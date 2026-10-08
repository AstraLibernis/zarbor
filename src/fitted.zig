// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Any fitted model, behind one interface, so callers switch on the algorithm
//! in one place: here.

const std = @import("std");
const Pool = @import("pool.zig").Pool;
const data = @import("data.zig");
const config = @import("config.zig");
const booster = @import("booster.zig");
const forest = @import("forest.zig");
const linear = @import("linear.zig");
const model = @import("model.zig");

/// Which scale `predictForReport` wrote. The booster reports raw log-odds;
/// the forest and the linear model already apply their own link.
pub const Scale = enum { raw, natural };

pub const Fitted = union(config.Algo) {
    gbdt: booster.Model,
    random_forest: forest.Forest,
    linear: linear.Linear,

    pub const Result = struct {
        model: Fitted,
        /// Trees grown, or solver epochs for the linear model.
        steps: u32,
        /// Time spent predicting and scoring the validation set.
        valid_ns: u64,
        /// How the linear solver finished; null for the ensembles.
        lin_fit: ?linear.Fit = null,
    };

    /// Fit the model `cfg.algo` names, with that model's own Params.
    pub fn train(
        gpa: std.mem.Allocator,
        pool: *Pool,
        ds: *const data.Dataset,
        valid: ?*const data.Dataset,
        cfg: config.Config,
        log: ?*std.Io.Writer,
    ) !Result {
        if (cfg.class_weight != .none) {
            // Train on a view of `ds` whose weights also carry each row's class weight.
            const w = try classWeights(gpa, ds, cfg);
            defer gpa.free(w);
            var view = ds.*;
            view.weights = w;
            var c = cfg;
            c.class_weight = .none;
            return train(gpa, pool, &view, valid, c, log);
        }
        return switch (cfg.algo) {
            .gbdt => blk: {
                const r = try booster.train(gpa, pool, ds, valid, cfg.gbdt, log);
                break :blk .{ .model = .{ .gbdt = r.model }, .steps = r.n_rounds, .valid_ns = r.valid_ns };
            },
            .random_forest => blk: {
                const r = try forest.train(gpa, pool, ds, valid, cfg.random_forest, log);
                break :blk .{ .model = .{ .random_forest = r.model }, .steps = r.n_trees, .valid_ns = r.valid_ns };
            },
            .linear => blk: {
                const r = try linear.train(gpa, pool, ds, valid, cfg.linear, log);
                break :blk .{ .model = .{ .linear = r.model }, .steps = r.epochs, .valid_ns = r.valid_ns, .lin_fit = r.fit };
            },
        };
    }

    /// Each row's sample weight (1 without one) times `n / (K * n_c)` for its class c, where
    /// with sample weights `n` and `n_c` are weight sums: scikit-learn's
    /// `compute_class_weight("balanced", sample_weight=...)`, as `LogisticRegression` calls it.
    /// A binary target's classes are `y > 0.5` and not. Caller frees.
    pub fn classWeights(gpa: std.mem.Allocator, ds: *const data.Dataset, cfg: config.Config) ![]f32 {
        const k: usize = if (cfg.objective() == .softmax) cfg.width() else 2;
        const counts = try gpa.alloc(f64, k);
        defer gpa.free(counts);
        @memset(counts, 0);
        var n: f64 = 0;
        for (ds.labels, 0..) |y, r| {
            const base: f64 = if (ds.weights.len != 0) ds.weights[r] else 1;
            counts[classOf(y, k)] += base;
            n += base;
        }
        const kf: f64 = @floatFromInt(k);
        const w = try gpa.alloc(f32, ds.n_rows);
        for (w, ds.labels, 0..) |*o, y, r| {
            const base: f64 = if (ds.weights.len != 0) ds.weights[r] else 1;
            o.* = @floatCast(base * n / (kf * counts[classOf(y, k)]));
        }
        return w;
    }

    fn classOf(y: f32, k: usize) usize {
        return if (k == 2 and y <= 1) @intFromBool(y > 0.5) else @intFromFloat(y);
    }

    pub fn deinit(m: *Fitted) void {
        switch (m.*) {
            inline else => |*x| x.deinit(),
        }
    }

    /// What one step of `Result.steps` is.
    pub fn stepName(m: Fitted) []const u8 {
        return switch (m) {
            // Softmax grows one tree per class each round.
            .gbdt => |x| if (x.num_class > 1) "round" else "tree",
            .random_forest => "tree",
            .linear => "epoch",
        };
    }

    /// Predictions on the natural scale: probabilities for logistic.
    pub fn predict(m: *const Fitted, pool: *Pool, ds: *const data.Dataset, out: []f32) void {
        switch (m.*) {
            inline else => |*x| x.predict(pool, ds, out),
        }
    }

    /// Scores for reporting a holdout, on whichever scale the model scores
    /// best on. For the booster that is raw log-odds: AUC is rank-based so
    /// the link does not matter, and logloss wants the raw scale anyway.
    pub fn predictForReport(m: *const Fitted, pool: *Pool, ds: *const data.Dataset, out: []f32) Scale {
        switch (m.*) {
            .gbdt => |*x| {
                x.predictRaw(pool, ds, out);
                return .raw;
            },
            inline else => |*x| {
                x.predict(pool, ds, out);
                return .natural;
            },
        }
    }

    /// Write the model to `path`. Takes ownership of `schema`.
    pub fn save(
        m: *const Fitted,
        gpa: std.mem.Allocator,
        io: std.Io,
        path: []const u8,
        schema: data.Schema,
        target: []const u8,
        enc: *const data.LabelEncoder,
    ) !void {
        var b = switch (m.*) {
            .gbdt => |*x| try model.fromBooster(gpa, x, schema),
            .random_forest => |*x| try model.fromForest(gpa, x, schema),
            // The bundle borrows the fitted model's design and weights rather
            // than copying, so it must not free them.
            .linear => |x| model.Bundle{
                .gpa = gpa,
                .kind = .linear,
                .schema = schema,
                .objective = x.objective,
                .num_class = x.num_class,
                .lin = x,
            },
        };
        // Dropping the borrowed model first lets the normal deinit clean up
        // everything the bundle does own.
        defer {
            b.lin = null;
            b.deinit();
        }
        try b.setLabel(target, enc);
        try model.save(gpa, io, path, &b);
    }
};
