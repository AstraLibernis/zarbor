// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Random forest: bagged, unshrunk trees, averaged.
//! Reuses the booster's `tree.zig` by an identity: the histogram search maximises
//! `G²/(H+lambda)` over children; with `g = -y`, `h = 1`, `lambda = 0` that is
//! `(Σy)²/n`, CART's variance reduction (Gini for 0/1 labels), and the leaf weight
//! `-G/(H+lambda)` is `mean(y)`. So a forest tree is a boosting tree fed constant
//! gradients with shrinkage off. Real differences: rows drawn with replacement,
//! every tree fits the same targets (not residuals), the ensemble averages.

const std = @import("std");
const Pool = @import("pool.zig").Pool;
const data = @import("data.zig");
const Dataset = data.Dataset;
const hist = @import("hist.zig");
const tree = @import("tree.zig");
const prof = @import("prof.zig");
const metric = @import("metric.zig");
const Objective = @import("objective.zig").Objective;

/// Tree defaults make it a random forest: deep, bootstrapped, unshrunk,
/// leaf-capped to bound the histogram budget. `colsample_bynode` gets Breiman's
/// sqrt(p) (or p/3) from `config.Config.applyForestFeatureDefault` once p is known.
pub const Params = struct {
    /// Number of bagged trees, fitted independently.
    n_rounds: u32 = 300,
    objective: Objective = .logistic,
    /// Print per-round metrics every N rounds. 0 silences training.
    verbose_eval: u32 = 10,
    tree: tree.Params = .{
        .learning_rate = 1.0,
        .bootstrap = true,
        .max_depth = 0,
        .max_leaves = 1024,
        .min_child_samples = 1,
        .min_child_weight = 0.0,
        .lambda = 0.0,
    },

    pub fn validate(p: Params) !void {
        if (p.n_rounds == 0) return error.NoRounds;
        // `train` forces the step to 1, so only that value is checked.
        var t = p.tree;
        t.learning_rate = 1.0;
        try t.validate();
        try p.tree.validateCapacity();
    }
};

pub const Forest = struct {
    gpa: std.mem.Allocator,
    trees: std.ArrayList(tree.Tree),
    objective: Objective,
    n_features: usize,

    pub fn deinit(m: *Forest) void {
        for (m.trees.items) |*t| t.deinit(m.gpa);
        m.trees.deinit(m.gpa);
        m.* = undefined;
    }

    /// Mean of member trees, already on the natural scale: leaves hold `mean(y)`,
    /// so for logistic this is a probability with no link to invert.
    pub fn predict(m: *const Forest, pool: *Pool, ds: *const Dataset, out: []f32) void {
        std.debug.assert(out.len == ds.n_rows);
        var ctx = PredictCtx{ .m = m, .ds = ds, .out = out };
        pool.parallelFor(ds.n_rows, &ctx, PredictCtx.run, 2048);
    }
};

const PredictCtx = struct {
    m: *const Forest,
    ds: *const Dataset,
    out: []f32,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *PredictCtx = @ptrCast(@alignCast(ctx));
        const n = self.m.trees.items.len;
        const inv: f32 = if (n == 0) 0 else 1.0 / @as(f32, @floatFromInt(n));
        const out = self.out[begin..end];

        // Trees outer, rows inner: a 300 x 1024-leaf forest is ~19.7 MB of nodes,
        // so row-major scattered each row across 300 uncached arrays; this keeps
        // one tree (~65 KB) and the chunk's bins in L2. Bit-exact with row-major:
        // each `out[r]` sums the same trees in the same order, rounding to f32 per step.
        @memset(out, 0);
        for (self.m.trees.items) |t| {
            for (out, begin..) |*o, r| o.* += t.predictBinned(self.ds, r);
        }
        for (out) |*o| o.* *= inv;
    }
};

/// Validation rows per pool chunk, at least. 4096 cut 133k rows into 33 chunks for 16 workers, so the
/// barrier waited on a worker with three while the average had two; 1024 gives 131. Rows are
/// independent, so the chunking changes nothing but the wait.
const valid_min_chunk = 1024;

/// Running sum of member predictions, so a validation curve costs one tree
/// traversal per round instead of re-predicting the whole ensemble.
const AccumCtx = struct {
    t: *const tree.Tree,
    ds: *const Dataset,
    sum: []f32,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *AccumCtx = @ptrCast(@alignCast(ctx));
        var r = begin;
        while (r < end) : (r += 1) self.sum[r] += self.t.predictBinned(self.ds, r);
    }
};

pub const TrainResult = struct {
    model: Forest,
    n_trees: u32,
    /// Validation score of the full ensemble, or NaN with no validation set.
    score: f64,
    /// Nanoseconds predicting and scoring validation, not fitting; see `booster.TrainResult.valid_ns`.
    valid_ns: u64,
};

pub fn train(
    gpa: std.mem.Allocator,
    pool: *Pool,
    ds: *const Dataset,
    valid: ?*const Dataset,
    cfg_in: Params,
    log: ?*std.Io.Writer,
) !TrainResult {
    var cfg = cfg_in;
    // Averaging controls variance; shrinking members too just scales a weak ensemble by eta.
    cfg.tree.learning_rate = 1.0;
    try cfg.validate();
    if (ds.labels.len == 0) return error.NoLabels;

    var model = Forest{
        .gpa = gpa,
        .trees = .empty,
        .objective = cfg.objective,
        .n_features = ds.n_features,
    };
    errdefer model.deinit();

    // Constant across every tree: each one fits the raw target, not a residual.
    const grads = try gpa.alloc(hist.GradPair, ds.n_rows);
    defer gpa.free(grads);
    for (ds.labels, 0..) |y, i| grads[i] = .{ .g = -y, .h = 1.0 };

    var valid_sum: []f32 = &.{};
    var valid_scratch: []f32 = &.{};
    if (valid) |v| {
        valid_sum = try gpa.alloc(f32, v.n_rows);
        valid_scratch = try gpa.alloc(f32, v.n_rows);
        @memset(valid_sum, 0);
    }
    defer if (valid_sum.len != 0) gpa.free(valid_sum);
    defer if (valid_scratch.len != 0) gpa.free(valid_scratch);

    var builder = try tree.Builder.init(gpa, pool, ds, cfg.tree);
    defer builder.deinit();

    var score: f64 = std.math.nan(f64);
    var valid_ns: u64 = 0;
    var round: u32 = 0;
    while (round < cfg.n_rounds) : (round += 1) {
        var t = try builder.grow(grads);
        errdefer t.deinit(gpa);
        try model.trees.append(gpa, t);

        if (valid) |v| {
            const wall0 = prof.now();
            const t_vp = prof.start();
            var actx = AccumCtx{
                .t = &model.trees.items[model.trees.items.len - 1],
                .ds = v,
                .sum = valid_sum,
            };
            pool.parallelFor(v.n_rows, &actx, AccumCtx.run, valid_min_chunk);
            prof.stop(.valid_predict, t_vp);

            // Score only when consumed: no early stopping, so only the last and logged
            // rounds use it, and the metric is a single-thread sort (133k rows: 1.18 s
            // of a 2.35 s fit when scored every round; the booster has the same fix).
            const log_due = log != null and cfg.verbose_eval != 0 and
                (round % cfg.verbose_eval == 0 or round + 1 == cfg.n_rounds);
            if (!log_due and round + 1 != cfg.n_rounds) {
                valid_ns += prof.now() - wall0;
                continue;
            }

            const t_vm = prof.start();
            const inv: f32 = 1.0 / @as(f32, @floatFromInt(model.trees.items.len));
            for (valid_scratch, valid_sum) |*dst, s| dst.* = s * inv;
            score = try evaluate(gpa, cfg.objective, valid_scratch, v.labels);
            prof.stop(.valid_metric, t_vm);
            valid_ns += prof.now() - wall0;

            if (log) |w| {
                if (cfg.verbose_eval != 0 and
                    (round % cfg.verbose_eval == 0 or round + 1 == cfg.n_rounds))
                {
                    try w.print("[{d:>4}] trees={d} valid={d:.6}\n", .{ round, model.trees.items.len, score });
                    try w.flush();
                }
            }
        } else if (log) |w| {
            if (cfg.verbose_eval != 0 and round % cfg.verbose_eval == 0) {
                try w.print("[{d:>4}] trees={d}\n", .{ round, model.trees.items.len });
                try w.flush();
            }
        }
    }

    return .{
        .model = model,
        .n_trees = @intCast(model.trees.items.len),
        .score = score,
        .valid_ns = valid_ns,
    };
}

/// Predictions are already probabilities/means: no raw-score scale to convert, unlike the booster.
fn evaluate(
    gpa: std.mem.Allocator,
    obj: Objective,
    pred: []f32,
    labels: []const f32,
) !f64 {
    return switch (obj) {
        .logistic => try metric.auc(gpa, pred, labels),
        .squared_error => metric.rmse(pred, labels),
    };
}
