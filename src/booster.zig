//! The boosting loop.
//!
//! The objective is dispatched through `inline else`, so each loss compiles to
//! its own specialised gradient loop with the sigmoid and the weighting
//! inlined — there is no indirect call anywhere in the per-row path.

const std = @import("std");
const Pool = @import("pool.zig").Pool;
const data = @import("data.zig");
const Dataset = data.Dataset;
const hist = @import("hist.zig");
const tree = @import("tree.zig");
const config = @import("config.zig");
const metric = @import("metric.zig");

const min_hessian: f32 = 1e-6;

pub const Model = struct {
    gpa: std.mem.Allocator,
    trees: std.ArrayList(tree.Tree),
    base_score: f32,
    objective: config.Objective,
    n_features: usize,

    pub fn deinit(m: *Model) void {
        for (m.trees.items) |*t| t.deinit(m.gpa);
        m.trees.deinit(m.gpa);
        m.* = undefined;
    }

    /// Raw scores (log-odds for logistic) for every row of `ds`.
    pub fn predictRaw(m: *const Model, pool: *Pool, ds: *const Dataset, out: []f32) void {
        std.debug.assert(out.len == ds.n_rows);
        var ctx = PredictCtx{ .m = m, .ds = ds, .out = out };
        pool.parallelFor(ds.n_rows, &ctx, PredictCtx.run, 2048);
    }

    /// Predictions on the objective's natural scale: probabilities for
    /// logistic, raw values for regression.
    pub fn predict(m: *const Model, pool: *Pool, ds: *const Dataset, out: []f32) void {
        m.predictRaw(pool, ds, out);
        if (m.objective == .logistic) {
            for (out) |*v| v.* = sigmoid(v.*);
        }
    }
};

const PredictCtx = struct {
    m: *const Model,
    ds: *const Dataset,
    out: []f32,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *PredictCtx = @ptrCast(@alignCast(ctx));
        var r = begin;
        while (r < end) : (r += 1) {
            var acc: f32 = self.m.base_score;
            for (self.m.trees.items) |t| acc += t.predictBinned(self.ds, r);
            self.out[r] = acc;
        }
    }
};

pub inline fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

// -------------------------------------------------------------- gradients

const GradCtx = struct {
    raw: []const f32,
    labels: []const f32,
    grads: []hist.GradPair,
    scale_pos_weight: f32,
    objective: config.Objective,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *GradCtx = @ptrCast(@alignCast(ctx));
        switch (self.objective) {
            inline else => |obj| self.runFor(obj, begin, end),
        }
    }

    fn runFor(self: *GradCtx, comptime obj: config.Objective, begin: usize, end: usize) void {
        var i = begin;
        while (i < end) : (i += 1) {
            const y = self.labels[i];
            switch (obj) {
                .logistic => {
                    const p = sigmoid(self.raw[i]);
                    const w: f32 = if (y > 0.5) self.scale_pos_weight else 1.0;
                    self.grads[i] = .{
                        .g = w * (p - y),
                        .h = w * @max(p * (1.0 - p), min_hessian),
                    };
                },
                .squared_error => {
                    self.grads[i] = .{ .g = self.raw[i] - y, .h = 1.0 };
                },
            }
        }
    }
};

const ApplyCtx = struct {
    spans: []const tree.LeafSpan,
    rows: []const u32,
    raw: []f32,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *ApplyCtx = @ptrCast(@alignCast(ctx));
        // Leaf spans are disjoint by construction, so distinct workers never
        // touch the same row and no synchronisation is needed.
        for (self.spans[begin..end]) |s| {
            for (self.rows[s.start..s.end]) |r| self.raw[r] += s.weight;
        }
    }
};

/// Applies a finished tree to *every* training row.
///
/// The span-based `ApplyCtx` only touches rows the tree actually saw, which is
/// correct when the tree saw all of them and wrong the moment any row sampling
/// is in play: an unsampled row's raw score would stay at the previous round's
/// value, and next round's gradient for it would be computed against a stale
/// ensemble. Sampled rounds pay for a full traversal instead.
const ApplyAllCtx = struct {
    t: *const tree.Tree,
    ds: *const Dataset,
    raw: []f32,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *ApplyAllCtx = @ptrCast(@alignCast(ctx));
        var r = begin;
        while (r < end) : (r += 1) self.raw[r] += self.t.predictBinned(self.ds, r);
    }
};

const ValidCtx = struct {
    t: *const tree.Tree,
    ds: *const Dataset,
    raw: []f32,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *ValidCtx = @ptrCast(@alignCast(ctx));
        var r = begin;
        while (r < end) : (r += 1) self.raw[r] += self.t.predictBinned(self.ds, r);
    }
};

// --------------------------------------------------------------------- GOSS

/// LightGBM's Gradient-based One-Side Sampling.
///
/// Rows with large |gradient| are the under-fitted ones and are kept in full;
/// the well-fitted remainder is sampled. Dropping most small-gradient rows
/// would bias the split gains, so the survivors are amplified by
/// `(1 - top_rate) / other_rate` — the reciprocal of their sampling rate —
/// which restores the expected gradient sum.
///
/// Mutates `grads` in place; the caller recomputes it every round anyway.
fn gossSelect(
    grads: []hist.GradPair,
    order: []u32,
    out: []u32,
    top_rate: f32,
    other_rate: f32,
    rng: std.Random,
) []u32 {
    const n = grads.len;
    for (order, 0..) |*o, i| o.* = @intCast(i);

    const Ctx = struct { g: []const hist.GradPair };
    std.sort.pdq(u32, order, Ctx{ .g = grads }, struct {
        fn lt(c: Ctx, a: u32, b: u32) bool {
            return @abs(c.g[a].g) > @abs(c.g[b].g);
        }
    }.lt);

    var top: usize = @intFromFloat(@round(@as(f32, @floatFromInt(n)) * top_rate));
    top = std.math.clamp(top, 1, n);
    const rest = n - top;

    var rand_n: usize = @intFromFloat(@round(@as(f32, @floatFromInt(n)) * other_rate));
    rand_n = @min(rand_n, rest);

    @memcpy(out[0..top], order[0..top]);

    var i: usize = 0;
    while (i < rand_n) : (i += 1) {
        const j = i + rng.uintLessThan(usize, rest - i);
        std.mem.swap(u32, &order[top + i], &order[top + j]);
        out[top + i] = order[top + i];
    }

    const amp: f32 = (1.0 - top_rate) / other_rate;
    for (out[top .. top + rand_n]) |r| {
        grads[r].g *= amp;
        grads[r].h *= amp;
    }
    return out[0 .. top + rand_n];
}

// ----------------------------------------------------------------- training

pub const TrainResult = struct {
    model: Model,
    /// Rounds kept after the post-hoc trim.
    n_rounds: u32,
    /// Rounds actually executed before early stopping fired. Distinct from
    /// `n_rounds`, which also reflects trimming the trees that came after the
    /// best round — the two differ, and conflating them hides whether early
    /// stopping ever triggered at all.
    rounds_run: u32,
    /// Best validation score seen, or NaN when no validation set was given.
    best_score: f64,
};

fn higherIsBetter(obj: config.Objective) bool {
    return obj == .logistic; // AUC for logistic, RMSE for regression
}

fn evaluate(
    gpa: std.mem.Allocator,
    obj: config.Objective,
    raw: []const f32,
    labels: []const f32,
    scratch: []f32,
) !f64 {
    return switch (obj) {
        .logistic => blk: {
            // AUC is rank-based, so raw log-odds rank identically to
            // probabilities and the sigmoid can be skipped.
            break :blk try metric.auc(gpa, raw, labels);
        },
        .squared_error => blk: {
            @memcpy(scratch, raw);
            break :blk metric.rmse(scratch, labels);
        },
    };
}

pub fn train(
    gpa: std.mem.Allocator,
    pool: *Pool,
    ds: *const Dataset,
    valid: ?*const Dataset,
    cfg: config.Config,
    log: ?*std.Io.Writer,
) !TrainResult {
    try cfg.validate();
    if (ds.labels.len == 0) return error.NoLabels;

    var model = Model{
        .gpa = gpa,
        .trees = .empty,
        .base_score = 0,
        .objective = cfg.objective,
        .n_features = ds.n_features,
    };
    errdefer model.deinit();

    // --- base score ---
    model.base_score = if (cfg.base_score) |b| b else blk: {
        var sum: f64 = 0;
        for (ds.labels) |y| sum += y;
        const mean = sum / @as(f64, @floatFromInt(ds.labels.len));
        break :blk switch (cfg.objective) {
            .logistic => b: {
                const p = std.math.clamp(mean, 1e-6, 1 - 1e-6);
                break :b @floatCast(@log(p / (1 - p)));
            },
            .squared_error => @floatCast(mean),
        };
    };

    const raw = try gpa.alloc(f32, ds.n_rows);
    defer gpa.free(raw);
    @memset(raw, model.base_score);

    const grads = try gpa.alloc(hist.GradPair, ds.n_rows);
    defer gpa.free(grads);

    var valid_raw: []f32 = &.{};
    var valid_scratch: []f32 = &.{};
    if (valid) |v| {
        valid_raw = try gpa.alloc(f32, v.n_rows);
        valid_scratch = try gpa.alloc(f32, v.n_rows);
        @memset(valid_raw, model.base_score);
    }
    defer if (valid_raw.len != 0) gpa.free(valid_raw);
    defer if (valid_scratch.len != 0) gpa.free(valid_scratch);

    var goss_order: []u32 = &.{};
    var goss_rows: []u32 = &.{};
    if (cfg.sampling == .goss) {
        goss_order = try gpa.alloc(u32, ds.n_rows);
        goss_rows = try gpa.alloc(u32, ds.n_rows);
    }
    defer if (goss_order.len != 0) gpa.free(goss_order);
    defer if (goss_rows.len != 0) gpa.free(goss_rows);
    var goss_rng: std.Random.DefaultPrng = .init(cfg.seed +% 0x9E3779B97F4A7C15);

    var builder = try tree.Builder.init(gpa, pool, ds, cfg);
    defer builder.deinit();

    const better = higherIsBetter(cfg.objective);
    var best_score: f64 = if (better) -std.math.inf(f64) else std.math.inf(f64);
    var best_round: u32 = 0;
    var since_best: u32 = 0;

    var round: u32 = 0;
    while (round < cfg.n_rounds) : (round += 1) {
        var gctx = GradCtx{
            .raw = raw,
            .labels = ds.labels,
            .grads = grads,
            .scale_pos_weight = cfg.scale_pos_weight,
            .objective = cfg.objective,
        };
        pool.parallelFor(ds.n_rows, &gctx, GradCtx.run, 8192);

        const subset: ?[]const u32 = if (cfg.sampling == .goss)
            gossSelect(grads, goss_order, goss_rows, cfg.top_rate, cfg.other_rate, goss_rng.random())
        else
            null;

        var t = try builder.growRows(grads, subset);
        errdefer t.deinit(gpa);
        try model.trees.append(gpa, t);

        // Spans only cover the rows the tree saw; that is every row only when
        // nothing was sampled away.
        if (builder.activeRows().len == ds.n_rows) {
            var actx = ApplyCtx{
                .spans = builder.leafSpans(),
                .rows = builder.rows,
                .raw = raw,
            };
            pool.parallelFor(builder.leafSpans().len, &actx, ApplyCtx.run, 1);
        } else {
            var actx = ApplyAllCtx{
                .t = &model.trees.items[model.trees.items.len - 1],
                .ds = ds,
                .raw = raw,
            };
            pool.parallelFor(ds.n_rows, &actx, ApplyAllCtx.run, 4096);
        }

        if (valid) |v| {
            var vctx = ValidCtx{ .t = &model.trees.items[model.trees.items.len - 1], .ds = v, .raw = valid_raw };
            pool.parallelFor(v.n_rows, &vctx, ValidCtx.run, 4096);

            const score = try evaluate(gpa, cfg.objective, valid_raw, v.labels, valid_scratch);
            const improved = if (better) score > best_score else score < best_score;
            if (improved) {
                best_score = score;
                best_round = round;
                since_best = 0;
            } else {
                since_best += 1;
            }

            if (log) |w| {
                if (cfg.verbose_eval != 0 and (round % cfg.verbose_eval == 0 or round + 1 == cfg.n_rounds)) {
                    try w.print("[{d:>4}] valid={d:.6}  best={d:.6} @{d}\n", .{ round, score, best_score, best_round });
                    try w.flush();
                }
            }

            if (cfg.early_stopping_rounds != 0 and since_best >= cfg.early_stopping_rounds) {
                if (log) |w| {
                    try w.print("early stop at round {d}; best {d:.6} @{d}\n", .{ round, best_score, best_round });
                    try w.flush();
                }
                round += 1;
                break;
            }
        } else if (log) |w| {
            if (cfg.verbose_eval != 0 and round % cfg.verbose_eval == 0) {
                try w.print("[{d:>4}] trees={d}\n", .{ round, model.trees.items.len });
                try w.flush();
            }
        }
    }

    // Drop the rounds that came after the best one; they only overfit.
    if (valid != null and cfg.early_stopping_rounds != 0) {
        const keep = best_round + 1;
        while (model.trees.items.len > keep) {
            var t = model.trees.pop().?;
            t.deinit(gpa);
        }
    }

    return .{
        .model = model,
        .n_rounds = @intCast(model.trees.items.len),
        .rounds_run = round,
        .best_score = if (valid != null) best_score else std.math.nan(f64),
    };
}
