// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Random forest: bagged, unshrunk trees, averaged.
//!
//! This shares `tree.zig` with the booster rather than reimplementing tree
//! induction, which works because of an identity worth spelling out. The
//! histogram search maximises `G²/(H+lambda)` summed over children. Feed it
//! `g = -y` and `h = 1` and that becomes `(Σy)²/n` — exactly CART's variance
//! reduction, and for 0/1 labels exactly the Gini criterion. The optimal leaf
//! weight `-G/(H+lambda)` likewise collapses to `mean(y)`.
//!
//! So a forest tree is a boosting tree fed constant gradients, with shrinkage
//! off and `lambda = 0`. The differences that remain are real ones: rows are
//! drawn with replacement, every tree sees the same targets rather than the
//! previous round's residuals, and the ensemble averages instead of summing.

const std = @import("std");
const Pool = @import("pool.zig").Pool;
const data = @import("data.zig");
const Dataset = data.Dataset;
const hist = @import("hist.zig");
const tree = @import("tree.zig");
const config = @import("config.zig");
const prof = @import("prof.zig");
const metric = @import("metric.zig");

pub const Forest = struct {
    gpa: std.mem.Allocator,
    trees: std.ArrayList(tree.Tree),
    objective: config.Objective,
    n_features: usize,

    pub fn deinit(m: *Forest) void {
        for (m.trees.items) |*t| t.deinit(m.gpa);
        m.trees.deinit(m.gpa);
        m.* = undefined;
    }

    /// Mean of the member trees. Already on the objective's natural scale:
    /// each leaf holds `mean(y)` over its rows, so for logistic this is a
    /// probability and there is no link function to invert.
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

        // Trees outer, rows inner. The other order walks every tree for one
        // row before moving on, and a 300 x 1024-leaf forest is ~19.7 MB of
        // nodes -- so each row scattered across 300 separate arrays, none of
        // which stayed cached. This way one tree's nodes (~65 KB) and the
        // chunk's bins stay in L2 for the whole pass.
        //
        // Bit-exact with the row-major order: each `out[r]` still accumulates
        // the same trees in the same sequence, rounding to f32 at each step
        // exactly as the register accumulator did.
        @memset(out, 0);
        for (self.m.trees.items) |t| {
            for (out, begin..) |*o, r| o.* += t.predictBinned(self.ds, r);
        }
        for (out) |*o| o.* *= inv;
    }
};

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
    /// Nanoseconds spent predicting and scoring the validation set; not part
    /// of fitting. See `booster.TrainResult.valid_ns`.
    valid_ns: u64,
};

pub fn train(
    gpa: std.mem.Allocator,
    pool: *Pool,
    ds: *const Dataset,
    valid: ?*const Dataset,
    cfg_in: config.Config,
    log: ?*std.Io.Writer,
) !TrainResult {
    var cfg = cfg_in;
    // Averaging is the variance control; shrinking each member as well would
    // just produce a weak ensemble scaled by eta.
    cfg.learning_rate = 1.0;
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

    var builder = try tree.Builder.init(gpa, pool, ds, cfg);
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
            pool.parallelFor(v.n_rows, &actx, AccumCtx.run, 4096);
            prof.stop(.valid_predict, t_vp);

            // Only score when something consumes it. A forest has no early
            // stopping, so every round but the last (and any logged one) threw
            // the number away — and the metric is a 133k-row sort on one
            // thread. That was 1.18 s of a 2.35 s fit, the same mistake the
            // booster made and had fixed.
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

/// Forest predictions are already probabilities/means, so unlike the booster
/// there is no raw-score scale to convert from.
fn evaluate(
    gpa: std.mem.Allocator,
    obj: config.Objective,
    pred: []f32,
    labels: []const f32,
) !f64 {
    return switch (obj) {
        .logistic => try metric.auc(gpa, pred, labels),
        .squared_error => metric.rmse(pred, labels),
    };
}
