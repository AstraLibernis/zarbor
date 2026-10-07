// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! The boosting loop. The objective dispatches through `inline else`, so each
//! loss compiles its own gradient loop with sigmoid and weighting inlined: no
//! indirect call in the per-row path.

const std = @import("std");
const Pool = @import("pool.zig").Pool;
const data = @import("data.zig");
const Dataset = data.Dataset;
const hist = @import("hist.zig");
const tree = @import("tree.zig");
const symmetric = @import("symmetric.zig");
const metric = @import("metric.zig");
const prof = @import("prof.zig");
const Objective = @import("objective.zig").Objective;
const goss = @import("goss.zig");
const n_radix = goss.n_radix;
const gossSelect = goss.gossSelect;

/// How rows are chosen for each tree.
pub const Sampling = enum {
    /// Uniform random subset of size `subsample`, without replacement.
    uniform,
    /// LightGBM's Gradient-based One-Side Sampling: keep large-gradient rows, sample
    /// the rest, amplify survivors so the gradient sum stays unbiased. Ignores `subsample`.
    goss,
};

/// Magnitude GOSS ranks rows by to keep in full. The one place the GOSS paper and
/// LightGBM's code differ; the choice moves accuracy measurably (docs/goss.md).
pub const GossRank = enum {
    /// `|g|`: Ke et al. (NeurIPS 2017) and LightGBM's `top_rate` docs. Keeps the
    /// largest residuals.
    gradient,
    /// `|g * h|`: what LightGBM computes in `goss.hpp`; select for LightGBM parity.
    /// The hessian pulls confidently-wrong rows *out* of the kept set: logistic
    /// `h = p(1-p)`, so p = 0.99 scores ~25x lower than p = 0.5 at equal residual.
    /// Identical to `gradient` on squared error (`h` = 1).
    gradient_hessian,
};

/// Everything gradient boosting needs; the trees' own settings are in `tree`.
pub const Params = struct {
    /// Boosting rounds (trees).
    n_rounds: u32 = 500,
    /// Initial raw score (log-odds for logistic). Null (the usual choice) derives
    /// it from the training label mean.
    base_score: ?f32 = null,
    objective: Objective = .logistic,
    /// Multiplier on positive-class gradients; >1 upweights an imbalanced minority.
    scale_pos_weight: f32 = 1.0,
    /// Row-selection strategy. `goss` is LightGBM's; `uniform` is XGBoost's.
    sampling: Sampling = .uniform,
    /// GOSS ranking key: the paper's `|g|` (measures better here) or LightGBM's
    /// `gradient_hessian`. Ignored unless `sampling = .goss`.
    goss_rank: GossRank = .gradient,
    /// GOSS: fraction of rows kept in full for the largest `goss_rank` key.
    top_rate: f32 = 0.2,
    /// GOSS: fraction of the *remaining* rows sampled uniformly.
    other_rate: f32 = 0.1,
    /// Stop after this many rounds without validation improvement; 0 disables.
    early_stopping_rounds: u32 = 0,
    /// Print per-round metrics every N rounds. 0 silences training.
    verbose_eval: u32 = 10,
    tree: tree.Params = .{},

    pub fn validate(p: Params) !void {
        if (p.n_rounds == 0) return error.NoRounds;
        try p.tree.validate();
        // With replacement a row's gradient counts more than once, meaningless when
        // next round's gradient comes from one accumulated score per row.
        if (p.tree.bootstrap) return error.BootstrapWithBoosting;
        // Target statistics count positives: a binary target.
        if (p.tree.cat_split == .ctr and p.objective != .logistic) return error.CtrNeedsLogistic;
        if (p.sampling == .goss) {
            if (p.tree.grow_policy == .symmetric) return error.SymmetricUnsupported;
            if (p.top_rate <= 0 or p.top_rate >= 1) return error.BadTopRate;
            if (p.other_rate <= 0 or p.other_rate >= 1) return error.BadOtherRate;
            if (p.top_rate + p.other_rate > 1) return error.GossRatesExceedOne;
        }
        try p.tree.validateCapacity();
    }
};

const min_hessian: f32 = 1e-6;

pub const Model = struct {
    gpa: std.mem.Allocator,
    trees: std.ArrayList(tree.Tree),
    base_score: f32,
    objective: Objective,
    n_features: usize,

    pub fn deinit(m: *Model) void {
        for (m.trees.items) |*t| t.deinit(m.gpa);
        m.trees.deinit(m.gpa);
        m.* = undefined;
    }

    /// Raw scores (log-odds for logistic) for every row of `ds`.
    pub fn predictRaw(m: *const Model, pool: *Pool, ds: *const Dataset, out: []f32) void {
        m.predictScaled(pool, ds, out, false);
    }

    /// Natural-scale predictions: probabilities for logistic, raw for regression. The sigmoid runs
    /// in the workers, as `model.zig` does it, not in a serial pass after them.
    pub fn predict(m: *const Model, pool: *Pool, ds: *const Dataset, out: []f32) void {
        m.predictScaled(pool, ds, out, m.objective == .logistic);
    }

    fn predictScaled(m: *const Model, pool: *Pool, ds: *const Dataset, out: []f32, prob: bool) void {
        std.debug.assert(out.len == ds.n_rows);
        var ctx = PredictCtx{ .m = m, .ds = ds, .out = out, .prob = prob };
        pool.parallelFor(ds.n_rows, &ctx, PredictCtx.run, 2048);
    }
};

const PredictCtx = struct {
    m: *const Model,
    ds: *const Dataset,
    out: []f32,
    /// Apply the sigmoid: probabilities, not log-odds.
    prob: bool,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *PredictCtx = @ptrCast(@alignCast(ctx));
        const out = self.out[begin..end];
        // Trees outer, rows inner; see the note in `model.zig`'s TreeCtx.
        // Bit-exact: same trees, same order, same f32 rounding per row.
        @memset(out, self.m.base_score);
        for (self.m.trees.items) |t| {
            for (out, begin..) |*o, r| o.* += t.predictBinned(self.ds, r);
        }
        if (self.prob) for (out) |*v| {
            v.* = sigmoid(v.*);
        };
    }
};

pub inline fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// Lanes in the vectorised gradient loop. Eight f32 is one AVX2 register.
pub const lanes = 8;
pub const F8 = @Vector(lanes, f32);

/// `sigmoid` for eight rows. Scalar `@exp` is a libm call per element (glibc `expf`
/// no better), and gradients are a noticeable share of a fit. Inline `2^(-x*log2e)`,
/// integer part into the exponent field, fraction by degree-5 minimax polynomial,
/// vectorises and is much faster. Error < 1e-6 absolute, pinned by `"vectorised
/// sigmoid matches the scalar one"`; relative to the true value it is tens of f32 ulp
/// in the far tail, not "a few" (docs/measurements.md). Still an approximation, so it
/// is confined to gradients; predictions use the scalar `sigmoid`.
pub inline fn sigmoid8(x: F8) F8 {
    const one: F8 = @splat(1.0);
    // exp overflows f32 past ~88; sigmoid has saturated long before, so clamp.
    const lim: F8 = @splat(88.0);
    const t = @min(@max(-x, -lim), lim);
    const y = t * @as(F8, @splat(1.44269504));
    const yr = @round(y);
    const f = y - yr;
    const p = one + f * (@as(F8, @splat(0.6931472)) +
        f * (@as(F8, @splat(0.2402265)) +
            f * (@as(F8, @splat(0.0555041)) +
                f * (@as(F8, @splat(0.0096181)) +
                    f * @as(F8, @splat(0.0013333))))));
    const ei: @Vector(lanes, i32) = @intFromFloat(yr);
    const bits: @Vector(lanes, i32) = (ei + @as(@Vector(lanes, i32), @splat(127))) <<
        @as(@Vector(lanes, u5), @splat(23));
    return one / (one + p * @as(F8, @bitCast(bits)));
}

/// `sigmoid8` for one row. The vector loop's tail must not use exact `sigmoid`:
/// chunk boundaries move with thread count, so the model would depend on `--n_threads`.
pub inline fn sigmoid1(x: f32) f32 {
    const v: F8 = @splat(x);
    return sigmoid8(v)[0];
}

// -------------------------------------------------------------- gradients

const GradCtx = struct {
    raw: []const f32,
    labels: []const f32,
    grads: []hist.GradPair,
    scale_pos_weight: f32,
    objective: Objective,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *GradCtx = @ptrCast(@alignCast(ctx));
        switch (self.objective) {
            inline else => |obj| self.runFor(obj, begin, end),
        }
    }

    fn runFor(self: *GradCtx, comptime obj: Objective, begin: usize, end: usize) void {
        var i = begin;
        if (obj == .logistic) {
            const one: F8 = @splat(1.0);
            const floor: F8 = @splat(min_hessian);
            const half: F8 = @splat(0.5);
            const spw: F8 = @splat(self.scale_pos_weight);
            while (i + lanes <= end) : (i += lanes) {
                const raw: F8 = self.raw[i..][0..lanes].*;
                const y: F8 = self.labels[i..][0..lanes].*;
                const p = sigmoid8(raw);
                const w = @select(f32, y > half, spw, one);
                const g = w * (p - y);
                const h = w * @max(p * (one - p), floor);
                // GradPair is {g, h} interleaved: write per row, not two vector scatters.
                const ga: [lanes]f32 = g;
                const ha: [lanes]f32 = h;
                for (0..lanes) |k| self.grads[i + k] = .{ .g = ga[k], .h = ha[k] };
            }
        }
        while (i < end) : (i += 1) {
            const y = self.labels[i];
            switch (obj) {
                .logistic => {
                    const p = sigmoid1(self.raw[i]);
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

/// Adds each leaf's weight to its rows, split over positions in `rows`, not over leaves: one task
/// per leaf left the barrier waiting on the biggest leaf, and leaves' rows interleave in row-id
/// space, so workers on different leaves wrote the same cache lines of `raw`. The spans are
/// sorted by start and tile `rows[0..n]`, so a chunk finds its first span and walks on. Boosting
/// rows are distinct (bootstrap is rejected for it), so chunks never share a row, and every row
/// still gets its one add: bit-identical.
const ApplyCtx = struct {
    spans: []const tree.LeafSpan,
    rows: []const u32,
    raw: []f32,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *ApplyCtx = @ptrCast(@alignCast(ctx));
        const spans = self.spans;
        // The span holding `begin`: the last one starting at or before it.
        var lo: usize = 0;
        var hi: usize = spans.len;
        while (hi - lo > 1) {
            const m = lo + (hi - lo) / 2;
            if (spans[m].start <= begin) lo = m else hi = m;
        }
        var si = lo;
        var i = begin;
        while (i < end) : (si += 1) {
            const s = spans[si];
            const stop = @min(s.end, end);
            for (self.rows[i..stop]) |r| self.raw[r] += s.weight;
            i = @max(i, stop);
        }
    }

    fn byStart(_: void, a: tree.LeafSpan, b: tree.LeafSpan) bool {
        return a.start < b.start;
    }
};

/// Applies a finished tree to *every* training row. Span-based `ApplyCtx` touches
/// only rows the tree saw; under row sampling an unsampled row's raw score would go
/// stale and its next gradient be wrong. Sampled rounds pay a full traversal.
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

/// Validation rows per pool chunk, at least. Too coarse a chunk leaves a typical validation set
/// with only a couple of chunks per worker, so the barrier waits on whichever worker drew one
/// extra; smaller chunks even out the load. Rows are independent, so the chunking changes
/// nothing but the wait. `forest.valid_min_chunk` follows the same reasoning.
/// Adds a symmetric tree's leaf values by each row's leaf index, which the builder already holds.
/// The same f32 sum as predicting the tree, without walking it.
const SymApplyCtx = struct {
    leaf: []const u16,
    values: []const f64,
    raw: []f32,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *SymApplyCtx = @ptrCast(@alignCast(ctx));
        for (self.raw[begin..end], self.leaf[begin..end]) |*r, l| r.* += @floatCast(self.values[l]);
    }
};

const valid_min_chunk = 1024;

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

// ----------------------------------------------------------------- training

pub const TrainResult = struct {
    model: Model,
    /// Rounds kept after the post-hoc trim.
    n_rounds: u32,
    /// Rounds executed before early stopping fired. Unlike `n_rounds` (post-trim),
    /// this shows whether early stopping triggered at all.
    rounds_run: u32,
    /// Best validation score seen; NaN with no validation set. With
    /// `early_stopping_rounds == 0` only logged rounds and the last are scored.
    best_score: f64,
    /// Nanoseconds predicting and scoring validation. Kept apart from fitting so a
    /// comparison with a library run without an eval set is fair. Always measured,
    /// not only under `--profile`: two clock reads a round cost nothing.
    valid_ns: u64,
};

fn higherIsBetter(obj: Objective) bool {
    return obj == .logistic; // AUC for logistic, RMSE for regression
}

fn evaluate(
    gpa: std.mem.Allocator,
    obj: Objective,
    raw: []const f32,
    labels: []const f32,
) !f64 {
    return switch (obj) {
        .logistic => blk: {
            // AUC is rank-based: log-odds rank like probabilities, skip the sigmoid.
            break :blk try metric.auc(gpa, raw, labels);
        },
        .squared_error => metric.rmse(raw, labels),
    };
}

pub fn train(
    gpa: std.mem.Allocator,
    pool: *Pool,
    ds: *const Dataset,
    valid: ?*const Dataset,
    cfg: Params,
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
    if (valid) |v| {
        valid_raw = try gpa.alloc(f32, v.n_rows);
        @memset(valid_raw, model.base_score);
    }
    defer if (valid_raw.len != 0) gpa.free(valid_raw);

    var goss_counts: []u32 = &.{};
    var goss_others: []u32 = &.{};
    var goss_mask: []u64 = &.{};
    var goss_rows: []u32 = &.{};
    var goss_chunks: []usize = &.{};
    if (cfg.sampling == .goss) {
        goss_counts = try gpa.alloc(u32, n_radix);
        goss_chunks = try gpa.alloc(usize, goss.scratchLen(pool));
        goss_others = try gpa.alloc(u32, ds.n_rows);
        goss_mask = try gpa.alloc(u64, (ds.n_rows + 63) / 64);
        goss_rows = try gpa.alloc(u32, ds.n_rows);
    }
    defer if (goss_counts.len != 0) gpa.free(goss_counts);
    defer if (goss_others.len != 0) gpa.free(goss_others);
    defer if (goss_mask.len != 0) gpa.free(goss_mask);
    defer if (goss_rows.len != 0) gpa.free(goss_rows);
    defer if (goss_chunks.len != 0) gpa.free(goss_chunks);
    var goss_rng: std.Random.DefaultPrng = .init(cfg.tree.seed +% 0x9E3779B97F4A7C15);
    // The finished tree's leaf spans, sorted by position for `ApplyCtx`.
    const span_buf = try gpa.alloc(tree.LeafSpan, cfg.tree.leafBudget());
    defer gpa.free(span_buf);

    // Exactly one of the two builders exists: symmetric trees have their own (symmetric.zig).
    var sym: ?symmetric.Builder = if (cfg.tree.grow_policy == .symmetric) try symmetric.Builder.init(gpa, pool, ds, .{
        .depth = cfg.tree.max_depth,
        .lambda = cfg.tree.lambda,
        .learning_rate = cfg.tree.learning_rate,
        .score = cfg.tree.score_function,
        .leaf_iterations = cfg.tree.leaf_estimation_iterations,
        .bootstrap = cfg.tree.bootstrap_type,
        .subsample = cfg.tree.subsample,
        .bagging_temperature = cfg.tree.bagging_temperature,
        .mvs_reg = if (cfg.tree.mvs_reg) |m| m else null,
        .random_strength = cfg.tree.random_strength,
        .seed = cfg.tree.seed,
        .ctr = cfg.tree.cat_split == .ctr,
        .one_hot_max_size = cfg.tree.one_hot_max_size,
        .model_size_reg = cfg.tree.model_size_reg,
        .ordered = cfg.tree.boosting_type == .ordered,
    }) else null;
    defer if (sym) |*s| s.deinit();
    var builder_opt: ?tree.Builder = if (sym == null) try tree.Builder.init(gpa, pool, ds, cfg.tree) else null;
    defer if (builder_opt) |*b| b.deinit();

    const better = higherIsBetter(cfg.objective);
    var best_score: f64 = if (better) -std.math.inf(f64) else std.math.inf(f64);
    var best_round: u32 = 0;
    var since_best: u32 = 0;

    var valid_ns: u64 = 0;
    var round: u32 = 0;
    while (round < cfg.n_rounds) : (round += 1) {
        var gctx = GradCtx{
            .raw = raw,
            .labels = ds.labels,
            .grads = grads,
            .scale_pos_weight = cfg.scale_pos_weight,
            .objective = cfg.objective,
        };
        const t_g = prof.start();
        pool.parallelFor(ds.n_rows, &gctx, GradCtx.run, 8192);
        prof.stop(.grad, t_g);

        const subset: ?[]const u32 = if (cfg.sampling == .goss) blk: {
            const t_gs = prof.start();
            defer prof.stop(.goss_select, t_gs);
            break :blk gossSelect(pool, grads, goss_counts, goss_chunks, goss_others, goss_mask, goss_rows, cfg.top_rate, cfg.other_rate, cfg.goss_rank, goss_rng.random());
        } else null;

        if (sym) |*s| {
            var t = try s.grow(grads, raw, ds.labels, cfg.objective, cfg.scale_pos_weight);
            errdefer t.deinit(gpa);
            try model.trees.append(gpa, t);
            const t_ap = prof.start();
            var sctx = SymApplyCtx{ .leaf = s.leaf, .values = s.values, .raw = raw };
            pool.parallelFor(ds.n_rows, &sctx, SymApplyCtx.run, 8192);
            prof.stop(.apply, t_ap);
        } else {
            const builder = &builder_opt.?;
            var t = try builder.growRows(grads, subset);
            errdefer t.deinit(gpa);
            try model.trees.append(gpa, t);

            // Spans cover only rows the tree saw (all rows only without sampling).
            const t_ap = prof.start();
            // Span fast path needs constant leaves; a linear leaf has no single constant.
            if (builder.activeRows().len == ds.n_rows and !cfg.tree.linear_leaves) {
                const spans = span_buf[0..builder.leafSpans().len];
                @memcpy(spans, builder.leafSpans());
                std.sort.pdq(tree.LeafSpan, spans, {}, ApplyCtx.byStart);
                var actx = ApplyCtx{
                    .spans = spans,
                    .rows = builder.rows,
                    .raw = raw,
                };
                pool.parallelFor(builder.activeRows().len, &actx, ApplyCtx.run, 8192);
            } else {
                var actx = ApplyAllCtx{
                    .t = &model.trees.items[model.trees.items.len - 1],
                    .ds = ds,
                    .raw = raw,
                };
                pool.parallelFor(ds.n_rows, &actx, ApplyAllCtx.run, 4096);
            }
            prof.stop(.apply, t_ap);
        }

        if (valid) |v| {
            const wall0 = prof.now();
            const t_vp = prof.start();
            var vctx = ValidCtx{ .t = &model.trees.items[model.trees.items.len - 1], .ds = v, .raw = valid_raw };
            pool.parallelFor(v.n_rows, &vctx, ValidCtx.run, valid_min_chunk);
            prof.stop(.valid_predict, t_vp);

            // Score only when consumed: early stopping (every round), a log line, or the
            // final round (`best_score`). Unguarded, the metric's single-thread sort ran
            // every round and was mostly discarded, a large share of training time.
            const log_due = log != null and cfg.verbose_eval != 0 and
                (round % cfg.verbose_eval == 0 or round + 1 == cfg.n_rounds);
            if (cfg.early_stopping_rounds == 0 and !log_due and round + 1 != cfg.n_rounds) {
                valid_ns += prof.now() - wall0;
                continue;
            }

            const t_vm = prof.start();
            const score = try evaluate(gpa, cfg.objective, valid_raw, v.labels);
            prof.stop(.valid_metric, t_vm);
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

            valid_ns += prof.now() - wall0;

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
        .valid_ns = valid_ns,
    };
}
