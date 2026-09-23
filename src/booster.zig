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
const prof = @import("prof.zig");

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
        const out = self.out[begin..end];
        // Trees outer, rows inner; see the note in `model.zig`'s TreeCtx.
        // Bit-exact: same trees, same order, same f32 rounding per row.
        @memset(out, self.m.base_score);
        for (self.m.trees.items) |t| {
            for (out, begin..) |*o, r| o.* += t.predictBinned(self.ds, r);
        }
    }
};

pub inline fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// Lanes in the vectorised gradient loop. Eight f32 is one AVX2 register.
const lanes = 8;
const F8 = @Vector(lanes, f32);

/// `sigmoid` for eight rows at once.
///
/// `@exp` on a scalar is a libm call — 7.8 ns an element here, and glibc's
/// `expf` is no faster, so the cost is scalar transcendental evaluation
/// rather than anyone's implementation. Computing it inline as
/// `2^(-x*log2e)`, with the integer part folded into the exponent field and
/// the fraction from a degree-5 minimax polynomial, vectorises to 0.67 ns an
/// element: 11.7x, and gradients were 15% of a fit.
///
/// Accurate to under 1e-6 absolute, which `"vectorised sigmoid matches the
/// scalar one"` pins. That is a few f32 ulp, and it feeds gradients that are
/// themselves stored as f32 — but it is an approximation, so it is confined
/// to the gradient loop. Predictions, which are what a caller actually reads,
/// go through the scalar `sigmoid`.
inline fn sigmoid8(x: F8) F8 {
    const one: F8 = @splat(1.0);
    // exp overflows f32 past ~88; clamping there costs nothing since sigmoid
    // has long since saturated.
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

/// The same approximation for a single row.
///
/// The tail of the vectorised loop must not use the *exact* `sigmoid`: chunk
/// boundaries move with the thread count, so a row would get a different
/// gradient at one thread than at eight, and the model would quietly depend
/// on `--n_threads`. Same function, one lane.
inline fn sigmoid1(x: f32) f32 {
    const v: F8 = @splat(x);
    return sigmoid8(v)[0];
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
                // GradPair is {g, h} interleaved, so write it a row at a time
                // rather than trying to scatter two vectors into it.
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

/// Buckets in one radix pass over a magnitude's bit pattern.
pub const radix_bits = 16;
pub const n_radix = 1 << radix_bits;

/// `|g|` as a sortable integer.
///
/// For a non-negative float the IEEE-754 bit pattern is monotonic in the
/// value, so the pattern can be bucketed directly with no conversion. NaN
/// sorts above everything, which is where a blown-up gradient belongs.
inline fn magBits(g: f32) u32 {
    return @bitCast(@abs(g));
}

/// The cut separating the `k` largest magnitudes from the rest.
pub const Cut = struct {
    /// Rows whose pattern is strictly greater are all in.
    t: u32,
    /// How many rows with pattern exactly `t` are in, taken in row order.
    take: usize,
};

/// Find the cut with two counting passes over the magnitude bit patterns.
///
/// This replaced a quickselect that was 768 ms of a 1.57 s GOSS fit — 49% of
/// the run, on a step that is not part of the model. The problem was not its
/// O(n): Hoare's two scans branch on a comparison that is a coin flip by
/// construction, so nearly every element cost a misprediction. Counting
/// passes branch predictably and stream the input, and sixteen bits at a time
/// means two of them pin the threshold exactly.
///
/// Exact, not approximate: afterwards every chosen row's magnitude is >= every
/// unchosen row's, with exact bit-ties broken by row order.
pub fn gossCut(grads: []const hist.GradPair, counts: []u32, k: usize) Cut {
    std.debug.assert(counts.len == n_radix);
    std.debug.assert(k >= 1 and k <= grads.len);

    @memset(counts, 0);
    for (grads) |g| counts[magBits(g.g) >> radix_bits] += 1;

    var above: usize = 0;
    var hi: u32 = n_radix - 1;
    while (true) {
        const c = counts[hi];
        if (above + c >= k or hi == 0) break;
        above += c;
        hi -= 1;
    }

    @memset(counts, 0);
    const lo_mask: u32 = n_radix - 1;
    for (grads) |g| {
        const b = magBits(g.g);
        if (b >> radix_bits == hi) counts[b & lo_mask] += 1;
    }

    var lo: u32 = n_radix - 1;
    while (true) {
        const c = counts[lo];
        if (above + c >= k or lo == 0) break;
        above += c;
        lo -= 1;
    }

    return .{ .t = (hi << radix_bits) | lo, .take = k - above };
}

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
    counts: []u32,
    others: []u32,
    mask: []u64,
    out: []u32,
    top_rate: f32,
    other_rate: f32,
    rng: std.Random,
) []u32 {
    const n = grads.len;
    var top: usize = @intFromFloat(@round(@as(f32, @floatFromInt(n)) * top_rate));
    top = std.math.clamp(top, 1, n);
    const rest = n - top;

    const cut = gossCut(grads, counts, top);

    // One ordered pass: mark the large-gradient rows and collect the rest, so
    // the random sample below has something contiguous to draw from.
    const words = (n + 63) / 64;
    @memset(mask[0..words], 0);
    var take = cut.take;
    var n_other: usize = 0;
    for (grads, 0..) |g, r| {
        const b = magBits(g.g);
        const keep = b > cut.t or (b == cut.t and take > 0);
        if (keep) {
            if (b == cut.t) take -= 1;
            mask[r >> 6] |= @as(u64, 1) << @truncate(r);
        } else {
            others[n_other] = @intCast(r);
            n_other += 1;
        }
    }

    var rand_n: usize = @intFromFloat(@round(@as(f32, @floatFromInt(n)) * other_rate));
    rand_n = @min(rand_n, @min(rest, n_other));

    // Partial Fisher-Yates over the unchosen rows.
    var i: usize = 0;
    while (i < rand_n) : (i += 1) {
        const j = i + rng.uintLessThan(usize, n_other - i);
        std.mem.swap(u32, &others[i], &others[j]);
    }

    const amp: f32 = (1.0 - top_rate) / other_rate;
    for (others[0..rand_n]) |r| {
        grads[r].g *= amp;
        grads[r].h *= amp;
        mask[r >> 6] |= @as(u64, 1) << @truncate(r);
    }

    // Emit in ascending row order, from the bitset rather than by sorting.
    // A scrambled row set is the worst case for the histogram kernel, which
    // reads the row-major bin matrix and wants a forward walk; sorting 160k
    // ids would cost more than the selection does.
    var k: usize = 0;
    for (mask[0..words], 0..) |w0, wi| {
        var w = w0;
        while (w != 0) {
            out[k] = @intCast(wi * 64 + @ctz(w));
            k += 1;
            w &= w - 1;
        }
    }
    return out[0..k];
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
    /// Best validation score seen. With `early_stopping_rounds == 0` the
    /// metric is only computed on logged rounds and the last one, so this is
    /// the best of *those*, not of every round. NaN with no validation set.
    best_score: f64,
    /// Nanoseconds spent predicting and scoring the validation set.
    ///
    /// Separated because it is not part of fitting the model, and a
    /// comparison against a library called without an eval set would
    /// otherwise charge us for work it never did. Always measured, not only
    /// under `--profile`: two clock reads a round cost nothing.
    valid_ns: u64,
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

    var goss_counts: []u32 = &.{};
    var goss_others: []u32 = &.{};
    var goss_mask: []u64 = &.{};
    var goss_rows: []u32 = &.{};
    if (cfg.sampling == .goss) {
        goss_counts = try gpa.alloc(u32, n_radix);
        goss_others = try gpa.alloc(u32, ds.n_rows);
        goss_mask = try gpa.alloc(u64, (ds.n_rows + 63) / 64);
        goss_rows = try gpa.alloc(u32, ds.n_rows);
    }
    defer if (goss_counts.len != 0) gpa.free(goss_counts);
    defer if (goss_others.len != 0) gpa.free(goss_others);
    defer if (goss_mask.len != 0) gpa.free(goss_mask);
    defer if (goss_rows.len != 0) gpa.free(goss_rows);
    var goss_rng: std.Random.DefaultPrng = .init(cfg.seed +% 0x9E3779B97F4A7C15);

    var builder = try tree.Builder.init(gpa, pool, ds, cfg);
    defer builder.deinit();

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

        const subset: ?[]const u32 = if (cfg.sampling == .goss)
            blk: {
                const t_gs = prof.start();
                defer prof.stop(.goss_select, t_gs);
                break :blk gossSelect(grads, goss_counts, goss_others, goss_mask, goss_rows, cfg.top_rate, cfg.other_rate, goss_rng.random());
            }
        else
            null;

        var t = try builder.growRows(grads, subset);
        errdefer t.deinit(gpa);
        try model.trees.append(gpa, t);

        // Spans only cover the rows the tree saw; that is every row only when
        // nothing was sampled away.
        const t_ap = prof.start();
        // A linear leaf has no single constant to add to a whole span, so
        // the span fast path is only valid for constant leaves.
        if (builder.activeRows().len == ds.n_rows and !cfg.linear_leaves) {
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
        prof.stop(.apply, t_ap);

        if (valid) |v| {
            const wall0 = prof.now();
            const t_vp = prof.start();
            var vctx = ValidCtx{ .t = &model.trees.items[model.trees.items.len - 1], .ds = v, .raw = valid_raw };
            pool.parallelFor(v.n_rows, &vctx, ValidCtx.run, 4096);
            prof.stop(.valid_predict, t_vp);

            // Only score when something will actually consume it. Early
            // stopping needs every round; a log line needs its own round; the
            // final round fills in `best_score` for the caller. Without this
            // guard the metric ran unconditionally, and on 668k rows that was
            // 52% of total training time — a full 133k-row sort per round,
            // single-threaded, discarded 199 times out of 200.
            const log_due = log != null and cfg.verbose_eval != 0 and
                (round % cfg.verbose_eval == 0 or round + 1 == cfg.n_rounds);
            if (cfg.early_stopping_rounds == 0 and !log_due and round + 1 != cfg.n_rounds) {
                valid_ns += prof.now() - wall0;
                continue;
            }

            const t_vm = prof.start();
            const score = try evaluate(gpa, cfg.objective, valid_raw, v.labels, valid_scratch);
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

const testing = std.testing;

test "vectorised sigmoid matches the scalar one" {
    // The gradient loop uses an inlined 2^x rather than libm's expf, which is
    // 11.7x faster and approximate. This pins how approximate: anything worse
    // than a few f32 ulp would start moving split decisions around.
    var max_err: f32 = 0;
    var x: f32 = -60.0;
    while (x <= 60.0) : (x += 0.0007) {
        const want = sigmoid(x);
        const got = sigmoid1(x);
        max_err = @max(max_err, @abs(got - want));
    }
    try testing.expect(max_err < 1e-6);

    // Saturation, and no NaN where exp would overflow.
    for ([_]f32{ -1e30, -1000, -89, 0, 89, 1000, 1e30 }) |v| {
        const got = sigmoid1(v);
        try testing.expect(!std.math.isNan(got));
        try testing.expect(got >= 0.0 and got <= 1.0);
        try testing.expect(@abs(got - sigmoid(v)) < 1e-6);
    }
    try testing.expectApproxEqAbs(@as(f32, 0.5), sigmoid1(0), 1e-7);

    // Every lane must agree with lane 0, or the tail of the gradient loop
    // would not match its body.
    const v = F8{ -3.25, -1.0, -0.125, 0, 0.125, 1.0, 3.25, 7.5 };
    const got: [lanes]f32 = sigmoid8(v);
    const src: [lanes]f32 = v;
    for (got, src) |g, sx| try testing.expectEqual(sigmoid1(sx), g);
}

test "gossCut separates exactly the k largest |g|" {
    // A selection step is easy to get subtly wrong — off by one at the
    // threshold, or wrong when every value is identical. Checked against a
    // full sort on random, already-sorted, reverse-sorted, all-equal and
    // sign-mixed inputs, since boosting produces all of them.
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(4);
    const r = prng.random();

    const counts = try gpa.alloc(u32, n_radix);
    defer gpa.free(counts);

    const shapes = enum { random, ascending, descending, all_equal, signed, tiny_spread };
    for (std.enums.values(shapes)) |shape| {
        for ([_]usize{ 1, 2, 7, 64, 1000, 5000 }) |n| {
            const grads = try gpa.alloc(hist.GradPair, n);
            defer gpa.free(grads);
            for (grads, 0..) |*g, i| {
                const v: f32 = switch (shape) {
                    .random => r.floatNorm(f32),
                    .ascending => @floatFromInt(i),
                    .descending => @floatFromInt(n - i),
                    .all_equal => 1.0,
                    // Magnitude is what is ranked, so signs must not matter.
                    .signed => if (i % 2 == 0) -r.float(f32) else r.float(f32),
                    // Values inside one radix bucket, which is what forces the
                    // second pass to do the work.
                    .tiny_spread => 1.0 + @as(f32, @floatFromInt(i % 3)) * 1e-7,
                };
                g.* = .{ .g = v, .h = 1 };
            }

            const mags = try gpa.alloc(f32, n);
            defer gpa.free(mags);
            for (mags, grads) |*m, g| m.* = @abs(g.g);
            std.sort.pdq(f32, mags, {}, std.sort.desc(f32));

            for ([_]usize{ 1, n / 3, n / 2, n - 1, n }) |k| {
                if (k == 0 or k > n) continue;
                const cut = gossCut(grads, counts, k);

                // Replay the selection rule the caller uses.
                const chosen = try gpa.alloc(bool, n);
                defer gpa.free(chosen);
                var take = cut.take;
                var n_chosen: usize = 0;
                for (grads, 0..) |g, i| {
                    const b: u32 = @bitCast(@abs(g.g));
                    const keep = b > cut.t or (b == cut.t and take > 0);
                    chosen[i] = keep;
                    if (keep) {
                        if (b == cut.t) take -= 1;
                        n_chosen += 1;
                    }
                }

                // Exactly k rows, and every one of them at least as large as
                // every row left behind. That is the whole contract.
                try testing.expectEqual(k, n_chosen);
                const kth = mags[k - 1];
                for (grads, chosen) |g, c| {
                    if (c) {
                        try testing.expect(@abs(g.g) >= kth);
                    } else {
                        try testing.expect(@abs(g.g) <= kth);
                    }
                }
            }
        }
    }
}
