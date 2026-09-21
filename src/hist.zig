//! Histogram construction and split search.
//!
//! This is the whole cost of boosting. Two decisions carry the performance:
//!
//!  1. Gradients for a node are gathered into a contiguous buffer **once**,
//!     then reused across all features. The gather is the expensive part of
//!     the inner loop, and a node builds one histogram per feature, so paying
//!     it once instead of per-feature removes most of the random access.
//!
//!  2. Accumulation uses per-worker private histograms and a vectorised
//!     reduction, so the inner loop needs no atomics at all. A private
//!     histogram is tens of KB and stays in that core's L2.
//!
//! Sums are f64 even though gradients are f32: a root node adds hundreds of
//! thousands of terms into a few hundred accumulators, and f32 loses real
//! precision over that many additions.

const std = @import("std");
const Pool = @import("pool.zig").Pool;
const Dataset = @import("data.zig").Dataset;

/// First-order and second-order derivative of the loss at one row.
pub const GradPair = extern struct {
    g: f32,
    h: f32,
};

pub const Bin = extern struct {
    g: f64 = 0,
    h: f64 = 0,
    n: u32 = 0,
    _pad: u32 = 0,

    pub inline fn add(a: Bin, b: Bin) Bin {
        return .{ .g = a.g + b.g, .h = a.h + b.h, .n = a.n + b.n };
    }

    pub inline fn sub(a: Bin, b: Bin) Bin {
        return .{ .g = a.g - b.g, .h = a.h - b.h, .n = a.n - b.n };
    }
};

/// Scratch for one tree's histograms.
///
/// `stride` is the per-feature bin count rounded up so each feature's slice
/// starts on its own cache line, which keeps two workers writing adjacent
/// features off the same line.
pub const Bank = struct {
    gpa: std.mem.Allocator,
    n_workers: usize,
    n_features: usize,
    stride: usize,
    /// `n_workers * n_features * stride`
    private: []Bin,

    /// Bins per stride step, chosen so that `step * @sizeOf(Bin)` is a whole
    /// number of cache lines.
    ///
    /// The obvious `cache_line / @sizeOf(Bin)` is wrong, and wrong in a way
    /// that only shows up on some CPUs. `Bin` is 24 bytes, and Zig reports a
    /// 128-byte cache line on Zen 5 (64 on much else), so that expression is
    /// 128/24 = 5 here — not a power of two, which `alignForward` requires.
    /// Debug catches it with an assert; ReleaseFast elides the assert and
    /// silently produces a stride smaller than `max_bins`, overlapping the
    /// per-feature histogram slices and corrupting every split search.
    ///
    /// Dividing by the gcd gives the smallest step whose byte size is a cache
    /// line multiple: 128/gcd(24,128) = 16 bins here, 8 where the line is 64.
    const bin_step: usize = @max(1, std.atomic.cache_line / std.math.gcd(@sizeOf(Bin), std.atomic.cache_line));

    pub fn init(gpa: std.mem.Allocator, n_workers: usize, n_features: usize, max_bins: usize) !Bank {
        // Plain multiple-rounding, not alignForward: `bin_step` is a count of
        // bins and carries no power-of-two guarantee.
        const stride = ((max_bins + bin_step - 1) / bin_step) * bin_step;
        const private = try gpa.alloc(Bin, n_workers * n_features * stride);
        return .{
            .gpa = gpa,
            .n_workers = n_workers,
            .n_features = n_features,
            .stride = stride,
            .private = private,
        };
    }

    pub fn deinit(b: *Bank) void {
        b.gpa.free(b.private);
        b.* = undefined;
    }

    /// Bins in one node's histogram slot.
    pub fn slotLen(b: *const Bank) usize {
        return b.n_features * b.stride;
    }

    pub inline fn featureSlice(b: *const Bank, hist: []Bin, f: usize) []Bin {
        return hist[f * b.stride ..][0..b.stride];
    }

};

const BuildCtx = struct {
    bank: *Bank,
    ds: *const Dataset,
    /// Row ids belonging to this node, contiguous.
    rows: []const u32,
    /// `grads[i]` is the gradient of `rows[i]` — pre-gathered, so this is read
    /// sequentially.
    grads: []const GradPair,
    features: []const u32,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        const self: *BuildCtx = @ptrCast(@alignCast(ctx));
        const bank = self.bank;
        const base = worker * bank.n_features * bank.stride;
        const mine = bank.private[base..][0 .. bank.n_features * bank.stride];

        for (self.features) |fid| {
            const col = self.ds.column(fid);
            const h = mine[fid * bank.stride ..][0..bank.stride];
            var i = begin;
            // Unrolled by four. The accumulators are independent unless two
            // rows share a bin, so this exposes enough ILP to hide the
            // read-modify-write latency in the common case.
            while (i + 4 <= end) : (i += 4) {
                const b0 = col[self.rows[i + 0]];
                const b1 = col[self.rows[i + 1]];
                const b2 = col[self.rows[i + 2]];
                const b3 = col[self.rows[i + 3]];
                const g0 = self.grads[i + 0];
                const g1 = self.grads[i + 1];
                const g2 = self.grads[i + 2];
                const g3 = self.grads[i + 3];
                h[b0].g += g0.g;
                h[b0].h += g0.h;
                h[b0].n += 1;
                h[b1].g += g1.g;
                h[b1].h += g1.h;
                h[b1].n += 1;
                h[b2].g += g2.g;
                h[b2].h += g2.h;
                h[b2].n += 1;
                h[b3].g += g3.g;
                h[b3].h += g3.h;
                h[b3].n += 1;
            }
            while (i < end) : (i += 1) {
                const b = col[self.rows[i]];
                const g = self.grads[i];
                h[b].g += g.g;
                h[b].h += g.h;
                h[b].n += 1;
            }
        }
    }
};

/// Reduces over a flat index space of `features.len * stride`, so the whole
/// merge is one barrier rather than one per feature.
const ReduceCtx = struct {
    bank: *Bank,
    features: []const u32,
    active: usize,
    out: []Bin,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *ReduceCtx = @ptrCast(@alignCast(ctx));
        const bank = self.bank;
        const stride = bank.stride;
        const span = bank.n_features * stride;

        var i = begin;
        while (i < end) {
            const fi = i / stride;
            const run_end = @min(end, (fi + 1) * stride);
            const base = @as(usize, self.features[fi]) * stride + (i % stride);
            var idx = base;
            while (i < run_end) : ({
                i += 1;
                idx += 1;
            }) {
                var acc = bank.private[idx];
                var w: usize = 1;
                while (w < self.active) : (w += 1) {
                    acc = acc.add(bank.private[w * span + idx]);
                }
                self.out[idx] = acc;
            }
        }
    }
};

/// Below this many rows a node is not worth fanning out: the barrier and the
/// private-histogram merge cost more than the accumulation itself.
pub const parallel_threshold: usize = 8192;

/// Build the histogram for one node into `out`, which must be one slot's
/// worth (`n_features * stride`).
///
/// Only `features` are populated; the caller must not read the others.
pub fn build(
    pool: *Pool,
    bank: *Bank,
    ds: *const Dataset,
    rows: []const u32,
    grads: []const GradPair,
    features: []const u32,
    out: []Bin,
) void {
    const span = bank.n_features * bank.stride;

    // Whether to fan out is decided here, not left to the pool, because the
    // clear and the merge below must cover exactly the workers that ran. If
    // those disagreed, a stale private histogram would be merged in.
    const parallel = bank.n_workers > 1 and rows.len > parallel_threshold;
    const active: usize = if (parallel) bank.n_workers else 1;

    // Clearing every worker's copy of every feature would dominate for a
    // shallow node, so only the features in play are touched.
    for (0..active) |w| {
        for (features) |fid| {
            @memset(bank.private[w * span + fid * bank.stride ..][0..bank.stride], .{});
        }
    }

    var bctx = BuildCtx{
        .bank = bank,
        .ds = ds,
        .rows = rows,
        .grads = grads,
        .features = features,
    };
    if (parallel) {
        pool.parallelFor(rows.len, &bctx, BuildCtx.run, parallel_threshold);
    } else {
        BuildCtx.run(&bctx, 0, 0, rows.len);
    }

    if (active == 1) {
        for (features) |fid| {
            const lo = fid * bank.stride;
            @memcpy(out[lo..][0..bank.stride], bank.private[lo..][0..bank.stride]);
        }
        return;
    }

    var rctx = ReduceCtx{ .bank = bank, .features = features, .active = active, .out = out };
    pool.parallelFor(features.len * bank.stride, &rctx, ReduceCtx.run, 512);
}

/// `out = parent - sibling`, the subtraction trick: a node's histogram is
/// derivable from its parent and its sibling, so only the cheaper child of
/// each pair ever needs a real build.
const SubCtx = struct {
    bank: *Bank,
    features: []const u32,
    out: []Bin,
    parent: []const Bin,
    sibling: []const Bin,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *SubCtx = @ptrCast(@alignCast(ctx));
        const stride = self.bank.stride;
        var i = begin;
        while (i < end) : (i += 1) {
            const fid = self.features[i / stride];
            const k = fid * stride + (i % stride);
            self.out[k] = self.parent[k].sub(self.sibling[k]);
        }
    }
};

/// A node's histogram is its parent's minus its sibling's.
///
/// Worth parallelising despite being pure memory traffic: it runs once per
/// internal node, and at 256 bins over a dozen features that is enough bytes
/// per node to show up plainly in a profile of tree building.
pub fn subtract(
    pool: *Pool,
    bank: *Bank,
    out: []Bin,
    parent: []const Bin,
    sibling: []const Bin,
    features: []const u32,
) void {
    var ctx = SubCtx{
        .bank = bank,
        .features = features,
        .out = out,
        .parent = parent,
        .sibling = sibling,
    };
    pool.parallelFor(features.len * bank.stride, &ctx, SubCtx.run, 1024);
}

// ------------------------------------------------------------ split search

pub const Split = struct {
    feature: u32 = 0,
    /// Rows with `bin <= threshold` go left (before the missing adjustment).
    threshold: u8 = 0,
    /// Whether the missing bin joins the left child.
    missing_left: bool = true,
    gain: f64 = -std.math.inf(f64),
    left: Bin = .{},
    right: Bin = .{},

    pub fn valid(s: Split) bool {
        return s.gain > 0;
    }
};

pub const SplitParams = struct {
    lambda: f64,
    alpha: f64,
    min_split_gain: f64,
    min_child_weight: f64,
    min_child_samples: u32,
    max_delta_step: f64,
};

inline fn softThreshold(g: f64, alpha: f64) f64 {
    if (alpha == 0) return g;
    if (g > alpha) return g - alpha;
    if (g < -alpha) return g + alpha;
    return 0;
}

/// Structure score of a node: the loss reduction its optimal leaf weight buys.
pub inline fn nodeScore(g: f64, h: f64, p: SplitParams) f64 {
    const t = softThreshold(g, p.alpha);
    return (t * t) / (h + p.lambda);
}

/// Optimal leaf weight under L1/L2 with an optional trust-region cap.
pub inline fn leafWeight(g: f64, h: f64, p: SplitParams) f64 {
    var w = -softThreshold(g, p.alpha) / (h + p.lambda);
    if (p.max_delta_step > 0) w = std.math.clamp(w, -p.max_delta_step, p.max_delta_step);
    return w;
}

inline fn consider(
    best: *Split,
    fid: u32,
    thr: u8,
    missing_left: bool,
    left: Bin,
    right: Bin,
    parent_score: f64,
    p: SplitParams,
) void {
    if (left.n < p.min_child_samples or right.n < p.min_child_samples) return;
    if (left.h < p.min_child_weight or right.h < p.min_child_weight) return;
    const gain = 0.5 * (nodeScore(left.g, left.h, p) + nodeScore(right.g, right.h, p) - parent_score) - p.min_split_gain;
    if (gain > best.gain) {
        best.* = .{
            .feature = fid,
            .threshold = thr,
            .missing_left = missing_left,
            .gain = gain,
            .left = left,
            .right = right,
        };
    }
}

/// Best split for one node over `features`.
///
/// Bin 0 holds the missing mass and is never itself a threshold. Each feature
/// is scanned twice — once sending missing left, once right — which is how the
/// default direction is learned per split rather than fixed in advance.
pub fn bestSplit(
    bank: *const Bank,
    hist: []const Bin,
    ds: *const Dataset,
    features: []const u32,
    total: Bin,
    p: SplitParams,
) Split {
    var best: Split = .{};
    const parent_score = nodeScore(total.g, total.h, p);

    for (features) |fid| {
        const h = hist[fid * bank.stride ..][0..bank.stride];
        const nb = ds.n_bins[fid];
        if (nb < 3) continue; // missing bin plus one real bin: nothing to cut
        const missing = h[0];

        // missing -> left
        var acc = missing;
        var b: usize = 1;
        while (b + 1 < nb) : (b += 1) {
            acc = acc.add(h[b]);
            consider(&best, fid, @intCast(b), true, acc, total.sub(acc), parent_score, p);
        }

        // missing -> right
        acc = .{};
        b = 1;
        while (b + 1 < nb) : (b += 1) {
            acc = acc.add(h[b]);
            consider(&best, fid, @intCast(b), false, acc, total.sub(acc), parent_score, p);
        }
    }
    return best;
}
