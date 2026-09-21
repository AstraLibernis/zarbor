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
/// Features get **packed per-feature offsets**, not a uniform stride sized by
/// the widest feature. That distinction is worth more than it sounds. On a
/// typical table the widths are wildly uneven — 234 and 202 bins for two
/// continuous columns, but 3 or 4 for the categoricals — so a uniform stride
/// allocated 13x240 = 3120 slots for 558 real bins, a 5.6x waste.
///
/// The waste was not merely memory. Clearing and reducing the private
/// histograms is a *fixed* cost per node, paid whatever the node's size, so it
/// dominated exactly where most of the nodes are: the bottom of the tree.
/// Measured per node, packing is 1.08x at 500k rows but 3.15x at 8k and 4.09x
/// at 2k, and a depth-6 tree keeps half its internal nodes in the last level.
///
/// Each feature still starts on a cache-line boundary, which keeps the clear
/// and the reduce off partial lines.
pub const Bank = struct {
    gpa: std.mem.Allocator,
    n_workers: usize,
    n_features: usize,
    /// `offsets[f]..offsets[f+1]` is feature `f`'s bin range within a slot.
    /// Length `n_features + 1`; the last entry is the slot length.
    offsets: []u32,
    /// `n_workers * slotLen()`
    private: []Bin,

    /// Bins per alignment step, chosen so `step * @sizeOf(Bin)` is a whole
    /// number of cache lines.
    ///
    /// The obvious `cache_line / @sizeOf(Bin)` is wrong, and wrong in a way
    /// that only shows up on some CPUs. `Bin` is 24 bytes, and Zig reports a
    /// 128-byte cache line on Zen 5 (64 on much else), so that expression is
    /// 128/24 = 5 here — not a power of two, which `alignForward` requires.
    /// Debug catches it with an assert; ReleaseFast elides the assert and
    /// silently produces a stride smaller than the bin count, overlapping the
    /// per-feature histogram slices and corrupting every split search.
    ///
    /// Dividing by the gcd gives the smallest step whose byte size is a cache
    /// line multiple: 128/gcd(24,128) = 16 bins here, 8 where the line is 64.
    const bin_step: usize = @max(1, std.atomic.cache_line / std.math.gcd(@sizeOf(Bin), std.atomic.cache_line));

    inline fn roundUp(n: usize) usize {
        // Plain multiple-rounding, not alignForward: `bin_step` is a count of
        // bins and carries no power-of-two guarantee.
        return ((n + bin_step - 1) / bin_step) * bin_step;
    }

    pub fn init(
        gpa: std.mem.Allocator,
        n_workers: usize,
        n_features: usize,
        n_bins: []const u16,
    ) !Bank {
        std.debug.assert(n_bins.len == n_features);
        const offsets = try gpa.alloc(u32, n_features + 1);
        errdefer gpa.free(offsets);
        var acc: usize = 0;
        for (0..n_features) |f| {
            offsets[f] = @intCast(acc);
            acc += roundUp(n_bins[f]);
        }
        offsets[n_features] = @intCast(acc);

        const private = try gpa.alloc(Bin, n_workers * acc);
        return .{
            .gpa = gpa,
            .n_workers = n_workers,
            .n_features = n_features,
            .offsets = offsets,
            .private = private,
        };
    }

    pub fn deinit(b: *Bank) void {
        b.gpa.free(b.private);
        b.gpa.free(b.offsets);
        b.* = undefined;
    }

    /// Bins in one node's histogram slot.
    pub fn slotLen(b: *const Bank) usize {
        return b.offsets[b.n_features];
    }

    pub inline fn featureSlice(b: *const Bank, hist: []Bin, f: usize) []Bin {
        return hist[b.offsets[f]..b.offsets[f + 1]];
    }

    pub inline fn featureSliceConst(b: *const Bank, hist: []const Bin, f: usize) []const Bin {
        return hist[b.offsets[f]..b.offsets[f + 1]];
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
        const span = bank.slotLen();
        const mine = bank.private[worker * span ..][0..span];

        for (self.features) |fid| {
            const col = self.ds.column(fid);
            const h = mine[bank.offsets[fid]..bank.offsets[fid + 1]];
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

/// Reduces the private histograms into `out` over the whole slot.
///
/// Reducing every feature rather than only the selected ones is deliberate
/// now that the slot is packed: it is a flat, branch-free loop over ~16 KB,
/// and skipping features would cost more in index arithmetic than it saves.
/// Unselected features end up holding whatever their private copies did,
/// which the caller already must not read.
const ReduceCtx = struct {
    bank: *Bank,
    active: usize,
    out: []Bin,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *ReduceCtx = @ptrCast(@alignCast(ctx));
        const span = self.bank.slotLen();
        var i = begin;
        while (i < end) : (i += 1) {
            var acc = self.bank.private[i];
            var w: usize = 1;
            while (w < self.active) : (w += 1) acc = acc.add(self.bank.private[w * span + i]);
            self.out[i] = acc;
        }
    }
};

/// Below this many rows a node is built on one thread.
///
/// Tuned, not guessed. The cost this trades against is the per-node fixed
/// cost — clearing every worker's private histogram and reducing them — which
/// packing the bins cut by ~4.5x. That made fanning out worthwhile at much
/// smaller nodes than before: sweeping the threshold, 8192 (the old value)
/// costs 1516 ms on a 200-tree fit where 512 costs 1213 ms, and the curve is
/// flat between 256 and 768 before rising again below 128, where the chunks
/// get too small for the barrier. Output is identical at every setting.
pub const parallel_threshold: usize = 512;

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
    const span = bank.slotLen();

    // Whether to fan out is decided here, not left to the pool, because the
    // clear and the merge below must cover exactly the workers that ran. If
    // those disagreed, a stale private histogram would be merged in.
    const parallel = bank.n_workers > 1 and rows.len > parallel_threshold;
    const active: usize = if (parallel) bank.n_workers else 1;

    // One flat clear per worker. With a packed slot this is ~16 KB, which is
    // cheaper than the old per-feature clear of only the selected features
    // was over a uniform stride.
    @memset(bank.private[0 .. active * span], .{});

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
        @memcpy(out[0..span], bank.private[0..span]);
        return;
    }

    var rctx = ReduceCtx{ .bank = bank, .active = active, .out = out };
    pool.parallelFor(span, &rctx, ReduceCtx.run, 512);
}

const SubCtx = struct {
    out: []Bin,
    parent: []const Bin,
    sibling: []const Bin,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *SubCtx = @ptrCast(@alignCast(ctx));
        var i = begin;
        while (i < end) : (i += 1) self.out[i] = self.parent[i].sub(self.sibling[i]);
    }
};

/// A node's histogram is its parent's minus its sibling's.
///
/// Flat over the packed slot: one contiguous pass, no per-feature indexing.
pub fn subtract(pool: *Pool, bank: *Bank, out: []Bin, parent: []const Bin, sibling: []const Bin) void {
    const span = bank.slotLen();
    var ctx = SubCtx{ .out = out[0..span], .parent = parent[0..span], .sibling = sibling[0..span] };
    pool.parallelFor(span, &ctx, SubCtx.run, 1024);
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
        const h = bank.featureSliceConst(hist, fid);
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
