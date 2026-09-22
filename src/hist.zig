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
const data = @import("data.zig");
const Dataset = data.Dataset;
const prof = @import("prof.zig");

/// First-order and second-order derivative of the loss at one row.
pub const GradPair = extern struct {
    g: f32,
    h: f32,
};

/// One histogram cell: gradient sum, hessian sum, row count.
///
/// Four f64 lanes, 32-byte aligned, rather than two doubles and a `u32`
/// count. The count in a lane makes the whole update a *single* aligned
/// vector load-add-store instead of three separate scattered
/// read-modify-writes, and the dead fourth lane pays for itself by making the
/// element size a power of two, so indexing is a shift rather than a multiply.
///
/// Measured against the 24-byte shape on the real bin widths: 1.35x on its
/// own, and it composes with the row-major layout rather than overlapping
/// with it (1.74x layout alone, 2.34x together, every node size from 2k to
/// 500k rows).
pub const Bin = extern struct {
    g: f64 align(32) = 0,
    h: f64 = 0,
    n: f64 = 0,
    _pad: f64 = 0,

    pub const Vec = @Vector(4, f64);

    pub inline fn vec(b: Bin) Vec {
        return @bitCast(b);
    }

    pub inline fn fromVec(x: Vec) Bin {
        return @bitCast(x);
    }

    pub inline fn add(a: Bin, b: Bin) Bin {
        return fromVec(a.vec() + b.vec());
    }

    pub inline fn sub(a: Bin, b: Bin) Bin {
        return fromVec(a.vec() - b.vec());
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
    /// Build sequence number each worker last cleared its slot for, one per
    /// cache line so the stamps do not share a line.
    ///
    /// Clearing the private slots was 108 ms of a 773 ms fit — 15%, on the
    /// main thread, and entirely overhead: 147 KB memset per node build,
    /// 6400 of them. A worker now clears its own slot on its first chunk, so
    /// the clear is spread across the cores that are about to use it, and a
    /// worker that never got a chunk is neither cleared nor reduced.
    stamps: []u64,
    /// Scratch for the workers that took part in the last build.
    parts: []usize,
    seq: u64 = 0,

    const stamp_stride: usize = @max(1, std.atomic.cache_line / @sizeOf(u64));

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
        errdefer gpa.free(private);
        const stamps = try gpa.alloc(u64, n_workers * stamp_stride);
        errdefer gpa.free(stamps);
        @memset(stamps, 0);
        const parts = try gpa.alloc(usize, n_workers);
        return .{
            .gpa = gpa,
            .n_workers = n_workers,
            .n_features = n_features,
            .offsets = offsets,
            .private = private,
            .stamps = stamps,
            .parts = parts,
        };
    }

    pub fn deinit(b: *Bank) void {
        b.gpa.free(b.private);
        b.gpa.free(b.stamps);
        b.gpa.free(b.parts);
        b.gpa.free(b.offsets);
        b.* = undefined;
    }

    inline fn stamp(b: *const Bank, worker: usize) *u64 {
        return &b.stamps[worker * stamp_stride];
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

/// Row-outer accumulation over the row-major bin matrix.
///
/// The obvious loop is feature-outer over the column-major bins, and it is
/// what this was for a long time. Two things make row-outer 1.74x faster on
/// real data:
///
///  * A row's bins are one contiguous run, so `rows[i]` and `grads[i]` are
///    read once per row instead of once per row *per feature*.
///  * The updates within a row hit thirteen different feature histograms, so
///    they are independent. Feature-outer updates one histogram repeatedly,
///    and on a 3-bin categorical consecutive rows collide constantly, which
///    serialises the whole loop on store-to-load forwarding.
///
/// The second point is why this was measured at only 1.07x once before: that
/// benchmark gave every feature 256 bins, where collisions are rare and the
/// histogram spills to L2. Real tables are lopsided — here two columns hold
/// 437 of the 559 bins and the other eleven have three to seven each.
///
/// No prefetching and no unrolling: both measured slower. The rows arrive
/// ascending from the stable partition, so the hardware prefetcher already
/// has the stream, and extra rows in flight only add register pressure.
const BuildCtx = struct {
    bank: *Bank,
    ds: *const Dataset,
    /// Row ids belonging to this node, contiguous.
    rows: []const u32,
    /// Gradients indexed by original row id, so `grads[rows[i]]`. Not
    /// permuted to match `rows`: keeping the permutation in step tripled what
    /// the partition had to move, and the gather here measures at 4 ms across
    /// a 200-tree fit.
    grads: []const GradPair,
    features: []const u32,
    seq: u64,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        const self: *BuildCtx = @ptrCast(@alignCast(ctx));
        const bank = self.bank;
        const span = bank.slotLen();
        const mine = bank.private[worker * span ..][0..span];
        // First chunk of this build for this worker: clear, and leave the
        // stamp behind so the reduce knows this slot holds data. Only the
        // worker itself writes its stamp, and the pool's barrier publishes it.
        const st = bank.stamp(worker);
        if (st.* != self.seq) {
            @memset(mine, .{});
            st.* = self.seq;
        }
        const nf = self.ds.n_features;
        const rm = self.ds.bins_rm;
        const offs = bank.offsets;

        var i = begin;
        while (i < end) : (i += 1) {
            const row: usize = self.rows[i];
            const rb = rm[row * nf ..][0..nf];
            const g = self.grads[row];
            const v = Bin.Vec{ g.g, g.h, 1, 0 };
            for (self.features) |fid| {
                const cell: *Bin.Vec = @ptrCast(&mine[offs[fid] + rb[fid]]);
                cell.* += v;
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
    /// Workers that actually took a chunk, ascending.
    parts: []const usize,
    out: []Bin,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *ReduceCtx = @ptrCast(@alignCast(ctx));
        const span = self.bank.slotLen();
        const parts = self.parts;
        var i = begin;
        while (i < end) : (i += 1) {
            var acc = self.bank.private[parts[0] * span + i];
            for (parts[1..]) |w| acc = acc.add(self.bank.private[w * span + i]);
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

    // The clear happens inside the accumulate, on whichever workers take a
    // chunk; see `Bank.stamps`.
    bank.seq += 1;
    var bctx = BuildCtx{
        .bank = bank,
        .ds = ds,
        .rows = rows,
        .grads = grads,
        .features = features,
        .seq = bank.seq,
    };
    const t_a = prof.start();
    if (parallel) {
        pool.parallelFor(rows.len, &bctx, BuildCtx.run, parallel_threshold);
    } else {
        BuildCtx.run(&bctx, 0, 0, rows.len);
    }
    prof.stop(.hist_accum, t_a);

    const t_r = prof.start();
    defer prof.stop(.hist_reduce, t_r);

    // Ascending worker order, and slots nobody touched are skipped rather
    // than added as zeros — which is why this stays bit-identical to reducing
    // all of them.
    var n_parts: usize = 0;
    for (0..active) |w| {
        if (bank.stamp(w).* == bank.seq) {
            bank.parts[n_parts] = w;
            n_parts += 1;
        }
    }
    if (n_parts == 0) {
        @memset(out[0..span], .{});
        return;
    }
    if (n_parts == 1) {
        @memcpy(out[0..span], bank.private[bank.parts[0] * span ..][0..span]);
        return;
    }

    var rctx = ReduceCtx{ .bank = bank, .parts = bank.parts[0..n_parts], .out = out };
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

/// Hard cap on how many levels one categorical split may send left, and so on
/// the size of the id list a split carries.
///
/// The set used to be a bitmask over every bin. That was affordable at 256
/// bins and is not at 65,535, where it would be 8 KiB per split node. A sorted
/// list of the ids on the left is smaller than the mask at every cardinality,
/// because `max_cat_threshold` already bounds it.
pub const max_cat_ids: usize = 32;

/// Ascending; binary search rather than a scan, since a split can carry 64 of
/// them and this runs once per node per row at prediction time.
pub inline fn catContains(ids: []const data.BinIdx, bin: data.BinIdx) bool {
    var lo: usize = 0;
    var hi: usize = ids.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (ids[mid] < bin) lo = mid + 1 else hi = mid;
    }
    return lo < ids.len and ids[lo] == bin;
}

pub const Split = struct {
    feature: u32 = 0,
    /// Rows with `bin <= threshold` go left (before the missing adjustment).
    /// Unused when `is_cat`.
    threshold: data.BinIdx = 0,
    /// Whether the missing bin joins the left child. Applies to categorical
    /// splits too: bin 0 is never in `cat_ids`.
    missing_left: bool = true,
    /// The split tests membership of `cat_ids[0..n_cat]`, not a threshold.
    is_cat: bool = false,
    n_cat: u8 = 0,
    cat_ids: [max_cat_ids]data.BinIdx = @splat(0),
    gain: f64 = -std.math.inf(f64),
    left: Bin = .{},
    right: Bin = .{},

    pub fn valid(s: Split) bool {
        return s.gain > 0;
    }
};

/// The four things the partition's row loop actually needs, lifted out of
/// `Split`.
///
/// `Split` is 160 bytes, because a categorical split carries its id array
/// inline. The row loop was reading through that, and the size showed up
/// directly in wall clock: partition ran at 2.2x until this was separated.
/// Nothing here is per-row state -- it is built once per partition.
pub const SplitTest = struct {
    missing_left: bool,
    is_cat: bool,
    threshold: data.BinIdx,
    ids: []const data.BinIdx,

    pub fn of(sp: *const Split) SplitTest {
        return .{
            .missing_left = sp.missing_left,
            .is_cat = sp.is_cat,
            .threshold = sp.threshold,
            .ids = sp.cat_ids[0..sp.n_cat],
        };
    }
};

pub const SplitParams = struct {
    lambda: f64,
    alpha: f64,
    min_split_gain: f64,
    min_child_weight: f64,
    min_child_samples: u32,
    max_delta_step: f64,
    /// Search categorical features by gradient order rather than by a cut on
    /// the dictionary id.
    cat_optimal: bool = false,
    cat_smooth: f64 = 10.0,
    cat_l2: f64 = 10.0,
    max_cat_threshold: u32 = 32,
    max_cat_to_onehot: u32 = 4,
    min_data_per_group: u32 = 100,
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
    thr: data.BinIdx,
    missing_left: bool,
    left: Bin,
    right: Bin,
    parent_score: f64,
    p: SplitParams,
) void {
    const min_n: f64 = @floatFromInt(p.min_child_samples);
    if (left.n < min_n or right.n < min_n) return;
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

/// Sort key for a categorical level: its gradient per unit hessian, with the
/// hessian padded so a level carrying three rows cannot reach an extreme of
/// the order on the strength of those three.
const CatKey = struct {
    bin: data.BinIdx,
    key: f64,

    fn lessThan(_: void, a: CatKey, b: CatKey) bool {
        if (a.key != b.key) return a.key < b.key;
        // Ties broken on the bin index so the order is total and the search
        // does not depend on the sort's internal choices.
        return a.bin < b.bin;
    }
};

/// Best split for one categorical feature.
///
/// Follows LightGBM's `FindBestThresholdCategoricalInner` (v4.7.0,
/// src/treelearner/feature_histogram.cpp) step for step. It was not written
/// that way first, and `docs/vs-lightgbm.md` records what the six differences
/// cost: zarbor recovered 64% of the gain LightGBM got from the same columns.
///
/// Two of them are worth naming here because they are not obvious:
///
///   * `cat_smooth` is the *participation* threshold, in rows, not just the
///     padding in the sort key. A level with fewer rows than it does not enter
///     the order at all and falls to the right child.
///   * `cat_l2` goes into the children's scores and **not** into the parent's.
///     Adding it to both, which reads as the self-consistent choice, biases
///     the comparison against categorical splits rather than regularising
///     them.
///
/// Missing is not searched. A categorical split always sends bin 0 right,
/// which is `default_left = false` in LightGBM, and is why `missing_left` is
/// forced false here rather than being left to a search that would find a
/// direction LightGBM never considers.
fn bestCatSplit(
    best: *Split,
    fid: u32,
    h: []const Bin,
    nb: u16,
    total: Bin,
    p: SplitParams,
    scratch: *[data.max_bins]CatKey,
) void {
    const min_n: f64 = @floatFromInt(p.min_child_samples);
    const missing = h[0];

    // A handful of levels: one against the rest, and without `cat_l2` -- in
    // LightGBM the extra penalty is added only on the sorted-partition path.
    if (nb <= p.max_cat_to_onehot) {
        const parent_score = nodeScore(total.g, total.h, p);
        var b: usize = 1;
        while (b < nb) : (b += 1) {
            const left = h[b];
            const right = total.sub(left);
            if (left.n < min_n or right.n < min_n) continue;
            if (left.h < p.min_child_weight or right.h < p.min_child_weight) continue;
            const gain = 0.5 * (nodeScore(left.g, left.h, p) +
                nodeScore(right.g, right.h, p) - parent_score) - p.min_split_gain;
            if (gain > best.gain) {
                var ids: [max_cat_ids]data.BinIdx = @splat(0);
                ids[0] = @intCast(b);
                best.* = .{
                    .feature = fid,
                    .is_cat = true,
                    .n_cat = 1,
                    .cat_ids = ids,
                    .missing_left = false,
                    .gain = gain,
                    .left = left,
                    .right = right,
                };
            }
        }
        return;
    }

    // Participation is by row count against `cat_smooth`.
    var n_ord: usize = 0;
    var b: usize = 1;
    while (b < nb) : (b += 1) {
        if (h[b].n < p.cat_smooth) continue;
        scratch[n_ord] = .{ .bin = @intCast(b), .key = h[b].g / (h[b].h + p.cat_smooth) };
        n_ord += 1;
    }
    if (n_ord < 2) return;

    // Stable, allocation-free, and at most 255 elements. The sort is nowhere
    // near the cost of the histogram that produced these bins.
    std.sort.insertion(CatKey, scratch[0..n_ord], {}, CatKey.lessThan);

    // Children carry the extra L2; the parent does not.
    var pc = p;
    pc.lambda = p.lambda + p.cat_l2;
    const parent_score = nodeScore(total.g, total.h, p);
    const group_floor: f64 = @floatFromInt(p.min_data_per_group);
    const max_num_cat = @min(@min(@as(usize, p.max_cat_threshold), max_cat_ids), (n_ord + 1) / 2);

    var best_gain = best.gain;
    var best_k: usize = 0;
    var best_from_low = true;
    var found = false;

    for ([2]bool{ true, false }) |from_low| {
        var acc: Bin = .{};
        var group: f64 = 0;
        var k: usize = 0;
        while (k < n_ord and k < max_num_cat) : (k += 1) {
            const idx = if (from_low) k else n_ord - 1 - k;
            acc = acc.add(h[scratch[idx].bin]);
            group += h[scratch[idx].bin].n;

            if (acc.n < min_n or acc.h < p.min_child_weight) continue;
            const right = total.sub(acc);
            // Once the right child is too small every longer prefix is too,
            // so this stops rather than skipping.
            if (right.n < min_n or right.n < group_floor) break;
            if (right.h < p.min_child_weight) break;
            // Pace the candidates: another cut is only considered once enough
            // rows have accumulated since the last one.
            if (group < group_floor) continue;
            group = 0;

            const gain = 0.5 * (nodeScore(acc.g, acc.h, pc) +
                nodeScore(right.g, right.h, pc) - parent_score) - p.min_split_gain;
            if (gain > best_gain) {
                best_gain = gain;
                best_k = k;
                best_from_low = from_low;
                found = true;
            }
        }
    }
    if (!found) return;

    // Materialise the winner once, rather than carrying the id list through
    // every candidate.
    var ids: [max_cat_ids]data.BinIdx = @splat(0);
    var n_cat: u8 = 0;
    var acc: Bin = .{};
    var k: usize = 0;
    while (k <= best_k) : (k += 1) {
        const idx = if (best_from_low) k else n_ord - 1 - k;
        ids[n_cat] = scratch[idx].bin;
        n_cat += 1;
        acc = acc.add(h[scratch[idx].bin]);
    }
    _ = missing;
    // Gradient order on the way in, bin order on the way out: the search wants
    // the first, the binary search at prediction time wants the second.
    std.sort.insertion(data.BinIdx, ids[0..n_cat], {}, std.sort.asc(data.BinIdx));
    best.* = .{
        .feature = fid,
        .is_cat = true,
        .n_cat = n_cat,
        .cat_ids = ids,
        .missing_left = false,
        .gain = best_gain,
        .left = acc,
        .right = total.sub(acc),
    };
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
    var cat_scratch: [data.max_bins]CatKey = undefined;

    for (features) |fid| {
        const h = bank.featureSliceConst(hist, fid);
        const nb = ds.n_bins[fid];
        if (nb < 3) continue; // missing bin plus one real bin: nothing to cut

        if (p.cat_optimal and ds.kinds[fid] == .categorical) {
            bestCatSplit(&best, fid, h, nb, total, p, &cat_scratch);
            continue;
        }

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
