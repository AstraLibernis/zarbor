// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Split search: the best cut of a node's histograms, numeric or categorical.

const std = @import("std");
const data = @import("data.zig");
const Dataset = data.Dataset;
const Bank = @import("hist.zig").Bank;
const Bin = @import("hist.zig").Bin;

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
/// Bin 0 holds the missing mass and is never itself a threshold. A feature
/// with missing rows at this node is scanned twice — once sending missing
/// left, once right — which is how the default direction is learned per split
/// rather than fixed in advance.
///
/// A feature with **no** missing rows at this node is scanned once. Both
/// directions score identically there (the missing bin contributes nothing to
/// either side), so the second scan recomputes the same gains and the winner
/// falls out of a tie-break: `consider` keeps the first strictly-better
/// candidate, so `missing_left = true` won every time, chosen by loop order
/// and not by evidence.
///
/// That flag is not inert. It is what routes a missing value at *prediction*
/// time, and a column can be complete in training and have holes later --
/// which is the normal case, not a corner one. On Kaggle's House Prices,
/// fifteen columns are missing in the test half and never in the training
/// half, so fifteen default directions were being set by loop order.
///
/// With no evidence, the defensible choice is the larger child: it is the
/// side holding more of the node's distribution, so it is the smaller bet.
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

        if (missing.n == 0) {
            // One scan, and the direction comes from the split rather than
            // from which loop ran first. The gains are unchanged, so the
            // chosen threshold is bit-identical to before -- only the flag
            // that nothing in training could inform is decided differently.
            var acc: Bin = .{};
            var b: usize = 1;
            while (b + 1 < nb) : (b += 1) {
                acc = acc.add(h[b]);
                const right = total.sub(acc);
                consider(&best, fid, @intCast(b), acc.n >= right.n, acc, right, parent_score, p);
            }
            continue;
        }

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
