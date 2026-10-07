// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Split search: the best cut of a node's histograms, numeric or categorical.

const std = @import("std");
const data = @import("data.zig");
const Dataset = data.Dataset;
const Bank = @import("hist.zig").Bank;
const Bin = @import("hist.zig").Bin;

/// Hard cap on levels one categorical split sends left, hence on its id-list size. A sorted id
/// list, not a bitmask over every bin: the mask was fine at 256 bins but 8 KiB per split node at
/// 65,535, and the list is smaller at every cardinality since `max_cat_threshold` bounds it.
pub const max_cat_ids: usize = 32;

/// Ascending ids; binary search, not a scan: a split can carry `max_cat_ids` and this runs per node per row at
/// predict.
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
    /// Rows with `bin <= threshold` go left (before the missing adjustment). Unused when `is_cat`.
    threshold: data.BinIdx = 0,
    /// Whether the missing bin joins the left child; also for categorical splits (bin 0 never in
    /// `cat_ids`).
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

/// The four fields the partition row loop needs, lifted out of `Split`, built once per partition.
/// `Split` is large (`@sizeOf(Split)`, inline categorical id array); reading through it slowed
/// partition severely (see docs/measurements.md).
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
    /// Search categoricals by gradient order rather than a cut on the dictionary id.
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
    // No 0.5: the unhalved score sum is XGBoost's `loss_chg` and LightGBM's split gain, which their
    // `gamma` / `min_gain_to_split` are compared against. Halving it made `min_split_gain` act as
    // twice the same number there.
    const gain = nodeScore(left.g, left.h, p) + nodeScore(right.g, right.h, p) - parent_score - p.min_split_gain;
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

/// Sort key for a categorical level: gradient per unit hessian, the hessian padded so a three-row
/// level cannot reach an extreme of the order on those three rows.
const CatKey = struct {
    bin: data.BinIdx,
    key: f64,

    fn lessThan(_: void, a: CatKey, b: CatKey) bool {
        if (a.key != b.key) return a.key < b.key;
        // Tie-break on bin so the order is total and independent of the sort's internals.
        return a.bin < b.bin;
    }
};

/// Levels the sorted categorical search orders on the stack. Sized to cover
/// `data.BinParams.max_cat_levels`'s default; wider columns go through `bestCatSplitWide`.
/// bestSplit used to declare a `data.max_bins`-long `CatKey` scratch on every call, cat search or
/// not: Debug and ReleaseSafe fill an `undefined` array with 0xAA, and that fill was most of a
/// large ReleaseSafe slowdown on a forest (see docs/measurements.md); the big frame also cost a
/// stack probe per call in every mode.
const cat_scratch_small = 256;

/// The `data.max_bins` scratch, in its own frame so only a categorical search over a wide column
/// pays for it.
noinline fn bestCatSplitWide(best: *Split, fid: u32, h: []const Bin, nb: u16, total: Bin, p: SplitParams) void {
    var scratch: [data.max_bins]CatKey = undefined; // zsnag:ok — R015: wide columns only
    bestCatSplit(best, fid, h, nb, total, p, &scratch);
}

/// Best split for one categorical feature. Follows LightGBM `FindBestThresholdCategoricalInner`
/// (v4.7.0, src/treelearner/feature_histogram.cpp) step for step; docs/vs-lightgbm.md records the
/// cost of six earlier differences (zarbor fell well short of LightGBM's gain on the same columns;
/// see docs/measurements.md). Subtle:
/// - `cat_smooth` is also the participation threshold in rows, not just sort-key padding; a level
///   with fewer rows stays out of the order and falls right.
/// - `cat_l2` goes into the children's scores, not the parent's. Adding it to both looks consistent
///   but biases the comparison against categorical splits instead of regularising them.
/// Missing is not searched: bin 0 always goes right (LightGBM `default_left = false`), so
/// `missing_left` is forced false rather than searched in a direction LightGBM never considers.
/// `scratch` holds at least `nb` entries.
fn bestCatSplit(
    best: *Split,
    fid: u32,
    h: []const Bin,
    nb: u16,
    total: Bin,
    p: SplitParams,
    scratch: []CatKey,
) void {
    std.debug.assert(scratch.len >= nb);
    const min_n: f64 = @floatFromInt(p.min_child_samples);

    // Few bins (the missing bin counts): one level vs the rest, without `cat_l2` (LightGBM adds
    // it only on the sorted path).
    if (nb <= p.max_cat_to_onehot) {
        const parent_score = nodeScore(total.g, total.h, p);
        var b: usize = 1;
        while (b < nb) : (b += 1) {
            const left = h[b];
            const right = total.sub(left);
            if (left.n < min_n or right.n < min_n) continue;
            if (left.h < p.min_child_weight or right.h < p.min_child_weight) continue;
            const gain = nodeScore(left.g, left.h, p) +
                nodeScore(right.g, right.h, p) - parent_score - p.min_split_gain;
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

    // Stable, allocation-free; one element per level present, so few at the default
    // `data.BinParams.max_cat_levels`, where it is far cheaper than the histogram. Quadratic, so
    // it grows fast on columns wider than that.
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
            // Right child too small: every longer prefix is too, so stop rather than skip.
            if (right.n < min_n or right.n < group_floor) break;
            if (right.h < p.min_child_weight) break;
            // Pace candidates: a cut is considered only once enough rows accumulated since the
            // last.
            if (group < group_floor) continue;
            group = 0;

            const gain = nodeScore(acc.g, acc.h, pc) +
                nodeScore(right.g, right.h, pc) - parent_score - p.min_split_gain;
            if (gain > best_gain) {
                best_gain = gain;
                best_k = k;
                best_from_low = from_low;
                found = true;
            }
        }
    }
    if (!found) return;

    // Materialise the winner once rather than carrying the id list through every candidate.
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
    // Gradient order for the search, bin order for the binary search at prediction time.
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

/// Best split for one node over `features`. Bin 0 holds the missing mass; it is a threshold only in
/// the missing-vs-present split (`threshold = 0`, missing left).
/// With missing rows at this node a feature is scanned twice (missing left, then right), learning
/// the default direction per split. With none it is scanned once: both directions score
/// identically, and `consider` keeps the first strictly-better candidate, so `missing_left = true`
/// would win by loop order, not evidence. That flag routes missing values at prediction time, and a
/// column complete in training often has holes later (Kaggle House Prices has many columns missing
/// only in the test half). With no evidence, send missing to the larger child: it holds more of the
/// node, so it is the smaller bet.
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
        if (nb < 2) continue;
        const missing = h[0];

        // Missing alone against every present value: `threshold = 0` sends only bin 0 left. No scan
        // below reaches this partition (each keeps at least one real bin with the missing mass), so
        // without it a column whose signal is *being* missing could not be split on at all; XGBoost
        // reaches it from its scan's end points. Numeric or categorical alike: bins >= 1 fail
        // `bin <= 0`, so it needs no level list. The mirror (missing right, all present left) is the
        // same partition.
        if (missing.n != 0) consider(&best, fid, 0, true, missing, total.sub(missing), parent_score, p);
        if (nb < 3) continue; // one real bin: missing-vs-present was the only cut

        if (p.cat_optimal and ds.kinds[fid] == .categorical) {
            if (nb <= cat_scratch_small) {
                var scratch: [cat_scratch_small]CatKey = undefined;
                bestCatSplit(&best, fid, h, nb, total, p, &scratch);
            } else bestCatSplitWide(&best, fid, h, nb, total, p);
            continue;
        }

        if (missing.n == 0) {
            // One scan; the direction comes from the split, not loop order. Gains are unchanged, so
            // the threshold is bit-identical; only the flag training could not inform is decided
            // differently.
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
