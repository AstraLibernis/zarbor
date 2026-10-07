// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Regression trees over binned features: the grow parameters (`Params`), the fitted `Tree` with
//! its `Node`s and linear-leaf terms, and binned prediction. Construction lives in builder.zig,
//! re-exported here as `Builder`.

const std = @import("std");
const data = @import("data.zig");
const Dataset = data.Dataset;
const hist = @import("hist.zig");

pub const Builder = @import("builder.zig").Builder;
const split = @import("split.zig");

/// How a tree is expanded once its root histogram exists.
pub const GrowPolicy = enum {
    /// Expand all nodes at depth d before any at depth d+1 (XGBoost default).
    depthwise,
    /// Split the highest-gain leaf first (LightGBM-style). Stronger per tree; cap via `max_leaves`.
    lossguide,
    /// One split per depth shared by every node at that depth (CatBoost's oblivious trees):
    /// `max_depth` questions, 2^depth leaves. Grown by symmetric.zig.
    symmetric,
};

pub const ScoreFunction = @import("symmetric.zig").ScoreFunction;
pub const BootstrapType = @import("symmetric.zig").BootstrapType;
pub const BoostingType = enum { plain, ordered };

/// How a categorical feature's levels are partitioned at a split.
pub const CatSplit = enum {
    /// Cut the dictionary id like a numeric bin. Ids follow first appearance, so the reachable
    /// partitions are prefixes of an arbitrary order.
    ordinal,
    /// Sort levels by smoothed gradient ratio and cut that. See docs/archive/categorical-splits.md.
    optimal,
    /// CatBoost's: one-hot up to `one_hot_max_size` levels, ordered target statistics above.
    /// `symmetric` trees and logistic loss only (docs/archive/catboost.md).
    ctr,
};

/// How one tree is grown. Booster and forest each carry their own copy with their own defaults.
pub const Params = struct {
    grow_policy: GrowPolicy = .depthwise,
    /// Depth cap; 0 disables, safe only with `max_leaves` (else histogram budget is unbounded).
    max_depth: u32 = 6,
    /// Leaf cap. 0 disables.
    max_leaves: u32 = 0,
    /// Histogram memory per tree, MiB (LightGBM's `histogram_pool_size`). A tree wanting more
    /// than fits evicts queued nodes' histograms and builds their children directly: slower, not
    /// refused. Only past the size of three histograms does training stop.
    histogram_pool_size: u32 = 2048,
    /// Minimum training rows per leaf.
    min_child_samples: u32 = 20,
    /// Minimum leaf hessian sum: for logistic, confidence mass not rows; the more principled cap.
    min_child_weight: f32 = 1.0,
    /// Leaf shrinkage (XGBoost `eta`). Forced to 1.0 for `random_forest`, which averages instead.
    learning_rate: f32 = 0.1,
    /// L2 penalty on leaf weights; lambda in the gain formula.
    lambda: f32 = 1.0,
    /// L1 penalty on leaf weights, applied by soft-thresholding the gradient.
    alpha: f32 = 0.0,
    /// Minimum gain to accept a split (XGBoost `gamma`).
    min_split_gain: f32 = 0.0,
    /// Cap on |leaf weight|; 0 disables. Stabilises logistic loss on heavily imbalanced data.
    max_delta_step: f32 = 0.0,
    /// Row fraction per tree; no replacement unless `bootstrap`. Ignored under `sampling = .goss`.
    subsample: f32 = 1.0,
    /// Sample rows with replacement: makes bagging a random forest. On only for `random_forest`.
    bootstrap: bool = false,
    /// Feature fractions sampled per tree, then from those per depth level, then per split.
    colsample_bytree: f32 = 1.0,
    colsample_bylevel: f32 = 1.0,
    colsample_bynode: f32 = 1.0,
    /// Categorical level partitioning. Default `ordinal`, what every model before this option used.
    cat_split: CatSplit = .ordinal,
    /// Added to a level's hessian in the sort key, so a low-mass level can't reach an extreme.
    cat_smooth: f32 = 10.0,
    /// Extra L2 on categorical gain only: a K-way choice fits noise more ways than a threshold.
    cat_l2: f32 = 10.0,
    /// Cap on levels sent left; further bounded by half the participating levels, as LightGBM does.
    max_cat_threshold: u32 = 32,
    /// Up to this many bins, the missing bin included: one level vs the rest, no `cat_l2`; so few
    /// levels leave little to search.
    max_cat_to_onehot: u32 = 4,
    /// Rows needed since the last evaluated cut before the next, and a right-child floor. Paces
    /// the scan; filtering which levels take part is `cat_smooth`'s job.
    min_data_per_group: u32 = 100,
    /// Per-leaf affine fit on the path's numeric features. Off: all older models are constant.
    linear_leaves: bool = false,
    /// Ridge on slopes only; intercept unpenalised, so huge values give the constant leaf, not 0.
    lin_leaf_lambda: f32 = 1.0,
    /// Cap on path features in one leaf's fit.
    lin_leaf_max_terms: u32 = 8,
    /// How `symmetric` scores a level's split; `auto` is CatBoost's cosine there. Depthwise and
    /// lossguide always use gain.
    score_function: ScoreFunction = .auto,
    /// Newton steps per leaf under `symmetric` (CatBoost's `leaf_estimation_iterations`, without
    /// its backtracking): >1 re-takes the derivatives at the moved score.
    leaf_estimation_iterations: u32 = 1,
    /// Per-tree row weights for split scoring under `symmetric` (CatBoost's `bootstrap_type`);
    /// `bernoulli` and `mvs` keep `subsample` of the rows in expectation.
    bootstrap_type: BootstrapType = .none,
    /// `bayesian` weights are `(-ln u)^bagging_temperature`; 0 makes them all 1.
    bagging_temperature: f32 = 1.0,
    /// MVS lambda; unset follows CatBoost (previous tree's mean |leaf| squared).
    mvs_reg: ?f32 = null,
    /// Normal noise on `symmetric` split scores, scaled by the gradients and fading as the model
    /// grows (CatBoost's `random_strength`, default 1 there). 0 is deterministic.
    random_strength: f32 = 0.0,
    /// Under `cat_split = ctr`: categoricals with up to this many levels are one-hot (CatBoost's
    /// default 2), wider ones get target statistics.
    one_hot_max_size: u32 = 2,
    /// Under `cat_split = ctr`: shrinks a target statistic no tree has used yet, the more distinct
    /// levels it has (CatBoost's default 0.5).
    model_size_reg: f32 = 0.5,
    /// Under `cat_split = ctr`: most columns one target statistic may combine (CatBoost's
    /// `max_ctr_complexity`, default 4 there); 1 keeps every statistic to a single column.
    max_ctr_complexity: u32 = 4,
    /// Rows are in time order (CatBoost's `has_time`): target statistics and ordered boosting read
    /// them in file order, one fold. Off (CatBoost's default), they use `permutation_count - 1`
    /// random orderings, one drawn per tree. Only matters under `cat_split = ctr` or `ordered`.
    has_time: bool = false,
    permutation_count: u32 = 4,
    /// Rows shuffled together; 0 is CatBoost's `min(256, n / 1000 + 1)`.
    permutation_block: u32 = 0,
    /// `ordered`: CatBoost's ordered boosting under `symmetric` (about 4x the work of `plain`).
    boosting_type: BoostingType = .plain,
    seed: u64 = 0,

    /// Upper bound on leaves for allocation sizing.
    pub fn leafBudget(p: Params) u32 {
        if (p.max_leaves != 0) return p.max_leaves;
        // Depthwise with a depth cap: 2^depth leaves, depth clamped to 20.
        const d = @min(p.max_depth, 20);
        return @as(u32, 1) << @intCast(d);
    }

    /// Step size, sampling rates and penalties.
    pub fn validate(p: Params) !void {
        if (p.learning_rate <= 0 or p.learning_rate > 1) return error.BadLearningRate;
        if (p.subsample <= 0 or p.subsample > 1) return error.BadSubsample;
        if (p.colsample_bytree <= 0 or p.colsample_bytree > 1) return error.BadColsample;
        if (p.colsample_bylevel <= 0 or p.colsample_bylevel > 1) return error.BadColsample;
        if (p.colsample_bynode <= 0 or p.colsample_bynode > 1) return error.BadColsample;
        if (p.lambda < 0 or p.alpha < 0) return error.NegativeRegularisation;
        if (p.leaf_estimation_iterations == 0) return error.BadLeafIterations;
        if (p.random_strength < 0 or p.bagging_temperature < 0) return error.NegativeRandomness;
        if (p.grow_policy != .symmetric and (p.bootstrap_type != .none or p.random_strength != 0 or p.cat_split == .ctr or p.boosting_type == .ordered))
            return error.OnlyForSymmetric;
        // Ordered scoring is defined for CatBoost's Cosine and L2; one leaf step is what is verified.
        if (p.boosting_type == .ordered and (p.score_function == .gain or p.leaf_estimation_iterations != 1))
            return error.OrderedUnsupported;
        if (p.model_size_reg < 0) return error.NegativeRegularisation;
        if (p.grow_policy == .symmetric) {
            if (p.max_depth == 0 or p.max_depth > @import("symmetric.zig").max_depth) return error.BadSymmetricDepth;
            // Rows are subsampled only through the bootstrap types that define it.
            if (p.subsample != 1 and p.bootstrap_type != .bernoulli and p.bootstrap_type != .mvs)
                return error.SymmetricSubsampleNeedsBootstrapType;
            // Not built for symmetric trees yet; refused rather than silently ignored.
            if (p.colsample_bytree != 1 or p.colsample_bylevel != 1 or
                p.colsample_bynode != 1 or p.bootstrap or p.linear_leaves or
                p.cat_split == .optimal or p.alpha != 0 or p.max_delta_step != 0)
                return error.SymmetricUnsupported;
        }
    }

    /// An unbounded tree is only safe if *something* caps its leaves.
    pub fn validateCapacity(p: Params) !void {
        if (p.max_depth == 0 and p.max_leaves == 0) return error.UnboundedTree;
    }
};

/// A bin's value when a leaf needs a number. Schema edge midpoints, not `ds.means`: means belong to
/// whichever file was binned, so a row would score differently depending on its file-mates.
pub inline fn binValue(ds: *const Dataset, f: u32, bin: data.BinIdx) f32 {
    const e = ds.edges[f];
    if (e.len == 0) return 0;
    if (bin == 0) return data.binMidpoint(e, e.len / 2); // missing -> centre
    return data.binMidpoint(e, @as(usize, bin) - 1);
}

pub const Node = extern struct {
    feature: u32 = 0,
    left: u32 = 0,
    right: u32 = 0,
    weight: f32 = 0,
    /// Start of this split's level ids in `Tree.cat_ids`; meaningless unless `is_cat`.
    cat_ofs: u32 = 0,
    threshold: data.BinIdx = 0,
    missing_left: bool = true,
    is_leaf: bool = true,
    /// Split tests membership of `cat_ids[cat_ofs..][0..n_cat]`, not `threshold`.
    is_cat: bool = false,
    /// Level ids sent left. Bounded by `split.max_cat_ids`, so a byte suffices.
    n_cat: u8 = 0,
    /// Slope terms in this leaf. 0 is a constant leaf, which every leaf is unless `linear_leaves`.
    n_lin: u8 = 0,
    /// `split` (threshold or level set, by `is_cat`), or `combo`: `feature` indexes `Tree.combos`
    /// and `threshold` is a bucket, rows with `bucket <= threshold` going left.
    kind: NodeKind = .split,
    /// Where this leaf's terms start in `Tree.lin`.
    lin_ofs: u32 = 0,
};

pub const NodeKind = enum(u8) { split = 0, combo = 1 };

/// One piece of a combination's key.
pub const PartKind = enum(u8) {
    /// The column's level (its bin).
    cat = 0,
    /// Whether the column's bin is above `value`.
    bin = 1,
    /// Whether the column's bin equals `value`.
    onehot = 2,
};

pub const Part = struct {
    kind: PartKind,
    feature: u32,
    value: data.BinIdx = 0,

    /// Canonical order: by kind, then column, then value.
    pub fn lessThan(_: void, a: Part, b: Part) bool {
        if (a.kind != b.kind) return @intFromEnum(a.kind) < @intFromEnum(b.kind);
        if (a.feature != b.feature) return a.feature < b.feature;
        return a.value < b.value;
    }
};

/// A target statistic over several columns at once (CatBoost's feature combinations): a row's key
/// mixes the levels of its `cat` parts with the bits of its `bin` and `onehot` parts, and its
/// bucket is looked up from counts over every training row. A key never seen in training gets
/// `unseen`, the bucket of the prior alone.
pub const Combo = struct {
    parts: []Part,
    /// Sorted ascending, with `buckets[i]` the bucket of `keys[i]`.
    keys: []u64,
    buckets: []u8,
    unseen: u8,

    pub fn deinit(c: *Combo, gpa: std.mem.Allocator) void {
        gpa.free(c.parts);
        gpa.free(c.keys);
        gpa.free(c.buckets);
        c.* = undefined;
    }

    /// A deep copy the caller owns.
    pub fn dupe(c: *const Combo, gpa: std.mem.Allocator) !Combo {
        const parts = try gpa.dupe(Part, c.parts);
        errdefer gpa.free(parts);
        const keys = try gpa.dupe(u64, c.keys);
        errdefer gpa.free(keys);
        return .{ .parts = parts, .keys = keys, .buckets = try gpa.dupe(u8, c.buckets), .unseen = c.unseen };
    }

    pub fn bucket(c: *const Combo, key: u64) u8 {
        var lo: usize = 0;
        var hi: usize = c.keys.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (c.keys[mid] < key) lo = mid + 1 else hi = mid;
        }
        return if (lo < c.keys.len and c.keys[lo] == key) c.buckets[lo] else c.unseen;
    }
};

/// The key of a row given its bins per column (`bin_of(f)`), for `parts` in canonical order.
/// Training and prediction share this, so the two cannot disagree on a key.
pub fn comboKey(parts: []const Part, ctx: anytype, comptime bin_of: fn (@TypeOf(ctx), u32) data.BinIdx) u64 {
    var x: u64 = 0x243F6A8885A308D3;
    for (parts) |p| {
        const b = bin_of(ctx, p.feature);
        const v: u64 = switch (p.kind) {
            .cat => b,
            .bin => @intFromBool(b > p.value),
            .onehot => @intFromBool(b == p.value),
        };
        x +%= 0x9E3779B97F4A7C15 +% v;
        var z = x;
        z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
        z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
        x = z ^ (z >> 31);
    }
    return x;
}

fn rowBin(rb: []const data.BinIdx, f: u32) data.BinIdx {
    return rb[f];
}

/// One linear-leaf slope term. `center` is the feature's within-leaf mean at fit time; subtracting
/// it here, not in the intercept, avoids f32 cancellation of a large intercept vs `coef * x`.
pub const LinTerm = extern struct {
    feature: u32,
    coef: f32,
    center: f32,
};

pub const Tree = struct {
    nodes: []Node,
    /// Flat store of sorted categorical level ids (`Node.cat_ofs`, `n_cat`). Empty with no categorical
    /// split, i.e. always under `cat_split = ordinal`.
    cat_ids: []data.BinIdx = &.{},
    /// Flat store of this tree's leaf slope terms. Empty unless `linear_leaves`.
    lin: []LinTerm = &.{},
    /// Combination statistics the tree's `combo` nodes index. Empty except under `cat_split = ctr`.
    combos: []Combo = &.{},

    pub fn deinit(t: *Tree, gpa: std.mem.Allocator) void {
        gpa.free(t.nodes);
        if (t.cat_ids.len != 0) gpa.free(t.cat_ids);
        if (t.lin.len != 0) gpa.free(t.lin);
        for (t.combos) |*c| c.deinit(gpa);
        if (t.combos.len != 0) gpa.free(t.combos);
        t.* = undefined;
    }

    /// Raw score for one row binned with the same edges. Row-major: a walk visits a new feature per
    /// level, so column-major cost a cache line per level; a row is `n_features` contiguous `u16`
    /// bins (26 bytes on 13 features).
    pub fn predictBinned(t: *const Tree, ds: *const Dataset, row: usize) f32 {
        const rb = ds.bins_rm[row * ds.n_features ..][0..ds.n_features];
        var i: u32 = 0;
        while (!t.nodes[i].is_leaf) {
            const n = t.nodes[i];
            if (n.kind == .combo) {
                const c = &t.combos[n.feature];
                i = if (c.bucket(comboKey(c.parts, rb, rowBin)) <= n.threshold) n.left else n.right;
                continue;
            }
            const b = rb[n.feature];
            const go_left = if (b == 0)
                n.missing_left
            else if (n.is_cat)
                split.catContains(t.cat_ids[n.cat_ofs..][0..n.n_cat], b)
            else
                b <= n.threshold;
            i = if (go_left) n.left else n.right;
        }
        const leaf = t.nodes[i];
        if (leaf.n_lin == 0) return leaf.weight;
        var v: f32 = leaf.weight;
        for (t.lin[leaf.lin_ofs..][0..leaf.n_lin]) |term| {
            v += term.coef * (binValue(ds, term.feature, rb[term.feature]) - term.center);
        }
        return v;
    }
};

/// A leaf's contiguous row span, so the booster updates raw scores by range, not per-row traversal.
pub const LeafSpan = struct {
    start: usize,
    end: usize,
    weight: f32,
};
