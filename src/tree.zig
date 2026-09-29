// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Regression-tree construction over binned features.
//!
//! Rows belonging to a node are kept contiguous in `rows`. Gradients are
//! *not* permuted alongside them: they stay indexed by original row id and
//! the gradient gather happens once per tree, not once per node per feature,
//! and every histogram pass then reads gradients sequentially.
//!
//! Sibling histograms use the subtraction identity — a node's histogram is its
//! parent's minus its sibling's — so only the cheaper child of each pair is
//! ever accumulated.

const std = @import("std");
const data = @import("data.zig");
const Dataset = data.Dataset;
const hist = @import("hist.zig");

/// The tree grower lives in `builder.zig`.
pub const Builder = @import("builder.zig").Builder;
const split = @import("split.zig");

/// How a tree is expanded once its root histogram exists.
pub const GrowPolicy = enum {
    /// Expand all nodes at depth d before any at depth d+1 (XGBoost default).
    depthwise,
    /// Always split the leaf with the highest gain (LightGBM-style). Usually
    /// stronger per tree, and needs `max_leaves` rather than `max_depth` to
    /// control capacity.
    lossguide,
};

/// How a categorical feature's levels are partitioned at a split.
pub const CatSplit = enum {
    /// Cut the dictionary id like a numeric bin. The ids are assigned in
    /// order of first appearance, so the reachable partitions are prefixes of
    /// an arbitrary order.
    ordinal,
    /// Sort the levels present in the node by their smoothed gradient ratio
    /// and cut that order instead. See docs/categorical-splits.md.
    optimal,
};

/// How one tree is grown. Shared by the booster and the forest, which each
/// carry their own copy with their own defaults.
pub const Params = struct {
    grow_policy: GrowPolicy = .depthwise,
    /// Hard depth cap. 0 disables the cap, which is only safe when
    /// `max_leaves` is set — otherwise the histogram budget is unbounded.
    max_depth: u32 = 6,
    /// Leaf cap. 0 disables.
    max_leaves: u32 = 0,
    /// Minimum number of training rows that must land in a leaf.
    min_child_samples: u32 = 20,
    /// Minimum summed hessian in a leaf. For logistic this is a measure of
    /// confidence mass, not row count, and is the more principled of the two.
    min_child_weight: f32 = 1.0,
    /// Shrinkage applied to each tree's leaf values. XGBoost's `eta`.
    /// Forced to 1.0 for `random_forest`, which averages instead of shrinking.
    learning_rate: f32 = 0.1,
    /// L2 penalty on leaf weights. Appears as lambda in the gain formula.
    lambda: f32 = 1.0,
    /// L1 penalty on leaf weights, applied by soft-thresholding the gradient.
    alpha: f32 = 0.0,
    /// Minimum gain required to accept a split. XGBoost's `gamma`.
    min_split_gain: f32 = 0.0,
    /// Cap on the absolute leaf weight. 0 disables. Stabilises logistic loss
    /// on heavily imbalanced data.
    max_delta_step: f32 = 0.0,
    /// Fraction of rows sampled per tree. Without replacement unless
    /// `bootstrap` is set. Ignored when `sampling = .goss`.
    subsample: f32 = 1.0,
    /// Sample rows *with* replacement. This is what makes a bagged ensemble a
    /// random forest rather than a subsample ensemble; on by default for
    /// `random_forest` and off for everything else.
    bootstrap: bool = false,
    /// Fraction of features sampled once per tree.
    colsample_bytree: f32 = 1.0,
    /// Fraction of the per-tree features sampled again at each depth level.
    colsample_bylevel: f32 = 1.0,
    /// Fraction of the per-level features sampled again at each split.
    colsample_bynode: f32 = 1.0,
    /// How a categorical column's levels are partitioned. Defaults to
    /// `ordinal`, which is what every model written before this existed used.
    cat_split: CatSplit = .ordinal,
    /// Added to a level's hessian in the sort key, so a level carrying little
    /// mass cannot reach an extreme of the order on a handful of rows.
    cat_smooth: f32 = 10.0,
    /// Extra L2 applied to the gain of a categorical split only. A K-way
    /// choice has more ways to fit noise than a single threshold does.
    cat_l2: f32 = 10.0,
    /// Cap on how many levels may land in the left child. Further bounded by
    /// half the levels that took part, as LightGBM does.
    max_cat_threshold: u32 = 32,
    /// A categorical with no more bins than this is split one level against
    /// the rest instead of by a sorted partition, and without `cat_l2`. At
    /// four levels or fewer the partition search has little to search.
    max_cat_to_onehot: u32 = 4,
    /// Rows that must accumulate since the last evaluated cut before another
    /// is considered, and a floor on the right child. This paces the scan; it
    /// is not a filter on which levels take part, which is what `cat_smooth`
    /// does.
    min_data_per_group: u32 = 100,
    /// Fit an affine function of the root-to-leaf path's numeric features in
    /// each leaf instead of emitting a constant. Off by default: a constant
    /// leaf is what every model written before this existed used.
    linear_leaves: bool = false,
    /// Ridge on the leaf's slope terms. The intercept is left unpenalised, so
    /// setting this very high recovers the constant leaf rather than shrinking
    /// the leaf toward zero.
    lin_leaf_lambda: f32 = 1.0,
    /// Cap on how many path features enter one leaf's fit.
    lin_leaf_max_terms: u32 = 8,
    seed: u64 = 0,

    /// Upper bound on leaves for allocation sizing.
    pub fn leafBudget(p: Params) u32 {
        if (p.max_leaves != 0) return p.max_leaves;
        // depthwise with a depth cap: 2^depth leaves, clamped to something sane.
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
    }

    /// An unbounded tree is only safe if *something* caps its leaves.
    pub fn validateCapacity(p: Params) !void {
        if (p.max_depth == 0 and p.max_leaves == 0) return error.UnboundedTree;
        if (p.grow_policy == .lossguide and p.max_leaves == 0 and p.max_depth == 0)
            return error.UnboundedLossguide;
    }
};

/// The value a bin stands for when a leaf needs a number rather than an index.
///
/// Midpoints of the schema's edges, not `ds.means`: means are a property of
/// whichever file was binned, so a model using them would score a row
/// differently depending on what else was in the file with it.
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
    /// Where this split's level ids start in `Tree.cat_ids`. Meaningless
    /// unless `is_cat`.
    cat_ofs: u32 = 0,
    threshold: data.BinIdx = 0,
    missing_left: bool = true,
    is_leaf: bool = true,
    /// The split tests membership of `cat_ids[cat_ofs..][0..n_cat]` rather
    /// than comparing against `threshold`.
    is_cat: bool = false,
    /// How many level ids this split sends left. Bounded by
    /// `split.max_cat_ids`, which is why a byte is enough.
    n_cat: u8 = 0,
    /// Slope terms this leaf carries. 0 is a plain constant leaf, which is
    /// every leaf unless `linear_leaves` is on.
    n_lin: u8 = 0,
    _pad: u8 = 0,
    /// Where this leaf's terms start in `Tree.lin`.
    lin_ofs: u32 = 0,
};

/// One slope term of a linear leaf. `center` is the feature's within-leaf mean
/// at fit time; subtracting it here rather than folding it into the intercept
/// keeps the evaluation away from the cancellation that a large intercept and
/// a large `coef * x` would produce in f32.
pub const LinTerm = extern struct {
    feature: u32,
    coef: f32,
    center: f32,
};

pub const Tree = struct {
    nodes: []Node,
    /// Flat store of the categorical masks this tree's splits refer to,
    /// `hist.cat_words` words each. Empty when no split is categorical, which
    /// is every tree grown under `cat_split = ordinal`.
    cat_ids: []data.BinIdx = &.{},
    /// Flat store of the slope terms this tree's leaves refer to. Empty
    /// unless `linear_leaves` is on.
    lin: []LinTerm = &.{},

    pub fn deinit(t: *Tree, gpa: std.mem.Allocator) void {
        gpa.free(t.nodes);
        if (t.cat_ids.len != 0) gpa.free(t.cat_ids);
        if (t.lin.len != 0) gpa.free(t.lin);
        t.* = undefined;
    }

    /// Raw score for one row of a dataset binned with the same edges.
    ///
    /// Reads the row-major matrix: a walk visits a different feature at every
    /// level, so column-major touched one cache line per level of depth,
    /// while a whole row is thirteen contiguous bytes.
    pub fn predictBinned(t: *const Tree, ds: *const Dataset, row: usize) f32 {
        const rb = ds.bins_rm[row * ds.n_features ..][0..ds.n_features];
        var i: u32 = 0;
        while (!t.nodes[i].is_leaf) {
            const n = t.nodes[i];
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

/// A leaf's contiguous row span, so the booster can update raw scores by
/// walking ranges instead of re-traversing the tree for every row.
pub const LeafSpan = struct {
    start: usize,
    end: usize,
    weight: f32,
};
