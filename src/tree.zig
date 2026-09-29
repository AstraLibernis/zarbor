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
const Pool = @import("pool.zig").Pool;
const data = @import("data.zig");
const Dataset = data.Dataset;
const hist = @import("hist.zig");
const prof = @import("prof.zig");
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

/// Hard cap on the root-to-leaf features a linear leaf may use. Independent
/// of `max_depth` so the array stays a fixed size in `Work`.
const max_path: usize = 8;

const Work = struct {
    node: u32,
    start: usize,
    end: usize,
    depth: u32,
    total: hist.Bin,
    slot: u32,
    split: split.Split,
    /// Distinct numeric features tested between the root and this node.
    /// Empty unless `linear_leaves` is on -- tracking it otherwise is pure
    /// cost for something nothing reads.
    path: [max_path]u32 = undefined,
    n_path: u8 = 0,
};

/// In-place Cholesky solve of a small symmetric positive-definite system.
///
/// `a` is read as its upper triangle and overwritten; `rhs` carries the
/// solution out. Returns false when the factorisation meets a non-positive
/// pivot, which is what a feature constant within the leaf looks like and is
/// common near the bottom of a tree. The caller treats that as "no slopes",
/// which is the admissible `beta = 0` solution rather than a failure.
fn choleskySolve(a: *[max_path][max_path]f64, rhs: *[max_path]f64, n: usize) bool {
    var l: [max_path][max_path]f64 = undefined;
    for (0..n) |i| for (0..n) |j| {
        l[i][j] = 0;
    };
    for (0..n) |i| {
        for (0..i + 1) |j| {
            var sum: f64 = if (j <= i) a[j][i] else a[i][j];
            for (0..j) |k| sum -= l[i][k] * l[j][k];
            if (i == j) {
                if (!(sum > 1e-12)) return false;
                l[i][i] = @sqrt(sum);
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    // Forward, then back.
    for (0..n) |i| {
        var sum = rhs[i];
        for (0..i) |k| sum -= l[i][k] * rhs[k];
        rhs[i] = sum / l[i][i];
    }
    var i = n;
    while (i > 0) {
        i -= 1;
        var sum = rhs[i];
        for (i + 1..n) |k| sum -= l[k][i] * rhs[k];
        rhs[i] = sum / l[i][i];
    }
    return true;
}

/// Hard ceiling on histogram slot memory. Beyond this the caller is asked to
/// reduce tree size rather than have the allocator decide for them.
const slot_memory_budget: usize = 2 << 30;

pub const Builder = struct {
    gpa: std.mem.Allocator,
    pool: *Pool,
    ds: *const Dataset,
    cfg: Params,
    bank: hist.Bank,

    /// `n_slots` histograms, each `bank.slotLen()` bins.
    slots: []hist.Bin,
    free_slots: std.ArrayList(u32),

    rows: []u32,
    /// Gradients indexed by original row id, borrowed for the current tree.
    ///
    /// These used to be permuted to match `rows`, so that a node's gradients
    /// were contiguous. That paid for itself when the histogram loop was
    /// feature-outer and read each gradient once *per feature*; row-outer
    /// reads it once per row, and the permutation then costs far more than it
    /// saves -- it tripled what the partition had to move (12 bytes a row
    /// against 4). Measured: the row-id gather costs 4 ms across a 200-tree
    /// fit, the moving cost over 100.
    g: []const hist.GradPair,
    /// Destination for the parallel partition, which cannot be done in place.
    rows_out: []u32,
    /// Per-chunk partial sums for `totalOf`.
    total_partial: []hist.Bin,
    /// Per-chunk left-hand counts, plus one for the total.
    part_counts: []usize,
    part_left: []usize,
    part_right: []usize,
    /// Rows active for the current tree: `rows[0..n_active]`.
    n_active: usize,

    nodes: std.ArrayList(Node),
    cat_ids: std.ArrayList(data.BinIdx),
    lin: std.ArrayList(LinTerm),
    queue: std.ArrayList(Work),
    leaves: std.ArrayList(LeafSpan),

    all_features: []u32,
    tree_features: []u32,
    level_features: []u32,
    node_features: []u32,
    n_tree_features: usize,
    n_level_features: usize,
    level_depth: i64,

    rng: std.Random.DefaultPrng,

    pub fn init(
        gpa: std.mem.Allocator,
        pool: *Pool,
        ds: *const Dataset,
        cfg: Params,
    ) !Builder {
        var bank = try hist.Bank.init(gpa, pool.workerCount(), ds.n_features, ds.n_bins);
        errdefer bank.deinit();

        const slot_len = bank.slotLen();
        const want_slots: usize = @as(usize, cfg.leafBudget()) + 2;
        const affordable = slot_memory_budget / (slot_len * @sizeOf(hist.Bin));
        if (want_slots > affordable) return error.TreeTooLargeForHistogramBudget;

        const slots = try gpa.alloc(hist.Bin, want_slots * slot_len);
        errdefer gpa.free(slots);

        var free_slots: std.ArrayList(u32) = .empty;
        errdefer free_slots.deinit(gpa);
        try free_slots.ensureTotalCapacity(gpa, want_slots);
        var s: u32 = 0;
        while (s < want_slots) : (s += 1) free_slots.appendAssumeCapacity(s);

        const rows = try gpa.alloc(u32, ds.n_rows);
        errdefer gpa.free(rows);
        const rows_out = try gpa.alloc(u32, ds.n_rows);
        errdefer gpa.free(rows_out);
        const total_partial = try gpa.alloc(hist.Bin, total_chunks);
        errdefer gpa.free(total_partial);

        const n_chunks = pool.workerCount() * 4 + 2;
        const part_counts = try gpa.alloc(usize, n_chunks);
        errdefer gpa.free(part_counts);
        const part_left = try gpa.alloc(usize, n_chunks);
        errdefer gpa.free(part_left);
        const part_right = try gpa.alloc(usize, n_chunks);
        errdefer gpa.free(part_right);

        const all_features = try gpa.alloc(u32, ds.n_features);
        errdefer gpa.free(all_features);
        for (all_features, 0..) |*f, i| f.* = @intCast(i);
        const tree_features = try gpa.alloc(u32, ds.n_features);
        errdefer gpa.free(tree_features);
        const level_features = try gpa.alloc(u32, ds.n_features);
        errdefer gpa.free(level_features);
        const node_features = try gpa.alloc(u32, ds.n_features);

        return .{
            .gpa = gpa,
            .pool = pool,
            .ds = ds,
            .cfg = cfg,
            .bank = bank,
            .slots = slots,
            .free_slots = free_slots,
            .rows = rows,
            .g = &.{},
            .rows_out = rows_out,
            .total_partial = total_partial,
            .part_counts = part_counts,
            .part_left = part_left,
            .part_right = part_right,
            .n_active = 0,
            .nodes = .empty,
            .cat_ids = .empty,
            .lin = .empty,
            .queue = .empty,
            .leaves = .empty,
            .all_features = all_features,
            .tree_features = tree_features,
            .level_features = level_features,
            .node_features = node_features,
            .n_tree_features = 0,
            .n_level_features = 0,
            .level_depth = -1,
            .rng = .init(cfg.seed),
        };
    }

    pub fn deinit(b: *Builder) void {
        const gpa = b.gpa;
        b.bank.deinit();
        gpa.free(b.slots);
        b.free_slots.deinit(gpa);
        gpa.free(b.rows);
        gpa.free(b.rows_out);
        gpa.free(b.total_partial);
        gpa.free(b.part_counts);
        gpa.free(b.part_left);
        gpa.free(b.part_right);
        gpa.free(b.all_features);
        gpa.free(b.tree_features);
        gpa.free(b.level_features);
        gpa.free(b.node_features);
        b.nodes.deinit(gpa);
        b.cat_ids.deinit(gpa);
        b.lin.deinit(gpa);
        b.queue.deinit(gpa);
        b.leaves.deinit(gpa);
        b.* = undefined;
    }

    inline fn slot(b: *Builder, id: u32) []hist.Bin {
        const len = b.bank.slotLen();
        return b.slots[@as(usize, id) * len ..][0..len];
    }

    fn takeSlot(b: *Builder) u32 {
        return b.free_slots.pop().?;
    }

    fn giveSlot(b: *Builder, id: u32) void {
        b.free_slots.appendAssumeCapacity(id);
    }

    /// Partial Fisher-Yates: sample `k` of `src[0..n]` into `dst`.
    fn sample(b: *Builder, src: []u32, n: usize, dst: []u32, rate: f32) usize {
        if (rate >= 1.0) {
            @memcpy(dst[0..n], src[0..n]);
            return n;
        }
        var k: usize = @intFromFloat(@round(@as(f32, @floatFromInt(n)) * rate));
        k = std.math.clamp(k, 1, n);
        const r = b.rng.random();
        var i: usize = 0;
        while (i < k) : (i += 1) {
            const j = i + r.uintLessThan(usize, n - i);
            std.mem.swap(u32, &src[i], &src[j]);
            dst[i] = src[i];
        }
        return k;
    }

    /// Features whose histograms are materialised for every node of this tree.
    ///
    /// This is deliberately the per-*tree* sample and never the per-level or
    /// per-node one. A node's histogram is derived from its parent's by
    /// subtraction, which is only valid where parent and child cover the same
    /// features; sampling per node would subtract against bins the parent
    /// never accumulated. Level/node sampling therefore restricts which
    /// features are *considered* for a split, not which are built — which is
    /// also what XGBoost and LightGBM do.
    fn treeFeatures(b: *const Builder) []const u32 {
        return b.tree_features[0..b.n_tree_features];
    }

    fn featuresFor(b: *Builder, depth: u32) []u32 {
        if (b.level_depth != @as(i64, depth)) {
            b.n_level_features = b.sample(
                b.tree_features,
                b.n_tree_features,
                b.level_features,
                b.cfg.colsample_bylevel,
            );
            b.level_depth = @intCast(depth);
        }
        const k = b.sample(
            b.level_features,
            b.n_level_features,
            b.node_features,
            b.cfg.colsample_bynode,
        );
        return b.node_features[0..k];
    }

    fn splitParams(b: *const Builder) split.SplitParams {
        return .{
            .lambda = b.cfg.lambda,
            .alpha = b.cfg.alpha,
            .min_split_gain = b.cfg.min_split_gain,
            .min_child_weight = b.cfg.min_child_weight,
            .min_child_samples = b.cfg.min_child_samples,
            .max_delta_step = b.cfg.max_delta_step,
            .cat_optimal = b.cfg.cat_split == .optimal,
            .cat_smooth = b.cfg.cat_smooth,
            .cat_l2 = b.cfg.cat_l2,
            .max_cat_threshold = b.cfg.max_cat_threshold,
            .max_cat_to_onehot = b.cfg.max_cat_to_onehot,
            .min_data_per_group = b.cfg.min_data_per_group,
        };
    }

    /// Chunks the root sum is split into. Fixed, not derived from the thread
    /// count, so the result does not depend on `--n_threads`.
    const total_chunks: usize = 64;
    /// Below this many rows the barrier costs more than the sum saves.
    const total_parallel_min: usize = 1 << 15;

    /// Below this many rows the barriers cost more than the scan saves.
    ///
    /// Tuned rather than guessed: sweeping it, 32768 (the original guess)
    /// spends 360 ms in partition on a 200-tree fit where 2048 spends 271 ms,
    /// and the curve rises monotonically above that. Output is identical at
    /// every setting.
    const parallel_partition_min: usize = 2048;

    /// Takes the four scalars the decision needs, not the whole `Split`.
    ///
    /// `Split` is 160 bytes once a categorical split carries its id array
    /// inline, and this runs once per row per partition pass. Reading the
    /// decision through it put partition at 2.2x; the element width of the
    /// bin, which is what two earlier guesses blamed, was worth nothing.
    inline fn goesLeft(bin: data.BinIdx, t: split.SplitTest) bool {
        if (bin == 0) return t.missing_left;
        if (t.is_cat) return split.catContains(t.ids, bin);
        return bin <= t.threshold;
    }

    /// Reorder `rows[start..end]` so the left child's rows come first.
    /// Only row ids move; gradients are looked up by row id.
    ///
    /// This is per-level O(rows) work and used to be the serial half of tree
    /// building: with it single-threaded, 16 threads bought only 1.66x over 1,
    /// and time scaled with tree *depth* rather than node count. It is now a
    /// count / prefix-sum / scatter, which is three parallel passes instead of
    /// one serial one.
    fn partition(b: *Builder, start: usize, end: usize, sp: split.Split) usize {
        if (b.ds.isWide(sp.feature))
            return b.partitionOn(data.BinIdx, b.ds.columnWide(sp.feature), start, end, sp);
        return b.partitionOn(u8, b.ds.columnNarrow(sp.feature), start, end, sp);
    }

    fn partitionOn(
        b: *Builder,
        comptime C: type,
        col: []const C,
        start: usize,
        end: usize,
        sp: split.Split,
    ) usize {
        const n = end - start;
        if (n < parallel_partition_min or b.pool.workerCount() == 1)
            return b.partitionSerialOn(C, col, start, end, sp);

        const n_chunks = b.pool.workerCount() * 4;
        const csize = (n + n_chunks - 1) / n_chunks;
        const used = (n + csize - 1) / csize;
        std.debug.assert(used <= b.part_counts.len);

        var ctx = PartCtx(C){
            .rows = b.rows,
            .rows_out = b.rows_out,
            .col = col,
            .sp = split.SplitTest.of(&sp),
            .start = start,
            .chunk = csize,
            .counts = b.part_counts[0..used],
            .left = b.part_left[0..used],
            .right = b.part_right[0..used],
        };

        // 1. How many of each chunk's rows go left.
        const t_c = prof.start();
        b.pool.parallelFor(n, &ctx, PartCtx(C).count, csize);
        prof.stop(.part_count, t_c);

        // 2. Prefix sums. `used` is a few dozen at most, so serial is right.
        var total_left: usize = 0;
        for (ctx.counts) |c| total_left += c;
        var l: usize = start;
        var r: usize = start + total_left;
        for (ctx.counts, 0..) |c, i| {
            ctx.left[i] = l;
            ctx.right[i] = r;
            l += c;
            r += (@min((i + 1) * csize, n) - i * csize) - c;
        }

        // 3. Scatter to the scratch buffers, then copy the touched range back.
        const t_s = prof.start();
        b.pool.parallelFor(n, &ctx, PartCtx(C).scatter, csize);
        prof.stop(.part_scatter, t_s);
        const t_b = prof.start();
        b.pool.parallelFor(n, &ctx, PartCtx(C).copyBack, 8192);
        prof.stop(.part_copy, t_b);

        return start + total_left;
    }

    /// The one-thread path, and stable like the parallel one.
    ///
    /// It used to be an in-place Hoare swap, which is fewer passes but leaves
    /// a node's rows in arbitrary order. That matters now: the histogram
    /// kernel reads the row-major bin matrix, and an ascending row order is
    /// what lets the hardware prefetcher follow it. Two branchless passes and
    /// a memcpy beat one pass of mispredicted swaps anyway.
    fn partitionSerialOn(
        b: *Builder,
        comptime C: type,
        col: []const C,
        start: usize,
        end: usize,
        sp: split.Split,
    ) usize {
        const rows = b.rows[start..end];
        const t = split.SplitTest.of(&sp);
        var n_left: usize = 0;
        for (rows) |row| n_left += @intFromBool(goesLeft(col[row], t));

        var li = start;
        var ri = start + n_left;
        for (rows) |row| {
            const left = goesLeft(col[row], t);
            const dst = if (left) li else ri;
            b.rows_out[dst] = row;
            li += @intFromBool(left);
            ri += @intFromBool(!left);
        }
        @memcpy(rows, b.rows_out[start..end]);
        return start + n_left;
    }

    fn makeLeaf(b: *Builder, w: Work) !void {
        const p = b.splitParams();
        // Shrinkage is applied here, so a finished tree's output is already
        // its contribution to the ensemble and no caller has to remember eta.
        const raw = split.leafWeight(w.total.g, w.total.h, p);
        const weight: f32 = @floatCast(raw * b.cfg.learning_rate);
        var n_lin: u8 = 0;
        var lin_ofs: u32 = 0;
        if (b.cfg.linear_leaves and w.n_path != 0) {
            lin_ofs = @intCast(b.lin.items.len);
            n_lin = try b.fitLinearLeaf(w);
        }
        b.nodes.items[w.node] = .{
            .is_leaf = true,
            .weight = weight,
            .n_lin = n_lin,
            .lin_ofs = lin_ofs,
        };
        try b.leaves.append(b.gpa, .{ .start = w.start, .end = w.end, .weight = weight });
        b.giveSlot(w.slot);
    }

    /// Slopes for one leaf, appended to `b.lin`. Returns how many were kept.
    ///
    /// Centring each feature on its *hessian-weighted* within-leaf mean is not
    /// cosmetic. It makes every cross term between the intercept and a slope
    /// vanish, so the intercept is exactly the constant leaf weight already
    /// computed and only the slope block has to be solved. A failure there is
    /// therefore a fallback to the constant leaf rather than an error.
    fn fitLinearLeaf(b: *Builder, w: Work) !u8 {
        const n = @min(@as(usize, w.n_path), @as(usize, b.cfg.lin_leaf_max_terms));
        const rows = b.rows[w.start..w.end];
        if (rows.len <= n + 1) return 0;

        var center: [max_path]f64 = undefined;
        var sum_h: f64 = 0;
        for (0..n) |j| center[j] = 0;
        for (rows) |r| {
            const h: f64 = b.g[r].h;
            sum_h += h;
            const rb = b.ds.bins_rm[r * b.ds.n_features ..][0..b.ds.n_features];
            for (0..n) |j| center[j] += h * binValue(b.ds, w.path[j], rb[w.path[j]]);
        }
        if (sum_h <= 0) return 0;
        for (0..n) |j| center[j] /= sum_h;

        // Upper triangle of the slope system, and its right-hand side.
        var a: [max_path][max_path]f64 = undefined;
        var rhs: [max_path]f64 = undefined;
        for (0..n) |j| {
            rhs[j] = 0;
            for (0..n) |k| a[j][k] = 0;
        }
        var x: [max_path]f64 = undefined;
        for (rows) |r| {
            const gp = b.g[r];
            const rb = b.ds.bins_rm[r * b.ds.n_features ..][0..b.ds.n_features];
            for (0..n) |j| x[j] = binValue(b.ds, w.path[j], rb[w.path[j]]) - center[j];
            for (0..n) |j| {
                rhs[j] -= @as(f64, gp.g) * x[j];
                for (j..n) |k| a[j][k] += @as(f64, gp.h) * x[j] * x[k];
            }
        }
        // Standardise before the ridge. `a[j][j]` is `sum_h * var_j`, so a
        // flat additive lambda means something different on a column measured
        // in dollars than on one measured in people -- it is not a ridge at
        // all, it is an arbitrary per-column shrinkage. Scaling each axis by
        // its own weighted sd makes lambda comparable across columns, which
        // is what `linear.zig` does to its design matrix for the same reason.
        var sd: [max_path]f64 = undefined;
        for (0..n) |j| {
            const v = a[j][j] / sum_h;
            sd[j] = if (v > 1e-30) @sqrt(v) else 0;
        }
        for (0..n) |j| {
            if (sd[j] == 0) return 0; // constant within the leaf: no slope to fit
            rhs[j] /= sd[j];
            for (j..n) |k| a[j][k] /= (sd[j] * sd[k]);
        }
        for (0..n) |j| a[j][j] += b.cfg.lin_leaf_lambda * sum_h;

        if (!choleskySolve(&a, &rhs, n)) return 0;
        for (0..n) |j| rhs[j] /= sd[j];

        var kept: u8 = 0;
        for (0..n) |j| {
            const c = rhs[j] * b.cfg.learning_rate;
            if (!std.math.isFinite(c) or c == 0) continue;
            try b.lin.append(b.gpa, .{
                .feature = w.path[j],
                .coef = @floatCast(c),
                .center = @floatCast(center[j]),
            });
            kept += 1;
        }
        return kept;
    }

    /// Gradient and hessian sums over a node's rows.
    ///
    /// Only the root needs this — every other node inherits its total from
    /// its parent's split — but the root is *every* row, so on 668k rows over
    /// 200 trees the serial version was 46 ms, 7% of a fit and one of the
    /// three things holding 8-thread scaling to 4.8x against xgboost's 5.3x.
    ///
    /// Summed in a fixed number of chunks, reduced in chunk order, regardless
    /// of how many threads are running. A plain `parallelFor` over rows would
    /// group the additions differently at one thread than at eight and make
    /// the model depend on `--n_threads`, which is a property worth more than
    /// the milliseconds.
    fn totalOf(b: *Builder, rows: []const u32) hist.Bin {
        const n = rows.len;
        // Deliberately *not* also conditioned on the worker count: that would
        // make one thread group the additions differently from eight, which
        // is the thing this is arranged to avoid. `parallelFor` already runs
        // the chunks inline when there is one worker.
        if (n < total_parallel_min) {
            var g: f64 = 0;
            var h: f64 = 0;
            for (rows) |r| {
                const p = b.g[r];
                g += p.g;
                h += p.h;
            }
            return .{ .g = g, .h = h, .n = @floatFromInt(n) };
        }

        var ctx = TotalCtx{
            .g = b.g,
            .rows = rows,
            .size = (n + total_chunks - 1) / total_chunks,
            .partial = b.total_partial,
        };
        b.pool.parallelFor(total_chunks, &ctx, TotalCtx.run, 1);

        var g: f64 = 0;
        var h: f64 = 0;
        for (b.total_partial) |t| {
            g += t.g;
            h += t.h;
        }
        return .{ .g = g, .h = h, .n = @floatFromInt(n) };
    }

    /// Fill `rows[0..n_active]` with the rows this tree will see.
    ///
    /// An explicit `subset` wins outright. Otherwise `bootstrap` draws
    /// `subsample * n` rows *with* replacement (bagging, which is what makes a
    /// forest a forest), and without it we shuffle and take a prefix.
    fn selectRows(b: *Builder, subset: ?[]const u32) void {
        const n_all = b.ds.n_rows;

        if (subset) |sel| {
            @memcpy(b.rows[0..sel.len], sel);
            b.n_active = sel.len;
            return;
        }

        var k: usize = n_all;
        if (b.cfg.subsample < 1.0) {
            k = @intFromFloat(@round(@as(f32, @floatFromInt(n_all)) * b.cfg.subsample));
            k = std.math.clamp(k, 1, n_all);
        }

        const r = b.rng.random();
        if (b.cfg.bootstrap) {
            // With replacement: duplicates are the point. A row drawn twice
            // simply carries twice the weight in this tree's histograms.
            for (b.rows[0..k]) |*slot_row| slot_row.* = r.uintLessThan(u32, @intCast(n_all));
            b.n_active = k;
            return;
        }

        for (b.rows, 0..) |*row, i| row.* = @intCast(i);
        if (k < n_all) {
            var i: usize = 0;
            while (i < k) : (i += 1) {
                const j = i + r.uintLessThan(usize, n_all - i);
                std.mem.swap(u32, &b.rows[i], &b.rows[j]);
            }
        }
        b.n_active = k;
    }

    /// Grow one tree against `gradients` (indexed by original row id).
    /// Returns the tree; `leafSpans()` describes where its leaves' rows landed.
    pub fn grow(b: *Builder, gradients: []const hist.GradPair) !Tree {
        return b.growRows(gradients, null);
    }

    /// As `grow`, but over an explicit row set. GOSS uses this to hand the
    /// builder the rows it chose; passing null falls back to the config's own
    /// `subsample`/`bootstrap` policy.
    pub fn growRows(
        b: *Builder,
        gradients: []const hist.GradPair,
        subset: ?[]const u32,
    ) !Tree {
        b.nodes.clearRetainingCapacity();
        b.cat_ids.clearRetainingCapacity();
        b.lin.clearRetainingCapacity();
        b.queue.clearRetainingCapacity();
        b.leaves.clearRetainingCapacity();
        b.level_depth = -1;

        const t_sel = prof.start();
        b.selectRows(subset);
        prof.stop(.select_rows, t_sel);

        b.g = gradients;

        b.n_tree_features = b.sample(
            b.all_features,
            b.ds.n_features,
            b.tree_features,
            b.cfg.colsample_bytree,
        );

        const p = b.splitParams();
        const leaf_cap: usize = if (b.cfg.max_leaves != 0) b.cfg.max_leaves else std.math.maxInt(usize);

        // --- root ---
        try b.nodes.append(b.gpa, .{});
        const root_slot = b.takeSlot();
        const root_total = b.totalOf(b.rows[0..b.n_active]);
        const tree_feats = b.treeFeatures();
        const root_search = b.featuresFor(0);
        const t_rh = prof.start();
        hist.build(
            b.pool,
            &b.bank,
            b.ds,
            b.rows[0..b.n_active],
            b.g,
            tree_feats,
            b.slot(root_slot),
        );
        prof.stop(.hist_build, t_rh);
        const t_rs = prof.start();
        const root_split = split.bestSplit(&b.bank, b.slot(root_slot), b.ds, root_search, root_total, p);
        prof.stop(.best_split, t_rs);
        try b.queue.append(b.gpa, .{
            .node = 0,
            .start = 0,
            .end = b.n_active,
            .depth = 0,
            .total = root_total,
            .slot = root_slot,
            .split = root_split,
        });

        // --- expansion ---
        while (b.queue.items.len != 0) {
            const idx = b.pickNext();
            const w = b.queue.orderedRemove(idx);

            const leaf_count = b.leaves.items.len + b.queue.items.len + 1;
            const depth_capped = b.cfg.max_depth != 0 and w.depth >= b.cfg.max_depth;
            if (!w.split.valid() or depth_capped or leaf_count >= leaf_cap) {
                try b.makeLeaf(w);
                continue;
            }

            const t_part = prof.start();
            const mid = b.partition(w.start, w.end, w.split);
            prof.stop(.partition, t_part);
            // A split the histogram endorsed but the partition cannot realise
            // (every row on one side) would loop forever; treat it as a leaf.
            if (mid == w.start or mid == w.end) {
                try b.makeLeaf(w);
                continue;
            }

            const li: u32 = @intCast(b.nodes.items.len);
            try b.nodes.append(b.gpa, .{});
            const ri: u32 = @intCast(b.nodes.items.len);
            try b.nodes.append(b.gpa, .{});
            var cat_ofs: u32 = 0;
            if (w.split.is_cat) {
                cat_ofs = @intCast(b.cat_ids.items.len);
                try b.cat_ids.appendSlice(b.gpa, w.split.cat_ids[0..w.split.n_cat]);
            }
            b.nodes.items[w.node] = .{
                .feature = w.split.feature,
                .threshold = w.split.threshold,
                .missing_left = w.split.missing_left,
                .is_cat = w.split.is_cat,
                .n_cat = w.split.n_cat,
                .cat_ofs = cat_ofs,
                .is_leaf = false,
                .left = li,
                .right = ri,
            };

            // Will these children only ever become leaves?
            //
            // `makeLeaf` reads a node's total, its row span and its slot --
            // never its histogram or its split. So for a child that is
            // certain to be a leaf, building the histogram and searching it
            // is pure waste, and it is not a small amount: half of a
            // depthwise tree's nodes are its last level, so half of all
            // histogram builds and split searches were thrown away.
            //
            // Two cases are decidable here. The depth cap is static. The leaf
            // cap is too, because `leaf_count` at pop time never decreases --
            // a pop either makes a leaf (queue -1, leaves +1) or splits
            // (queue -1 +2, leaves +0) -- so once the budget is reached every
            // remaining pop is a leaf. `b.queue` is post-removal here, and
            // the next pop will see `leaves + queue + 2`.
            const at_depth_cap = b.cfg.max_depth != 0 and w.depth + 1 >= b.cfg.max_depth;
            const at_leaf_cap = b.leaves.items.len + b.queue.items.len + 2 >= leaf_cap;
            const children_are_leaves = at_depth_cap or at_leaf_cap;

            const slot_l = b.takeSlot();
            const slot_r = b.takeSlot();

            // Accumulate the smaller side; derive the larger by subtraction.
            const n_left = mid - w.start;
            const n_right = w.end - mid;
            if (children_are_leaves) {
                // Nothing to build. The slots are still taken and released
                // through the normal path so the free list behaves the same.
            } else if (n_left <= n_right) {
                const t_hb = prof.start();
                hist.build(b.pool, &b.bank, b.ds, b.rows[w.start..mid], b.g, tree_feats, b.slot(slot_l));
                prof.stop(.hist_build, t_hb);
                const t_hs = prof.start();
                hist.subtract(b.pool, &b.bank, b.slot(slot_r), b.slot(w.slot), b.slot(slot_l));
                prof.stop(.hist_subtract, t_hs);
            } else {
                const t_hb = prof.start();
                hist.build(b.pool, &b.bank, b.ds, b.rows[mid..w.end], b.g, tree_feats, b.slot(slot_r));
                prof.stop(.hist_build, t_hb);
                const t_hs = prof.start();
                hist.subtract(b.pool, &b.bank, b.slot(slot_l), b.slot(w.slot), b.slot(slot_r));
                prof.stop(.hist_subtract, t_hs);
            }
            b.giveSlot(w.slot);

            // Each child draws its own candidate features, as XGBoost does.
            // The draws happen even when the search is skipped: they consume
            // the builder's RNG, and skipping them would shift every later
            // sample and change the trees under `colsample_bylevel/bynode`.
            const t_bs = prof.start();
            var left_split: split.Split = .{};
            var right_split: split.Split = .{};
            // Draw, use, draw, use. `featuresFor` hands back a slice of one
            // shared buffer, so hoisting both draws above both searches makes
            // them alias and the left child gets searched with the right
            // child's sample -- silently, and only when a colsample is below
            // 1.0. The draws happen even when the search is skipped: they
            // consume the builder's RNG, and dropping them would shift every
            // later sample.
            const left_search = b.featuresFor(w.depth + 1);
            if (!children_are_leaves)
                left_split = split.bestSplit(&b.bank, b.slot(slot_l), b.ds, left_search, w.split.left, p);
            const right_search = b.featuresFor(w.depth + 1);
            if (!children_are_leaves)
                right_split = split.bestSplit(&b.bank, b.slot(slot_r), b.ds, right_search, w.split.right, p);
            prof.stop(.best_split, t_bs);

            // Children inherit the path and add the feature just tested,
            // unless it is categorical (a dictionary id is not a number to
            // fit a slope in) or already present.
            var path = w.path;
            var n_path = w.n_path;
            if (b.cfg.linear_leaves and
                b.ds.kinds[w.split.feature] == .numeric and
                n_path < max_path)
            {
                var seen = false;
                for (path[0..n_path]) |f| seen = seen or f == w.split.feature;
                if (!seen) {
                    path[n_path] = w.split.feature;
                    n_path += 1;
                }
            }

            try b.queue.append(b.gpa, .{
                .node = li,
                .start = w.start,
                .end = mid,
                .depth = w.depth + 1,
                .total = w.split.left,
                .slot = slot_l,
                .split = left_split,
                .path = path,
                .n_path = n_path,
            });
            try b.queue.append(b.gpa, .{
                .node = ri,
                .start = mid,
                .end = w.end,
                .depth = w.depth + 1,
                .total = w.split.right,
                .slot = slot_r,
                .split = right_split,
                .path = path,
                .n_path = n_path,
            });
        }

        const nodes = try b.gpa.dupe(Node, b.nodes.items);
        errdefer b.gpa.free(nodes);
        const ids: []data.BinIdx = if (b.cat_ids.items.len == 0)
            &.{}
        else
            try b.gpa.dupe(data.BinIdx, b.cat_ids.items);
        errdefer if (ids.len != 0) b.gpa.free(ids);
        const lin: []LinTerm = if (b.lin.items.len == 0)
            &.{}
        else
            try b.gpa.dupe(LinTerm, b.lin.items);
        return .{ .nodes = nodes, .cat_ids = ids, .lin = lin };
    }

    /// Depthwise takes the oldest pending node (breadth-first); lossguide takes
    /// the one whose split buys the most.
    fn pickNext(b: *Builder) usize {
        if (b.cfg.grow_policy == .depthwise) return 0;
        var best: usize = 0;
        var best_gain = -std.math.inf(f64);
        for (b.queue.items, 0..) |w, i| {
            if (w.split.gain > best_gain) {
                best_gain = w.split.gain;
                best = i;
            }
        }
        return best;
    }

    pub fn leafSpans(b: *const Builder) []const LeafSpan {
        return b.leaves.items;
    }

    pub fn activeRows(b: *const Builder) []const u32 {
        return b.rows[0..b.n_active];
    }
};

/// Shared state for the three passes of a parallel partition.
///
/// `parallelFor` hands out fixed-size chunks from a cursor, so `begin / chunk`
/// recovers which chunk a worker got — which is what lets the counting pass
/// and the scatter pass agree on where each chunk's output belongs without any
/// coordination between them.
/// One chunk of the root's gradient sum.
const TotalCtx = struct {
    g: []const hist.GradPair,
    rows: []const u32,
    size: usize,
    partial: []hist.Bin,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *TotalCtx = @ptrCast(@alignCast(ctx));
        // `begin`/`end` index chunks, not rows, so a chunk always covers the
        // same rows however the pool hands them out.
        var c = begin;
        while (c < end) : (c += 1) {
            const lo = @min(c * self.size, self.rows.len);
            const hi = @min(lo + self.size, self.rows.len);
            var g: f64 = 0;
            var h: f64 = 0;
            for (self.rows[lo..hi]) |r| {
                const p = self.g[r];
                g += p.g;
                h += p.h;
            }
            self.partial[c] = .{ .g = g, .h = h, .n = @floatFromInt(hi - lo) };
        }
    }
};

/// Generic over the column's element type so a table that fits in a byte
/// keeps reading bytes here.
///
/// This is the one place the bin width is visibly paid. Partitioning walks a
/// single feature down a node's rows, and those rows are scattered once the
/// tree is more than a level deep, so each tends to want its own cache line
/// and the element size is paid in full rather than amortised. Measured when
/// the whole matrix went to `u16`: 51 ms -> 110 ms on adult, while the
/// row-major accumulate did not move. The row-major mirror is therefore
/// uniformly `u16` and only this stays narrow.
fn PartCtx(comptime C: type) type {
    return struct {
        const Self = @This();

        rows: []u32,
        rows_out: []u32,
        col: []const C,
        sp: split.SplitTest,
        start: usize,
        chunk: usize,
        counts: []usize,
        left: []usize,
        right: []usize,

        fn count(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
            _ = worker;
            const self: *Self = @ptrCast(@alignCast(ctx));
            var n: usize = 0;
            for (self.rows[self.start + begin .. self.start + end]) |row| {
                if (Builder.goesLeft(self.col[row], self.sp)) n += 1;
            }
            self.counts[begin / self.chunk] = n;
        }

        fn scatter(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
            _ = worker;
            const self: *Self = @ptrCast(@alignCast(ctx));
            const c = begin / self.chunk;
            var li = self.left[c];
            var ri = self.right[c];
            var i = self.start + begin;
            while (i < self.start + end) : (i += 1) {
                const row = self.rows[i];
                // Branchless: which side a row takes is close to a coin flip near
                // a good split, so a branch here mispredicts about half the time.
                // Selecting the cursor instead costs a cmov and nothing else.
                const left = Builder.goesLeft(self.col[row], self.sp);
                const dst = if (left) li else ri;
                self.rows_out[dst] = row;
                li += @intFromBool(left);
                ri += @intFromBool(!left);
            }
        }

        fn copyBack(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
            _ = worker;
            const self: *Self = @ptrCast(@alignCast(ctx));
            const a = self.start + begin;
            const b_ = self.start + end;
            @memcpy(self.rows[a..b_], self.rows_out[a..b_]);
        }
    };
}
