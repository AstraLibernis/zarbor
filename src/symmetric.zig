// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Symmetric (oblivious) trees, CatBoost's default: one split per depth, shared by every node at
//! that depth, so a depth-D tree is D yes/no questions and 2^D leaves. Re-implemented from
//! CatBoost 1.2.10's documented behaviour and source reading (docs/catboost.md); no CatBoost code.
//!
//! The fitted tree is an ordinary `tree.Tree`: a complete binary tree whose nodes at depth d all
//! test split d, so prediction, the model file and blending need nothing new. A row's leaf index is
//! `sum_d [bin > threshold_d] << d`, the left child holding bins <= threshold (missing included,
//! CatBoost's `nan_mode = Min`).

const std = @import("std");
const data = @import("data.zig");
const Dataset = data.Dataset;
const hist = @import("hist.zig");
const tree = @import("tree.zig");
const Pool = @import("pool.zig").Pool;
const Objective = @import("objective.zig").Objective;

/// Deepest supported tree: 2^16 leaves, and the leaf index fits a `u16`.
pub const max_depth: u32 = 16;

/// How a level's candidate split is scored, summed over every leaf at that level.
pub const ScoreFunction = enum {
    /// Cosine for `symmetric`, gain otherwise.
    auto,
    /// XGBoost/LightGBM Newton gain, `G^2 / (H + lambda)` per side.
    gain,
    /// CatBoost's default: the cosine between the gradient vector and the step the split would take,
    /// leaves estimated as `G / (n + lambda)`. First derivatives only.
    cosine,
    /// CatBoost's `L2`: `sum v * G` with the same leaf estimates.
    l2,
};

/// Row weights drawn once per tree for split scoring only; leaf values always use every row
/// unweighted, as CatBoost's do.
pub const BootstrapType = enum {
    none,
    /// Keep each row with probability `subsample`, weight 1.
    bernoulli,
    /// Weight `(-ln u)^bagging_temperature`, u uniform: every row kept, weights vary.
    bayesian,
    /// Minimal variance sampling (CatBoost's CPU default): keep rows with large gradients more
    /// often, reweighted by 1/probability so the sums stay unbiased.
    mvs,
};

/// Rows per MVS block, each with its own threshold and random stream (CatBoost's `BlockSize`).
const mvs_block: usize = 8192;

/// One histogram cell: gradient sum, row count, hessian sum.
const Cell = struct {
    g: f64 = 0,
    n: f64 = 0,
    h: f64 = 0,

    fn add(a: Cell, b: Cell) Cell {
        return .{ .g = a.g + b.g, .n = a.n + b.n, .h = a.h + b.h };
    }
    fn sub(a: Cell, b: Cell) Cell {
        return .{ .g = a.g - b.g, .n = a.n - b.n, .h = a.h - b.h };
    }
};

pub const Settings = struct {
    depth: u32,
    lambda: f64,
    learning_rate: f64,
    score: ScoreFunction,
    /// Newton steps per leaf; >1 re-evaluates the derivatives at the moved score (no backtracking).
    leaf_iterations: u32,
    bootstrap: BootstrapType = .none,
    /// Expected kept fraction for `bernoulli` and `mvs`.
    subsample: f64 = 1,
    bagging_temperature: f64 = 1,
    /// MVS's lambda; null takes CatBoost's: the previous tree's mean |leaf value| squared, or on
    /// the first tree the mean |gradient| squared.
    mvs_reg: ?f64 = null,
    /// Scale of the normal noise added to split scores (CatBoost's `random_strength`), shrinking as
    /// the model grows; 0 disables.
    random_strength: f64 = 0,
    seed: u64 = 0,
    /// Categorical columns as CatBoost has them: one-hot up to `one_hot_max_size` levels, ordered
    /// target statistics (CTRs) above. Off: categoricals split on their ordinal ids.
    ctr: bool = false,
    one_hot_max_size: u32 = 2,
    /// Penalty on CTRs not yet used in any tree, by their number of distinct keys.
    model_size_reg: f64 = 0.5,
    /// CatBoost's ordered boosting: splits are scored by leaf estimates from a prefix of the rows
    /// against the rows just after it, so no row's gradient is judged by a model that saw it.
    ordered: bool = false,
};

/// One prefix of the ordered rows: a model trained on rows `0..body` scores rows `body..tail`.
pub const BodyTail = struct { body: usize, tail: usize };

/// CatBoost's dynamic folds (fold.cpp): the first body is `min(100, n / 50)` rows (1 when
/// `n <= 500`), each tail twice its body, each next body the previous tail, until all rows.
pub fn bodyTails(gpa: std.mem.Allocator, n: usize) ![]BodyTail {
    var list: std.ArrayList(BodyTail) = .empty;
    errdefer list.deinit(gpa);
    var body: usize = if (n > 500) @min(100, n / 50) else 1;
    while (true) {
        const tail = @min(n, 2 * body);
        try list.append(gpa, .{ .body = body, .tail = tail });
        if (tail >= n) break;
        body = tail;
    }
    return list.toOwnedSlice(gpa);
}

/// What a candidate split tests.
pub const CandKind = enum {
    /// `bin > threshold` on a dataset column (numeric, or a categorical's ordinal ids).
    numeric,
    /// `bin == level` on a categorical with at most `one_hot_max_size` levels.
    onehot,
    /// `bucket > threshold` on one of a categorical's ordered target statistics.
    ctr,
};

/// CTR types per categorical, in CatBoost's default order: Borders with priors 0, 0.5 and 1, then
/// Counter.
const ctr_priors = [3]f32{ 0, 0.5, 1 };
const n_ctr_types = ctr_priors.len + 1;
/// A CTR value in [0, 1] becomes `trunc(value * 15)`: 16 buckets.
const ctr_buckets = 16;

pub const Cand = struct {
    kind: CandKind,
    /// Dataset column.
    feature: u32,
    /// CTR type: 0..2 Borders with `ctr_priors[t]`, 3 Counter.
    ctr_type: u8 = 0,
    /// Bins of the column this candidate scores.
    nb: u16,
    /// CTR: training bucket per row (online: counts over earlier rows only).
    col: []u8 = &.{},
    /// CTR: bucket per level from counts over every training row, what prediction uses.
    final: []u8 = &.{},
    /// CTR: distinct levels in training, for `model_size_reg`.
    uniq: u32 = 0,
    /// One-hot: which levels occur in training (only those are split on).
    present: []bool = &.{},
    /// CTR: chosen at some level already, so `model_size_reg` no longer applies. CatBoost keys
    /// this on (Borders or Counter, column), so choosing one prior frees the other two.
    used: bool = false,
};

pub const Builder = struct {
    gpa: std.mem.Allocator,
    pool: *Pool,
    ds: *const Dataset,
    s: Settings,
    /// Each row's leaf index in the tree being grown.
    leaf: []u16,
    /// Split candidates, CatBoost's order: numeric columns, then one-hot, then CTRs. Ties go to
    /// the earlier one.
    cands: []Cand,
    /// Histograms for every candidate over the deepest scored level: candidate c's leaf l bin b is
    /// `cells[off[c] + l * nb_c + b]`.
    cells: []Cell,
    off: []usize,
    /// Whether candidate c's column has a missing row: the missing-vs-present cut (threshold 0) is
    /// offered only then, as CatBoost adds its `nan_mode = Min` border only for columns with NaN.
    has_missing: []bool,
    /// Per-candidate best split of the level being searched.
    best: []Best,
    /// Largest `uniq` among CTR candidates.
    max_uniq: u32 = 0,
    /// Leaf values of the last grown tree, already scaled by the learning rate.
    values: []f64,
    /// Bootstrap weight per row for the tree being grown; empty without bootstrap.
    weights: []f32,
    /// MVS scratch: one candidate per row, each block sorting its own slice.
    mvs_scratch: []f64,
    /// Trees grown so far: the noise decay and every random stream key on it.
    iteration: u32 = 0,
    /// Mean |leaf value| of the previous tree, for MVS's lambda.
    prev_mean_leaf: ?f64 = null,
    /// Split-score noise scale for the tree being grown.
    sigma: f64 = 0,
    /// Ordered boosting: the prefixes, each prefix model's score on its rows (`approx[ap_off[k]..]`,
    /// `tail` long), its derivatives there, a second histogram bank for the tails, per-candidate
    /// score accumulators, and the derivative each row is weighted by (its own tail's).
    bts: []BodyTail = &.{},
    approx: []f64 = &.{},
    deriv: []f64 = &.{},
    ap_off: []usize = &.{},
    cells_tail: []Cell = &.{},
    acc_num: []f64 = &.{},
    acc_den: []f64 = &.{},
    acc_off: []usize = &.{},
    tail_grads: []hist.GradPair = &.{},

    const Best = struct { score: f64, threshold: u32 };

    pub fn init(gpa: std.mem.Allocator, pool: *Pool, ds: *const Dataset, s: Settings) !Builder {
        if (s.depth == 0 or s.depth > max_depth) return error.BadSymmetricDepth;
        const cands = try candidates(gpa, ds, s);
        errdefer freeCands(gpa, cands);
        var max_uniq: u32 = 0;
        for (cands) |c| if (c.kind == .ctr) {
            max_uniq = @max(max_uniq, c.uniq);
        };
        const leaf = try gpa.alloc(u16, ds.n_rows);
        errdefer gpa.free(leaf);
        const off = try gpa.alloc(usize, cands.len + 1);
        errdefer gpa.free(off);
        const scored_leaves: usize = @as(usize, 1) << @intCast(s.depth - 1);
        off[0] = 0;
        for (cands, 0..) |c, i| off[i + 1] = off[i] + scored_leaves * c.nb;
        const cells = try gpa.alloc(Cell, off[cands.len]);
        errdefer gpa.free(cells);
        const has_missing = try gpa.alloc(bool, cands.len);
        errdefer gpa.free(has_missing);
        for (has_missing, cands) |*m, c| {
            m.* = false;
            if (c.kind != .numeric) continue;
            for (0..ds.n_rows) |r| if (binAt(ds, c.feature, r) == 0) {
                m.* = true;
                break;
            };
        }
        const best = try gpa.alloc(Best, cands.len);
        errdefer gpa.free(best);
        const values = try gpa.alloc(f64, @as(usize, 1) << @intCast(s.depth));
        errdefer gpa.free(values);
        const weights = try gpa.alloc(f32, if (s.bootstrap == .none) 0 else ds.n_rows);
        errdefer gpa.free(weights);
        const mvs_scratch = try gpa.alloc(f64, if (s.bootstrap == .mvs) ds.n_rows else 0);
        errdefer gpa.free(mvs_scratch);
        var ord: Ordered = .{};
        if (s.ordered) ord = try Ordered.init(gpa, ds, cands, cells.len);
        return .{
            .gpa = gpa,
            .pool = pool,
            .ds = ds,
            .s = s,
            .leaf = leaf,
            .cands = cands,
            .max_uniq = max_uniq,
            .cells = cells,
            .off = off,
            .has_missing = has_missing,
            .best = best,
            .values = values,
            .weights = weights,
            .mvs_scratch = mvs_scratch,
            .bts = ord.bts,
            .approx = ord.approx,
            .deriv = ord.deriv,
            .ap_off = ord.ap_off,
            .cells_tail = ord.cells_tail,
            .acc_num = ord.acc_num,
            .acc_den = ord.acc_den,
            .acc_off = ord.acc_off,
            .tail_grads = ord.tail_grads,
        };
    }

    pub fn deinit(b: *Builder) void {
        freeCands(b.gpa, b.cands);
        b.gpa.free(b.leaf);
        b.gpa.free(b.cells);
        b.gpa.free(b.off);
        b.gpa.free(b.has_missing);
        b.gpa.free(b.best);
        b.gpa.free(b.values);
        b.gpa.free(b.weights);
        b.gpa.free(b.mvs_scratch);
        if (b.bts.len != 0) {
            const ord: Ordered = .{ .bts = b.bts, .approx = b.approx, .deriv = b.deriv, .ap_off = b.ap_off, .cells_tail = b.cells_tail, .acc_num = b.acc_num, .acc_den = b.acc_den, .acc_off = b.acc_off, .tail_grads = b.tail_grads };
            ord.deinit(b.gpa);
        }
        b.* = undefined;
    }

    /// Grow one tree on `grads` (taken at `raw`). Leaf values come from `leaf_iterations` Newton
    /// steps on all rows; the caller adds `values[leaf[r]]` to `raw[r]` (or predicts the tree).
    /// Caller owns the returned tree.
    pub fn grow(
        b: *Builder,
        grads: []const hist.GradPair,
        raw: []const f32,
        labels: []const f32,
        objective: Objective,
        scale_pos_weight: f32,
    ) !tree.Tree {
        const ds = b.ds;
        @memset(b.leaf, 0);
        var splits: [max_depth]Split = undefined;
        var depth: u32 = 0;
        // Under ordered boosting the derivatives a split is judged by are each prefix model's, and
        // bootstrap weights and noise follow the tail rows' (CatBoost's mvs.cpp, greedy_tensor_search).
        if (b.s.ordered) {
            // Every prefix model starts where the booster does.
            if (b.iteration == 0) for (b.bts, 0..) |bt, k| {
                for (b.approx[b.ap_off[k]..][0..bt.tail], raw[0..bt.tail]) |*a, r0| a.* = r0;
            };
            b.prefixDerivatives(labels, objective, scale_pos_weight);
        }
        const sample_grads = if (b.s.ordered) b.tail_grads else grads;
        if (b.s.bootstrap != .none) b.sampleWeights(sample_grads);
        b.sigma = if (b.s.random_strength > 0) b.noiseScale(sample_grads) else 0;

        while (depth < b.s.depth) {
            const n_leaves = @as(usize, 1) << @intCast(depth);
            if (b.s.ordered) {
                @memset(b.acc_num, 0);
                @memset(b.acc_den, 0);
                for (b.bts, 0..) |_, k| {
                    var octx = OrderedCtx{ .b = b, .bt = k, .n_leaves = n_leaves };
                    b.pool.parallelFor(b.cands.len, &octx, OrderedCtx.run, 1);
                }
            } else {
                var hctx = HistCtx{ .b = b, .grads = grads, .n_leaves = n_leaves };
                b.pool.parallelFor(b.cands.len, &hctx, HistCtx.run, 1);
            }
            var sctx = ScoreCtx{ .b = b, .n_leaves = n_leaves, .level = depth };
            b.pool.parallelFor(b.cands.len, &sctx, ScoreCtx.run, 1);

            // Strict `>` in candidate order: ties go to the earlier candidate, then (within one,
            // in `ScoreCtx`) to the lower threshold, as CatBoost's scan does. With noise, each
            // candidate's best (noise-free) score gets a fresh draw here, in order. A CTR not yet
            // used by any tree is scaled down by `model_size_reg` (CatBoost's `GetCatFeatureWeight`).
            var sel = std.Random.DefaultPrng.init(mix(b.s.seed, b.iteration, depth, std.math.maxInt(u32)));
            var win: ?usize = null;
            var win_score: f64 = -std.math.inf(f64);
            for (b.best, b.cands, 0..) |c, cand, f| {
                if (!std.math.isFinite(c.score)) continue;
                var s = if (b.sigma > 0) c.score + b.sigma * sel.random().floatNorm(f64) else c.score;
                if (cand.kind == .ctr and !cand.used and b.s.model_size_reg > 0) {
                    const ratio = @as(f64, @floatFromInt(cand.uniq)) / @as(f64, @floatFromInt(b.max_uniq));
                    s *= std.math.pow(f64, 1 + ratio, -b.s.model_size_reg);
                }
                if (win == null or s > win_score) {
                    win = f;
                    win_score = s;
                }
            }
            const f = win orelse break;
            const sp: Split = .{ .cand = @intCast(f), .threshold = @intCast(b.best[f].threshold) };
            // Marked on choice, as CatBoost does: later levels of this tree see it as used, even
            // if the redundancy rule removes the split.
            if (b.cands[f].kind == .ctr) {
                const counter = b.cands[f].ctr_type == ctr_priors.len;
                for (b.cands) |*c| {
                    if (c.kind == .ctr and c.feature == b.cands[f].feature and (c.ctr_type == ctr_priors.len) == counter) c.used = true;
                }
            }
            splits[depth] = sp;

            var actx = ApplySplit{ .b = b, .sp = sp, .bit = @intCast(depth) };
            b.pool.parallelFor(ds.n_rows, &actx, ApplySplit.run, 8192);
            depth += 1;

            // CatBoost stops early when a split (any, not only the new one) separates nothing:
            // every pair of leaves differing only in its bit has an empty member. It is removed
            // and the tree ends there.
            if (try b.redundant(depth)) |j| {
                b.removeBit(j);
                var k = j;
                while (k + 1 < depth) : (k += 1) splits[k] = splits[k + 1];
                depth -= 1;
                break;
            }
        }

        try b.leafValues(depth, grads, raw, labels, objective, scale_pos_weight);
        if (b.s.ordered) try b.updatePrefixes(depth, labels, objective, scale_pos_weight);
        const n_leaves = @as(usize, 1) << @intCast(depth);
        var sum_abs: f64 = 0;
        for (b.values[0..n_leaves]) |v| sum_abs += @abs(v);
        b.prev_mean_leaf = sum_abs / @as(f64, @floatFromInt(n_leaves));
        b.iteration += 1;
        return b.toTree(splits[0..depth]);
    }

    /// Each prefix model's derivatives on its rows, and per row its own tail's (rows of the first
    /// body take the first prefix's): what ordered scoring, bootstrap and noise read.
    fn prefixDerivatives(b: *Builder, labels: []const f32, objective: Objective, scale_pos_weight: f32) void {
        for (b.bts, 0..) |bt, k| {
            const a = b.approx[b.ap_off[k]..][0..bt.tail];
            const d = b.deriv[b.ap_off[k]..][0..bt.tail];
            for (a, d, labels[0..bt.tail]) |x, *o, y| o.* = derivatives(objective, x, y, scale_pos_weight).g;
            const from = if (k == 0) 0 else bt.body;
            for (d[from..], b.tail_grads[from..bt.tail]) |x, *o| o.* = .{ .g = @floatCast(x), .h = 0 };
        }
    }

    /// After a tree: each prefix model takes Newton steps fitted on its body rows only and applies
    /// them to every row it scores (CatBoost's approx_calcer, ordered branch).
    fn updatePrefixes(b: *Builder, depth: u32, labels: []const f32, objective: Objective, scale_pos_weight: f32) !void {
        const n_leaves = @as(usize, 1) << @intCast(depth);
        const g = try b.gpa.alloc(f64, n_leaves);
        defer b.gpa.free(g);
        const h = try b.gpa.alloc(f64, n_leaves);
        defer b.gpa.free(h);
        for (b.bts, 0..) |bt, k| {
            const a = b.approx[b.ap_off[k]..][0..bt.tail];
            @memset(g, 0);
            @memset(h, 0);
            for (a[0..bt.body], labels[0..bt.body], b.leaf[0..bt.body]) |x, y, l| {
                const d = derivatives(objective, x, y, scale_pos_weight);
                g[l] += d.g;
                h[l] += d.h;
            }
            for (g, h) |*gs, hs| gs.* = if (hs + b.s.lambda > 0) -gs.* / (hs + b.s.lambda) * b.s.learning_rate else 0;
            for (a, b.leaf[0..bt.tail]) |*x, l| x.* += g[l];
        }
    }

    /// CatBoost's noise scale: `random_strength * sqrt(mean g^2) * decay`, the decay a logistic in
    /// `ln n - iteration * learning_rate`, so the noise fades once the model has grown. Ordered
    /// boosting takes the mean over tail rows only, the first body excluded.
    fn noiseScale(b: *Builder, grads: []const hist.GradPair) f64 {
        const from = if (b.s.ordered) b.bts[0].body else 0;
        var s2: f64 = 0;
        for (grads[from..]) |gp| s2 += @as(f64, gp.g) * gp.g;
        const n_all: f64 = @floatFromInt(grads.len);
        const n: f64 = @floatFromInt(grads.len - from);
        const model_length = @as(f64, @floatFromInt(b.iteration)) * b.s.learning_rate;
        const e = @exp(@log(n_all) - model_length);
        return b.s.random_strength * @sqrt(s2 / n) * (e / (1 + e));
    }

    /// This tree's bootstrap weights. Each block of rows has its own random stream keyed on the
    /// tree and the block, so the weights do not depend on the thread count.
    fn sampleWeights(b: *Builder, grads: []const hist.GradPair) void {
        var lambda: f64 = 0;
        if (b.s.bootstrap == .mvs) lambda = b.s.mvs_reg orelse blk: {
            const m = b.prev_mean_leaf orelse first: {
                var s: f64 = 0;
                for (grads) |gp| s += @abs(@as(f64, gp.g));
                break :first s / @as(f64, @floatFromInt(grads.len));
            };
            break :blk m * m;
        };
        var ctx = WeightCtx{ .b = b, .grads = grads, .lambda = lambda };
        const n_blocks = (b.ds.n_rows + mvs_block - 1) / mvs_block;
        b.pool.parallelFor(n_blocks, &ctx, WeightCtx.run, 1);
    }

    /// The first split bit `j` whose every leaf pair differing only in `j` has an empty side.
    fn redundant(b: *Builder, depth: u32) !?u32 {
        const n_leaves = @as(usize, 1) << @intCast(depth);
        const filled = try b.gpa.alloc(bool, n_leaves);
        defer b.gpa.free(filled);
        @memset(filled, false);
        for (b.leaf) |l| filled[l] = true;
        var j: u32 = 0;
        while (j < depth) : (j += 1) {
            const bit = @as(usize, 1) << @intCast(j);
            var useless = true;
            for (0..n_leaves) |i| {
                if (i & bit != 0) continue;
                if (filled[i] and filled[i | bit]) {
                    useless = false;
                    break;
                }
            }
            if (useless) return j;
        }
        return null;
    }

    /// Drop bit `j` from every row's leaf index; the bits above it move down one.
    fn removeBit(b: *Builder, j: u32) void {
        const low: u16 = @intCast((@as(u32, 1) << @intCast(j)) - 1);
        for (b.leaf) |*l| l.* = (l.* & low) | ((l.* >> @intCast(j + 1)) << @intCast(j));
    }

    /// Newton leaf values over all rows: `-G / (H + lambda)`, repeated `leaf_iterations` times with
    /// the derivatives re-taken at the moved score, scaled by the learning rate at the end. As in
    /// CatBoost, G carries no `lambda * value` term, so lambda damps each step and repeated steps
    /// head for the unregularised leaf optimum.
    fn leafValues(
        b: *Builder,
        depth: u32,
        grads: []const hist.GradPair,
        raw: []const f32,
        labels: []const f32,
        objective: Objective,
        scale_pos_weight: f32,
    ) !void {
        const n_leaves = @as(usize, 1) << @intCast(depth);
        const g = try b.gpa.alloc(f64, n_leaves);
        defer b.gpa.free(g);
        const h = try b.gpa.alloc(f64, n_leaves);
        defer b.gpa.free(h);
        const delta = b.values[0..n_leaves];
        @memset(delta, 0);

        var it: u32 = 0;
        while (it < b.s.leaf_iterations) : (it += 1) {
            @memset(g, 0);
            @memset(h, 0);
            if (it == 0) {
                for (b.leaf, grads) |l, gp| {
                    g[l] += gp.g;
                    h[l] += gp.h;
                }
            } else {
                for (b.leaf, raw, labels) |l, r0, y| {
                    const d = derivatives(objective, @as(f64, r0) + delta[l], y, scale_pos_weight);
                    g[l] += d.g;
                    h[l] += d.h;
                }
            }
            for (delta, g, h) |*dl, gs, hs| {
                if (hs + b.s.lambda > 0) dl.* += -gs / (hs + b.s.lambda);
            }
        }
        for (delta) |*dl| dl.* *= b.s.learning_rate;
    }

    /// A complete binary tree: every node at depth d tests split d. A numeric split sends
    /// `bin <= threshold` left (leaf bit 0). One-hot and CTR splits become categorical set nodes;
    /// a CTR's set is the levels whose *final* bucket (counts over all training rows) is <= the
    /// threshold, which is what CatBoost predicts with. A node whose stored set holds the bit-1
    /// side is `inverted`: its left child is the bit-1 subtree.
    fn toTree(b: *Builder, sp: []const Split) !tree.Tree {
        const d = sp.len;
        const n_nodes = (@as(usize, 1) << @intCast(d + 1)) - 1;
        const nodes = try b.gpa.alloc(tree.Node, n_nodes);
        errdefer b.gpa.free(nodes);
        var ids: std.ArrayList(data.BinIdx) = .empty;
        defer ids.deinit(b.gpa);

        var proto: [max_depth]tree.Node = undefined;
        var inverted = [_]bool{false} ** max_depth;
        for (sp, 0..) |s, level| {
            const c = b.cands[s.cand];
            switch (c.kind) {
                .numeric => proto[level] = .{ .feature = c.feature, .threshold = @intCast(s.threshold), .missing_left = true, .is_leaf = false },
                .onehot => {
                    // `bin == level` is leaf bit 1, and the set {level} goes left.
                    const ofs = ids.items.len;
                    try ids.append(b.gpa, @intCast(s.threshold));
                    proto[level] = .{ .feature = c.feature, .is_cat = true, .cat_ofs = @intCast(ofs), .n_cat = 1, .missing_left = s.threshold == 0, .is_leaf = false };
                    inverted[level] = true;
                },
                .ctr => {
                    // Bit 0 is `final bucket <= threshold`; store whichever side is smaller.
                    var low: usize = 0;
                    for (c.final[1..]) |bk| low += @intFromBool(bk <= s.threshold);
                    const high = c.final.len - 1 - low;
                    const keep_low = low <= high;
                    if (@min(low, high) > std.math.maxInt(u8)) return error.CtrSetTooLarge;
                    const ofs = ids.items.len;
                    for (c.final[1..], 1..) |bk, lvl| {
                        if ((bk <= s.threshold) == keep_low) try ids.append(b.gpa, @intCast(lvl));
                    }
                    const missing_low = c.final[0] <= s.threshold;
                    proto[level] = .{
                        .feature = c.feature,
                        .is_cat = true,
                        .cat_ofs = @intCast(ofs),
                        .n_cat = @intCast(ids.items.len - ofs),
                        .missing_left = missing_low == keep_low,
                        .is_leaf = false,
                    };
                    inverted[level] = !keep_low;
                },
            }
        }

        // Breadth-first: node i's children are 2i+1 and 2i+2; depth-d nodes start at 2^d - 1.
        for (nodes, 0..) |*nd, i| {
            const level = std.math.log2_int(usize, i + 1);
            if (level < d) {
                nd.* = proto[level];
                nd.left = @intCast(2 * i + 1);
                nd.right = @intCast(2 * i + 2);
            } else {
                // Position among the leaves, read root to leaf, has bit `level - 1 - k` set when
                // the k-th split went right; the leaf index has split k at bit k, flipped where the
                // node is inverted.
                const pos = i + 1 - (@as(usize, 1) << @intCast(d));
                var idx: usize = 0;
                for (0..d) |k| {
                    const went_right = (pos >> @intCast(d - 1 - k)) & 1 != 0;
                    if (went_right != inverted[k]) idx |= @as(usize, 1) << @intCast(k);
                }
                nd.* = .{ .weight = @floatCast(b.values[idx]), .is_leaf = true };
            }
        }
        return .{ .nodes = nodes, .cat_ids = try ids.toOwnedSlice(b.gpa) };
    }
};

/// Candidate index and its threshold (for one-hot, the level).
const Split = struct { cand: u32, threshold: u32 };

pub fn freeCands(gpa: std.mem.Allocator, cands: []Cand) void {
    for (cands) |c| {
        if (c.col.len != 0) gpa.free(c.col);
        if (c.final.len != 0) gpa.free(c.final);
        if (c.present.len != 0) gpa.free(c.present);
    }
    gpa.free(cands);
}

/// Numeric columns in order, then one-hot categoricals, then four CTRs per wider categorical. A
/// categorical with one level in training offers nothing and is skipped. Without `ctr` every
/// column is numeric (ordinal ids for categoricals). Caller owns the result (`freeCands`).
pub fn candidates(gpa: std.mem.Allocator, ds: *const Dataset, s: Settings) ![]Cand {
    var list: std.ArrayList(Cand) = .empty;
    errdefer {
        for (list.items) |c| {
            if (c.col.len != 0) gpa.free(c.col);
            if (c.final.len != 0) gpa.free(c.final);
            if (c.present.len != 0) gpa.free(c.present);
        }
        list.deinit(gpa);
    }
    for (0..ds.n_features) |f| {
        if (s.ctr and ds.kinds[f] == .categorical) continue;
        try list.append(gpa, .{ .kind = .numeric, .feature = @intCast(f), .nb = ds.n_bins[f] });
    }
    if (!s.ctr) return list.toOwnedSlice(gpa);
    if (ds.labels.len != ds.n_rows) return error.CtrNeedsLabels;

    // Levels present in training, per categorical: one-hot or CTR.
    var wide: std.ArrayList(u32) = .empty;
    defer wide.deinit(gpa);
    for (0..ds.n_features) |f| {
        if (ds.kinds[f] != .categorical) continue;
        const present = try gpa.alloc(bool, ds.n_bins[f]);
        var kept = false;
        defer if (!kept) gpa.free(present);
        @memset(present, false);
        for (0..ds.n_rows) |r| present[binAt(ds, f, r)] = true;
        var levels: u32 = 0;
        for (present) |p| levels += @intFromBool(p);
        if (levels <= 1) continue;
        if (levels <= s.one_hot_max_size) {
            try list.append(gpa, .{ .kind = .onehot, .feature = @intCast(f), .nb = ds.n_bins[f], .present = present });
            kept = true;
        } else try wide.append(gpa, @intCast(f));
    }
    for (wide.items) |f| {
        for (0..n_ctr_types) |t| {
            var c: Cand = .{ .kind = .ctr, .feature = f, .ctr_type = @intCast(t), .nb = ctr_buckets };
            c.col = try gpa.alloc(u8, ds.n_rows);
            errdefer gpa.free(c.col);
            c.final = try gpa.alloc(u8, ds.n_bins[f]);
            errdefer gpa.free(c.final);
            c.uniq = try ctrColumn(gpa, ds, f, @intCast(t), c.col, c.final);
            try list.append(gpa, c);
        }
    }
    return list.toOwnedSlice(gpa);
}

/// `trunc(value * 15)` in f32, as CatBoost buckets a CTR value in [0, 1].
inline fn bucketOf(value: f32) u8 {
    return @intFromFloat(@min(@trunc(value * 15), 15));
}

/// One CTR's training buckets (`col`, online: each row sees only the rows before it, in row
/// order, CatBoost's `has_time`) and final buckets per level (`final`, counts over every row).
/// Borders: `(positives + prior) / (count + 1)`. Counter: `count / (largest count + 1)` over all
/// rows, online or not (CatBoost's `SkipTest`). Returns the number of levels present.
pub fn ctrColumn(gpa: std.mem.Allocator, ds: *const Dataset, f: u32, t: u8, col: []u8, final: []u8) !u32 {
    const nl = ds.n_bins[f];
    const total = try gpa.alloc(u32, nl);
    defer gpa.free(total);
    const good = try gpa.alloc(u32, nl);
    defer gpa.free(good);
    @memset(total, 0);
    @memset(good, 0);
    if (t < ctr_priors.len) {
        const prior = ctr_priors[t];
        for (col, 0..) |*o, r| {
            const k = binAt(ds, f, r);
            o.* = bucketOf((@as(f32, @floatFromInt(good[k])) + prior) / (@as(f32, @floatFromInt(total[k])) + 1));
            total[k] += 1;
            good[k] += @intFromBool(ds.labels[r] > 0.5);
        }
        for (final, total, good) |*o, n, g| o.* = bucketOf((@as(f32, @floatFromInt(g)) + prior) / (@as(f32, @floatFromInt(n)) + 1));
    } else {
        for (0..ds.n_rows) |r| total[binAt(ds, f, r)] += 1;
        var largest: u32 = 0;
        for (total) |n| largest = @max(largest, n);
        const den: f32 = @floatFromInt(largest + 1);
        for (final, total) |*o, n| o.* = bucketOf(@as(f32, @floatFromInt(n)) / den);
        for (col, 0..) |*o, r| o.* = final[binAt(ds, f, r)];
    }
    var uniq: u32 = 0;
    for (total) |n| uniq += @intFromBool(n != 0);
    return uniq;
}

/// A random stream key from four integers (splitmix64 rounds), so every stream is fixed by what it
/// is for, not by which worker draws it.
fn mix(seed: u64, a: u64, c: u64, d: u64) u64 {
    var x = seed;
    for ([3]u64{ a, c, d }) |v| {
        x +%= 0x9E3779B97F4A7C15 +% v;
        var z = x;
        z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
        z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
        x = z ^ (z >> 31);
    }
    return x;
}

const WeightCtx = struct {
    b: *Builder,
    grads: []const hist.GradPair,
    lambda: f64,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *WeightCtx = @ptrCast(@alignCast(ctx));
        for (begin..end) |blk| self.block(blk);
    }

    fn block(self: *WeightCtx, blk: usize) void {
        const b = self.b;
        const lo = blk * mvs_block;
        const hi = @min(lo + mvs_block, b.ds.n_rows);
        var prng = std.Random.DefaultPrng.init(mix(b.s.seed, b.iteration, blk, 0xB007));
        const r = prng.random();
        const w = b.weights[lo..hi];
        switch (b.s.bootstrap) {
            .none => @memset(w, 1),
            .bernoulli => for (w) |*x| {
                x.* = if (r.float(f64) < b.s.subsample) 1 else 0;
            },
            .bayesian => for (w) |*x| {
                // u in (0, 1]: -ln u is finite and >= 0.
                const u = 1.0 - r.float(f64);
                x.* = @floatCast(std.math.pow(f64, -@log(u), b.s.bagging_temperature));
            },
            .mvs => {
                if (b.s.subsample >= 1) return @memset(w, 1);
                const cand = b.mvs_scratch[lo..hi];
                for (cand, self.grads[lo..hi]) |*c, gp| c.* = @sqrt(@as(f64, gp.g) * gp.g + self.lambda);
                const mu = mvsThreshold(cand, b.s.subsample * @as(f64, @floatFromInt(hi - lo)));
                for (w, self.grads[lo..hi]) |*x, gp| {
                    const c = @sqrt(@as(f64, gp.g) * gp.g + self.lambda);
                    const p = if (c > mu) 1.0 else c / mu;
                    x.* = if (p > std.math.floatEps(f64) and r.float(f64) < p) @floatCast(1 / p) else 0;
                }
            },
        }
    }
};

/// The threshold mu with `sum min(1, c / mu) = sample`: rows above mu are kept for sure, the rest
/// with probability c / mu. Sorts `cand` in place.
pub fn mvsThreshold(cand: []f64, sample: f64) f64 {
    std.sort.pdq(f64, cand, {}, std.sort.asc(f64));
    var small: f64 = 0;
    for (cand) |c| small += c;
    // k rows taken for sure, from the top; the rest share what is left of the sample.
    var k: usize = 0;
    while (k < cand.len) : (k += 1) {
        const left = sample - @as(f64, @floatFromInt(k));
        if (left <= 0) break;
        const mu = small / left;
        if (cand[cand.len - 1 - k] <= mu) return mu;
        small -= cand[cand.len - 1 - k];
    }
    // Every row is above any threshold that fits: keep them all.
    return 0;
}

inline fn binAt(ds: *const Dataset, f: usize, r: usize) data.BinIdx {
    return if (ds.isWide(f)) ds.columnWide(f)[r] else ds.columnNarrow(f)[r];
}

/// Per-feature histograms of the current level: each task owns one feature's slice, rows in order,
/// so the sums are the same at any thread count.
const HistCtx = struct {
    b: *Builder,
    grads: []const hist.GradPair,
    n_leaves: usize,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *HistCtx = @ptrCast(@alignCast(ctx));
        const b = self.b;
        for (begin..end) |f| {
            const cand = b.cands[f];
            const nb: usize = cand.nb;
            const slice = b.cells[b.off[f]..][0 .. self.n_leaves * nb];
            @memset(slice, .{});
            const w = b.weights;
            for (b.leaf, self.grads, 0..) |l, gp, r| {
                const s: f64 = if (w.len != 0) w[r] else 1;
                if (s == 0) continue;
                const bin: usize = if (cand.kind == .ctr) cand.col[r] else binAt(b.ds, cand.feature, r);
                const c = &slice[@as(usize, l) * nb + bin];
                c.g += s * gp.g;
                c.n += s;
                c.h += s * gp.h;
            }
        }
    }
};

/// Each feature's best threshold at this level, scored over all leaves.
const ScoreCtx = struct {
    b: *Builder,
    n_leaves: usize,
    level: u32,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *ScoreCtx = @ptrCast(@alignCast(ctx));
        for (begin..end) |f| self.b.best[f] = self.feature(f);
    }

    /// One leaf side's contribution to the level's score.
    inline fn side(b: *const Builder, c: Cell, num: *f64, den: *f64) void {
        switch (b.s.score) {
            .cosine, .l2, .auto => {
                const v = if (c.n > 0) c.g / (c.n + b.s.lambda) else 0;
                num.* += v * c.g;
                den.* += v * v * c.n;
            },
            .gain => num.* += c.g * c.g / (c.h + b.s.lambda),
        }
    }

    inline fn total(b: *const Builder, num: f64, den: f64) f64 {
        return switch (b.s.score) {
            .cosine, .auto => num / @sqrt(den + 1e-100),
            .l2, .gain => num,
        };
    }

    fn feature(self: *ScoreCtx, f: usize) Builder.Best {
        const b = self.b;
        const nb: usize = b.cands[f].nb;
        var best: Builder.Best = .{ .score = -std.math.inf(f64), .threshold = 0 };
        if (nb < 2) return best;
        if (b.s.ordered) return self.fromAccumulators(f);
        if (b.cands[f].kind == .onehot) return self.oneHot(f);
        const slice = b.cells[b.off[f]..][0 .. self.n_leaves * nb];
        // Each leaf's bins become running sums in place (rebuilt every level): `slice[l*nb + k]`
        // is then the left side of threshold k and the last cell the leaf's total.
        for (0..self.n_leaves) |l| {
            const cells = slice[l * nb ..][0..nb];
            for (1..nb) |i| cells[i] = cells[i - 1].add(cells[i]);
        }
        // A CTR bucket has no missing value: every threshold is a cut.
        const first: usize = if (b.has_missing[f] or b.cands[f].kind == .ctr) 0 else 1;
        // With noise, thresholds compete on noisy scores and the winner keeps its noise-free one;
        // the comparison across features draws fresh noise (CatBoost's `SetBestScore`).
        var prng = std.Random.DefaultPrng.init(mix(b.s.seed, b.iteration, self.level, f));
        var best_noisy = -std.math.inf(f64);
        // Thresholds k = first .. nb-2: left holds bins <= k.
        var k = first;
        while (k + 1 < nb) : (k += 1) {
            var num: f64 = 0;
            var den: f64 = 0;
            for (0..self.n_leaves) |l| {
                const left = slice[l * nb + k];
                // CatBoost forms the right side by subtraction from the total; so does this.
                side(b, left, &num, &den);
                side(b, slice[l * nb + nb - 1].sub(left), &num, &den);
            }
            const s = total(b, num, den);
            const noisy = if (b.sigma > 0) s + b.sigma * prng.random().floatNorm(f64) else s;
            if (noisy > best_noisy) {
                best_noisy = noisy;
                best = .{ .score = s, .threshold = @intCast(k) };
            }
        }
        return best;
    }

    /// Ordered boosting: the per-threshold sums every prefix added (`OrderedCtx`). For one-hot
    /// candidates index v is `bin == v`; levels no row has are skipped.
    fn fromAccumulators(self: *ScoreCtx, f: usize) Builder.Best {
        const b = self.b;
        const c = b.cands[f];
        const nb: usize = c.nb;
        var best: Builder.Best = .{ .score = -std.math.inf(f64), .threshold = 0 };
        var prng = std.Random.DefaultPrng.init(mix(b.s.seed, b.iteration, self.level, f));
        var best_noisy = -std.math.inf(f64);
        const num = b.acc_num[b.acc_off[f]..][0..nb];
        const den = b.acc_den[b.acc_off[f]..][0..nb];
        const lo: usize = if (c.kind == .onehot or b.has_missing[f] or c.kind == .ctr) 0 else 1;
        const hi: usize = if (c.kind == .onehot) nb else nb - 1;
        for (lo..hi) |k| {
            if (c.kind == .onehot and !c.present[k]) continue;
            const s = total(b, num[k], den[k]);
            const noisy = if (b.sigma > 0) s + b.sigma * prng.random().floatNorm(f64) else s;
            if (noisy > best_noisy) {
                best_noisy = noisy;
                best = .{ .score = s, .threshold = @intCast(k) };
            }
        }
        return best;
    }

    /// `bin == v` against the rest, for each level v present in training, levels in id order.
    fn oneHot(self: *ScoreCtx, f: usize) Builder.Best {
        const b = self.b;
        const nb: usize = b.cands[f].nb;
        const slice = b.cells[b.off[f]..][0 .. self.n_leaves * nb];
        var best: Builder.Best = .{ .score = -std.math.inf(f64), .threshold = 0 };
        var prng = std.Random.DefaultPrng.init(mix(b.s.seed, b.iteration, self.level, f));
        var best_noisy = -std.math.inf(f64);
        for (0..nb) |v| {
            var present = false;
            for (0..self.n_leaves) |l| present = present or slice[l * nb + v].n > 0;
            if (!present) continue;
            var num: f64 = 0;
            var den: f64 = 0;
            for (0..self.n_leaves) |l| {
                var all: Cell = .{};
                for (slice[l * nb ..][0..nb]) |c| all = all.add(c);
                const eq = slice[l * nb + v];
                side(b, all.sub(eq), &num, &den);
                side(b, eq, &num, &den);
            }
            const s = total(b, num, den);
            const noisy = if (b.sigma > 0) s + b.sigma * prng.random().floatNorm(f64) else s;
            if (noisy > best_noisy) {
                best_noisy = noisy;
                best = .{ .score = s, .threshold = @intCast(v) };
            }
        }
        return best;
    }
};

/// One prefix's contribution to every candidate's ordered score: histograms of its body (prefix
/// model's derivative sums and counts) and its tail (sums and bootstrap-weighted counts), then for
/// each threshold and leaf the body's leaf estimates scored against the tail. One task per
/// candidate; prefixes run one after another, so the sums keep a fixed order.
const OrderedCtx = struct {
    b: *Builder,
    bt: usize,
    n_leaves: usize,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *OrderedCtx = @ptrCast(@alignCast(ctx));
        for (begin..end) |f| self.cand(f);
    }

    fn cand(self: *OrderedCtx, f: usize) void {
        const b = self.b;
        const c = b.cands[f];
        const nb: usize = c.nb;
        const bt = b.bts[self.bt];
        const d = b.deriv[b.ap_off[self.bt]..][0..bt.tail];
        const body = b.cells[b.off[f]..][0 .. self.n_leaves * nb];
        const tail = b.cells_tail[b.off[f]..][0 .. self.n_leaves * nb];
        @memset(body, .{});
        @memset(tail, .{});
        for (0..bt.tail) |r| {
            const bin: usize = if (c.kind == .ctr) c.col[r] else binAt(b.ds, c.feature, r);
            const i = @as(usize, b.leaf[r]) * nb + bin;
            if (r < bt.body) {
                body[i].g += d[r];
                body[i].n += 1;
            } else {
                const s: f64 = if (b.weights.len != 0) b.weights[r] else 1;
                if (s == 0) continue;
                tail[i].g += s * d[r];
                tail[i].n += s;
            }
        }
        const num = b.acc_num[b.acc_off[f]..][0..nb];
        const den = b.acc_den[b.acc_off[f]..][0..nb];
        const lambda = b.s.lambda;
        const est = struct {
            fn v(cell: Cell, lam: f64) f64 {
                return if (cell.n > 0) cell.g / (cell.n + lam) else 0;
            }
        }.v;
        if (c.kind == .onehot) {
            for (0..nb) |v| {
                if (!c.present[v]) continue;
                for (0..self.n_leaves) |l| {
                    var ball: Cell = .{};
                    var tall: Cell = .{};
                    for (body[l * nb ..][0..nb], tail[l * nb ..][0..nb]) |x, y| {
                        ball = ball.add(x);
                        tall = tall.add(y);
                    }
                    const be = body[l * nb + v];
                    const te = tail[l * nb + v];
                    const ae = est(be, lambda);
                    const ar = est(ball.sub(be), lambda);
                    num[v] += ae * te.g + ar * (tall.g - te.g);
                    den[v] += ae * ae * te.n + ar * ar * (tall.n - te.n);
                }
            }
            return;
        }
        for (0..self.n_leaves) |l| {
            const bc = body[l * nb ..][0..nb];
            const tc = tail[l * nb ..][0..nb];
            for (1..nb) |i| {
                bc[i] = bc[i - 1].add(bc[i]);
                tc[i] = tc[i - 1].add(tc[i]);
            }
            for (0..nb - 1) |k| {
                const al = est(bc[k], lambda);
                const ar = est(bc[nb - 1].sub(bc[k]), lambda);
                const tr = tc[nb - 1].sub(tc[k]);
                num[k] += al * tc[k].g + ar * tr.g;
                den[k] += al * al * tc[k].n + ar * ar * tr.n;
            }
        }
    }
};

/// Ordered boosting's buffers; owned by the builder, gathered here so init and deinit agree.
const Ordered = struct {
    bts: []BodyTail = &.{},
    approx: []f64 = &.{},
    deriv: []f64 = &.{},
    ap_off: []usize = &.{},
    cells_tail: []Cell = &.{},
    acc_num: []f64 = &.{},
    acc_den: []f64 = &.{},
    acc_off: []usize = &.{},
    tail_grads: []hist.GradPair = &.{},

    fn init(gpa: std.mem.Allocator, ds: *const Dataset, cands: []const Cand, n_cells: usize) !Ordered {
        var o: Ordered = .{};
        errdefer o.deinit(gpa);
        o.bts = try bodyTails(gpa, ds.n_rows);
        o.ap_off = try gpa.alloc(usize, o.bts.len + 1);
        o.ap_off[0] = 0;
        for (o.bts, 0..) |bt, k| o.ap_off[k + 1] = o.ap_off[k] + bt.tail;
        o.approx = try gpa.alloc(f64, o.ap_off[o.bts.len]);
        @memset(o.approx, 0);
        o.deriv = try gpa.alloc(f64, o.ap_off[o.bts.len]);
        o.cells_tail = try gpa.alloc(Cell, n_cells);
        o.acc_off = try gpa.alloc(usize, cands.len + 1);
        o.acc_off[0] = 0;
        for (cands, 0..) |c, i| o.acc_off[i + 1] = o.acc_off[i] + c.nb;
        o.acc_num = try gpa.alloc(f64, o.acc_off[cands.len]);
        o.acc_den = try gpa.alloc(f64, o.acc_off[cands.len]);
        o.tail_grads = try gpa.alloc(hist.GradPair, ds.n_rows);
        return o;
    }

    fn deinit(o: Ordered, gpa: std.mem.Allocator) void {
        if (o.bts.len != 0) gpa.free(o.bts);
        if (o.approx.len != 0) gpa.free(o.approx);
        if (o.deriv.len != 0) gpa.free(o.deriv);
        if (o.ap_off.len != 0) gpa.free(o.ap_off);
        if (o.cells_tail.len != 0) gpa.free(o.cells_tail);
        if (o.acc_num.len != 0) gpa.free(o.acc_num);
        if (o.acc_den.len != 0) gpa.free(o.acc_den);
        if (o.acc_off.len != 0) gpa.free(o.acc_off);
        if (o.tail_grads.len != 0) gpa.free(o.tail_grads);
    }
};

const ApplySplit = struct {
    b: *Builder,
    sp: Split,
    bit: u4,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *ApplySplit = @ptrCast(@alignCast(ctx));
        const ds = self.b.ds;
        const c = self.b.cands[self.sp.cand];
        for (begin..end) |r| {
            const one = switch (c.kind) {
                .numeric => binAt(ds, c.feature, r) > self.sp.threshold,
                .onehot => binAt(ds, c.feature, r) == self.sp.threshold,
                .ctr => c.col[r] > self.sp.threshold,
            };
            self.b.leaf[r] |= @as(u16, @intFromBool(one)) << self.bit;
        }
    }
};

const Deriv = struct { g: f64, h: f64 };

/// Exact derivatives for a re-evaluated Newton step (the first step reuses the booster's).
fn derivatives(objective: Objective, raw: f64, y: f32, scale_pos_weight: f32) Deriv {
    return switch (objective) {
        .logistic => blk: {
            const p = 1.0 / (1.0 + @exp(-raw));
            const w: f64 = if (y > 0.5) scale_pos_weight else 1.0;
            break :blk .{ .g = w * (p - y), .h = w * @max(p * (1 - p), 1e-6) };
        },
        .squared_error => .{ .g = raw - y, .h = 1 },
    };
}
