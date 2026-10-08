// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Symmetric (oblivious) trees, CatBoost's default: one split per depth, shared by every node at
//! that depth, so a depth-D tree is D yes/no questions and 2^D leaves. Re-implemented from
//! CatBoost 1.2.10's documented behaviour and source reading (docs/archive/catboost.md); no CatBoost code.
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
const prof = @import("prof.zig");
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
    /// Most columns a CTR may combine (CatBoost's `max_ctr_complexity`); 1 is single columns only.
    max_ctr_complexity: u32 = 4,
    /// Rows are in time order: one learning fold in file order, shared with the model. Off, with
    /// target statistics or ordered boosting, CatBoost's permutations apply (`Fold`).
    has_time: bool = true,
    /// Learning folds are `permutation_count - 1` (CatBoost's default 4, so 3).
    permutation_count: u32 = 4,
    /// Rows shuffled together as a block in folds after the first; 0 is CatBoost's
    /// `min(256, n / 1000 + 1)`.
    permutation_block: u32 = 0,
    /// Most bytes ordered scoring may keep in per-prefix histograms to subtract from at the next
    /// level; above it every level counts every row (same model, slower).
    ordered_bank_limit: usize = 1 << 30,
};

/// One ordering of the training rows and the state that follows it (CatBoost's `TFold`): its
/// online target statistics, and its own scores (plain: one per row; ordered: each prefix
/// model's). Learning folds choose tree structure, one drawn per tree, and each takes its own
/// Newton step after every tree; the averaging fold's leaf membership gives the model's leaves.
/// With `has_time` (or nothing to permute) there is one fold in file order, and its scores are
/// the model's own.
const Fold = struct {
    /// Position to row; empty is file order.
    order: []u32 = &.{},
    /// Online buckets per single-column CTR (`Cand.slot`), by row.
    ctr: [][]u8 = &.{},
    /// Plain boosting with its own scores: one per row. Empty: the model's scores.
    approx: []f64 = &.{},
    /// Ordered boosting: prefix models' scores and derivatives, by position (`Builder.ap_off`).
    papprox: []f64 = &.{},
    pderiv: []f64 = &.{},

    inline fn row(f: *const Fold, pos: usize) usize {
        return if (f.order.len == 0) pos else f.order[pos];
    }

    fn deinit(f: *Fold, gpa: std.mem.Allocator) void {
        if (f.order.len != 0) gpa.free(f.order);
        for (f.ctr) |c| gpa.free(c);
        if (f.ctr.len != 0) gpa.free(f.ctr);
        if (f.approx.len != 0) gpa.free(f.approx);
        if (f.papprox.len != 0) gpa.free(f.papprox);
        if (f.pderiv.len != 0) gpa.free(f.pderiv);
        f.* = .{};
    }
};

/// CatBoost's block shuffle (fold.cpp, `Shuffle`): positions cut into blocks of `block` rows, the
/// blocks permuted, rows keeping their order inside a block. Caller owns the result.
pub fn blockShuffle(gpa: std.mem.Allocator, base: []const u32, n: usize, block: usize, r: std.Random) ![]u32 {
    const n_blocks = (n + block - 1) / block;
    const blocks = try gpa.alloc(u32, n_blocks);
    defer gpa.free(blocks);
    for (blocks, 0..) |*x, i| x.* = @intCast(i);
    r.shuffle(u32, blocks);
    const out = try gpa.alloc(u32, n);
    var o: usize = 0;
    for (blocks) |bi| {
        const lo = @as(usize, bi) * block;
        for (lo..@min(lo + block, n)) |pos| {
            out[o] = if (base.len == 0) @intCast(pos) else base[pos];
            o += 1;
        }
    }
    return out;
}

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
    /// Combination CTR: its projection (canonical order), and its table from counts over every
    /// training row (sorted keys and their buckets; `unseen` for any other key). Empty for a
    /// single-column CTR, whose projection is its `feature` and whose table is `final`.
    parts: []tree.Part = &.{},
    keys: []u64 = &.{},
    kbuckets: []u8 = &.{},
    unseen: u8 = 0,
    /// The four CTRs of a combination share `parts`; the first owns it.
    owns_parts: bool = false,
    /// Single-column CTR: its index in every fold's `ctr`. Its `col` then views the current
    /// learning fold's column and is not owned.
    slot: u32 = 0,
    owns_col: bool = true,

    fn isCounter(c: Cand) bool {
        return c.ctr_type == ctr_priors.len;
    }

    /// A CTR's projection: its parts, or for a single column a one-part view in `buf`.
    fn projection(c: *const Cand, buf: *[1]tree.Part) []const tree.Part {
        if (c.parts.len != 0) return c.parts;
        buf[0] = .{ .kind = .cat, .feature = c.feature };
        return buf;
    }

    fn freeOwned(c: Cand, gpa: std.mem.Allocator) void {
        if (c.owns_col and c.col.len != 0) gpa.free(c.col);
        if (c.final.len != 0) gpa.free(c.final);
        if (c.present.len != 0) gpa.free(c.present);
        if (c.owns_parts) gpa.free(c.parts);
        if (c.keys.len != 0) gpa.free(c.keys);
        if (c.kbuckets.len != 0) gpa.free(c.kbuckets);
    }
};

/// A (Borders or Counter, projection) pair some split has chosen: `model_size_reg` stops applying
/// to every CTR of that type over that projection (CatBoost's `UsedCtrSplits`).
const Used = struct { counter: bool, parts: []tree.Part };

pub const Builder = struct {
    gpa: std.mem.Allocator,
    pool: *Pool,
    ds: *const Dataset,
    s: Settings,
    /// Each row's leaf index in the tree being grown, on the learning fold that chose it.
    leaf: []u16,
    /// Each row's leaf on the averaging fold: what the model's leaves are fitted on and what the
    /// caller applies `values` by. `leaf` itself when the two folds agree.
    leaf_model: []u16,
    leaf_buf: []u16,
    leaf_tmp: []u16,
    /// Learning folds (`folds[0]` doubles as the averaging fold when `avg_alias`), the averaging
    /// fold otherwise, and the fold the current tree searches.
    folds: []Fold,
    avg: Fold = .{},
    avg_alias: bool = true,
    fold_k: usize = 0,
    /// Derivatives on a learning fold with its own scores; empty when the fold is the model.
    fold_grads: []hist.GradPair = &.{},
    combo_buf: []u8 = &.{},
    /// Split candidates of the level being searched, CatBoost's order: numeric columns, then
    /// one-hot, then single-column CTRs, then this level's combinations. Ties go to the earlier
    /// one. A view of `all`, whose first `n_static` are fixed and whose rest each level rebuilds.
    cands: []Cand,
    all: []Cand,
    n_static: usize,
    /// Categoricals with target statistics, ascending: what a combination may add.
    wide: []u32,
    used: std.ArrayList(Used) = .empty,
    /// The current tree's chosen combinations, already in stored form.
    tree_combos: std.ArrayList(tree.Combo) = .empty,
    /// Histograms for every candidate over the deepest scored level: candidate c's leaf l bin b is
    /// `cells[off[c] + l * nb_c + b]`.
    cells: []Cell,
    off: []usize,
    /// Whether candidate c's column has a missing row: the missing-vs-present cut (threshold 0) is
    /// offered only then, as CatBoost adds its `nan_mode = Min` border only for columns with NaN.
    has_missing: []bool,
    /// Per-candidate best split of the level being searched.
    best: []Best,
    /// Largest `uniq` among this level's CTR candidates, and among the static ones.
    max_uniq: u32 = 0,
    static_max_uniq: u32 = 0,
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
    ap_off: []usize = &.{},
    cells_tail: []Cell = &.{},
    acc_num: []f64 = &.{},
    acc_den: []f64 = &.{},
    acc_off: []usize = &.{},
    tail_grads: []hist.GradPair = &.{},
    /// Plain scoring below the root (CatBoost's smallest-side trick): the rows on the smaller side
    /// of the newest split, ascending, with their weighted derivatives; whether candidate c's cells
    /// hold its raw histogram of the level above; and per-worker room for the running sums
    /// `ScoreCtx` scans, so the raw histograms survive to be subtracted from.
    sel: Selected = .{},
    /// One derivative pair per row position, computed in parallel and then summed in order.
    dscratch: []Deriv = &.{},
    cached: []bool = &.{},
    /// Ordered scoring's per-prefix body and tail histograms of the static candidates, kept from
    /// the level above for the same subtraction (`obody[k * ostride + off[c] ..]`); empty when
    /// they would pass `Settings.ordered_bank_limit`, and every level then counts every row.
    obody: []Cell = &.{},
    otail: []Cell = &.{},
    ostride: usize = 0,
    scratch: []Cell = &.{},
    max_slice: usize = 0,

    const Best = struct { score: f64, threshold: u32 };

    pub fn init(gpa: std.mem.Allocator, pool: *Pool, ds: *const Dataset, s: Settings) !Builder {
        if (s.depth == 0 or s.depth > max_depth) return error.BadSymmetricDepth;
        const static = try candidates(gpa, ds, s);
        var static_owned = true;
        defer if (static_owned) freeCands(gpa, static) else gpa.free(static);
        var n_wide: usize = 0;
        for (static) |c| n_wide += @intFromBool(c.kind == .ctr and c.ctr_type == 0);
        const wide = try gpa.alloc(u32, n_wide);
        errdefer gpa.free(wide);
        n_wide = 0;
        for (static) |c| if (c.kind == .ctr and c.ctr_type == 0) {
            wide[n_wide] = c.feature;
            n_wide += 1;
        };
        // Combinations per level, at most: one base per split above it plus the split-bits base,
        // times each categorical, times the four CTR types.
        const max_combos: usize = if (s.ctr and s.max_ctr_complexity > 1) s.depth * wide.len * n_ctr_types else 0;
        const all = try gpa.alloc(Cand, static.len + max_combos);
        errdefer gpa.free(all);
        @memcpy(all[0..static.len], static);
        static_owned = false; // `all` owns the static candidates' arrays now
        errdefer freeCands(gpa, all[0..static.len]);
        const cands = all[0..static.len];
        var max_uniq: u32 = 0;
        for (cands) |c| if (c.kind == .ctr) {
            max_uniq = @max(max_uniq, c.uniq);
        };
        const leaf = try gpa.alloc(u16, ds.n_rows);
        errdefer gpa.free(leaf);
        const off = try gpa.alloc(usize, all.len + 1);
        errdefer gpa.free(off);
        const scored_leaves: usize = @as(usize, 1) << @intCast(s.depth - 1);
        off[0] = 0;
        for (cands, 0..) |c, i| off[i + 1] = off[i] + scored_leaves * c.nb;
        const cells = try gpa.alloc(Cell, off[cands.len] + max_combos * scored_leaves * ctr_buckets);
        errdefer gpa.free(cells);
        const has_missing = try gpa.alloc(bool, all.len);
        errdefer gpa.free(has_missing);
        @memset(has_missing, false);
        for (has_missing[0..cands.len], cands) |*m, c| {
            m.* = false;
            if (c.kind != .numeric) continue;
            for (0..ds.n_rows) |r| if (binAt(ds, c.feature, r) == 0) {
                m.* = true;
                break;
            };
        }
        const best = try gpa.alloc(Best, all.len);
        errdefer gpa.free(best);
        const values = try gpa.alloc(f64, @as(usize, 1) << @intCast(s.depth));
        errdefer gpa.free(values);
        const weights = try gpa.alloc(f32, if (s.bootstrap == .none) 0 else ds.n_rows);
        errdefer gpa.free(weights);
        const mvs_scratch = try gpa.alloc(f64, if (s.bootstrap == .mvs) ds.n_rows else 0);
        errdefer gpa.free(mvs_scratch);
        var ord: Ordered = .{};
        if (s.ordered) ord = try Ordered.init(gpa, ds, cands, all.len, cells.len, max_combos * ctr_buckets);
        errdefer ord.deinit(gpa);

        // Single-column CTRs get a slot in every fold; their columns move into the folds.
        var n_slots: u32 = 0;
        for (cands) |*c| if (c.kind == .ctr) {
            c.slot = n_slots;
            n_slots += 1;
            if (c.owns_col and c.col.len != 0) gpa.free(c.col);
            c.col = &.{};
            c.owns_col = false;
        };
        const folds_made = try makeFolds(gpa, ds, s, n_slots, cands, ord.ap_off);
        errdefer {
            for (folds_made.learning) |*f| f.deinit(gpa);
            gpa.free(folds_made.learning);
            var a = folds_made.avg;
            if (!folds_made.avg_alias) a.deinit(gpa);
        }
        const leaf_buf = try gpa.alloc(u16, ds.n_rows);
        errdefer gpa.free(leaf_buf);
        const leaf_tmp = try gpa.alloc(u16, ds.n_rows);
        errdefer gpa.free(leaf_tmp);
        const fold_grads = try gpa.alloc(hist.GradPair, if (folds_made.learning[0].approx.len != 0) ds.n_rows else 0);
        errdefer gpa.free(fold_grads);
        const combo_buf = try gpa.alloc(u8, if (max_combos != 0) ds.n_rows else 0);
        errdefer gpa.free(combo_buf);
        var max_slice: usize = scored_leaves * ctr_buckets;
        for (cands) |c| max_slice = @max(max_slice, scored_leaves * c.nb);
        const cached = try gpa.alloc(bool, all.len);
        errdefer gpa.free(cached);
        @memset(cached, false);
        // Ordered scoring keeps a body and a tail histogram per worker.
        const scratch = try gpa.alloc(Cell, pool.workerCount() * max_slice * @as(usize, if (s.ordered) 2 else 1));
        errdefer gpa.free(scratch);
        var sel = try Selected.init(gpa, ds.n_rows);
        errdefer sel.deinit(gpa);
        const ostride = off[cands.len];
        const bank_cells = if (s.ordered) ord.bts.len * ostride else 0;
        const keep_banks = bank_cells != 0 and 2 * bank_cells * @sizeOf(Cell) <= s.ordered_bank_limit;
        const obody = try gpa.alloc(Cell, if (keep_banks) bank_cells else 0);
        errdefer gpa.free(obody);
        const otail = try gpa.alloc(Cell, if (keep_banks) bank_cells else 0);
        errdefer gpa.free(otail);
        const dscratch = try gpa.alloc(Deriv, ds.n_rows);
        errdefer gpa.free(dscratch);
        return .{
            .gpa = gpa,
            .pool = pool,
            .ds = ds,
            .s = s,
            .leaf = leaf,
            .cands = cands,
            .all = all,
            .n_static = static.len,
            .wide = wide,
            .max_uniq = max_uniq,
            .static_max_uniq = max_uniq,
            .cells = cells,
            .off = off,
            .has_missing = has_missing,
            .best = best,
            .values = values,
            .weights = weights,
            .mvs_scratch = mvs_scratch,
            .leaf_model = leaf,
            .leaf_buf = leaf_buf,
            .leaf_tmp = leaf_tmp,
            .folds = folds_made.learning,
            .avg = folds_made.avg,
            .avg_alias = folds_made.avg_alias,
            .fold_grads = fold_grads,
            .combo_buf = combo_buf,
            .bts = ord.bts,
            .ap_off = ord.ap_off,
            .cells_tail = ord.cells_tail,
            .acc_num = ord.acc_num,
            .acc_den = ord.acc_den,
            .acc_off = ord.acc_off,
            .tail_grads = ord.tail_grads,
            .sel = sel,
            .dscratch = dscratch,
            .cached = cached,
            .obody = obody,
            .otail = otail,
            .ostride = ostride,
            .scratch = scratch,
            .max_slice = max_slice,
        };
    }

    pub fn deinit(b: *Builder) void {
        for (b.cands) |c| c.freeOwned(b.gpa);
        b.gpa.free(b.all);
        b.gpa.free(b.wide);
        for (b.used.items) |u| b.gpa.free(u.parts);
        b.used.deinit(b.gpa);
        for (b.tree_combos.items) |*c| c.deinit(b.gpa);
        b.tree_combos.deinit(b.gpa);
        b.gpa.free(b.leaf);
        b.gpa.free(b.cells);
        b.gpa.free(b.off);
        b.gpa.free(b.has_missing);
        b.gpa.free(b.best);
        b.gpa.free(b.values);
        b.gpa.free(b.weights);
        b.gpa.free(b.mvs_scratch);
        if (b.bts.len != 0) {
            const ord: Ordered = .{ .bts = b.bts, .ap_off = b.ap_off, .cells_tail = b.cells_tail, .acc_num = b.acc_num, .acc_den = b.acc_den, .acc_off = b.acc_off, .tail_grads = b.tail_grads };
            ord.deinit(b.gpa);
        }
        for (b.folds) |*f| f.deinit(b.gpa);
        b.gpa.free(b.folds);
        if (!b.avg_alias) b.avg.deinit(b.gpa);
        b.gpa.free(b.leaf_buf);
        b.gpa.free(b.leaf_tmp);
        b.gpa.free(b.fold_grads);
        b.gpa.free(b.combo_buf);
        b.sel.deinit(b.gpa);
        b.gpa.free(b.dscratch);
        b.gpa.free(b.cached);
        b.gpa.free(b.obody);
        b.gpa.free(b.otail);
        b.gpa.free(b.scratch);
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
        const t_setup = prof.start();
        @memset(b.leaf, 0);
        var splits: [max_depth]Split = undefined;
        var depth: u32 = 0;

        // Every fold's own scores start where the booster's do.
        if (b.iteration == 0) for (b.folds) |*fd| {
            if (fd.approx.len != 0) for (fd.approx, raw) |*a, r0| {
                a.* = r0;
            };
            if (fd.papprox.len != 0) for (b.bts, 0..) |bt, k| {
                for (fd.papprox[b.ap_off[k]..][0..bt.tail], 0..) |*a, pos| a.* = raw[fd.row(pos)];
            };
        };
        // The learning fold this tree searches on (CatBoost: `Folds[rand % count]`), and its
        // columns for the single-column CTRs.
        b.fold_k = if (b.folds.len == 1) 0 else @intCast(mix(b.s.seed, b.iteration, 0xF01D, 0) % b.folds.len);
        const fold = &b.folds[b.fold_k];
        for (b.all[0..b.n_static]) |*c| if (c.kind == .ctr) {
            c.col = fold.ctr[c.slot];
        };
        // The derivatives a split is judged by: the fold's own (each prefix model's, under ordered;
        // bootstrap weights and noise follow the tail rows'), or the model's when they coincide.
        var search_grads = grads;
        if (b.s.ordered) {
            b.prefixDerivatives(fold, labels, objective, scale_pos_weight);
        } else if (fold.approx.len != 0) {
            var pctx = PosCtx{ .b = b, .fold = fold, .labels = labels, .objective = objective, .spw = scale_pos_weight, .mode = .fold_grads };
            b.pool.parallelFor(ds.n_rows, &pctx, PosCtx.run, pos_chunk);
            search_grads = b.fold_grads;
        }
        const sample_grads = if (b.s.ordered) b.tail_grads else search_grads;
        if (b.s.bootstrap != .none) b.sampleWeights(sample_grads);
        b.sigma = if (b.s.random_strength > 0) b.noiseScale(sample_grads) else 0;
        prof.stop(.sym_setup, t_setup);

        defer b.dropCombos();
        @memset(b.cached, false);
        while (depth < b.s.depth) {
            const n_leaves = @as(usize, 1) << @intCast(depth);
            const t_combo = prof.start();
            try b.levelCombos(splits[0..depth], fold.order);
            prof.stop(.sym_combo, t_combo);
            const t_hist = prof.start();
            if (b.s.ordered) {
                @memset(b.acc_num, 0);
                @memset(b.acc_den, 0);
                const side: ?bool = if (depth == 0 or b.obody.len == 0) null else b.selectPositions(fold, depth - 1);
                for (b.bts, 0..) |_, k| {
                    var octx = OrderedCtx{ .b = b, .fold = fold, .bt = k, .n_leaves = n_leaves, .side = side };
                    b.pool.parallelFor(b.cands.len, &octx, OrderedCtx.run, 1);
                }
                // Every prefix's banks now hold this level for the static candidates.
                if (b.obody.len != 0) @memset(b.cached[0..b.n_static], true);
            } else {
                const side: ?bool = if (depth == 0) null else b.selectSide(search_grads, depth - 1);
                var hctx = HistCtx{ .b = b, .grads = search_grads, .n_leaves = n_leaves, .side = side };
                b.pool.parallelFor(b.cands.len, &hctx, HistCtx.run, 1);
            }
            prof.stop(.sym_hist, t_hist);
            const t_score = prof.start();
            var sctx = ScoreCtx{ .b = b, .n_leaves = n_leaves, .level = depth };
            b.pool.parallelFor(b.cands.len, &sctx, ScoreCtx.run, 1);

            // Strict `>` in candidate order: ties go to the earlier candidate, then (within one,
            // in `ScoreCtx`) to the lower threshold, as CatBoost's scan does. With noise, each
            // candidate's best (noise-free) score gets a fresh draw here, in order. A CTR not yet
            // used by any tree is scaled down by `model_size_reg` (CatBoost's `GetCatFeatureWeight`).
            var sel = std.Random.DefaultPrng.init(mix(b.s.seed, b.iteration, depth, std.math.maxInt(u32)));
            var win: ?usize = null;
            var win_score: f64 = -std.math.inf(f64);
            for (b.best[0..b.cands.len], b.cands, 0..) |c, cand, f| {
                if (!std.math.isFinite(c.score)) continue;
                var s = if (b.sigma > 0) c.score + b.sigma * sel.random().floatNorm(f64) else c.score;
                if (cand.kind == .ctr and b.s.model_size_reg > 0 and !b.isUsed(&cand)) {
                    const ratio = @as(f64, @floatFromInt(cand.uniq)) / @as(f64, @floatFromInt(b.max_uniq));
                    s *= std.math.pow(f64, 1 + ratio, -b.s.model_size_reg);
                }
                if (win == null or s > win_score) {
                    win = f;
                    win_score = s;
                }
            }
            prof.stop(.sym_score, t_score);
            const f = win orelse break;
            var sp: Split = .{ .cand = @intCast(f), .threshold = @intCast(b.best[f].threshold) };
            // Marked on choice, as CatBoost does: later levels of this tree see it as used, even
            // if the redundancy rule removes the split.
            if (b.cands[f].kind == .ctr) try b.markUsed(&b.cands[f]);
            // A chosen combination is stored now: next level rebuilds the candidates it lives in.
            if (b.cands[f].parts.len != 0) {
                sp.combo = @intCast(b.tree_combos.items.len);
                sp.ctr_type = b.cands[f].ctr_type;
                try b.tree_combos.append(b.gpa, try b.storedCombo(&b.cands[f]));
            }
            splits[depth] = sp;

            const t_split = prof.start();
            var actx = ApplySplit{ .b = b, .sp = sp, .bit = @intCast(depth) };
            b.pool.parallelFor(ds.n_rows, &actx, ApplySplit.run, 8192);
            depth += 1;
            prof.stop(.sym_split, t_split);

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

        // The model's leaves: membership on the averaging fold, whose online statistics may place a
        // row differently from the fold that chose the splits.
        const t_leaves = prof.start();
        b.leaf_model = b.leaf;
        if (!(b.avg_alias and b.fold_k == 0)) {
            try b.assignLeaves(&b.avg, splits[0..depth], b.leaf_buf);
            b.leaf_model = b.leaf_buf;
        }
        try b.leafValues(depth, grads, raw, labels, objective, scale_pos_weight, b.leaf_model);
        prof.stop(.sym_leaves, t_leaves);
        const t_folds = prof.start();
        defer prof.stop(.sym_folds, t_folds);
        // Every learning fold takes its own step, on its own membership.
        for (b.folds, 0..) |*fd, j| {
            if (fd.approx.len == 0 and fd.papprox.len == 0) continue;
            var lj = b.leaf;
            if (j != b.fold_k) {
                try b.assignLeaves(fd, splits[0..depth], b.leaf_tmp);
                lj = b.leaf_tmp;
            }
            if (b.s.ordered) try b.updatePrefixes(fd, depth, lj, labels, objective, scale_pos_weight) else try b.updatePlain(fd, depth, lj, labels, objective, scale_pos_weight);
        }
        const n_leaves = @as(usize, 1) << @intCast(depth);
        var sum_abs: f64 = 0;
        for (b.values[0..n_leaves]) |v| sum_abs += @abs(v);
        b.prev_mean_leaf = sum_abs / @as(f64, @floatFromInt(n_leaves));
        b.iteration += 1;
        return b.toTree(splits[0..depth]);
    }

    /// CatBoost's `SetSmallestSideControl`: of the rows a histogram counts (nonzero weight), the
    /// side of split bit `bit` with fewer of them (`true` = bit set; a tie takes it), compacted
    /// into `sel` in ascending row order with the weighted derivatives the histograms add.
    fn selectSide(b: *Builder, grads: []const hist.GradPair, bit: u32) bool {
        const n = b.ds.n_rows;
        const mask = @as(u16, 1) << @intCast(bit);
        var cctx = SelectCtx{ .b = b, .grads = grads, .mask = mask, .side = false, .chunk = (n + select_chunks - 1) / select_chunks };
        b.pool.parallelFor(select_chunks, &cctx, SelectCtx.count, 1);
        var set: usize = 0;
        var kept: usize = 0;
        for (b.sel.chunk_set, b.sel.chunk_kept) |x, k| {
            set += x;
            kept += k;
        }
        cctx.side = !(set * 2 > kept);
        // Each chunk's start in `sel`: the chunks before it, in order, on the chosen side.
        var at: usize = 0;
        for (b.sel.chunk_set, b.sel.chunk_kept, b.sel.chunk_at) |x, k, *o| {
            o.* = at;
            at += if (cctx.side) x else k - x;
        }
        b.sel.len = at;
        b.pool.parallelFor(select_chunks, &cctx, SelectCtx.fill, 1);
        return cctx.side;
    }

    /// Ordered scoring's counterpart of `selectSide`: fold positions (ascending) whose row is on the
    /// side of split bit `bit` with fewer rows, into `sel.row`. Weights are applied when counting.
    fn selectPositions(b: *Builder, fold: *const Fold, bit: u32) bool {
        const n = b.ds.n_rows;
        const mask = @as(u16, 1) << @intCast(bit);
        var cctx = SelectCtx{ .b = b, .grads = &.{}, .mask = mask, .side = false, .chunk = (n + select_chunks - 1) / select_chunks, .fold = fold };
        b.pool.parallelFor(select_chunks, &cctx, SelectCtx.count, 1);
        var set: usize = 0;
        for (b.sel.chunk_set) |x| set += x;
        cctx.side = !(set * 2 > n);
        var at: usize = 0;
        for (b.sel.chunk_set, b.sel.chunk_kept, b.sel.chunk_at) |x, k, *o| {
            o.* = at;
            at += if (cctx.side) x else k - x;
        }
        b.sel.len = at;
        b.pool.parallelFor(select_chunks, &cctx, SelectCtx.fill, 1);
        return cctx.side;
    }

    /// Each prefix model's derivatives on its rows, and per row its own tail's (rows of the first
    /// body take the first prefix's): what ordered scoring, bootstrap and noise read.
    fn prefixDerivatives(b: *Builder, fold: *const Fold, labels: []const f32, objective: Objective, scale_pos_weight: f32) void {
        for (b.bts, 0..) |bt, k| {
            var pctx = PosCtx{ .b = b, .fold = fold, .labels = labels, .objective = objective, .spw = scale_pos_weight, .mode = .prefix_deriv, .k = k };
            b.pool.parallelFor(bt.tail, &pctx, PosCtx.run, pos_chunk);
        }
    }

    /// After a tree: each prefix model takes Newton steps fitted on its body rows only and applies
    /// them to every row it scores (CatBoost's approx_calcer, ordered branch).
    fn updatePrefixes(b: *Builder, fold: *Fold, depth: u32, leaf: []const u16, labels: []const f32, objective: Objective, scale_pos_weight: f32) !void {
        const n_leaves = @as(usize, 1) << @intCast(depth);
        const g = try b.gpa.alloc(f64, n_leaves);
        defer b.gpa.free(g);
        const h = try b.gpa.alloc(f64, n_leaves);
        defer b.gpa.free(h);
        for (b.bts, 0..) |bt, k| {
            // Derivatives in parallel, then summed in position order as before.
            var pctx = PosCtx{ .b = b, .fold = fold, .labels = labels, .objective = objective, .spw = scale_pos_weight, .mode = .body_deriv, .k = k, .leaf = leaf, .step = g };
            b.pool.parallelFor(bt.body, &pctx, PosCtx.run, pos_chunk);
            @memset(g, 0);
            @memset(h, 0);
            for (b.dscratch[0..bt.body], 0..) |d, pos| {
                const l = leaf[fold.row(pos)];
                g[l] += d.g;
                h[l] += d.h;
            }
            for (g, h) |*gs, hs| gs.* = if (hs + b.s.lambda > 0) -gs.* / (hs + b.s.lambda) * b.s.learning_rate else 0;
            pctx.mode = .prefix_apply;
            b.pool.parallelFor(bt.tail, &pctx, PosCtx.run, pos_chunk);
        }
    }

    /// A plain learning fold's own step: Newton over all rows on its membership, as the model's
    /// leaves are (one step; CatBoost's `UpdateLearningFold`).
    fn updatePlain(b: *Builder, fold: *Fold, depth: u32, leaf: []const u16, labels: []const f32, objective: Objective, scale_pos_weight: f32) !void {
        const n_leaves = @as(usize, 1) << @intCast(depth);
        const g = try b.gpa.alloc(f64, n_leaves);
        defer b.gpa.free(g);
        const h = try b.gpa.alloc(f64, n_leaves);
        defer b.gpa.free(h);
        var pctx = PosCtx{ .b = b, .fold = fold, .labels = labels, .objective = objective, .spw = scale_pos_weight, .mode = .plain_deriv, .leaf = leaf, .step = g };
        b.pool.parallelFor(fold.approx.len, &pctx, PosCtx.run, pos_chunk);
        @memset(g, 0);
        @memset(h, 0);
        for (b.dscratch[0..fold.approx.len], leaf) |d, l| {
            g[l] += d.g;
            h[l] += d.h;
        }
        for (g, h) |*gs, hs| gs.* = if (hs + b.s.lambda > 0) -gs.* / (hs + b.s.lambda) * b.s.learning_rate else 0;
        pctx.mode = .plain_apply;
        b.pool.parallelFor(fold.approx.len, &pctx, PosCtx.run, pos_chunk);
    }

    /// Each row's leaf under `splits` on `fold`: its own online statistics for CTR splits.
    fn assignLeaves(b: *Builder, fold: *const Fold, splits: []const Split, out: []u16) !void {
        @memset(out, 0);
        for (splits, 0..) |sp, d| {
            if (sp.combo != no_combo) try comboOnline(b.gpa, b.ds, b.tree_combos.items[sp.combo].parts, sp.ctr_type, fold.order, b.combo_buf);
            var ctx = LeafBitCtx{ .b = b, .fold = fold, .sp = sp, .bit = @intCast(d), .out = out };
            b.pool.parallelFor(out.len, &ctx, LeafBitCtx.run, pos_chunk);
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

    fn isUsed(b: *const Builder, c: *const Cand) bool {
        var buf: [1]tree.Part = undefined;
        const proj = c.projection(&buf);
        for (b.used.items) |u| {
            if (u.counter == c.isCounter() and partsEql(u.parts, proj)) return true;
        }
        return false;
    }

    fn markUsed(b: *Builder, c: *const Cand) !void {
        if (b.isUsed(c)) return;
        var buf: [1]tree.Part = undefined;
        const parts = try b.gpa.dupe(tree.Part, c.projection(&buf));
        errdefer b.gpa.free(parts);
        try b.used.append(b.gpa, .{ .counter = c.isCounter(), .parts = parts });
    }

    /// A copy of a combination candidate's projection and table, owned by the caller.
    fn storedCombo(b: *Builder, c: *const Cand) !tree.Combo {
        const parts = try b.gpa.dupe(tree.Part, c.parts);
        errdefer b.gpa.free(parts);
        const keys = try b.gpa.dupe(u64, c.keys);
        errdefer b.gpa.free(keys);
        const buckets = try b.gpa.dupe(u8, c.kbuckets);
        return .{ .parts = parts, .keys = keys, .buckets = buckets, .unseen = c.unseen };
    }

    /// Frees this level's combination candidates and shrinks the view back to the static ones.
    fn dropCombos(b: *Builder) void {
        for (b.cands[b.n_static..]) |c| c.freeOwned(b.gpa);
        b.cands = b.all[0..b.n_static];
    }

    /// CatBoost's tree CTRs (greedy_tensor_search.cpp, `AddTreeCtrs`): from the splits above this
    /// level, the bases are every numeric and one-hot split taken together as one projection, and
    /// each CTR projection chosen in this tree. Each base gains each categorical not already in
    /// it, while the projection's length stays within `max_ctr_complexity`; duplicates are
    /// skipped. Length is CatBoost's `GetFullProjectionLength`: its categoricals, plus one if it
    /// has any split bits at all, however many.
    /// Every new projection becomes four CTR candidates after the static ones.
    fn levelCombos(b: *Builder, above: []const Split, order: []const u32) !void {
        b.dropCombos();
        b.max_uniq = b.static_max_uniq;
        if (!b.s.ctr or b.s.max_ctr_complexity <= 1 or above.len == 0 or b.wide.len == 0) return;
        const gpa = b.gpa;
        var bases: std.ArrayList([]tree.Part) = .empty;
        defer {
            for (bases.items) |x| gpa.free(x);
            bases.deinit(gpa);
        }
        var bits: std.ArrayList(tree.Part) = .empty;
        defer bits.deinit(gpa);
        // A combination split's candidate slot is reused by later levels: its projection lives in
        // `tree_combos`, and it is never a split bit.
        for (above) |sp| {
            if (sp.combo != no_combo) continue;
            const c = b.all[sp.cand];
            switch (c.kind) {
                .numeric => try bits.append(gpa, .{ .kind = .bin, .feature = c.feature, .value = @intCast(sp.threshold) }),
                .onehot => try bits.append(gpa, .{ .kind = .onehot, .feature = c.feature, .value = @intCast(sp.threshold) }),
                .ctr => {},
            }
        }
        if (bits.items.len != 0) {
            std.sort.pdq(tree.Part, bits.items, {}, tree.Part.lessThan);
            try bases.append(gpa, try gpa.dupe(tree.Part, bits.items));
        }
        for (above) |sp| {
            var buf: [1]tree.Part = undefined;
            const proj: []const tree.Part = if (sp.combo != no_combo)
                b.tree_combos.items[sp.combo].parts
            else if (b.all[sp.cand].kind == .ctr)
                b.all[sp.cand].projection(&buf)
            else
                continue;
            var seen = false;
            for (bases.items) |x| seen = seen or partsEql(x, proj);
            if (!seen) try bases.append(gpa, try gpa.dupe(tree.Part, proj));
        }

        var n = b.n_static;
        var max_uniq: u32 = 0;
        for (b.all[0..b.n_static]) |c| if (c.kind == .ctr) {
            max_uniq = @max(max_uniq, c.uniq);
        };
        // The level's new projections, deduplicated in discovery order; their candidates are then
        // built in parallel, each into its own slots, so the result does not depend on threads.
        var projs: std.ArrayList([]tree.Part) = .empty;
        defer {
            for (projs.items) |x| if (x.len != 0) gpa.free(x);
            projs.deinit(gpa);
        }
        for (bases.items) |base| {
            var n_cats: usize = 0;
            var bits_part: usize = 0;
            for (base) |part| {
                if (part.kind == .cat) n_cats += 1 else bits_part = 1;
            }
            if (n_cats + 1 + bits_part > b.s.max_ctr_complexity) continue;
            for (b.wide) |w| {
                var has = false;
                for (base) |part| has = has or (part.kind == .cat and part.feature == w);
                if (has) continue;
                const proj = try gpa.alloc(tree.Part, base.len + 1);
                var proj_owned = true;
                defer if (proj_owned) gpa.free(proj);
                @memcpy(proj[0..base.len], base);
                proj[base.len] = .{ .kind = .cat, .feature = w };
                std.sort.pdq(tree.Part, proj, {}, tree.Part.lessThan);
                var dup = false;
                for (projs.items) |x| dup = dup or partsEql(x, proj);
                if (dup) continue;
                try projs.append(gpa, proj);
                proj_owned = false;
            }
        }
        std.debug.assert(n + projs.items.len * n_ctr_types <= b.all.len);
        const n_new = projs.items.len * n_ctr_types;
        const out = b.all[n..][0..n_new];
        for (out) |*c| c.* = .{ .kind = .ctr, .feature = std.math.maxInt(u32), .nb = ctr_buckets };
        // Every candidate's arrays are freed on error; none owns its projection until the end.
        errdefer for (out) |*c| {
            c.freeOwned(gpa);
            c.* = .{ .kind = .ctr, .feature = std.math.maxInt(u32), .nb = ctr_buckets };
        };
        const work = try gpa.alloc(ComboWork, projs.items.len);
        defer gpa.free(work);
        for (work) |*w| w.* = .{ .proj = &.{} };
        defer for (work) |*w| w.deinit(gpa);
        for (work, projs.items) |*w, x| w.* = try ComboWork.init(gpa, b.ds, x);
        for (out, 0..) |*c, i| {
            c.ctr_type = @intCast(i % n_ctr_types);
            c.col = try gpa.alloc(u8, b.ds.n_rows);
        }
        const errs = try gpa.alloc(?anyerror, projs.items.len);
        defer gpa.free(errs);
        @memset(errs, null);
        var cctx = ComboCtx{ .gpa = gpa, .ds = b.ds, .order = order, .work = work, .out = out, .errs = errs, .pass = .codes };
        const n_tasks = projs.items.len * combo_chunks;
        b.pool.parallelFor(n_tasks, &cctx, ComboCtx.run, 1);
        for (errs) |e| if (e) |err| return err;
        for (work) |*w| try w.tables(gpa, b.ds.n_rows);
        cctx.pass = .count;
        b.pool.parallelFor(n_tasks, &cctx, ComboCtx.run, 1);
        for (work) |*w| w.prefix();
        cctx.pass = .replay;
        b.pool.parallelFor(n_tasks, &cctx, ComboCtx.run, 1);
        cctx.pass = .counter;
        b.pool.parallelFor(n_tasks, &cctx, ComboCtx.run, 1);
        for (work, 0..) |*w, i| try w.finish(gpa, out[i * n_ctr_types ..][0..n_ctr_types]);
        for (projs.items, 0..) |*x, i| {
            const first = &out[i * n_ctr_types];
            first.parts = x.*;
            first.owns_parts = true;
            for (out[i * n_ctr_types + 1 ..][0 .. n_ctr_types - 1]) |*c| c.parts = x.*;
            max_uniq = @max(max_uniq, first.uniq);
            x.* = &.{}; // the first candidate owns it now
        }
        n += projs.items.len * n_ctr_types;
        b.max_uniq = max_uniq;
        b.cands = b.all[0..n];
        // Histogram layout for the combinations, after the static candidates'.
        const scored_leaves: usize = @as(usize, 1) << @intCast(b.s.depth - 1);
        for (b.n_static..n) |i| {
            b.off[i + 1] = b.off[i] + scored_leaves * ctr_buckets;
            b.has_missing[i] = false;
        }
        if (b.s.ordered) for (b.n_static..n) |i| {
            b.acc_off[i + 1] = b.acc_off[i] + ctr_buckets;
        };
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
        leaf: []const u16,
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
                for (leaf, grads) |l, gp| {
                    g[l] += gp.g;
                    h[l] += gp.h;
                }
            } else {
                for (leaf, raw, labels) |l, r0, y| {
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
        var combos: std.ArrayList(tree.Combo) = .empty;
        errdefer {
            for (combos.items) |*x| x.deinit(b.gpa);
            combos.deinit(b.gpa);
        }
        for (sp, 0..) |s, level| {
            if (s.combo != no_combo) {
                // Moved out of the builder's list; the slot left behind is emptied below.
                const idx: u32 = @intCast(combos.items.len);
                try combos.append(b.gpa, b.tree_combos.items[s.combo]);
                b.tree_combos.items[s.combo] = .{ .parts = &.{}, .keys = &.{}, .buckets = &.{}, .unseen = 0 };
                proto[level] = .{ .feature = idx, .threshold = @intCast(s.threshold), .kind = .combo, .is_leaf = false };
                continue;
            }
            const c = b.all[s.cand];
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
        // Combinations chosen but removed by the redundancy rule were never moved out.
        for (b.tree_combos.items) |*x| if (x.parts.len != 0) x.deinit(b.gpa);
        b.tree_combos.clearRetainingCapacity();
        const cat_ids = try ids.toOwnedSlice(b.gpa);
        errdefer b.gpa.free(cat_ids);
        return .{ .nodes = nodes, .cat_ids = cat_ids, .combos = try combos.toOwnedSlice(b.gpa) };
    }
};

const MadeFolds = struct { learning: []Fold, avg: Fold, avg_alias: bool };

/// CatBoost's folds (learn_context.cpp, `TFoldsCreationParams`): without `has_time`, target
/// statistics or ordered boosting make `permutation_count - 1` learning folds. With target
/// statistics the rows are first shuffled once; fold 0 keeps that order, later folds and the
/// averaging fold block-shuffle it. Ordered boosting alone shuffles only learning folds after the
/// first. Otherwise one fold in file order, shared with the model and as the averaging fold.
fn makeFolds(gpa: std.mem.Allocator, ds: *const Dataset, s: Settings, n_slots: u32, cands: []const Cand, ap_off: []const usize) !MadeFolds {
    const n = ds.n_rows;
    const has_ctr = n_slots != 0;
    const permute = !s.has_time and (has_ctr or s.ordered);
    const n_folds: usize = if (permute) @max(1, s.permutation_count -| 1) else 1;
    const block: usize = if (s.permutation_block != 0) s.permutation_block else @min(256, n / 1000 + 1);
    var prng = std.Random.DefaultPrng.init(mix(s.seed, 0xF01D5, 0, 0));
    const r = prng.random();

    const learning = try gpa.alloc(Fold, n_folds);
    for (learning) |*f| f.* = .{};
    var made: MadeFolds = .{ .learning = learning, .avg = .{}, .avg_alias = !permute };
    errdefer {
        for (made.learning) |*f| f.deinit(gpa);
        gpa.free(made.learning);
        if (!made.avg_alias) made.avg.deinit(gpa);
    }
    // The order fold 0 keeps: a full shuffle when there are statistics to make order-free.
    if (!s.has_time and has_ctr) {
        const base = try gpa.alloc(u32, n);
        for (base, 0..) |*x, i| x.* = @intCast(i);
        r.shuffle(u32, base);
        learning[0].order = base;
    }
    for (learning[1..]) |*f| f.order = try blockShuffle(gpa, learning[0].order, n, block, r);
    if (!made.avg_alias and !s.has_time and has_ctr) made.avg.order = try blockShuffle(gpa, learning[0].order, n, block, r);

    for (learning) |*f| f.ctr = try foldColumns(gpa, ds, cands, n_slots, f.order);
    if (!made.avg_alias) made.avg.ctr = try foldColumns(gpa, ds, cands, n_slots, made.avg.order);

    if (permute) {
        for (learning) |*f| {
            if (s.ordered) {
                f.papprox = try gpa.alloc(f64, ap_off[ap_off.len - 1]);
                @memset(f.papprox, 0);
                f.pderiv = try gpa.alloc(f64, ap_off[ap_off.len - 1]);
            } else {
                f.approx = try gpa.alloc(f64, n);
                @memset(f.approx, 0);
            }
        }
    } else if (s.ordered) {
        learning[0].papprox = try gpa.alloc(f64, ap_off[ap_off.len - 1]);
        @memset(learning[0].papprox, 0);
        learning[0].pderiv = try gpa.alloc(f64, ap_off[ap_off.len - 1]);
    }
    if (made.avg_alias) made.avg = learning[0];
    return made;
}

/// Online columns of every single-column CTR along `order`. Caller owns the result.
fn foldColumns(gpa: std.mem.Allocator, ds: *const Dataset, cands: []const Cand, n_slots: u32, order: []const u32) ![][]u8 {
    const cols = try gpa.alloc([]u8, n_slots);
    var made: usize = 0;
    errdefer {
        for (cols[0..made]) |c| gpa.free(c);
        gpa.free(cols);
    }
    var widest: usize = 0;
    for (ds.n_bins) |nb| widest = @max(widest, nb);
    const scratch = try gpa.alloc(u8, widest);
    defer gpa.free(scratch);
    for (cands) |c| {
        if (c.kind != .ctr) continue;
        cols[c.slot] = try gpa.alloc(u8, ds.n_rows);
        made += 1;
        _ = try ctrColumn(gpa, ds, c.feature, c.ctr_type, order, cols[c.slot], scratch[0..ds.n_bins[c.feature]]);
    }
    return cols;
}

/// Candidate index and its threshold (for one-hot, the level); a chosen combination also records
/// its slot in the builder's `tree_combos`.
const Split = struct { cand: u32, threshold: u32, combo: u32 = no_combo, ctr_type: u8 = 0 };
const no_combo = std.math.maxInt(u32);

fn partsEql(a: []const tree.Part, c: []const tree.Part) bool {
    if (a.len != c.len) return false;
    for (a, c) |x, y| if (x.kind != y.kind or x.feature != y.feature or x.value != y.value) return false;
    return true;
}

const RowCtx = struct { ds: *const Dataset, r: usize };
fn rowCtxBin(ctx: RowCtx, f: u32) data.BinIdx {
    return binAt(ctx.ds, f, ctx.r);
}

/// Each row's dense id for `proj` (0.. in order of first appearance by row) into `ids`, and every
/// id's key, in id order, into `key_of`. When the parts' value combinations fit `direct_ids`, a row's
/// mixed-radix code indexes a table; otherwise one hash lookup per row. Either way ids and keys are
/// the same, and the key is computed once per id.
fn comboIds(gpa: std.mem.Allocator, ds: *const Dataset, proj: []const tree.Part, ids: []u32, key_of: *std.ArrayList(u64)) !void {
    var space: usize = 1;
    for (proj) |part| {
        const card: usize = if (part.kind == .cat) ds.n_bins[part.feature] else 2;
        space = std.math.mul(usize, space, card) catch std.math.maxInt(usize);
    }
    if (space <= direct_ids) {
        const table = try gpa.alloc(u32, space);
        defer gpa.free(table);
        @memset(table, unseen_id);
        for (ids, 0..) |*id, r| {
            var code: usize = 0;
            for (proj) |part| {
                const bin = binAt(ds, part.feature, r);
                code = switch (part.kind) {
                    .cat => code * ds.n_bins[part.feature] + bin,
                    .bin => code * 2 + @intFromBool(bin > part.value),
                    .onehot => code * 2 + @intFromBool(bin == part.value),
                };
            }
            if (table[code] == unseen_id) {
                table[code] = @intCast(key_of.items.len);
                try key_of.append(gpa, tree.comboKey(proj, RowCtx{ .ds = ds, .r = r }, rowCtxBin));
            }
            id.* = table[code];
        }
        return;
    }
    var index: std.AutoHashMapUnmanaged(u64, u32) = .empty;
    defer index.deinit(gpa);
    for (ids, 0..) |*id, r| {
        const k = tree.comboKey(proj, RowCtx{ .ds = ds, .r = r }, rowCtxBin);
        const gop = try index.getOrPut(gpa, k);
        if (!gop.found_existing) {
            gop.value_ptr.* = @intCast(key_of.items.len);
            try key_of.append(gpa, k);
        }
        id.* = gop.value_ptr.*;
    }
}

/// Largest value space `comboIds` indexes directly (4 MiB of u32 per projection being built).
const direct_ids = 1 << 20;
const unseen_id = std.math.maxInt(u32);

/// One CTR type's online buckets for a combination along `order` (by row), for a fold other than
/// the one that chose it.
fn comboOnline(gpa: std.mem.Allocator, ds: *const Dataset, proj: []const tree.Part, t: u8, order: []const u32, col: []u8) !void {
    const ids = try gpa.alloc(u32, ds.n_rows);
    defer gpa.free(ids);
    var key_of: std.ArrayList(u64) = .empty;
    defer key_of.deinit(gpa);
    try comboIds(gpa, ds, proj, ids, &key_of);
    const total = try gpa.alloc(u32, key_of.items.len);
    defer gpa.free(total);
    @memset(total, 0);
    if (t < ctr_priors.len) {
        const good = try gpa.alloc(u32, key_of.items.len);
        defer gpa.free(good);
        @memset(good, 0);
        for (0..ds.n_rows) |pos| {
            const r = if (order.len == 0) pos else order[pos];
            const id = ids[r];
            col[r] = bucketOf((@as(f32, @floatFromInt(good[id])) + ctr_priors[t]) / (@as(f32, @floatFromInt(total[id])) + 1));
            total[id] += 1;
            good[id] += @intFromBool(ds.labels[r] > 0.5);
        }
        return;
    }
    for (ids) |id| total[id] += 1;
    var largest: u32 = 0;
    for (total) |x| largest = @max(largest, x);
    const den: f32 = @floatFromInt(largest + 1);
    for (col, ids) |*o, id| o.* = bucketOf(@as(f32, @floatFromInt(total[id])) / den);
}

/// Chunks a combination's rows are split into for its parallel passes, a fixed count so the
/// result does not depend on threads.
const combo_chunks = 16;

/// One new combination of a level, built in passes that each split the rows into chunks:
/// every row's code (its mixed-radix part values when they fit `direct_ids`, else a dense id from
/// hashing), per-chunk counts of each code along the fold order, each chunk's starting counts
/// (a prefix over chunks), then the online buckets replayed per chunk. Counts are integers, so
/// this equals one sequential pass exactly.
const ComboWork = struct {
    proj: []const tree.Part,
    /// Part cardinalities for the direct codes; empty on the hashed path.
    card: []u32 = &.{},
    code: []u32 = &.{},
    n_codes: usize = 0,
    /// Hashed path: each id's key.
    key_of: std.ArrayList(u64) = .empty,
    /// Chunks along the fold order: `combo_chunks`, or 1 when a chunk's count table would be
    /// larger than the chunk.
    chunks: usize = 1,
    /// `chunks * n_codes`: counts per chunk, then each chunk's starting counts.
    good: []u32 = &.{},
    total: []u32 = &.{},
    /// Counts over every row.
    full_total: []u32 = &.{},
    full_good: []u32 = &.{},
    den: f32 = 0,

    fn deinit(w: *ComboWork, gpa: std.mem.Allocator) void {
        if (w.card.len != 0) gpa.free(w.card);
        if (w.code.len != 0) gpa.free(w.code);
        w.key_of.deinit(gpa);
        if (w.good.len != 0) gpa.free(w.good);
        if (w.total.len != 0) gpa.free(w.total);
        if (w.full_total.len != 0) gpa.free(w.full_total);
        if (w.full_good.len != 0) gpa.free(w.full_good);
        w.* = .{ .proj = &.{} };
    }

    fn init(gpa: std.mem.Allocator, ds: *const Dataset, proj: []const tree.Part) !ComboWork {
        var w: ComboWork = .{ .proj = proj };
        errdefer w.deinit(gpa);
        var space: usize = 1;
        for (proj) |part| {
            const card: usize = if (part.kind == .cat) ds.n_bins[part.feature] else 2;
            space = std.math.mul(usize, space, card) catch std.math.maxInt(usize);
        }
        w.code = try gpa.alloc(u32, ds.n_rows);
        if (space <= direct_ids and proj.len <= max_combo_parts) {
            w.card = try gpa.alloc(u32, proj.len);
            for (proj, w.card) |part, *c| c.* = if (part.kind == .cat) ds.n_bins[part.feature] else 2;
            w.n_codes = space;
        }
        return w;
    }

    /// After the codes exist: chunking and count tables.
    fn tables(w: *ComboWork, gpa: std.mem.Allocator, n: usize) !void {
        if (w.card.len == 0) w.n_codes = w.key_of.items.len;
        w.chunks = if (w.n_codes * combo_chunks <= n) combo_chunks else 1;
        w.good = try gpa.alloc(u32, w.chunks * w.n_codes);
        @memset(w.good, 0);
        w.total = try gpa.alloc(u32, w.chunks * w.n_codes);
        @memset(w.total, 0);
        w.full_good = try gpa.alloc(u32, w.n_codes);
        w.full_total = try gpa.alloc(u32, w.n_codes);
    }

    /// Chunk `c` of `chunks` over `[0, n)`.
    fn span(n: usize, chunks: usize, c: usize) [2]usize {
        return .{ n * c / chunks, n * (c + 1) / chunks };
    }

    /// Each chunk's counts become its starting counts; the sums over all chunks are the totals.
    fn prefix(w: *ComboWork) void {
        var largest: u32 = 0;
        for (0..w.n_codes) |k| {
            var g: u32 = 0;
            var t: u32 = 0;
            for (0..w.chunks) |c| {
                const i = c * w.n_codes + k;
                const gc = w.good[i];
                const tc = w.total[i];
                w.good[i] = g;
                w.total[i] = t;
                g += gc;
                t += tc;
            }
            w.full_good[k] = g;
            w.full_total[k] = t;
            largest = @max(largest, t);
        }
        w.den = @floatFromInt(largest + 1);
    }

    /// The key of code `k`: from its digits (direct) or remembered (hashed).
    fn key(w: *const ComboWork, k: usize) u64 {
        if (w.card.len == 0) return w.key_of.items[k];
        var values: [max_combo_parts]u64 = undefined;
        var rest = k;
        var i = w.proj.len;
        while (i > 0) {
            i -= 1;
            values[i] = rest % w.card[i];
            rest /= w.card[i];
        }
        return tree.comboKeyValues(values[0..w.proj.len]);
    }

    /// The four candidates' stored tables (keys ascending) from the full counts.
    fn finish(w: *const ComboWork, gpa: std.mem.Allocator, out: []Cand) !void {
        var present: usize = 0;
        for (w.full_total) |t| present += @intFromBool(t != 0);
        const Entry = struct { key: u64, code: u32 };
        const entries = try gpa.alloc(Entry, present);
        defer gpa.free(entries);
        var at: usize = 0;
        for (w.full_total, 0..) |t, k| if (t != 0) {
            entries[at] = .{ .key = w.key(k), .code = @intCast(k) };
            at += 1;
        };
        std.sort.pdq(Entry, entries, {}, struct {
            fn lt(_: void, x: Entry, y: Entry) bool {
                return x.key < y.key;
            }
        }.lt);
        for (out, 0..) |*c, t| {
            c.uniq = @intCast(present);
            c.keys = try gpa.alloc(u64, present);
            c.kbuckets = try gpa.alloc(u8, present);
            for (entries, c.keys, c.kbuckets) |e, *k, *bk| {
                k.* = e.key;
                bk.* = if (t < ctr_priors.len)
                    bucketOf((@as(f32, @floatFromInt(w.full_good[e.code])) + ctr_priors[t]) / (@as(f32, @floatFromInt(w.full_total[e.code])) + 1))
                else
                    bucketOf(@as(f32, @floatFromInt(w.full_total[e.code])) / w.den);
            }
            c.unseen = if (t < ctr_priors.len) bucketOf(ctr_priors[t]) else 0;
        }
    }
};

/// Longest projection: the split bits of a tree (one per level) plus its categoricals.
const max_combo_parts = 2 * max_depth;

/// The passes of `ComboWork` over every new combination of a level at once: task `i` is chunk
/// `i % combo_chunks` of combination `i / combo_chunks`. Each task writes only its own chunk.
const ComboCtx = struct {
    gpa: std.mem.Allocator,
    ds: *const Dataset,
    order: []const u32,
    work: []ComboWork,
    out: []Cand,
    errs: []?anyerror,
    pass: enum { codes, count, replay, counter },

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *ComboCtx = @ptrCast(@alignCast(ctx));
        for (begin..end) |i| self.task(i / combo_chunks, i % combo_chunks) catch |e| {
            self.errs[i / combo_chunks] = e;
        };
    }

    fn task(self: *ComboCtx, p: usize, c: usize) !void {
        const w = &self.work[p];
        const ds = self.ds;
        const n = ds.n_rows;
        switch (self.pass) {
            .codes => {
                if (w.card.len == 0) {
                    // Hashed: one sequential pass, dense ids by first appearance.
                    if (c == 0) try comboIds(self.gpa, ds, w.proj, w.code, &w.key_of);
                    return;
                }
                const lo, const hi = ComboWork.span(n, combo_chunks, c);
                for (lo..hi) |r| {
                    var code: u32 = 0;
                    for (w.proj, w.card) |part, card| code = code * card + @as(u32, @intCast(tree.partValue(part, binAt(ds, part.feature, r))));
                    w.code[r] = code;
                }
            },
            .count => {
                if (c >= w.chunks) return;
                const lo, const hi = ComboWork.span(n, w.chunks, c);
                const good = w.good[c * w.n_codes ..][0..w.n_codes];
                const total = w.total[c * w.n_codes ..][0..w.n_codes];
                for (lo..hi) |pos| {
                    const r = if (self.order.len == 0) pos else self.order[pos];
                    const k = w.code[r];
                    total[k] += 1;
                    good[k] += @intFromBool(ds.labels[r] > 0.5);
                }
            },
            .replay => {
                if (c >= w.chunks) return;
                const lo, const hi = ComboWork.span(n, w.chunks, c);
                const good = w.good[c * w.n_codes ..][0..w.n_codes];
                const total = w.total[c * w.n_codes ..][0..w.n_codes];
                const out = self.out[p * n_ctr_types ..][0..n_ctr_types];
                for (lo..hi) |pos| {
                    const r = if (self.order.len == 0) pos else self.order[pos];
                    const k = w.code[r];
                    const g: f32 = @floatFromInt(good[k]);
                    const t: f32 = @floatFromInt(total[k]);
                    for (ctr_priors, 0..) |prior, i| out[i].col[r] = bucketOf((g + prior) / (t + 1));
                    total[k] += 1;
                    good[k] += @intFromBool(ds.labels[r] > 0.5);
                }
            },
            .counter => {
                const lo, const hi = ComboWork.span(n, combo_chunks, c);
                const col = self.out[p * n_ctr_types + ctr_priors.len].col;
                for (lo..hi) |r| col[r] = bucketOf(@as(f32, @floatFromInt(w.full_total[w.code[r]])) / w.den);
            },
        }
    }
};

pub fn freeCands(gpa: std.mem.Allocator, cands: []Cand) void {
    for (cands) |c| c.freeOwned(gpa);
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
            c.uniq = try ctrColumn(gpa, ds, f, @intCast(t), &.{}, c.col, c.final);
            try list.append(gpa, c);
        }
    }
    return list.toOwnedSlice(gpa);
}

/// `trunc(value * 15)` in f32, as CatBoost buckets a CTR value in [0, 1].
inline fn bucketOf(value: f32) u8 {
    return @intFromFloat(@min(@trunc(value * 15), 15));
}

/// One CTR's training buckets (`col`, by row, online: each row sees only the rows before it in
/// `order`, file order when empty) and final buckets per level (`final`, counts over every row).
/// Borders: `(positives + prior) / (count + 1)`. Counter: `count / (largest count + 1)` over all
/// rows, online or not (CatBoost's `SkipTest`). Returns the number of levels present.
pub fn ctrColumn(gpa: std.mem.Allocator, ds: *const Dataset, f: u32, t: u8, order: []const u32, col: []u8, final: []u8) !u32 {
    const nl = ds.n_bins[f];
    const total = try gpa.alloc(u32, nl);
    defer gpa.free(total);
    const good = try gpa.alloc(u32, nl);
    defer gpa.free(good);
    @memset(total, 0);
    @memset(good, 0);
    if (t < ctr_priors.len) {
        const prior = ctr_priors[t];
        for (0..ds.n_rows) |pos| {
            const r = if (order.len == 0) pos else order[pos];
            const k = binAt(ds, f, r);
            col[r] = bucketOf((@as(f32, @floatFromInt(good[k])) + prior) / (@as(f32, @floatFromInt(total[k])) + 1));
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

/// Fixed chunk count for `selectSide`, so `sel` is the same at any thread count.
const select_chunks = 64;

const Selected = struct {
    row: []u32 = &.{},
    g: []f64 = &.{},
    n: []f64 = &.{},
    h: []f64 = &.{},
    len: usize = 0,
    chunk_set: []usize = &.{},
    chunk_kept: []usize = &.{},
    chunk_at: []usize = &.{},

    fn init(gpa: std.mem.Allocator, n: usize) !Selected {
        var x: Selected = .{};
        errdefer x.deinit(gpa);
        x.row = try gpa.alloc(u32, n);
        x.g = try gpa.alloc(f64, n);
        x.n = try gpa.alloc(f64, n);
        x.h = try gpa.alloc(f64, n);
        x.chunk_set = try gpa.alloc(usize, select_chunks);
        x.chunk_kept = try gpa.alloc(usize, select_chunks);
        x.chunk_at = try gpa.alloc(usize, select_chunks);
        return x;
    }

    fn deinit(x: *Selected, gpa: std.mem.Allocator) void {
        if (x.row.len != 0) gpa.free(x.row);
        if (x.g.len != 0) gpa.free(x.g);
        if (x.n.len != 0) gpa.free(x.n);
        if (x.h.len != 0) gpa.free(x.h);
        if (x.chunk_set.len != 0) gpa.free(x.chunk_set);
        if (x.chunk_kept.len != 0) gpa.free(x.chunk_kept);
        if (x.chunk_at.len != 0) gpa.free(x.chunk_at);
        x.* = .{};
    }
};

const SelectCtx = struct {
    b: *Builder,
    grads: []const hist.GradPair,
    mask: u16,
    side: bool,
    chunk: usize,
    /// Set for ordered scoring: chunks are fold positions, every row is kept, and only
    /// positions are written.
    fold: ?*const Fold = null,

    /// Chunk `c`'s rows are `[start(c), start(c + 1))`.
    fn start(self: *const SelectCtx, c: usize) usize {
        return @min(self.b.ds.n_rows, c * self.chunk);
    }

    fn count(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *SelectCtx = @ptrCast(@alignCast(ctx));
        const b = self.b;
        const w = b.weights;
        for (begin..end) |c| {
            const lo = self.start(c);
            const hi = self.start(c + 1);
            if (self.fold) |fold| {
                var set: usize = 0;
                for (lo..hi) |pos| set += @intFromBool(b.leaf[fold.row(pos)] & self.mask != 0);
                b.sel.chunk_set[c] = set;
                b.sel.chunk_kept[c] = hi - lo;
                continue;
            }
            var set: usize = 0;
            var kept: usize = 0;
            for (lo..hi) |r| {
                if (w.len != 0 and w[r] == 0) continue;
                kept += 1;
                set += @intFromBool(b.leaf[r] & self.mask != 0);
            }
            b.sel.chunk_set[c] = set;
            b.sel.chunk_kept[c] = kept;
        }
    }

    fn fill(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *SelectCtx = @ptrCast(@alignCast(ctx));
        const b = self.b;
        const w = b.weights;
        const sel = &b.sel;
        for (begin..end) |c| {
            const lo = self.start(c);
            const hi = self.start(c + 1);
            var at = sel.chunk_at[c];
            if (self.fold) |fold| {
                for (lo..hi) |pos| {
                    if ((b.leaf[fold.row(pos)] & self.mask != 0) != self.side) continue;
                    sel.row[at] = @intCast(pos);
                    at += 1;
                }
                continue;
            }
            for (lo..hi) |r| {
                const s: f64 = if (w.len != 0) w[r] else 1;
                if (s == 0) continue;
                if ((b.leaf[r] & self.mask != 0) != self.side) continue;
                const gp = self.grads[r];
                sel.row[at] = @intCast(r);
                sel.g[at] = s * gp.g;
                sel.n[at] = s;
                sel.h[at] = s * gp.h;
                at += 1;
            }
        }
    }
};

/// Per-feature histograms of the current level: each task owns one feature's slice, rows in order,
/// so the sums are the same at any thread count. At the root, and for a combination (new each
/// level), every row is counted; otherwise only the rows of the newest split's smaller side, and
/// the other side is the level above's histogram minus them, as CatBoost does (`FixUpStats`).
const HistCtx = struct {
    b: *Builder,
    grads: []const hist.GradPair,
    n_leaves: usize,
    /// The side `sel` holds, below the root.
    side: ?bool,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *HistCtx = @ptrCast(@alignCast(ctx));
        const b = self.b;
        for (begin..end) |f| {
            if (self.side) |side| {
                if (b.cached[f]) {
                    self.smallerSide(f, side);
                    continue;
                }
            }
            self.allRows(f);
            if (f < b.n_static) b.cached[f] = true;
        }
    }

    fn allRows(self: *HistCtx, f: usize) void {
        const b = self.b;
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

    fn smallerSide(self: *HistCtx, f: usize, side: bool) void {
        const b = self.b;
        const cand = b.cands[f];
        const nb: usize = cand.nb;
        const half = self.n_leaves / 2 * nb;
        const slice = b.cells[b.off[f]..][0 .. self.n_leaves * nb];
        // The level above's histogram is the lower half; its leaf l splits into l and l + half.
        if (!side) @memcpy(slice[half..], slice[0..half]);
        const fresh = if (side) slice[half..] else slice[0..half];
        const other = if (side) slice[0..half] else slice[half..];
        @memset(fresh, .{});
        const sel = &b.sel;
        // An index loop: a four-slice `for` here crashed the Debug tests (the stack walker read a
        // bad frame) though LLVM Debug, ReleaseSafe and valgrind found nothing; suspected
        // self-hosted backend bug, not reduced to a minimal case.
        for (0..sel.len) |i| {
            const r = sel.row[i];
            const bin: usize = if (cand.kind == .ctr) cand.col[r] else binAt(b.ds, cand.feature, r);
            const c = &slice[@as(usize, b.leaf[r]) * nb + bin];
            c.g += sel.g[i];
            c.n += sel.n[i];
            c.h += sel.h[i];
        }
        for (other, fresh) |*o, x| o.* = o.sub(x);
    }
};

/// Each feature's best threshold at this level, scored over all leaves.
const ScoreCtx = struct {
    b: *Builder,
    n_leaves: usize,
    level: u32,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        const self: *ScoreCtx = @ptrCast(@alignCast(ctx));
        for (begin..end) |f| self.b.best[f] = self.feature(f, worker);
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

    fn feature(self: *ScoreCtx, f: usize, worker: usize) Builder.Best {
        const b = self.b;
        const nb: usize = b.cands[f].nb;
        var best: Builder.Best = .{ .score = -std.math.inf(f64), .threshold = 0 };
        if (nb < 2) return best;
        if (b.s.ordered) return self.fromAccumulators(f);
        if (b.cands[f].kind == .onehot) return self.oneHot(f);
        // Each leaf's bins as running sums, in this worker's scratch (the raw histogram is the
        // next level's parent): `slice[l*nb + k]` is the left side of threshold k and the last
        // cell the leaf's total.
        const slice = b.scratch[worker * b.max_slice ..][0 .. self.n_leaves * nb];
        @memcpy(slice, b.cells[b.off[f]..][0 .. self.n_leaves * nb]);
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
    fold: *const Fold,
    bt: usize,
    n_leaves: usize,
    /// Below the root with banks kept: the side `sel` holds (positions, ascending).
    side: ?bool,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        const self: *OrderedCtx = @ptrCast(@alignCast(ctx));
        for (begin..end) |f| self.cand(f, worker);
    }

    fn cand(self: *OrderedCtx, f: usize, worker: usize) void {
        const b = self.b;
        const c = b.cands[f];
        const nb: usize = c.nb;
        const bt = b.bts[self.bt];
        const d = self.fold.pderiv[b.ap_off[self.bt]..][0..bt.tail];
        const len = self.n_leaves * nb;
        // A static candidate's histograms live in this prefix's banks, kept across levels; a
        // combination's (new each level), or all of them without banks, in the shared cells.
        const banked = f < b.n_static and b.obody.len != 0;
        const body = if (banked) b.obody[self.bt * b.ostride + b.off[f] ..][0..len] else b.cells[b.off[f]..][0..len];
        const tail = if (banked) b.otail[self.bt * b.ostride + b.off[f] ..][0..len] else b.cells_tail[b.off[f]..][0..len];
        if (self.side != null and banked and b.cached[f]) {
            self.smallerSide(c, d, bt, body, tail, self.side.?);
        } else {
            @memset(body, .{});
            @memset(tail, .{});
            for (0..bt.tail) |pos| self.add(c, d, bt, body, tail, pos);
        }
        // Running sums in this worker's scratch: the banks stay raw for the next level.
        const scratch = b.scratch[worker * 2 * b.max_slice ..][0 .. 2 * b.max_slice];
        const bs = scratch[0..len];
        const ts = scratch[b.max_slice..][0..len];
        @memcpy(bs, body);
        @memcpy(ts, tail);
        self.score(f, bs, ts);
    }

    /// Position `pos` of prefix `bt`: its body sums (prefix model's derivative, count) or its tail
    /// sums (bootstrap-weighted).
    inline fn add(self: *OrderedCtx, c: Cand, d: []const f64, bt: BodyTail, body: []Cell, tail: []Cell, pos: usize) void {
        const b = self.b;
        const r = self.fold.row(pos);
        const bin: usize = if (c.kind == .ctr) c.col[r] else binAt(b.ds, c.feature, r);
        const i = @as(usize, b.leaf[r]) * c.nb + bin;
        if (pos < bt.body) {
            body[i].g += d[pos];
            body[i].n += 1;
        } else {
            const s: f64 = if (b.weights.len != 0) b.weights[r] else 1;
            if (s == 0) return;
            tail[i].g += s * d[pos];
            tail[i].n += s;
        }
    }

    /// The selected positions this prefix covers, then the other side as the parent minus them,
    /// for both banks (CatBoost caches every body-tail's statistics the same way).
    fn smallerSide(self: *OrderedCtx, c: Cand, d: []const f64, bt: BodyTail, body: []Cell, tail: []Cell, side: bool) void {
        const half = body.len / 2;
        inline for (.{ body, tail }) |bank| {
            if (!side) @memcpy(bank[half..], bank[0..half]);
            @memset(if (side) bank[half..] else bank[0..half], .{});
        }
        const sel = &self.b.sel;
        for (0..sel.len) |j| {
            const pos: usize = sel.row[j];
            if (pos >= bt.tail) break;
            self.add(c, d, bt, body, tail, pos);
        }
        inline for (.{ body, tail }) |bank| {
            const fresh = if (side) bank[half..] else bank[0..half];
            const other = if (side) bank[0..half] else bank[half..];
            for (other, fresh) |*o, x| o.* = o.sub(x);
        }
    }

    /// For each threshold and leaf, the body's leaf estimates scored against the tail, added to the
    /// candidate's accumulators. `body` and `tail` are scratch copies, made running sums here.
    fn score(self: *OrderedCtx, f: usize, body: []Cell, tail: []Cell) void {
        const b = self.b;
        const c = b.cands[f];
        const nb: usize = c.nb;
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
    ap_off: []usize = &.{},
    cells_tail: []Cell = &.{},
    acc_num: []f64 = &.{},
    acc_den: []f64 = &.{},
    acc_off: []usize = &.{},
    tail_grads: []hist.GradPair = &.{},

    fn init(gpa: std.mem.Allocator, ds: *const Dataset, cands: []const Cand, capacity: usize, n_cells: usize, extra_acc: usize) !Ordered {
        var o: Ordered = .{};
        errdefer o.deinit(gpa);
        o.bts = try bodyTails(gpa, ds.n_rows);
        o.ap_off = try gpa.alloc(usize, o.bts.len + 1);
        o.ap_off[0] = 0;
        for (o.bts, 0..) |bt, k| o.ap_off[k + 1] = o.ap_off[k] + bt.tail;
        o.cells_tail = try gpa.alloc(Cell, n_cells);
        o.acc_off = try gpa.alloc(usize, capacity + 1);
        o.acc_off[0] = 0;
        for (cands, 0..) |c, i| o.acc_off[i + 1] = o.acc_off[i] + c.nb;
        o.acc_num = try gpa.alloc(f64, o.acc_off[cands.len] + extra_acc);
        o.acc_den = try gpa.alloc(f64, o.acc_off[cands.len] + extra_acc);
        o.tail_grads = try gpa.alloc(hist.GradPair, ds.n_rows);
        return o;
    }

    fn deinit(o: Ordered, gpa: std.mem.Allocator) void {
        if (o.bts.len != 0) gpa.free(o.bts);
        if (o.ap_off.len != 0) gpa.free(o.ap_off);
        if (o.cells_tail.len != 0) gpa.free(o.cells_tail);
        if (o.acc_num.len != 0) gpa.free(o.acc_num);
        if (o.acc_den.len != 0) gpa.free(o.acc_den);
        if (o.acc_off.len != 0) gpa.free(o.acc_off);
        if (o.tail_grads.len != 0) gpa.free(o.tail_grads);
    }
};

/// One split's bit of every row's leaf on a fold (its own CTR columns; a combination's buckets
/// are in `combo_buf`). Per row, no reductions.
const LeafBitCtx = struct {
    b: *Builder,
    fold: *const Fold,
    sp: Split,
    bit: u4,
    out: []u16,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *LeafBitCtx = @ptrCast(@alignCast(ctx));
        const b = self.b;
        const sp = self.sp;
        if (sp.combo != no_combo) {
            for (self.out[begin..end], b.combo_buf[begin..end]) |*o, bk| o.* |= @as(u16, @intFromBool(bk > sp.threshold)) << self.bit;
            return;
        }
        const c = b.all[sp.cand];
        for (self.out[begin..end], begin..) |*o, r| {
            const one = switch (c.kind) {
                .numeric => binAt(b.ds, c.feature, r) > sp.threshold,
                .onehot => binAt(b.ds, c.feature, r) == sp.threshold,
                .ctr => self.fold.ctr[c.slot][r] > sp.threshold,
            };
            o.* |= @as(u16, @intFromBool(one)) << self.bit;
        }
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

/// Rows per task for `PosCtx`, whose work per row is one derivative or one add.
const pos_chunk = 8192;

/// Per-row work over fold positions with no reductions, so any thread count gives the same bits:
/// derivatives (prefix `k`'s, or the plain fold's), and adding a tree's step by leaf.
const PosCtx = struct {
    b: *Builder,
    fold: *const Fold,
    labels: []const f32,
    objective: Objective,
    spw: f32,
    mode: enum { fold_grads, prefix_deriv, body_deriv, prefix_apply, plain_deriv, plain_apply },
    k: usize = 0,
    leaf: []const u16 = &.{},
    step: []const f64 = &.{},

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *PosCtx = @ptrCast(@alignCast(ctx));
        const b = self.b;
        const fold = self.fold;
        switch (self.mode) {
            .fold_grads => for (begin..end) |r| {
                const d = derivatives(self.objective, fold.approx[r], self.labels[r], self.spw);
                b.fold_grads[r] = .{ .g = @floatCast(d.g), .h = @floatCast(d.h) };
            },
            .prefix_deriv => {
                const ap = b.ap_off[self.k];
                const from = if (self.k == 0) 0 else b.bts[self.k].body;
                for (begin..end) |pos| {
                    const r = fold.row(pos);
                    const g = derivatives(self.objective, fold.papprox[ap + pos], self.labels[r], self.spw).g;
                    fold.pderiv[ap + pos] = g;
                    if (pos >= from) b.tail_grads[r] = .{ .g = @floatCast(g), .h = 0 };
                }
            },
            .body_deriv => {
                const ap = b.ap_off[self.k];
                for (begin..end) |pos| b.dscratch[pos] = derivatives(self.objective, fold.papprox[ap + pos], self.labels[fold.row(pos)], self.spw);
            },
            .prefix_apply => {
                const ap = b.ap_off[self.k];
                for (begin..end) |pos| fold.papprox[ap + pos] += self.step[self.leaf[fold.row(pos)]];
            },
            .plain_deriv => for (begin..end) |r| {
                b.dscratch[r] = derivatives(self.objective, fold.approx[r], self.labels[r], self.spw);
            },
            .plain_apply => for (begin..end) |r| {
                fold.approx[r] += self.step[self.leaf[r]];
            },
        }
    }
};

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
