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
};

pub const Builder = struct {
    gpa: std.mem.Allocator,
    pool: *Pool,
    ds: *const Dataset,
    s: Settings,
    /// Each row's leaf index in the tree being grown.
    leaf: []u16,
    /// Histograms for every feature over the deepest scored level: feature f's leaf l bin b is
    /// `cells[off[f] + l * nb_f + b]`.
    cells: []Cell,
    off: []usize,
    /// Whether feature f has any missing row: the missing-vs-present cut (threshold 0) is offered
    /// only then, as CatBoost adds its `nan_mode = Min` border only for columns with NaN.
    has_missing: []bool,
    /// Per-feature best split of the level being searched.
    best: []Best,
    /// Leaf values of the last grown tree, already scaled by the learning rate.
    values: []f64,

    const Best = struct { score: f64, threshold: u32 };

    pub fn init(gpa: std.mem.Allocator, pool: *Pool, ds: *const Dataset, s: Settings) !Builder {
        if (s.depth == 0 or s.depth > max_depth) return error.BadSymmetricDepth;
        const leaf = try gpa.alloc(u16, ds.n_rows);
        errdefer gpa.free(leaf);
        const off = try gpa.alloc(usize, ds.n_features + 1);
        errdefer gpa.free(off);
        const scored_leaves: usize = @as(usize, 1) << @intCast(s.depth - 1);
        off[0] = 0;
        for (0..ds.n_features) |f| off[f + 1] = off[f] + scored_leaves * ds.n_bins[f];
        const cells = try gpa.alloc(Cell, off[ds.n_features]);
        errdefer gpa.free(cells);
        const has_missing = try gpa.alloc(bool, ds.n_features);
        errdefer gpa.free(has_missing);
        for (has_missing, 0..) |*m, f| {
            m.* = false;
            for (0..ds.n_rows) |r| if (binAt(ds, f, r) == 0) {
                m.* = true;
                break;
            };
        }
        const best = try gpa.alloc(Best, ds.n_features);
        errdefer gpa.free(best);
        const values = try gpa.alloc(f64, @as(usize, 1) << @intCast(s.depth));
        return .{
            .gpa = gpa,
            .pool = pool,
            .ds = ds,
            .s = s,
            .leaf = leaf,
            .cells = cells,
            .off = off,
            .has_missing = has_missing,
            .best = best,
            .values = values,
        };
    }

    pub fn deinit(b: *Builder) void {
        b.gpa.free(b.leaf);
        b.gpa.free(b.cells);
        b.gpa.free(b.off);
        b.gpa.free(b.has_missing);
        b.gpa.free(b.best);
        b.gpa.free(b.values);
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

        while (depth < b.s.depth) {
            const n_leaves = @as(usize, 1) << @intCast(depth);
            var hctx = HistCtx{ .b = b, .grads = grads, .n_leaves = n_leaves };
            b.pool.parallelFor(ds.n_features, &hctx, HistCtx.run, 1);
            var sctx = ScoreCtx{ .b = b, .n_leaves = n_leaves };
            b.pool.parallelFor(ds.n_features, &sctx, ScoreCtx.run, 1);

            // Strict `>` in feature order: ties go to the lower feature index, then (within a
            // feature, in `ScoreCtx`) to the lower threshold, as CatBoost's scan does.
            var win: ?usize = null;
            for (b.best, 0..) |c, f| {
                if (!std.math.isFinite(c.score)) continue;
                if (win == null or c.score > b.best[win.?].score) win = f;
            }
            const f = win orelse break;
            const sp: Split = .{ .feature = @intCast(f), .threshold = @intCast(b.best[f].threshold) };
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
        return b.toTree(splits[0..depth]);
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

    /// A complete binary tree: every node at depth d tests split d, left = `bin <= threshold`.
    fn toTree(b: *Builder, sp: []const Split) !tree.Tree {
        const d = sp.len;
        const n_nodes = (@as(usize, 1) << @intCast(d + 1)) - 1;
        const nodes = try b.gpa.alloc(tree.Node, n_nodes);
        errdefer b.gpa.free(nodes);
        // Breadth-first: node i's children are 2i+1 and 2i+2; depth-d nodes start at 2^d - 1.
        for (nodes, 0..) |*nd, i| {
            const level = std.math.log2_int(usize, i + 1);
            if (level < d) {
                nd.* = .{
                    .feature = sp[level].feature,
                    .threshold = @intCast(sp[level].threshold),
                    .missing_left = true,
                    .is_leaf = false,
                    .left = @intCast(2 * i + 1),
                    .right = @intCast(2 * i + 2),
                };
            } else {
                // Position among the leaves, read root to leaf, has bit `level - 1 - k` set when
                // the k-th split went right; the leaf index has split k at bit k.
                const pos = i + 1 - (@as(usize, 1) << @intCast(d));
                var idx: usize = 0;
                for (0..d) |k| {
                    if ((pos >> @intCast(d - 1 - k)) & 1 != 0) idx |= @as(usize, 1) << @intCast(k);
                }
                nd.* = .{ .weight = @floatCast(b.values[idx]), .is_leaf = true };
            }
        }
        return .{ .nodes = nodes };
    }
};

const Split = struct { feature: u32, threshold: u32 };

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
            const nb: usize = b.ds.n_bins[f];
            const slice = b.cells[b.off[f]..][0 .. self.n_leaves * nb];
            @memset(slice, .{});
            for (b.leaf, self.grads, 0..) |l, gp, r| {
                const c = &slice[@as(usize, l) * nb + binAt(b.ds, f, r)];
                c.g += gp.g;
                c.n += 1;
                c.h += gp.h;
            }
        }
    }
};

/// Each feature's best threshold at this level, scored over all leaves.
const ScoreCtx = struct {
    b: *Builder,
    n_leaves: usize,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *ScoreCtx = @ptrCast(@alignCast(ctx));
        for (begin..end) |f| self.b.best[f] = self.feature(f);
    }

    fn feature(self: *ScoreCtx, f: usize) Builder.Best {
        const b = self.b;
        const nb: usize = b.ds.n_bins[f];
        var best: Builder.Best = .{ .score = -std.math.inf(f64), .threshold = 0 };
        if (nb < 2) return best;
        const slice = b.cells[b.off[f]..][0 .. self.n_leaves * nb];
        // Each leaf's bins become running sums in place (rebuilt every level): `slice[l*nb + k]`
        // is then the left side of threshold k and the last cell the leaf's total.
        for (0..self.n_leaves) |l| {
            const cells = slice[l * nb ..][0..nb];
            for (1..nb) |i| cells[i] = cells[i - 1].add(cells[i]);
        }
        const lambda = b.s.lambda;
        const first: usize = if (b.has_missing[f]) 0 else 1;
        // Thresholds k = first .. nb-2: left holds bins <= k.
        var k = first;
        while (k + 1 < nb) : (k += 1) {
            var num: f64 = 0;
            var den: f64 = 0;
            for (0..self.n_leaves) |l| {
                const left = slice[l * nb + k];
                // CatBoost forms the right side by subtraction from the total; so does this.
                const right = slice[l * nb + nb - 1].sub(left);
                for ([2]Cell{ left, right }) |side| switch (b.s.score) {
                    .cosine, .l2, .auto => {
                        const v = if (side.n > 0) side.g / (side.n + lambda) else 0;
                        num += v * side.g;
                        den += v * v * side.n;
                    },
                    .gain => num += side.g * side.g / (side.h + lambda),
                };
            }
            const s = switch (b.s.score) {
                .cosine, .auto => num / @sqrt(den + 1e-100),
                .l2, .gain => num,
            };
            if (s > best.score) best = .{ .score = s, .threshold = @intCast(k) };
        }
        return best;
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
        for (begin..end) |r| {
            const right = binAt(ds, self.sp.feature, r) > self.sp.threshold;
            self.b.leaf[r] |= @as(u16, @intFromBool(right)) << self.bit;
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
