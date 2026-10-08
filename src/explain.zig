// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! What a trained model uses: feature importance from the trees' stored node statistics, and
//! SHAP values per row. For trees, exact TreeSHAP (Lundberg, Erion and Lee, "Consistent
//! individualized feature attribution for tree ensembles", 2018, Algorithm 2), with each node's
//! training cover weighting the paths a row does not take: the hessian sum as XGBoost does, or
//! the row count as LightGBM and CatBoost do. A row's values and the bias sum to its raw score.
//! For the linear model, a coefficient times the column's distance from its training mean.

const std = @import("std");
const data = @import("data.zig");
const tree = @import("tree.zig");
const split = @import("split.zig");
const model = @import("model.zig");
const linear = @import("linear.zig");
const Pool = @import("pool.zig").Pool;
const Dataset = data.Dataset;

/// Which node statistic weights the paths a row does not take.
pub const Cover = enum {
    /// Hessian sum (XGBoost's `cover`).
    hessian,
    /// Training rows (LightGBM's `internal_count`, CatBoost's leaf weights without sample weights).
    count,
};

fn coverOf(st: tree.NodeStat, c: Cover) f64 {
    return switch (c) {
        .hessian => st.hess,
        .count => st.count,
    };
}

// ------------------------------------------------------------- importance

/// One feature's use across every split: summed gain and cover, and how many splits test it.
pub const Importance = struct {
    gain: f64 = 0,
    cover: f64 = 0,
    splits: u32 = 0,
};

/// Per-feature importance into `out` (one per feature, zeroed here). A combination split (CatBoost
/// target statistics over several columns) is shared equally among its columns. Needs the
/// statistics a format-7 model stores.
pub fn importance(trees: []const tree.Tree, cover: Cover, out: []Importance) !void {
    @memset(out, .{});
    for (trees) |t| {
        if (t.stats.len != t.nodes.len) return error.ModelHasNoNodeStats;
        for (t.nodes, t.stats) |n, st| {
            if (n.is_leaf) continue;
            if (n.kind == .combo) {
                const parts = t.combos[n.feature].parts;
                const share: f64 = 1 / @as(f64, @floatFromInt(parts.len));
                for (parts) |p| {
                    out[p.feature].gain += share * st.gain;
                    out[p.feature].cover += share * coverOf(st, cover);
                    out[p.feature].splits += 1;
                }
                continue;
            }
            out[n.feature].gain += st.gain;
            out[n.feature].cover += coverOf(st, cover);
            out[n.feature].splits += 1;
        }
    }
}

// ------------------------------------------------------------------ SHAP

/// What every tree method needs to know about a model to explain it.
pub const TreeModel = struct {
    trees: []const tree.Tree,
    /// Scores per row: classes under softmax, else 1. Tree `i` adds to score `i % width`.
    width: usize,
    /// Starting raw score per output (`width` long).
    base: []const f32,
    /// A forest averages its trees: each tree's share is `1 / trees.len`.
    average: bool,
    cover: Cover,
};

/// SHAP values for `ds`'s rows into `out`, row-major: per row and output, one value per feature,
/// then the bias (`width * (n_features + 1)` per row). Rows are independent, so any thread count
/// gives the same bits.
pub fn treeShap(gpa: std.mem.Allocator, pool: *Pool, m: TreeModel, ds: *const Dataset, out: []f64) !void {
    const f = ds.n_features;
    std.debug.assert(out.len == ds.n_rows * m.width * (f + 1));
    var max_depth: usize = 0;
    for (m.trees) |t| {
        if (t.stats.len != t.nodes.len) return error.ModelHasNoNodeStats;
        for (t.nodes) |n| {
            if (n.n_lin != 0) return error.ShapLinearLeavesUnsupported;
            if (!n.is_leaf and n.kind == .combo) return error.ShapCombinationUnsupported;
        }
        max_depth = @max(max_depth, depthOf(t));
    }
    // Symmetric trees are explained with their levels reversed, the last split at the root, as
    // CatBoost walks them (`CalcObliviousInternalShapValuesForLeafRecursive`). The function is the
    // same; path-dependent SHAP weighs unknown features in path order, so the attribution differs.
    const trees = try gpa.alloc(tree.Tree, m.trees.len);
    var n_made: usize = 0;
    defer {
        for (trees[0..n_made], m.trees[0..n_made]) |*t, orig| if (t.nodes.ptr != orig.nodes.ptr) {
            gpa.free(t.nodes);
            gpa.free(t.stats);
        };
        gpa.free(trees);
    }
    for (m.trees, trees) |orig, *t| {
        t.* = if (obliviousDepth(orig)) |d| try reversed(gpa, orig, d) else orig;
        n_made += 1;
    }
    var mr = m;
    mr.trees = trees;
    // Each tree's expected value, its contribution to the bias: leaves weighted by cover.
    const means = try gpa.alloc(f64, m.trees.len);
    defer gpa.free(means);
    for (mr.trees, means) |t, *mv| mv.* = nodeMean(t, 0, m.cover);

    // Path storage per worker: the path at depth d is a copy of its parent's plus one element,
    // so a branch needs (D + 1)(D + 2) / 2 elements at most (D = depth + 1 for the root's).
    const d = max_depth + 2;
    const per_worker = d * (d + 1) / 2;
    const paths = try gpa.alloc(PathElement, per_worker * pool.workerCount());
    defer gpa.free(paths);

    var ctx = ShapCtx{ .m = mr, .ds = ds, .out = out, .means = means, .paths = paths, .per_worker = per_worker };
    pool.parallelFor(ds.n_rows, &ctx, ShapCtx.run, 64);
}

const PathElement = struct {
    feature: i64,
    zero: f64,
    one: f64,
    weight: f64,
};

const ShapCtx = struct {
    m: TreeModel,
    ds: *const Dataset,
    out: []f64,
    means: []const f64,
    paths: []PathElement,
    per_worker: usize,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        const self: *ShapCtx = @ptrCast(@alignCast(ctx));
        const m = self.m;
        const f = self.ds.n_features;
        const per_out = f + 1;
        const path = self.paths[worker * self.per_worker ..][0..self.per_worker];
        const share: f64 = if (m.average) 1 / @as(f64, @floatFromInt(m.trees.len)) else 1;
        for (begin..end) |r| {
            const row_out = self.out[r * m.width * per_out ..][0 .. m.width * per_out];
            @memset(row_out, 0);
            const rb = self.ds.bins_rm[r * f ..][0..f];
            for (m.trees, self.means, 0..) |*t, mean, i| {
                const phi = row_out[(i % m.width) * per_out ..][0..per_out];
                recurse(t, m.cover, rb, phi, share, path, 0, 0, 0, 1, 1, -1);
                phi[f] += share * mean;
            }
            for (0..m.width) |c| row_out[c * per_out + f] += m.base[c];
        }
    }
};

/// The depth of a complete tree whose nodes at each level all test the same split (a symmetric
/// tree, as `symmetric.zig` builds them), or null.
fn obliviousDepth(t: tree.Tree) ?usize {
    const n = t.nodes.len;
    if (n < 3 or !std.math.isPowerOfTwo(n + 1)) return null;
    const d = std.math.log2_int(usize, n + 1) - 1;
    for (0..d) |level| {
        const first = (@as(usize, 1) << @intCast(level)) - 1;
        const a = t.nodes[first];
        for (t.nodes[first .. 2 * first + 1]) |b| {
            if (b.is_leaf or b.kind != a.kind or b.feature != a.feature or b.threshold != a.threshold or
                b.is_cat != a.is_cat or b.missing_left != a.missing_left or b.cat_ofs != a.cat_ofs) return null;
        }
    }
    for (t.nodes[(@as(usize, 1) << @intCast(d)) - 1 ..]) |leaf| if (!leaf.is_leaf) return null;
    return d;
}

/// A symmetric tree of depth `d` with its levels in reverse order: the same split per level, the
/// leaf at root-to-leaf position `p` moved to `p` with its bits reversed, and internal node
/// statistics summed again from the leaves. Same predictions; caller frees `nodes` and `stats`.
fn reversed(gpa: std.mem.Allocator, t: tree.Tree, d: usize) !tree.Tree {
    const n = t.nodes.len;
    const nodes = try gpa.alloc(tree.Node, n);
    errdefer gpa.free(nodes);
    const stats = try gpa.alloc(tree.NodeStat, n);
    errdefer gpa.free(stats);
    const n_internal = (@as(usize, 1) << @intCast(d)) - 1;
    for (0..n_internal) |i| {
        const level = std.math.log2_int(usize, i + 1);
        nodes[i] = t.nodes[(@as(usize, 1) << @intCast(d - 1 - level)) - 1];
        nodes[i].left = @intCast(2 * i + 1);
        nodes[i].right = @intCast(2 * i + 2);
    }
    for (0..n_internal + 1) |p| {
        const q = @bitReverse(p) >> @intCast(@bitSizeOf(usize) - d);
        nodes[n_internal + q] = t.nodes[n_internal + p];
        stats[n_internal + q] = t.stats[n_internal + p];
    }
    var i = n_internal;
    while (i > 0) {
        i -= 1;
        const l = stats[2 * i + 1];
        const r = stats[2 * i + 2];
        stats[i] = .{ .gain = 0, .hess = l.hess + r.hess, .count = l.count + r.count };
    }
    return .{ .nodes = nodes, .stats = stats, .cat_ids = t.cat_ids, .lin = t.lin, .combos = t.combos };
}

fn depthOf(t: tree.Tree) usize {
    return depthFrom(t, 0);
}

fn depthFrom(t: tree.Tree, i: u32) usize {
    const n = t.nodes[i];
    if (n.is_leaf) return 0;
    return 1 + @max(depthFrom(t, n.left), depthFrom(t, n.right));
}

/// The cover-weighted mean of the leaves below node `i`.
fn nodeMean(t: tree.Tree, i: u32, cover: Cover) f64 {
    const n = t.nodes[i];
    if (n.is_leaf) return n.weight;
    const cl = coverOf(t.stats[n.left], cover);
    const cr = coverOf(t.stats[n.right], cover);
    if (cl + cr == 0) return 0;
    return (nodeMean(t, n.left, cover) * cl + nodeMean(t, n.right, cover) * cr) / (cl + cr);
}

/// The child row `rb` goes to, as `Tree.predictBinned` decides it.
fn hotChild(t: *const tree.Tree, n: tree.Node, rb: []const data.BinIdx) u32 {
    const b = rb[n.feature];
    const go_left = if (b == 0)
        n.missing_left
    else if (n.is_cat)
        split.catContains(t.cat_ids[n.cat_ofs..][0..n.n_cat], b)
    else
        b <= n.threshold;
    return if (go_left) n.left else n.right;
}

/// Add a split to the path: its feature joins with the fractions of rows that reach the child
/// when the feature is unknown (`zero`) and known (`one`), and every subset weight is updated.
fn extend(path: []PathElement, depth: usize, zero: f64, one: f64, feature: i64) void {
    path[depth] = .{ .feature = feature, .zero = zero, .one = one, .weight = if (depth == 0) 1 else 0 };
    const d1: f64 = @floatFromInt(depth + 1);
    var i = depth;
    while (i > 0) {
        i -= 1;
        const fi: f64 = @floatFromInt(i);
        const fd: f64 = @floatFromInt(depth);
        path[i + 1].weight += one * path[i].weight * (fi + 1) / d1;
        path[i].weight = zero * path[i].weight * (fd - fi) / d1;
    }
}

/// Undo `extend` for the element at `at`: a feature split on again further down.
fn unwind(path: []PathElement, depth: usize, at: usize) void {
    const one = path[at].one;
    const zero = path[at].zero;
    var next = path[depth].weight;
    const d1: f64 = @floatFromInt(depth + 1);
    const fd: f64 = @floatFromInt(depth);
    var i = depth;
    while (i > 0) {
        i -= 1;
        const fi: f64 = @floatFromInt(i);
        if (one != 0) {
            const tmp = path[i].weight;
            path[i].weight = next * d1 / ((fi + 1) * one);
            next = tmp - path[i].weight * zero * (fd - fi) / d1;
        } else {
            path[i].weight = path[i].weight * d1 / (zero * (fd - fi));
        }
    }
    for (at..depth) |j| {
        path[j].feature = path[j + 1].feature;
        path[j].zero = path[j + 1].zero;
        path[j].one = path[j + 1].one;
    }
}

/// The total subset weight the path would have without the element at `at`, without unwinding.
fn unwoundSum(path: []const PathElement, depth: usize, at: usize) f64 {
    const one = path[at].one;
    const zero = path[at].zero;
    var next = path[depth].weight;
    var total: f64 = 0;
    const d1: f64 = @floatFromInt(depth + 1);
    const fd: f64 = @floatFromInt(depth);
    var i = depth;
    while (i > 0) {
        i -= 1;
        const fi: f64 = @floatFromInt(i);
        if (one != 0) {
            const tmp = next * d1 / ((fi + 1) * one);
            total += tmp;
            next = path[i].weight - tmp * zero * (fd - fi) / d1;
        } else if (zero != 0) {
            total += path[i].weight / zero / ((fd - fi) / d1);
        }
    }
    return total;
}

/// Lundberg's recursion at node `node`. Paths share one buffer: this call's path starts at
/// `buf[off..]`, right after its parent's (`buf[off - depth ..][0..depth]`), and begins as a copy of
/// it, so each sibling starts from the same parent path (XGBoost's `TreeShap`).
fn recurse(
    t: *const tree.Tree,
    cover: Cover,
    rb: []const data.BinIdx,
    phi: []f64,
    share: f64,
    buf: []PathElement,
    off: usize,
    node: u32,
    depth_in: usize,
    zero: f64,
    one: f64,
    feature: i64,
) void {
    const path = buf[off..];
    var depth = depth_in;
    if (depth != 0) @memcpy(path[0..depth], buf[off - depth ..][0..depth]);
    extend(path, depth, zero, one, feature);

    const n = t.nodes[node];
    if (n.is_leaf) {
        const value: f64 = n.weight;
        for (1..depth + 1) |i| {
            const w = unwoundSum(path, depth, i);
            const el = path[i];
            phi[@intCast(el.feature)] += w * (el.one - el.zero) * value * share;
        }
        return;
    }

    const hot = hotChild(t, n, rb);
    const cold = if (hot == n.left) n.right else n.left;
    // A branch no training row reached (symmetric trees have them) takes no share of the
    // unknown-feature expectation: its fractions are 0, not 0/0.
    const c = coverOf(t.stats[node], cover);
    const hot_zero = if (c > 0) coverOf(t.stats[hot], cover) / c else 0;
    const cold_zero = if (c > 0) coverOf(t.stats[cold], cover) / c else 0;
    var in_zero: f64 = 1;
    var in_one: f64 = 1;

    // A feature already on the path is taken off it first: its fractions carry into the children.
    const f: i64 = n.feature;
    var at: usize = 0;
    while (at <= depth) : (at += 1) {
        if (path[at].feature == f) break;
    }
    if (at <= depth) {
        in_zero = path[at].zero;
        in_one = path[at].one;
        unwind(path, depth, at);
        depth -= 1;
    }
    // A child the row can reach neither with the feature known nor unknown adds nothing: every
    // subset weight below it is 0. Skipping it also keeps `unwind` from dividing by a zero fraction.
    if (hot_zero * in_zero != 0 or in_one != 0)
        recurse(t, cover, rb, phi, share, buf, off + depth + 1, hot, depth + 1, hot_zero * in_zero, in_one, f);
    if (cold_zero * in_zero != 0)
        recurse(t, cover, rb, phi, share, buf, off + depth + 1, cold, depth + 1, cold_zero * in_zero, 0, f);
}

// ----------------------------------------------------------- linear SHAP

/// The linear model's SHAP values: each design column's coefficient times its distance from the
/// training mean, summed over a feature's columns; the bias is the intercept. Exact for a linear
/// model with independent features. A standardised design (the default) is centred on the training
/// means, so the distance is the column's value; without standardisation the means were not kept.
pub fn linearShap(m: *const linear.Linear, ds: *const Dataset, out: []f64) !void {
    const f = ds.n_features;
    const k = m.width();
    std.debug.assert(out.len == ds.n_rows * k * (f + 1));
    const d = &m.design;
    for (d.cols) |c| if (c.scale == 1 and c.center == 0) return error.ShapNeedsStandardizedLinear;
    const p = d.cols.len;
    @memset(out, 0);
    for (0..ds.n_rows) |r| for (0..k) |cls| {
        const phi = out[(r * k + cls) * (f + 1) ..][0 .. f + 1];
        const w = m.w[cls * p ..][0..p];
        for (d.cols, w) |c, coef| phi[c.feature] += @as(f64, coef) * d.value(c, ds, r);
        phi[f] = if (k > 1) m.intercepts[cls] else m.intercept;
    };
}

/// SHAP values for any saved model (`model.Bundle`), laid out as `treeShap`'s.
pub fn bundleShap(gpa: std.mem.Allocator, pool: *Pool, b: *const model.Bundle, ds: *const Dataset, cover: Cover, out: []f64) !void {
    switch (b.kind) {
        .linear => try linearShap(&b.lin.?, ds, out),
        .gbdt, .forest => {
            const k = b.width();
            const base = try gpa.alloc(f32, k);
            defer gpa.free(base);
            if (b.class_base.len != 0) @memcpy(base, b.class_base) else @memset(base, if (b.kind == .gbdt) b.base_score else 0);
            try treeShap(gpa, pool, .{ .trees = b.trees, .width = k, .base = base, .average = b.kind == .forest, .cover = cover }, ds, out);
        },
    }
}
