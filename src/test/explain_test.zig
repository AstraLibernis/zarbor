// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Importance and SHAP. Agreement with XGBoost's `pred_contribs`, LightGBM's `pred_contrib` and
//! CatBoost's `ShapValues` on the same trees is checked by `bench/parity/parity.py`; these pin the
//! algebra on hand-built trees, the sum to the raw score for every model kind, and thread
//! independence.

const std = @import("std");
const data = @import("../data.zig");
const config = @import("../config.zig");
const booster = @import("../booster.zig");
const forest = @import("../forest.zig");
const linear = @import("../linear.zig");
const tree = @import("../tree.zig");
const model_mod = @import("../model.zig");
const explain = @import("../explain.zig");
const Pool = @import("../pool.zig").Pool;
const fromBins = @import("split_test.zig").fromBins;

const testing = std.testing;

const n_rows = 1600;

/// Three 8-level features with signal in the first two (the third is noise), labels 0/1.
fn fixture(gpa: std.mem.Allocator, seed: u64) !data.Dataset {
    var cols: [3][n_rows]u8 = undefined;
    var y: [n_rows]f32 = undefined;
    var prng: std.Random.DefaultPrng = .init(seed);
    const r = prng.random();
    for (0..n_rows) |i| {
        for (&cols) |*c| c[i] = r.intRangeAtMost(u8, 1, 8);
        const s = @as(f32, @floatFromInt(cols[0][i])) * 0.5 - @as(f32, @floatFromInt(cols[1][i])) * 0.3 + r.floatNorm(f32);
        y[i] = if (s > 0.6) 1 else 0;
    }
    return fromBins(gpa, &.{ &cols[0], &cols[1], &cols[2] }, &.{ 9, 9, 9 }, &y);
}

fn leaf(w: f32) tree.Node {
    return .{ .is_leaf = true, .weight = w };
}

fn splitOn(f: u32, t: u16, l: u32, r: u32) tree.Node {
    return .{ .feature = f, .threshold = t, .is_leaf = false, .left = l, .right = r };
}

test "a stump: the row's leaf less the cover-weighted mean, and that mean in the bias" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 1);
    defer pool.deinit();
    var ds = try fixture(gpa, 1);
    defer ds.deinit();
    var nodes = [_]tree.Node{ splitOn(0, 4, 1, 2), leaf(-1.5), leaf(2.5) };
    var stats = [_]tree.NodeStat{ .{ .gain = 9, .hess = 40, .count = 400 }, .{ .hess = 10, .count = 300 }, .{ .hess = 30, .count = 100 } };
    const t = tree.Tree{ .nodes = &nodes, .stats = &stats };
    for ([_]explain.Cover{ .hessian, .count }) |cover| {
        const cl: f64 = if (cover == .hessian) 10 else 300;
        const cr: f64 = if (cover == .hessian) 30 else 100;
        const mean = (-1.5 * cl + 2.5 * cr) / (cl + cr);
        const phi = try gpa.alloc(f64, n_rows * 4);
        defer gpa.free(phi);
        try explain.treeShap(gpa, pool, .{ .trees = &.{t}, .width = 1, .base = &.{0.25}, .average = false, .cover = cover }, &ds, phi);
        for (0..n_rows) |r| {
            const row = phi[r * 4 ..][0..4];
            const own: f64 = if (ds.columnNarrow(0)[r] <= 4) -1.5 else 2.5;
            try testing.expectApproxEqAbs(own - mean, row[0], 1e-12);
            try testing.expectEqual(@as(f64, 0), row[1]);
            try testing.expectEqual(@as(f64, 0), row[2]);
            try testing.expectApproxEqAbs(0.25 + mean, row[3], 1e-12);
        }
    }
}

test "a branch no training row reached adds nothing and makes no NaN" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 1);
    defer pool.deinit();
    var ds = try fixture(gpa, 2);
    defer ds.deinit();
    // Root on feature 0; its left child on feature 1 has an empty left leaf.
    var nodes = [_]tree.Node{ splitOn(0, 4, 1, 2), splitOn(1, 3, 3, 4), leaf(0.7), leaf(5), leaf(-0.4) };
    var stats = [_]tree.NodeStat{
        .{ .gain = 1, .hess = 50, .count = 50 }, .{ .gain = 1, .hess = 20, .count = 20 }, .{ .hess = 30, .count = 30 },
        .{ .hess = 0, .count = 0 },              .{ .hess = 20, .count = 20 },
    };
    const t = tree.Tree{ .nodes = &nodes, .stats = &stats };
    const phi = try gpa.alloc(f64, n_rows * 4);
    defer gpa.free(phi);
    try explain.treeShap(gpa, pool, .{ .trees = &.{t}, .width = 1, .base = &.{0}, .average = false, .cover = .count }, &ds, phi);
    for (0..n_rows) |r| {
        var sum: f64 = 0;
        for (phi[r * 4 ..][0..4]) |v| {
            try testing.expect(std.math.isFinite(v));
            sum += v;
        }
        try testing.expectApproxEqAbs(@as(f64, t.predictBinned(&ds, r)), sum, 1e-6);
    }
}

/// Train `cfg`, then check every row's SHAP values and bias sum to its raw score.
fn sumsToRaw(gpa: std.mem.Allocator, pool: *Pool, ds: *const data.Dataset, cfg: config.Config) !void {
    var res = try @import("../fitted.zig").Fitted.train(gpa, pool, ds, null, cfg, null);
    defer res.model.deinit();
    var b: model_mod.Bundle = switch (res.model) {
        .gbdt => |*x| try model_mod.fromBooster(gpa, x, try data.Schema.fromDataset(gpa, ds)),
        .random_forest => |*x| try model_mod.fromForest(gpa, x, try data.Schema.fromDataset(gpa, ds)),
        .linear => |x| .{ .gpa = gpa, .kind = .linear, .schema = try data.Schema.fromDataset(gpa, ds), .objective = x.objective, .num_class = x.num_class, .lin = x },
    };
    defer {
        if (b.kind == .linear) b.lin = null;
        b.deinit();
    }
    const k = b.width();
    const per = ds.n_features + 1;
    const phi = try gpa.alloc(f64, ds.n_rows * k * per);
    defer gpa.free(phi);
    try explain.bundleShap(gpa, pool, &b, ds, .hessian, phi);
    const raw = try gpa.alloc(f32, ds.n_rows * k);
    defer gpa.free(raw);
    b.predictRaw(pool, ds, raw);
    for (raw, 0..) |want, i| {
        var sum: f64 = 0;
        for (phi[i * per ..][0..per]) |v| sum += v;
        try testing.expectApproxEqAbs(@as(f64, want), sum, 1e-4 * @max(1, @abs(@as(f64, want))));
    }
}

test "every model kind's SHAP values and bias sum to its raw score" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 3);
    defer pool.deinit();
    var ds = try fixture(gpa, 3);
    defer ds.deinit();
    try sumsToRaw(gpa, pool, &ds, config.Config.from(.{ .n_rounds = 20, .max_depth = 5, .verbose_eval = 0 }));
    try sumsToRaw(gpa, pool, &ds, config.Config.from(.{ .n_rounds = 20, .grow_policy = .lossguide, .max_leaves = 15, .max_depth = 0, .subsample = 0.7, .verbose_eval = 0 }));
    // Symmetric trees (explained with their levels reversed) at a depth that leaves empty leaves.
    try sumsToRaw(gpa, pool, &ds, config.Config.from(.{ .n_rounds = 20, .grow_policy = .symmetric, .max_depth = 7, .verbose_eval = 0 }));
    try sumsToRaw(gpa, pool, &ds, config.Config.from(.{ .algo = .random_forest, .n_rounds = 15, .verbose_eval = 0 }));
    try sumsToRaw(gpa, pool, &ds, config.Config.from(.{ .algo = .linear, .verbose_eval = 0 }));
    try sumsToRaw(gpa, pool, &ds, config.Config.from(.{ .objective = .squared_error, .n_rounds = 15, .verbose_eval = 0 }));
    var soft = config.Config.from(.{ .objective = .softmax, .n_rounds = 10, .max_depth = 4, .verbose_eval = 0 });
    for (ds.labels, 0..) |*y, i| y.* = @floatFromInt(@as(u32, @intFromFloat(y.*)) + @as(u32, @intCast(i % 2)));
    soft.setNumClass(3);
    try sumsToRaw(gpa, pool, &ds, soft);
    soft.algo = .linear;
    try sumsToRaw(gpa, pool, &ds, soft);
}

test "SHAP values are the same to the bit at 1, 3 and 16 threads; an unused feature gets 0" {
    const gpa = testing.allocator;
    // The fixture's three features plus a constant one, which no tree can split on.
    var base = try fixture(gpa, 4);
    defer base.deinit();
    var cols: [4][n_rows]u8 = undefined;
    for (0..3) |f| @memcpy(&cols[f], base.columnNarrow(f));
    @memset(&cols[3], 1);
    var ds = try fromBins(gpa, &.{ &cols[0], &cols[1], &cols[2], &cols[3] }, &.{ 9, 9, 9, 2 }, base.labels);
    defer ds.deinit();
    const per = ds.n_features + 1;
    var want: [n_rows * 5]f64 = undefined;
    for ([_]u32{ 1, 3, 16 }) |threads| {
        const pool = try Pool.init(gpa, threads);
        defer pool.deinit();
        var res = try booster.train(gpa, pool, &ds, null, config.Config.from(.{ .n_rounds = 10, .max_depth = 4, .verbose_eval = 0 }).gbdt, null);
        defer res.model.deinit();
        var got: [n_rows * 5]f64 = undefined;
        try explain.treeShap(gpa, pool, .{ .trees = res.model.trees.items, .width = 1, .base = &.{res.model.base_score}, .average = false, .cover = .hessian }, &ds, &got);
        if (threads == 1) want = got else try testing.expectEqualSlices(f64, &want, &got);

        var imp: [4]explain.Importance = undefined;
        try explain.importance(res.model.trees.items, .hessian, &imp);
        try testing.expectEqual(@as(u32, 0), imp[3].splits);
        for (0..n_rows) |r| try testing.expectEqual(@as(f64, 0), got[r * per + 3]);
    }
}

test "importance sums every split's stored gain and cover, and counts the splits" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();
    var ds = try fixture(gpa, 5);
    defer ds.deinit();
    var res = try booster.train(gpa, pool, &ds, null, config.Config.from(.{ .n_rounds = 12, .max_depth = 4, .min_split_gain = 0.5, .verbose_eval = 0 }).gbdt, null);
    defer res.model.deinit();
    var imp: [3]explain.Importance = undefined;
    try explain.importance(res.model.trees.items, .count, &imp);
    var gain: f64 = 0;
    var cover: f64 = 0;
    var splits: u32 = 0;
    for (res.model.trees.items) |t| for (t.nodes, t.stats) |n, st| {
        if (n.is_leaf) continue;
        // The stored gain is the split's improvement before `min_split_gain` comes off: at least it.
        try testing.expect(st.gain >= 0.5);
        // A node's rows and hessian are its children's.
        try testing.expectApproxEqRel(st.count, t.stats[n.left].count + t.stats[n.right].count, 1e-6);
        try testing.expectApproxEqRel(st.hess, t.stats[n.left].hess + t.stats[n.right].hess, 1e-5);
        gain += st.gain;
        cover += st.count;
        splits += 1;
    };
    var g2: f64 = 0;
    var c2: f64 = 0;
    var s2: u32 = 0;
    for (imp) |x| {
        g2 += x.gain;
        c2 += x.cover;
        s2 += x.splits;
    }
    try testing.expectApproxEqRel(gain, g2, 1e-9);
    try testing.expectApproxEqRel(cover, c2, 1e-9);
    try testing.expectEqual(splits, s2);
    // The noise feature is used least.
    try testing.expect(imp[2].gain < imp[0].gain and imp[2].gain < imp[1].gain);
}

test "a model without node statistics is refused, not explained with zeros" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 1);
    defer pool.deinit();
    var ds = try fixture(gpa, 6);
    defer ds.deinit();
    var nodes = [_]tree.Node{ splitOn(0, 4, 1, 2), leaf(-1), leaf(1) };
    const t = tree.Tree{ .nodes = &nodes };
    const phi = try gpa.alloc(f64, n_rows * 4);
    defer gpa.free(phi);
    try testing.expectError(error.ModelHasNoNodeStats, explain.treeShap(gpa, pool, .{ .trees = &.{t}, .width = 1, .base = &.{0}, .average = false, .cover = .count }, &ds, phi));
    var imp: [3]explain.Importance = undefined;
    try testing.expectError(error.ModelHasNoNodeStats, explain.importance(&.{t}, .count, &imp));
}

/// The path-dependent expectation of a tree given only the features in `known` (bit f set):
/// follow a known feature's split as the row does, average an unknown one's children by cover.
/// A node no training row reached weighs nothing. Algorithm 1 of the TreeSHAP paper.
fn expectGiven(t: *const tree.Tree, i: u32, ds: *const data.Dataset, r: usize, known: u32) f64 {
    const n = t.nodes[i];
    if (n.is_leaf) return n.weight;
    if (known & (@as(u32, 1) << @intCast(n.feature)) != 0) {
        const b = ds.columnNarrow(n.feature)[r];
        return expectGiven(t, if (b <= n.threshold) n.left else n.right, ds, r, known);
    }
    const c = t.stats[i].hess;
    if (c == 0) return 0;
    return (t.stats[n.left].hess * expectGiven(t, n.left, ds, r, known) +
        t.stats[n.right].hess * expectGiven(t, n.right, ds, r, known)) / c;
}

test "TreeSHAP equals the Shapley values computed by enumerating every feature subset" {
    // Brute force over all 2^M subsets with the same expectation: any slip in the path
    // bookkeeping (a feature met twice on a path, the subset weights) changes some value here,
    // even where the row's values still sum to its score.
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();
    var ds = try fixture(gpa, 7);
    defer ds.deinit();
    // Deep trees on three features: each path tests some feature more than once.
    var res = try booster.train(gpa, pool, &ds, null, config.Config.from(.{ .n_rounds = 3, .max_depth = 6, .min_child_samples = 5, .verbose_eval = 0 }).gbdt, null);
    defer res.model.deinit();
    const m = 3;
    const per = m + 1;
    const phi = try gpa.alloc(f64, n_rows * per);
    defer gpa.free(phi);
    for (res.model.trees.items) |*t| {
        var repeats = false;
        for (t.nodes) |n| if (!n.is_leaf and !t.nodes[n.left].is_leaf and t.nodes[n.left].feature == n.feature) {
            repeats = true;
        };
        try testing.expect(repeats);
        try explain.treeShap(gpa, pool, .{ .trees = &.{t.*}, .width = 1, .base = &.{0}, .average = false, .cover = .hessian }, &ds, phi);
        for (0..200) |r| for (0..m) |f| {
            var want: f64 = 0;
            const bit = @as(u32, 1) << @intCast(f);
            for (0..(1 << m)) |s| {
                const set: u32 = @intCast(s);
                if (set & bit != 0) continue;
                const size = @popCount(set);
                // |S|! (M - |S| - 1)! / M!
                const fact = [_]f64{ 1, 1, 2, 6 };
                const w = fact[size] * fact[m - size - 1] / fact[m];
                want += w * (expectGiven(t, 0, &ds, r, set | bit) - expectGiven(t, 0, &ds, r, set));
            }
            try testing.expectApproxEqAbs(want, phi[r * per + f], 1e-9);
        };
    }
}

test "symmetric trees' node statistics count every training row" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();
    var ds = try fixture(gpa, 8);
    defer ds.deinit();
    var res = try booster.train(gpa, pool, &ds, null, config.Config.from(.{ .n_rounds = 5, .grow_policy = .symmetric, .max_depth = 4, .verbose_eval = 0 }).gbdt, null);
    defer res.model.deinit();
    for (res.model.trees.items) |t| {
        try testing.expectEqual(@as(f32, n_rows), t.stats[0].count);
        try testing.expect(t.stats[0].hess > 0);
        try testing.expect(t.stats[0].gain > 0);
    }
}
