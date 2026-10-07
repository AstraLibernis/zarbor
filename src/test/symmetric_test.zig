// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Symmetric (CatBoost-style) trees. Row-for-row parity with catboost 1.2.10 is measured outside
//! the unit tests (docs/catboost.md); these pin the pieces a wrong change would break silently:
//! the shared split per depth, the leaf-index mapping, Newton leaves, the score functions and the
//! redundant-split stop.

const std = @import("std");
const data = @import("../data.zig");
const config = @import("../config.zig");
const booster = @import("../booster.zig");
const tree = @import("../tree.zig");
const Pool = @import("../pool.zig").Pool;
const fromBins = @import("split_test.zig").fromBins;

const testing = std.testing;

const n_rows = 1500;

/// Three 6-bin features (bins 1..5, no missing) with signal in the first two.
fn fixture(gpa: std.mem.Allocator, seed: u64) !data.Dataset {
    var cols: [3][n_rows]u8 = undefined;
    var y: [n_rows]f32 = undefined;
    var prng: std.Random.DefaultPrng = .init(seed);
    const r = prng.random();
    for (0..n_rows) |i| {
        for (&cols) |*c| c[i] = r.intRangeAtMost(u8, 1, 5);
        const s: f32 = @as(f32, @floatFromInt(cols[0][i])) * 0.6 - @as(f32, @floatFromInt(cols[1][i])) * 0.4 + r.floatNorm(f32);
        y[i] = if (s > 0.5) 1 else 0;
    }
    return fromBins(gpa, &.{ &cols[0], &cols[1], &cols[2] }, &.{ 6, 6, 6 }, &y);
}

fn train(gpa: std.mem.Allocator, pool: *Pool, ds: *const data.Dataset, opts: anytype) !booster.Model {
    var cfg = config.Config.from(.{ .grow_policy = .symmetric, .verbose_eval = 0, .base_score = 0 }).gbdt;
    inline for (@typeInfo(@TypeOf(opts)).@"struct".fields) |f| {
        if (@hasField(booster.Params, f.name)) @field(cfg, f.name) = @field(opts, f.name) else @field(cfg.tree, f.name) = @field(opts, f.name);
    }
    const res = try booster.train(gpa, pool, ds, null, cfg, null);
    return res.model;
}

fn binOf(ds: *const data.Dataset, f: usize, r: usize) data.BinIdx {
    return ds.columnNarrow(f)[r];
}

/// The split each depth uses, read from the first node of that depth, after checking every node
/// at the depth agrees.
fn levelSplits(t: tree.Tree, out: [][2]u32) !usize {
    var depth: usize = 0;
    while ((@as(usize, 1) << @intCast(depth)) - 1 < t.nodes.len and !t.nodes[(@as(usize, 1) << @intCast(depth)) - 1].is_leaf) : (depth += 1) {
        const first = (@as(usize, 1) << @intCast(depth)) - 1;
        const n0 = t.nodes[first];
        for (t.nodes[first .. 2 * first + 1]) |nd| {
            try testing.expect(!nd.is_leaf);
            try testing.expectEqual(n0.feature, nd.feature);
            try testing.expectEqual(n0.threshold, nd.threshold);
        }
        out[depth] = .{ n0.feature, n0.threshold };
    }
    return depth;
}

test "every depth shares one split, and leaves are Newton steps over the rows they select" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 3);
    defer pool.deinit();
    var ds = try fixture(gpa, 1);
    defer ds.deinit();

    const lr: f64 = 0.5;
    const lambda: f64 = 2;
    var m = try train(gpa, pool, &ds, .{ .n_rounds = 1, .max_depth = 3, .learning_rate = @as(f32, @floatCast(lr)), .lambda = @as(f32, @floatCast(lambda)) });
    defer m.deinit();
    try testing.expectEqual(@as(usize, 1), m.trees.items.len);

    var sp: [16][2]u32 = undefined;
    const depth = try levelSplits(m.trees.items[0], &sp);
    try testing.expectEqual(@as(usize, 3), depth);

    // Leaf index = sum over depths of [bin > threshold] << depth; base score 0 means p = 0.5,
    // g = 0.5 - y and h = 0.25 for every row.
    var g = [_]f64{0} ** 8;
    var h = [_]f64{0} ** 8;
    var idx: [n_rows]usize = undefined;
    for (0..n_rows) |r| {
        var i: usize = 0;
        for (0..depth) |d| if (binOf(&ds, sp[d][0], r) > sp[d][1]) {
            i |= @as(usize, 1) << @intCast(d);
        };
        idx[r] = i;
        g[i] += 0.5 - ds.labels[r];
        h[i] += 0.25;
    }
    var raw: [n_rows]f32 = undefined;
    m.predictRaw(pool, &ds, &raw);
    for (raw, idx) |got, i| {
        const want = -g[i] / (h[i] + lambda) * lr;
        try testing.expectApproxEqAbs(want, got, 1e-5);
    }
}

/// Brute force over the root's candidates: index of the best (feature, threshold) by `score`.
fn bruteRoot(ds: *const data.Dataset, lambda: f64, score: tree.ScoreFunction) [2]u32 {
    var best: [2]u32 = .{ 0, 0 };
    var best_s = -std.math.inf(f64);
    for (0..ds.n_features) |f| {
        const nb = ds.n_bins[f];
        var k: u32 = 1; // no missing rows, so threshold 0 is not offered
        while (k + 1 < nb) : (k += 1) {
            var gs = [2]f64{ 0, 0 };
            var ns = [2]f64{ 0, 0 };
            for (0..ds.n_rows) |r| {
                const side: usize = @intFromBool(binOf(ds, f, r) > k);
                gs[side] += 0.5 - ds.labels[r];
                ns[side] += 1;
            }
            var num: f64 = 0;
            var den: f64 = 0;
            for (gs, ns) |gg, nn| switch (score) {
                .gain => num += gg * gg / (0.25 * nn + lambda),
                else => {
                    const v = if (nn > 0) gg / (nn + lambda) else 0;
                    num += v * gg;
                    den += v * v * nn;
                },
            };
            const s = if (score == .cosine) num / @sqrt(den + 1e-100) else num;
            if (s > best_s) {
                best_s = s;
                best = .{ @intCast(f), k };
            }
        }
    }
    return best;
}

test "the root split is the argmax of the chosen score function" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();
    // A large lambda pulls the three scores apart.
    const lambda: f32 = 400;
    for (0..4) |seed| {
        var ds = try fixture(gpa, 10 + seed);
        defer ds.deinit();
        for ([_]tree.ScoreFunction{ .cosine, .l2, .gain }) |sf| {
            var m = try train(gpa, pool, &ds, .{ .n_rounds = 1, .max_depth = 1, .lambda = lambda, .score_function = sf });
            defer m.deinit();
            const root = m.trees.items[0].nodes[0];
            const want = bruteRoot(&ds, lambda, sf);
            try testing.expectEqual(want[0], root.feature);
            try testing.expectEqual(want[1], @as(u32, root.threshold));
        }
    }
}

test "a split that separates nothing stops the tree" {
    // One two-valued feature: after splitting on it, any further split leaves one side of every
    // leaf pair empty, so CatBoost's redundancy rule removes it and the tree ends at depth 1.
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();
    var col: [400]u8 = undefined;
    var y: [400]f32 = undefined;
    for (&col, &y, 0..) |*c, *l, i| {
        c.* = if (i % 3 == 0) 1 else 2;
        l.* = if (i % 3 == 0 and i % 2 == 0) 1 else 0;
    }
    var ds = try fromBins(gpa, &.{&col}, &.{3}, &y);
    defer ds.deinit();
    var m = try train(gpa, pool, &ds, .{ .n_rounds = 1, .max_depth = 4 });
    defer m.deinit();
    try testing.expectEqual(@as(usize, 3), m.trees.items[0].nodes.len);
}

test "more Newton steps solve each leaf; squared error needs one" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();
    var ds = try fixture(gpa, 3);
    defer ds.deinit();

    // Logistic, one tree, full step. Each step is -G / (H + lambda) at the moved score, with no
    // lambda * v term in G (CatBoost's walk, matched to 6e-7 in docs/catboost.md), so lambda only
    // damps the steps and they converge to the unregularised leaf optimum, sum (sigmoid(v) - y) = 0
    // over the leaf's rows. One step does not get there.
    var resid: [2]f64 = undefined;
    for ([_]u32{ 1, 30 }, 0..) |iters, which| {
        var m = try train(gpa, pool, &ds, .{ .n_rounds = 1, .max_depth = 2, .learning_rate = @as(f32, 1.0), .lambda = @as(f32, 1.0), .leaf_estimation_iterations = iters });
        defer m.deinit();
        var raw: [n_rows]f32 = undefined;
        m.predictRaw(pool, &ds, &raw);
        // Group rows by their (shared) leaf value.
        var worst: f64 = 0;
        for (raw) |v0| {
            var s: f64 = 0;
            for (raw, ds.labels) |v, yy| if (v == v0) {
                s += 1 / (1 + @exp(-@as(f64, v))) - yy;
            };
            worst = @max(worst, @abs(s));
        }
        resid[which] = worst;
    }
    try testing.expect(resid[0] > 1.0);
    try testing.expect(resid[1] < 1e-2);

    // Squared error without lambda: the Newton step is exact, so further steps change nothing.
    // (With lambda > 0 they move on toward the unregularised mean, as above.)
    var raws: [2][n_rows]f32 = undefined;
    for ([_]u32{ 1, 5 }, 0..) |iters, which| {
        var cfg = config.Config.from(.{ .grow_policy = .symmetric, .verbose_eval = 0, .objective = .squared_error, .n_rounds = 3, .max_depth = 2, .lambda = 0, .leaf_estimation_iterations = iters }).gbdt;
        cfg.base_score = 0;
        var res = try booster.train(gpa, pool, &ds, null, cfg, null);
        defer res.model.deinit();
        res.model.predictRaw(pool, &ds, &raws[which]);
    }
    for (raws[0], raws[1]) |a, b| try testing.expectApproxEqAbs(a, b, 1e-5);
}

test "symmetric refuses options it does not implement" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 1);
    defer pool.deinit();
    var ds = try fixture(gpa, 4);
    defer ds.deinit();
    const bad = [_]config.Config{
        config.Config.from(.{ .grow_policy = .symmetric, .subsample = 0.5 }),
        config.Config.from(.{ .grow_policy = .symmetric, .colsample_bytree = 0.5 }),
        config.Config.from(.{ .grow_policy = .symmetric, .cat_split = .optimal }),
        config.Config.from(.{ .grow_policy = .symmetric, .sampling = .goss }),
        config.Config.from(.{ .grow_policy = .symmetric, .max_depth = 0 }),
    };
    for (bad) |c| {
        if (booster.train(gpa, pool, &ds, null, c.gbdt, null)) |res| {
            var m = res.model;
            m.deinit();
            return error.TestExpectedError;
        } else |_| {}
    }
}

test "an exact tie goes to the lower feature index" {
    // Two identical columns score identically at every threshold; CatBoost keeps the first
    // maximum in feature order, and so must the tree.
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();
    var col: [600]u8 = undefined;
    var y: [600]f32 = undefined;
    for (&col, &y, 0..) |*c, *l, i| {
        c.* = @intCast(1 + i % 4);
        l.* = if (i % 4 >= 2 and i % 7 != 0) 1 else 0;
    }
    var ds = try fromBins(gpa, &.{ &col, &col }, &.{ 5, 5 }, &y);
    defer ds.deinit();
    var m = try train(gpa, pool, &ds, .{ .n_rounds = 1, .max_depth = 1 });
    defer m.deinit();
    try testing.expectEqual(@as(u32, 0), m.trees.items[0].nodes[0].feature);
}

const symmetric = @import("../symmetric.zig");

test "the MVS threshold solves sum min(1, c / mu) = sample" {
    var prng: std.Random.DefaultPrng = .init(21);
    const r = prng.random();
    var buf: [8192]f64 = undefined;
    for ([_]usize{ 1, 7, 100, 8192 }) |n| {
        for ([_]f64{ 0.05, 0.5, 0.8, 0.999 }) |rate| {
            const c = buf[0..n];
            // Heavy-tailed candidates, so some rows sit above the threshold.
            for (c) |*x| x.* = 0.01 + r.float(f64) * r.float(f64) * r.float(f64) * 50;
            const sample = rate * @as(f64, @floatFromInt(n));
            const mu = symmetric.mvsThreshold(c, sample);
            var s: f64 = 0;
            for (c) |x| s += if (mu == 0 or x > mu) 1 else x / mu;
            try testing.expectApproxEqRel(sample, s, 1e-9);
        }
    }
    // Equal candidates: every row gets the same probability.
    var eq = [_]f64{2} ** 10;
    const mu = symmetric.mvsThreshold(&eq, 4);
    try testing.expectApproxEqRel(@as(f64, 5), mu, 1e-12);
}

fn predictWith(gpa: std.mem.Allocator, threads: u32, ds: *const data.Dataset, opts: anytype) ![n_rows]f32 {
    const pool = try Pool.init(gpa, threads);
    defer pool.deinit();
    var m = try train(gpa, pool, ds, opts);
    defer m.deinit();
    var raw: [n_rows]f32 = undefined;
    m.predictRaw(pool, ds, &raw);
    return raw;
}

test "bootstrap and noise give the same model at 1, 3 and 16 threads" {
    const gpa = testing.allocator;
    var ds = try fixture(gpa, 6);
    defer ds.deinit();
    inline for (.{ .mvs, .bernoulli, .bayesian }) |bt| {
        const opts = .{ .n_rounds = 8, .max_depth = 4, .bootstrap_type = bt, .subsample = @as(f32, if (bt == .bayesian) 1.0 else 0.6), .random_strength = @as(f32, 1.0), .seed = @as(u64, 7) };
        const one = try predictWith(gpa, 1, &ds, opts);
        for ([_]u32{ 3, 16 }) |t| {
            const other = try predictWith(gpa, t, &ds, opts);
            try testing.expectEqualSlices(f32, &one, &other);
        }
    }
}

test "bootstrap weights reach split scoring only, never the leaves" {
    // Same check as the first test, under Bayesian weights: whatever splits were chosen, every
    // leaf is the full-data, unweighted Newton step over its rows.
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();
    var ds = try fixture(gpa, 8);
    defer ds.deinit();
    var m = try train(gpa, pool, &ds, .{ .n_rounds = 1, .max_depth = 3, .learning_rate = @as(f32, 1.0), .lambda = @as(f32, 2.0), .bootstrap_type = .bayesian, .bagging_temperature = @as(f32, 3.0), .seed = @as(u64, 5) });
    defer m.deinit();
    var sp: [16][2]u32 = undefined;
    const depth = try levelSplits(m.trees.items[0], &sp);
    var g = [_]f64{0} ** 8;
    var h = [_]f64{0} ** 8;
    var idx: [n_rows]usize = undefined;
    for (0..n_rows) |r| {
        var i: usize = 0;
        for (0..depth) |d| if (binOf(&ds, sp[d][0], r) > sp[d][1]) {
            i |= @as(usize, 1) << @intCast(d);
        };
        idx[r] = i;
        g[i] += 0.5 - ds.labels[r];
        h[i] += 0.25;
    }
    var raw: [n_rows]f32 = undefined;
    m.predictRaw(pool, &ds, &raw);
    for (raw, idx) |got, i| try testing.expectApproxEqAbs(-g[i] / (h[i] + 2.0), got, 1e-5);
}

test "weights that are all one change nothing; noise changes the model only when on" {
    const gpa = testing.allocator;
    var ds = try fixture(gpa, 9);
    defer ds.deinit();
    const base = try predictWith(gpa, 2, &ds, .{ .n_rounds = 6, .max_depth = 4 });
    // Temperature 0 makes every Bayesian weight 1; MVS at subsample 1 keeps every row at weight 1.
    const bay0 = try predictWith(gpa, 2, &ds, .{ .n_rounds = 6, .max_depth = 4, .bootstrap_type = .bayesian, .bagging_temperature = @as(f32, 0.0) });
    const mvs1 = try predictWith(gpa, 2, &ds, .{ .n_rounds = 6, .max_depth = 4, .bootstrap_type = .mvs });
    try testing.expectEqualSlices(f32, &base, &bay0);
    try testing.expectEqualSlices(f32, &base, &mvs1);
    // And weights that are not all one do reach the splits.
    const bay3 = try predictWith(gpa, 2, &ds, .{ .n_rounds = 6, .max_depth = 4, .bootstrap_type = .bayesian, .bagging_temperature = @as(f32, 3.0) });
    try testing.expect(!std.mem.eql(f32, &base, &bay3));
    // Without noise the seed is irrelevant; with it, two seeds grow different models.
    const s1 = try predictWith(gpa, 2, &ds, .{ .n_rounds = 6, .max_depth = 4, .seed = @as(u64, 1) });
    try testing.expectEqualSlices(f32, &base, &s1);
    const n1 = try predictWith(gpa, 2, &ds, .{ .n_rounds = 6, .max_depth = 4, .random_strength = @as(f32, 5.0), .seed = @as(u64, 1) });
    const n2 = try predictWith(gpa, 2, &ds, .{ .n_rounds = 6, .max_depth = 4, .random_strength = @as(f32, 5.0), .seed = @as(u64, 2) });
    try testing.expect(!std.mem.eql(f32, &n1, &n2));
}

test "a CTR row sees only the rows before it; Counter sees every row" {
    // One categorical, levels 1 2 1 1 2 3, labels 1 0 0 1 1 0. Borders with prior p for row i is
    // (positives before i with i's level + p) / (rows before i with i's level + 1), bucketed by
    // trunc(15 v); Counter is count(level) / (largest count + 1) over all rows.
    const gpa = testing.allocator;
    const col = [_]u8{ 1, 2, 1, 1, 2, 3 };
    const y = [_]f32{ 1, 0, 0, 1, 1, 0 };
    var ds = try fromBins(gpa, &.{&col}, &.{4}, &y);
    defer ds.deinit();
    ds.kinds[0] = .categorical;

    const Case = struct { t: u8, want: [6]u8, final: [4]u8 };
    const cases = [_]Case{
        // prior 0: 0/1, 0/1, 1/2, 1/3, 0/2, 0/1
        .{ .t = 0, .want = .{ 0, 0, 7, 5, 0, 0 }, .final = .{ 0, 7, 5, 0 } },
        // prior 0.5: .5/1, .5/1, 1.5/2, 1.5/3, .5/2, .5/1
        .{ .t = 1, .want = .{ 7, 7, 11, 7, 3, 7 }, .final = .{ 7, 9, 7, 3 } },
        // prior 1: 1/1, 1/1, 2/2, 2/3, 1/2, 1/1
        .{ .t = 2, .want = .{ 15, 15, 15, 10, 7, 15 }, .final = .{ 15, 11, 10, 7 } },
        // Counter: counts 3, 2, 1 of 6 rows; largest 3, so value = count / 4
        .{ .t = 3, .want = .{ 11, 7, 11, 11, 7, 3 }, .final = .{ 0, 11, 7, 3 } },
    };
    for (cases) |c| {
        var got: [6]u8 = undefined;
        var final: [4]u8 = undefined;
        const uniq = try symmetric.ctrColumn(gpa, &ds, 0, c.t, &.{}, &got, &final);
        try testing.expectEqual(@as(u32, 3), uniq);
        try testing.expectEqualSlices(u8, &c.want, &got);
        try testing.expectEqualSlices(u8, &c.final, &final);
    }
}

test "categoricals: one-hot up to one_hot_max_size levels, four CTRs above, one level dropped" {
    const gpa = testing.allocator;
    var a: [12]u8 = undefined; // two levels
    var w: [12]u8 = undefined; // four levels
    var k: [12]u8 = undefined; // one level
    var y: [12]f32 = undefined;
    for (0..12) |i| {
        a[i] = @intCast(1 + i % 2);
        w[i] = @intCast(1 + i % 4);
        k[i] = 1;
        y[i] = @floatFromInt(i % 3 % 2);
    }
    var ds = try fromBins(gpa, &.{ &a, &w, &k, &a }, &.{ 3, 5, 2, 3 }, &y);
    defer ds.deinit();
    ds.kinds[0] = .categorical;
    ds.kinds[1] = .categorical;
    ds.kinds[2] = .categorical;
    // Column 3 stays numeric.
    const cands = try symmetric.candidates(gpa, &ds, .{ .depth = 2, .lambda = 1, .learning_rate = 0.1, .score = .auto, .leaf_iterations = 1, .ctr = true });
    defer symmetric.freeCands(gpa, cands);
    try testing.expectEqual(@as(usize, 6), cands.len);
    try testing.expectEqual(symmetric.CandKind.numeric, cands[0].kind);
    try testing.expectEqual(@as(u32, 3), cands[0].feature);
    try testing.expectEqual(symmetric.CandKind.onehot, cands[1].kind);
    try testing.expectEqual(@as(u32, 0), cands[1].feature);
    for (cands[2..], 0..) |c, t| {
        try testing.expectEqual(symmetric.CandKind.ctr, c.kind);
        try testing.expectEqual(@as(u32, 1), c.feature);
        try testing.expectEqual(@as(u8, @intCast(t)), c.ctr_type);
    }
}

test "ordered boosting's prefixes: body min(100, n/50), each tail twice its body" {
    const gpa = testing.allocator;
    // CatBoost's own example (fold.cpp): n = 3000 gives (60,120), (120,240), ..., (1920,3000).
    const bt = try symmetric.bodyTails(gpa, 3000);
    defer gpa.free(bt);
    const want = [_][2]usize{ .{ 60, 120 }, .{ 120, 240 }, .{ 240, 480 }, .{ 480, 960 }, .{ 960, 1920 }, .{ 1920, 3000 } };
    try testing.expectEqual(want.len, bt.len);
    for (bt, want) |got, w| {
        try testing.expectEqual(w[0], got.body);
        try testing.expectEqual(w[1], got.tail);
    }
    // Small data starts from one row; the last tail is clipped to n.
    const small = try symmetric.bodyTails(gpa, 10);
    defer gpa.free(small);
    try testing.expectEqual(@as(usize, 1), small[0].body);
    try testing.expectEqual(@as(usize, 10), small[small.len - 1].tail);
    try testing.expectEqual(@as(usize, 8), small[small.len - 1].body);
}

test "under ordered boosting the model's leaves are still full-data Newton steps" {
    // Ordering changes which split is chosen and the prefix models behind the scenes; the leaves
    // a prediction uses come from every row at the model's own score, as in plain boosting.
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();
    var ds = try fixture(gpa, 12);
    defer ds.deinit();
    var m = try train(gpa, pool, &ds, .{ .n_rounds = 1, .max_depth = 3, .learning_rate = @as(f32, 1.0), .lambda = @as(f32, 2.0), .boosting_type = .ordered });
    defer m.deinit();
    var sp: [16][2]u32 = undefined;
    const depth = try levelSplits(m.trees.items[0], &sp);
    var g = [_]f64{0} ** 8;
    var h = [_]f64{0} ** 8;
    var idx: [n_rows]usize = undefined;
    for (0..n_rows) |r| {
        var i: usize = 0;
        for (0..depth) |d| if (binOf(&ds, sp[d][0], r) > sp[d][1]) {
            i |= @as(usize, 1) << @intCast(d);
        };
        idx[r] = i;
        g[i] += 0.5 - ds.labels[r];
        h[i] += 0.25;
    }
    var raw: [n_rows]f32 = undefined;
    m.predictRaw(pool, &ds, &raw);
    for (raw, idx) |got, i| try testing.expectApproxEqAbs(-g[i] / (h[i] + 2.0), got, 1e-5);
}

test "ordered boosting gives the same model at 1, 3 and 16 threads, and differs from plain" {
    const gpa = testing.allocator;
    var ds = try fixture(gpa, 13);
    defer ds.deinit();
    const opts = .{ .n_rounds = 12, .max_depth = 4, .boosting_type = .ordered, .bootstrap_type = .mvs, .subsample = @as(f32, 0.7), .random_strength = @as(f32, 1.0), .seed = @as(u64, 3) };
    const one = try predictWith(gpa, 1, &ds, opts);
    for ([_]u32{ 3, 16 }) |t| {
        const other = try predictWith(gpa, t, &ds, opts);
        try testing.expectEqualSlices(f32, &one, &other);
    }
    const plain = try predictWith(gpa, 2, &ds, .{ .n_rounds = 12, .max_depth = 4 });
    const ordered = try predictWith(gpa, 2, &ds, .{ .n_rounds = 12, .max_depth = 4, .boosting_type = .ordered });
    try testing.expect(!std.mem.eql(f32, &plain, &ordered));
}

test "ordered boosting refuses what is not verified" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 1);
    defer pool.deinit();
    var ds = try fixture(gpa, 14);
    defer ds.deinit();
    const bad = [_]config.Config{
        config.Config.from(.{ .grow_policy = .symmetric, .boosting_type = .ordered, .score_function = .gain }),
        config.Config.from(.{ .grow_policy = .symmetric, .boosting_type = .ordered, .leaf_estimation_iterations = 3 }),
        config.Config.from(.{ .boosting_type = .ordered }),
    };
    for (bad) |c| {
        if (booster.train(gpa, pool, &ds, null, c.gbdt, null)) |res| {
            var m = res.model;
            m.deinit();
            return error.TestExpectedError;
        } else |_| {}
    }
}

test "the ordered root split is the argmax of body estimates scored against tails" {
    // Brute force of CatBoost's ordered Cosine at the root, first tree (every prefix model still at
    // the base score 0, so each row's derivative is 0.5 - y): for each prefix, leaf estimates from
    // its body rows, scored against its tail rows, summed over prefixes.
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();
    const lambda: f64 = 3;
    for (0..3) |seed| {
        var ds = try fixture(gpa, 30 + seed);
        defer ds.deinit();
        const bts = try symmetric.bodyTails(gpa, ds.n_rows);
        defer gpa.free(bts);
        var best: [2]u32 = .{ 0, 0 };
        var best_s = -std.math.inf(f64);
        for (0..ds.n_features) |f| {
            var k: u32 = 1;
            while (k + 1 < ds.n_bins[f]) : (k += 1) {
                var num: f64 = 0;
                var den: f64 = 1e-100;
                for (bts) |bt| {
                    var bs = [2]f64{ 0, 0 };
                    var bc = [2]f64{ 0, 0 };
                    var ts = [2]f64{ 0, 0 };
                    var tc = [2]f64{ 0, 0 };
                    for (0..bt.tail) |r| {
                        const side: usize = @intFromBool(binOf(&ds, f, r) > k);
                        const d = 0.5 - @as(f64, ds.labels[r]);
                        if (r < bt.body) {
                            bs[side] += d;
                            bc[side] += 1;
                        } else {
                            ts[side] += d;
                            tc[side] += 1;
                        }
                    }
                    for (0..2) |s| {
                        const v = if (bc[s] > 0) bs[s] / (bc[s] + lambda) else 0;
                        num += v * ts[s];
                        den += v * v * tc[s];
                    }
                }
                const sc = num / @sqrt(den);
                if (sc > best_s) {
                    best_s = sc;
                    best = .{ @intCast(f), k };
                }
            }
        }
        var m = try train(gpa, pool, &ds, .{ .n_rounds = 1, .max_depth = 1, .lambda = @as(f32, 3.0), .boosting_type = .ordered, .has_time = true });
        defer m.deinit();
        const root = m.trees.items[0].nodes[0];
        try testing.expectEqual(best[0], root.feature);
        try testing.expectEqual(best[1], @as(u32, root.threshold));
    }
}

const model_mod = @import("../model.zig");

test "a model with combination splits survives save and load exactly" {
    // Two categoricals whose interaction carries the signal and a numeric column, so the trees use
    // combinations; the bundle path (fromBooster, serialise, deserialise) is the one the CLI takes,
    // and the one that once dropped the combination tables.
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();
    const n = 3000;
    var a: [n]u8 = undefined;
    var c: [n]u8 = undefined;
    var x: [n]u8 = undefined;
    var y: [n]f32 = undefined;
    var prng: std.Random.DefaultPrng = .init(41);
    const r = prng.random();
    for (0..n) |i| {
        a[i] = r.intRangeAtMost(u8, 1, 6);
        c[i] = r.intRangeAtMost(u8, 1, 5);
        x[i] = r.intRangeAtMost(u8, 1, 8);
        const s: f32 = (if ((a[i] + c[i]) % 3 == 0) @as(f32, 1.2) else -0.6) + @as(f32, @floatFromInt(x[i])) * 0.1 + r.floatNorm(f32) * 0.5;
        y[i] = if (s > 0) 1 else 0;
    }
    var ds = try fromBins(gpa, &.{ &a, &c, &x }, &.{ 7, 6, 9 }, &y);
    defer ds.deinit();
    ds.kinds[0] = .categorical;
    ds.kinds[1] = .categorical;
    var m = try train(gpa, pool, &ds, .{ .n_rounds = 30, .max_depth = 4, .cat_split = .ctr, .max_ctr_complexity = 3 });
    defer m.deinit();

    var combos: usize = 0;
    for (m.trees.items) |t| combos += t.combos.len;
    try testing.expect(combos > 0);

    var schema = try data.Schema.fromDataset(gpa, &ds);
    var bundle = model_mod.fromBooster(gpa, &m, schema) catch |e| {
        schema.deinit();
        return e;
    };
    defer bundle.deinit();
    const bytes = try model_mod.serialise(gpa, &bundle);
    defer gpa.free(bytes);
    var back = try model_mod.deserialise(gpa, bytes);
    defer back.deinit();

    var want: [n]f32 = undefined;
    var got: [n]f32 = undefined;
    m.predict(pool, &ds, &want);
    back.predict(pool, &ds, &got);
    try testing.expectEqualSlices(f32, &want, &got);
}

test "a block shuffle permutes whole blocks and keeps each block's order" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(77);
    for ([_][2]usize{ .{ 1000, 7 }, .{ 10, 3 }, .{ 5, 10 }, .{ 256, 1 } }) |nb| {
        const n = nb[0];
        const block = nb[1];
        const out = try symmetric.blockShuffle(gpa, &.{}, n, block, prng.random());
        defer gpa.free(out);
        var seen = try gpa.alloc(bool, n);
        defer gpa.free(seen);
        @memset(seen, false);
        for (out) |r| {
            try testing.expect(!seen[r]);
            seen[r] = true;
        }
        // Inside a run that starts a block, rows follow on by one.
        var pos: usize = 0;
        while (pos < n) {
            const start = out[pos];
            try testing.expect(start % block == 0);
            const len = @min(block, n - start);
            for (0..len) |i| try testing.expectEqual(start + @as(u32, @intCast(i)), out[pos + i]);
            pos += len;
        }
        // And it does move them: 143 blocks left in place would be a 1-in-143! accident.
        if (n == 1000) {
            var moved = false;
            for (out, 0..) |rr, i| moved = moved or rr != i;
            try testing.expect(moved);
        }
    }
}

fn ctrFixture(gpa: std.mem.Allocator) !data.Dataset {
    const n = n_rows;
    var a: [n]u8 = undefined;
    var c: [n]u8 = undefined;
    var x: [n]u8 = undefined;
    var y: [n]f32 = undefined;
    var prng: std.Random.DefaultPrng = .init(52);
    const r = prng.random();
    for (0..n) |i| {
        a[i] = r.intRangeAtMost(u8, 1, 6);
        c[i] = r.intRangeAtMost(u8, 1, 4);
        x[i] = r.intRangeAtMost(u8, 1, 8);
        const s: f32 = (if (a[i] % 3 == 0) @as(f32, 1.0) else -0.5) + @as(f32, @floatFromInt(x[i])) * 0.1 + r.floatNorm(f32) * 0.6;
        y[i] = if (s > 0) 1 else 0;
    }
    var ds = try fromBins(gpa, &.{ &a, &c, &x }, &.{ 7, 5, 9 }, &y);
    ds.kinds[0] = .categorical;
    ds.kinds[1] = .categorical;
    return ds;
}

test "permutations: the same model at 1, 3 and 16 threads; the seed matters only without has_time" {
    const gpa = testing.allocator;
    var ds = try ctrFixture(gpa);
    defer ds.deinit();
    inline for (.{ .plain, .ordered }) |bt| {
        const opts = .{ .n_rounds = 10, .max_depth = 3, .cat_split = .ctr, .boosting_type = bt, .seed = @as(u64, 4) };
        const one = try predictWith(gpa, 1, &ds, opts);
        for ([_]u32{ 3, 16 }) |t| {
            const other = try predictWith(gpa, t, &ds, opts);
            try testing.expectEqualSlices(f32, &one, &other);
        }
        const s5 = try predictWith(gpa, 2, &ds, .{ .n_rounds = 10, .max_depth = 3, .cat_split = .ctr, .boosting_type = bt, .seed = @as(u64, 5) });
        try testing.expect(!std.mem.eql(f32, &one, &s5));
        const t4 = try predictWith(gpa, 2, &ds, .{ .n_rounds = 10, .max_depth = 3, .cat_split = .ctr, .boosting_type = bt, .has_time = true, .seed = @as(u64, 4) });
        const t5 = try predictWith(gpa, 2, &ds, .{ .n_rounds = 10, .max_depth = 3, .cat_split = .ctr, .boosting_type = bt, .has_time = true, .seed = @as(u64, 5) });
        try testing.expectEqualSlices(f32, &t4, &t5);
    }
}
