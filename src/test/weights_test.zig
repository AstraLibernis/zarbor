// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Sample weights (`--weight-col`). Tree-for-tree parity with XGBoost's and LightGBM's weighted
//! fits, and the linear objectives against scikit-learn's `sample_weight`, are measured outside
//! the unit tests (`bench/parity/parity.py`, docs/correctness.md); these pin the identities a
//! weight must keep: 1 is no weight, 0 is a dropped row, 2 is a duplicated row.

const std = @import("std");
const data = @import("../data.zig");
const config = @import("../config.zig");
const metric = @import("../metric.zig");
const csv = @import("../csv.zig");
const Fitted = @import("../fitted.zig").Fitted;
const Pool = @import("../pool.zig").Pool;
const fromBins = @import("split_test.zig").fromBins;

const testing = std.testing;

const n_rows = 1200;

/// Three 8-level features, a 0/1 label with signal in the first two, and the label's
/// regression twin (a noisy score) in `score`.
fn fixture(gpa: std.mem.Allocator, seed: u64, score: *[n_rows]f32) !data.Dataset {
    var cols: [3][n_rows]u8 = undefined;
    var y: [n_rows]f32 = undefined;
    var prng: std.Random.DefaultPrng = .init(seed);
    const r = prng.random();
    for (0..n_rows) |i| {
        for (&cols) |*c| c[i] = r.intRangeAtMost(u8, 1, 8);
        const s = @as(f32, @floatFromInt(cols[0][i])) * 0.5 - @as(f32, @floatFromInt(cols[1][i])) * 0.3 + r.floatNorm(f32);
        score[i] = s;
        y[i] = if (s > 0.6) 1 else 0;
    }
    return fromBins(gpa, &.{ &cols[0], &cols[1], &cols[2] }, &.{ 9, 9, 9 }, &y);
}

fn predictions(gpa: std.mem.Allocator, pool: *Pool, train: *const data.Dataset, on: *const data.Dataset, cfg: config.Config) ![]f32 {
    var res = try Fitted.train(gpa, pool, train, null, cfg, null);
    defer res.model.deinit();
    const out = try gpa.alloc(f32, on.n_rows * cfg.width());
    res.model.predict(pool, on, out);
    return out;
}

/// The models whose weighting is tested: XGBoost- and LightGBM-style boosting, regression, the
/// forest and the linear model. No row-count floor, which a zero-weight row would still count.
fn configs() [7]config.Config {
    var soft = config.Config.from(.{ .objective = .softmax, .n_rounds = 10, .max_depth = 4, .min_child_samples = 0, .verbose_eval = 0 });
    soft.setNumClass(2);
    var lsoft = config.Config.from(.{ .algo = .linear, .objective = .softmax, .lin_tol = 1e-12, .lin_epochs = 2000, .verbose_eval = 0 });
    lsoft.setNumClass(2);
    return .{
        soft,
        lsoft,
        config.Config.from(.{ .n_rounds = 15, .max_depth = 4, .min_child_samples = 0, .verbose_eval = 0 }),
        config.Config.from(.{ .n_rounds = 15, .grow_policy = .lossguide, .max_depth = 0, .max_leaves = 12, .min_child_samples = 0, .verbose_eval = 0 }),
        config.Config.from(.{ .objective = .squared_error, .n_rounds = 15, .max_depth = 4, .min_child_samples = 0, .verbose_eval = 0 }),
        config.Config.from(.{ .algo = .random_forest, .n_rounds = 8, .bootstrap = false, .min_child_samples = 0, .verbose_eval = 0 }),
        config.Config.from(.{ .algo = .linear, .lin_tol = 1e-12, .lin_epochs = 2000, .verbose_eval = 0 }),
    };
}

test "weights of 1 give the unweighted model, to the bit" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 3);
    defer pool.deinit();
    var score: [n_rows]f32 = undefined;
    var base = try fixture(gpa, 1, &score);
    defer base.deinit();
    for (configs()) |cfg| {
        var plain = try data.subset(gpa, &base, &iota);
        defer plain.deinit();
        if (cfg.objective() == .squared_error) @memcpy(plain.labels, &score);
        var ones = try data.subset(gpa, &plain, &iota);
        defer ones.deinit();
        ones.weights = try gpa.alloc(f32, n_rows);
        @memset(ones.weights, 1);
        const a = try predictions(gpa, pool, &plain, &plain, cfg);
        defer gpa.free(a);
        const b = try predictions(gpa, pool, &ones, &plain, cfg);
        defer gpa.free(b);
        try testing.expectEqualSlices(f32, a, b);
    }
}

const iota = blk: {
    @setEvalBranchQuota(4 * n_rows);
    var x: [n_rows]u32 = undefined;
    for (&x, 0..) |*v, i| v.* = i;
    break :blk x;
};

test "a weight of 0 is a dropped row, and a weight of 2 a duplicated one" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 3);
    defer pool.deinit();
    var score: [n_rows]f32 = undefined;
    var base = try fixture(gpa, 2, &score);
    defer base.deinit();
    var prng: std.Random.DefaultPrng = .init(9);
    // Rows to drop (weight 0), to double (weight 2), and the rest (weight 1).
    var w: [n_rows]f32 = undefined;
    var kept: std.ArrayList(u32) = .empty;
    defer kept.deinit(gpa);
    for (&w, 0..) |*v, i| {
        const u = prng.random().float(f32);
        v.* = if (u < 0.15) 0 else if (u < 0.35) 2 else 1;
        if (v.* >= 1) try kept.append(gpa, @intCast(i));
        if (v.* == 2) try kept.append(gpa, @intCast(i));
    }
    for (configs()) |cfg| {
        var labelled = try data.subset(gpa, &base, &iota);
        defer labelled.deinit();
        if (cfg.objective() == .squared_error) @memcpy(labelled.labels, &score);
        var weighted = try data.subset(gpa, &labelled, &iota);
        defer weighted.deinit();
        weighted.weights = try gpa.dupe(f32, &w);
        var expanded = try data.subset(gpa, &labelled, kept.items);
        defer expanded.deinit();
        const a = try predictions(gpa, pool, &weighted, &labelled, cfg);
        defer gpa.free(a);
        const b = try predictions(gpa, pool, &expanded, &labelled, cfg);
        defer gpa.free(b);
        // Equal up to the order f32 sums are taken in, which the row layout changes.
        for (a, b) |x, y| try testing.expectApproxEqAbs(y, x, 2e-4);
    }
}

test "boosting with weights is the same to the bit at 1, 3 and 16 threads" {
    const gpa = testing.allocator;
    var score: [n_rows]f32 = undefined;
    var ds = try fixture(gpa, 3, &score);
    defer ds.deinit();
    ds.weights = try gpa.alloc(f32, n_rows);
    var prng: std.Random.DefaultPrng = .init(4);
    for (ds.weights) |*v| v.* = 0.05 + 6 * prng.random().float(f32) * prng.random().float(f32);
    var want: [n_rows]f32 = undefined;
    for ([_]u32{ 1, 3, 16 }) |threads| {
        const pool = try Pool.init(gpa, threads);
        defer pool.deinit();
        const got = try predictions(gpa, pool, &ds, &ds, config.Config.from(.{ .n_rounds = 12, .max_depth = 5, .subsample = 0.8, .colsample_bytree = 0.7, .verbose_eval = 0 }));
        defer gpa.free(got);
        if (threads == 1) @memcpy(&want, got) else try testing.expectEqualSlices(f32, &want, got);
    }
}

test "weighted metrics: weights of 1 are the plain metric, a weight of 2 a duplicated row" {
    const gpa = testing.allocator;
    const s = [_]f32{ 0.9, 0.2, 0.7, 0.7, 0.1, 0.5, 0.5, 0.3 };
    const y = [_]f32{ 1, 0, 1, 0, 0, 1, 0, 1 };
    const one = [_]f32{1} ** 8;
    try testing.expectApproxEqAbs(try metric.auc(gpa, &s, &y), try metric.aucW(gpa, &s, &y, &one), 1e-12);
    try testing.expectApproxEqAbs(metric.rmse(&s, &y), metric.rmseW(&s, &y, &one), 1e-12);
    try testing.expectApproxEqAbs(metric.loglossProb(&s, &y), metric.loglossProbW(&s, &y, &one), 1e-12);
    // Row 3 (a tied negative) doubled, row 4 dropped.
    const w = [_]f32{ 1, 1, 1, 2, 0, 1, 1, 1 };
    const sd = [_]f32{ 0.9, 0.2, 0.7, 0.7, 0.7, 0.5, 0.5, 0.3 };
    const yd = [_]f32{ 1, 0, 1, 0, 0, 1, 0, 1 };
    try testing.expectApproxEqAbs(try metric.auc(gpa, &sd, &yd), try metric.aucW(gpa, &s, &y, &w), 1e-12);
    try testing.expectApproxEqAbs(metric.rmse(&sd, &yd), metric.rmseW(&s, &y, &w), 1e-12);
}

/// A frame with one numeric column, `values`.
fn oneColumn(gpa: std.mem.Allocator, values: []const f32) !csv.Frame {
    const names = try gpa.alloc([]u8, 1);
    names[0] = try gpa.dupe(u8, "w");
    const kinds = try gpa.alloc(data.ColumnKind, 1);
    kinds[0] = .numeric;
    const vals = try gpa.alloc([]f32, 1);
    vals[0] = try gpa.dupe(f32, values);
    const lv = try gpa.alloc([][]u8, 1);
    lv[0] = &.{};
    return .{ .gpa = gpa, .n_rows = values.len, .names = names, .kinds = kinds, .values = vals, .levels = lv };
}

test "a weight column with a missing, negative or all-zero weight is refused" {
    const gpa = testing.allocator;
    const cases = .{
        .{ &[_]f32{ 1, std.math.nan(f32), 2 }, error.MissingWeight },
        .{ &[_]f32{ 1, -0.5, 2 }, error.NegativeWeight },
        .{ &[_]f32{ 0, 0, 0 }, error.WeightsSumToZero },
    };
    inline for (cases) |c| {
        var f = try oneColumn(gpa, c[0]);
        defer f.deinit();
        try testing.expectError(c[1], data.readWeights(gpa, &f, 0));
    }
    var f = try oneColumn(gpa, &.{ 0, 1.5, 3 });
    defer f.deinit();
    const w = try data.readWeights(gpa, &f, 0);
    defer gpa.free(w);
    try testing.expectEqualSlices(f32, &.{ 0, 1.5, 3 }, w);
}

test "a subset keeps each row's weight" {
    const gpa = testing.allocator;
    var score: [n_rows]f32 = undefined;
    var ds = try fixture(gpa, 5, &score);
    defer ds.deinit();
    ds.weights = try gpa.alloc(f32, n_rows);
    for (ds.weights, 0..) |*v, i| v.* = @floatFromInt(i % 7);
    const rows = [_]u32{ 9, 3, 700, 3, 1199 };
    var sub = try data.subset(gpa, &ds, &rows);
    defer sub.deinit();
    for (rows, sub.weights) |r, v| try testing.expectEqual(ds.weights[r], v);
}

const lin_solve = @import("../lin_solve.zig");
const linear = @import("../linear.zig");

test "weighted linear objective: the gradient is its derivative (central differences)" {
    // The line search accepts steps by the objective while the step follows the gradient: a
    // weight applied to one and not the other still converges, to the wrong point's
    // neighbourhood, so the pair is checked directly, binary and multinomial.
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();
    var score: [n_rows]f32 = undefined;
    var ds = try fixture(gpa, 6, &score);
    defer ds.deinit();
    ds.weights = try gpa.alloc(f32, n_rows);
    var prng: std.Random.DefaultPrng = .init(8);
    for (ds.weights) |*v| v.* = 4 * prng.random().float(f32);
    var design = try linear.buildDesign(gpa, &ds, true);
    defer design.deinit();
    const tables = try design.valueTables(gpa, &ds);
    defer linear.Design.freeTables(gpa, tables);
    const p = design.cols.len;
    inline for (.{ .{ .logistic, 1 }, .{ .squared_error, 1 }, .{ .softmax, 2 } }) |c| {
        const k: usize = c[1];
        const theta = try gpa.alloc(f64, (p + 1) * k);
        defer gpa.free(theta);
        for (theta) |*t| t.* = prng.random().floatNorm(f64) * 0.3;
        const wf = try gpa.alloc(f32, p * k);
        defer gpa.free(wf);
        const z = try gpa.alloc(f32, n_rows * k);
        defer gpa.free(z);
        const resid = try gpa.alloc(f32, n_rows * k);
        defer gpa.free(resid);
        var loss_part: [1]f64 = undefined;
        var rsum_part: [2]f64 = undefined;
        var pr = lin_solve.Problem{
            .pool = pool, .design = &design, .tables = tables, .ds = &ds, .objective = c[0], .scale_pos_weight = 1,
            .l2 = 0.02, .l1 = 0, .theta = theta, .k = k, .wf = wf, .z = z, .resid = resid,
            .loss_part = &loss_part, .rsum_part = rsum_part[0..k], .chunks = 1, .size = n_rows,
        };
        const g = try gpa.alloc(f64, theta.len);
        defer gpa.free(g);
        _ = pr.value(theta, true);
        pr.grad(theta, g);
        const probe = try gpa.dupe(f64, theta);
        defer gpa.free(probe);
        const h = 1e-3;
        for (0..theta.len) |i| {
            probe[i] = theta[i] + h;
            const up = pr.value(probe, true);
            probe[i] = theta[i] - h;
            const down = pr.value(probe, true);
            probe[i] = theta[i];
            try testing.expectApproxEqAbs((up - down) / (2 * h), g[i], 5e-4);
        }
    }
}

test "CatBoost-style trees: weights of 1 change nothing, and scaling every weight changes nothing" {
    // CatBoost scales `l2_leaf_reg` by the mean weight (by each prefix body's under ordered
    // boosting), so multiplying every weight by c multiplies derivatives, counts and L2 alike:
    // leaf values and the chosen splits stay put. Boosting without that scaling would not.
    // Which mean (the fold's or each prefix body's) and whether the prefix models' derivatives
    // are weighted are invisible to this identity; `bench/parity/parity.py` (cat_weights) pins
    // them against CatBoost row for row.
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 3);
    defer pool.deinit();
    var score: [n_rows]f32 = undefined;
    var base = try fixture(gpa, 7, &score);
    defer base.deinit();
    inline for (.{ .plain, .ordered }) |bt| {
        const cfg = config.Config.from(.{ .grow_policy = .symmetric, .boosting_type = bt, .has_time = true, .lambda = 3, .n_rounds = 12, .max_depth = 4, .verbose_eval = 0 });
        const plain = try predictions(gpa, pool, &base, &base, cfg);
        defer gpa.free(plain);
        var prng: std.Random.DefaultPrng = .init(11);
        var w: [n_rows]f32 = undefined;
        for (&w) |*v| v.* = 0.2 + 3 * prng.random().float(f32);
        var outs: [3][]f32 = undefined;
        for ([_]f32{ 1, 1, 7 }, 0..) |c, i| {
            var d = try data.subset(gpa, &base, &iota);
            defer d.deinit();
            d.weights = try gpa.alloc(f32, n_rows);
            for (d.weights, w) |*dst, v| dst.* = if (i == 0) 1 else c * v;
            outs[i] = try predictions(gpa, pool, &d, &base, cfg);
        }
        defer for (outs) |o| gpa.free(o);
        try testing.expectEqualSlices(f32, plain, outs[0]);
        try testing.expect(!std.mem.eql(f32, plain, outs[1]));
        for (outs[1], outs[2]) |a, b| try testing.expectApproxEqAbs(a, b, 1e-5);
    }
}

test "balanced class weights: n / (K n_c), from weight sums when rows are weighted" {
    const gpa = testing.allocator;
    var score: [n_rows]f32 = undefined;
    var ds = try fixture(gpa, 9, &score);
    defer ds.deinit();
    // Binary: each class's total weight becomes n / 2.
    const cfg = config.Config.from(.{ .class_weight = .balanced });
    const w = try Fitted.classWeights(gpa, &ds, cfg);
    defer gpa.free(w);
    var pos: f64 = 0;
    var neg: f64 = 0;
    for (w, ds.labels) |v, y| {
        if (y > 0.5) pos += v else neg += v;
    }
    try testing.expectApproxEqRel(@as(f64, n_rows) / 2, pos, 1e-6);
    try testing.expectApproxEqRel(@as(f64, n_rows) / 2, neg, 1e-6);

    // With sample weights, the classes' weighted totals are equal (scikit-learn 1.9).
    ds.weights = try gpa.alloc(f32, n_rows);
    for (ds.weights, 0..) |*v, i| v.* = @floatFromInt(1 + i % 5);
    const ww = try Fitted.classWeights(gpa, &ds, cfg);
    defer gpa.free(ww);
    var total: f64 = 0;
    for (ds.weights) |v| total += v;
    pos = 0;
    neg = 0;
    for (ww, ds.labels) |v, y| {
        if (y > 0.5) pos += v else neg += v;
    }
    try testing.expectApproxEqRel(total / 2, pos, 1e-6);
    try testing.expectApproxEqRel(total / 2, neg, 1e-6);

    // Softmax: three classes, each totals n / 3.
    var soft = config.Config.from(.{ .objective = .softmax, .class_weight = .balanced });
    soft.setNumClass(3);
    gpa.free(ds.weights);
    ds.weights = &.{};
    for (ds.labels, 0..) |*y, i| y.* = @floatFromInt(i % 7 % 3);
    const ws = try Fitted.classWeights(gpa, &ds, soft);
    defer gpa.free(ws);
    var per = [_]f64{0} ** 3;
    for (ws, ds.labels) |v, y| per[@intFromFloat(y)] += v;
    for (per) |t| try testing.expectApproxEqRel(@as(f64, n_rows) / 3, t, 1e-6);

    try testing.expectError(error.ClassWeightNeedsClasses, config.Config.from(.{ .objective = .squared_error, .class_weight = .balanced }).validate());
}

test "balanced classes of equal size weigh exactly 1: the unweighted model, to the bit" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();
    var score: [n_rows]f32 = undefined;
    var ds = try fixture(gpa, 10, &score);
    defer ds.deinit();
    for (ds.labels, 0..) |*y, i| y.* = @floatFromInt(i % 2);
    const a = try predictions(gpa, pool, &ds, &ds, config.Config.from(.{ .n_rounds = 8, .verbose_eval = 0 }));
    defer gpa.free(a);
    const b = try predictions(gpa, pool, &ds, &ds, config.Config.from(.{ .n_rounds = 8, .class_weight = .balanced, .verbose_eval = 0 }));
    defer gpa.free(b);
    try testing.expectEqualSlices(f32, a, b);

    // Unequal classes: the weights move the model (and lift the rare class's probabilities).
    for (ds.labels, 0..) |*y, i| y.* = if (i % 5 == 0) 1 else 0;
    const c = try predictions(gpa, pool, &ds, &ds, config.Config.from(.{ .n_rounds = 8, .verbose_eval = 0 }));
    defer gpa.free(c);
    const d = try predictions(gpa, pool, &ds, &ds, config.Config.from(.{ .n_rounds = 8, .class_weight = .balanced, .verbose_eval = 0 }));
    defer gpa.free(d);
    var mc: f64 = 0;
    var md: f64 = 0;
    for (c, d) |x, y| {
        mc += x;
        md += y;
    }
    try testing.expect(md > mc + 0.1 * n_rows);
}
