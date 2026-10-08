// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! The regression losses: absolute error and quantile (LightGBM's, with leaves renewed to a
//! residual percentile), pseudo-Huber and Poisson (XGBoost's). Tree-for-tree parity with each is
//! measured by `bench/parity/parity.py`; these pin the pieces a wrong change would break quietly.

const std = @import("std");
const data = @import("../data.zig");
const config = @import("../config.zig");
const booster = @import("../booster.zig");
const metric = @import("../metric.zig");
const model_mod = @import("../model.zig");
const Pool = @import("../pool.zig").Pool;
const fromBins = @import("split_test.zig").fromBins;

const testing = std.testing;

const n_rows = 1500;

/// Three 8-level features and a skewed, outlier-heavy target driven by the first two.
fn fixture(gpa: std.mem.Allocator, seed: u64, counts: bool) !data.Dataset {
    var cols: [3][n_rows]u8 = undefined;
    var y: [n_rows]f32 = undefined;
    var prng: std.Random.DefaultPrng = .init(seed);
    const r = prng.random();
    for (0..n_rows) |i| {
        for (&cols) |*c| c[i] = r.intRangeAtMost(u8, 1, 8);
        const s = @as(f32, @floatFromInt(cols[0][i])) * 0.4 - @as(f32, @floatFromInt(cols[1][i])) * 0.25;
        if (counts) {
            // A Poisson count with mean exp(s / 2), by inversion.
            const mean = @exp(s / 2);
            var k: f32 = 0;
            var p = @exp(-mean);
            var cdf = p;
            const u = r.float(f32);
            while (u > cdf and k < 60) {
                k += 1;
                p *= mean / k;
                cdf += p;
            }
            y[i] = k;
        } else {
            const noise = r.floatNorm(f32);
            y[i] = s + noise + if (r.float(f32) < 0.05) 25 * noise else 0;
        }
    }
    return fromBins(gpa, &.{ &cols[0], &cols[1], &cols[2] }, &.{ 9, 9, 9 }, &y);
}

test "percentile follows LightGBM's rules, plain and weighted" {
    const gpa = testing.allocator;
    // Plain: (n - 1)(1 - alpha) from the top, interpolated. Sorted descending 9 7 5 3 1.
    var v = [_]f64{ 3, 9, 1, 7, 5 };
    try testing.expectEqual(@as(f64, 5), try booster.percentile(gpa, &v, &.{}, 0.5));
    v = .{ 3, 9, 1, 7, 5 };
    // pos = int(4 * 0.2) + 1 = 1, bias 0.8: 9 - (9 - 7) * 0.8.
    try testing.expectApproxEqAbs(@as(f64, 7.4), try booster.percentile(gpa, &v, &.{}, 0.8), 1e-12);
    v = .{ 3, 9, 1, 7, 5 };
    try testing.expectEqual(@as(f64, 9), try booster.percentile(gpa, &v, &.{}, 1.0));
    var one = [_]f64{4.5};
    try testing.expectEqual(@as(f64, 4.5), try booster.percentile(gpa, &one, &.{}, 0.3));
    // Weighted: ascending 1 3 5 7 9 with weights 1 1 4 1 1, cumulative 1 2 6 7 8; at 0.5 the
    // threshold 4 lands in 5's span, past a gap >= 1: 3 + (4 - 2) / 4 * (5 - 3).
    var w_v = [_]f64{ 9, 1, 5, 3, 7 };
    const w = [_]f32{ 1, 1, 4, 1, 1 };
    try testing.expectApproxEqAbs(@as(f64, 4), try booster.percentile(gpa, &w_v, &w, 0.5), 1e-12);
    // A span under 1 takes the lower value outright.
    var s_v = [_]f64{ 1, 2, 3, 4 };
    const sw = [_]f32{ 0.2, 0.3, 0.3, 0.2 };
    try testing.expectEqual(@as(f64, 2), try booster.percentile(gpa, &s_v, &sw, 0.6));
}

fn params(opts: anytype) booster.Params {
    var cfg = config.Config.from(.{ .verbose_eval = 0 }).gbdt;
    inline for (@typeInfo(@TypeOf(opts)).@"struct".fields) |f| {
        if (@hasField(booster.Params, f.name)) @field(cfg, f.name) = @field(opts, f.name) else @field(cfg.tree, f.name) = @field(opts, f.name);
    }
    return cfg;
}

test "absolute error and quantile: each leaf is its rows' residual median or percentile" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 3);
    defer pool.deinit();
    var ds = try fixture(gpa, 1, false);
    defer ds.deinit();
    inline for (.{ .{ .absolute_error, 0.5 }, .{ .quantile, 0.8 } }) |c| {
        // One tree at learning rate 1: the model is the start plus the leaf values.
        var res = try booster.train(gpa, pool, &ds, null, params(.{ .objective = c[0], .quantile_alpha = c[1], .n_rounds = 1, .learning_rate = 1, .max_depth = 3 }), null);
        defer res.model.deinit();
        const m = &res.model;
        // The start is the label percentile.
        var ys: [n_rows]f64 = undefined;
        for (&ys, ds.labels) |*o, y| o.* = y;
        try testing.expectApproxEqAbs(@as(f32, @floatCast(try booster.percentile(gpa, &ys, &.{}, c[1]))), m.base_score, 1e-6);
        const t = m.trees.items[0];
        var leaves: usize = 0;
        for (t.nodes, 0..) |node, i| {
            if (!node.is_leaf) continue;
            var res_rows: std.ArrayList(f64) = .empty;
            defer res_rows.deinit(gpa);
            for (0..n_rows) |r| {
                var j: u32 = 0;
                while (!t.nodes[j].is_leaf) j = if (ds.columnNarrow(t.nodes[j].feature)[r] <= t.nodes[j].threshold) t.nodes[j].left else t.nodes[j].right;
                if (j == i) try res_rows.append(gpa, @as(f64, ds.labels[r]) - m.base_score);
            }
            if (res_rows.items.len == 0) continue;
            const want = try booster.percentile(gpa, res_rows.items, &.{}, c[1]);
            try testing.expectApproxEqAbs(@as(f32, @floatCast(want)), node.weight, 1e-5);
            leaves += 1;
        }
        try testing.expect(leaves >= 4);
    }
}

test "pseudo-Huber with a huge slope is squared error, and its fit resists outliers at a small one" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();
    var ds = try fixture(gpa, 2, false);
    defer ds.deinit();
    var big = try booster.train(gpa, pool, &ds, null, params(.{ .objective = .pseudo_huber, .huber_slope = 1e4, .n_rounds = 10, .max_depth = 3 }), null);
    defer big.model.deinit();
    var sq = try booster.train(gpa, pool, &ds, null, params(.{ .objective = .squared_error, .n_rounds = 10, .max_depth = 3 }), null);
    defer sq.model.deinit();
    var a: [n_rows]f32 = undefined;
    var b: [n_rows]f32 = undefined;
    big.model.predict(pool, &ds, &a);
    sq.model.predict(pool, &ds, &b);
    for (a, b) |x, y| try testing.expectApproxEqAbs(y, x, 1e-3);
    // The hessian is XGBoost's s^2 / ((s^2 + z^2) sqrt(1 + z^2 / s^2)): the root's stored cover
    // is its sum over the rows at the starting score.
    var one = try booster.train(gpa, pool, &ds, null, params(.{ .objective = .pseudo_huber, .huber_slope = 1.5, .n_rounds = 1, .max_depth = 2 }), null);
    defer one.model.deinit();
    var want_h: f64 = 0;
    for (ds.labels) |y| {
        const z = one.model.base_score - y;
        const s2: f32 = 1.5 * 1.5;
        want_h += s2 / ((s2 + z * z) * @sqrt(1 + z * z / s2));
    }
    try testing.expectApproxEqRel(want_h, one.model.trees.items[0].stats[0].hess, 1e-4);
    // At slope 1 the outliers (5% of rows, 25x noise) pull the fit less: lower absolute error.
    var small = try booster.train(gpa, pool, &ds, null, params(.{ .objective = .pseudo_huber, .huber_slope = 1, .n_rounds = 60, .max_depth = 3 }), null);
    defer small.model.deinit();
    var sq60 = try booster.train(gpa, pool, &ds, null, params(.{ .objective = .squared_error, .n_rounds = 60, .max_depth = 3 }), null);
    defer sq60.model.deinit();
    small.model.predict(pool, &ds, &a);
    sq60.model.predict(pool, &ds, &b);
    try testing.expect(metric.mae(&a, ds.labels, &.{}) < metric.mae(&b, ds.labels, &.{}));
}

test "Poisson: predictions are exp(raw), and leaves are clipped at its max_delta_step" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();
    var ds = try fixture(gpa, 3, true);
    defer ds.deinit();
    var res = try booster.train(gpa, pool, &ds, null, params(.{ .objective = .poisson, .n_rounds = 15, .max_depth = 4, .learning_rate = 1 }), null);
    defer res.model.deinit();
    var raw: [n_rows]f32 = undefined;
    var mu: [n_rows]f32 = undefined;
    res.model.predictRaw(pool, &ds, &raw);
    res.model.predict(pool, &ds, &mu);
    for (raw, mu) |r, m| try testing.expectApproxEqRel(@exp(r), m, 1e-6);
    var hit = false;
    for (res.model.trees.items) |t| for (t.nodes) |n| if (n.is_leaf) {
        try testing.expect(@abs(n.weight) <= 0.7 + 1e-6);
        hit = hit or @abs(n.weight) > 0.69;
    };
    try testing.expect(hit);
    // A saved model applies the same link.
    var bundle = try model_mod.fromBooster(gpa, &res.model, try data.Schema.fromDataset(gpa, &ds));
    defer bundle.deinit();
    var saved: [n_rows]f32 = undefined;
    bundle.predict(pool, &ds, &saved);
    try testing.expectEqualSlices(f32, &mu, &saved);
    // Its metric decreases as the model learns.
    var weak = try booster.train(gpa, pool, &ds, null, params(.{ .objective = .poisson, .n_rounds = 1, .max_depth = 1, .learning_rate = 0.01 }), null);
    defer weak.model.deinit();
    var mw: [n_rows]f32 = undefined;
    weak.model.predict(pool, &ds, &mw);
    try testing.expect(metric.poissonNloglik(&mu, ds.labels, &.{}) < metric.poissonNloglik(&mw, ds.labels, &.{}));
}

test "loss metrics: pinball at 0.5 is half the absolute error; mphe tends to half the squared error" {
    const p = [_]f32{ 1, 2, 3, 4 };
    const y = [_]f32{ 1.5, 1, 3.5, 2 };
    try testing.expectApproxEqAbs(metric.mae(&p, &y, &.{}) / 2, metric.pinball(&p, &y, &.{}, 0.5), 1e-12);
    // Above the label at 0.8: 0.2 per unit; below: 0.8 per unit.
    try testing.expectApproxEqAbs(@as(f64, (0.8 * 0.5 + 0.2 * 1 + 0.8 * 0.5 + 0.2 * 2) / 4.0), metric.pinball(&p, &y, &.{}, 0.8), 1e-12);
    const r = metric.rmse(&p, &y);
    try testing.expectApproxEqRel(r * r / 2, metric.mphe(&p, &y, &.{}, 1e4), 1e-6);
    // Poisson: mu - y log(mu) + lgamma(y + 1) at mu = 2, y = 3.
    try testing.expectApproxEqAbs(2 - 3 * @log(@as(f64, 2)) + @log(@as(f64, 6)), metric.poissonNloglik(&.{2}, &.{3}, &.{}), 1e-12);
}

test "the new losses are the same to the bit at 1, 3 and 16 threads, and refused where not built" {
    const gpa = testing.allocator;
    var ds = try fixture(gpa, 4, false);
    defer ds.deinit();
    inline for (.{ .absolute_error, .quantile, .pseudo_huber }) |obj| {
        var want: [n_rows]f32 = undefined;
        for ([_]u32{ 1, 3, 16 }) |threads| {
            const pool = try Pool.init(gpa, threads);
            defer pool.deinit();
            var res = try booster.train(gpa, pool, &ds, null, params(.{ .objective = obj, .n_rounds = 8, .max_depth = 4, .subsample = 0.8 }), null);
            defer res.model.deinit();
            var got: [n_rows]f32 = undefined;
            res.model.predict(pool, &ds, &got);
            if (threads == 1) want = got else try testing.expectEqualSlices(f32, &want, &got);
        }
    }
    try testing.expectError(error.ObjectiveSymmetricUnsupported, params(.{ .objective = .quantile, .grow_policy = .symmetric }).validate());
    try testing.expectError(error.BadQuantileAlpha, params(.{ .objective = .quantile, .quantile_alpha = 1 }).validate());
    try testing.expectError(error.BadHuberSlope, params(.{ .objective = .pseudo_huber, .huber_slope = 0 }).validate());
    try testing.expectError(error.ObjectiveNeedsGbdt, config.Config.from(.{ .algo = .linear, .objective = .poisson }).validate());
    var enc = data.LabelEncoder{ .gpa = gpa };
    try testing.expectError(error.LabelOutOfRange, enc.validate(&.{ 1, -1 }, .poisson));
}
