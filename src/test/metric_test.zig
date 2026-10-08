// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

const std = @import("std");
const metric = @import("../metric.zig");
const auc = metric.auc;

const testing = std.testing;

test "auc is 1 for a perfect ranking" {
    const s = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
    const y = [_]f32{ 0, 0, 1, 1 };
    try testing.expectApproxEqAbs(@as(f64, 1.0), try auc(testing.allocator, &s, &y), 1e-12);
}

test "auc is 0.5 when every score ties" {
    const s = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    const y = [_]f32{ 0, 1, 0, 1 };
    try testing.expectApproxEqAbs(@as(f64, 0.5), try auc(testing.allocator, &s, &y), 1e-12);
}

test "auc is 0 for a perfectly inverted ranking" {
    const s = [_]f32{ 0.9, 0.8, 0.2, 0.1 };
    const y = [_]f32{ 0, 0, 1, 1 };
    try testing.expectApproxEqAbs(@as(f64, 0.0), try auc(testing.allocator, &s, &y), 1e-12);
}

/// The comparison-sort AUC the radix version replaced, kept as the reference it must match.
fn aucReference(gpa: std.mem.Allocator, scores: []const f32, labels: []const f32) !f64 {
    const n = scores.len;
    const idx = try gpa.alloc(u32, n);
    defer gpa.free(idx);
    for (idx, 0..) |*v, i| v.* = @intCast(i);
    const Ctx = struct {
        s: []const f32,
        fn lessThan(c: @This(), a: u32, b: u32) bool {
            return c.s[a] < c.s[b];
        }
    };
    std.sort.pdq(u32, idx, Ctx{ .s = scores }, Ctx.lessThan);
    var pos: f64 = 0;
    var neg: f64 = 0;
    var rank_sum: f64 = 0;
    var i: usize = 0;
    while (i < n) {
        var j = i;
        while (j + 1 < n and scores[idx[j + 1]] == scores[idx[i]]) j += 1;
        const avg_rank = (@as(f64, @floatFromInt(i + 1)) + @as(f64, @floatFromInt(j + 1))) / 2.0;
        var k = i;
        while (k <= j) : (k += 1) {
            if (labels[idx[k]] > 0.5) {
                pos += 1;
                rank_sum += avg_rank;
            } else {
                neg += 1;
            }
        }
        i = j + 1;
    }
    if (pos == 0 or neg == 0) return 0.5;
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg);
}

test "radix auc equals the comparison-sort auc to the bit" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(42);
    const r = prng.random();
    const n = 20_000;
    const s = try gpa.alloc(f32, n);
    defer gpa.free(s);
    const y = try gpa.alloc(f32, n);
    defer gpa.free(y);
    // Continuous scores, heavy ties (like rows sharing a leaf), mixed signs, and both zeros.
    for (0..4) |shape| {
        for (s, y) |*sv, *yv| {
            sv.* = switch (shape) {
                0 => r.float(f32),
                1 => @floatFromInt(r.uintLessThan(u32, 40)),
                2 => (r.float(f32) - 0.5) * 1e6,
                else => if (r.boolean()) -0.0 else @as(f32, @floatFromInt(r.uintLessThan(u32, 5))) - 2.0,
            };
            yv.* = if (r.float(f32) < sv.* * 1e-7 + 0.3) 1 else 0;
        }
        try testing.expectEqual(try aucReference(gpa, s, y), try auc(gpa, s, y));
    }
}

// ------------------------------------------------------------ --eval_metric

// Reference values from scikit-learn 1.9 in float64 (`average_precision_score`, `r2_score`).
const ap_scores = [_]f32{ 0.9, 0.8, 0.8, 0.7, 0.6, 0.6, 0.6, 0.3, 0.2, 0.1 };
const ap_labels = [_]f32{ 1, 1, 0, 1, 0, 1, 0, 0, 1, 0 };
const ap_weights = [_]f32{ 1.0, 0.5, 2.0, 1.5, 1.0, 0.25, 3.0, 1.0, 0.5, 2.0 };

test "average precision equals scikit-learn's, ties and weights included" {
    const gpa = testing.allocator;
    try testing.expectApproxEqAbs(@as(f64, 0.7087301587301587), try metric.averagePrecision(gpa, &ap_scores, &ap_labels, &.{}), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.633744575139924), try metric.averagePrecision(gpa, &ap_scores, &ap_labels, &ap_weights), 1e-12);
    // Shuffled rows give the same value: the order of a tied block does not matter.
    const perm = [_]usize{ 6, 2, 9, 0, 5, 3, 8, 1, 7, 4 };
    var s: [10]f32 = undefined;
    var y: [10]f32 = undefined;
    var w: [10]f32 = undefined;
    for (perm, 0..) |p, i| {
        s[i] = ap_scores[p];
        y[i] = ap_labels[p];
        w[i] = ap_weights[p];
    }
    try testing.expectApproxEqAbs(@as(f64, 0.633744575139924), try metric.averagePrecision(gpa, &s, &y, &w), 1e-12);
    // A perfect ranking is 1; no positives is 0.
    try testing.expectEqual(@as(f64, 1), try metric.averagePrecision(gpa, &.{ 0.1, 0.9, 0.8 }, &.{ 0, 1, 1 }, &.{}));
    try testing.expectEqual(@as(f64, 0), try metric.averagePrecision(gpa, &.{ 0.1, 0.9 }, &.{ 0, 0 }, &.{}));
}

test "r2 equals scikit-learn's, weighted and not, and its constant-target convention" {
    const p = [_]f32{ 1.0, 2.5, 2.0, 4.0, 3.5, 0.5 };
    const y = [_]f32{ 1.5, 2.0, 2.5, 3.0, 4.5, 1.0 };
    const w = [_]f32{ 1, 2, 0.5, 1, 3, 0.25 };
    try testing.expectApproxEqAbs(@as(f64, 0.6108108108108108), metric.r2(&p, &y, &.{}), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.5925124792013311), metric.r2(&p, &y, &w), 1e-12);
    try testing.expectEqual(@as(f64, 1), metric.r2(&.{ 2, 2 }, &.{ 2, 2 }, &.{}));
    try testing.expectEqual(@as(f64, 0), metric.r2(&.{ 1, 2 }, &.{ 2, 2 }, &.{}));
}

test "evaluate: raw and natural scales agree for every metric that takes either" {
    const gpa = testing.allocator;
    const raw = [_]f32{ -2.0, 0.5, 1.5, -0.25, 3.0, -1.0 };
    const y = [_]f32{ 0, 1, 1, 1, 1, 0 };
    const w = [_]f32{ 1, 2, 0.5, 1, 3, 0.25 };
    var prob: [6]f32 = undefined;
    for (&prob, raw) |*q, r| q.* = 1 / (1 + @exp(-r));
    const ctx: metric.Ctx = .{ .objective = .logistic };
    inline for (.{ .auc, .aucpr, .logloss, .error_rate, .accuracy }) |m| for ([_][]const f32{ &.{}, &w }) |ws| {
        const a = try metric.evaluate(gpa, m, ctx, &raw, .raw, &y, ws);
        const b = try metric.evaluate(gpa, m, ctx, &prob, .natural, &y, ws);
        try testing.expectApproxEqAbs(a, b, 1e-6);
    };
    // One row is on the wrong side (raw -0.25, label 1).
    try testing.expectApproxEqAbs(@as(f64, 1.0 / 6.0), try metric.evaluate(gpa, .error_rate, ctx, &raw, .raw, &y, &.{}), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 5.0 / 6.0), try metric.evaluate(gpa, .accuracy, ctx, &raw, .raw, &y, &.{}), 1e-12);
    // Poisson: a raw score is a log mean.
    const pc: metric.Ctx = .{ .objective = .poisson };
    const counts = [_]f32{ 0, 1, 3, 2, 5, 0 };
    var mu: [6]f32 = undefined;
    for (&mu, raw) |*m, r| m.* = @exp(r);
    inline for (.{ .rmse, .mae, .r2, .poisson_nloglik }) |m| {
        try testing.expectApproxEqAbs(try metric.evaluate(gpa, m, pc, &mu, .natural, &counts, &w), try metric.evaluate(gpa, m, pc, &raw, .raw, &counts, &w), 1e-6);
    }
}

test "every objective's default metric fits it, and directions are right" {
    inline for (@typeInfo(@import("../objective.zig").Objective).@"enum".fields) |f| {
        const obj: @import("../objective.zig").Objective = @enumFromInt(f.value);
        try testing.expect(metric.Metric.defaultFor(obj).fits(obj));
    }
    try testing.expect(!metric.Metric.auc.fits(.squared_error));
    try testing.expect(!metric.Metric.rmse.fits(.logistic));
    try testing.expect(!metric.Metric.poisson_nloglik.fits(.squared_error));
    try testing.expect(!metric.Metric.mlogloss.fits(.logistic));
    try testing.expect(metric.Metric.accuracy.fits(.logistic) and metric.Metric.accuracy.fits(.softmax));
    for ([_]metric.Metric{ .auc, .aucpr, .accuracy, .r2 }) |m| try testing.expect(m.higherIsBetter());
    for ([_]metric.Metric{ .logloss, .error_rate, .mlogloss, .rmse, .mae, .pinball, .mphe, .poisson_nloglik }) |m| try testing.expect(!m.higherIsBetter());
}
