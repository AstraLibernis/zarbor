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
