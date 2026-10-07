// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Binning: the per-value floor, edge placement and each policy, at the
//! extremes. Bin-level parity with XGBoost, LightGBM and CatBoost is measured
//! outside the unit tests (docs/binning.md); these pin the rules themselves.

const std = @import("std");
const bin_edges = @import("../bin_edges.zig");
const data = @import("../data.zig");

const testing = std.testing;
const all_policies = [_]data.BinPolicy{ .quantile, .uniform, .greedy, .logsum };

/// Edges for `vals` (copied, so callers keep their order). Caller frees.
fn cutsOf(vals: []const f32, budget: usize, policy: data.BinPolicy, min_data_in_bin: u32) ![]f32 {
    const gpa = testing.allocator;
    const work = try gpa.dupe(f32, vals);
    defer gpa.free(work);
    var cuts: std.ArrayList(f32) = .empty;
    errdefer cuts.deinit(gpa);
    try bin_edges.cutsFor(gpa, work, budget, policy, min_data_in_bin, &cuts);
    return cuts.toOwnedSlice(gpa);
}

fn binOf(edges: []const f32, v: f32) usize {
    return bin_edges.lowerBound(edges, v);
}

test "a short column gets one bin per value under every policy, rare values included" {
    // The 2026-10-07 parity audit's first finding: quantile merged a value
    // held by 129 of 100,000 rows into its neighbour although the column had
    // six values and 255 bins to spend.
    var vals: [1000]f32 = undefined;
    for (&vals, 0..) |*v, i| v.* = @floatFromInt(1 + i % 5); // 1..5, 200 rows each
    vals[0] = 0; // one row of 0
    for (all_policies) |policy| {
        const e = try cutsOf(&vals, 255, policy, 1);
        defer testing.allocator.free(e);
        try testing.expectEqualSlices(f32, &.{ 0.5, 1.5, 2.5, 3.5, 4.5 }, e);
    }
}

test "the floor merges a value with fewer than min_data_in_bin rows into the next" {
    // LightGBM's rule, kept as the default; min_data_in_bin = 1 is XGBoost's.
    const vals = [_]f32{ 0, 0, 1, 1, 1, 2, 2, 2 }; // two 0s
    const e = try cutsOf(&vals, 255, .quantile, 3);
    defer testing.allocator.free(e);
    // 0 and 1 share a bin (2 rows < 3, so no cut after 0); 1 | 2 is cut.
    try testing.expectEqualSlices(f32, &.{1.5}, e);
}

test "edges sit midway, so an unseen value goes to the nearer training value" {
    const vals = [_]f32{ 0, 10, 10, 20 };
    const e = try cutsOf(&vals, 255, .quantile, 1);
    defer testing.allocator.free(e);
    try testing.expectEqualSlices(f32, &.{ 5, 15 }, e);
    // Training values land as they did with the edge on the lower value.
    try testing.expectEqual(@as(usize, 0), binOf(e, 0));
    try testing.expectEqual(@as(usize, 1), binOf(e, 10));
    try testing.expectEqual(@as(usize, 2), binOf(e, 20));
    // Unseen values: nearer the lower goes low, nearer the upper goes up.
    try testing.expectEqual(@as(usize, 0), binOf(e, 4));
    try testing.expectEqual(@as(usize, 1), binOf(e, 6));
}

test "with no float between two values, or an infinite one, the edge stays on the lower" {
    // Their f64 midpoint is exactly halfway, so it rounds to the even of the
    // two: from an odd-mantissa `a` that is `b`, an edge on `b` would put `b`
    // in `a`'s bin.
    const a: f32 = std.math.nextAfter(f32, 1.0, std.math.inf(f32));
    const b = std.math.nextAfter(f32, a, std.math.inf(f32));
    const inf = std.math.inf(f32);
    const vals = [_]f32{ -inf, a, b, inf };
    const e = try cutsOf(&vals, 255, .quantile, 1);
    defer testing.allocator.free(e);
    try testing.expectEqualSlices(f32, &.{ -inf, a, b }, e);
    // Every value still has a bin of its own.
    for (vals, 0..) |v, i| try testing.expectEqual(i, binOf(e, v));
}

test "-0.0 and +0.0 are one value" {
    const vals = [_]f32{ -0.0, 0.0, -0.0, 1, 1 };
    const e = try cutsOf(&vals, 255, .greedy, 1);
    defer testing.allocator.free(e);
    try testing.expectEqualSlices(f32, &.{0.5}, e);
}

test "degenerate columns and budgets give no cuts" {
    const one = [_]f32{ 3, 3, 3, 3 };
    const single = [_]f32{7};
    const two = [_]f32{ 1, 2, 3 };
    for (all_policies) |policy| {
        for ([_][]const f32{ &one, &single }) |vals| {
            const e = try cutsOf(vals, 255, policy, 1);
            defer testing.allocator.free(e);
            try testing.expectEqual(@as(usize, 0), e.len);
        }
        // A budget of one real bin has nothing to cut.
        const e1 = try cutsOf(&two, 1, policy, 1);
        defer testing.allocator.free(e1);
        try testing.expectEqual(@as(usize, 0), e1.len);
    }
}

/// A column with more distinct values than any budget below: a 60% atom at 0,
/// negatives and a long positive tail.
fn wideColumn(buf: []f32) void {
    var prng: std.Random.DefaultPrng = .init(11);
    const r = prng.random();
    for (buf) |*v| {
        const u = r.float(f32);
        v.* = if (u < 0.6) 0 else if (u < 0.7) -1 - r.float(f32) * 50 else r.float(f32) * r.float(f32) * 1e4;
    }
}

test "every policy respects the budget, cuts ascending, never inside a value" {
    var vals: [5000]f32 = undefined;
    wideColumn(&vals);
    for (all_policies) |policy| {
        for ([_]usize{ 2, 3, 16, 255 }) |budget| {
            const e = try cutsOf(&vals, budget, policy, 1);
            defer testing.allocator.free(e);
            try testing.expect(e.len >= 1);
            try testing.expect(e.len <= budget - 1);
            for (e[1..], e[0 .. e.len - 1]) |hi, lo| try testing.expect(hi > lo);
            // Data-driven policies put no bin out of reach of the training data.
            if (policy == .uniform) continue;
            var used = [_]bool{false} ** 256;
            for (vals) |v| used[binOf(e, v)] = true;
            for (used[0 .. e.len + 1]) |u| try testing.expect(u);
        }
    }
}

test "min_data_in_bin caps the bin count and floors every bin, under every policy" {
    var vals: [5000]f32 = undefined;
    wideColumn(&vals);
    for (all_policies) |policy| {
        const e = try cutsOf(&vals, 255, policy, 50);
        defer testing.allocator.free(e);
        // 5000 rows / 50 can fill at most 100 bins.
        try testing.expect(e.len >= 1);
        try testing.expect(e.len <= 99);
        // Uniform cuts are not run boundaries; only the count cap applies to it.
        if (policy == .uniform) continue;
        var counts = [_]usize{0} ** 256;
        for (vals) |v| counts[binOf(e, v)] += 1;
        for (counts[0..e.len]) |c| try testing.expect(c >= 50);
    }
}

test "quantile keeps its budget past an atom" {
    // 60% of rows are 0, so most rank targets of the first 60% land on it. The
    // old rule dropped each repeat; XGBoost's moves it to the next value.
    var vals: [5000]f32 = undefined;
    wideColumn(&vals);
    const e = try cutsOf(&vals, 255, .quantile, 1);
    defer testing.allocator.free(e);
    try testing.expectEqual(@as(usize, 254), e.len);
}

test "greedy gives zero a bin of its own, between the signs, even when zeros are few" {
    // LightGBM forces cuts at +-1e-35 whatever zero's share. With zeros at 2%
    // of rows they are far below a bin's worth, so only the forced cut keeps
    // them apart from their neighbours.
    var vals: [5000]f32 = undefined;
    var prng: std.Random.DefaultPrng = .init(5);
    const r = prng.random();
    for (&vals, 0..) |*v, i| {
        v.* = if (i % 50 == 0) 0 else if (i % 2 == 0) -r.float(f32) * 10 - 1e-3 else r.float(f32) * 10 + 1e-3;
    }
    for ([_]usize{ 3, 16, 255 }) |budget| {
        const e = try cutsOf(&vals, budget, .greedy, 3);
        defer testing.allocator.free(e);
        const z = binOf(e, 0);
        for (vals) |v| if (v != 0) try testing.expect(binOf(e, v) != z);
    }
}

test "logsum reproduces CatBoost's GreedyLogSum borders" {
    // Value k repeated (k*k mod 7) + 1 times, k = 0..29: 87 rows, 30 values,
    // uneven counts so the greedy order matters. Equal scores are a tie
    // CatBoost breaks by heap order and zarbor leftmost (at 9 borders four
    // candidates tie exactly), so only budgets that end clear of a tie are
    // pinned: 7 takes both of a tied pair. Expected borders from
    // catboost 1.2.10: Pool.quantize(border_count=c, feature_border_type=
    // "GreedyLogSum") then save_quantization_borders.
    var vals: [87]f32 = undefined;
    var i: usize = 0;
    for (0..30) |k| {
        for (0..(k * k % 7) + 1) |_| {
            vals[i] = @floatFromInt(k);
            i += 1;
        }
    }
    try testing.expectEqual(vals.len, i);
    const cases = [_]struct { border_count: usize, want: []const f32 }{
        .{ .border_count = 1, .want = &.{14.5} },
        .{ .border_count = 4, .want = &.{ 6.5, 14.5, 21.5, 24.5 } },
        .{ .border_count = 7, .want = &.{ 3.5, 6.5, 10.5, 14.5, 17.5, 21.5, 24.5 } },
        .{ .border_count = 8, .want = &.{ 3.5, 6.5, 10.5, 14.5, 17.5, 21.5, 24.5, 26.5 } },
    };
    for (cases) |c| {
        const e = try cutsOf(&vals, c.border_count + 1, .logsum, 1);
        defer testing.allocator.free(e);
        try testing.expectEqualSlices(f32, c.want, e);
    }
}

test "quantile reproduces XGBoost's cuts on an exact summary" {
    // 300 rows, 86 values, skewed with repeats. Expected bin starts from
    // xgboost 3.4.1 (hist, max_bin 16 and 32, DMatrix.get_quantile_cut,
    // sentinels dropped). XGBoost's summary holds 8 * max_bin entries, so at
    // these budgets it is exact and the cut rule itself is compared; with more
    // distinct values than that the sketch approximates, and zarbor, which
    // counts exactly, does not follow it there (docs/binning.md).
    var vals: [300]f32 = undefined;
    for (&vals, 0..) |*v, i| {
        const x: f64 = @floatFromInt((i * 37) % 101);
        v.* = @floatCast(@floor(@floor(std.math.pow(f64, x, 1.5)) / 10));
    }
    const cases = [_]struct { max_bin: usize, starts: []const f32 }{
        .{ .max_bin = 16, .starts = &.{ 1, 4, 7, 12, 17, 22, 29, 35, 41, 48, 57, 64, 72, 82, 91 } },
        .{ .max_bin = 32, .starts = &.{ 1, 2, 3, 4, 5, 7, 9, 12, 14, 17, 19, 22, 25, 29, 32, 35, 38, 41, 45, 48, 53, 57, 61, 64, 68, 72, 78, 82, 86, 91, 95 } },
    };
    for (cases) |c| {
        const e = try cutsOf(&vals, c.max_bin, .quantile, 1);
        defer testing.allocator.free(e);
        for (vals) |v| {
            // XGBoost: a value at or above a bin start is in that bin or later.
            var xbin: usize = 0;
            for (c.starts) |s| xbin += @intFromBool(v >= s);
            try testing.expectEqual(xbin, binOf(e, v));
        }
    }
}
