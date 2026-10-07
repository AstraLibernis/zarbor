// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

const std = @import("std");
const config = @import("../config.zig");
const csv = @import("../csv.zig");
const data = @import("../data.zig");
const pool_mod = @import("../pool.zig");
const booster = @import("../booster.zig");
const forest = @import("../forest.zig");
const linear = @import("../linear.zig");
const metric = @import("../metric.zig");
const cv = @import("../cv.zig");
const assignFolds = cv.assignFolds;
const assignGroupFolds = cv.assignGroupFolds;

const testing = std.testing;

test "folds are equally sized and cover every row exactly once" {
    const gpa = testing.allocator;
    const n = 1000;
    const labels = try gpa.alloc(f32, n);
    defer gpa.free(labels);
    for (labels, 0..) |*y, i| y.* = if (i % 5 == 0) 1 else 0;

    const fold = try assignFolds(gpa, labels, 5, 7, true);
    defer gpa.free(fold);

    var count = [_]usize{0} ** 5;
    for (fold) |k| {
        try testing.expect(k < 5);
        count[k] += 1;
    }
    for (count) |c| try testing.expectEqual(@as(usize, 200), c);
}

test "stratification keeps each fold's class balance" {
    const gpa = testing.allocator;
    const n = 10_000;
    const labels = try gpa.alloc(f32, n);
    defer gpa.free(labels);
    // 10% positive, and deliberately contiguous: an unshuffled or
    // unstratified split would hand whole folds a wildly wrong balance.
    for (labels, 0..) |*y, i| y.* = if (i < n / 10) 1 else 0;

    const fold = try assignFolds(gpa, labels, 5, 3, true);
    defer gpa.free(fold);

    var pos = [_]usize{0} ** 5;
    var tot = [_]usize{0} ** 5;
    for (fold, labels) |k, y| {
        tot[k] += 1;
        if (y >= 0.5) pos[k] += 1;
    }
    // Exactly 1000 positives over 5 folds: dealing them round-robin from
    // their own shuffled list puts 200 in each, not merely "about" 200.
    for (pos) |p| try testing.expectEqual(@as(usize, 200), p);
    for (tot) |t| try testing.expectEqual(@as(usize, 2000), t);
}

test "the fold seed changes the partition, and repeating it reproduces one" {
    const gpa = testing.allocator;
    const labels = try gpa.alloc(f32, 500);
    defer gpa.free(labels);
    for (labels, 0..) |*y, i| y.* = if (i % 3 == 0) 1 else 0;

    const a = try assignFolds(gpa, labels, 5, 1, true);
    defer gpa.free(a);
    const b = try assignFolds(gpa, labels, 5, 1, true);
    defer gpa.free(b);
    const c = try assignFolds(gpa, labels, 5, 2, true);
    defer gpa.free(c);

    try testing.expectEqualSlices(u32, a, b);
    var same: usize = 0;
    for (a, c) |x, y| same += @intFromBool(x == y);
    // Two independent 5-way partitions agree on ~1/5 of rows by chance;
    // anything near total agreement would mean the seed is not being used.
    try testing.expect(same < labels.len / 2);
}

test "unstratified assignment still partitions completely" {
    const gpa = testing.allocator;
    const labels = try gpa.alloc(f32, 333);
    defer gpa.free(labels);
    for (labels) |*y| y.* = 0.5;

    const fold = try assignFolds(gpa, labels, 4, 11, false);
    defer gpa.free(fold);

    var count = [_]usize{0} ** 4;
    for (fold) |k| count[k] += 1;
    var total: usize = 0;
    for (count) |c| {
        try testing.expect(c == 83 or c == 84); // 333 does not divide by 4
        total += c;
    }
    try testing.expectEqual(@as(usize, 333), total);
}

test "grouped folds never split a group across folds" {
    const gpa = testing.allocator;
    const n = 1200;
    const groups = try gpa.alloc(f32, n);
    defer gpa.free(groups);
    // 300 groups of 4 rows each -- the panel shape this exists for.
    for (groups, 0..) |*g, i| g.* = @floatFromInt(i / 4);

    const fold = try assignGroupFolds(gpa, groups, 5, 3);
    defer gpa.free(fold);

    // Every row of a group shares its fold. A row-wise splitter puts the same
    // group in both halves, which is exactly the leak this prevents.
    var seen: std.AutoHashMapUnmanaged(u32, u32) = .empty;
    defer seen.deinit(gpa);
    for (groups, fold) |g, k| {
        const id: u32 = @intFromFloat(g);
        const e = try seen.getOrPut(gpa, id);
        if (e.found_existing) {
            try testing.expectEqual(e.value_ptr.*, k);
        } else e.value_ptr.* = k;
    }
    try testing.expectEqual(@as(usize, 300), seen.count());
}

test "grouped folds stay balanced when group sizes are uneven" {
    const gpa = testing.allocator;
    // Sizes 1..60: dealing these round-robin would leave one fold carrying
    // far more rows than another, so they are placed largest-first into
    // whichever fold is currently lightest.
    var list: std.ArrayList(f32) = .empty;
    defer list.deinit(gpa);
    for (1..61) |g| {
        for (0..g) |_| try list.append(gpa, @floatFromInt(g));
    }
    const fold = try assignGroupFolds(gpa, list.items, 5, 11);
    defer gpa.free(fold);

    var load = [_]usize{0} ** 5;
    for (fold) |k| load[k] += 1;
    var lo: usize = std.math.maxInt(usize);
    var hi: usize = 0;
    for (load) |v| {
        lo = @min(lo, v);
        hi = @max(hi, v);
    }
    // 1830 rows over 5 folds is 366 each; allow a little slack for the
    // largest group being indivisible.
    try testing.expect(hi - lo <= 60);
}

test "grouping is refused when there are fewer groups than folds" {
    const gpa = testing.allocator;
    const groups = [_]f32{ 1, 1, 2, 2, 3, 3 };
    try testing.expectError(
        error.FewerGroupsThanFolds,
        assignGroupFolds(gpa, &groups, 5, 1),
    );
}

test "the early-stopping slice is a tenth, stratified, and never splits a group" {
    const gpa = testing.allocator;
    const n = 2000;
    const labels = try gpa.alloc(f32, n);
    defer gpa.free(labels);
    const groups = try gpa.alloc(f32, n);
    defer gpa.free(groups);
    for (labels, groups, 0..) |*y, *g, i| {
        y.* = if (i % 4 == 0) 1 else 0;
        g.* = @floatFromInt(i / 5); // 400 groups of 5
    }
    const plain = try cv.assignEarlyStop(gpa, labels, null, 7, true);
    defer gpa.free(plain);
    var in_slice: usize = 0;
    var pos_in_slice: usize = 0;
    for (plain, labels) |p, y| if (p == 0) {
        in_slice += 1;
        if (y >= 0.5) pos_in_slice += 1;
    };
    try testing.expectEqual(@as(usize, n / cv.early_stop_parts), in_slice);
    try testing.expectEqual(in_slice / 4, pos_in_slice); // a quarter positive, like the data

    const grouped = try cv.assignEarlyStop(gpa, labels, groups, 7, true);
    defer gpa.free(grouped);
    for (0..n / 5) |gi| for (1..5) |j| try testing.expectEqual(grouped[gi * 5], grouped[gi * 5 + j]);
}
