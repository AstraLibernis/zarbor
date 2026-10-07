// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Split-search semantics pinned against XGBoost and LightGBM.
//!
//! Both were found by the 2026-10-07 reference audit (docs/parity.md): a
//! partition the scans could not reach, and a gain scale that made
//! `min_split_gain` mean twice XGBoost's `gamma`. Each test is built so the old
//! behaviour fails it.

const std = @import("std");
const data = @import("../data.zig");
const config = @import("../config.zig");
const booster = @import("../booster.zig");
const Pool = @import("../pool.zig").Pool;

const testing = std.testing;

/// A dataset from explicit column-major bins (bin 0 = missing) and labels.
/// Edges are 1, 2, ... so a real bin `b` stands for the value `b`.
pub fn fromBins(gpa: std.mem.Allocator, cols: []const []const u8, n_bins: []const u16, labels: []const f32) !data.Dataset {
    const n_features = cols.len;
    const n_rows = labels.len;

    const bins = try gpa.alloc(u8, n_features * n_rows);
    errdefer gpa.free(bins);
    for (cols, 0..) |c, f| @memcpy(bins[f * n_rows ..][0..n_rows], c);
    const bins_rm = try gpa.alloc(data.BinIdx, n_features * n_rows);
    errdefer gpa.free(bins_rm);
    for (0..n_rows) |r| for (0..n_features) |f| {
        bins_rm[r * n_features + f] = bins[f * n_rows + r];
    };
    const lab = try gpa.dupe(f32, labels);
    errdefer gpa.free(lab);
    const nb = try gpa.dupe(u16, n_bins);
    errdefer gpa.free(nb);

    const wide_cols = try gpa.alloc([]data.BinIdx, n_features);
    errdefer gpa.free(wide_cols);
    @memset(wide_cols, &.{});
    const kinds = try gpa.alloc(data.ColumnKind, n_features);
    errdefer gpa.free(kinds);
    @memset(kinds, .numeric);
    const levels = try gpa.alloc([][]u8, n_features);
    errdefer gpa.free(levels);
    @memset(levels, &.{});
    const means = try gpa.alloc([]f32, n_features);
    errdefer gpa.free(means);
    @memset(means, &.{});

    // Built one by one; on failure free what exists so far.
    const edges = try gpa.alloc([]f32, n_features);
    errdefer gpa.free(edges);
    const names = try gpa.alloc([]u8, n_features);
    errdefer gpa.free(names);
    var built: usize = 0;
    errdefer for (0..built) |f| {
        gpa.free(edges[f]);
        gpa.free(names[f]);
    };
    for (0..n_features) |f| {
        const e = try gpa.alloc(f32, n_bins[f] - 2);
        errdefer gpa.free(e);
        for (e, 0..) |*v, i| v.* = @floatFromInt(i + 1);
        names[f] = try std.fmt.allocPrint(gpa, "f{d}", .{f});
        edges[f] = e;
        built += 1;
    }

    return .{
        .gpa = gpa,
        .n_rows = n_rows,
        .n_features = n_features,
        .bins = bins,
        .wide_cols = wide_cols,
        .bins_rm = bins_rm,
        .n_bins = nb,
        .edges = edges,
        .means = means,
        .kinds = kinds,
        .names = names,
        .levels = levels,
        .labels = lab,
    };
}

/// One tree, one split at most, full step: the root decides everything.
fn stump(gpa: std.mem.Allocator, pool: *Pool, ds: *const data.Dataset, min_split_gain: f32) !booster.Model {
    const res = try booster.train(gpa, pool, ds, null, config.Config.from(.{
        .n_rounds = 1,
        .max_depth = 1,
        .learning_rate = 1.0,
        .min_split_gain = min_split_gain,
        .verbose_eval = 0,
    }).gbdt, null);
    return res.model;
}

test "a column that is only missing or one value can be split on" {
    // Missing-ness is the whole signal: bin 0 is mostly positive, bin 1 mostly
    // negative. With one real bin the old search skipped the feature (`nb < 3`)
    // and returned a stump.
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();

    const n = 1000;
    var col: [n]u8 = undefined;
    var y: [n]f32 = undefined;
    for (0..n) |i| {
        col[i] = if (i % 10 < 3) 0 else 1;
        // Two thirds positive when missing, one in seven when present.
        y[i] = if (col[i] == 0) @floatFromInt(@intFromBool(i % 10 != 0)) else @floatFromInt(@intFromBool(i % 10 == 9));
    }
    var ds = try fromBins(gpa, &.{&col}, &.{2}, &y);
    defer ds.deinit();

    var m = try stump(gpa, pool, &ds, 0);
    defer m.deinit();
    const root = m.trees.items[0].nodes[0];
    try testing.expect(!root.is_leaf);
    try testing.expectEqual(@as(data.BinIdx, 0), root.threshold);
    try testing.expect(root.missing_left);
}

test "missing-vs-present wins when missing-ness is the signal" {
    // Six real values carry no signal; being missing does. Every scan the old
    // search ran kept at least one real bin beside the missing mass, so it
    // could only approximate this partition.
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();

    const n = 1200;
    var col: [n]u8 = undefined;
    var y: [n]f32 = undefined;
    for (0..n) |i| {
        const is_missing = i % 4 == 0;
        col[i] = if (is_missing) 0 else @intCast(1 + (i / 4) % 6);
        const pos = if (is_missing) i % 10 != 0 else i % 10 == 0;
        y[i] = @floatFromInt(@intFromBool(pos));
    }
    var ds = try fromBins(gpa, &.{&col}, &.{8}, &y);
    defer ds.deinit();

    var m = try stump(gpa, pool, &ds, 0);
    defer m.deinit();
    const root = m.trees.items[0].nodes[0];
    try testing.expect(!root.is_leaf);
    try testing.expectEqual(@as(data.BinIdx, 0), root.threshold);
    try testing.expect(root.missing_left);
}

test "min_split_gain compares against the unhalved gain, as XGBoost's gamma does" {
    // Pinned by consequence: compute XGBoost's loss_chg for the only split by
    // hand, then require a split at 0.99x and none at 1.01x. With the old 0.5
    // factor the split already fails at 0.99x.
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();

    const n = 800;
    var col: [n]u8 = undefined;
    var y: [n]f32 = undefined;
    for (0..n) |i| {
        col[i] = if (i < n / 2) 1 else 2;
        // 30% positive on the left, 60% on the right.
        y[i] = @floatFromInt(@intFromBool(if (i < n / 2) i % 10 < 3 else i % 10 < 6));
    }
    var ds = try fromBins(gpa, &.{&col}, &.{3}, &y);
    defer ds.deinit();

    // Every row starts at the base score, so p, and hence h, is one number.
    var pos: f64 = 0;
    for (y) |v| pos += v;
    const p = pos / n;
    const h = p * (1 - p);
    const lambda: f64 = 1.0; // the default
    var gl: f64 = 0;
    var gr: f64 = 0;
    for (0..n) |i| {
        if (col[i] == 1) gl += p - y[i] else gr += p - y[i];
    }
    const hl = h * (n / 2);
    const hr = h * (n / 2);
    const score = struct {
        fn f(g: f64, hh: f64, l: f64) f64 {
            return g * g / (hh + l);
        }
    }.f;
    const loss_chg = score(gl, hl, lambda) + score(gr, hr, lambda) - score(gl + gr, hl + hr, lambda);

    var below = try stump(gpa, pool, &ds, @floatCast(0.99 * loss_chg));
    defer below.deinit();
    try testing.expect(!below.trees.items[0].nodes[0].is_leaf);

    var above = try stump(gpa, pool, &ds, @floatCast(1.01 * loss_chg));
    defer above.deinit();
    try testing.expect(above.trees.items[0].nodes[0].is_leaf);
}
