// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Bin edges: where a numeric column's cuts go, and quantising a column onto them.

const std = @import("std");
const csv = @import("csv.zig");
const Frame = csv.Frame;
const data = @import("data.zig");
const BinIdx = data.BinIdx;
const BinPolicy = data.BinPolicy;
const max_bins = data.max_bins;

/// The value a numeric bin stands for when no training value landed in it:
/// the midpoint of its edges, or the one finite edge for the two unbounded
/// end bins. `real_bin` is the bin index with the missing bin removed.
pub fn binMidpoint(edges: []const f32, real_bin: usize) f32 {
    if (edges.len == 0) return 0;
    if (real_bin == 0) return edges[0];
    if (real_bin >= edges.len) return edges[edges.len - 1];
    return 0.5 * (edges[real_bin - 1] + edges[real_bin]);
}

/// LightGBM's `GreedyFindBin` (v4.7.0, src/io/bin.cpp), which is the only
/// part of the binning that a plain quantile rule gets badly wrong.
///
/// A quantile rule places its cuts at fixed rank positions. On a column that
/// is 92% zeros -- `capital-gain` in the adult census set, say -- almost every
/// cut lands inside that mass, dedups against its neighbour, and the whole
/// budget collapses onto a handful of usable bins covering the 8% that
/// carries the signal. Measured: 0.6088 AUC against LightGBM's 0.6224 on that
/// column alone.
///
/// The fix is to notice the mode. Any distinct value holding at least a bin's
/// worth of rows takes a bin to itself; the remaining budget is then recomputed
/// over what is left, repeatedly, so the tail gets the resolution.
///
/// `present` must be sorted ascending. Cuts are inclusive upper bounds, which
/// is what `lowerBound` below expects -- LightGBM stores midpoints between
/// neighbouring distinct values, and the two assign every row to the same bin.
fn greedyBins(
    gpa: std.mem.Allocator,
    present: []f32,
    max_bin: usize,
    min_data_in_bin: u32,
    cuts: *std.ArrayList(f32),
) !void {
    std.sort.pdq(f32, present, {}, lessF32);
    const total: usize = present.len;
    if (total == 0 or max_bin < 2) return;

    // Run-length encode into distinct values and their counts.
    var distinct: std.ArrayList(f32) = .empty;
    defer distinct.deinit(gpa);
    var counts: std.ArrayList(usize) = .empty;
    defer counts.deinit(gpa);
    for (present) |v| {
        if (distinct.items.len != 0 and distinct.items[distinct.items.len - 1] == v) {
            counts.items[counts.items.len - 1] += 1;
        } else {
            try distinct.append(gpa, v);
            try counts.append(gpa, 1);
        }
    }
    const nd = distinct.items.len;
    if (nd <= 1) return;

    if (nd <= max_bin) {
        // Every distinct value can have its own bin, subject to the floor.
        var cur: usize = 0;
        for (0..nd - 1) |i| {
            cur += counts.items[i];
            if (cur < min_data_in_bin) continue;
            const val = distinct.items[i];
            if (cuts.items.len != 0 and cuts.items[cuts.items.len - 1] >= val) continue;
            try cuts.append(gpa, val);
            cur = 0;
        }
        return;
    }

    var budget = max_bin;
    if (min_data_in_bin > 0) budget = @max(1, @min(budget, total / min_data_in_bin));
    var mean_bin_size = @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(budget));

    var rest_bins: i64 = @intCast(budget);
    var rest_rows: i64 = @intCast(total);
    const big = try gpa.alloc(bool, nd);
    defer gpa.free(big);
    for (0..nd) |i| {
        big[i] = @as(f64, @floatFromInt(counts.items[i])) >= mean_bin_size;
        if (big[i]) {
            rest_bins -= 1;
            rest_rows -= @intCast(counts.items[i]);
        }
    }
    if (rest_bins > 0) mean_bin_size = @as(f64, @floatFromInt(rest_rows)) / @as(f64, @floatFromInt(rest_bins));

    var n_bins: usize = 0;
    var cur: usize = 0;
    for (0..nd - 1) |i| {
        if (!big[i]) rest_rows -= @intCast(counts.items[i]);
        cur += counts.items[i];
        const c: f64 = @floatFromInt(cur);
        const close = big[i] or c >= mean_bin_size or
            (big[i + 1] and c >= @max(1.0, mean_bin_size * 0.5));
        if (!close) continue;
        const val = distinct.items[i];
        if (cuts.items.len == 0 or cuts.items[cuts.items.len - 1] < val) try cuts.append(gpa, val);
        n_bins += 1;
        if (n_bins >= budget - 1) break;
        cur = 0;
        if (!big[i]) {
            rest_bins -= 1;
            if (rest_bins > 0)
                mean_bin_size = @as(f64, @floatFromInt(rest_rows)) / @as(f64, @floatFromInt(rest_bins));
        }
    }
}

pub fn lowerBound(edges: []const f32, v: f32) usize {
    var lo: usize = 0;
    var hi: usize = edges.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (edges[mid] < v) lo = mid + 1 else hi = mid;
    }
    return lo;
}

fn lessF32(_: void, a: f32, b: f32) bool {
    return a < b;
}

pub const BinCtx = struct {
    gpa: std.mem.Allocator,
    src: *const Frame,
    feature_cols: []const usize,
    bins: []BinIdx,
    n_bins: []u16,
    edges: [][]f32,
    means: [][]f32,
    n_rows: usize,
    max_bin: u16,
    min_data_in_bin: u32,
    /// Cardinality above which a categorical column is refused. Not a limit of
    /// the index type -- it stops a free-text column becoming a 50,000-bin
    /// histogram that would be legal and unusable.
    max_cat_levels: u32,
    policy: BinPolicy,
    failed: std.atomic.Value(bool),

    pub fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *BinCtx = @ptrCast(@alignCast(ctx));
        var f = begin;
        while (f < end) : (f += 1) {
            self.binOne(f) catch self.failed.store(true, .monotonic);
        }
    }

    fn binOne(self: *BinCtx, f: usize) !void {
        const col = self.feature_cols[f];
        const vals = self.src.values[col];
        const out = self.bins[f * self.n_rows ..][0..self.n_rows];

        if (self.src.kinds[col] == .categorical) {
            // Ids are already dense and small; bin j+1 is level j.
            const card = self.src.levels[col].len;
            // Checked here as well as at dictionary build time, because this
            // is where the cast is: a frame assembled by any other route must
            // not be able to reach it with an id that does not fit. The
            // configured limit is the one that usually bites -- it exists to
            // stop a free-text column becoming a 50,000-bin histogram, not
            // because the index cannot hold it.
            // `max_bin` deliberately does not appear here. It is the number of
            // quantile cuts a *numeric* column gets; a categorical's width is
            // its cardinality and nothing else decides it.
            if (card >= max_bins or card > self.max_cat_levels) return error.CategoricalTooWide;
            for (vals, out) |v, *b| {
                b.* = if (std.math.isNan(v)) 0 else @intCast(@as(usize, @intFromFloat(v)) + 1);
            }
            self.edges[f] = &.{};
            self.n_bins[f] = @intCast(card + 1);
            return;
        }

        // Real bins available after reserving bin 0 for missing.
        const budget: usize = @min(@as(usize, self.max_bin) - 1, self.n_rows);

        var sorted = try self.gpa.alloc(f32, vals.len);
        defer self.gpa.free(sorted);
        var n: usize = 0;
        for (vals) |v| {
            if (!std.math.isNan(v)) {
                sorted[n] = v;
                n += 1;
            }
        }
        const present = sorted[0..n];

        var cuts: std.ArrayList(f32) = .empty;
        errdefer cuts.deinit(self.gpa);

        if (n != 0) switch (self.policy) {
            .quantile => {
                std.sort.pdq(f32, present, {}, lessF32);
                var k: usize = 1;
                while (k < budget) : (k += 1) {
                    const idx = (k * n) / budget;
                    if (idx == 0 or idx >= n) continue;
                    const c = present[idx];
                    // Dedup: a repeated cut means the distribution has an atom
                    // there, and an empty bin would only waste histogram space.
                    if (cuts.items.len != 0 and cuts.items[cuts.items.len - 1] >= c) continue;
                    try cuts.append(self.gpa, c);
                }
            },
            .greedy => try greedyBins(self.gpa, present, budget, self.min_data_in_bin, &cuts),
            .uniform => {
                var lo = present[0];
                var hi = present[0];
                for (present) |v| {
                    lo = @min(lo, v);
                    hi = @max(hi, v);
                }
                if (hi > lo) {
                    const step = (hi - lo) / @as(f32, @floatFromInt(budget));
                    var k: usize = 1;
                    while (k < budget) : (k += 1) {
                        const c = lo + step * @as(f32, @floatFromInt(k));
                        if (cuts.items.len != 0 and cuts.items[cuts.items.len - 1] >= c) continue;
                        try cuts.append(self.gpa, c);
                    }
                }
            },
        };

        const edges = try cuts.toOwnedSlice(self.gpa);
        self.edges[f] = edges;
        self.n_bins[f] = @intCast(edges.len + 2);

        for (vals, out) |v, *b| {
            b.* = if (std.math.isNan(v)) 0 else @intCast(lowerBound(edges, v) + 1);
        }

        const nb: usize = edges.len + 2;
        const means = try self.gpa.alloc(f32, nb);
        // Published before the two fallible allocations below, so a failure
        // there leaves it for the caller's errdefer to free.
        self.means[f] = means;
        const sums = try self.gpa.alloc(f64, nb);
        defer self.gpa.free(sums);
        const cnts = try self.gpa.alloc(u32, nb);
        defer self.gpa.free(cnts);
        @memset(sums, 0);
        @memset(cnts, 0);
        for (vals, out) |v, b| {
            if (std.math.isNan(v)) continue;
            sums[b] += v;
            cnts[b] += 1;
        }
        var all: f64 = 0;
        var seen: u64 = 0;
        for (sums[1..], cnts[1..]) |sm, c| {
            all += sm;
            seen += c;
        }
        // Bin 0 is missing and holds no values of its own. The column mean
        // puts a missing entry at the centre, where it contributes nothing
        // once the column is standardised.
        means[0] = if (seen == 0) 0 else @floatCast(all / @as(f64, @floatFromInt(seen)));
        for (means[1..], sums[1..], cnts[1..], 1..) |*m, sm, c, b| {
            m.* = if (c == 0)
                binMidpoint(edges, b - 1)
            else
                @floatCast(sm / @as(f64, @floatFromInt(c)));
        }
    }
};
