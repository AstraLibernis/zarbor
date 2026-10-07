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

/// A sorted column as runs of equal values: `values[j]` ascending and distinct,
/// `ends[j]` the number of rows with value <= `values[j]`, so value `j` owns
/// rows `start(j)..ends[j]` of the sorted column.
///
/// Every policy below places cuts between runs, never inside one: a cut inside
/// a run would put equal values in different bins, which no split could use.
/// Each emits *boundaries*, a boundary `j` meaning "between `values[j]` and
/// `values[j + 1]`", ascending and at most `values.len - 2`; `edgesFrom` turns
/// them into edges. Policy and edge placement stay separate, so the rule for
/// where an edge sits between two training values is written once.
pub const Runs = struct {
    values: []f32,
    ends: []usize,

    /// `sorted` ascending, no NaN. Caller owns the result (`deinit`).
    pub fn of(gpa: std.mem.Allocator, sorted: []const f32) !Runs {
        var values: std.ArrayList(f32) = .empty;
        errdefer values.deinit(gpa);
        var ends: std.ArrayList(usize) = .empty;
        errdefer ends.deinit(gpa);
        for (sorted, 0..) |v, i| {
            // `==` merges -0.0 with +0.0, which compare equal everywhere a bin is decided.
            if (values.items.len != 0 and values.items[values.items.len - 1] == v) {
                ends.items[ends.items.len - 1] = i + 1;
            } else {
                try values.append(gpa, v);
                try ends.append(gpa, i + 1);
            }
        }
        const vs = try values.toOwnedSlice(gpa);
        errdefer gpa.free(vs);
        return .{ .values = vs, .ends = try ends.toOwnedSlice(gpa) };
    }

    pub fn deinit(r: *Runs, gpa: std.mem.Allocator) void {
        gpa.free(r.values);
        gpa.free(r.ends);
        r.* = undefined;
    }

    pub fn len(r: Runs) usize {
        return r.values.len;
    }

    pub fn start(r: Runs, j: usize) usize {
        return if (j == 0) 0 else r.ends[j - 1];
    }

    pub fn count(r: Runs, j: usize) usize {
        return r.ends[j] - r.start(j);
    }

    /// Rows in runs `lo..hi`.
    fn rows(r: Runs, lo: usize, hi: usize) usize {
        return if (hi <= lo) 0 else r.ends[hi - 1] - r.start(lo);
    }
};

/// One bin per distinct value, a value with fewer than `min_data_in_bin` rows
/// merging forward into the next (LightGBM's `GreedyFindBin` when every value
/// fits). The floor every policy gets when the column has no more distinct
/// values than the budget: it refines any other binning of the column, so it
/// can only add splits, and it costs no histogram width beyond the budget.
/// XGBoost, LightGBM, CatBoost and scikit-learn all bin a short column this way.
/// Runs `lo..hi` only, so `greedy` can apply it to one sign at a time.
fn perValue(gpa: std.mem.Allocator, r: Runs, lo: usize, hi: usize, min_data_in_bin: u32, out: *std.ArrayList(u32)) !void {
    if (hi <= lo + 1) return;
    var cur: usize = 0;
    for (lo..hi - 1) |j| {
        cur += r.count(j);
        if (cur < min_data_in_bin) continue;
        try out.append(gpa, @intCast(j));
        cur = 0;
    }
}

/// Equal-count bins by XGBoost's `QueryCutValues` (v3.4.1 src/common/quantile.h)
/// on an exact summary. For k = 1 .. budget-1 the target rank `k * n / budget`
/// picks the run whose rank range it falls nearest (by the runs' mid-ranks),
/// and that run *starts* bin k. When the run is already a bin start, the cut
/// advances to the next run instead of being dropped: a run holding many
/// targets is an atom, and dropping the repeats used to leave a 90%-one-value
/// column a handful of bins for the other 10%.
fn quantile(gpa: std.mem.Allocator, r: Runs, budget: usize, out: *std.ArrayList(u32)) !void {
    const nd = r.len();
    const n: f64 = @floatFromInt(r.ends[nd - 1]);
    const lo = struct {
        fn f(rs: Runs, j: usize) f64 {
            return @floatFromInt(rs.start(j));
        }
    }.f;
    const hi = struct {
        fn f(rs: Runs, j: usize) f64 {
            return @floatFromInt(rs.ends[j]);
        }
    }.f;
    var q: usize = 0; // query cursor, never moves back
    var last: usize = 0; // run of the last bin start; run 0 starts bin 0
    for (1..budget) |k| {
        const rank2 = 2.0 * @as(f64, @floatFromInt(k)) * n / @as(f64, @floatFromInt(budget));
        while (q + 2 < nd and rank2 >= lo(r, q + 1) + hi(r, q + 1)) q += 1;
        var c = if (rank2 < hi(r, q) + lo(r, q + 1)) q else q + 1;
        if (c <= last) c = last + 1;
        if (c >= nd) break;
        try out.append(gpa, @intCast(c - 1)); // a bin starting at run c is a boundary after c - 1
        last = c;
    }
}

fn orderUsize(a: usize, b: usize) std.math.Order {
    return std.math.order(a, b);
}

/// LightGBM's `GreedyFindBin` (v4.7.0, src/io/bin.cpp) on runs `lo..hi` with
/// `budget` bins. A run with a bin's worth of rows gets its own bin and the
/// mean is recomputed over the rest, repeatedly; quantile-style cuts on adult's
/// 92%-zero `capital-gain` scored below LightGBM on that column alone
/// (docs/measurements.md). Where LightGBM's remaining-bin count reaches zero
/// and it divides by it, the old mean is kept.
fn greedyRange(
    gpa: std.mem.Allocator,
    r: Runs,
    lo: usize,
    hi: usize,
    max_bin: usize,
    min_data_in_bin: u32,
    out: *std.ArrayList(u32),
) !void {
    const nd = hi -| lo;
    if (nd <= 1 or max_bin < 2) return;
    if (nd <= max_bin) return perValue(gpa, r, lo, hi, min_data_in_bin, out);

    const total = r.rows(lo, hi);
    var budget = max_bin;
    if (min_data_in_bin > 0) budget = @max(1, @min(budget, total / min_data_in_bin));
    if (budget < 2) return;
    var mean_bin_size = @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(budget));

    var rest_bins: i64 = @intCast(budget);
    var rest_rows: i64 = @intCast(total);
    const big = try gpa.alloc(bool, nd);
    defer gpa.free(big);
    for (big, lo..) |*b, j| {
        b.* = @as(f64, @floatFromInt(r.count(j))) >= mean_bin_size;
        if (b.*) {
            rest_bins -= 1;
            rest_rows -= @intCast(r.count(j));
        }
    }
    if (rest_bins > 0) mean_bin_size = @as(f64, @floatFromInt(rest_rows)) / @as(f64, @floatFromInt(rest_bins));

    var n_bins: usize = 0;
    var cur: usize = 0;
    for (0..nd - 1) |i| {
        const j = lo + i;
        if (!big[i]) rest_rows -= @intCast(r.count(j));
        cur += r.count(j);
        const c: f64 = @floatFromInt(cur);
        const close = big[i] or c >= mean_bin_size or
            (big[i + 1] and c >= @max(1.0, mean_bin_size * 0.5));
        if (!close) continue;
        try out.append(gpa, @intCast(j));
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

/// LightGBM's `|x| <= kZeroThreshold` (include/LightGBM/meta.h).
const zero_threshold: f32 = 1e-35;

/// LightGBM's `FindBinWithZeroAsOneBin`: zero gets a bin of its own, and the
/// budget is split between negatives and positives by their share of the
/// nonzero rows, each side binned by `greedyRange`. LightGBM bins every
/// numeric column this way, so this is what makes `greedy` its binning rather
/// than an approximation of it. Empty bins (a zero bin with no zeros) need no
/// boundary, so the row partition matches LightGBM's without them.
fn greedy(gpa: std.mem.Allocator, r: Runs, budget: usize, min_data_in_bin: u32, out: *std.ArrayList(u32)) !void {
    const nd = r.len();
    var neg_end: usize = 0; // runs below -zero
    while (neg_end < nd and r.values[neg_end] <= -zero_threshold) neg_end += 1;
    var pos_start = neg_end; // first run above +zero
    while (pos_start < nd and r.values[pos_start] <= zero_threshold) pos_start += 1;

    const total = r.ends[nd - 1];
    const zeros = r.rows(neg_end, pos_start);
    var left_bins: usize = 0;
    if (neg_end > 0 and budget > 1) {
        const share = @as(f64, @floatFromInt(r.rows(0, neg_end))) / @as(f64, @floatFromInt(total - zeros));
        const left_budget: usize = @max(1, @as(usize, @intFromFloat(share * @as(f64, @floatFromInt(budget - 1)))));
        const before = out.items.len;
        try greedyRange(gpa, r, 0, neg_end, left_budget, min_data_in_bin, out);
        left_bins = out.items.len - before + 1;
        // The last negative bin closes below zero, whatever follows.
        if (neg_end < nd) try out.append(gpa, @intCast(neg_end - 1));
    }
    const right_budget = budget -| (1 + left_bins);
    if (pos_start < nd and right_budget > 0) {
        if (pos_start > neg_end) try out.append(gpa, @intCast(pos_start - 1)); // zeros end
        try greedyRange(gpa, r, pos_start, nd, right_budget, min_data_in_bin, out);
    }
}

/// CatBoost's default `GreedyLogSum` (v1.2.10, library/cpp/grid_creator/
/// binarization.cpp, `TFeatureBin` and `GreedySplit`). Each bin offers one
/// candidate: the run edge nearest its middle row, either end of the run that
/// holds it, the better of the two. A split scores
/// `log(L) + log(R) - log(L + R)` in rows, which favours even bins and splits
/// a big run off its small neighbours early. The best-scoring bin splits until
/// the budget is used or no bin holds two runs. Ties go to the leftmost bin,
/// where CatBoost's are heap order.
fn logsum(gpa: std.mem.Allocator, r: Runs, budget: usize, out: *std.ArrayList(u32)) !void {
    const Bin = struct {
        lo: usize, // runs lo..hi
        hi: usize,
        cut: usize, // best split: runs lo..cut | cut..hi; == lo when none
        score: f64,
    };
    const score = struct {
        fn f(l: usize, rr: usize) f64 {
            const a: f64 = @floatFromInt(l);
            const b: f64 = @floatFromInt(rr);
            return @log(a + 1e-8) + @log(b + 1e-8) - @log(a + b + 1e-8);
        }
    }.f;
    const best = struct {
        fn f(rs: Runs, lo: usize, hi: usize) Bin {
            var b: Bin = .{ .lo = lo, .hi = hi, .cut = lo, .score = -std.math.inf(f64) };
            if (hi <= lo + 1) return b;
            const row_lo = rs.start(lo);
            const row_hi = rs.ends[hi - 1];
            const mid = row_lo + (row_hi - row_lo) / 2;
            const jm = std.sort.upperBound(usize, rs.ends[lo..hi], mid, orderUsize) + lo;
            // Either end of the middle run; a cut at the bin's own edge is no split.
            for ([2]usize{ jm, jm + 1 }) |c| {
                if (c <= lo or c >= hi) continue;
                const s = score(rs.start(c) - row_lo, row_hi - rs.start(c));
                if (s > b.score) {
                    b.cut = c;
                    b.score = s;
                }
            }
            return b;
        }
    }.f;

    // Best score first, ties to the leftmost bin; a bin that cannot split scores -inf, so once one
    // is on top none can. A heap, not a scan: at a budget of thousands of bins a scan per split is
    // quadratic.
    const Queue = std.PriorityQueue(Bin, void, struct {
        fn order(_: void, a: Bin, c: Bin) std.math.Order {
            if (a.score > c.score) return .lt;
            if (a.score < c.score) return .gt;
            return std.math.order(a.lo, c.lo);
        }
    }.order);
    var queue = Queue.initContext({});
    defer queue.deinit(gpa);
    try queue.push(gpa, best(r, 0, r.len()));
    var n_bins: usize = 1;
    while (n_bins < budget) {
        const b = queue.pop() orelse break;
        if (b.cut == b.lo) {
            try queue.push(gpa, b);
            break;
        }
        try queue.push(gpa, best(r, b.lo, b.cut));
        try queue.push(gpa, best(r, b.cut, b.hi));
        n_bins += 1;
    }
    // Every bin but the first starts after a boundary.
    const first = out.items.len;
    while (queue.pop()) |b| if (b.lo != 0) try out.append(gpa, @intCast(b.lo - 1));
    std.sort.pdq(u32, out.items[first..], {}, std.sort.asc(u32));
}

/// Boundary `j` becomes the midpoint of `values[j]` and `values[j + 1]`, as an
/// inclusive upper edge. Every training value lands where it did with the edge
/// at `values[j]`; a value never seen in training falls to whichever side it is
/// nearer, as in LightGBM, CatBoost and scikit-learn (XGBoost sends it to the
/// lower). Where no f32 lies strictly between the two, or one is infinite, the
/// edge stays at `values[j]`.
fn edgesFrom(gpa: std.mem.Allocator, r: Runs, boundaries: []const u32, cuts: *std.ArrayList(f32)) !void {
    try cuts.ensureUnusedCapacity(gpa, boundaries.len);
    for (boundaries) |j| {
        const a = r.values[j];
        const b = r.values[j + 1];
        var e = a;
        if (std.math.isFinite(a) and std.math.isFinite(b)) {
            const m: f32 = @floatCast((@as(f64, a) + @as(f64, b)) * 0.5);
            if (m >= a and m < b) e = m;
        }
        cuts.appendAssumeCapacity(e);
    }
}

/// Edges for one numeric column: `present` holds its non-missing values (sorted
/// in place here), `budget` the real bins allowed. A column with no more
/// distinct values than the budget gets one bin per value whatever the policy
/// (`perValue`); otherwise the policy decides.
pub fn cutsFor(
    gpa: std.mem.Allocator,
    present: []f32,
    budget: usize,
    policy: BinPolicy,
    min_data_in_bin: u32,
    cuts: *std.ArrayList(f32),
) !void {
    std.sort.pdq(f32, present, {}, lessF32);
    var r = try Runs.of(gpa, present);
    defer r.deinit(gpa);
    const nd = r.len();
    if (nd <= 1 or budget < 2) return;

    var bounds: std.ArrayList(u32) = .empty;
    defer bounds.deinit(gpa);
    if (nd <= budget) {
        try perValue(gpa, r, 0, nd, min_data_in_bin, &bounds);
        return edgesFrom(gpa, r, bounds.items, cuts);
    }
    // `greedy` applies `min_data_in_bin` itself, per sign, as LightGBM does.
    // The other policies get the same floor here: no more bins than the rows
    // can fill, then any bin still short merges forward (`floorBounds`).
    const rows_cap = if (min_data_in_bin > 0) r.ends[nd - 1] / min_data_in_bin else budget;
    const capped = if (policy == .greedy) budget else @max(1, @min(budget, rows_cap));
    if (capped < 2) return;
    switch (policy) {
        .quantile => try quantile(gpa, r, capped, &bounds),
        .greedy => try greedy(gpa, r, budget, min_data_in_bin, &bounds),
        .logsum => try logsum(gpa, r, capped, &bounds),
        .uniform => {
            // Equal widths are not cuts between runs, so no boundaries: edges
            // directly, and only the bin-count cap applies, not the per-bin floor.
            const lo = r.values[0];
            const hi = r.values[nd - 1];
            const step = (hi - lo) / @as(f32, @floatFromInt(capped));
            for (1..capped) |k| {
                const c = lo + step * @as(f32, @floatFromInt(k));
                if (cuts.items.len != 0 and cuts.items[cuts.items.len - 1] >= c) continue;
                try cuts.append(gpa, c);
            }
            return;
        },
    }
    if (policy != .greedy) floorBounds(r, &bounds, min_data_in_bin);
    return edgesFrom(gpa, r, bounds.items, cuts);
}

/// Drops boundaries until every bin but the last holds at least
/// `min_data_in_bin` rows, a short bin merging forward as in `perValue`.
/// `bounds` are run indices that end a bin, ascending.
fn floorBounds(r: Runs, bounds: *std.ArrayList(u32), min_data_in_bin: u32) void {
    if (min_data_in_bin <= 1) return;
    var kept: usize = 0;
    var prev_end: usize = 0; // rows before the current bin
    for (bounds.items) |b| {
        const end = r.ends[b];
        if (end - prev_end < min_data_in_bin) continue;
        bounds.items[kept] = b;
        kept += 1;
        prev_end = end;
    }
    bounds.shrinkRetainingCapacity(kept);
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
    /// Categorical cardinality refused. Not an index-type limit: it stops a
    /// free-text column becoming a legal but unusable 50,000-bin histogram.
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
            // Rechecked here because this is where the cast is: a frame built
            // by any route must not reach it with an id that does not fit.
            // `max_bin` deliberately absent: it counts numeric quantile cuts; a
            // categorical's width is its cardinality alone.
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
        if (n != 0) try cutsFor(self.gpa, present, budget, self.policy, self.min_data_in_bin, &cuts);

        const edges = try cuts.toOwnedSlice(self.gpa);
        self.edges[f] = edges;
        self.n_bins[f] = @intCast(edges.len + 2);

        for (vals, out) |v, *b| {
            b.* = if (std.math.isNan(v)) 0 else @intCast(lowerBound(edges, v) + 1);
        }

        const nb: usize = edges.len + 2;
        const means = try self.gpa.alloc(f32, nb);
        // Published first so the caller's errdefer frees it on later failure.
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
        // Missing bin gets the column mean: zero once standardised.
        means[0] = if (seen == 0) 0 else @floatCast(all / @as(f64, @floatFromInt(seen)));
        for (means[1..], sums[1..], cnts[1..], 1..) |*m, sm, c, b| {
            m.* = if (c == 0)
                binMidpoint(edges, b - 1)
            else
                @floatCast(sm / @as(f64, @floatFromInt(c)));
        }
    }
};
