// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! CSV ingest and quantisation. Every column becomes bin indices up front so
//! the design stays in cache (668k rows x 13 features: 17 MB as bins, 35 MB
//! as f32). Split finding uses bins; raw values only report thresholds.

const std = @import("std");
const Pool = @import("pool.zig").Pool;

/// `u16` so categoricals can exceed 255 levels (ZIP3, city, metro). 0.98-1.01x
/// of `u8` in the accumulation kernel (bound by the scattered histogram
/// update), so it costs memory only. See docs/wide-categoricals.md.
pub const BinIdx = u16;

/// Hard ceiling from the index type; `n_bins` (`u16`) must hold it.
pub const max_bins = std.math.maxInt(BinIdx);

pub const csv = @import("csv.zig");

/// Strategy for choosing bin edges when quantising a numeric column.
pub const BinPolicy = enum {
    /// Equal-count bins from the empirical distribution. Robust to skew.
    quantile,
    /// Equal-width bins between min and max. Cheaper, worse on skewed data.
    uniform,
    /// LightGBM's `GreedyFindBin`: a value with a bin's worth of rows gets its
    /// own bin, the rest share the budget, so a 92%-one-value column spends
    /// its cuts on the other 8% rather than the mode.
    greedy,
};

/// How `quantise` turns columns into bins. Shared by every model.
pub const BinParams = struct {
    bin_policy: BinPolicy = .quantile,
    /// Rows a bin must hold under `greedy` before a cut. LightGBM's `min_data_in_bin`.
    min_data_in_bin: u32 = 3,
    /// Bins per feature. Capped at 256 so the column-major `bins` stays one
    /// byte per bin, which keeps the feature matrix inside L3.
    max_bin: u16 = 256,
    /// Categorical cardinality refused outright, so a free-text column cannot
    /// become an unaffordable histogram. Default 255 (the old `u8` limit) so
    /// widening `BinIdx` alone changes no existing run.
    max_cat_levels: u32 = 255,

    pub fn validate(p: BinParams) !void {
        if (p.max_bin < 2 or p.max_bin > 256) return error.BadMaxBin;
    }
};

/// Re-exported from `csv.zig`, `label_encoder.zig`, `schema.zig` so call
/// sites did not move when the code did.
pub const ColumnKind = csv.ColumnKind;
pub const Frame = csv.Frame;
pub const readCsv = csv.readCsv;

const label_encoder = @import("label_encoder.zig");
pub const LabelEncoder = label_encoder.LabelEncoder;
pub const LabelSpec = label_encoder.LabelSpec;
const schema = @import("schema.zig");
pub const Schema = schema.Schema;
pub const applySchema = schema.applySchema;
pub const explainWidth = schema.explainWidth;
const bin_edges = @import("bin_edges.zig");
pub const BinCtx = bin_edges.BinCtx;
pub const binMidpoint = bin_edges.binMidpoint;
pub const lowerBound = bin_edges.lowerBound;

// ------------------------------------------------------------------ binning

/// A quantised training matrix. Bin 0 is "missing", real values start at 1;
/// that lets split finding pick a default direction per split.
pub const Dataset = struct {
    gpa: std.mem.Allocator,
    n_rows: usize,
    n_features: usize,
    /// Column-major, one byte per bin: `bins[f * n_rows + r]`, read by partition
    /// one feature over scattered rows. Too-wide columns go to `wide_cols`.
    /// The 51 -> 110 ms partition slowdown once blamed on `u16` bins was
    /// `goesLeft` taking a 160-byte `Split` by value; the bin width "was never
    /// the cost" (docs/wide-categoricals.md, "Three wrong guesses").
    bins: []u8,
    /// Column-major bins for features with more than 256 bins; empty otherwise.
    wide_cols: [][]BinIdx,
    /// Row-major copy: `bins_rm[r * n_features + f]`. Histogram building gains
    /// 1.74x from a contiguous row; partition would touch 13x the cache lines
    /// here. Cost: 2 bytes per feature per row (13 features: 17.4 MB on 668k
    /// rows; the 93 MB peak was measured at 1 byte) and one transpose at load.
    bins_rm: []BinIdx,
    /// Bins actually in use per feature, including the missing bin.
    n_bins: []u16,
    /// `edges[f][i]`: inclusive upper bound of real-value bin `i`; length `n_bins[f] - 2`.
    edges: [][]f32,
    /// `means[f][b]`: mean training value in bin `b` of numeric `f` (slot 0:
    /// column mean); empty for categoricals. Read only by the linear model:
    /// the edge midpoint is biased for skewed bins, the norm under quantile
    /// cuts on heavy tails. Trees compare bin indices only.
    means: [][]f32,
    kinds: []ColumnKind,
    names: [][]u8,
    /// Categorical level strings by dictionary id; empty for numeric. A bin
    /// *is* its first-appearance id, so without these a second file cannot be
    /// binned the same way and a saved model cannot score new data.
    levels: [][][]u8,
    /// Empty when the frame carried no target column.
    labels: []f32,

    pub fn deinit(d: *Dataset) void {
        const gpa = d.gpa;
        gpa.free(d.bins);
        for (d.wide_cols) |c| if (c.len != 0) gpa.free(c);
        gpa.free(d.wide_cols);
        gpa.free(d.bins_rm);
        gpa.free(d.n_bins);
        for (d.edges) |e| gpa.free(e);
        gpa.free(d.edges);
        for (d.means) |m| if (m.len != 0) gpa.free(m);
        gpa.free(d.means);
        gpa.free(d.kinds);
        for (d.names) |n| gpa.free(n);
        gpa.free(d.names);
        for (d.levels) |ls| {
            for (ls) |l| gpa.free(l);
            gpa.free(ls);
        }
        gpa.free(d.levels);
        if (d.labels.len != 0) gpa.free(d.labels);
        d.* = undefined;
    }

    /// True when feature `f`'s bins do not fit in a byte.
    pub inline fn isWide(d: *const Dataset, f: usize) bool {
        return d.wide_cols[f].len != 0;
    }

    /// Asserts, not optional: both callers dispatch on `isWide` first, and a
    /// silent wrong answer is a mis-partitioned tree.
    pub inline fn columnNarrow(d: *const Dataset, f: usize) []const u8 {
        std.debug.assert(!d.isWide(f));
        return d.bins[f * d.n_rows ..][0..d.n_rows];
    }

    pub inline fn columnWide(d: *const Dataset, f: usize) []const BinIdx {
        return d.wide_cols[f];
    }

    /// Largest bin count across features; sizes the histogram allocation.
    pub fn maxBins(d: *const Dataset) u16 {
        var m: u16 = 0;
        for (d.n_bins) |b| m = @max(m, b);
        return m;
    }
};

/// Fills the row-major mirror. Parallel over row blocks: sequential column
/// reads, one contiguous write run; by-feature would scatter every write.
const TransposeCtx = struct {
    bins: []const BinIdx,
    out: []BinIdx,
    n_rows: usize,
    n_features: usize,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *TransposeCtx = @ptrCast(@alignCast(ctx));
        const nf = self.n_features;
        for (0..nf) |f| {
            const col = self.bins[f * self.n_rows ..][0..self.n_rows];
            var r = begin;
            while (r < end) : (r += 1) self.out[r * nf + f] = col[r];
        }
    }
};

/// Split the `BinIdx` scratch into the byte matrix partition reads plus
/// per-feature wide columns. The only place that knows which are wide.
pub fn splitByWidth(
    gpa: std.mem.Allocator,
    scratch: []const BinIdx,
    n_bins: []const u16,
    n_rows: usize,
    n_features: usize,
) !struct { narrow: []u8, wide: [][]BinIdx } {
    const narrow = try gpa.alloc(u8, n_features * n_rows);
    errdefer gpa.free(narrow);
    const wide = try gpa.alloc([]BinIdx, n_features);
    errdefer gpa.free(wide);
    @memset(wide, &.{});
    var made: usize = 0;
    errdefer for (wide[0..made]) |c| if (c.len != 0) gpa.free(c);

    while (made < n_features) : (made += 1) {
        const src = scratch[made * n_rows ..][0..n_rows];
        if (n_bins[made] <= 256) {
            const dst = narrow[made * n_rows ..][0..n_rows];
            for (src, dst) |v, *d| d.* = @intCast(v);
        } else {
            wide[made] = try gpa.dupe(BinIdx, src);
            @memset(narrow[made * n_rows ..][0..n_rows], 0);
        }
    }
    return .{ .narrow = narrow, .wide = wide };
}

pub fn buildRowMajor(
    gpa: std.mem.Allocator,
    pool: *Pool,
    bins: []const BinIdx,
    n_rows: usize,
    n_features: usize,
) ![]BinIdx {
    const out = try gpa.alloc(BinIdx, n_rows * n_features);
    errdefer gpa.free(out);
    var ctx = TransposeCtx{ .bins = bins, .out = out, .n_rows = n_rows, .n_features = n_features };
    pool.parallelFor(n_rows, &ctx, TransposeCtx.run, 4096);
    return out;
}

/// Quantise `src`. `label` names and encodes the target (excluded from
/// features); `skip` names columns to drop, e.g. an id.
pub fn quantise(
    gpa: std.mem.Allocator,
    pool: *Pool,
    src: *const Frame,
    cfg: BinParams,
    label: ?LabelSpec,
    skip: []const []const u8,
) !Dataset {
    var feats: std.ArrayList(usize) = .empty;
    defer feats.deinit(gpa);
    outer: for (0..src.names.len) |c| {
        if (label) |ls| if (c == ls.col) continue;
        for (skip) |s| if (std.mem.eql(u8, src.names[c], s)) continue :outer;
        try feats.append(gpa, c);
    }
    const n_features = feats.items.len;
    if (n_features == 0) return error.NoFeatures;

    // Checked here, after `skip`: parallel binning collapses errors into
    // `BinningFailed` and cannot name the column.
    for (feats.items) |c| {
        if (src.kinds[c] == .categorical and src.levels[c].len > cfg.max_cat_levels)
            return error.CategoricalTooWide;
    }

    // Mutable, errdefer keyed on length: freed midway, then cleared so a
    // later failure does not double free.
    var bins = try gpa.alloc(BinIdx, n_features * src.n_rows);
    errdefer if (bins.len != 0) gpa.free(bins);
    const n_bins = try gpa.alloc(u16, n_features);
    errdefer gpa.free(n_bins);
    const edges = try gpa.alloc([]f32, n_features);
    errdefer gpa.free(edges);
    @memset(edges, &.{});
    // Free inner slices too: on `BinningFailed` other workers have allocated.
    errdefer for (edges) |e| if (e.len != 0) gpa.free(e);
    const means = try gpa.alloc([]f32, n_features);
    errdefer gpa.free(means);
    @memset(means, &.{});
    errdefer for (means) |m| if (m.len != 0) gpa.free(m);
    const kinds = try gpa.alloc(ColumnKind, n_features);
    errdefer gpa.free(kinds);
    const names = try gpa.alloc([]u8, n_features);
    errdefer gpa.free(names);
    var named: usize = 0;
    errdefer for (names[0..named]) |n| gpa.free(n);
    while (named < n_features) : (named += 1) {
        names[named] = try gpa.dupe(u8, src.names[feats.items[named]]);
        kinds[named] = src.kinds[feats.items[named]];
    }

    var ctx = BinCtx{
        .gpa = gpa,
        .src = src,
        .feature_cols = feats.items,
        .bins = bins,
        .n_bins = n_bins,
        .edges = edges,
        .means = means,
        .n_rows = src.n_rows,
        .max_cat_levels = cfg.max_cat_levels,
        .min_data_in_bin = cfg.min_data_in_bin,
        .max_bin = cfg.max_bin,
        .policy = cfg.bin_policy,
        .failed = .init(false),
    };
    // One feature per item: independent sort + linear pass, balances well.
    pool.parallelFor(n_features, &ctx, BinCtx.run, 1);
    if (ctx.failed.load(.monotonic)) return error.BinningFailed;

    const levels = try gpa.alloc([][]u8, n_features);
    errdefer gpa.free(levels);
    @memset(levels, &.{});
    errdefer for (levels) |ls| {
        for (ls) |l| if (l.len != 0) gpa.free(l);
        if (ls.len != 0) gpa.free(ls);
    };
    for (0..n_features) |f| {
        const col = feats.items[f];
        if (kinds[f] != .categorical) continue;
        const src_levels = src.levels[col];
        const copy = try gpa.alloc([]u8, src_levels.len);
        // Emptied before the fallible dupes so the errdefer walks valid slices.
        @memset(copy, &.{});
        levels[f] = copy;
        for (src_levels, copy) |from, *to| to.* = try gpa.dupe(u8, from);
    }

    const bins_rm = try buildRowMajor(gpa, pool, bins, src.n_rows, n_features);
    errdefer gpa.free(bins_rm);

    const split = try splitByWidth(gpa, bins, n_bins, src.n_rows, n_features);
    errdefer {
        gpa.free(split.narrow);
        for (split.wide) |c| if (c.len != 0) gpa.free(c);
        gpa.free(split.wide);
    }
    gpa.free(bins);
    bins = &.{};

    var labels: []f32 = &.{};
    errdefer if (labels.len != 0) gpa.free(labels);
    if (label) |ls| labels = try ls.enc.encode(gpa, src, ls.col);

    return .{
        .gpa = gpa,
        .n_rows = src.n_rows,
        .n_features = n_features,
        .bins = split.narrow,
        .wide_cols = split.wide,
        .bins_rm = bins_rm,
        .n_bins = n_bins,
        .edges = edges,
        .means = means,
        .kinds = kinds,
        .names = names,
        .levels = levels,
        .labels = labels,
    };
}

/// A dataset of only `rows`, sharing the source's edges. For train/validation
/// splits: a threshold must mean the same on both halves.
pub fn subset(gpa: std.mem.Allocator, ds: *const Dataset, rows: []const u32) !Dataset {
    const n = rows.len;
    const bins = try gpa.alloc(u8, ds.n_features * n);
    errdefer gpa.free(bins);
    const wide = try gpa.alloc([]BinIdx, ds.n_features);
    errdefer gpa.free(wide);
    @memset(wide, &.{});
    var wmade: usize = 0;
    errdefer for (wide[0..wmade]) |c| if (c.len != 0) gpa.free(c);
    while (wmade < ds.n_features) : (wmade += 1) {
        const f = wmade;
        const dst = bins[f * n ..][0..n];
        if (ds.isWide(f)) {
            const src = ds.columnWide(f);
            const w = try gpa.alloc(BinIdx, n);
            for (rows, w) |r, *d| d.* = src[r];
            wide[f] = w;
            @memset(dst, 0);
        } else {
            const src = ds.columnNarrow(f);
            for (rows, dst) |r, *d| d.* = src[r];
        }
    }

    // No transpose: each source row is contiguous, so short memcpys.
    const nf = ds.n_features;
    const bins_rm = try gpa.alloc(BinIdx, nf * n);
    errdefer gpa.free(bins_rm);
    for (rows, 0..) |r, i| @memcpy(bins_rm[i * nf ..][0..nf], ds.bins_rm[@as(usize, r) * nf ..][0..nf]);

    const n_bins = try gpa.dupe(u16, ds.n_bins);
    errdefer gpa.free(n_bins);
    const kinds = try gpa.dupe(ColumnKind, ds.kinds);
    errdefer gpa.free(kinds);

    const edges = try gpa.alloc([]f32, ds.edges.len);
    errdefer gpa.free(edges);
    @memset(edges, &.{});
    var e: usize = 0;
    errdefer for (edges[0..e]) |x| gpa.free(x);
    while (e < ds.edges.len) : (e += 1) edges[e] = try gpa.dupe(f32, ds.edges[e]);

    const means = try gpa.alloc([]f32, ds.means.len);
    errdefer gpa.free(means);
    @memset(means, &.{});
    var mn: usize = 0;
    errdefer for (means[0..mn]) |x| if (x.len != 0) gpa.free(x);
    while (mn < ds.means.len) : (mn += 1) means[mn] = try gpa.dupe(f32, ds.means[mn]);

    const names = try gpa.alloc([]u8, ds.names.len);
    errdefer gpa.free(names);
    var nm: usize = 0;
    errdefer for (names[0..nm]) |x| gpa.free(x);
    while (nm < ds.names.len) : (nm += 1) names[nm] = try gpa.dupe(u8, ds.names[nm]);

    const levels = try gpa.alloc([][]u8, ds.levels.len);
    errdefer gpa.free(levels);
    @memset(levels, &.{});
    var lv: usize = 0;
    errdefer for (levels[0..lv]) |ls| {
        for (ls) |l| gpa.free(l);
        gpa.free(ls);
    };
    while (lv < ds.levels.len) : (lv += 1) {
        if (ds.levels[lv].len == 0) continue;
        const copy = try gpa.alloc([]u8, ds.levels[lv].len);
        for (ds.levels[lv], copy) |from, *to| to.* = try gpa.dupe(u8, from);
        levels[lv] = copy;
    }

    var labels: []f32 = &.{};
    if (ds.labels.len != 0) {
        labels = try gpa.alloc(f32, n);
        for (rows, labels) |r, *l| l.* = ds.labels[r];
    }

    return .{
        .gpa = gpa,
        .n_rows = n,
        .n_features = ds.n_features,
        .bins = bins,
        .wide_cols = wide,
        .bins_rm = bins_rm,
        .n_bins = n_bins,
        .edges = edges,
        .means = means,
        .kinds = kinds,
        .names = names,
        .levels = levels,
        .labels = labels,
    };
}
