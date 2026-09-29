// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! CSV ingest and quantisation.
//!
//! The booster never sees raw feature values. Every column is reduced to a
//! bin index up front, which is what lets the whole design stay in cache:
//! 668k rows x 13 features is 17 MB as bins, versus 35 MB as f32. Split
//! finding then works on bin indices alone and touches the original values
//! only to report thresholds.

const std = @import("std");
const Pool = @import("pool.zig").Pool;

/// A bin index. Was `u8`, which capped a categorical column at 255 levels and
/// put ZIP3, city and metro names out of reach entirely. Measured at
/// 0.98-1.01x of `u8` in the accumulation kernel -- that loop is bound by the
/// scattered histogram update, not by reading the index -- so the width is
/// paid in memory only. See docs/wide-categoricals.md.
pub const BinIdx = u16;

/// Hard ceiling, set by the index type. `n_bins` is a `u16` and must be able
/// to hold a count this large.
pub const max_bins = std.math.maxInt(BinIdx);

pub const csv = @import("csv.zig");

/// Strategy for choosing bin edges when quantising a numeric column.
pub const BinPolicy = enum {
    /// Equal-count bins from the empirical distribution. Robust to skew.
    quantile,
    /// Equal-width bins between min and max. Cheaper, worse on skewed data.
    uniform,
    /// LightGBM's `GreedyFindBin`. Any distinct value carrying at least a
    /// bin's worth of rows gets a bin to itself, and the remaining budget is
    /// spread over what is left -- so a column that is 92% one value spends
    /// its cuts on the other 8% instead of collapsing them all onto the mode.
    greedy,
};

/// How `quantise` turns columns into bins. Shared by every model.
pub const BinParams = struct {
    bin_policy: BinPolicy = .quantile,
    /// Rows a bin must hold under `greedy` before a cut is placed after it.
    /// LightGBM's `min_data_in_bin`.
    min_data_in_bin: u32 = 3,
    /// Bins per feature. Capped at 256 because bins are stored as u8, which
    /// is what keeps the feature matrix inside L3.
    max_bin: u16 = 256,
    /// Cardinality above which a categorical column is refused outright. The
    /// bin index can hold far more; this exists so a free-text column cannot
    /// turn into a histogram nobody can afford. 255 is what the `u8` bin used
    /// to enforce, and is kept as the default so widening the index changes
    /// no existing run on its own.
    max_cat_levels: u32 = 255,

    pub fn validate(p: BinParams) !void {
        if (p.max_bin < 2 or p.max_bin > 256) return error.BadMaxBin;
    }
};

/// Parsing (`csv.zig`), label encoding (`label_encoder.zig`) and the saved
/// binning (`schema.zig`) live in their own files. Re-exported here because
/// every caller of this module needs them; moving the code out should not
/// move every call site.
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

/// A quantised training matrix.
///
/// Bin 0 of every feature is reserved for "missing". Real values start at 1.
/// Keeping missing in its own bin is what lets split finding choose a default
/// direction per split instead of imposing one globally.
pub const Dataset = struct {
    gpa: std.mem.Allocator,
    n_rows: usize,
    n_features: usize,
    /// Column-major: `bins[f * n_rows + r]`. Partitioning a node reads one
    /// feature for every row, so that pass wants the column contiguous.
    /// Column-major, **one byte per bin**: `bins[f * n_rows + r]`.
    ///
    /// Partitioning reads one feature down a node's rows, and those rows are
    /// scattered once the tree is more than a level deep, so each one tends to
    /// want its own cache line and the element size is paid in full. Measured
    /// when the whole matrix went to `u16`: partition went from 51 ms to
    /// 110 ms on adult while the row-major accumulate did not move at all.
    /// So this stays a byte, and a column too wide for one lives in
    /// `wide_cols` instead. See docs/wide-categoricals.md.
    bins: []u8,
    /// Column-major storage for features whose bin count exceeds 256. Empty
    /// slice for every other feature, which is almost all of them.
    wide_cols: [][]BinIdx,
    /// The same bins row-major: `bins_rm[r * n_features + f]`.
    ///
    /// Both layouts are kept because the two hot passes want opposite things.
    /// Histogram building reads every selected feature of a row and gains
    /// 1.74x from having them contiguous; partitioning reads a single feature
    /// down the rows and would touch thirteen times the cache lines in that
    /// layout. The copy costs 13 bytes a row -- 8.7 MB on 668k rows, against
    /// a 93 MB peak -- and one transpose pass at load time.
    bins_rm: []BinIdx,
    /// Bins actually in use per feature, including the missing bin.
    n_bins: []u16,
    /// Cut points per feature; `edges[f][i]` is the inclusive upper bound of
    /// real-value bin `i`. Length is `n_bins[f] - 2`.
    edges: [][]f32,
    /// `means[f][b]` is the mean of the training values that landed in bin `b`
    /// of numeric feature `f`; empty for categorical features. Slot 0 is the
    /// missing bin and holds the column mean instead.
    ///
    /// Only the linear model reads this: it has to pick one number to stand
    /// for a whole bin, and the midpoint of the bin's two edges is a biased
    /// stand-in whenever the values inside are skewed -- which, under quantile
    /// cuts on a heavy-tailed column, they usually are. Trees never care,
    /// since they only ever compare bin indices.
    means: [][]f32,
    kinds: []ColumnKind,
    names: [][]u8,
    /// Level strings for each categorical feature, indexed by dictionary id;
    /// empty for numeric features.
    ///
    /// Retained because a categorical bin *is* its dictionary id, and those
    /// ids are assigned in order of first appearance. Without the strings, a
    /// second file cannot be binned the same way as the first, and a saved
    /// model could not score anything it had not been trained on.
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

    /// Column-major bins for a narrow feature. Asserting rather than
    /// returning an optional: the two callers both dispatch on `isWide`
    /// first, and a silent wrong answer here is a mis-partitioned tree.
    pub inline fn columnNarrow(d: *const Dataset, f: usize) []const u8 {
        std.debug.assert(!d.isWide(f));
        return d.bins[f * d.n_rows ..][0..d.n_rows];
    }

    pub inline fn columnWide(d: *const Dataset, f: usize) []const BinIdx {
        return d.wide_cols[f];
    }

    /// Every feature's bin for one row, contiguous.
    pub inline fn row(d: *const Dataset, r: usize) []const u8 {
        return d.bins_rm[r * d.n_features ..][0..d.n_features];
    }

    /// Largest bin count across features; sizes the histogram allocation.
    pub fn maxBins(d: *const Dataset) u16 {
        var m: u16 = 0;
        for (d.n_bins) |b| m = @max(m, b);
        return m;
    }
};

/// Fills the row-major mirror of a column-major bin matrix.
///
/// Parallel over row blocks: each worker reads `n_features` column streams
/// sequentially and writes one contiguous run, which the prefetcher handles
/// on both sides. Transposing by feature instead would scatter every write.
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

/// Split a wide column-major scratch matrix into the byte mirror the
/// partition reads and per-feature overrides for the columns that do not fit.
///
/// The scratch is what binning produces and what the row-major transpose
/// consumes; neither wants to know which columns are wide. This is the one
/// place that does.
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

/// Quantise `src` into a `Dataset`. `label`, when given, names the target
/// column and how to encode it; that column is excluded from the feature set.
/// `skip` names columns to drop (an id column, typically).
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

    // Width is checked here, on the columns that survived `skip`, rather than
    // inside the parallel binning below -- which collapses every worker error
    // into one `BinningFailed` and so cannot say which column was at fault.
    for (feats.items) |c| {
        if (src.kinds[c] == .categorical and src.levels[c].len > cfg.max_cat_levels)
            return error.CategoricalTooWide;
    }

    // Mutable, and the errdefer guards on length: the scratch is handed back
    // to the allocator partway through this function, and an `errdefer` on a
    // pointer that has already been freed is a double free on any later
    // failure. Clearing the slice is how the guard is told it is spent.
    var bins = try gpa.alloc(BinIdx, n_features * src.n_rows);
    errdefer if (bins.len != 0) gpa.free(bins);
    const n_bins = try gpa.alloc(u16, n_features);
    errdefer gpa.free(n_bins);
    const edges = try gpa.alloc([]f32, n_features);
    errdefer gpa.free(edges);
    @memset(edges, &.{});
    // The workers below fill these, and if one of them fails the others have
    // already allocated. Freeing only the outer array leaked every numeric
    // feature's cut points on the `BinningFailed` path.
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
    // One feature per work item: each is an independent sort plus a linear
    // assignment pass, which balances well across cores.
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
        // Published empty *before* the fallible dupes, so a failure partway
        // leaves the errdefer above a well-formed array to walk rather than
        // uninitialised pointers.
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

/// A new dataset holding only `rows`, sharing the source's bin edges.
///
/// Used for train/validation splits: both halves must be quantised with the
/// same cut points, or a threshold learned on one would mean something
/// different on the other.
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

    // The row-major mirror needs no transpose here: one selected row is
    // already contiguous in the source, so this is a run of short memcpys.
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
