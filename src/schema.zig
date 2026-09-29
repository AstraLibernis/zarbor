// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! The binning a trained model carries, and how new data is binned with it.

const std = @import("std");
const Pool = @import("pool.zig").Pool;
const csv = @import("csv.zig");
const ColumnKind = csv.ColumnKind;
const Frame = csv.Frame;
const data = @import("data.zig");
const BinIdx = data.BinIdx;
const Dataset = data.Dataset;
const LabelSpec = data.LabelSpec;
pub const buildRowMajor = data.buildRowMajor;
pub const lowerBound = data.lowerBound;
pub const splitByWidth = data.splitByWidth;

/// Everything needed to bin a new table exactly as a training table was.
///
/// This is what makes a saved model usable: bin edges alone are not enough,
/// because a categorical bin *is* its dictionary id and those ids are assigned
/// in order of first appearance. Scoring a second file without the original
/// level strings would map "Male" to whichever id that file happened to give
/// it, silently producing a different model input.
pub const Schema = struct {
    gpa: std.mem.Allocator,
    n_features: usize,
    names: [][]u8,
    kinds: []ColumnKind,
    n_bins: []u16,
    edges: [][]f32,
    levels: [][][]u8,

    pub fn deinit(s: *Schema) void {
        const gpa = s.gpa;
        for (s.names) |n| gpa.free(n);
        gpa.free(s.names);
        gpa.free(s.kinds);
        gpa.free(s.n_bins);
        for (s.edges) |e| gpa.free(e);
        gpa.free(s.edges);
        for (s.levels) |ls| {
            for (ls) |l| gpa.free(l);
            gpa.free(ls);
        }
        gpa.free(s.levels);
        s.* = undefined;
    }

    pub fn fromDataset(gpa: std.mem.Allocator, ds: *const Dataset) !Schema {
        var s = Schema{
            .gpa = gpa,
            .n_features = ds.n_features,
            .names = try gpa.alloc([]u8, ds.n_features),
            .kinds = try gpa.dupe(ColumnKind, ds.kinds),
            .n_bins = try gpa.dupe(u16, ds.n_bins),
            .edges = try gpa.alloc([]f32, ds.n_features),
            .levels = try gpa.alloc([][]u8, ds.n_features),
        };
        @memset(s.names, &.{});
        @memset(s.edges, &.{});
        @memset(s.levels, &.{});
        errdefer s.deinit();
        for (0..ds.n_features) |f| {
            s.names[f] = try gpa.dupe(u8, ds.names[f]);
            s.edges[f] = try gpa.dupe(f32, ds.edges[f]);
            if (ds.levels[f].len != 0) {
                const copy = try gpa.alloc([]u8, ds.levels[f].len);
                for (ds.levels[f], copy) |from, *to| to.* = try gpa.dupe(u8, from);
                s.levels[f] = copy;
            }
        }
        return s;
    }

    pub fn columnIndex(s: *const Schema, name: []const u8) ?usize {
        for (s.names, 0..) |n, i| if (std.mem.eql(u8, n, name)) return i;
        return null;
    }
};

const ApplyCtx = struct {
    src: *const Frame,
    schema: *const Schema,
    /// Column of `src` supplying each schema feature.
    src_col: []const usize,
    bins: []BinIdx,
    n_rows: usize,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *ApplyCtx = @ptrCast(@alignCast(ctx));
        var f = begin;
        while (f < end) : (f += 1) {
            const vals = self.src.values[self.src_col[f]];
            const out = self.bins[f * self.n_rows ..][0..self.n_rows];
            switch (self.schema.kinds[f]) {
                .numeric => {
                    const edges = self.schema.edges[f];
                    for (vals, out) |v, *b| {
                        b.* = if (std.math.isNan(v)) 0 else @intCast(lowerBound(edges, v) + 1);
                    }
                },
                .categorical => {
                    // The new frame built its own dictionary, so translate
                    // through the strings. A level the model never saw becomes
                    // the missing bin, which every split already handles.
                    const src_levels = self.src.levels[self.src_col[f]];
                    for (vals, out) |v, *b| {
                        if (std.math.isNan(v)) {
                            b.* = 0;
                            continue;
                        }
                        const id: usize = @intFromFloat(v);
                        b.* = 0;
                        if (id >= src_levels.len) continue;
                        for (self.schema.levels[f], 0..) |lvl, j| {
                            if (std.mem.eql(u8, lvl, src_levels[id])) {
                                b.* = @intCast(j + 1);
                                break;
                            }
                        }
                    }
                },
            }
        }
    }
};

/// Bin `src` under an existing `schema`, matching columns by name.
///
/// `label` names an optional target column to carry through, so a saved model
/// can be scored against a labelled holdout as well as used for prediction.
/// It must be encoded with the *model's* encoder: a holdout file orders its
/// own categorical dictionary by first appearance, so reading the raw ids
/// would invert the target whenever the two files disagree on which class
/// appears first.
pub fn applySchema(
    gpa: std.mem.Allocator,
    pool: *Pool,
    src: *const Frame,
    schema: *const Schema,
    label: ?LabelSpec,
) !Dataset {
    const n = schema.n_features;
    const src_col = try gpa.alloc(usize, n);
    defer gpa.free(src_col);
    for (0..n) |f| {
        src_col[f] = src.columnIndex(schema.names[f]) orelse return error.MissingFeatureColumn;
        // A column that was categorical in training but parses as numeric here
        // (or the reverse) would bin into a different space entirely.
        if (src.kinds[src_col[f]] != schema.kinds[f]) return error.FeatureKindMismatch;
    }

    var bins = try gpa.alloc(BinIdx, n * src.n_rows);
    errdefer if (bins.len != 0) gpa.free(bins);

    var ctx = ApplyCtx{
        .src = src,
        .schema = schema,
        .src_col = src_col,
        .bins = bins,
        .n_rows = src.n_rows,
    };
    pool.parallelFor(n, &ctx, ApplyCtx.run, 1);

    const bins_rm = try buildRowMajor(gpa, pool, bins, src.n_rows, n);
    errdefer gpa.free(bins_rm);

    const split = try splitByWidth(gpa, bins, schema.n_bins, src.n_rows, n);
    errdefer {
        gpa.free(split.narrow);
        for (split.wide) |c| if (c.len != 0) gpa.free(c);
        gpa.free(split.wide);
    }
    gpa.free(bins);
    bins = &.{};

    var out = Dataset{
        .gpa = gpa,
        .n_rows = src.n_rows,
        .n_features = n,
        .bins = split.narrow,
        .wide_cols = split.wide,
        .bins_rm = bins_rm,
        .n_bins = try gpa.dupe(u16, schema.n_bins),
        .edges = try gpa.alloc([]f32, n),
        // Prediction reads the bin-to-value mapping saved with the model, not
        // this one, and a table being scored has no training values to
        // average anyway.
        .means = try gpa.alloc([]f32, n),
        .kinds = try gpa.dupe(ColumnKind, schema.kinds),
        .names = try gpa.alloc([]u8, n),
        .levels = try gpa.alloc([][]u8, n),
        .labels = &.{},
    };
    @memset(out.edges, &.{});
    @memset(out.means, &.{});
    @memset(out.names, &.{});
    @memset(out.levels, &.{});
    errdefer out.deinit();
    for (0..n) |f| {
        out.edges[f] = try gpa.dupe(f32, schema.edges[f]);
        out.names[f] = try gpa.dupe(u8, schema.names[f]);
        if (schema.levels[f].len != 0) {
            const copy = try gpa.alloc([]u8, schema.levels[f].len);
            for (schema.levels[f], copy) |from, *to| to.* = try gpa.dupe(u8, from);
            out.levels[f] = copy;
        }
    }
    if (label) |ls| out.labels = try ls.enc.encode(gpa, src, ls.col);
    return out;
}

/// Name the columns that blew the bin cap, and how wide they are.
///
/// `error.CategoricalTooWide` on its own sends the reader back to count
/// distinct values by hand across every column in the file. The frame already
/// knows, so it may as well say.
///
/// It lives here, next to the `quantise` that raises the error, because it
/// used to live in `main.zig` where only the train path could reach it: `cv`
/// and `tune` printed a bare `error: CategoricalTooWide` and left the reader
/// with nothing. An explanation attached to one caller of a shared failure is
/// an explanation that mostly does not appear.
pub fn explainWidth(
    out: *std.Io.Writer,
    frame: *const Frame,
    max_bin: u16,
    dropped: []const []const u8,
) !void {
    try out.print(
        \\error: a categorical column has more levels than a bin can address
        \\       (limit {d}, set by --max_bin; a bin is stored as a u8)
        \\
    , .{max_bin});
    outer: for (frame.names, frame.kinds, frame.levels) |name, kind, levels| {
        if (kind != .categorical or levels.len < max_bin) continue;
        // A column the caller already dropped is not their problem.
        for (dropped) |d| if (std.mem.eql(u8, d, name)) continue :outer;
        try out.print("  {s: <28} {d} levels\n", .{ name, levels.len });
    }
    try out.writeAll(
        \\
        \\Either --drop the column, or replace it with something numeric --
        \\for a high-cardinality group, a statistic of that group (its mean
        \\target, size, or rank) is usually more useful than its identity.
        \\
    );
    try out.flush();
}
