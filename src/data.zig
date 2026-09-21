//! CSV ingest and quantisation.
//!
//! The booster never sees raw feature values. Every column is reduced to a
//! `u8` bin index up front, which is what lets the whole design stay in cache:
//! 668k rows x 13 features is 8.7 MB as bins, versus 35 MB as f32. Split
//! finding then works on bin indices alone and touches the original values
//! only to report thresholds.

const std = @import("std");
const Pool = @import("pool.zig").Pool;
const config = @import("config.zig");

pub const max_bins = 256;

pub const ColumnKind = enum { numeric, categorical };

/// A parsed but not yet quantised table. Column-major: a whole column is
/// contiguous, because every pass over the data is per-column.
pub const Frame = struct {
    gpa: std.mem.Allocator,
    n_rows: usize,
    names: [][]u8,
    kinds: []ColumnKind,
    /// For numeric columns the value; for categorical, the dictionary id as a
    /// float. NaN marks a missing entry.
    values: [][]f32,
    /// Dictionary for each categorical column, indexed by id. Empty otherwise.
    levels: [][][]u8,

    pub fn deinit(f: *Frame) void {
        const gpa = f.gpa;
        for (f.names) |n| gpa.free(n);
        gpa.free(f.names);
        for (f.values) |v| gpa.free(v);
        gpa.free(f.values);
        for (f.levels) |ls| {
            for (ls) |l| gpa.free(l);
            gpa.free(ls);
        }
        gpa.free(f.levels);
        gpa.free(f.kinds);
        f.* = undefined;
    }

    pub fn columnIndex(f: *const Frame, name: []const u8) ?usize {
        for (f.names, 0..) |n, i| if (std.mem.eql(u8, n, name)) return i;
        return null;
    }
};

// --------------------------------------------------------------- CSV parsing

const sniff_rows = 1000;

/// Offsets of each row's first byte, plus a terminating offset.
fn rowOffsets(gpa: std.mem.Allocator, text: []const u8) ![]usize {
    var offs: std.ArrayList(usize) = .empty;
    errdefer offs.deinit(gpa);
    try offs.append(gpa, 0);
    var i: usize = 0;
    while (std.mem.indexOfScalarPos(u8, text, i, '\n')) |nl| {
        i = nl + 1;
        if (i < text.len) try offs.append(gpa, i);
    }
    try offs.append(gpa, text.len);
    return offs.toOwnedSlice(gpa);
}

fn trimField(s: []const u8) []const u8 {
    return std.mem.trim(u8, s, " \t\r\n");
}

/// A field is numeric if it parses as a float, or is empty (missing).
fn looksNumeric(s: []const u8) bool {
    const t = trimField(s);
    if (t.len == 0) return true;
    _ = std.fmt.parseFloat(f32, t) catch return false;
    return true;
}

/// Split one CSV row into `out`, returning how many fields were written.
/// Quoted fields are supported to the extent Kaggle emits them: a leading
/// quote runs to the matching quote, with "" as an escaped quote.
fn splitRow(line: []const u8, out: [][]const u8) usize {
    var n: usize = 0;
    var i: usize = 0;
    while (n < out.len) {
        if (i < line.len and line[i] == '"') {
            const start = i + 1;
            var j = start;
            while (j < line.len) : (j += 1) {
                if (line[j] == '"') {
                    if (j + 1 < line.len and line[j + 1] == '"') {
                        j += 1;
                    } else break;
                }
            }
            out[n] = line[start..@min(j, line.len)];
            n += 1;
            i = j + 1;
            if (i < line.len and line[i] == ',') i += 1 else break;
        } else {
            const comma = std.mem.indexOfScalarPos(u8, line, i, ',');
            if (comma) |c| {
                out[n] = line[i..c];
                n += 1;
                i = c + 1;
            } else {
                out[n] = line[i..];
                n += 1;
                break;
            }
        }
    }
    return n;
}

const ParseCtx = struct {
    text: []const u8,
    offs: []const usize,
    kinds: []const ColumnKind,
    values: [][]f32,
    dicts: []const ?*const std.StringHashMapUnmanaged(u32),
    n_cols: usize,
    /// Set by any worker that meets a categorical level absent from the
    /// dictionary. Only ever written to `true`, so a plain store is enough.
    unseen: std.atomic.Value(bool),

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *ParseCtx = @ptrCast(@alignCast(ctx));
        var fields_buf: [512][]const u8 = undefined;
        const fields = fields_buf[0..@min(self.n_cols, fields_buf.len)];

        var r = begin;
        while (r < end) : (r += 1) {
            const line = trimField(self.text[self.offs[r]..self.offs[r + 1]]);
            const got = splitRow(line, fields);
            for (0..self.n_cols) |c| {
                var v: f32 = std.math.nan(f32);
                if (c < got) {
                    const t = trimField(fields[c]);
                    if (t.len != 0) {
                        switch (self.kinds[c]) {
                            .numeric => v = std.fmt.parseFloat(f32, t) catch std.math.nan(f32),
                            .categorical => {
                                if (self.dicts[c].?.get(t)) |id| {
                                    v = @floatFromInt(id);
                                } else {
                                    self.unseen.store(true, .monotonic);
                                }
                            },
                        }
                    }
                }
                self.values[c][r] = v;
            }
        }
    }
};

/// Read and parse a CSV. `pool` is used for the row-parsing pass, which is
/// where essentially all the time goes.
pub fn readCsv(
    gpa: std.mem.Allocator,
    io: std.Io,
    pool: *Pool,
    path: []const u8,
    max_bytes: usize,
) !Frame {
    const text = try std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        gpa,
        std.Io.Limit.limited(max_bytes),
    );
    defer gpa.free(text);

    const offs = try rowOffsets(gpa, text);
    defer gpa.free(offs);
    if (offs.len < 3) return error.EmptyCsv;

    // ---- header ----
    const header_line = trimField(text[offs[0]..offs[1]]);
    var hdr_buf: [512][]const u8 = undefined;
    const n_cols = splitRow(header_line, &hdr_buf);
    if (n_cols == 0) return error.NoColumns;

    const names = try gpa.alloc([]u8, n_cols);
    errdefer gpa.free(names);
    var named: usize = 0;
    errdefer for (names[0..named]) |n| gpa.free(n);
    while (named < n_cols) : (named += 1) {
        names[named] = try gpa.dupe(u8, trimField(hdr_buf[named]));
    }

    const n_rows = offs.len - 2; // minus header, minus terminator

    // ---- sniff column kinds from a prefix ----
    const kinds = try gpa.alloc(ColumnKind, n_cols);
    errdefer gpa.free(kinds);
    @memset(kinds, .numeric);
    {
        var fields_buf: [512][]const u8 = undefined;
        const fields = fields_buf[0..n_cols];
        const limit = @min(n_rows, sniff_rows);
        for (0..limit) |i| {
            const line = trimField(text[offs[i + 1]..offs[i + 2]]);
            const got = splitRow(line, fields);
            for (0..@min(got, n_cols)) |c| {
                if (kinds[c] == .numeric and !looksNumeric(fields[c])) kinds[c] = .categorical;
            }
        }
    }

    // ---- collect categorical levels ----
    // Cardinalities here are small, so a single scan building one dictionary
    // costs less than the cross-thread merge a parallel scan would need.
    var dict_storage = try gpa.alloc(?std.StringHashMapUnmanaged(u32), n_cols);
    defer gpa.free(dict_storage);
    @memset(dict_storage, null);
    defer for (dict_storage) |*d| if (d.*) |*m| m.deinit(gpa);

    const levels = try gpa.alloc([][]u8, n_cols);
    errdefer gpa.free(levels);
    var levelled: usize = 0;
    errdefer for (levels[0..levelled]) |ls| {
        for (ls) |l| gpa.free(l);
        gpa.free(ls);
    };

    {
        var lists = try gpa.alloc(std.ArrayList([]u8), n_cols);
        defer gpa.free(lists);
        for (lists) |*l| l.* = .empty;
        defer for (lists) |*l| l.deinit(gpa);

        var fields_buf: [512][]const u8 = undefined;
        const fields = fields_buf[0..n_cols];
        for (0..n_rows) |i| {
            const line = trimField(text[offs[i + 1]..offs[i + 2]]);
            const got = splitRow(line, fields);
            for (0..@min(got, n_cols)) |c| {
                if (kinds[c] != .categorical) continue;
                const t = trimField(fields[c]);
                if (t.len == 0) continue;
                if (dict_storage[c] == null) dict_storage[c] = .empty;
                const m = &dict_storage[c].?;
                if (m.contains(t)) continue;
                if (lists[c].items.len >= max_bins) return error.CategoricalTooWide;
                const owned = try gpa.dupe(u8, t);
                errdefer gpa.free(owned);
                try m.put(gpa, owned, @intCast(lists[c].items.len));
                try lists[c].append(gpa, owned);
            }
        }
        while (levelled < n_cols) : (levelled += 1) {
            levels[levelled] = try lists[levelled].toOwnedSlice(gpa);
        }
    }

    // ---- parse values in parallel ----
    const values = try gpa.alloc([]f32, n_cols);
    errdefer gpa.free(values);
    var allocated: usize = 0;
    errdefer for (values[0..allocated]) |v| gpa.free(v);
    while (allocated < n_cols) : (allocated += 1) {
        values[allocated] = try gpa.alloc(f32, n_rows);
    }

    const dict_ptrs = try gpa.alloc(?*const std.StringHashMapUnmanaged(u32), n_cols);
    defer gpa.free(dict_ptrs);
    for (0..n_cols) |c| {
        dict_ptrs[c] = if (dict_storage[c]) |*m| m else null;
    }

    var ctx = ParseCtx{
        .text = text,
        .offs = offs[1..],
        .kinds = kinds,
        .values = values,
        .dicts = dict_ptrs,
        .n_cols = n_cols,
        .unseen = .init(false),
    };
    pool.parallelFor(n_rows, &ctx, ParseCtx.run, 4096);

    return .{
        .gpa = gpa,
        .n_rows = n_rows,
        .names = names,
        .kinds = kinds,
        .values = values,
        .levels = levels,
    };
}

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
    /// Column-major: `bins[f * n_rows + r]`. One feature's rows are contiguous,
    /// so histogram building walks memory linearly and the prefetcher wins.
    bins: []u8,
    /// Bins actually in use per feature, including the missing bin.
    n_bins: []u16,
    /// Cut points per feature; `edges[f][i]` is the inclusive upper bound of
    /// real-value bin `i`. Length is `n_bins[f] - 2`.
    edges: [][]f32,
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
        gpa.free(d.n_bins);
        for (d.edges) |e| gpa.free(e);
        gpa.free(d.edges);
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

    pub inline fn column(d: *const Dataset, f: usize) []const u8 {
        return d.bins[f * d.n_rows ..][0..d.n_rows];
    }

    /// Largest bin count across features; sizes the histogram allocation.
    pub fn maxBins(d: *const Dataset) u16 {
        var m: u16 = 0;
        for (d.n_bins) |b| m = @max(m, b);
        return m;
    }
};

fn lowerBound(edges: []const f32, v: f32) usize {
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

const BinCtx = struct {
    gpa: std.mem.Allocator,
    src: *const Frame,
    feature_cols: []const usize,
    bins: []u8,
    n_bins: []u16,
    edges: [][]f32,
    n_rows: usize,
    max_bin: u16,
    policy: config.BinPolicy,
    failed: std.atomic.Value(bool),

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
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
    }
};

/// Quantise `src` into a `Dataset`. `label_col`, when given, is excluded from
/// the feature set and copied out as the target. `skip` names columns to drop
/// (an id column, typically).
pub fn quantise(
    gpa: std.mem.Allocator,
    pool: *Pool,
    src: *const Frame,
    cfg: config.Config,
    label_col: ?usize,
    skip: []const []const u8,
) !Dataset {
    var feats: std.ArrayList(usize) = .empty;
    defer feats.deinit(gpa);
    outer: for (0..src.names.len) |c| {
        if (label_col) |lc| if (c == lc) continue;
        for (skip) |s| if (std.mem.eql(u8, src.names[c], s)) continue :outer;
        try feats.append(gpa, c);
    }
    const n_features = feats.items.len;
    if (n_features == 0) return error.NoFeatures;

    const bins = try gpa.alloc(u8, n_features * src.n_rows);
    errdefer gpa.free(bins);
    const n_bins = try gpa.alloc(u16, n_features);
    errdefer gpa.free(n_bins);
    const edges = try gpa.alloc([]f32, n_features);
    errdefer gpa.free(edges);
    @memset(edges, &.{});
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
        .n_rows = src.n_rows,
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
    for (0..n_features) |f| {
        const col = feats.items[f];
        if (kinds[f] != .categorical) continue;
        const src_levels = src.levels[col];
        const copy = try gpa.alloc([]u8, src_levels.len);
        for (src_levels, copy) |from, *to| to.* = try gpa.dupe(u8, from);
        levels[f] = copy;
    }

    var labels: []f32 = &.{};
    if (label_col) |lc| {
        labels = try gpa.dupe(f32, src.values[lc]);
    }

    return .{
        .gpa = gpa,
        .n_rows = src.n_rows,
        .n_features = n_features,
        .bins = bins,
        .n_bins = n_bins,
        .edges = edges,
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
    for (0..ds.n_features) |f| {
        const src = ds.column(f);
        const dst = bins[f * n ..][0..n];
        for (rows, dst) |r, *d| d.* = src[r];
    }

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
        .n_bins = n_bins,
        .edges = edges,
        .kinds = kinds,
        .names = names,
        .levels = levels,
        .labels = labels,
    };
}

// ------------------------------------------------------------------ schema

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
    bins: []u8,
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
pub fn applySchema(
    gpa: std.mem.Allocator,
    pool: *Pool,
    src: *const Frame,
    schema: *const Schema,
    label: ?[]const u8,
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

    const bins = try gpa.alloc(u8, n * src.n_rows);
    errdefer gpa.free(bins);

    var ctx = ApplyCtx{
        .src = src,
        .schema = schema,
        .src_col = src_col,
        .bins = bins,
        .n_rows = src.n_rows,
    };
    pool.parallelFor(n, &ctx, ApplyCtx.run, 1);

    var out = Dataset{
        .gpa = gpa,
        .n_rows = src.n_rows,
        .n_features = n,
        .bins = bins,
        .n_bins = try gpa.dupe(u16, schema.n_bins),
        .edges = try gpa.alloc([]f32, n),
        .kinds = try gpa.dupe(ColumnKind, schema.kinds),
        .names = try gpa.alloc([]u8, n),
        .levels = try gpa.alloc([][]u8, n),
        .labels = &.{},
    };
    @memset(out.edges, &.{});
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
    if (label) |name| {
        if (src.columnIndex(name)) |lc| out.labels = try gpa.dupe(f32, src.values[lc]);
    }
    return out;
}
