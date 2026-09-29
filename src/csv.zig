// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! CSV parsing: bytes on disk to a column-major `Frame`.
//!
//! Split out of `data.zig` because parsing is not a model and not a
//! quantisation concern -- every model in this library pays this cost before
//! it sees a single bin, so it deserves its own module, its own tests and its
//! own timings. `data.zig` re-exports `Frame`, `ColumnKind` and `readCsv` so
//! existing call sites keep working.
//!
//! Measured against `pandas.read_csv` on the same 32 MB / 534,932-row file:
//! 117 ms here against 179 ms there. See docs/arena.md.

const std = @import("std");
const Pool = @import("pool.zig").Pool;
const zsift = @import("vendor/zsift/csv.zig");

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
    /// Per column, how many non-empty fields failed to parse as a number in a
    /// column the sniff had classified numeric -- and were not one of the
    /// recognised missing markers either. A nonzero entry means the file
    /// disagrees with itself: either the sniff prefix was unrepresentative or
    /// the column holds junk. Both are worth saying out loud rather than
    /// silently storing NaN, which is what used to happen.
    unparsed: []u32 = &.{},

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
        gpa.free(f.unparsed);
        f.* = undefined;
    }

    pub fn columnIndex(f: *const Frame, name: []const u8) ?usize {
        for (f.names, 0..) |n, i| if (std.mem.eql(u8, n, name)) return i;
        return null;
    }
};

// --------------------------------------------------------------- CSV parsing

const sniff_rows = 1000;

fn trimField(s: []const u8) []const u8 {
    return std.mem.trim(u8, s, " \t\r\n");
}

/// Tokens that mean "no value recorded here".
///
/// These are the markers R, pandas, Excel and SQL exports actually emit. The
/// set is deliberately short: every entry is a word no one uses as data.
const missing_tokens = [_][]const u8{
    "na", "n/a", "#n/a", "nan", "null", "none", "nil", "?",
};

/// Is this field one of the recognised missing markers (or empty)?
///
/// Read this together with `looksNumeric`, because the pair encodes the one
/// design decision that matters here: **a marker only means "missing" in a
/// column whose other values are numbers.**
///
/// The Ames housing data is the clean example of why. `LotFrontage` holds
/// numbers and `NA`; the `NA` is a frontage nobody recorded. `PoolQC` holds
/// `Ex`/`Gd`/`TA` and `NA`; there the data description defines `NA` as the
/// level "No Pool" -- real information about the house. Treating every `NA`
/// as missing, which is what `pandas.read_csv` does by default, silently
/// converts 14 of that dataset's columns from "documented category" to
/// "unknown". Treating none of them as missing -- what this parser did before
/// -- is worse the other way: `MasVnrArea` becomes a 328-level categorical
/// and the file will not load at all.
///
/// Deciding per column costs one extra comparison in the sniff pass and gets
/// both right with no configuration.
pub fn isMissingToken(s: []const u8) bool {
    const t = trimField(s);
    if (t.len == 0) return true;
    if (t.len > 4) return false;
    for (missing_tokens) |m| {
        if (std.ascii.eqlIgnoreCase(t, m)) return true;
    }
    return false;
}

/// A field keeps a column numeric if it parses as a float, or is missing.
fn looksNumeric(s: []const u8) bool {
    const t = trimField(s);
    if (isMissingToken(t)) return true;
    _ = std.fmt.parseFloat(f32, t) catch return false;
    return true;
}

/// Ceiling on distinct values in one column while reading. Far above any
/// usable categorical -- it exists so a free-text column cannot allocate a
/// dictionary the size of the file. The bin-width limit that actually matters
/// is `max_bins`, checked in `binOne` after drops are applied.
const max_levels: usize = 1 << 20;

/// Column kinds that are already known, keyed by name.
///
/// The parser decides a column's kind by looking at the file, which is all it
/// can do when nobody knows better. At prediction time somebody does: the
/// model was trained on a schema and stored it. Letting the parser re-decide
/// lets a file disagree with the model for reasons that have nothing to do
/// with the column.
///
/// The case that found this: a 298-row holdout slice of House Prices in which
/// every `PoolQC` happened to be `NA`. A column of nothing but missing markers
/// is numeric-compatible, so it sniffed numeric where training had it
/// categorical, and the prediction died with `FeatureKindMismatch` on data
/// that was perfectly well formed. The column had not changed; the evidence
/// available about it had.
pub const KindHint = struct {
    names: []const []const u8,
    kinds: []const ColumnKind,

    fn get(h: KindHint, name: []const u8) ?ColumnKind {
        for (h.names, h.kinds) |n, k| {
            if (std.mem.eql(u8, n, name)) return k;
        }
        return null;
    }
};

/// Read and parse a CSV with the vendored zsift parser (`vendor/zsift`): quoted fields
/// may hold delimiters, newlines and escaped `""`, and there is no column limit. The
/// data rows are parsed on up to `pool.workerCount()` workers through `io` (zsift picks
/// fewer, or one, for small files); each worker keeps its own columns and category
/// dictionaries, merged in file order so level ids are first-appearance order exactly
/// as a serial read gives. A file with a stray quote inside an unquoted field (not
/// RFC 4180, which zsift's strict path rejects) is re-read serially by zsift's lenient
/// parser, which treats that quote as data.
pub fn readCsv(
    gpa: std.mem.Allocator,
    io: std.Io,
    pool: *Pool,
    path: []const u8,
    max_bytes: usize,
) !Frame {
    return readCsvHinted(gpa, io, pool, path, max_bytes, null);
}

/// `hint` pins the kind of any column it names; the rest are sniffed as usual.
/// A column the hint does not mention is not an error -- the new file may
/// carry extras the model never saw, and dropping them is the caller's job.
pub fn readCsvHinted(
    gpa: std.mem.Allocator,
    io: std.Io,
    pool: *Pool,
    path: []const u8,
    max_bytes: usize,
    hint: ?KindHint,
) !Frame {
    const text = try std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        gpa,
        std.Io.Limit.limited(max_bytes),
    );
    defer gpa.free(text);
    const workers = @min(pool.workerCount(), zsift.parallel.max_workers);
    return parseText(gpa, io, text, workers, hint, .strict) catch |e| switch (e) {
        // Not RFC 4180 (a quote inside an unquoted field): read it leniently instead.
        error.InvalidQuote => parseText(gpa, io, text, 1, hint, .lenient),
        else => e,
    };
}

const Mode = enum { strict, lenient };

/// Bytes of CSV per worker. zsift's own size rule (`parallel.forEachField`, serial
/// below 2 MiB) is tuned for a sink that does almost nothing per field; this one
/// converts every field, so splitting pays far sooner. Swept 32–256 KiB against the
/// previous all-core loader on 7 real files (0.7–45 MB): 64 KiB was the only size
/// faster on every file in every round (medians 1.34–1.73×, worst round 1.08×).
const bytes_per_worker: usize = 64 << 10;

/// Scratch per worker for unescaping one quoted field with `""` in it; a longer
/// escaped field is a `ScratchTooSmall` error, not a truncation.
const field_scratch_max: usize = 16 << 20;

fn parseText(gpa: std.mem.Allocator, io: std.Io, text: []const u8, workers: usize, hint: ?KindHint, comptime mode: Mode) !Frame {
    const P = if (mode == .strict) zsift.SimdParser else zsift.Parser;
    const scratch_len = @min(text.len + 64, field_scratch_max);

    // ---- header, then the column kinds from the first `sniff_rows` data rows ----
    var names: std.ArrayList([]u8) = .empty;
    errdefer {
        for (names.items) |n| gpa.free(n);
        names.deinit(gpa);
    }
    var kinds: std.ArrayList(ColumnKind) = .empty;
    errdefer kinds.deinit(gpa);
    {
        const scratch = try gpa.alloc(u8, scratch_len);
        defer gpa.free(scratch);
        var it = try P.init(text, scratch, .{});
        while (try it.next()) |f| {
            try names.append(gpa, try gpa.dupe(u8, trimField(f.bytes)));
            it.resetScratch();
            if (f.last_in_record) break;
        }
        if (names.items.len == 0) return error.NoColumns;
        try kinds.appendNTimes(gpa, .numeric, names.items.len);
        // A pinned column is not sniffed at all: the hint is evidence from the whole
        // training set, and this file's prefix cannot outvote it.
        const pinned = try gpa.alloc(bool, names.items.len);
        defer gpa.free(pinned);
        @memset(pinned, false);
        if (hint) |hh| for (names.items, kinds.items, pinned) |n, *k, *pin| {
            if (hh.get(n)) |hk| {
                k.* = hk;
                pin.* = true;
            }
        };
        var rows: usize = 0;
        var c: usize = 0;
        while (rows < sniff_rows) {
            const f = (try it.next()) orelse break;
            if (c < names.items.len and !pinned[c] and kinds.items[c] == .numeric and !looksNumeric(f.bytes))
                kinds.items[c] = .categorical;
            it.resetScratch();
            c += 1;
            if (f.last_in_record) {
                c = 0;
                rows += 1;
            }
        }
    }

    // ---- every row, on up to `workers` ranges; range 0 skips the header ----
    const k: usize = switch (mode) {
        .strict => @max(1, @min(workers, text.len / bytes_per_worker)),
        .lenient => 1,
    };
    const sinks = try gpa.alloc(RangeSink, k);
    defer gpa.free(sinks);
    for (sinks, 0..) |*s, i| s.* = .{ .gpa = gpa, .kinds = kinds.items, .skip_header = i == 0 };
    defer for (sinks) |*s| s.deinit();
    for (sinks) |*s| try s.alloc();
    const ptrs = try gpa.alloc(*RangeSink, k);
    defer gpa.free(ptrs);
    const scratches = try gpa.alloc([]u8, k);
    defer gpa.free(scratches);
    var n_scratch: usize = 0;
    defer for (scratches[0..n_scratch]) |sc| gpa.free(sc);
    for (ptrs, scratches, sinks) |*ptr, *sc, *s| {
        ptr.* = s;
        sc.* = try gpa.alloc(u8, scratch_len);
        n_scratch += 1;
    }
    switch (mode) {
        .strict => try zsift.parallel.forEachFieldExact(io, text, .{}, scratches, ptrs, RangeSink.on),
        .lenient => {
            var it = try zsift.Parser.init(text, scratches[0], .{});
            while (try it.next()) |f| {
                sinks[0].on(f.bytes, f.last_in_record);
                it.resetScratch();
            }
        },
    }
    for (sinks) |*s| if (s.err) |e| return e;

    // ---- merge the ranges in file order ----
    var n_rows: usize = 0;
    for (sinks) |*s| n_rows += s.rows;
    if (n_rows == 0) return error.EmptyCsv;
    const n_cols = names.items.len;

    const values = try gpa.alloc([]f32, n_cols);
    errdefer gpa.free(values);
    var filled: usize = 0;
    errdefer for (values[0..filled]) |v| gpa.free(v);
    const levels = try gpa.alloc([][]u8, n_cols);
    errdefer gpa.free(levels);
    var levelled: usize = 0;
    errdefer for (levels[0..levelled]) |ls| {
        for (ls) |l| gpa.free(l);
        gpa.free(ls);
    };
    const unparsed = try gpa.alloc(u32, n_cols);
    errdefer gpa.free(unparsed);

    for (0..n_cols) |c| {
        values[c] = try gpa.alloc(f32, n_rows);
        filled += 1;
        unparsed[c] = 0;
        for (sinks) |*s| unparsed[c] += s.bad[c];
        levels[c] = try mergeColumn(gpa, sinks, c, values[c]);
        levelled += 1;
    }

    return .{
        .gpa = gpa,
        .n_rows = n_rows,
        .names = try names.toOwnedSlice(gpa),
        .kinds = try kinds.toOwnedSlice(gpa),
        .values = values,
        .levels = levels,
        .unparsed = unparsed,
    };
}

/// Concatenate column `c` of every range into `out`. For a categorical column the
/// ranges' local level ids are renumbered into one dictionary, walked in range order,
/// so each level's id is its first appearance in the file. Returns the levels, owned.
fn mergeColumn(gpa: std.mem.Allocator, sinks: []RangeSink, c: usize, out: []f32) ![][]u8 {
    var at: usize = 0;
    if (sinks[0].kinds[c] == .numeric) {
        for (sinks) |*s| {
            @memcpy(out[at..][0..s.cols[c].items.len], s.cols[c].items);
            at += s.cols[c].items.len;
        }
        return &.{};
    }
    var global: std.StringHashMapUnmanaged(u32) = .empty;
    defer global.deinit(gpa);
    var list: std.ArrayList([]u8) = .empty;
    errdefer {
        for (list.items) |l| gpa.free(l);
        list.deinit(gpa);
    }
    var remap: std.ArrayList(u32) = .empty;
    defer remap.deinit(gpa);
    for (sinks) |*s| {
        remap.clearRetainingCapacity();
        for (s.levels[c].items) |*lvl| {
            const gop = try global.getOrPut(gpa, lvl.*);
            if (!gop.found_existing) {
                if (list.items.len >= max_levels) return error.TooManyLevels;
                gop.value_ptr.* = @intCast(list.items.len);
                try list.append(gpa, lvl.*);
                lvl.* = &.{}; // ownership moved to `list`
            }
            try remap.append(gpa, gop.value_ptr.*);
        }
        for (s.cols[c].items) |v| {
            out[at] = if (std.math.isNan(v)) v else @floatFromInt(remap.items[@intFromFloat(v)]);
            at += 1;
        }
    }
    return list.toOwnedSlice(gpa);
}

/// One range's rows, column-major, with the range's own category dictionaries (local
/// ids in first-appearance order within the range). Written by one worker only.
const RangeSink = struct {
    gpa: std.mem.Allocator,
    kinds: []const ColumnKind,
    /// Range 0 begins with the header record, which is not a data row.
    skip_header: bool,
    col: usize = 0,
    rows: usize = 0,
    cols: []std.ArrayList(f32) = &.{},
    dicts: []std.StringHashMapUnmanaged(u32) = &.{},
    levels: []std.ArrayList([]u8) = &.{},
    bad: []u32 = &.{},
    err: ?anyerror = null,

    fn alloc(s: *RangeSink) !void {
        const n = s.kinds.len;
        s.cols = try s.gpa.alloc(std.ArrayList(f32), n);
        for (s.cols) |*x| x.* = .empty;
        s.dicts = try s.gpa.alloc(std.StringHashMapUnmanaged(u32), n);
        for (s.dicts) |*x| x.* = .empty;
        s.levels = try s.gpa.alloc(std.ArrayList([]u8), n);
        for (s.levels) |*x| x.* = .empty;
        s.bad = try s.gpa.alloc(u32, n);
        @memset(s.bad, 0);
    }

    fn deinit(s: *RangeSink) void {
        for (s.cols) |*x| x.deinit(s.gpa);
        s.gpa.free(s.cols);
        for (s.dicts) |*x| x.deinit(s.gpa);
        s.gpa.free(s.dicts);
        for (s.levels) |*x| {
            for (x.items) |l| s.gpa.free(l); // levels merged away were emptied
            x.deinit(s.gpa);
        }
        s.gpa.free(s.levels);
        s.gpa.free(s.bad);
    }

    fn on(s: *RangeSink, bytes: []const u8, last: bool) void {
        if (s.err != null) return;
        if (s.skip_header) {
            if (last) s.skip_header = false;
            return;
        }
        if (s.col < s.kinds.len) s.put(s.col, bytes) catch |e| {
            s.err = e;
        };
        s.col += 1;
        if (last) {
            // A short record: its missing columns are NaN. Extra fields were ignored.
            while (s.col < s.kinds.len) : (s.col += 1) s.cols[s.col].append(s.gpa, std.math.nan(f32)) catch |e| {
                s.err = e;
            };
            s.col = 0;
            s.rows += 1;
        }
    }

    fn put(s: *RangeSink, c: usize, bytes: []const u8) !void {
        const t = trimField(bytes);
        var v: f32 = std.math.nan(f32);
        // Only a genuinely empty field is missing in a categorical column: its `NA`
        // is a level (see `isMissingToken`).
        if (t.len != 0) switch (s.kinds[c]) {
            .numeric => v = std.fmt.parseFloat(f32, t) catch blk: {
                if (!isMissingToken(t)) s.bad[c] += 1;
                break :blk std.math.nan(f32);
            },
            .categorical => {
                const gop = try s.dicts[c].getOrPut(s.gpa, t);
                if (!gop.found_existing) {
                    // A memory guard, not the bin cap (that is `binOne`'s): a free-text
                    // column must not build a dictionary the size of the file.
                    if (s.levels[c].items.len >= max_levels) return error.TooManyLevels;
                    const owned = try s.gpa.dupe(u8, t);
                    gop.key_ptr.* = owned;
                    gop.value_ptr.* = @intCast(s.levels[c].items.len);
                    try s.levels[c].append(s.gpa, owned);
                }
                v = @floatFromInt(gop.value_ptr.*);
            },
        };
        try s.cols[c].append(s.gpa, v);
    }
};

// ------------------------------------------------------------- profiling

// Everything below answers "what is actually in this file" before a model
// sees it. It lives here rather than in a per-dataset script because every
// dataset needs it and the parser has already touched every byte: the counts
// are a second pass over memory that is still warm, not a second read.
//
// What it deliberately does not do is judge. It reports a count of values
// outside the Tukey fence; it does not call them errors, because on house
// prices or incomes the tail is the data. The reader decides.

/// One column's shape. Numeric fields are meaningless for a categorical
/// column and are left at zero.
pub const ColumnStat = struct {
    name: []const u8,
    kind: ColumnKind,
    /// NaN entries: empty fields, recognised missing markers, and anything
    /// that failed to parse.
    missing: usize,
    /// Of those, the ones that were neither empty nor a known marker.
    unparsed: u32,
    /// Level count for a categorical column; distinct finite values for a
    /// numeric one.
    distinct: usize,
    min: f64 = 0,
    q1: f64 = 0,
    median: f64 = 0,
    q3: f64 = 0,
    max: f64 = 0,
    mean: f64 = 0,
    /// Values past the Tukey *outer* fence, q1 - 3*IQR and q3 + 3*IQR. The
    /// outer fence rather than the usual 1.5*IQR because a skewed column --
    /// sale price, lot area, income -- puts a tenth of its rows outside the
    /// inner one legitimately, and a flag that fires on a tenth of the data
    /// is not a flag.
    far_low: usize = 0,
    far_high: usize = 0,

    pub fn allMissing(s: ColumnStat) bool {
        return s.distinct == 0;
    }

    pub fn constant(s: ColumnStat) bool {
        return s.distinct == 1;
    }
};

fn quantile(sorted: []const f64, q: f64) f64 {
    if (sorted.len == 0) return std.math.nan(f64);
    if (sorted.len == 1) return sorted[0];
    const pos = q * @as(f64, @floatFromInt(sorted.len - 1));
    const lo: usize = @intFromFloat(@floor(pos));
    const hi = @min(lo + 1, sorted.len - 1);
    const frac = pos - @floor(pos);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

/// Measure every column. Caller frees the returned slice.
pub fn profile(gpa: std.mem.Allocator, f: *const Frame) ![]ColumnStat {
    const stats = try gpa.alloc(ColumnStat, f.names.len);
    errdefer gpa.free(stats);

    const buf = try gpa.alloc(f64, f.n_rows);
    defer gpa.free(buf);

    for (f.names, f.kinds, f.values, 0..) |name, kind, vals, c| {
        var n: usize = 0;
        var sum: f64 = 0;
        for (vals) |v| {
            if (std.math.isNan(v)) continue;
            buf[n] = v;
            n += 1;
            sum += v;
        }
        stats[c] = .{
            .name = name,
            .kind = kind,
            .missing = f.n_rows - n,
            .unparsed = if (c < f.unparsed.len) f.unparsed[c] else 0,
            .distinct = 0,
        };
        if (kind == .categorical) {
            stats[c].distinct = f.levels[c].len;
            continue;
        }
        if (n == 0) continue;

        const v = buf[0..n];
        std.mem.sort(f64, v, {}, std.sort.asc(f64));
        var distinct: usize = 1;
        for (1..n) |i| {
            if (v[i] != v[i - 1]) distinct += 1;
        }
        const q1 = quantile(v, 0.25);
        const q3 = quantile(v, 0.75);
        const fence = 3.0 * (q3 - q1);
        var lo_n: usize = 0;
        var hi_n: usize = 0;
        if (fence > 0) {
            for (v) |x| {
                if (x < q1 - fence) lo_n += 1;
                if (x > q3 + fence) hi_n += 1;
            }
        }
        stats[c].distinct = distinct;
        stats[c].min = v[0];
        stats[c].q1 = q1;
        stats[c].median = quantile(v, 0.5);
        stats[c].q3 = q3;
        stats[c].max = v[n - 1];
        stats[c].mean = sum / @as(f64, @floatFromInt(n));
        stats[c].far_low = lo_n;
        stats[c].far_high = hi_n;
    }
    return stats;
}

/// The one line every train/cv run prints: enough to notice a file that was
/// read wrong, short enough that nobody scrolls past it.
pub fn writeSummary(out: *std.Io.Writer, f: *const Frame, stats: []const ColumnStat) !void {
    var num: usize = 0;
    var missing: usize = 0;
    var unparsed: u64 = 0;
    var empty_cols: usize = 0;
    var const_cols: usize = 0;
    for (stats) |s| {
        if (s.kind == .numeric) num += 1;
        missing += s.missing;
        unparsed += s.unparsed;
        if (s.allMissing()) empty_cols += 1 else if (s.constant()) const_cols += 1;
    }
    const cells = f.n_rows * stats.len;
    const pct = if (cells == 0) 0.0 else 100.0 * @as(f64, @floatFromInt(missing)) /
        @as(f64, @floatFromInt(cells));
    try out.print("parse   {d} cols ({d} numeric, {d} categorical)  {d} missing ({d:.2}%)\n", .{
        stats.len, num, stats.len - num, missing, pct,
    });
    if (unparsed != 0 or empty_cols != 0 or const_cols != 0) {
        try out.writeAll("        ");
        if (unparsed != 0) try out.print("{d} unparsed  ", .{unparsed});
        if (empty_cols != 0) try out.print("{d} all-missing column(s)  ", .{empty_cols});
        if (const_cols != 0) try out.print("{d} constant column(s)", .{const_cols});
        try out.writeAll("\n");
    }
}

/// The full table, for `zgbdt profile`.
pub fn writeProfile(out: *std.Io.Writer, f: *const Frame, stats: []const ColumnStat) !void {
    try out.print("rows    {d}\ncols    {d}\n\n", .{ f.n_rows, stats.len });
    try out.print("{s: <24}{s: <6}{s: >9}{s: >7}{s: >9}{s: >13}{s: >13}{s: >13}{s: >9}\n", .{
        "column", "kind", "missing", "miss%", "distinct", "min", "median", "max", "far out",
    });

    for (stats) |s| {
        const pct = if (f.n_rows == 0) 0.0 else 100.0 * @as(f64, @floatFromInt(s.missing)) /
            @as(f64, @floatFromInt(f.n_rows));
        try out.print("{s: <24}{s: <6}{d: >9}{d: >6.1}%{d: >9}", .{
            s.name,
            if (s.kind == .numeric) "num" else "cat",
            s.missing,
            pct,
            s.distinct,
        });
        if (s.kind == .numeric and !s.allMissing()) {
            try out.print("{d: >13.4}{d: >13.4}{d: >13.4}{d: >9}\n", .{
                s.min, s.median, s.max, s.far_low + s.far_high,
            });
        } else {
            try out.print("{s: >13}{s: >13}{s: >13}{s: >9}\n", .{ "-", "-", "-", "-" });
        }
    }

    // The footer is the part worth acting on, so it says what to do.
    var said = false;
    for (stats) |s| {
        if (s.unparsed == 0) continue;
        if (!said) {
            try out.writeAll(
                \\
                \\Values that were neither a number nor a recognised missing marker, in
                \\a column read as numeric. The kind is sniffed from the first 1000 rows,
                \\so this usually means the column changes character further down:
                \\
            );
            said = true;
        }
        try out.print("  {s: <24}{d} unparsed\n", .{ s.name, s.unparsed });
    }

    said = false;
    for (stats) |s| {
        if (!s.allMissing() and !s.constant()) continue;
        if (!said) {
            try out.writeAll("\nColumns that cannot inform a split:\n");
            said = true;
        }
        try out.print("  {s: <24}{s}\n", .{
            s.name,
            if (s.allMissing()) "every value missing" else "one distinct value",
        });
    }

    var wide = false;
    for (stats) |s| {
        if (s.kind != .categorical or s.distinct < 256) continue;
        if (!wide) {
            try out.writeAll(
                \\
                \\Categorical columns too wide for a u8 bin. These must be dropped or
                \\replaced with a numeric summary before training:
                \\
            );
            wide = true;
        }
        try out.print("  {s: <24}{d} levels\n", .{ s.name, s.distinct });
    }
    try out.flush();
}

// ----------------------------------------------------------------- tests

const testing = std.testing;

test "missing markers are recognised, and real values are not" {
    for ([_][]const u8{ "", "  ", "NA", "na", "n/a", "N/A", "#N/A", "NaN", "null", "None", "nil", "?" }) |t| {
        try testing.expect(isMissingToken(t));
    }
    // The near misses matter more than the hits. `NAmes` is a real
    // Neighborhood level in the Ames data and `None` is a real MasVnrType,
    // so a prefix match or a case-folded contains() would corrupt both.
    for ([_][]const u8{ "NAmes", "Names", "NAN1", "nullable", "N", "0", "-1" }) |t| {
        try testing.expect(!isMissingToken(t));
    }
}

// The behaviour the whole design rests on, end to end through `readCsv`:
// the same token, `NA`, is a hole in one column and a level in the next.
test "NA is missing in a numeric column and a level in a categorical one" {
    const gpa = testing.allocator;
    const io = testing.io;

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    // frontage: numbers + NA -> numeric, NA is a hole
    // poolqc:   Ex/Gd + NA   -> categorical, NA is the level "No Pool"
    try tmp.dir.writeFile(io, .{ .sub_path = "d.csv", .data =
        \\frontage,poolqc
        \\65,Ex
        \\NA,NA
        \\80,Gd
        \\NA,NA
        \\
    });

    var path_buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, ".zig-cache/tmp/{s}/d.csv", .{tmp.sub_path});

    var pool = try Pool.init(gpa, 1);
    defer pool.deinit();

    var frame = try readCsv(gpa, io, pool, path, 1 << 20);
    defer frame.deinit();

    try testing.expectEqual(@as(usize, 4), frame.n_rows);

    const fr = frame.columnIndex("frontage").?;
    try testing.expectEqual(ColumnKind.numeric, frame.kinds[fr]);
    try testing.expect(std.math.isNan(frame.values[fr][1]));
    try testing.expectEqual(@as(f32, 80), frame.values[fr][2]);
    try testing.expectEqual(@as(u32, 0), frame.unparsed[fr]);

    const pq = frame.columnIndex("poolqc").?;
    try testing.expectEqual(ColumnKind.categorical, frame.kinds[pq]);
    // Three levels, not two: NA is one of them, and no row is missing.
    try testing.expectEqual(@as(usize, 3), frame.levels[pq].len);
    for (frame.values[pq]) |v| try testing.expect(!std.math.isNan(v));

    const stats = try profile(gpa, &frame);
    defer gpa.free(stats);
    try testing.expectEqual(@as(usize, 2), stats[fr].missing);
    try testing.expectEqual(@as(usize, 2), stats[fr].distinct);
    try testing.expectEqual(@as(usize, 0), stats[pq].missing);
    try testing.expectEqual(@as(usize, 3), stats[pq].distinct);
}

// The `unparsed` counter exists for exactly one situation, so the test has to
// reproduce it: the column kind is sniffed from the first 1000 rows, and junk
// *inside* that window simply makes the column categorical -- correctly, and
// with nothing to report. Only a value past row 1000 lands in a column already
// committed to numeric, where it used to become a silent NaN.
test "a non-numeric value past the sniff window is counted, not swallowed" {
    const gpa = testing.allocator;
    const io = testing.io;

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(gpa);
    try text.appendSlice(gpa, "v\n");
    for (0..sniff_rows + 5) |i| {
        if (i == sniff_rows + 2) {
            try text.appendSlice(gpa, "oops\n");
        } else {
            var nb: [24]u8 = undefined;
            try text.appendSlice(gpa, try std.fmt.bufPrint(&nb, "{d}\n", .{i}));
        }
    }
    try tmp.dir.writeFile(io, .{ .sub_path = "d.csv", .data = text.items });

    var path_buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, ".zig-cache/tmp/{s}/d.csv", .{tmp.sub_path});

    var pool = try Pool.init(gpa, 1);
    defer pool.deinit();
    var frame = try readCsv(gpa, io, pool, path, 1 << 20);
    defer frame.deinit();

    try testing.expectEqual(ColumnKind.numeric, frame.kinds[0]);
    try testing.expectEqual(@as(u32, 1), frame.unparsed[0]);
    try testing.expect(std.math.isNan(frame.values[0][sniff_rows + 2]));

    // A recognised marker in the same position must NOT be counted: it is a
    // declared absence, not a disagreement about the column.
    var clean: std.ArrayList(u8) = .empty;
    defer clean.deinit(gpa);
    try clean.appendSlice(gpa, "v\n");
    for (0..sniff_rows + 5) |i| {
        if (i == sniff_rows + 2) {
            try clean.appendSlice(gpa, "NA\n");
        } else {
            var nb: [24]u8 = undefined;
            try clean.appendSlice(gpa, try std.fmt.bufPrint(&nb, "{d}\n", .{i}));
        }
    }
    try tmp.dir.writeFile(io, .{ .sub_path = "c.csv", .data = clean.items });
    const path2 = try std.fmt.bufPrint(&path_buf, ".zig-cache/tmp/{s}/c.csv", .{tmp.sub_path});
    var f2 = try readCsv(gpa, io, pool, path2, 1 << 20);
    defer f2.deinit();
    try testing.expectEqual(@as(u32, 0), f2.unparsed[0]);
    try testing.expect(std.math.isNan(f2.values[0][sniff_rows + 2]));
}

test "profile: quantiles, the outer fence, and degenerate columns" {
    const gpa = testing.allocator;

    // 0..99 plus one value far above the fence. q1=24.75, q3=74.25,
    // IQR=49.5, so the outer fence sits at 74.25 + 148.5 = 222.75.
    var vals: [101]f32 = undefined;
    for (0..100) |i| vals[i] = @floatFromInt(i);
    vals[100] = 1000;

    var flat = [_]f32{ 7, 7, 7, 7 };
    var gone = [_]f32{ std.math.nan(f32), std.math.nan(f32) };

    var names = [_][]u8{ @constCast("spread"), @constCast("flat"), @constCast("gone") };
    var kinds = [_]ColumnKind{ .numeric, .numeric, .numeric };
    var values = [_][]f32{ &vals, &flat, &gone };
    var levels = [_][][]u8{ &.{}, &.{}, &.{} };

    // Ragged on purpose: only the first column's length is read as n_rows,
    // so keep the others long enough. Instead, profile each separately.
    inline for (.{
        .{ 0, @as(usize, 101) },
        .{ 1, @as(usize, 4) },
        .{ 2, @as(usize, 2) },
    }) |case| {
        const f = Frame{
            .gpa = gpa,
            .n_rows = case[1],
            .names = names[case[0] .. case[0] + 1],
            .kinds = kinds[case[0] .. case[0] + 1],
            .values = values[case[0] .. case[0] + 1],
            .levels = levels[case[0] .. case[0] + 1],
        };
        const stats = try profile(gpa, &f);
        defer gpa.free(stats);
        const s = stats[0];
        switch (case[0]) {
            0 => {
                try testing.expectEqual(@as(usize, 0), s.missing);
                try testing.expectApproxEqAbs(@as(f64, 50), s.median, 0.6);
                try testing.expectEqual(@as(f64, 1000), s.max);
                // Exactly one value past the outer fence, and none below it.
                try testing.expectEqual(@as(usize, 1), s.far_high);
                try testing.expectEqual(@as(usize, 0), s.far_low);
            },
            1 => {
                try testing.expect(s.constant());
                try testing.expect(!s.allMissing());
                // A zero-width IQR must not make every row an outlier.
                try testing.expectEqual(@as(usize, 0), s.far_high + s.far_low);
            },
            2 => {
                try testing.expect(s.allMissing());
                try testing.expectEqual(@as(usize, 2), s.missing);
            },
            else => unreachable,
        }
    }
}

// A slice of a file can carry no usable evidence about a column while being
// perfectly valid data. Found on a 298-row holdout of House Prices where every
// `PoolQC` was `NA`: markers are numeric-compatible, so the column sniffed
// numeric where training had it categorical, and predicting on it failed with
// `FeatureKindMismatch`. The fix is not to sniff harder -- it is to stop
// sniffing a column somebody already knows the answer for.
test "a schema hint outranks the evidence in the file" {
    const gpa = testing.allocator;
    const io = testing.io;

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    // Every value in `poolqc` is a missing marker, and `grade` holds digits --
    // both read as numeric with nothing to say otherwise.
    try tmp.dir.writeFile(io, .{ .sub_path = "hold.csv", .data =
        \\poolqc,grade
        \\NA,3
        \\NA,1
        \\NA,2
        \\
    });

    var path_buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, ".zig-cache/tmp/{s}/hold.csv", .{tmp.sub_path});

    var pool = try Pool.init(gpa, 1);
    defer pool.deinit();

    {
        var unhinted = try readCsv(gpa, io, pool, path, 1 << 20);
        defer unhinted.deinit();
        try testing.expectEqual(ColumnKind.numeric, unhinted.kinds[0]);
    }

    const names = [_][]const u8{ "poolqc", "grade" };
    const kinds = [_]ColumnKind{ .categorical, .numeric };
    var hinted = try readCsvHinted(gpa, io, pool, path, 1 << 20, .{
        .names = &names,
        .kinds = &kinds,
    });
    defer hinted.deinit();

    try testing.expectEqual(ColumnKind.categorical, hinted.kinds[0]);
    try testing.expectEqual(ColumnKind.numeric, hinted.kinds[1]);
    // Pinned categorical, so `NA` is interned as a level rather than dropped,
    // which is what lets it match the level the model learned.
    try testing.expectEqual(@as(usize, 1), hinted.levels[0].len);
    try testing.expectEqualStrings("NA", hinted.levels[0][0]);

    // A name the hint does not mention is still sniffed; extras in a
    // prediction file are the caller's to drop, not the parser's to reject.
    const partial = [_][]const u8{"poolqc"};
    const partial_kinds = [_]ColumnKind{.categorical};
    var mixed = try readCsvHinted(gpa, io, pool, path, 1 << 20, .{
        .names = &partial,
        .kinds = &partial_kinds,
    });
    defer mixed.deinit();
    try testing.expectEqual(ColumnKind.categorical, mixed.kinds[0]);
    try testing.expectEqual(ColumnKind.numeric, mixed.kinds[1]);
}

// ---- the loader on zsift: the three row-splitting bugs it fixed, and its fallback ----

fn readText(io: std.Io, gpa: std.mem.Allocator, pool: *Pool, text: []const u8) !Frame {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "t.csv", .data = text });
    var path_buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, ".zig-cache/tmp/{s}/t.csv", .{tmp.sub_path});
    return readCsv(gpa, io, pool, path, 1 << 28);
}

test "a newline inside a quoted field does not split the row" {
    const gpa = testing.allocator;
    var pool = try Pool.init(gpa, 1);
    defer pool.deinit();
    // The old splitter cut this into 5 rows and turned `id` categorical.
    var f = try readText(testing.io, gpa, pool, "id,note,val\n1,plain,1.5\n2,\"two\nlines\",2.5\n3,x,3.5\n");
    defer f.deinit();
    try testing.expectEqual(@as(usize, 3), f.n_rows);
    try testing.expectEqual(ColumnKind.numeric, f.kinds[0]);
    try testing.expectEqualStrings("two\nlines", f.levels[1][1]);
    try testing.expectEqual(@as(f32, 2.5), f.values[2][1]);
}

test "escaped quotes are unescaped in category levels" {
    const gpa = testing.allocator;
    var pool = try Pool.init(gpa, 1);
    defer pool.deinit();
    var f = try readText(testing.io, gpa, pool, "name\n\"say \"\"hi\"\"\"\nplain\n");
    defer f.deinit();
    try testing.expectEqualStrings("say \"hi\"", f.levels[0][0]);
}

test "more than 512 columns are all kept" {
    const gpa = testing.allocator;
    var pool = try Pool.init(gpa, 1);
    defer pool.deinit();
    var b: std.ArrayList(u8) = .empty;
    defer b.deinit(gpa);
    var buf: [16]u8 = undefined;
    for (0..3) |r| {
        for (0..600) |c| {
            if (c > 0) try b.append(gpa, ',');
            try b.appendSlice(gpa, if (r == 0) try std.fmt.bufPrint(&buf, "c{d}", .{c}) else try std.fmt.bufPrint(&buf, "{d}", .{c}));
        }
        try b.append(gpa, '\n');
    }
    var f = try readText(testing.io, gpa, pool, b.items);
    defer f.deinit();
    try testing.expectEqual(@as(usize, 600), f.names.len);
    try testing.expectEqual(@as(f32, 599), f.values[599][1]);
}

test "a stray quote in an unquoted field falls back to the lenient parser" {
    const gpa = testing.allocator;
    var pool = try Pool.init(gpa, 4);
    defer pool.deinit();
    var f = try readText(testing.io, gpa, pool, "item,v\n3\" pipe,1.5\nplain,2.5\n");
    defer f.deinit();
    try testing.expectEqual(@as(usize, 2), f.n_rows);
    try testing.expectEqualStrings("3\" pipe", f.levels[0][0]);
    try testing.expectEqual(@as(f32, 2.5), f.values[1][1]);
}

test "a file large enough to parse on many workers loads exactly as on one" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    // > 2 MiB so zsift really splits it; categorical levels first seen late, quoted
    // fields with newlines, NA in numeric and categorical columns, short rows.
    var prng = std.Random.DefaultPrng.init(11);
    const r = prng.random();
    var b: std.ArrayList(u8) = .empty;
    defer b.deinit(gpa);
    try b.appendSlice(gpa, "x,cat,y,note\n");
    var buf: [96]u8 = undefined;
    for (0..150_000) |i| {
        const cat = r.uintLessThan(u32, 40 + @as(u32, @intCast(i / 2000)));
        const x = if (r.uintLessThan(u8, 20) == 0) "NA" else try std.fmt.bufPrint(buf[48..], "{d}", .{r.uintLessThan(u32, 1000)});
        const line = if (r.uintLessThan(u8, 50) == 0)
            try std.fmt.bufPrint(&buf, "{s},c{d}\n", .{ x, cat })
        else
            try std.fmt.bufPrint(&buf, "{s},c{d},{d},\"n\n{d}\"\n", .{ x, cat, r.uintLessThan(u32, 100), i % 7 });
        try b.appendSlice(gpa, line);
    }
    try testing.expect(b.items.len > 2 << 20);

    var one = try Pool.init(gpa, 1);
    defer one.deinit();
    var many = try Pool.init(gpa, 8);
    defer many.deinit();
    var a = try readText(io, gpa, one, b.items);
    defer a.deinit();
    var m = try readText(io, gpa, many, b.items);
    defer m.deinit();

    try testing.expectEqual(a.n_rows, m.n_rows);
    for (0..a.names.len) |c| {
        try testing.expectEqual(a.kinds[c], m.kinds[c]);
        try testing.expectEqual(a.unparsed[c], m.unparsed[c]);
        try testing.expectEqual(a.levels[c].len, m.levels[c].len);
        for (a.levels[c], m.levels[c]) |x, y| try testing.expectEqualStrings(x, y);
        for (a.values[c], m.values[c]) |x, y| try testing.expectEqual(@as(u32, @bitCast(x)), @as(u32, @bitCast(y)));
    }
}
