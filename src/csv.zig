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

pub const sniff_rows = 1000;

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

/// The full table, for `zarbor profile`.
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
