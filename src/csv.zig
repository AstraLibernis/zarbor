// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! CSV parsing: bytes on disk to a column-major `Frame`. Every model pays
//! this before any bin, so it has its own module, tests and timings;
//! `data.zig` re-exports `Frame`, `ColumnKind`, `readCsv`. Faster than
//! `pandas.read_csv`; see docs/measurements.md and docs/archive/arena.md.

const std = @import("std");
const Pool = @import("pool.zig").Pool;
const zsift = @import("vendor/zsift/csv.zig");

pub const ColumnKind = enum { numeric, categorical };

/// A parsed, unquantised table. Column-major: every pass is per-column.
pub const Frame = struct {
    gpa: std.mem.Allocator,
    n_rows: usize,
    names: [][]u8,
    kinds: []ColumnKind,
    /// Numeric value, or categorical dictionary id as float. NaN = missing.
    values: [][]f32,
    /// Dictionary for each categorical column, indexed by id. Empty otherwise.
    levels: [][][]u8,
    /// Per column: non-empty, non-marker fields that failed to parse in a
    /// sniffed-numeric column. Nonzero means an unrepresentative sniff prefix
    /// or junk; reported rather than silently stored as NaN.
    unparsed: []u32 = &.{},
    /// Records with more fields than the header. Their extra fields are not
    /// stored, so a nonzero count means the columns may not line up with the
    /// header: callers refuse the file (`checkShape`).
    long_rows: usize = 0,
    /// The first such record, counting data records from 1.
    first_long: usize = 0,
    /// Most fields in any record.
    max_fields: usize = 0,
    /// Records with fewer fields than the header; their missing fields are NaN.
    short_rows: usize = 0,

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

/// Missing markers R, pandas, Excel and SQL exports emit. Deliberately
/// short: no entry is a word anyone uses as data.
const missing_tokens = [_][]const u8{
    "na", "n/a", "#n/a", "nan", "null", "none", "nil", "?",
};

/// Empty or a missing marker. With `looksNumeric`: **a marker means
/// "missing" only in a column whose other values are numbers.** Ames: in
/// `LotFrontage` `NA` is unrecorded; in `PoolQC` it is the level "No Pool".
/// All-missing (pandas default) turns 14 columns' documented categories into
/// "unknown"; never-missing makes `MasVnrArea` a 328-level categorical and the
/// file will not load. Per column costs one sniff comparison, no config.
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

/// Read-time distinct-value ceiling: stops a free-text column allocating a
/// file-sized dictionary. The bin limit is in `binOne`, which refuses a
/// categorical with `card >= max_bins` or `card > max_cat_levels`.
const max_levels: usize = 1 << 20;

/// Known column kinds by name, from the model's stored schema, so a
/// prediction file cannot re-sniff differently. A 298-row House Prices
/// holdout with all-`NA` `PoolQC` sniffed numeric and died with
/// `FeatureKindMismatch` on well-formed data.
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

/// Parse a CSV with vendored zsift (`vendor/zsift`): quoted fields may hold
/// delimiters, newlines, `""`; no column limit. Up to `pool.workerCount()`
/// workers, capped at `zsift.parallel.max_workers` and at one per
/// `bytes_per_worker` of text (`parseText`), each with own columns and
/// dictionaries, merged in file order so level ids match a serial read. A stray quote in an
/// unquoted field (not RFC 4180) is re-read serially by the lenient parser.
pub fn readCsv(
    gpa: std.mem.Allocator,
    io: std.Io,
    pool: *Pool,
    path: []const u8,
    max_bytes: usize,
) !Frame {
    return readCsvHinted(gpa, io, pool, path, max_bytes, null);
}

/// `hint` pins named columns' kinds; others are sniffed. Unmentioned columns
/// are not an error: dropping extras is the caller's job.
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
    // `parseText` frees the text once every field is parsed, before the merge: the file, the
    // per-range columns and the merged columns were all alive at once, the peak of every command.
    var owned: ?[]u8 = text;
    defer if (owned) |t| gpa.free(t);
    const workers = @min(pool.workerCount(), zsift.parallel.max_workers);
    return parseText(gpa, io, text, &owned, workers, hint, .strict) catch |e| switch (e) {
        // Not RFC 4180 (a quote inside an unquoted field): read it leniently instead. The strict
        // pass fails while parsing, so the text has not been freed.
        error.InvalidQuote => parseText(gpa, io, text, &owned, 1, hint, .lenient),
        else => e,
    };
}

const Mode = enum { strict, lenient };

/// CSV bytes per worker. zsift's rule (`parallel.forEachField`, serial below
/// `parallel.min_parallel_bytes`) suits a near-free sink; this one converts
/// every field. Chosen by a sweep; see docs/measurements.md (csv.zig
/// `bytes_per_worker`).
const bytes_per_worker: usize = 64 << 10;

/// Scratch per worker for unescaping one quoted field with `""` in it; a longer
/// escaped field is a `ScratchTooSmall` error, not a truncation.
const field_scratch_max: usize = 16 << 20;

/// Unescape scratch is allocated raw, skipping `alloc`'s fill: Debug and ReleaseSafe write 0xAA over
/// every allocation and again on free, and this is up to `field_scratch_max` (16 MiB) per worker.
/// Only a field containing `""` writes it.
fn allocScratch(gpa: std.mem.Allocator, len: usize) ![]u8 {
    const p = gpa.rawAlloc(len, .of(u8), @returnAddress()) orelse return error.OutOfMemory;
    return p[0..len];
}

fn freeScratch(gpa: std.mem.Allocator, scratch: []u8) void {
    gpa.rawFree(scratch, .of(u8), @returnAddress());
}

/// `owner` holds `text` if the caller owns it; it is freed (and set to null) once all fields are
/// parsed, since nothing after that reads it: categorical levels are copied.
fn parseText(
    gpa: std.mem.Allocator,
    io: std.Io,
    text: []const u8,
    owner: *?[]u8,
    workers: usize,
    hint: ?KindHint,
    comptime mode: Mode,
) !Frame {
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
        const scratch = try allocScratch(gpa, scratch_len);
        defer freeScratch(gpa, scratch);
        var it = try P.init(text, scratch, .{});
        while (try it.next()) |f| {
            try names.append(gpa, try gpa.dupe(u8, trimField(f.bytes)));
            it.resetScratch();
            if (f.last_in_record) break;
        }
        if (names.items.len == 0) return error.NoColumns;
        try kinds.appendNTimes(gpa, .numeric, names.items.len);
        // Pinned columns are not sniffed: whole-training-set evidence beats a prefix.
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
    defer for (scratches[0..n_scratch]) |sc| freeScratch(gpa, sc);
    for (ptrs, scratches, sinks) |*ptr, *sc, *s| {
        ptr.* = s;
        sc.* = try allocScratch(gpa, scratch_len);
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
    if (owner.*) |t| {
        gpa.free(t);
        owner.* = null;
    }
    for (scratches[0..n_scratch]) |sc| freeScratch(gpa, sc);
    n_scratch = 0;

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
        // Merged: the ranges' copy of this column is dead weight while the next is built.
        for (sinks) |*s| s.cols[c].clearAndFree(gpa);
    }

    var shape: struct { long: usize = 0, first: usize = 0, max: usize = 0, short: usize = 0 } = .{};
    var before: usize = 0;
    for (sinks) |*s| {
        if (s.first_long) |r| if (shape.long == 0) {
            shape.first = before + r + 1;
        };
        shape.long += s.long_rows;
        shape.short += s.short_rows;
        shape.max = @max(shape.max, s.max_fields);
        before += s.rows;
    }

    return .{
        .gpa = gpa,
        .n_rows = n_rows,
        .names = try names.toOwnedSlice(gpa),
        .kinds = try kinds.toOwnedSlice(gpa),
        .values = values,
        .levels = levels,
        .unparsed = unparsed,
        .long_rows = shape.long,
        .first_long = shape.first,
        .max_fields = shape.max,
        .short_rows = shape.short,
    };
}

/// Refuses a frame whose records have more fields than its header, saying
/// why, since the extra fields were dropped and the columns may be shifted;
/// mentions short records, which are kept with NaN for their missing fields.
pub fn checkShape(f: *const Frame, out: *std.Io.Writer) !void {
    if (f.short_rows != 0) try out.print(
        "note: {d} record(s) have fewer fields than the header's {d}; their missing fields are read as missing\n",
        .{ f.short_rows, f.names.len },
    );
    if (f.long_rows == 0) return;
    try out.print(
        \\{d} record(s) have more fields than the header's {d} (up to {d}), the first being data
        \\record {d}. Their extra fields would be dropped and the columns may not line up. Usually
        \\the first line is not the header (a comment or title line), or a field holds an unquoted
        \\comma. Fix the file; zarbor does not guess.
        \\
    , .{ f.long_rows, f.names.len, f.max_fields, f.first_long });
    try out.flush();
    return error.RaggedRows;
}

/// Append `extra`'s rows to `main`, matching columns by name: the result is what reading the two
/// files concatenated would give (categorical ids keep `main`'s, and a level only `extra` has
/// takes the next id, in `extra`'s order). Read `extra` with `main`'s kinds as hints so the two
/// agree. A column of `main` that `extra` lacks is refused unless named in `may_lack` (it is
/// then missing on the appended rows); a column only `extra` has is refused. On error `main` is
/// unchanged.
pub fn appendFrame(gpa: std.mem.Allocator, main: *Frame, extra: *const Frame, may_lack: []const []const u8) !void {
    for (extra.names) |n| if (main.columnIndex(n) == null) return error.ExtraColumnNotInMain;
    const n_cols = main.names.len;
    const src = try gpa.alloc(?usize, n_cols);
    defer gpa.free(src);
    for (main.names, main.kinds, src) |n, kind, *s| {
        s.* = extra.columnIndex(n);
        if (s.*) |e| {
            if (extra.kinds[e] != kind) return error.ExtraColumnKindDiffers;
        } else {
            var ok = false;
            for (may_lack) |m| ok = ok or std.mem.eql(u8, m, n);
            if (!ok) return error.ExtraColumnMissing;
        }
    }

    const n_total = main.n_rows + extra.n_rows;
    // Built in full before anything of `main` changes, so a failure leaves it as it was.
    const new_values = try gpa.alloc([]f32, n_cols);
    @memset(new_values, &.{});
    defer gpa.free(new_values);
    errdefer for (new_values) |v| gpa.free(v);
    const new_levels = try gpa.alloc([][]u8, n_cols);
    @memset(new_levels, &.{});
    defer gpa.free(new_levels);
    // Only the levels added here are owned by `new_levels` until commit; `main`'s are borrowed.
    errdefer for (new_levels, main.levels) |nl, ml| {
        for (nl[@min(ml.len, nl.len)..]) |l| gpa.free(l);
        gpa.free(nl);
    };

    for (0..n_cols) |c| {
        const v = try gpa.alloc(f32, n_total);
        new_values[c] = v;
        @memcpy(v[0..main.n_rows], main.values[c]);
        const tail = v[main.n_rows..];
        const e = src[c] orelse {
            @memset(tail, std.math.nan(f32));
            continue;
        };
        if (main.kinds[c] == .numeric) {
            @memcpy(tail, extra.values[e]);
            continue;
        }
        // Categorical: map each of `extra`'s level ids to `main`'s, adding new levels in order.
        var ids: std.StringHashMapUnmanaged(u32) = .empty;
        defer ids.deinit(gpa);
        for (main.levels[c], 0..) |l, i| try ids.put(gpa, l, @intCast(i));
        var levels: std.ArrayList([]u8) = .empty;
        errdefer {
            for (levels.items[@min(main.levels[c].len, levels.items.len)..]) |l| gpa.free(l);
            levels.deinit(gpa);
        }
        try levels.appendSlice(gpa, main.levels[c]);
        const map = try gpa.alloc(f32, extra.levels[e].len);
        defer gpa.free(map);
        for (extra.levels[e], map) |l, *m| {
            if (ids.get(l)) |id| {
                m.* = @floatFromInt(id);
            } else {
                const own = try gpa.dupe(u8, l);
                levels.append(gpa, own) catch |err| {
                    gpa.free(own);
                    return err;
                };
                try ids.put(gpa, own, @intCast(levels.items.len - 1));
                m.* = @floatFromInt(levels.items.len - 1);
            }
        }
        for (tail, extra.values[e]) |*t, x| t.* = if (std.math.isNan(x)) x else map[@intFromFloat(x)];
        new_levels[c] = try levels.toOwnedSlice(gpa);
    }

    // Commit: nothing below fails.
    for (0..n_cols) |c| {
        gpa.free(main.values[c]);
        main.values[c] = new_values[c];
        if (main.kinds[c] == .categorical and src[c] != null) {
            gpa.free(main.levels[c]); // the outer slice only: its strings live on in `new_levels`
            main.levels[c] = new_levels[c];
        }
    }
    main.n_rows = n_total;
}

/// Concatenate column `c` of every range into `out`; categorical local ids are
/// renumbered in range order (first appearance in file). Returns owned levels.
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

/// One range's rows, column-major, with local first-appearance dictionaries. One writer.
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
    long_rows: usize = 0,
    /// The first long record, counting this range's records from 0.
    first_long: ?usize = null,
    max_fields: usize = 0,
    short_rows: usize = 0,

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
        // A blank line is no record, as in pandas: it would otherwise be a row with every
        // field missing, an extra prediction row in `predict`'s output.
        if (last and s.col == 0 and trimField(bytes).len == 0) return;
        if (s.col < s.kinds.len) s.put(s.col, bytes) catch |e| {
            s.err = e;
        };
        s.col += 1;
        if (last) {
            s.max_fields = @max(s.max_fields, s.col);
            if (s.col > s.kinds.len) {
                s.long_rows += 1;
                if (s.first_long == null) s.first_long = s.rows;
            } else if (s.col < s.kinds.len) s.short_rows += 1;
            // A short record: its missing columns are NaN. Extra fields are counted, not stored.
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
        // Categorical: only an empty field is missing; `NA` is a level.
        if (t.len != 0) switch (s.kinds[c]) {
            .numeric => v = std.fmt.parseFloat(f32, t) catch blk: {
                if (!isMissingToken(t)) s.bad[c] += 1;
                break :blk std.math.nan(f32);
            },
            .categorical => {
                const gop = try s.dicts[c].getOrPut(s.gpa, t);
                if (!gop.found_existing) {
                    // Memory guard, not the bin cap (`binOne`'s).
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
