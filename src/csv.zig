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
    /// Per column, failures to parse a field that was neither empty nor a
    /// recognised missing marker. Written from every worker, so atomic; the
    /// increment only runs on the failure path and costs nothing otherwise.
    bad: []std.atomic.Value(u32),
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
                            .numeric => v = std.fmt.parseFloat(f32, t) catch blk: {
                                if (!isMissingToken(t)) _ = self.bad[c].fetchAdd(1, .monotonic);
                                break :blk std.math.nan(f32);
                            },
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
        // A pinned column is not sniffed at all: the hint is evidence from
        // the whole training set, and this file's prefix cannot outvote it.
        const pinned = try gpa.alloc(bool, n_cols);
        defer gpa.free(pinned);
        @memset(pinned, false);
        if (hint) |hh| {
            for (0..n_cols) |c| {
                if (hh.get(names[c])) |k| {
                    kinds[c] = k;
                    pinned[c] = true;
                }
            }
        }

        var fields_buf: [512][]const u8 = undefined;
        const fields = fields_buf[0..n_cols];
        const limit = @min(n_rows, sniff_rows);
        for (0..limit) |i| {
            const line = trimField(text[offs[i + 1]..offs[i + 2]]);
            const got = splitRow(line, fields);
            for (0..@min(got, n_cols)) |c| {
                if (pinned[c]) continue;
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
                // Only genuinely empty fields are skipped. A categorical
                // column's "NA" is kept as a level on purpose -- see
                // `isMissingToken` for why that is not an oversight.
                if (t.len == 0) continue;
                if (dict_storage[c] == null) dict_storage[c] = .empty;
                const m = &dict_storage[c].?;
                if (m.contains(t)) continue;
                // The `u8` bin cap is NOT enforced here. It used to be, and
                // that made `--drop` useless against a wide column: this pass
                // runs over every column in the file, before `quantise` has
                // seen the drop list, so a column the caller had explicitly
                // excluded still killed the read. The cap belongs where the
                // `u8` cast is, in `binOne`, which only runs for columns that
                // survived the drops.
                //
                // What remains here is a memory guard: a free-text column in a
                // large file would otherwise build a dictionary the size of
                // the file before anyone objected.
                if (lists[c].items.len >= max_levels) return error.TooManyLevels;
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

    const bad = try gpa.alloc(std.atomic.Value(u32), n_cols);
    defer gpa.free(bad);
    for (bad) |*b| b.* = .init(0);

    var ctx = ParseCtx{
        .text = text,
        .offs = offs[1..],
        .kinds = kinds,
        .values = values,
        .dicts = dict_ptrs,
        .bad = bad,
        .n_cols = n_cols,
        .unseen = .init(false),
    };
    pool.parallelFor(n_rows, &ctx, ParseCtx.run, 4096);

    const unparsed = try gpa.alloc(u32, n_cols);
    errdefer gpa.free(unparsed);
    for (unparsed, bad) |*u, *b| u.* = b.load(.monotonic);

    return .{
        .gpa = gpa,
        .n_rows = n_rows,
        .names = names,
        .kinds = kinds,
        .values = values,
        .levels = levels,
        .unparsed = unparsed,
    };
}

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
