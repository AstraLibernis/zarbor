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
/// Ceiling on distinct values in one column while reading. Far above any
/// usable categorical -- it exists so a free-text column cannot allocate a
/// dictionary the size of the file. The bin-width limit that actually matters
/// is `max_bins`, checked in `binOne` after drops are applied.
const max_levels: usize = 1 << 20;

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
