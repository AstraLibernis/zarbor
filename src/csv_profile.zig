// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! What is in a loaded CSV: per-column statistics and the report `zarbor profile` prints.

const std = @import("std");
const radix = @import("radix.zig");
const csv = @import("csv.zig");
const ColumnKind = csv.ColumnKind;
const Frame = csv.Frame;

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

    // Sorted by radix on (key << 32 | bits), not std.mem.sort on f64: the stable block sort was
    // 171 ms of a 221 ms load on 668k rows x 8 numeric columns. The order is the same element for
    // element: f32 -> f64 is exact and order-preserving, the key orders as `<` does (-0 and +0
    // equal), and LSD radix is stable as the block sort was, so even signed zeros keep their places.
    const keys = try gpa.alloc(u64, 2 * f.n_rows);
    defer gpa.free(keys);
    const buf = try gpa.alloc(f64, f.n_rows);
    defer gpa.free(buf);

    for (f.names, f.kinds, f.values, 0..) |name, kind, vals, c| {
        var n: usize = 0;
        var sum: f64 = 0;
        for (vals) |v| {
            if (std.math.isNan(v)) continue;
            keys[n] = @as(u64, radix.f32Key(v)) << 32 | @as(u32, @bitCast(v));
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

        const sorted = radix.sortHigh32(keys[0..n], keys[f.n_rows..][0..n]);
        const v = buf[0..n];
        for (v, sorted) |*x, k| x.* = @as(f32, @bitCast(@as(u32, @truncate(k))));
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
                \\Categorical columns over the default --max_cat_levels (255). Drop
                \\them, replace them with a numeric summary, or raise the limit:
                \\
            );
            wide = true;
        }
        try out.print("  {s: <24}{d} levels\n", .{ s.name, s.distinct });
    }
    try out.flush();
}
