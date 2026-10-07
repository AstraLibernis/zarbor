// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

const std = @import("std");
const Pool = @import("../pool.zig").Pool;
const zsift = @import("../vendor/zsift/csv.zig");
const csv = @import("../csv.zig");
const csv_profile = @import("../csv_profile.zig");
const ColumnKind = csv.ColumnKind;
const Frame = csv.Frame;
const isMissingToken = csv.isMissingToken;
const readCsv = csv.readCsv;
const sniff_rows = csv.sniff_rows;
const profile = csv_profile.profile;
const readCsvHinted = csv.readCsvHinted;

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
// reproduce it: the column kind is sniffed from the first `sniff_rows` rows, and junk
// *inside* that window simply makes the column categorical -- correctly, and
// with nothing to report. Only a value past that window lands in a column already
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
        // Two long records, both past the first range: the first must be found in file order.
        const line = if (i == 100_000 or i == 120_000)
            try std.fmt.bufPrint(&buf, "{s},c{d},1,n,extra\n", .{ x, cat })
        else if (r.uintLessThan(u8, 50) == 0)
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
    try testing.expectEqual(@as(usize, 2), m.long_rows);
    try testing.expectEqual(@as(usize, 100_001), m.first_long);
    try testing.expectEqual(@as(usize, 5), m.max_fields);
    try testing.expectEqual(a.first_long, m.first_long);
    try testing.expectEqual(a.short_rows, m.short_rows);
    try testing.expect(m.short_rows > 0);
    for (0..a.names.len) |c| {
        try testing.expectEqual(a.kinds[c], m.kinds[c]);
        try testing.expectEqual(a.unparsed[c], m.unparsed[c]);
        try testing.expectEqual(a.levels[c].len, m.levels[c].len);
        for (a.levels[c], m.levels[c]) |x, y| try testing.expectEqualStrings(x, y);
        for (a.values[c], m.values[c]) |x, y| try testing.expectEqual(@as(u32, @bitCast(x)), @as(u32, @bitCast(y)));
    }
}

test "records longer than the header are counted and refused, short ones noted" {
    const gpa = testing.allocator;
    var pool = try Pool.init(gpa, 1);
    defer pool.deinit();
    // A title line read as the header: one column, then rows of three.
    var f = try readText(testing.io, gpa, pool, "a,b\n1,2\n3,4,5\n6\n7,8,9,10\n");
    defer f.deinit();
    try testing.expectEqual(@as(usize, 4), f.n_rows);
    try testing.expectEqual(@as(usize, 2), f.long_rows);
    try testing.expectEqual(@as(usize, 2), f.first_long);
    try testing.expectEqual(@as(usize, 4), f.max_fields);
    try testing.expectEqual(@as(usize, 1), f.short_rows);
    var sink: std.Io.Writer.Allocating = .init(gpa);
    defer sink.deinit();
    try testing.expectError(error.RaggedRows, csv.checkShape(&f, &sink.writer));
    try testing.expect(std.mem.find(u8, sink.written(), "record 2.") != null);
}

test "blank lines are no records, and a well-formed file passes the shape check" {
    const gpa = testing.allocator;
    var pool = try Pool.init(gpa, 1);
    defer pool.deinit();
    for ([_][]const u8{
        "a,b\n1,2\n3,4\n",       "a,b\n1,2\n3,4",          "a,b\r\n1,2\r\n3,4\r\n",
        "a,b\n1,2\n\n3,4\n\n", "a,b\n1,2\n  \n3,4\r\n\r\n",
    }) |text| {
        var f = try readText(testing.io, gpa, pool, text);
        defer f.deinit();
        try testing.expectEqual(@as(usize, 2), f.n_rows);
        try testing.expectEqual(@as(usize, 0), f.long_rows);
        try testing.expectEqual(@as(usize, 0), f.short_rows);
        try testing.expectEqual(@as(f32, 3), f.values[0][1]);
    }
}
