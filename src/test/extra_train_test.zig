// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! `--extra-train`: a second file's rows appended to the first's (`csv.appendFrame`), and
//! cross-validation that trains on them in every fold without ever scoring them.

const std = @import("std");
const csv = @import("../csv.zig");
const data = @import("../data.zig");
const cv = @import("../cv.zig");
const config = @import("../config.zig");
const Fitted = @import("../fitted.zig").Fitted;
const Pool = @import("../pool.zig").Pool;
const fromBins = @import("split_test.zig").fromBins;

const testing = std.testing;

fn read(gpa: std.mem.Allocator, pool: *Pool, dir: []const u8, name: []const u8, hint: ?csv.KindHint) !csv.Frame {
    var buf: [160]u8 = undefined;
    const path = try std.fmt.bufPrint(&buf, ".zig-cache/tmp/{s}/{s}", .{ dir, name });
    return csv.readCsvHinted(gpa, testing.io, pool, path, 1 << 20, hint);
}

fn expectSameFrame(want: *const csv.Frame, got: *const csv.Frame) !void {
    try testing.expectEqual(want.n_rows, got.n_rows);
    try testing.expectEqual(want.names.len, got.names.len);
    for (want.names, got.names, want.kinds, got.kinds, want.values, got.values, want.levels, got.levels) |wn, gn, wk, gk, wv, gv, wl, gl| {
        try testing.expectEqualStrings(wn, gn);
        try testing.expectEqual(wk, gk);
        for (wv, gv) |a, b| {
            if (std.math.isNan(a)) try testing.expect(std.math.isNan(b)) else try testing.expectEqual(a, b);
        }
        try testing.expectEqual(wl.len, gl.len);
        for (wl, gl) |a, b| try testing.expectEqualStrings(a, b);
    }
}

test "appending a file equals reading the two concatenated, new levels and missing columns included" {
    const gpa = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "main.csv", .data =
        \\id,num,cat,y
        \\1,0.5,red,1
        \\2,NA,blue,0
        \\3,2.25,red,0
        \\
    });
    // No id (a dropped column), columns in another order, a shared level, two new ones, a hole.
    try tmp.dir.writeFile(io, .{ .sub_path = "extra.csv", .data =
        \\cat,y,num
        \\green,1,7
        \\blue,0,
        \\amber,1,-3.5
        \\green,0,1
        \\
    });
    try tmp.dir.writeFile(io, .{ .sub_path = "both.csv", .data =
        \\id,num,cat,y
        \\1,0.5,red,1
        \\2,NA,blue,0
        \\3,2.25,red,0
        \\,7,green,1
        \\,,blue,0
        \\,-3.5,amber,1
        \\,1,green,0
        \\
    });
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();
    var main = try read(gpa, pool, &tmp.sub_path, "main.csv", null);
    defer main.deinit();
    var extra = try read(gpa, pool, &tmp.sub_path, "extra.csv", .{ .names = main.names, .kinds = main.kinds });
    defer extra.deinit();
    var both = try read(gpa, pool, &tmp.sub_path, "both.csv", null);
    defer both.deinit();

    try csv.appendFrame(gpa, &main, &extra, &.{"id"});
    try expectSameFrame(&both, &main);
}

test "appending refuses a missing or unknown column, or a kind that differs, and changes nothing" {
    const gpa = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "main.csv", .data = "id,num,cat,y\n1,0.5,red,1\n2,1.5,blue,0\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "no_num.csv", .data = "cat,y\nred,1\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "more.csv", .data = "num,cat,y,other\n1,red,1,9\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "numcat.csv", .data = "num,cat,y\n1,4,1\n" });
    const pool = try Pool.init(gpa, 1);
    defer pool.deinit();
    var main = try read(gpa, pool, &tmp.sub_path, "main.csv", null);
    defer main.deinit();
    var before = try read(gpa, pool, &tmp.sub_path, "main.csv", null);
    defer before.deinit();

    var no_num = try read(gpa, pool, &tmp.sub_path, "no_num.csv", null);
    defer no_num.deinit();
    try testing.expectError(error.ExtraColumnMissing, csv.appendFrame(gpa, &main, &no_num, &.{"id"}));
    var more = try read(gpa, pool, &tmp.sub_path, "more.csv", null);
    defer more.deinit();
    try testing.expectError(error.ExtraColumnNotInMain, csv.appendFrame(gpa, &main, &more, &.{"id"}));
    // Read without hints, the categorical column sniffs numeric.
    var numcat = try read(gpa, pool, &tmp.sub_path, "numcat.csv", null);
    defer numcat.deinit();
    try testing.expectError(error.ExtraColumnKindDiffers, csv.appendFrame(gpa, &main, &numcat, &.{"id"}));
    try expectSameFrame(&before, &main);
}

const n_main = 900;
const n_extra = 400;

/// A binary target from two of three 8-level features, the extra rows drawn the same way
/// with a shifted intercept, as an original dataset differs a little from a synthetic one.
fn fixture(gpa: std.mem.Allocator) !data.Dataset {
    const n = n_main + n_extra;
    var cols: [3][n]u8 = undefined;
    var y: [n]f32 = undefined;
    var prng: std.Random.DefaultPrng = .init(11);
    const r = prng.random();
    for (0..n) |i| {
        for (&cols) |*c| c[i] = r.intRangeAtMost(u8, 1, 8);
        const s = @as(f32, @floatFromInt(cols[0][i])) * 0.5 - @as(f32, @floatFromInt(cols[1][i])) * 0.4 + if (i >= n_main) @as(f32, 0.3) else 0;
        y[i] = if (r.float(f32) < 1 / (1 + @exp(-s))) 1 else 0;
    }
    return fromBins(gpa, &.{ &cols[0], &cols[1], &cols[2] }, &.{ 9, 9, 9 }, &y);
}

test "cv trains every fold on the extra rows, scores only the file's, and stops early on the file's" {
    const gpa = testing.allocator;
    var ds = try fixture(gpa);
    defer ds.deinit();
    const folds = 4;
    const fold_of = try cv.assignFolds(gpa, ds.labels[0..n_main], folds, 3, true);
    defer gpa.free(fold_of);
    for ([_]u32{ 1, 3, 16 }) |threads| {
        const pool = try Pool.init(gpa, threads);
        defer pool.deinit();
        for ([_]u32{ 0, 5 }) |stop_rounds| {
            const cfg = config.Config.from(.{ .n_rounds = 40, .max_depth = 3, .learning_rate = 0.3, .verbose_eval = 0, .early_stopping_rounds = stop_rounds });
            var oof: [n_main + n_extra]f32 = undefined;
            @memset(&oof, -1);
            const o = try cv.crossValidate(gpa, testing.io, pool, &ds, cfg, fold_of, folds, .{ .oof = &oof, .early_stop_seed = 9 });
            // The extra rows are never predicted.
            for (oof[n_main..]) |v| try testing.expectEqual(@as(f32, -1), v);

            // The same, by hand: each fold fits on its training rows of the file, in order, then
            // every extra row; under early stopping the slice is the file's rows of inner part 0.
            const es = try cv.assignEarlyStop(gpa, ds.labels[0..n_main], null, 9, true);
            defer gpa.free(es);
            var want: [n_main]f32 = undefined;
            for (0..folds) |k| {
                var fit: std.ArrayList(u32) = .empty;
                defer fit.deinit(gpa);
                var stop: std.ArrayList(u32) = .empty;
                defer stop.deinit(gpa);
                var valid: std.ArrayList(u32) = .empty;
                defer valid.deinit(gpa);
                for (fold_of, 0..) |f, i| {
                    const row: u32 = @intCast(i);
                    if (f == k) try valid.append(gpa, row) else if (stop_rounds != 0 and es[i] == 0) try stop.append(gpa, row) else try fit.append(gpa, row);
                }
                for (n_main..n_main + n_extra) |i| try fit.append(gpa, @intCast(i));
                var fit_ds = try data.subset(gpa, &ds, fit.items);
                defer fit_ds.deinit();
                var valid_ds = try data.subset(gpa, &ds, valid.items);
                defer valid_ds.deinit();
                var stop_ds: ?data.Dataset = if (stop_rounds != 0) try data.subset(gpa, &ds, stop.items) else null;
                defer if (stop_ds) |*d| d.deinit();
                var res = try Fitted.train(gpa, pool, &fit_ds, if (stop_ds) |*d| d else null, cfg, null);
                defer res.model.deinit();
                const pred = try gpa.alloc(f32, valid.items.len);
                defer gpa.free(pred);
                res.model.predict(pool, &valid_ds, pred);
                for (valid.items, pred) |row, p| want[row] = p;
            }
            try testing.expectEqualSlices(f32, &want, oof[0..n_main]);
            if (stop_rounds != 0) try testing.expect(o.stopped_early);
        }
    }
}

test "extra rows weigh W, times any weight they had; W = 1 leaves the data unweighted" {
    const gpa = testing.allocator;
    var ds = try fixture(gpa);
    defer ds.deinit();
    try ds.weighRowsFrom(n_main, 1);
    try testing.expectEqual(@as(usize, 0), ds.weights.len);
    try ds.weighRowsFrom(n_main, 0.25);
    for (ds.weights, 0..) |w, i| try testing.expectEqual(@as(f32, if (i < n_main) 1 else 0.25), w);
    // An existing weight is multiplied, not replaced.
    ds.weights[n_main] = 3;
    ds.weights[0] = 2;
    try ds.weighRowsFrom(n_main, 0.5);
    try testing.expectEqual(@as(f32, 2), ds.weights[0]);
    try testing.expectEqual(@as(f32, 1.5), ds.weights[n_main]);
    try testing.expectEqual(@as(f32, 0.125), ds.weights[n_main + 1]);
    try testing.expectError(error.BadExtraWeight, ds.weighRowsFrom(n_main, -1));
    try testing.expectError(error.BadExtraWeight, ds.weighRowsFrom(n_main, std.math.nan(f32)));
    try testing.expectError(error.BadExtraWeight, ds.weighRowsFrom(n_main, std.math.inf(f32)));
}
