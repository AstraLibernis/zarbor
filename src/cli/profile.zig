// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! `zarbor profile`: what is in a CSV, before any model.

const std = @import("std");
const zarbor = @import("zarbor");
const args = @import("args.zig");
const data = zarbor.data;
const pool_mod = zarbor.pool;
const csv = zarbor.csv;

/// `zarbor profile <data.csv>` -- describe a file without training on it.
///
/// This exists because the alternative is a throwaway pandas script per
/// dataset, and that script is where the understanding then lives: outside
/// the tool, unversioned, and different every time. The parser has already
/// walked every byte, so it is the right place to answer what is in the file.
pub fn run(init: std.process.Init, gpa: std.mem.Allocator, out: *std.Io.Writer) !void {
    const io = init.io;
    var path: ?[]const u8 = null;
    var max_bytes: usize = 1 << 31;

    var it = args.Iterator.init(init.minimal.args, 1);
    while (it.next()) |arg| switch (arg) {
        .positional => |a| path = a,
        .flag => |f| if (std.mem.eql(u8, f.key, "max-bytes")) {
            max_bytes = try std.fmt.parseInt(usize, f.val, 10);
        },
        .bare => {},
    };
    const p = path orelse {
        try out.writeAll("usage: zarbor profile <data.csv> [--max-bytes=N]\n");
        try out.flush();
        return error.NoCsvPath;
    };

    const pool = try pool_mod.Pool.init(gpa, 0);
    defer pool.deinit();

    const t0 = std.Io.Timestamp.now(io, .awake).toNanoseconds();
    var frame = try data.readCsv(gpa, io, pool, p, max_bytes);
    defer frame.deinit();
    const t_read = std.Io.Timestamp.now(io, .awake).toNanoseconds();

    const stats = try csv.profile(gpa, &frame);
    defer gpa.free(stats);
    const t_prof = std.Io.Timestamp.now(io, .awake).toNanoseconds();

    try out.print("file    {s}\nread    {d} ms\nprofile {d} ms\n", .{
        p, @divTrunc(t_read - t0, 1_000_000), @divTrunc(t_prof - t_read, 1_000_000),
    });
    try csv.writeProfile(out, &frame, stats);
}
