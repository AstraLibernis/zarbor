// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! The one argument syntax every command takes: positionals and `--key=value`.
//! Each command decides what an unknown or valueless flag means.

const std = @import("std");

pub const Arg = union(enum) {
    positional: []const u8,
    flag: struct { key: []const u8, val: []const u8 },
    /// `--key` with no `=value`.
    bare: []const u8,
};

pub const Iterator = struct {
    inner: std.process.Args.Iterator,

    /// Skips argv[0], then `skip` more (a subcommand name).
    pub fn init(args: std.process.Args, skip: usize) Iterator {
        var inner = std.process.Args.Iterator.init(args);
        for (0..skip + 1) |_| _ = inner.skip();
        return .{ .inner = inner };
    }

    pub fn next(it: *Iterator) ?Arg {
        const arg = it.inner.next() orelse return null;
        if (!std.mem.startsWith(u8, arg, "--")) return .{ .positional = arg };
        const body = arg[2..];
        const eq = std.mem.findScalar(u8, body, '=') orelse return .{ .bare = body };
        return .{ .flag = .{ .key = body[0..eq], .val = body[eq + 1 ..] } };
    }
};
