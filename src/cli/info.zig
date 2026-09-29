// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! `zgbdt info`: what a saved model holds.

const std = @import("std");
const zarbor = @import("zarbor");
const args = @import("args.zig");
const forest = zarbor.forest;
const linear = zarbor.linear;
const model_mod = zarbor.model;

pub fn run(init: std.process.Init, gpa: std.mem.Allocator, out: *std.Io.Writer) !void {
    var path: ?[]const u8 = null;
    var it = args.Iterator.init(init.minimal.args, 1);
    while (it.next()) |arg| switch (arg) {
        .flag => |f| if (std.mem.eql(u8, f.key, "model")) {
            path = f.val;
        },
        else => {},
    };
    const p = path orelse return error.NoModel;

    var b = try model_mod.load(gpa, init.io, p);
    defer b.deinit();

    try out.print(
        \\file      {s}
        \\kind      {s}
        \\objective {s}
        \\features  {d}
        \\
    , .{ p, @tagName(b.kind), @tagName(b.objective), b.schema.n_features });

    if (b.label.len != 0) {
        try out.print("label     {s}", .{b.label});
        for (b.classes, 0..) |c, i| try out.print("{s}\"{s}\"={d}", .{ if (i == 0) "  " else ", ", c, i });
        if (b.classes.len == 0) try out.writeAll("  (numeric)");
        try out.writeAll("\n");
    }

    switch (b.kind) {
        .gbdt, .forest => {
            var nodes: usize = 0;
            var leaves: usize = 0;
            for (b.trees) |t| {
                nodes += t.nodes.len;
                for (t.nodes) |n| {
                    if (n.is_leaf) leaves += 1;
                }
            }
            try out.print("trees     {d} ({d} nodes, {d} leaves)\n", .{ b.trees.len, nodes, leaves });
            if (b.kind == .gbdt) try out.print("base      {d:.6}\n", .{b.base_score});
        },
        .linear => {
            var zero: usize = 0;
            for (b.lin.?.w) |c| {
                if (c == 0) zero += 1;
            }
            try out.print("coefs     {d} ({d} zero)\nintercept {d:.6}\n", .{ b.lin.?.w.len, zero, b.lin.?.intercept });
        },
    }

    try out.print("\nschema\n", .{});
    for (0..b.schema.n_features) |f| {
        try out.print("  {s:<32} {s:<12} {d:>4} bins\n", .{
            b.schema.names[f],
            @tagName(b.schema.kinds[f]),
            b.schema.n_bins[f],
        });
    }
    try out.flush();
}
