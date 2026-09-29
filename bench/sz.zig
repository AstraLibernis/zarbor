// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

const std = @import("std");
const z = @import("zarbor");
pub fn main() void {
    std.debug.print("Split={d}  Node={d}  BinIdx={d}\n", .{ @sizeOf(z.split.Split), @sizeOf(z.tree.Node), @sizeOf(z.data.BinIdx) });
}
