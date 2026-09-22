const std = @import("std");
const z = @import("zarbor");
pub fn main() void {
    std.debug.print("Split={d}  Node={d}  BinIdx={d}\n", .{ @sizeOf(z.hist.Split), @sizeOf(z.tree.Node), @sizeOf(z.data.BinIdx) });
}
