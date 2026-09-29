// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

const std = @import("std");
const pool_mod = @import("../pool.zig");
const Pool = pool_mod.Pool;
const spin_budget = pool_mod.spin_budget;

const testing = std.testing;

test "parallelFor covers every index exactly once" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 0);
    defer pool.deinit();

    const n = 100_000;
    const marks = try gpa.alloc(u8, n);
    defer gpa.free(marks);
    @memset(marks, 0);

    const Ctx = struct {
        marks: []u8,
        fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
            _ = worker;
            const self: *@This() = @ptrCast(@alignCast(ctx));
            for (self.marks[begin..end]) |*m| m.* += 1;
        }
    };
    var ctx = Ctx{ .marks = marks };
    pool.parallelFor(n, &ctx, Ctx.run, 1);

    for (marks) |m| try testing.expectEqual(@as(u8, 1), m);
}

test "workers park and wake correctly across an idle gap" {
    // The fast tests never leave a gap long enough for a worker to exceed the
    // spin budget, so they exercise the spin path only. This one stalls the
    // main thread past that budget, forcing every helper onto the futex, and
    // then checks a subsequent round still completes — which is where a lost
    // wakeup would hang forever.
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 4);
    defer pool.deinit();

    const n = 10_000;
    const vals = try gpa.alloc(u32, n);
    defer gpa.free(vals);
    @memset(vals, 0);

    const Ctx = struct {
        vals: []u32,
        fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
            _ = worker;
            const self: *@This() = @ptrCast(@alignCast(ctx));
            for (self.vals[begin..end]) |*v| v.* += 1;
        }
    };
    var ctx = Ctx{ .vals = vals };

    for (0..3) |_| {
        pool.parallelFor(n, &ctx, Ctx.run, 1);
        // Stall well past `spin_budget` so the helpers actually sleep.
        var sink: u64 = 0;
        for (0..spin_budget * 64) |i| sink +%= i;
        std.mem.doNotOptimizeAway(sink);
    }

    for (vals) |v| try testing.expectEqual(@as(u32, 3), v);
}

test "parallelFor is reusable across rounds" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 4);
    defer pool.deinit();

    const n = 10_000;
    const vals = try gpa.alloc(u32, n);
    defer gpa.free(vals);
    @memset(vals, 0);

    const Ctx = struct {
        vals: []u32,
        fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
            _ = worker;
            const self: *@This() = @ptrCast(@alignCast(ctx));
            for (self.vals[begin..end]) |*v| v.* += 1;
        }
    };
    var ctx = Ctx{ .vals = vals };
    for (0..50) |_| pool.parallelFor(n, &ctx, Ctx.run, 1);

    for (vals) |v| try testing.expectEqual(@as(u32, 50), v);
}
