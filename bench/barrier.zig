// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! What does one parallel region cost?
//!
//! Tree building submits a `parallelFor` per node -- per histogram build, per
//! reduce, per partition pass -- where xgboost batches a whole level. That
//! looked like an obvious explanation for a scaling gap, so it was measured
//! before anything was restructured: ~0.51 us at 8 threads, against roughly
//! 57,000 submissions in a 200-tree fit, is about 29 ms. Not the answer. The
//! answer turned out to be serial work inside the loop (the root gradient
//! sum, the split search) rather than the cost of going parallel at all.
//!
//! Build: zig build-exe -O ReleaseFast --dep pool -Mroot=bench/barrier.zig \
//!            -Mpool=src/pool.zig -femit-bin=/tmp/bar

const std = @import("std");
const linux = std.os.linux;
const Pool = @import("pool").Pool;

fn now() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

var sink: std.atomic.Value(u64) = .init(0);

const Ctx = struct {
    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = ctx;
        _ = worker;
        _ = begin;
        _ = end;
    }
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var buf: [4096]u8 = undefined;
    var fw = std.Io.File.stdout().writerStreaming(init.io, &buf);
    const out = &fw.interface;

    for ([_]u32{ 1, 2, 4, 8 }) |nt| {
        const pool = try Pool.init(gpa, nt);
        defer pool.deinit();
        var c = Ctx{};
        const reps: usize = 20000;
        // warm
        for (0..1000) |_| pool.parallelFor(100000, &c, Ctx.run, 1);
        var best: u64 = std.math.maxInt(u64);
        for (0..5) |_| {
            const t0 = now();
            for (0..reps) |_| pool.parallelFor(100000, &c, Ctx.run, 1);
            best = @min(best, now() - t0);
        }
        try out.print("threads={d:>2}  empty parallelFor: {d:.2} us each\n", .{
            nt, @as(f64, @floatFromInt(best)) / @as(f64, @floatFromInt(reps)) / 1000.0,
        });
        try out.flush();
    }
}
