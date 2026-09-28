// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! How expensive is `sigmoid`, and whose fault is it?
//!
//! Gradients were 15% of a boosting fit and almost all of it was `@exp`.
//! Three candidates: Zig's `@exp` (compiler_rt), glibc's `expf`, and an
//! inlined vector `2^x`. glibc is *not* faster -- scalar transcendental
//! evaluation is the cost, not anyone's implementation -- and the vector form
//! is 11.7x, accurate to under 1e-6 absolute.
//!
//! Build: zig build-exe bench/expbench.zig -O ReleaseFast -lc -femit-bin=/tmp/eb

const std = @import("std");
const linux = std.os.linux;

extern fn expf(f32) f32;

fn now() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

const N = 1 << 20;

var sink: f32 = 0;

fn consume(ys: []const f32) void {
    var t: f32 = 0;
    for (ys) |y| t += y;
    sink += t;
}

/// 8-wide sigmoid: exp2 by exponent construction plus a degree-5 polynomial.
/// exp(-x) = 2^(-x*log2e), split into integer and fractional parts.
inline fn sigmoid8(x: @Vector(8, f32)) @Vector(8, f32) {
    const V = @Vector(8, f32);
    const lim: V = @splat(@as(f32, 88.0));
    const t = @min(@max(-x, -lim), lim);
    const log2e: V = @splat(@as(f32, 1.44269504));
    const y = t * log2e;
    const yr = @round(y);
    const f = y - yr;
    // 2^f on [-0.5, 0.5], minimax degree 5.
    const c1: V = @splat(@as(f32, 0.6931472));
    const c2: V = @splat(@as(f32, 0.2402265));
    const c3: V = @splat(@as(f32, 0.0555041));
    const c4: V = @splat(@as(f32, 0.0096181));
    const c5: V = @splat(@as(f32, 0.0013333));
    const one: V = @splat(@as(f32, 1.0));
    const p = one + f * (c1 + f * (c2 + f * (c3 + f * (c4 + f * c5))));
    const ei: @Vector(8, i32) = @intFromFloat(yr);
    const bits: @Vector(8, i32) = (ei + @as(@Vector(8, i32), @splat(127))) << @splat(@as(u5, 23));
    const scale: V = @bitCast(bits);
    return one / (one + p * scale);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var buf: [4096]u8 = undefined;
    var fw = std.Io.File.stdout().writerStreaming(init.io, &buf);
    const out = &fw.interface;

    const xs = try gpa.alloc(f32, N);
    defer gpa.free(xs);
    const ys = try gpa.alloc(f32, N);
    defer gpa.free(ys);
    var prng: std.Random.DefaultPrng = .init(3);
    const r = prng.random();
    for (xs) |*x| x.* = (r.float(f32) - 0.5) * 12.0;

    const reps = 200;
    var best: u64 = std.math.maxInt(u64);
    for (0..5) |_| {
        const t0 = now();
        for (0..reps) |_| {
            for (xs, ys) |x, *y| y.* = 1.0 / (1.0 + @exp(-x));
            consume(ys);
        }
        best = @min(best, now() - t0);
    }
    const base = @as(f64, @floatFromInt(best)) / @as(f64, @floatFromInt(N * reps));
    try out.print("zig @exp sigmoid      {d:.3} ns/elem  1.00x\n", .{base});

    best = std.math.maxInt(u64);
    for (0..5) |_| {
        const t0 = now();
        for (0..reps) |_| {
            for (xs, ys) |x, *y| y.* = 1.0 / (1.0 + expf(-x));
            consume(ys);
        }
        best = @min(best, now() - t0);
    }
    var ns = @as(f64, @floatFromInt(best)) / @as(f64, @floatFromInt(N * reps));
    try out.print("glibc expf sigmoid    {d:.3} ns/elem  {d:.2}x\n", .{ ns, base / ns });

    best = std.math.maxInt(u64);
    for (0..5) |_| {
        const t0 = now();
        for (0..reps) |_| {
            var i: usize = 0;
            while (i + 8 <= N) : (i += 8) {
                const v: @Vector(8, f32) = xs[i..][0..8].*;
                ys[i..][0..8].* = sigmoid8(v);
            }
            consume(ys);
        }
        best = @min(best, now() - t0);
    }
    ns = @as(f64, @floatFromInt(best)) / @as(f64, @floatFromInt(N * reps));
    try out.print("vector sigmoid8       {d:.3} ns/elem  {d:.2}x\n", .{ ns, base / ns });

    // Accuracy of the vector version against the scalar one.
    var max_abs: f32 = 0;
    var i: usize = 0;
    while (i + 8 <= N) : (i += 8) {
        const v: @Vector(8, f32) = xs[i..][0..8].*;
        const got: [8]f32 = sigmoid8(v);
        for (got, 0..) |g, k| {
            const want = 1.0 / (1.0 + @exp(-xs[i + k]));
            max_abs = @max(max_abs, @abs(g - want));
        }
    }
    try out.print("vector max abs error  {e:.3}\n", .{max_abs});
    try out.print("(sink {d:.1})\n", .{sink});
    try out.flush();
}
