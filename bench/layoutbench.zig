// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Uniform stride vs packed per-feature offsets, including the reduce.
//!
//! The real dataset's per-feature bin counts are `REAL_BINS`: two wide columns
//! and many narrow ones. A uniform stride sized by the widest one allocates
//! several times as many slots as there are real bins (the bench prints both),
//! so every zeroing, reduction and subtraction does that many times the
//! necessary work, and the histogram spills a cache level it need not.

const std = @import("std");
const linux = std.os.linux;

const N_ROWS: usize = 500_000;
const REAL_BINS = [_]u16{ 47, 234, 202, 6, 17, 22, 7, 4, 4, 5, 3, 3, 4 };
const N_FEAT = REAL_BINS.len;
const N_WORKERS: usize = 8;
const REPS: usize = 20;

const GradPair = extern struct { g: f32, h: f32 };
const Bin = extern struct {
    g: f64 = 0,
    h: f64 = 0,
    n: u32 = 0,
    _pad: u32 = 0,
    inline fn add(a: Bin, b: Bin) Bin {
        return .{ .g = a.g + b.g, .h = a.h + b.h, .n = a.n + b.n };
    }
};

fn now() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

const step: usize = @max(1, std.atomic.cache_line / std.math.gcd(@sizeOf(Bin), std.atomic.cache_line));
fn roundUp(n: usize) usize {
    return ((n + step - 1) / step) * step;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var buf: [8192]u8 = undefined;
    var fw = std.Io.File.stdout().writerStreaming(init.io, &buf);
    const o = &fw.interface;

    // --- the two layouts ---
    var maxb: u16 = 0;
    for (REAL_BINS) |b| maxb = @max(maxb, b);
    const uniform_stride = roundUp(maxb + 1);
    const uniform_len = N_FEAT * uniform_stride;

    var offsets: [N_FEAT + 1]usize = undefined;
    offsets[0] = 0;
    for (REAL_BINS, 0..) |b, i| offsets[i + 1] = offsets[i] + roundUp(b + 1);
    const packed_len = offsets[N_FEAT];

    try o.print(
        \\uniform stride : {d} features x {d} = {d} bins  ({d} KB/slot)
        \\packed offsets : {d} bins  ({d} KB/slot)
        \\reduction      : {d:.2}x fewer bins
        \\
        \\
    , .{
        N_FEAT,                            uniform_stride, uniform_len,
        uniform_len * @sizeOf(Bin) / 1024, packed_len,     packed_len * @sizeOf(Bin) / 1024,
        @as(f64, @floatFromInt(uniform_len)) /
            @as(f64, @floatFromInt(packed_len)),
    });

    var prng: std.Random.DefaultPrng = .init(3);
    const r = prng.random();
    const bins = try gpa.alloc(u8, N_ROWS * N_FEAT);
    defer gpa.free(bins);
    for (0..N_FEAT) |f| for (0..N_ROWS) |i| {
        bins[f * N_ROWS + i] = r.intRangeAtMost(u8, 1, @intCast(REAL_BINS[f] - 1));
    };
    const rows = try gpa.alloc(u32, N_ROWS);
    defer gpa.free(rows);
    for (rows, 0..) |*x, i| x.* = @intCast(i); // ascending: what the stable partition gives
    const grads = try gpa.alloc(GradPair, N_ROWS);
    defer gpa.free(grads);
    for (grads) |*g| g.* = .{ .g = r.floatNorm(f32), .h = 1.0 };

    const priv_u = try gpa.alloc(Bin, N_WORKERS * uniform_len);
    defer gpa.free(priv_u);
    const out_u = try gpa.alloc(Bin, uniform_len);
    defer gpa.free(out_u);
    const priv_p = try gpa.alloc(Bin, N_WORKERS * packed_len);
    defer gpa.free(priv_p);
    const out_p = try gpa.alloc(Bin, packed_len);
    defer gpa.free(out_p);

    // Sweep node sizes: a depth-6 tree's levels hold roughly the full row
    // count, then halves. The clear and the reduce are paid per node whatever
    // its size, so the waste factor bites hardest at the bottom of the tree —
    // which is also where most of the nodes are.
    try o.print("  {s:>10} {s:>12} {s:>12} {s:>8}\n", .{ "rows/node", "uniform ms", "packed ms", "speedup" });
    const sizes = [_]usize{ 500_000, 125_000, 30_000, 8_000, 2_000 };
    for (sizes) |node_rows| {
        const share = @max(node_rows / N_WORKERS, 1);

        var bu: u64 = std.math.maxInt(u64);
        for (0..REPS) |_| {
            const t0 = now();
            for (0..N_WORKERS) |w| for (0..N_FEAT) |f| {
                @memset(priv_u[w * uniform_len + f * uniform_stride ..][0..uniform_stride], .{});
            };
            for (0..N_FEAT) |f| {
                const col = bins[f * N_ROWS ..][0..N_ROWS];
                const h = priv_u[f * uniform_stride ..][0..uniform_stride];
                for (rows[0..share], grads[0..share]) |row, g| {
                    const b = col[row];
                    h[b].g += g.g;
                    h[b].h += g.h;
                    h[b].n += 1;
                }
            }
            for (0..uniform_len) |i| {
                var acc = priv_u[i];
                for (1..N_WORKERS) |w| acc = acc.add(priv_u[w * uniform_len + i]);
                out_u[i] = acc;
            }
            bu = @min(bu, now() - t0);
        }

        var bp: u64 = std.math.maxInt(u64);
        for (0..REPS) |_| {
            const t0 = now();
            for (0..N_WORKERS) |w| for (0..N_FEAT) |f| {
                @memset(priv_p[w * packed_len + offsets[f] .. w * packed_len + offsets[f + 1]], .{});
            };
            for (0..N_FEAT) |f| {
                const col = bins[f * N_ROWS ..][0..N_ROWS];
                const h = priv_p[offsets[f]..offsets[f + 1]];
                for (rows[0..share], grads[0..share]) |row, g| {
                    const b = col[row];
                    h[b].g += g.g;
                    h[b].h += g.h;
                    h[b].n += 1;
                }
            }
            for (0..packed_len) |i| {
                var acc = priv_p[i];
                for (1..N_WORKERS) |w| acc = acc.add(priv_p[w * packed_len + i]);
                out_p[i] = acc;
            }
            bp = @min(bp, now() - t0);
        }

        const mu2 = @as(f64, @floatFromInt(bu)) / 1e6;
        const mp2 = @as(f64, @floatFromInt(bp)) / 1e6;
        try o.print("  {d:>10} {d:>12.4} {d:>12.4} {d:>7.2}x\n", .{ node_rows, mu2, mp2, mu2 / mp2 });
    }
    try o.flush();
}
