// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! What shape should a histogram bin be?
//!
//! `histbench.zig` answered "which loop order", but it models every feature as
//! 256 bins wide. Real tables are not like that: the benchmark dataset has 559
//! bins across 13 features, and 437 of those belong to just two columns. The
//! whole histogram is 16 KB and L1-resident, where the uniform-256 model makes
//! it 80 KB and L2-resident — a different regime, and the regime the code
//! actually runs in. This file uses the real widths and the real packed
//! per-feature offsets.
//!
//! The question it exists to answer: the count field. Dropping it is worth
//! 1.5x in histbench, but `min_child_samples` needs it. Three scattered
//! read-modify-writes per row-feature is what costs — so the candidate is to
//! make them *one*: pad the bin to four f64 lanes and let a single 32-byte
//! vector load-add-store carry g, h and n together.
//!
//! Run: zig build bench-binshape -Doptimize=ReleaseFast

const std = @import("std");
const linux = std.os.linux;

const N_ROWS: usize = 500_000;
const REPS: usize = 30;

/// The benchmark dataset's actual bin widths, in column order.
const WIDTHS = [_]usize{ 47, 234, 203, 6, 17, 22, 7, 4, 4, 5, 3, 3, 4 };
const N_FEAT = WIDTHS.len;

fn now() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

const GradPair = extern struct { g: f32, h: f32 };

/// Packed per-feature offsets, each feature starting on a cache line, exactly
/// as `hist.Bank` lays them out.
fn offsetsFor(comptime T: type) [N_FEAT + 1]u32 {
    @setEvalBranchQuota(10_000);
    const line = std.atomic.cache_line;
    const step = @max(1, line / std.math.gcd(@sizeOf(T), line));
    var o: [N_FEAT + 1]u32 = undefined;
    var acc: usize = 0;
    for (WIDTHS, 0..) |w, i| {
        o[i] = @intCast(acc);
        acc += ((w + step - 1) / step) * step;
    }
    o[N_FEAT] = @intCast(acc);
    return o;
}

// ------------------------------------------------------------- bin shapes

/// Today's shape: two doubles, a count, padding. Three separate RMWs.
const Bin24 = extern struct { g: f64 = 0, h: f64 = 0, n: u32 = 0, _pad: u32 = 0 };
/// The count dropped. Two RMWs, and the index is a shift rather than a
/// multiply. This is the 1.5x, and the semantics we cannot afford to lose.
const Bin16 = extern struct { g: f64 = 0, h: f64 = 0 };
/// Four f64 lanes: g, h, n, unused. 32 bytes, naturally aligned, so the whole
/// update is one vector load, one add, one store -- counts included.
const Vec4 = @Vector(4, f64);
/// Two lanes: gradient and hessian only, 16 bytes. This is xgboost's shape —
/// it constrains splits by hessian sum and never stores a row count. Here to
/// price what the count lane costs in the layout the code actually uses.
const Vec2 = @Vector(2, f64);

// ---------------------------------------------------------------- kernels

fn scalar24(hist: []Bin24, bins: []const u8, rows: []const u32, grads: []const GradPair) void {
    const off = comptime offsetsFor(Bin24);
    for (0..N_FEAT) |f| {
        const col = bins[f * N_ROWS ..][0..N_ROWS];
        const h = hist[off[f]..off[f + 1]];
        for (rows, 0..) |row, i| {
            const b = col[row];
            const g = grads[i];
            h[b].g += g.g;
            h[b].h += g.h;
            h[b].n += 1;
        }
    }
}

fn scalar16(hist: []Bin16, bins: []const u8, rows: []const u32, grads: []const GradPair) void {
    const off = comptime offsetsFor(Bin16);
    for (0..N_FEAT) |f| {
        const col = bins[f * N_ROWS ..][0..N_ROWS];
        const h = hist[off[f]..off[f + 1]];
        for (rows, 0..) |row, i| {
            const b = col[row];
            const g = grads[i];
            h[b].g += g.g;
            h[b].h += g.h;
        }
    }
}

/// Feature-outer, one 32-byte vector RMW per row.
fn vec4FeatureOuter(hist: []Vec4, bins: []const u8, rows: []const u32, grads: []const GradPair) void {
    const off = comptime offsetsFor(Vec4);
    for (0..N_FEAT) |f| {
        const col = bins[f * N_ROWS ..][0..N_ROWS];
        const h = hist[off[f]..off[f + 1]];
        for (rows, 0..) |row, i| {
            const b = col[row];
            const g = grads[i];
            h[b] += Vec4{ g.g, g.h, 1, 0 };
        }
    }
}

/// The same, unrolled by four to expose independent chains.
fn vec4Unrolled(hist: []Vec4, bins: []const u8, rows: []const u32, grads: []const GradPair) void {
    const off = comptime offsetsFor(Vec4);
    for (0..N_FEAT) |f| {
        const col = bins[f * N_ROWS ..][0..N_ROWS];
        const h = hist[off[f]..off[f + 1]];
        var i: usize = 0;
        while (i + 4 <= rows.len) : (i += 4) {
            const b0 = col[rows[i + 0]];
            const b1 = col[rows[i + 1]];
            const b2 = col[rows[i + 2]];
            const b3 = col[rows[i + 3]];
            const g0 = grads[i + 0];
            const g1 = grads[i + 1];
            const g2 = grads[i + 2];
            const g3 = grads[i + 3];
            h[b0] += Vec4{ g0.g, g0.h, 1, 0 };
            h[b1] += Vec4{ g1.g, g1.h, 1, 0 };
            h[b2] += Vec4{ g2.g, g2.h, 1, 0 };
            h[b3] += Vec4{ g3.g, g3.h, 1, 0 };
        }
        while (i < rows.len) : (i += 1) {
            const b = col[rows[i]];
            const g = grads[i];
            h[b] += Vec4{ g.g, g.h, 1, 0 };
        }
    }
}

/// Today's scalar shape, unrolled by four -- the actual current kernel.
fn scalar24Unrolled(hist: []Bin24, bins: []const u8, rows: []const u32, grads: []const GradPair) void {
    const off = comptime offsetsFor(Bin24);
    for (0..N_FEAT) |f| {
        const col = bins[f * N_ROWS ..][0..N_ROWS];
        const h = hist[off[f]..off[f + 1]];
        var i: usize = 0;
        while (i + 4 <= rows.len) : (i += 4) {
            const b0 = col[rows[i + 0]];
            const b1 = col[rows[i + 1]];
            const b2 = col[rows[i + 2]];
            const b3 = col[rows[i + 3]];
            const g0 = grads[i + 0];
            const g1 = grads[i + 1];
            const g2 = grads[i + 2];
            const g3 = grads[i + 3];
            h[b0].g += g0.g;
            h[b0].h += g0.h;
            h[b0].n += 1;
            h[b1].g += g1.g;
            h[b1].h += g1.h;
            h[b1].n += 1;
            h[b2].g += g2.g;
            h[b2].h += g2.h;
            h[b2].n += 1;
            h[b3].g += g3.g;
            h[b3].h += g3.h;
            h[b3].n += 1;
        }
        while (i < rows.len) : (i += 1) {
            const b = col[rows[i]];
            const g = grads[i];
            h[b].g += g.g;
            h[b].h += g.h;
            h[b].n += 1;
        }
    }
}

/// Vector bins, row-major matrix, row-outer: `rows` and `grads` are read once
/// per row instead of once per row per feature.
fn vec4RowMajor(hist: []Vec4, bins_rm: []const u8, rows: []const u32, grads: []const GradPair, comptime D: usize) void {
    const off = comptime offsetsFor(Vec4);
    for (rows, 0..) |row, i| {
        if (i + D < rows.len) @prefetch(&bins_rm[rows[i + D] * N_FEAT], .{ .locality = 0 });
        const rb = bins_rm[row * N_FEAT ..][0..N_FEAT];
        const g = grads[i];
        const v = Vec4{ g.g, g.h, 1, 0 };
        inline for (0..N_FEAT) |f| hist[off[f] + rb[f]] += v;
    }
}

/// Today's 24-byte bin, but row-major and row-outer. Separates the layout
/// change from the bin-shape change: whatever this gains is the layout's, and
/// whatever Vec4 row-major gains on top of it is the vector RMW's.
fn scalar24RowMajor(hist: []Bin24, bins_rm: []const u8, rows: []const u32, grads: []const GradPair, comptime D: usize) void {
    const off = comptime offsetsFor(Bin24);
    for (rows, 0..) |row, i| {
        if (i + D < rows.len) @prefetch(&bins_rm[rows[i + D] * N_FEAT], .{ .locality = 0 });
        const rb = bins_rm[row * N_FEAT ..][0..N_FEAT];
        const g = grads[i];
        inline for (0..N_FEAT) |f| {
            const k = off[f] + rb[f];
            hist[k].g += g.g;
            hist[k].h += g.h;
            hist[k].n += 1;
        }
    }
}

/// Vector bins, row-major, no prefetch: is the prefetch doing anything, or is
/// the hardware prefetcher already covering the ascending walk?
fn vec4RowMajorNoPf(hist: []Vec4, bins_rm: []const u8, rows: []const u32, grads: []const GradPair) void {
    const off = comptime offsetsFor(Vec4);
    for (rows, 0..) |row, i| {
        const rb = bins_rm[row * N_FEAT ..][0..N_FEAT];
        const g = grads[i];
        const v = Vec4{ g.g, g.h, 1, 0 };
        inline for (0..N_FEAT) |f| hist[off[f] + rb[f]] += v;
    }
}

/// The honest version of row-major: the feature set is a runtime slice, not
/// a comptime range, because `colsample` picks it per tree. The offsets are
/// then runtime loads and the 13 updates cannot be unrolled into independent
/// immediate-offset chains. If most of the gain above was the `inline for`,
/// it disappears here -- which is the whole reason to measure it.
fn vec4RowMajorRuntime(hist: []Vec4, bins_rm: []const u8, rows: []const u32, grads: []const GradPair, off: []const u32, feats: []const u32, comptime D: usize) void {
    for (rows, 0..) |row, i| {
        if (i + D < rows.len) @prefetch(&bins_rm[rows[i + D] * N_FEAT], .{ .locality = 0 });
        const rb = bins_rm[row * N_FEAT ..][0..N_FEAT];
        const g = grads[i];
        const v = Vec4{ g.g, g.h, 1, 0 };
        for (feats) |fid| hist[off[fid] + rb[fid]] += v;
    }
}

/// Runtime features, but four rows in flight at once so the four independent
/// chains per feature overlap.
fn vec4RowMajorRuntime4(hist: []Vec4, bins_rm: []const u8, rows: []const u32, grads: []const GradPair, off: []const u32, feats: []const u32, comptime D: usize) void {
    var i: usize = 0;
    while (i + 4 <= rows.len) : (i += 4) {
        if (i + D + 4 <= rows.len) {
            @prefetch(&bins_rm[rows[i + D] * N_FEAT], .{ .locality = 0 });
            @prefetch(&bins_rm[rows[i + D + 2] * N_FEAT], .{ .locality = 0 });
        }
        const r0 = bins_rm[rows[i + 0] * N_FEAT ..][0..N_FEAT];
        const r1 = bins_rm[rows[i + 1] * N_FEAT ..][0..N_FEAT];
        const r2 = bins_rm[rows[i + 2] * N_FEAT ..][0..N_FEAT];
        const r3 = bins_rm[rows[i + 3] * N_FEAT ..][0..N_FEAT];
        const g0 = grads[i + 0];
        const g1 = grads[i + 1];
        const g2 = grads[i + 2];
        const g3 = grads[i + 3];
        const v0 = Vec4{ g0.g, g0.h, 1, 0 };
        const v1 = Vec4{ g1.g, g1.h, 1, 0 };
        const v2 = Vec4{ g2.g, g2.h, 1, 0 };
        const v3 = Vec4{ g3.g, g3.h, 1, 0 };
        for (feats) |fid| {
            const base = off[fid];
            hist[base + r0[fid]] += v0;
            hist[base + r1[fid]] += v1;
            hist[base + r2[fid]] += v2;
            hist[base + r3[fid]] += v3;
        }
    }
    while (i < rows.len) : (i += 1) {
        const rb = bins_rm[rows[i] * N_FEAT ..][0..N_FEAT];
        const g = grads[i];
        const v = Vec4{ g.g, g.h, 1, 0 };
        for (feats) |fid| hist[off[fid] + rb[fid]] += v;
    }
}

/// Row-major, two-lane bins: no count.
fn vec2RowMajor(hist: []Vec2, bins_rm: []const u8, rows: []const u32, grads: []const GradPair) void {
    const off = comptime offsetsFor(Vec2);
    for (rows, 0..) |row, i| {
        const rb = bins_rm[row * N_FEAT ..][0..N_FEAT];
        const g = grads[i];
        const v = Vec2{ g.g, g.h };
        inline for (0..N_FEAT) |f| hist[off[f] + rb[f]] += v;
    }
}

/// Two independent vector accumulators per feature, combined at the end.
/// Breaks the serial dependency when consecutive rows land in the same bin,
/// which is common for the narrow categorical columns.
fn vec4TwoBank(hist: []Vec4, bins: []const u8, rows: []const u32, grads: []const GradPair) void {
    const off = comptime offsetsFor(Vec4);
    const span = off[N_FEAT];
    for (0..N_FEAT) |f| {
        const col = bins[f * N_ROWS ..][0..N_ROWS];
        const h0 = hist[off[f]..off[f + 1]];
        const h1 = hist[span + off[f] ..][0 .. off[f + 1] - off[f]];
        var i: usize = 0;
        while (i + 2 <= rows.len) : (i += 2) {
            const b0 = col[rows[i + 0]];
            const b1 = col[rows[i + 1]];
            const g0 = grads[i + 0];
            const g1 = grads[i + 1];
            h0[b0] += Vec4{ g0.g, g0.h, 1, 0 };
            h1[b1] += Vec4{ g1.g, g1.h, 1, 0 };
        }
        while (i < rows.len) : (i += 1) {
            const b = col[rows[i]];
            const g = grads[i];
            h0[b] += Vec4{ g.g, g.h, 1, 0 };
        }
    }
}

// ----------------------------------------------------------------- driver

fn run(
    name: []const u8,
    comptime T: type,
    out: *std.Io.Writer,
    base: *?f64,
    n_rows: usize,
    f: anytype,
    args: anytype,
) !void {
    const gpa = std.heap.page_allocator;
    const off = offsetsFor(T);
    // Two slots always: the two-bank variant uses the second half, and every
    // other variant simply leaves it zero. Keeping one signature means the
    // allocation is identical across variants, so it cannot skew the timing.
    const hist = try gpa.alignedAlloc(T, .fromByteUnits(64), 2 * off[N_FEAT]);
    defer gpa.free(hist);

    var best: u64 = std.math.maxInt(u64);
    var checksum: f64 = 0;
    var count: f64 = 0;
    for (0..REPS) |_| {
        @memset(hist, std.mem.zeroes(T));
        const t0 = now();
        @call(.auto, f, .{hist} ++ args);
        const dt = now() - t0;
        best = @min(best, dt);
        checksum = 0;
        count = 0;
        for (hist) |b| {
            if (T == Vec4) {
                checksum += b[0];
                count += b[2];
            } else if (T == Vec2) {
                checksum += b[0];
            } else {
                checksum += b.g;
                if (@hasField(T, "n")) count += @floatFromInt(b.n);
            }
        }
    }
    const ms = @as(f64, @floatFromInt(best)) / 1e6;
    const updates: f64 = @floatFromInt(n_rows * N_FEAT);
    var rel: f64 = 1.0;
    if (base.*) |b| rel = b / ms else base.* = ms;
    try out.print("  {s:<32} {d:>8.3} ms {d:>6.3} ns/upd {d:>6.2}x  sum={d:.1} n={d:.0}\n", .{
        name, ms, @as(f64, @floatFromInt(best)) / updates, rel, checksum, count,
    });
    try out.flush();
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var buf: [8192]u8 = undefined;
    var fw = std.Io.File.stdout().writerStreaming(init.io, &buf);
    const out = &fw.interface;

    // Node sizes matter as much as the kernel: a depth-6 tree spends half its
    // node-builds in the last level, on a few thousand rows each.
    var sizes: []const usize = &.{ N_ROWS, 60_000, 8_000, 2_000 };
    {
        var it = std.process.Args.Iterator.init(init.minimal.args);
        _ = it.skip();
        while (it.next()) |arg| {
            if (std.mem.eql(u8, arg, "--full")) sizes = &.{N_ROWS};
        }
    }

    var prng: std.Random.DefaultPrng = .init(7);
    const r = prng.random();

    const bins_cm = try gpa.alloc(u8, N_ROWS * N_FEAT);
    defer gpa.free(bins_cm);
    const bins_rm = try gpa.alloc(u8, N_ROWS * N_FEAT);
    defer gpa.free(bins_rm);
    for (0..N_FEAT) |f| for (0..N_ROWS) |i| {
        // Real widths, so the narrow categorical columns collide constantly
        // and the two wide numeric ones barely do.
        const v = r.intRangeLessThan(u8, 1, @intCast(WIDTHS[f]));
        bins_cm[f * N_ROWS + i] = v;
        bins_rm[i * N_FEAT + f] = v;
    };

    const rows = try gpa.alloc(u32, N_ROWS);
    defer gpa.free(rows);
    // Ascending: the stable partition leaves a node's rows in order, and a
    // shuffled benchmark recommends the opposite optimisations.
    for (rows, 0..) |*x, i| x.* = @intCast(i);

    const grads = try gpa.alloc(GradPair, N_ROWS);
    defer gpa.free(grads);
    for (grads) |*g| g.* = .{ .g = r.floatNorm(f32), .h = 1.0 };

    const off24 = offsetsFor(Bin24);
    const off32 = offsetsFor(Vec4);
    try out.print(
        \\bin shape: {d} features, {d} real bins, ascending rows, best of {d}
        \\slot: Bin24 {d} bins = {d:.1} KB | Vec4 {d} bins = {d:.1} KB
        \\
    , .{
        N_FEAT,
        comptime blk: {
            var t: usize = 0;
            for (WIDTHS) |w| t += w;
            break :blk t;
        },
        REPS,
        off24[N_FEAT],
        @as(f64, @floatFromInt(off24[N_FEAT] * 24)) / 1024.0,
        off32[N_FEAT],
        @as(f64, @floatFromInt(off32[N_FEAT] * 32)) / 1024.0,
    });

    // Runtime copies of what the comptime variants get for free.
    const off32rt = offsetsFor(Vec4);
    var feats: [N_FEAT]u32 = undefined;
    for (&feats, 0..) |*x, i| x.* = @intCast(i);

    for (sizes) |n| {
        try out.print("\n-- node of {d} rows --\n", .{n});
        const rs = rows[0..n];
        const gs = grads[0..n];
        var base: ?f64 = null;
        try run("current: Bin24 x4 unrolled", Bin24, out, &base, n, scalar24Unrolled, .{ bins_cm, rs, gs });
        try run("Bin24 plain loop", Bin24, out, &base, n, scalar24, .{ bins_cm, rs, gs });
        try run("Bin16 no count (not viable)", Bin16, out, &base, n, scalar16, .{ bins_cm, rs, gs });
        try run("Vec4 32B, one RMW", Vec4, out, &base, n, vec4FeatureOuter, .{ bins_cm, rs, gs });
        try run("Vec4 x4 unrolled", Vec4, out, &base, n, vec4Unrolled, .{ bins_cm, rs, gs });
        try run("Vec4 two banks", Vec4, out, &base, n, vec4TwoBank, .{ bins_cm, rs, gs });
        try run("Bin24 row-major D=8", Bin24, out, &base, n, scalar24RowMajor, .{ bins_rm, rs, gs, 8 });
        try run("Vec4 row-major D=8", Vec4, out, &base, n, vec4RowMajor, .{ bins_rm, rs, gs, 8 });
        try run("Vec4 row-major, no prefetch", Vec4, out, &base, n, vec4RowMajorNoPf, .{ bins_rm, rs, gs });
        try run("Vec2 row-major (no count)", Vec2, out, &base, n, vec2RowMajor, .{ bins_rm, rs, gs });
        try run("Vec4 rm, runtime feats", Vec4, out, &base, n, vec4RowMajorRuntime, .{ bins_rm, rs, gs, &off32rt, &feats, 8 });
        try run("Vec4 rm, runtime feats, 4 rows", Vec4, out, &base, n, vec4RowMajorRuntime4, .{ bins_rm, rs, gs, &off32rt, &feats, 8 });
    }
}
