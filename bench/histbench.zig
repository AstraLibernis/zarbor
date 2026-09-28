// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Standalone microbenchmark for the histogram accumulation kernel.
//!
//! hist_build is 45-63% of training, so this isolates it from tree building
//! entirely: synthetic bins, a shuffled row order standing in for the arbitrary
//! ids a node holds after partitioning, and one variant per idea. Iterating
//! here takes seconds instead of a two-minute training run, and removes every
//! confound the full pipeline brings.
//!
//! Build: zig build-exe bench/histbench.zig -O ReleaseFast -femit-bin=/tmp/hb

const std = @import("std");
const linux = std.os.linux;

const N_ROWS: usize = 500_000;
const N_FEAT: usize = 13;
const N_BINS: usize = 256;
const REPS: usize = 20;

fn now() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

const GradPair = extern struct { g: f32, h: f32 };

/// What the code uses today: two doubles, a count, and padding to 24 bytes.
const Bin24 = extern struct {
    g: f64 = 0,
    h: f64 = 0,
    n: u32 = 0,
    _pad: u32 = 0,
};
/// Counts dropped: 16 bytes, two per cache line quarter.
const Bin16 = extern struct { g: f64 = 0, h: f64 = 0 };
/// Single-precision sums with the count kept: 12 bytes.
const Bin12 = extern struct { g: f32 = 0, h: f32 = 0, n: u32 = 0 };
/// Single precision, no count: 8 bytes.
const Bin8 = extern struct { g: f32 = 0, h: f32 = 0 };
/// Single-precision sums with the count, padded to a 16-byte power of two.
/// This is the shape that keeps everything the split search needs.
const Bin16f = extern struct { g: f32 = 0, h: f32 = 0, n: u32 = 0, _pad: u32 = 0 };
/// Doubles plus count, padded to 32: isolates alignment from size.
const Bin32 = extern struct { g: f64 = 0, h: f64 = 0, n: u32 = 0, _p0: u32 = 0, _p1: u64 = 0 };

fn strideFor(comptime T: type) usize {
    const line = std.atomic.cache_line;
    const step = @max(1, line / std.math.gcd(@sizeOf(T), line));
    return ((N_BINS + 1 + step - 1) / step) * step;
}

// --------------------------------------------------------------- variants

/// Current implementation: feature-outer, unrolled by four.
fn featureOuter(comptime T: type, hist: []T, bins: []const u8, rows: []const u32, grads: []const GradPair) void {
    const stride = strideFor(T);
    for (0..N_FEAT) |f| {
        const col = bins[f * N_ROWS ..][0..N_ROWS];
        const h = hist[f * stride ..][0..stride];
        var i: usize = 0;
        while (i + 4 <= rows.len) : (i += 4) {
            const b0 = col[rows[i + 0]];
            const b1 = col[rows[i + 1]];
            const b2 = col[rows[i + 2]];
            const b3 = col[rows[i + 3]];
            accum(T, &h[b0], grads[i + 0]);
            accum(T, &h[b1], grads[i + 1]);
            accum(T, &h[b2], grads[i + 2]);
            accum(T, &h[b3], grads[i + 3]);
        }
        while (i < rows.len) : (i += 1) accum(T, &h[col[rows[i]]], grads[i]);
    }
}

/// Same, plus an explicit prefetch of the gather target D iterations ahead.
/// This is the technique xgboost's binary is full of (~4,900 prefetch
/// instructions, and no AVX-512 at all).
fn featureOuterPrefetch(comptime T: type, hist: []T, bins: []const u8, rows: []const u32, grads: []const GradPair, comptime D: usize) void {
    const stride = strideFor(T);
    for (0..N_FEAT) |f| {
        const col = bins[f * N_ROWS ..][0..N_ROWS];
        const h = hist[f * stride ..][0..stride];
        var i: usize = 0;
        while (i + 4 <= rows.len) : (i += 4) {
            if (i + D + 4 <= rows.len) {
                // Pull the *column* bytes in — that is the random access.
                @prefetch(&col[rows[i + D + 0]], .{ .locality = 0 });
                @prefetch(&col[rows[i + D + 2]], .{ .locality = 0 });
                @prefetch(&rows[i + D * 2], .{ .locality = 1 });
            }
            const b0 = col[rows[i + 0]];
            const b1 = col[rows[i + 1]];
            const b2 = col[rows[i + 2]];
            const b3 = col[rows[i + 3]];
            accum(T, &h[b0], grads[i + 0]);
            accum(T, &h[b1], grads[i + 1]);
            accum(T, &h[b2], grads[i + 2]);
            accum(T, &h[b3], grads[i + 3]);
        }
        while (i < rows.len) : (i += 1) accum(T, &h[col[rows[i]]], grads[i]);
    }
}

/// Row-outer over a row-major bin matrix: every feature of a row is
/// contiguous, so `rows` and `grads` are read once instead of N_FEAT times.
fn rowOuterRowMajor(comptime T: type, hist: []T, bins_rm: []const u8, rows: []const u32, grads: []const GradPair) void {
    const stride = strideFor(T);
    for (rows, 0..) |row, i| {
        const rb = bins_rm[row * N_FEAT ..][0..N_FEAT];
        const g = grads[i];
        inline for (0..N_FEAT) |f| accum(T, &hist[f * stride + rb[f]], g);
    }
}

/// Row-outer, row-major, with the next row's bins prefetched.
fn rowOuterRowMajorPrefetch(comptime T: type, hist: []T, bins_rm: []const u8, rows: []const u32, grads: []const GradPair, comptime D: usize) void {
    const stride = strideFor(T);
    for (rows, 0..) |row, i| {
        if (i + D < rows.len) @prefetch(&bins_rm[rows[i + D] * N_FEAT], .{ .locality = 0 });
        const rb = bins_rm[row * N_FEAT ..][0..N_FEAT];
        const g = grads[i];
        inline for (0..N_FEAT) |f| accum(T, &hist[f * stride + rb[f]], g);
    }
}

/// Gradient sums in a 16-byte bin, counts in a separate u32 array.
///
/// The counts are what force the bin up to 24 bytes today. Split out, the
/// gradient array is 13x272x16 = 56 KB and the count array 14 KB — and the
/// count array is small enough to stay in L1 while the gradients stream.
fn rowOuterSplitCounts(hist: []Bin16, counts: []u32, bins_rm: []const u8, rows: []const u32, grads: []const GradPair, comptime D: usize) void {
    const stride = strideFor(Bin16);
    for (rows, 0..) |row, i| {
        if (i + D < rows.len) @prefetch(&bins_rm[rows[i + D] * N_FEAT], .{ .locality = 0 });
        const rb = bins_rm[row * N_FEAT ..][0..N_FEAT];
        const g = grads[i];
        inline for (0..N_FEAT) |f| {
            const k = f * stride + rb[f];
            hist[k].g += g.g;
            hist[k].h += g.h;
            counts[k] += 1;
        }
    }
}

/// Stage a block of rows' bins, prefetch each histogram slot, then accumulate.
/// This is the shape LightGBM uses: the bin read and the scattered
/// read-modify-write are separated so the latter's misses overlap.
fn rowOuterBlocked(hist: []Bin16, counts: []u32, bins_rm: []const u8, rows: []const u32, grads: []const GradPair, comptime B: usize, comptime D: usize) void {
    const stride = strideFor(Bin16);
    var buf: [B][N_FEAT]u8 = undefined;
    var i: usize = 0;
    while (i + B <= rows.len) : (i += B) {
        for (0..B) |k| {
            if (i + D + k < rows.len) @prefetch(&bins_rm[rows[i + D + k] * N_FEAT], .{ .locality = 0 });
            const rb = bins_rm[rows[i + k] * N_FEAT ..][0..N_FEAT];
            inline for (0..N_FEAT) |f| buf[k][f] = rb[f];
        }
        for (0..B) |k| inline for (0..N_FEAT) |f| {
            @prefetch(&hist[f * stride + buf[k][f]], .{ .rw = .write, .locality = 3 });
        };
        for (0..B) |k| {
            const g = grads[i + k];
            inline for (0..N_FEAT) |f| {
                const idx = f * stride + buf[k][f];
                hist[idx].g += g.g;
                hist[idx].h += g.h;
                counts[idx] += 1;
            }
        }
    }
    while (i < rows.len) : (i += 1) {
        const rb = bins_rm[rows[i] * N_FEAT ..][0..N_FEAT];
        const g = grads[i];
        inline for (0..N_FEAT) |f| {
            const idx = f * stride + rb[f];
            hist[idx].g += g.g;
            hist[idx].h += g.h;
            counts[idx] += 1;
        }
    }
}

/// Runs the split-count variants, which need a second output array and so do
/// not fit the generic `bench` signature.
fn benchSplit(name: []const u8, out: *std.Io.Writer, base: f64, f: anytype, args: anytype) !void {
    const gpa = std.heap.page_allocator;
    const stride = strideFor(Bin16);
    const hist = try gpa.alloc(Bin16, N_FEAT * stride);
    defer gpa.free(hist);
    const counts = try gpa.alloc(u32, N_FEAT * stride);
    defer gpa.free(counts);

    var best: u64 = std.math.maxInt(u64);
    var checksum: f64 = 0;
    var total: u64 = 0;
    for (0..REPS) |_| {
        @memset(hist, .{});
        @memset(counts, 0);
        const t0 = now();
        @call(.auto, f, .{ hist, counts } ++ args);
        best = @min(best, now() - t0);
        checksum = 0;
        total = 0;
        for (hist) |b| checksum += b.g;
        for (counts) |c| total += c;
    }
    const ms = @as(f64, @floatFromInt(best)) / 1e6;
    try out.print("  {s:<34} {d:>8.2} ms  {d:>5.2} ns/update  {d:>5.2}x  sum={d:.1} n={d}\n", .{
        name, ms, @as(f64, @floatFromInt(best)) / @as(f64, @floatFromInt(N_ROWS * N_FEAT)), base / ms, checksum, total,
    });
    try out.flush();
}

inline fn accum(comptime T: type, h: *T, g: GradPair) void {
    switch (T) {
        Bin24 => {
            h.g += g.g;
            h.h += g.h;
            h.n += 1;
        },
        Bin16 => {
            h.g += g.g;
            h.h += g.h;
        },
        Bin12 => {
            h.g += g.g;
            h.h += g.h;
            h.n += 1;
        },
        Bin8 => {
            h.g += g.g;
            h.h += g.h;
        },
        Bin16f, Bin32 => {
            h.g += g.g;
            h.h += g.h;
            h.n += 1;
        },
        else => unreachable,
    }
}

// ------------------------------------------------------------------ driver

fn bench(
    name: []const u8,
    comptime T: type,
    out: *std.Io.Writer,
    baseline: *?f64,
    f: anytype,
    args: anytype,
) !void {
    const gpa = std.heap.page_allocator;
    const stride = strideFor(T);
    const hist = try gpa.alloc(T, N_FEAT * stride);
    defer gpa.free(hist);

    var best: u64 = std.math.maxInt(u64);
    var checksum: f64 = 0;
    for (0..REPS) |_| {
        @memset(hist, .{});
        const t0 = now();
        @call(.auto, f, .{ T, hist } ++ args);
        const dt = now() - t0;
        best = @min(best, dt);
        checksum = 0;
        for (hist) |b| checksum += b.g;
    }
    const ms = @as(f64, @floatFromInt(best)) / 1e6;
    const updates = @as(f64, @floatFromInt(N_ROWS * N_FEAT));
    const ns_per = @as(f64, @floatFromInt(best)) / updates;

    var rel: f64 = 1.0;
    if (baseline.*) |b| rel = b / ms else baseline.* = ms;

    try out.print("  {s:<34} {d:>8.2} ms  {d:>5.2} ns/update  {d:>5.2}x  sum={d:.1}\n", .{ name, ms, ns_per, rel, checksum });
    try out.flush();
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var buf: [8192]u8 = undefined;
    var fw = std.Io.File.stdout().writerStreaming(init.io, &buf);
    const out = &fw.interface;

    var prng: std.Random.DefaultPrng = .init(7);
    const r = prng.random();

    const bins_cm = try gpa.alloc(u8, N_ROWS * N_FEAT); // column-major
    defer gpa.free(bins_cm);
    const bins_rm = try gpa.alloc(u8, N_ROWS * N_FEAT); // row-major
    defer gpa.free(bins_rm);
    for (0..N_FEAT) |f| for (0..N_ROWS) |i| {
        const v = r.intRangeAtMost(u8, 1, N_BINS - 1);
        bins_cm[f * N_ROWS + i] = v;
        bins_rm[i * N_FEAT + f] = v;
    };

    const rows = try gpa.alloc(u32, N_ROWS);
    defer gpa.free(rows);

    // Which order a node's rows arrive in decides the whole question, and it
    // is a property of the partition, not a free choice. The parallel
    // partition is a stable count/prefix/scatter, so each child keeps its
    // rows in ascending order; the old Hoare two-pointer swap did not.
    // ZB_SHUFFLE=1 reproduces the unordered case for comparison.
    var shuffled = false;
    {
        var it = std.process.Args.Iterator.init(init.minimal.args);
        _ = it.skip();
        while (it.next()) |arg| {
            if (std.mem.eql(u8, arg, "--shuffle")) shuffled = true;
        }
    }
    for (rows, 0..) |*x, i| x.* = @intCast(i);
    if (shuffled) r.shuffle(u32, rows);

    const grads = try gpa.alloc(GradPair, N_ROWS);
    defer gpa.free(grads);
    for (grads) |*g| g.* = .{ .g = r.floatNorm(f32), .h = 1.0 };

    try out.print(
        \\histogram kernel: {d} rows x {d} features x {d} bins, row order: {s}
        \\({d} reps, best-of; 'x' is speedup over the current implementation)
        \\
        \\
    , .{ N_ROWS, N_FEAT, N_BINS, if (shuffled) "SHUFFLED" else "ASCENDING", REPS });

    var base: ?f64 = null;
    try bench("current: Bin24 feature-outer", Bin24, out, &base, featureOuter, .{ bins_cm, rows, grads });

    try out.print("\n  -- narrower bins, same loop --\n", .{});
    var b2: ?f64 = base;
    try bench("Bin16 (drop count)", Bin16, out, &b2, featureOuter, .{ bins_cm, rows, grads });
    b2 = base;
    try bench("Bin12 (f32 sums, keep count)", Bin12, out, &b2, featureOuter, .{ bins_cm, rows, grads });
    b2 = base;
    try bench("Bin8  (f32 sums, no count)", Bin8, out, &b2, featureOuter, .{ bins_cm, rows, grads });

    try out.print("\n  -- keeping the count: size vs alignment --\n", .{});
    b2 = base;
    try bench("Bin16f (f32 sums + count, 16B)", Bin16f, out, &b2, featureOuter, .{ bins_cm, rows, grads });
    b2 = base;
    try bench("Bin32 (f64 sums + count, 32B)", Bin32, out, &b2, featureOuter, .{ bins_cm, rows, grads });

    try out.print("\n  -- prefetch distance, current layout --\n", .{});
    inline for (.{ 8, 16, 32, 64 }) |D| {
        b2 = base;
        try bench(std.fmt.comptimePrint("Bin24 + prefetch D={d}", .{D}), Bin24, out, &b2, featureOuterPrefetch, .{ bins_cm, rows, grads, D });
    }

    try out.print("\n  -- row-major layout, row-outer loop --\n", .{});
    b2 = base;
    try bench("Bin24 row-major row-outer", Bin24, out, &b2, rowOuterRowMajor, .{ bins_rm, rows, grads });
    inline for (.{ 8, 16, 32 }) |D| {
        b2 = base;
        try bench(std.fmt.comptimePrint("Bin24 row-major + prefetch D={d}", .{D}), Bin24, out, &b2, rowOuterRowMajorPrefetch, .{ bins_rm, rows, grads, D });
    }
    inline for (.{ 4, 8, 12, 16 }) |D| {
        b2 = base;
        try bench(std.fmt.comptimePrint("Bin16  row-major + prefetch D={d}", .{D}), Bin16, out, &b2, rowOuterRowMajorPrefetch, .{ bins_rm, rows, grads, D });
    }
    inline for (.{ 4, 8, 16 }) |D| {
        b2 = base;
        try bench(std.fmt.comptimePrint("Bin16f row-major + prefetch D={d}", .{D}), Bin16f, out, &b2, rowOuterRowMajorPrefetch, .{ bins_rm, rows, grads, D });
    }
    b2 = base;
    try bench("Bin32  row-major + prefetch D=8", Bin32, out, &b2, rowOuterRowMajorPrefetch, .{ bins_rm, rows, grads, 8 });

    try out.print("\n  -- f64 sums AND counts, counts in their own array --\n", .{});
    inline for (.{ 4, 8, 16 }) |D| {
        try benchSplit(std.fmt.comptimePrint("split counts + prefetch D={d}", .{D}), out, base.?, rowOuterSplitCounts, .{ bins_rm, rows, grads, D });
    }

    try out.print("\n  -- blocked: stage bins, prefetch hist slots, accumulate --\n", .{});
    inline for (.{ 8, 16, 32 }) |B| {
        try benchSplit(std.fmt.comptimePrint("blocked B={d} D=16", .{B}), out, base.?, rowOuterBlocked, .{ bins_rm, rows, grads, B, 16 });
    }

    try out.flush();
}
