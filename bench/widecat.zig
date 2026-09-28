// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! What does one wide categorical column cost?
//!
//! zarbor stores a bin in a `u8`, so a column with 256 or more levels is
//! refused. Lifting that ceiling is only worth doing if the tree builder can
//! afford the wider histogram it implies, and this measures that before the
//! rewrite rather than after.
//!
//! Two halves, because one of them can be measured with today's code and the
//! other cannot:
//!
//!  A. **Everything except the scatter.** `Bank` already takes a per-feature
//!     bin count as a `u16`, so a feature can be *declared* 753 bins wide
//!     today. The rows only ever land in bins 0..255, but the per-node clear,
//!     the reduce and the split search all walk the full declared width --
//!     which is exactly what they would do after the change. This is the term
//!     zarbor's own notes say dominates at the bottom of a tree.
//!
//!  B. **The scatter.** Accumulation touches one cell per row per feature, so
//!     widening a column does not change the instruction count, only the
//!     working set. Half A cannot see that, because its rows crowd into the
//!     first 256 cells. Measured separately with a standalone u16 loop.
//!
//! Run: zig build bench-widecat -Doptimize=ReleaseFast

const std = @import("std");
const zarbor = @import("zarbor");
const data = zarbor.data;
const hist = zarbor.hist;
const tree = zarbor.tree;
const config = zarbor.config;
const Pool = zarbor.pool.Pool;

fn now() u64 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

const n_numeric: usize = 12;
const numeric_bins: u16 = 64;

/// `n_numeric` numeric columns plus one categorical *declared* `cat_bins` wide.
fn makeDs(gpa: std.mem.Allocator, n_rows: usize, cat_bins: u16, seed: u64) !data.Dataset {
    const nf = n_numeric + 1;
    const bins = try gpa.alloc(data.BinIdx, nf * n_rows);
    const bins_rm = try gpa.alloc(data.BinIdx, nf * n_rows);
    const labels = try gpa.alloc(f32, n_rows);
    var prng: std.Random.DefaultPrng = .init(seed);
    const r = prng.random();
    for (0..n_rows) |row| {
        var acc: f32 = 0;
        for (0..nf) |f| {
            const top: data.BinIdx = if (f == n_numeric)
                @intCast(@min(@as(usize, cat_bins) - 1, 255))
            else
                @intCast(numeric_bins - 1);
            const b = r.intRangeAtMost(data.BinIdx, 1, top);
            bins[f * n_rows + row] = b;
            bins_rm[row * nf + f] = b;
            if (f < 3) acc += @floatFromInt(b);
        }
        labels[row] = acc + r.floatNorm(f32) * 8;
    }
    const nb = try gpa.alloc(u16, nf);
    for (nb, 0..) |*v, f| v.* = if (f == n_numeric) cat_bins else numeric_bins;
    const kinds = try gpa.alloc(data.ColumnKind, nf);
    for (kinds, 0..) |*k, f| k.* = if (f == n_numeric) .categorical else .numeric;
    const edges = try gpa.alloc([]f32, nf);
    const names = try gpa.alloc([]u8, nf);
    const means = try gpa.alloc([]f32, nf);
    const levels = try gpa.alloc([][]u8, nf);
    for (0..nf) |f| {
        const e = try gpa.alloc(f32, @as(usize, nb[f]) - 2);
        for (e, 0..) |*v, i| v.* = @floatFromInt(i + 1);
        edges[f] = e;
        names[f] = try std.fmt.allocPrint(gpa, "f{d}", .{f});
        means[f] = &.{};
        levels[f] = &.{};
    }
    return .{
        .gpa = gpa,
        .n_rows = n_rows,
        .n_features = nf,
        .bins = bins,
        .bins_rm = bins_rm,
        .n_bins = nb,
        .edges = edges,
        .means = means,
        .kinds = kinds,
        .names = names,
        .levels = levels,
        .labels = labels,
    };
}

/// Half A: whole trees, real builder, varying declared width.
fn halfA(gpa: std.mem.Allocator, pool: *Pool, n_rows: usize, n_trees: usize) !void {
    const widths = [_]u16{ 51, 255, 753, 1024, 3114 };
    var base_ord: u64 = 0;
    var base_opt: u64 = 0;
    std.debug.print("\n  rows={d}, {d} numeric columns + 1 categorical, {d} trees, depth 6\n", .{ n_rows, n_numeric, n_trees });
    std.debug.print("  {s:>6}  {s:>10}  {s:>8}  {s:>10}  {s:>8}  {s:>9}\n", .{ "levels", "ordinal", "vs 51", "optimal", "vs 51", "slot KiB" });
    for (widths) |w| {
        var ds = try makeDs(gpa, n_rows, w, 99);
        defer ds.deinit();
        const grads = try gpa.alloc(hist.GradPair, n_rows);
        defer gpa.free(grads);
        for (grads, 0..) |*g, i| g.* = .{ .g = ds.labels[i] * 0.01, .h = 1 };

        var slot_kib: f64 = 0;
        var times: [2]u64 = undefined;
        for ([2]config.CatSplit{ .ordinal, .optimal }, 0..) |mode, mi| {
            var b = try tree.Builder.init(gpa, pool, &ds, .{
                .max_depth = 6,
                .cat_split = mode,
                .verbose_eval = 0,
            });
            defer b.deinit();
            slot_kib = @as(f64, @floatFromInt(b.bank.slotLen() * @sizeOf(hist.Bin))) / 1024.0;
            const t0 = now();
            for (0..n_trees) |_| {
                var t = try b.grow(grads);
                t.deinit(gpa);
            }
            times[mi] = now() - t0;
        }
        if (w == 51) {
            base_ord = times[0];
            base_opt = times[1];
        }
        const ms0 = @as(f64, @floatFromInt(times[0])) / 1e6;
        const ms1 = @as(f64, @floatFromInt(times[1])) / 1e6;
        std.debug.print("  {d:>6}  {d:>9.1}ms  {d:>7.2}x  {d:>9.1}ms  {d:>7.2}x  {d:>9.1}\n", .{
            w,
            ms0,
            ms0 / (@as(f64, @floatFromInt(base_ord)) / 1e6),
            ms1,
            ms1 / (@as(f64, @floatFromInt(base_opt)) / 1e6),
            slot_kib,
        });
    }
}

/// Half B: the scatter alone, over bins that really do span the full width.
fn halfB(gpa: std.mem.Allocator, n_rows: usize) !void {
    const widths = [_]u32{ 51, 255, 753, 1024, 3114, 16384 };
    const reps: usize = 40;
    std.debug.print("\n  scatter only: one column, {d} rows x {d} reps\n", .{ n_rows, reps });
    std.debug.print("  {s:>6}  {s:>12}  {s:>8}  {s:>10}\n", .{ "levels", "ns/row", "vs 51", "hist KiB" });
    var base: f64 = 0;
    for (widths) |w| {
        const bins = try gpa.alloc(u16, n_rows);
        defer gpa.free(bins);
        var prng: std.Random.DefaultPrng = .init(7);
        const r = prng.random();
        for (bins) |*b| b.* = @intCast(r.uintLessThan(u32, w));
        const h = try gpa.alloc(hist.Bin, w);
        defer gpa.free(h);
        const g = try gpa.alloc(hist.GradPair, n_rows);
        defer gpa.free(g);
        for (g) |*v| v.* = .{ .g = 0.5, .h = 1 };

        const t0 = now();
        for (0..reps) |_| {
            @memset(h, .{});
            for (bins, g) |b, gp| {
                const cell: *hist.Bin.Vec = @ptrCast(&h[b]);
                cell.* += hist.Bin.Vec{ gp.g, gp.h, 1, 0 };
            }
        }
        const ns = @as(f64, @floatFromInt(now() - t0)) / @as(f64, @floatFromInt(n_rows * reps));
        if (w == 51) base = ns;
        std.debug.print("  {d:>6}  {d:>11.3}   {d:>7.2}x  {d:>9.1}\n", .{
            w, ns, ns / base, @as(f64, @floatFromInt(w * @sizeOf(hist.Bin))) / 1024.0,
        });
    }
}

/// Half C: does a `u16` bin index cost the accumulation kernel anything?
///
/// This decides the architecture. If widening is near-free the bin type can
/// just become `u16` everywhere, one code path. If it is not, the type has to
/// become a comptime parameter with both widths instantiated, so a table that
/// fits in `u8` keeps today's cost -- a far larger refactor.
///
/// Row-major over `n_feat` features, which is the real kernel's shape: the
/// difference is entirely how many bytes of `bins_rm` a row occupies.
fn strideIdx(st: usize) usize {
    return switch (st) {
        1 => 0,
        4 => 1,
        else => 2,
    };
}

/// Half C: does a `u16` bin index cost the accumulation kernel anything?
///
/// This decides the architecture, and the first version of it got the answer
/// wrong. It scanned every row in order, which is what the *root* node does --
/// there one cache line of `bins_rm` serves several rows whatever the index
/// width, and widening looked free. Every node below the root holds an
/// ascending subset, and once the rows thin out each one wants its own line;
/// then doubling the row stride doubles the lines touched. `stride` here is
/// how sparse the node is: 1 is the root, 16 is roughly depth four.
fn halfC(gpa: std.mem.Allocator, n_rows: usize) !void {
    const reps: usize = 30;
    const n_feat: usize = 13;
    const bins_per: u32 = 64;
    std.debug.print("\n  accumulate, row-major, {d} features x {d} rows x {d} reps\n", .{ n_feat, n_rows, reps });
    std.debug.print("  {s:>9}  {s:>10}  {s:>12}  {s:>8}\n", .{ "bin type", "node holds", "ns/row", "vs u8" });
    var base_by: [3]f64 = .{ 1, 1, 1 };
    inline for ([2]type{ u8, u16 }) |B| {
        const rm = try gpa.alloc(B, n_rows * n_feat);
        defer gpa.free(rm);
        var prng: std.Random.DefaultPrng = .init(3);
        const r = prng.random();
        for (rm) |*b| b.* = @intCast(r.uintLessThan(u32, bins_per));
        const g = try gpa.alloc(hist.GradPair, n_rows);
        defer gpa.free(g);
        for (g) |*v| v.* = .{ .g = 0.5, .h = 1 };
        const rows = try gpa.alloc(u32, n_rows);
        defer gpa.free(rows);
        const h = try gpa.alloc(hist.Bin, n_feat * bins_per);
        defer gpa.free(h);

        for ([3]usize{ 1, 4, 16 }) |stride| {
            var n_sel: usize = 0;
            var i: usize = 0;
            while (i < n_rows) : (i += stride) {
                rows[n_sel] = @intCast(i);
                n_sel += 1;
            }
            const sel = rows[0..n_sel];
            const t0 = now();
            for (0..reps) |_| {
                @memset(h, .{});
                for (sel) |row| {
                    const rb = rm[row * n_feat ..][0..n_feat];
                    const gp = g[row];
                    const v = hist.Bin.Vec{ gp.g, gp.h, 1, 0 };
                    inline for (0..n_feat) |f| {
                        const cell: *hist.Bin.Vec = @ptrCast(&h[f * bins_per + rb[f]]);
                        cell.* += v;
                    }
                }
            }
            const ns = @as(f64, @floatFromInt(now() - t0)) / @as(f64, @floatFromInt(n_sel * reps));
            if (B == u8) base_by[strideIdx(stride)] = ns;
            std.debug.print("  {s:>9}  1 row in {d:>2}  {d:>11.3}   {d:>7.2}x\n", .{
                @typeName(B), stride, ns, ns / base_by[strideIdx(stride)],
            });
        }
    }
}

pub fn main() !void {
    var gpa_state: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();
    const pool = try Pool.init(gpa, 16);
    defer pool.deinit();

    std.debug.print("=== A. whole trees, real builder (scatter understated: rows crowd into bins 0..255)", .{});
    try halfA(gpa, pool, 6_830, 200); // the housing frame
    try halfA(gpa, pool, 500_000, 20);
    std.debug.print("\n=== B. scatter alone, bins spanning the full width", .{});
    try halfB(gpa, 500_000);
    std.debug.print("\n=== C. u8 vs u16 bin index in the accumulation kernel", .{});
    try halfC(gpa, 500_000);
    try halfC(gpa, 6_830);
    std.debug.print("\n", .{});
}
