// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Partitioning a node's rows between its children, in parallel.

const std = @import("std");
const data = @import("data.zig");
const prof = @import("prof.zig");
const split = @import("split.zig");
const tree = @import("tree.zig");
const builder = @import("builder.zig");
const Builder = builder.Builder;

/// Below this many rows the barriers cost more than the scan saves.
///
/// Tuned rather than guessed: sweeping it, 32768 (the original guess)
/// spends 360 ms in partition on a 200-tree fit where 2048 spends 271 ms,
/// and the curve rises monotonically above that. Output is identical at
/// every setting.
const parallel_partition_min: usize = 2048;

/// Takes the four scalars the decision needs, not the whole `Split`.
///
/// `Split` is 160 bytes once a categorical split carries its id array
/// inline, and this runs once per row per partition pass. Reading the
/// decision through it put partition at 2.2x; the element width of the
/// bin, which is what two earlier guesses blamed, was worth nothing.
pub inline fn goesLeft(bin: data.BinIdx, t: split.SplitTest) bool {
    if (bin == 0) return t.missing_left;
    if (t.is_cat) return split.catContains(t.ids, bin);
    return bin <= t.threshold;
}

/// Reorder `rows[start..end]` so the left child's rows come first.
/// Only row ids move; gradients are looked up by row id.
///
/// This is per-level O(rows) work and used to be the serial half of tree
/// building: with it single-threaded, 16 threads bought only 1.66x over 1,
/// and time scaled with tree *depth* rather than node count. It is now a
/// count / prefix-sum / scatter, which is three parallel passes instead of
/// one serial one.
pub fn partition(b: *Builder, start: usize, end: usize, sp: split.Split) usize {
    if (b.ds.isWide(sp.feature))
        return partitionOn(b, data.BinIdx, b.ds.columnWide(sp.feature), start, end, sp);
    return partitionOn(b, u8, b.ds.columnNarrow(sp.feature), start, end, sp);
}

pub fn partitionOn(
    b: *Builder,
    comptime C: type,
    col: []const C,
    start: usize,
    end: usize,
    sp: split.Split,
) usize {
    const n = end - start;
    if (n < parallel_partition_min or b.pool.workerCount() == 1)
        return partitionSerialOn(b, C, col, start, end, sp);

    const n_chunks = b.pool.workerCount() * 4;
    const csize = (n + n_chunks - 1) / n_chunks;
    const used = (n + csize - 1) / csize;
    std.debug.assert(used <= b.part_counts.len);

    var ctx = PartCtx(C){
        .rows = b.rows,
        .rows_out = b.rows_out,
        .col = col,
        .sp = split.SplitTest.of(&sp),
        .start = start,
        .chunk = csize,
        .counts = b.part_counts[0..used],
        .left = b.part_left[0..used],
        .right = b.part_right[0..used],
    };

    // 1. How many of each chunk's rows go left.
    const t_c = prof.start();
    b.pool.parallelFor(n, &ctx, PartCtx(C).count, csize);
    prof.stop(.part_count, t_c);

    // 2. Prefix sums. `used` is a few dozen at most, so serial is right.
    var total_left: usize = 0;
    for (ctx.counts) |c| total_left += c;
    var l: usize = start;
    var r: usize = start + total_left;
    for (ctx.counts, 0..) |c, i| {
        ctx.left[i] = l;
        ctx.right[i] = r;
        l += c;
        r += (@min((i + 1) * csize, n) - i * csize) - c;
    }

    // 3. Scatter to the scratch buffers, then copy the touched range back.
    const t_s = prof.start();
    b.pool.parallelFor(n, &ctx, PartCtx(C).scatter, csize);
    prof.stop(.part_scatter, t_s);
    const t_b = prof.start();
    b.pool.parallelFor(n, &ctx, PartCtx(C).copyBack, 8192);
    prof.stop(.part_copy, t_b);

    return start + total_left;
}

/// The one-thread path, and stable like the parallel one.
///
/// It used to be an in-place Hoare swap, which is fewer passes but leaves
/// a node's rows in arbitrary order. That matters now: the histogram
/// kernel reads the row-major bin matrix, and an ascending row order is
/// what lets the hardware prefetcher follow it. Two branchless passes and
/// a memcpy beat one pass of mispredicted swaps anyway.
pub fn partitionSerialOn(
    b: *Builder,
    comptime C: type,
    col: []const C,
    start: usize,
    end: usize,
    sp: split.Split,
) usize {
    const rows = b.rows[start..end];
    const t = split.SplitTest.of(&sp);
    var n_left: usize = 0;
    for (rows) |row| n_left += @intFromBool(goesLeft(col[row], t));

    var li = start;
    var ri = start + n_left;
    for (rows) |row| {
        const left = goesLeft(col[row], t);
        const dst = if (left) li else ri;
        b.rows_out[dst] = row;
        li += @intFromBool(left);
        ri += @intFromBool(!left);
    }
    @memcpy(rows, b.rows_out[start..end]);
    return start + n_left;
}

/// Generic over the column's element type so a table that fits in a byte
/// keeps reading bytes here.
///
/// This is the one place the bin width is visibly paid. Partitioning walks a
/// single feature down a node's rows, and those rows are scattered once the
/// tree is more than a level deep, so each tends to want its own cache line
/// and the element size is paid in full rather than amortised. Measured when
/// the whole matrix went to `u16`: 51 ms -> 110 ms on adult, while the
/// row-major accumulate did not move. The row-major mirror is therefore
/// uniformly `u16` and only this stays narrow.
fn PartCtx(comptime C: type) type {
    return struct {
        const Self = @This();

        rows: []u32,
        rows_out: []u32,
        col: []const C,
        sp: split.SplitTest,
        start: usize,
        chunk: usize,
        counts: []usize,
        left: []usize,
        right: []usize,

        fn count(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
            _ = worker;
            const self: *Self = @ptrCast(@alignCast(ctx));
            var n: usize = 0;
            for (self.rows[self.start + begin .. self.start + end]) |row| {
                if (goesLeft(self.col[row], self.sp)) n += 1;
            }
            self.counts[begin / self.chunk] = n;
        }

        fn scatter(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
            _ = worker;
            const self: *Self = @ptrCast(@alignCast(ctx));
            const c = begin / self.chunk;
            var li = self.left[c];
            var ri = self.right[c];
            var i = self.start + begin;
            while (i < self.start + end) : (i += 1) {
                const row = self.rows[i];
                // Branchless: which side a row takes is close to a coin flip near
                // a good split, so a branch here mispredicts about half the time.
                // Selecting the cursor instead costs a cmov and nothing else.
                const left = goesLeft(self.col[row], self.sp);
                const dst = if (left) li else ri;
                self.rows_out[dst] = row;
                li += @intFromBool(left);
                ri += @intFromBool(!left);
            }
        }

        fn copyBack(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
            _ = worker;
            const self: *Self = @ptrCast(@alignCast(ctx));
            const a = self.start + begin;
            const b_ = self.start + end;
            @memcpy(self.rows[a..b_], self.rows_out[a..b_]);
        }
    };
}
