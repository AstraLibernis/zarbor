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

/// Below this many rows the barriers cost more than the scan saves. Tuned: on a 200-tree fit 32768
/// (the original guess) spends 360 ms in partition, 2048 spends 271 ms; the curve rises
/// monotonically above that. Output identical at every setting.
const parallel_partition_min: usize = 2048;

/// Takes the four scalars the decision needs, not the whole 160-byte `Split` (inline categorical id
/// array): read per row per pass, it put partition at 2.2x. Bin element width, blamed twice, was
/// worth nothing.
pub inline fn goesLeft(bin: data.BinIdx, t: split.SplitTest) bool {
    if (bin == 0) return t.missing_left;
    if (t.is_cat) return split.catContains(t.ids, bin);
    return bin <= t.threshold;
}

/// A numeric split as one unsigned compare: `(bin - lo) <= hi` in wrapping u32 arithmetic, with
/// `lo = 1` when missing goes right (bin 0 wraps to the top and fails) and `lo = 0` when it goes
/// left (bin 0 passes). `goesLeft` compiled to three conditional jumps per row; near a good split
/// the side is a coin flip, so they mispredicted constantly. Same decision for every bin: bin 0 is
/// missing and never a threshold, so `threshold >= 1` and `hi` never wraps.
const NumTest = struct {
    lo: u32,
    hi: u32,

    fn of(t: split.SplitTest) NumTest {
        std.debug.assert(!t.is_cat and t.threshold != 0);
        const lo: u32 = @intFromBool(!t.missing_left);
        return .{ .lo = lo, .hi = @as(u32, t.threshold) - lo };
    }

    inline fn left(self: NumTest, bin: data.BinIdx) bool {
        return @as(u32, bin) -% self.lo <= self.hi;
    }
};

const CatTest = struct {
    t: split.SplitTest,

    inline fn left(self: CatTest, bin: data.BinIdx) bool {
        return goesLeft(bin, self.t);
    }
};

/// Reorder `rows[start..end]` so the left child's rows come first; only row ids move (gradients are
/// looked up by id). Count / prefix-sum / scatter in three parallel passes: done serially, this
/// per-level O(rows) work capped 16 threads at 1.66x over 1 and made time scale with depth, not
/// node count.
pub fn partition(b: *Builder, start: usize, end: usize, sp: split.Split) usize {
    if (b.ds.isWide(sp.feature))
        return partitionOn(b, data.BinIdx, b.ds.columnWide(sp.feature), start, end, sp);
    return partitionOn(b, u8, b.ds.columnNarrow(sp.feature), start, end, sp);
}

/// `partition` on the calling thread, for a node small enough that `partition` would run it serially
/// anyway (`parallel_partition_min`); safe from pool tasks on disjoint row ranges, which is how the
/// builder expands a batch of small nodes at once.
pub fn partitionSmall(b: *Builder, start: usize, end: usize, sp: split.Split) usize {
    std.debug.assert(end - start < parallel_partition_min);
    const t = split.SplitTest.of(&sp);
    if (b.ds.isWide(sp.feature)) return serialFor(b, data.BinIdx, b.ds.columnWide(sp.feature), start, end, t);
    return serialFor(b, u8, b.ds.columnNarrow(sp.feature), start, end, t);
}

fn serialFor(b: *Builder, comptime C: type, col: []const C, start: usize, end: usize, t: split.SplitTest) usize {
    if (t.is_cat) return partitionSerial(b, C, CatTest, col, start, end, .{ .t = t });
    return partitionSerial(b, C, NumTest, col, start, end, NumTest.of(t));
}

/// Picks the test once per partition, so the row loops are specialised at compile time.
fn partitionOn(b: *Builder, comptime C: type, col: []const C, start: usize, end: usize, sp: split.Split) usize {
    const t = split.SplitTest.of(&sp);
    if (t.is_cat) return partitionWith(b, C, CatTest, col, start, end, .{ .t = t });
    return partitionWith(b, C, NumTest, col, start, end, NumTest.of(t));
}

fn partitionWith(
    b: *Builder,
    comptime C: type,
    comptime T: type,
    col: []const C,
    start: usize,
    end: usize,
    t: T,
) usize {
    const n = end - start;
    if (n < parallel_partition_min or b.pool.workerCount() == 1)
        return partitionSerial(b, C, T, col, start, end, t);

    const n_chunks = b.pool.workerCount() * 4;
    const csize = (n + n_chunks - 1) / n_chunks;
    const used = (n + csize - 1) / csize;
    std.debug.assert(used <= b.part_counts.len);

    var ctx = PartCtx(C, T){
        .rows = b.rows,
        .rows_out = b.rows_out,
        .col = col,
        .t = t,
        .start = start,
        .chunk = csize,
        .counts = b.part_counts[0..used],
        .left = b.part_left[0..used],
        .right = b.part_right[0..used],
    };

    // 1. How many of each chunk's rows go left.
    const t_c = prof.start();
    b.pool.parallelFor(n, &ctx, PartCtx(C, T).count, csize);
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
    b.pool.parallelFor(n, &ctx, PartCtx(C, T).scatter, csize);
    prof.stop(.part_scatter, t_s);
    const t_b = prof.start();
    b.pool.parallelFor(n, &ctx, PartCtx(C, T).copyBack, 8192);
    prof.stop(.part_copy, t_b);

    return start + total_left;
}

/// The one-thread path, stable like the parallel one. Not an in-place Hoare swap: that leaves rows
/// in arbitrary order, and the histogram kernel needs ascending rows for the hardware prefetcher
/// over the row-major matrix. Two branchless passes and a memcpy beat mispredicted swaps anyway.
fn partitionSerial(
    b: *Builder,
    comptime C: type,
    comptime T: type,
    col: []const C,
    start: usize,
    end: usize,
    t: T,
) usize {
    const rows = b.rows[start..end];
    const out = b.rows_out;
    var n_left: usize = 0;
    for (rows) |row| n_left += @intFromBool(t.left(col[row]));

    var li = start;
    var ri = start + n_left;
    for (rows) |row| {
        const left = t.left(col[row]);
        const dst = if (left) li else ri;
        out[dst] = row;
        li += @intFromBool(left);
        ri += @intFromBool(!left);
    }
    @memcpy(rows, out[start..end]);
    return start + n_left;
}

/// Generic over the column element type so byte-wide tables keep reading bytes. The one place bin
/// width is visibly paid: one feature walked down scattered rows costs a cache line per row, so the
/// element size is not amortised. Whole matrix as `u16`: 51 -> 110 ms on adult, row-major
/// accumulate unmoved. So the row-major mirror is uniformly `u16` and only this stays narrow.
/// `parallelFor` hands out fixed-size chunks, so `begin / chunk` recovers the chunk: count and
/// scatter agree on each chunk's output position with no coordination.
/// The loops copy what they read into locals first: through `self` the compiler reloaded every
/// field per row, since a store to `rows_out` might alias it.
fn PartCtx(comptime C: type, comptime T: type) type {
    return struct {
        const Self = @This();

        rows: []u32,
        rows_out: []u32,
        col: []const C,
        t: T,
        start: usize,
        chunk: usize,
        counts: []usize,
        left: []usize,
        right: []usize,

        fn count(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
            _ = worker;
            const self: *Self = @ptrCast(@alignCast(ctx));
            const col = self.col;
            const t = self.t;
            var n: usize = 0;
            for (self.rows[self.start + begin .. self.start + end]) |row| n += @intFromBool(t.left(col[row]));
            self.counts[begin / self.chunk] = n;
        }

        fn scatter(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
            _ = worker;
            const self: *Self = @ptrCast(@alignCast(ctx));
            const c = begin / self.chunk;
            const col = self.col;
            const t = self.t;
            const out = self.rows_out;
            var li = self.left[c];
            var ri = self.right[c];
            for (self.rows[self.start + begin .. self.start + end]) |row| {
                // Branchless: near a good split the side is a coin flip, so a branch mispredicts
                // about half the time; selecting the cursor costs one cmov.
                const left = t.left(col[row]);
                const dst = if (left) li else ri;
                out[dst] = row;
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
