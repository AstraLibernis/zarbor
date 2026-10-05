// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! GOSS (gradient-based one-side sampling): which rows a boosting round keeps.

const std = @import("std");
const hist = @import("hist.zig");
const booster = @import("booster.zig");
const Pool = @import("pool.zig").Pool;
const GossRank = booster.GossRank;

/// Buckets in one radix pass over a magnitude's bit pattern.
pub const radix_bits = 16;
pub const n_radix = 1 << radix_bits;

/// `|g|` as a sortable integer: non-negative IEEE-754 bits are monotonic in
/// value. NaN sorts above all, where a blown-up gradient belongs.
inline fn magBits(g: f32) u32 {
    return @bitCast(@abs(g));
}

/// The value GOSS ranks a row by. `|g|` is the paper's; `|g*h|` is
/// LightGBM's. See `GossRank`.
inline fn rankKey(p: hist.GradPair, how: GossRank) u32 {
    return switch (how) {
        .gradient => magBits(p.g),
        .gradient_hessian => magBits(p.g * p.h),
    };
}

/// The cut separating the `k` largest magnitudes from the rest.
pub const Cut = struct {
    /// Rows whose pattern is strictly greater are all in.
    t: u32,
    /// How many rows with pattern exactly `t` are in, taken in row order.
    take: usize,
};

/// Two 16-bit counting passes over the bit patterns. Quickselect was 768 ms
/// of a 1.57 s GOSS fit (49%): Hoare's scans branch on a coin flip, so nearly
/// every element mispredicted; counting branches predictably and streams.
/// Exact: every chosen magnitude >= every unchosen one, bit-ties by row order.
pub fn gossCut(grads: []const hist.GradPair, counts: []u32, k: usize, how: GossRank) Cut {
    std.debug.assert(counts.len == n_radix);
    std.debug.assert(k >= 1 and k <= grads.len);

    @memset(counts, 0);
    for (grads) |g| counts[rankKey(g, how) >> radix_bits] += 1;

    var above: usize = 0;
    var hi: u32 = n_radix - 1;
    while (true) {
        const c = counts[hi];
        if (above + c >= k or hi == 0) break;
        above += c;
        hi -= 1;
    }

    @memset(counts, 0);
    const lo_mask: u32 = n_radix - 1;
    for (grads) |g| {
        const b = rankKey(g, how);
        if (b >> radix_bits == hi) counts[b & lo_mask] += 1;
    }

    var lo: u32 = n_radix - 1;
    while (true) {
        const c = counts[lo];
        if (above + c >= k or lo == 0) break;
        above += c;
        lo -= 1;
    }

    return .{ .t = (hi << radix_bits) | lo, .take = k - above };
}

/// Per-chunk counters for `gossSelect`'s parallel passes: 4 x `scratchChunks(pool)` entries.
pub fn scratchLen(pool: *const Pool) usize {
    return 4 * scratchChunks(pool);
}

/// At most as many chunks as `parallelFor` would make by itself, so the chunk size we ask for is
/// the one it uses, and `begin / chunk` names the chunk.
fn scratchChunks(pool: *const Pool) usize {
    return pool.workerCount() * 4;
}

/// LightGBM's Gradient-based One-Side Sampling. Large-|gradient| rows are
/// kept; the rest are sampled and amplified by `(1 - top_rate) / other_rate`
/// (inverse sampling rate) so split gains stay unbiased. Mutates `grads` in
/// place; the caller recomputes it every round.
///
/// Marking and the final bitset walk run on the pool: they were 65% of a serial 1.5 ms per round,
/// 35% of a GOSS fit. Every chunk is whole mask words, so each owns its words outright; integer
/// prefix counts give each chunk its first kept tie and its first slot in `others`, so the output
/// is the serial one exactly, for any thread count. The shuffle stays serial: it draws the RNG.
pub fn gossSelect(
    pool: *Pool,
    grads: []hist.GradPair,
    counts: []u32,
    chunk_scratch: []usize,
    others: []u32,
    mask: []u64,
    out: []u32,
    top_rate: f32,
    other_rate: f32,
    how: GossRank,
    rng: std.Random,
) []u32 {
    const n = grads.len;
    var top: usize = @intFromFloat(@round(@as(f32, @floatFromInt(n)) * top_rate));
    top = std.math.clamp(top, 1, n);
    const rest = n - top;

    const cut = gossCut(grads, counts, top, how);

    const max_chunks = scratchChunks(pool);
    std.debug.assert(chunk_scratch.len >= 4 * max_chunks);
    const csize = std.mem.alignForward(usize, @max((n + max_chunks - 1) / max_chunks, min_chunk), 64);
    const used = (n + csize - 1) / csize;
    var ctx = MarkCtx{
        .grads = grads,
        .how = how,
        .t = cut.t,
        .take = cut.take,
        .chunk = csize,
        .below = chunk_scratch[0..used],
        .ties = chunk_scratch[max_chunks..][0..used],
        .ties_before = chunk_scratch[2 * max_chunks ..][0..used],
        .other_at = chunk_scratch[3 * max_chunks ..][0..used],
        .others = others,
        .mask = mask,
        .out = out,
    };

    // Mark large-gradient rows; collect the rest contiguously, in row order, for sampling.
    pool.parallelFor(n, &ctx, MarkCtx.count, csize);
    var ties_seen: usize = 0;
    var n_other: usize = 0;
    for (ctx.below, ctx.ties, ctx.ties_before, ctx.other_at) |below, ties, *tb, *oa| {
        tb.* = ties_seen;
        oa.* = n_other;
        const kept_ties = @min(ties, cut.take -| ties_seen);
        n_other += below + ties - kept_ties;
        ties_seen += ties;
    }
    pool.parallelFor(n, &ctx, MarkCtx.mark, csize);

    var rand_n: usize = @intFromFloat(@round(@as(f32, @floatFromInt(n)) * other_rate));
    rand_n = @min(rand_n, @min(rest, n_other));

    // Partial Fisher-Yates over the unchosen rows.
    var i: usize = 0;
    while (i < rand_n) : (i += 1) {
        const j = i + rng.uintLessThan(usize, n_other - i);
        std.mem.swap(u32, &others[i], &others[j]);
    }

    const amp: f32 = (1.0 - top_rate) / other_rate;
    for (others[0..rand_n]) |r| {
        grads[r].g *= amp;
        grads[r].h *= amp;
        mask[r >> 6] |= @as(u64, 1) << @truncate(r);
    }

    // Ascending row order via the bitset: the histogram kernel wants a
    // forward walk, and sorting 160k ids would cost more than selection.
    // Per-chunk popcounts (reusing `below`) place each chunk's rows in `out`.
    pool.parallelFor(n, &ctx, MarkCtx.popcount, csize);
    var k: usize = 0;
    for (ctx.below, ctx.other_at) |pop, *at| {
        at.* = k;
        k += pop;
    }
    pool.parallelFor(n, &ctx, MarkCtx.emit, csize);
    return out[0..k];
}

/// Below this many rows a chunk is not worth a barrier.
const min_chunk = 16384;

/// Each pass is written per chunk, and `run` splits whatever range the pool hands it into chunks:
/// one worker (or a range under the pool's `min_chunk`) gets the whole range in one call, and the
/// per-chunk counters must still be filled for every chunk.
const MarkCtx = struct {
    grads: []const hist.GradPair,
    how: GossRank,
    t: u32,
    take: usize,
    chunk: usize,
    below: []usize,
    ties: []usize,
    ties_before: []usize,
    other_at: []usize,
    others: []u32,
    mask: []u64,
    out: []u32,

    fn run(comptime pass: fn (*MarkCtx, usize, usize, usize) void) fn (*anyopaque, usize, usize, usize) void {
        return struct {
            fn task(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
                _ = worker;
                const self: *MarkCtx = @ptrCast(@alignCast(ctx));
                var lo = begin;
                while (lo < end) {
                    const hi = @min(lo + self.chunk, end);
                    pass(self, lo / self.chunk, lo, hi);
                    lo = hi;
                }
            }
        }.task;
    }

    const count = run(countChunk);
    const mark = run(markChunk);
    const popcount = run(popcountChunk);
    const emit = run(emitChunk);

    fn countChunk(self: *MarkCtx, c: usize, begin: usize, end: usize) void {
        const how = self.how;
        const t = self.t;
        var below: usize = 0;
        var ties: usize = 0;
        for (self.grads[begin..end]) |g| {
            const b = rankKey(g, how);
            below += @intFromBool(b < t);
            ties += @intFromBool(b == t);
        }
        self.below[c] = below;
        self.ties[c] = ties;
    }

    /// One mask word at a time, built in a register: `mask[w] |=` per row was a read-modify-write
    /// chain through memory.
    fn markChunk(self: *MarkCtx, c: usize, begin: usize, end: usize) void {
        const how = self.how;
        const t = self.t;
        const grads = self.grads;
        const others = self.others;
        var ties_left = self.take -| self.ties_before[c];
        var o = self.other_at[c];
        var r = begin;
        while (r < end) : (r += 64) {
            const stop = @min(r + 64, end);
            var w: u64 = 0;
            for (r..stop) |row| {
                const b = rankKey(grads[row], how);
                const tie = b == t;
                const keep = b > t or (tie and ties_left > 0);
                ties_left -= @intFromBool(tie and keep);
                w |= @as(u64, @intFromBool(keep)) << @truncate(row);
                // A branch, not an unconditional store: a chunk ending in kept rows would write
                // one slot past its region, into the next chunk's first.
                if (!keep) {
                    others[o] = @intCast(row);
                    o += 1;
                }
            }
            self.mask[r >> 6] = w;
        }
    }

    fn popcountChunk(self: *MarkCtx, c: usize, begin: usize, end: usize) void {
        var pop: usize = 0;
        for (self.mask[begin >> 6 .. (end + 63) >> 6]) |w| pop += @popCount(w);
        self.below[c] = pop;
    }

    fn emitChunk(self: *MarkCtx, c: usize, begin: usize, end: usize) void {
        var k = self.other_at[c];
        const first = begin >> 6;
        for (self.mask[first .. (end + 63) >> 6], first..) |w0, wi| {
            var w = w0;
            while (w != 0) {
                self.out[k] = @intCast(wi * 64 + @ctz(w));
                k += 1;
                w &= w - 1;
            }
        }
    }
};
