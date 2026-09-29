// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! GOSS (gradient-based one-side sampling): which rows a boosting round keeps.

const std = @import("std");
const hist = @import("hist.zig");
const booster = @import("booster.zig");
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

/// LightGBM's Gradient-based One-Side Sampling. Large-|gradient| rows are
/// kept; the rest are sampled and amplified by `(1 - top_rate) / other_rate`
/// (inverse sampling rate) so split gains stay unbiased. Mutates `grads` in
/// place; the caller recomputes it every round.
pub fn gossSelect(
    grads: []hist.GradPair,
    counts: []u32,
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

    // Mark large-gradient rows; collect the rest contiguously for sampling.
    const words = (n + 63) / 64;
    @memset(mask[0..words], 0);
    var take = cut.take;
    var n_other: usize = 0;
    for (grads, 0..) |g, r| {
        const b = rankKey(g, how);
        const keep = b > cut.t or (b == cut.t and take > 0);
        if (keep) {
            if (b == cut.t) take -= 1;
            mask[r >> 6] |= @as(u64, 1) << @truncate(r);
        } else {
            others[n_other] = @intCast(r);
            n_other += 1;
        }
    }

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
    var k: usize = 0;
    for (mask[0..words], 0..) |w0, wi| {
        var w = w0;
        while (w != 0) {
            out[k] = @intCast(wi * 64 + @ctz(w));
            k += 1;
            w &= w - 1;
        }
    }
    return out[0..k];
}
