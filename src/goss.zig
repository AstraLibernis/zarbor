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

/// `|g|` as a sortable integer.
///
/// For a non-negative float the IEEE-754 bit pattern is monotonic in the
/// value, so the pattern can be bucketed directly with no conversion. NaN
/// sorts above everything, which is where a blown-up gradient belongs.
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

/// Find the cut with two counting passes over the magnitude bit patterns.
///
/// This replaced a quickselect that was 768 ms of a 1.57 s GOSS fit — 49% of
/// the run, on a step that is not part of the model. The problem was not its
/// O(n): Hoare's two scans branch on a comparison that is a coin flip by
/// construction, so nearly every element cost a misprediction. Counting
/// passes branch predictably and stream the input, and sixteen bits at a time
/// means two of them pin the threshold exactly.
///
/// Exact, not approximate: afterwards every chosen row's magnitude is >= every
/// unchosen row's, with exact bit-ties broken by row order.
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

/// LightGBM's Gradient-based One-Side Sampling.
///
/// Rows with large |gradient| are the under-fitted ones and are kept in full;
/// the well-fitted remainder is sampled. Dropping most small-gradient rows
/// would bias the split gains, so the survivors are amplified by
/// `(1 - top_rate) / other_rate` — the reciprocal of their sampling rate —
/// which restores the expected gradient sum.
///
/// Mutates `grads` in place; the caller recomputes it every round anyway.
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

    // One ordered pass: mark the large-gradient rows and collect the rest, so
    // the random sample below has something contiguous to draw from.
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

    // Emit in ascending row order, from the bitset rather than by sorting.
    // A scrambled row set is the worst case for the histogram kernel, which
    // reads the row-major bin matrix and wants a forward walk; sorting 160k
    // ids would cost more than the selection does.
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
