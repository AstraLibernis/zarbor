// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Evaluation metrics.

const std = @import("std");

/// ROC AUC by rank identity: `(sum of positive ranks - n_pos(n_pos+1)/2) / (n_pos * n_neg)`.
/// Ties share their average rank: rows sharing a leaf get equal scores, and
/// arbitrary tie order would bias the result.
///
/// Sorted by LSD radix on `sortKey(score) << 32 | label`, not a comparison sort through an index:
/// 9x faster on 133k rows (10.3 -> 1.2 ms), and it runs every `verbose_eval` round, every round
/// under early stopping. The value is unchanged to the bit: a tied block is still every row of
/// one score, and the rank sum adds half-integers far below 2^53, exact in any order.
/// (NaN scores, which the comparison sort left in undefined places, now group by bit pattern.)
pub fn auc(gpa: std.mem.Allocator, scores: []const f32, labels: []const f32) !f64 {
    std.debug.assert(scores.len == labels.len);
    const n = scores.len;
    if (n == 0) return 0.5;

    const buf = try gpa.alloc(u64, 2 * n);
    defer gpa.free(buf);
    var src = buf[0..n];
    var dst = buf[n..];
    for (src, scores, labels) |*v, s, y| v.* = @as(u64, sortKey(s)) << 32 | @intFromBool(y > 0.5);

    var shift: u6 = 32;
    while (true) : (shift += 8) {
        var count = [_]usize{0} ** 256;
        for (src) |v| count[@as(u8, @truncate(v >> shift))] += 1;
        // A byte every key shares cannot reorder anything.
        if (count[@as(u8, @truncate(src[0] >> shift))] != n) {
            var at: usize = 0;
            for (&count) |*c| {
                const k = c.*;
                c.* = at;
                at += k;
            }
            for (src) |v| {
                const b: u8 = @truncate(v >> shift);
                dst[count[b]] = v;
                count[b] += 1;
            }
            std.mem.swap([]u64, &src, &dst);
        }
        if (shift == 56) break;
    }

    var pos: f64 = 0;
    var neg: f64 = 0;
    var rank_sum: f64 = 0;

    var i: usize = 0;
    while (i < n) {
        const key = src[i] >> 32;
        var j = i;
        var p: usize = 0;
        while (j < n and src[j] >> 32 == key) : (j += 1) p += @intFromBool(src[j] & 1 == 1);
        // Ranks are 1-based; the block is [i, j) and each row takes its average rank.
        const avg_rank = (@as(f64, @floatFromInt(i + 1)) + @as(f64, @floatFromInt(j))) / 2.0;
        const fp: f64 = @floatFromInt(p);
        pos += fp;
        neg += @floatFromInt(j - i - p);
        rank_sum += avg_rank * fp; // exact: a half-integer times a count, far below 2^53
        i = j;
    }

    if (pos == 0 or neg == 0) return 0.5;
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg);
}

/// An f32 as a u32 whose unsigned order is the float order. -0 and +0 share a key, as they compare
/// equal under `==`, so they still fall into one tied block.
fn sortKey(s: f32) u32 {
    const bits: u32 = @bitCast(if (s == 0) @as(f32, 0) else s);
    return if (bits >> 31 == 1) ~bits else bits | 0x8000_0000;
}

/// Mean binary cross-entropy from raw log-odds.
pub fn logloss(raw: []const f32, labels: []const f32) f64 {
    var acc: f64 = 0;
    for (raw, labels) |r, y| {
        const z: f64 = r;
        // log(1+exp(z)) evaluated so neither tail overflows.
        const softplus = if (z > 0) z + @log(1 + @exp(-z)) else @log(1 + @exp(z));
        acc += softplus - @as(f64, y) * z;
    }
    return acc / @as(f64, @floatFromInt(raw.len));
}

/// Mean binary cross-entropy from probabilities. Clamped away from 0 and 1:
/// a pure forest leaf holds exactly 0 or 1, which would make the loss infinite.
pub fn loglossProb(prob: []const f32, labels: []const f32) f64 {
    var acc: f64 = 0;
    for (prob, labels) |pr, y| {
        const q = std.math.clamp(@as(f64, pr), 1e-7, 1 - 1e-7);
        acc += -(@as(f64, y) * @log(q) + (1 - @as(f64, y)) * @log(1 - q));
    }
    return acc / @as(f64, @floatFromInt(prob.len));
}

pub fn rmse(pred: []const f32, labels: []const f32) f64 {
    var acc: f64 = 0;
    for (pred, labels) |p, y| {
        const d: f64 = @as(f64, p) - @as(f64, y);
        acc += d * d;
    }
    return @sqrt(acc / @as(f64, @floatFromInt(pred.len)));
}
