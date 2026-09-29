// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Evaluation metrics.

const std = @import("std");

/// ROC AUC by rank identity: `(sum of positive ranks - n_pos(n_pos+1)/2) / (n_pos * n_neg)`.
/// Ties share their average rank: rows sharing a leaf get equal scores, and
/// arbitrary tie order would bias the result.
pub fn auc(gpa: std.mem.Allocator, scores: []const f32, labels: []const f32) !f64 {
    std.debug.assert(scores.len == labels.len);
    const n = scores.len;
    if (n == 0) return 0.5;

    const idx = try gpa.alloc(u32, n);
    defer gpa.free(idx);
    for (idx, 0..) |*v, i| v.* = @intCast(i);

    const Ctx = struct {
        s: []const f32,
        fn lessThan(c: @This(), a: u32, b: u32) bool {
            return c.s[a] < c.s[b];
        }
    };
    std.sort.pdq(u32, idx, Ctx{ .s = scores }, Ctx.lessThan);

    var pos: f64 = 0;
    var neg: f64 = 0;
    var rank_sum: f64 = 0;

    var i: usize = 0;
    while (i < n) {
        var j = i;
        while (j + 1 < n and scores[idx[j + 1]] == scores[idx[i]]) j += 1;
        // Ranks are 1-based; the average over the tied block is used for each.
        const avg_rank = (@as(f64, @floatFromInt(i + 1)) + @as(f64, @floatFromInt(j + 1))) / 2.0;
        var k = i;
        while (k <= j) : (k += 1) {
            if (labels[idx[k]] > 0.5) {
                pos += 1;
                rank_sum += avg_rank;
            } else {
                neg += 1;
            }
        }
        i = j + 1;
    }

    if (pos == 0 or neg == 0) return 0.5;
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg);
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
