// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Evaluation metrics.

const std = @import("std");
const radix = @import("radix.zig");

/// ROC AUC by rank identity: `(sum of positive ranks - n_pos(n_pos+1)/2) / (n_pos * n_neg)`.
/// Ties share their average rank: rows sharing a leaf get equal scores, and
/// arbitrary tie order would bias the result.
///
/// Sorted by LSD radix (`radix.zig`) on `f32Key(score) << 32 | label`, not a comparison sort through an index:
/// much faster (docs/measurements.md, metric.zig `auc`), and it runs every `verbose_eval` round, every round
/// under early stopping. The value is unchanged to the bit: a tied block is still every row of
/// one score, and the rank sum adds half-integers far below 2^53, exact in any order.
/// (NaN scores, which the comparison sort left in undefined places, now group by bit pattern.)
pub fn auc(gpa: std.mem.Allocator, scores: []const f32, labels: []const f32) !f64 {
    std.debug.assert(scores.len == labels.len);
    const n = scores.len;
    if (n == 0) return 0.5;

    const buf = try gpa.alloc(u64, 2 * n);
    defer gpa.free(buf);
    for (buf[0..n], scores, labels) |*v, x, y| v.* = @as(u64, radix.f32Key(x)) << 32 | @intFromBool(y > 0.5);
    const src = radix.sortHigh32(buf[0..n], buf[n..]);

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

/// Mean multiclass cross-entropy from raw scores, `raw[r * k + c]` for row r and class c:
/// log-softmax per row in f64, shifted by the row's largest score so no exp overflows.
pub fn mlogloss(raw: []const f32, labels: []const f32, k: usize) f64 {
    const n = labels.len;
    std.debug.assert(raw.len == n * k);
    var acc: f64 = 0;
    for (labels, 0..) |y, r| {
        const row = raw[r * k ..][0..k];
        var top: f64 = row[0];
        for (row[1..]) |v| top = @max(top, @as(f64, v));
        var sum: f64 = 0;
        for (row) |v| sum += @exp(@as(f64, v) - top);
        const c: usize = @intFromFloat(y);
        acc += top + @log(sum) - @as(f64, row[c]);
    }
    return acc / @as(f64, @floatFromInt(n));
}

/// Mean multiclass cross-entropy from probabilities, each clamped up from 1e-15 (LightGBM's
/// `kEpsilon`) so a class given exactly 0 does not make the loss infinite.
pub fn mloglossProb(prob: []const f32, labels: []const f32, k: usize) f64 {
    const n = labels.len;
    std.debug.assert(prob.len == n * k);
    var acc: f64 = 0;
    for (labels, 0..) |y, r| {
        const c: usize = @intFromFloat(y);
        acc -= @log(@max(@as(f64, prob[r * k + c]), 1e-15));
    }
    return acc / @as(f64, @floatFromInt(n));
}

/// Fraction of rows whose highest score (raw or probability; the first on a tie) is their
/// class.
pub fn accuracy(scores: []const f32, labels: []const f32, k: usize) f64 {
    const n = labels.len;
    std.debug.assert(scores.len == n * k);
    var hit: usize = 0;
    for (labels, 0..) |y, r| {
        const row = scores[r * k ..][0..k];
        var best: usize = 0;
        for (row, 0..) |v, c| if (v > row[best]) {
            best = c;
        };
        hit += @intFromBool(@as(f32, @floatFromInt(best)) == y);
    }
    return @as(f64, @floatFromInt(hit)) / @as(f64, @floatFromInt(n));
}

// ------------------------------------------------------------ weighted

/// Weighted ROC AUC (scikit-learn's `sample_weight`, XGBoost's weighted `auc`): each positive's
/// weight times the negative weight scored below it, a tie counting half, over the product of
/// the two classes' total weights. Rows sort by score through their index, so each keeps its
/// weight; ties are blocks of one score.
pub fn aucW(gpa: std.mem.Allocator, scores: []const f32, labels: []const f32, weights: []const f32) !f64 {
    std.debug.assert(scores.len == labels.len and weights.len == labels.len);
    const n = scores.len;
    if (n == 0) return 0.5;
    const buf = try gpa.alloc(u64, 2 * n);
    defer gpa.free(buf);
    for (buf[0..n], scores, 0..) |*v, x, i| v.* = @as(u64, radix.f32Key(x)) << 32 | @as(u32, @intCast(i));
    const src = radix.sortHigh32(buf[0..n], buf[n..]);

    var below: f64 = 0; // negative weight scored strictly lower
    var pos: f64 = 0;
    var neg: f64 = 0;
    var area: f64 = 0;
    var i: usize = 0;
    while (i < n) {
        const key = src[i] >> 32;
        var j = i;
        var p: f64 = 0;
        var q: f64 = 0;
        while (j < n and src[j] >> 32 == key) : (j += 1) {
            const r: usize = @intCast(src[j] & 0xFFFF_FFFF);
            if (labels[r] > 0.5) p += weights[r] else q += weights[r];
        }
        area += p * (below + 0.5 * q);
        below += q;
        pos += p;
        neg += q;
        i = j;
    }
    if (pos == 0 or neg == 0) return 0.5;
    return area / (pos * neg);
}

fn weightSum(weights: []const f32) f64 {
    var s: f64 = 0;
    for (weights) |w| s += w;
    return s;
}

/// Weighted mean binary cross-entropy from raw log-odds.
pub fn loglossW(raw: []const f32, labels: []const f32, weights: []const f32) f64 {
    var acc: f64 = 0;
    for (raw, labels, weights) |r, y, w| {
        const z: f64 = r;
        const softplus = if (z > 0) z + @log(1 + @exp(-z)) else @log(1 + @exp(z));
        acc += w * (softplus - @as(f64, y) * z);
    }
    return acc / weightSum(weights);
}

/// Weighted mean binary cross-entropy from probabilities, clamped as `loglossProb`.
pub fn loglossProbW(prob: []const f32, labels: []const f32, weights: []const f32) f64 {
    var acc: f64 = 0;
    for (prob, labels, weights) |pr, y, w| {
        const q = std.math.clamp(@as(f64, pr), 1e-7, 1 - 1e-7);
        acc += w * -(@as(f64, y) * @log(q) + (1 - @as(f64, y)) * @log(1 - q));
    }
    return acc / weightSum(weights);
}

/// Weighted root mean squared error.
pub fn rmseW(pred: []const f32, labels: []const f32, weights: []const f32) f64 {
    var acc: f64 = 0;
    for (pred, labels, weights) |p, y, w| {
        const d: f64 = @as(f64, p) - @as(f64, y);
        acc += w * d * d;
    }
    return @sqrt(acc / weightSum(weights));
}

/// Weighted multiclass cross-entropy from raw scores (`mlogloss`'s layout).
pub fn mloglossW(raw: []const f32, labels: []const f32, k: usize, weights: []const f32) f64 {
    var acc: f64 = 0;
    for (labels, weights, 0..) |y, w, r| {
        const row = raw[r * k ..][0..k];
        var top: f64 = row[0];
        for (row[1..]) |v| top = @max(top, @as(f64, v));
        var sum: f64 = 0;
        for (row) |v| sum += @exp(@as(f64, v) - top);
        acc += w * (top + @log(sum) - @as(f64, row[@intFromFloat(y)]));
    }
    return acc / weightSum(weights);
}

/// Weighted multiclass cross-entropy from probabilities, clamped as `mloglossProb`.
pub fn mloglossProbW(prob: []const f32, labels: []const f32, k: usize, weights: []const f32) f64 {
    var acc: f64 = 0;
    for (labels, weights, 0..) |y, w, r| acc -= w * @log(@max(@as(f64, prob[r * k + @as(usize, @intFromFloat(y))]), 1e-15));
    return acc / weightSum(weights);
}

/// Weighted share of rows whose highest score is their class.
pub fn accuracyW(scores: []const f32, labels: []const f32, k: usize, weights: []const f32) f64 {
    var hit: f64 = 0;
    for (labels, weights, 0..) |y, w, r| {
        const row = scores[r * k ..][0..k];
        var best: usize = 0;
        for (row, 0..) |v, c| if (v > row[best]) {
            best = c;
        };
        if (@as(f32, @floatFromInt(best)) == y) hit += w;
    }
    return hit / weightSum(weights);
}

// ------------------------------------------------------- regression losses

/// Weight of row `r`: 1 when there are none.
inline fn wOf(weights: []const f32, r: usize) f64 {
    return if (weights.len != 0) weights[r] else 1;
}

fn total(weights: []const f32, n: usize) f64 {
    return if (weights.len != 0) weightSum(weights) else @floatFromInt(n);
}

/// (Weighted) mean absolute error.
pub fn mae(pred: []const f32, labels: []const f32, weights: []const f32) f64 {
    var acc: f64 = 0;
    for (pred, labels, 0..) |p, y, r| acc += wOf(weights, r) * @abs(@as(f64, p) - y);
    return acc / total(weights, labels.len);
}

/// (Weighted) mean pinball loss at `alpha`: `alpha * (y - p)` above the prediction,
/// `(1 - alpha) * (p - y)` below. 0.5 is half the absolute error.
pub fn pinball(pred: []const f32, labels: []const f32, weights: []const f32, alpha: f64) f64 {
    var acc: f64 = 0;
    for (pred, labels, 0..) |p, y, r| {
        const d = @as(f64, y) - p;
        acc += wOf(weights, r) * (if (d >= 0) alpha * d else (alpha - 1) * d);
    }
    return acc / total(weights, labels.len);
}

/// (Weighted) mean pseudo-Huber error with `slope`, XGBoost's `mphe`:
/// `slope^2 (sqrt(1 + (r / slope)^2) - 1)`.
pub fn mphe(pred: []const f32, labels: []const f32, weights: []const f32, slope: f64) f64 {
    var acc: f64 = 0;
    for (pred, labels, 0..) |p, y, r| {
        const z = (@as(f64, p) - y) / slope;
        acc += wOf(weights, r) * slope * slope * (@sqrt(1 + z * z) - 1);
    }
    return acc / total(weights, labels.len);
}

/// (Weighted) mean Poisson negative log-likelihood of predicted means `mu` (natural scale),
/// XGBoost's `poisson-nloglik`: `mu - y log(mu) + lgamma(y + 1)`, with `mu` floored at 1e-16.
pub fn poissonNloglik(mu: []const f32, labels: []const f32, weights: []const f32) f64 {
    var acc: f64 = 0;
    for (mu, labels, 0..) |m, y, r| {
        const p = @max(@as(f64, m), 1e-16);
        acc += wOf(weights, r) * (p - @as(f64, y) * @log(p) + std.math.lgamma(f64, @as(f64, y) + 1));
    }
    return acc / total(weights, labels.len);
}
