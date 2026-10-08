// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Evaluation metrics.

const std = @import("std");
const radix = @import("radix.zig");
const Objective = @import("objective.zig").Objective;

/// What a holdout is scored by: early stopping, the logged validation score, `cv` and `tune`.
/// Names follow XGBoost's `eval_metric` where it has one. Each objective has a default
/// (`defaultFor`); `--eval_metric` picks another that fits the target (`fits`).
pub const Metric = enum {
    /// ROC AUC (binary).
    auc,
    /// Average precision, the area under the precision-recall curve as scikit-learn's
    /// `average_precision_score` takes it (binary).
    aucpr,
    /// Binary cross-entropy.
    logloss,
    /// Share of rows on the wrong side of probability 0.5 (XGBoost's `error`; binary).
    error_rate,
    /// Share of rows whose most likely class is their class (binary or multiclass).
    accuracy,
    /// Multiclass cross-entropy.
    mlogloss,
    rmse,
    mae,
    /// Coefficient of determination, `1 - SS_res / SS_tot` (scikit-learn's `r2_score`).
    r2,
    /// Mean pinball loss at `quantile_alpha`.
    pinball,
    /// Mean pseudo-Huber error at `huber_slope`.
    mphe,
    /// Mean Poisson negative log-likelihood (predictions must be positive: `poisson` only).
    poisson_nloglik,

    pub fn defaultFor(obj: Objective) Metric {
        return switch (obj) {
            .logistic => .auc,
            .squared_error => .rmse,
            .softmax => .mlogloss,
            .absolute_error => .mae,
            .quantile => .pinball,
            .pseudo_huber => .mphe,
            .poisson => .poisson_nloglik,
        };
    }

    pub fn higherIsBetter(m: Metric) bool {
        return switch (m) {
            .auc, .aucpr, .accuracy, .r2 => true,
            else => false,
        };
    }

    /// Whether `m` can score predictions of `obj`.
    pub fn fits(m: Metric, obj: Objective) bool {
        return switch (m) {
            .auc, .aucpr, .logloss, .error_rate => obj == .logistic,
            .accuracy => obj == .logistic or obj == .softmax,
            .mlogloss => obj == .softmax,
            .rmse, .mae, .r2, .pinball, .mphe => !obj.classifies(),
            .poisson_nloglik => obj == .poisson,
        };
    }
};

/// Whether scores are the model's raw output (log-odds, class scores, log mean) or its
/// predictions (probabilities, means).
pub const Scale = enum { raw, natural };

/// What a metric needs beyond predictions and labels.
pub const Ctx = struct {
    objective: Objective,
    /// Scores per row: the class count under softmax, else 1. Layout `scores[r * k + c]`.
    k: usize = 1,
    quantile_alpha: f64 = 0.5,
    huber_slope: f64 = 1.0,
};

/// Score `scores` (on `scale`) against `labels` by `m`; `weights` empty for none. Rank and
/// argmax metrics use raw scores as they are (the links are monotone), log losses use their
/// stable raw forms, and the rest score natural-scale predictions.
pub fn evaluate(
    gpa: std.mem.Allocator,
    m: Metric,
    ctx: Ctx,
    scores: []const f32,
    scale: Scale,
    labels: []const f32,
    weights: []const f32,
) !f64 {
    const w = weights.len != 0;
    switch (m) {
        .auc => return if (w) try aucW(gpa, scores, labels, weights) else try auc(gpa, scores, labels),
        .aucpr => return try averagePrecision(gpa, scores, labels, weights),
        .logloss => return switch (scale) {
            .raw => if (w) loglossW(scores, labels, weights) else logloss(scores, labels),
            .natural => if (w) loglossProbW(scores, labels, weights) else loglossProb(scores, labels),
        },
        .error_rate => return errorRate(scores, labels, weights, if (scale == .raw) 0 else 0.5),
        .accuracy => {
            if (ctx.k == 1) return 1 - errorRate(scores, labels, weights, if (scale == .raw) 0 else 0.5);
            return if (w) accuracyW(scores, labels, ctx.k, weights) else accuracy(scores, labels, ctx.k);
        },
        .mlogloss => return switch (scale) {
            .raw => if (w) mloglossW(scores, labels, ctx.k, weights) else mlogloss(scores, labels, ctx.k),
            .natural => if (w) mloglossProbW(scores, labels, ctx.k, weights) else mloglossProb(scores, labels, ctx.k),
        },
        else => {},
    }
    // Value metrics: Poisson's raw score is a log mean; every other regression raw is its value.
    var owned: []f32 = &.{};
    defer gpa.free(owned);
    var pred = scores;
    if (scale == .raw and ctx.objective == .poisson) {
        owned = try gpa.alloc(f32, scores.len);
        for (owned, scores) |*o, r| o.* = @exp(r);
        pred = owned;
    }
    return switch (m) {
        .rmse => if (w) rmseW(pred, labels, weights) else rmse(pred, labels),
        .mae => mae(pred, labels, weights),
        .r2 => r2(pred, labels, weights),
        .pinball => pinball(pred, labels, weights, ctx.quantile_alpha),
        .mphe => mphe(pred, labels, weights, ctx.huber_slope),
        .poisson_nloglik => poissonNloglik(pred, labels, weights),
        else => unreachable, // handled above
    };
}

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

// ------------------------------------------------------- added for --eval_metric

/// (Weighted) average precision, scikit-learn's `average_precision_score`:
/// `sum over thresholds of (R_t - R_{t-1}) P_t`, one threshold per distinct score, highest
/// first. Without positives it is 0, recall being undefined.
pub fn averagePrecision(gpa: std.mem.Allocator, scores: []const f32, labels: []const f32, weights: []const f32) !f64 {
    const n = scores.len;
    if (n == 0) return 0;
    const buf = try gpa.alloc(u64, 2 * n);
    defer gpa.free(buf);
    for (buf[0..n], scores, 0..) |*v, x, i| v.* = @as(u64, radix.f32Key(x)) << 32 | @as(u32, @intCast(i));
    const src = radix.sortHigh32(buf[0..n], buf[n..]);

    var pos_total: f64 = 0;
    for (labels, 0..) |y, r| {
        if (y > 0.5) pos_total += wOf(weights, r);
    }
    if (pos_total == 0) return 0;

    var tp: f64 = 0;
    var fp: f64 = 0;
    var ap: f64 = 0;
    var j = n; // blocks of one score, from the highest down: [i, j)
    while (j > 0) {
        const key = src[j - 1] >> 32;
        var i = j;
        var dtp: f64 = 0;
        while (i > 0 and src[i - 1] >> 32 == key) : (i -= 1) {
            const r: usize = @intCast(src[i - 1] & 0xFFFF_FFFF);
            const wr = wOf(weights, r);
            if (labels[r] > 0.5) dtp += wr else fp += wr;
        }
        tp += dtp;
        if (dtp > 0) ap += dtp / pos_total * (tp / (tp + fp));
        j = i;
    }
    return ap;
}

/// (Weighted) share of rows whose score is on the wrong side of `threshold` (above it counts as
/// positive, XGBoost's `error`).
pub fn errorRate(scores: []const f32, labels: []const f32, weights: []const f32, threshold: f32) f64 {
    var wrong: f64 = 0;
    for (scores, labels, 0..) |s, y, r| {
        if ((s > threshold) != (y > 0.5)) wrong += wOf(weights, r);
    }
    return wrong / total(weights, labels.len);
}

/// (Weighted) coefficient of determination, scikit-learn's `r2_score`: 1 for a perfect fit, 0
/// for predicting the (weighted) label mean. A constant target gives 1 if predicted exactly,
/// else 0, as scikit-learn does.
pub fn r2(pred: []const f32, labels: []const f32, weights: []const f32) f64 {
    var sw: f64 = 0;
    var sy: f64 = 0;
    for (labels, 0..) |y, r| {
        sw += wOf(weights, r);
        sy += wOf(weights, r) * y;
    }
    const mean = sy / sw;
    var res: f64 = 0;
    var tot: f64 = 0;
    for (pred, labels, 0..) |p, y, r| {
        const d = @as(f64, y) - p;
        const e = @as(f64, y) - mean;
        res += wOf(weights, r) * d * d;
        tot += wOf(weights, r) * e * e;
    }
    if (tot == 0) return if (res == 0) 1 else 0;
    return 1 - res / tot;
}
