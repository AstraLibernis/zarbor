// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Linear leaves: an affine fit over a leaf's root-to-leaf numeric features.

const std = @import("std");
const data = @import("data.zig");
const tree = @import("tree.zig");
const binValue = tree.binValue;
const builder = @import("builder.zig");
const Builder = builder.Builder;
const Work = builder.Work;
const max_path = builder.max_path;

/// In-place Cholesky solve of a small symmetric positive-definite system.
///
/// `a` is read as its upper triangle and overwritten; `rhs` carries the
/// solution out. Returns false when the factorisation meets a non-positive
/// pivot, which is what a feature constant within the leaf looks like and is
/// common near the bottom of a tree. The caller treats that as "no slopes",
/// which is the admissible `beta = 0` solution rather than a failure.
fn choleskySolve(a: *[max_path][max_path]f64, rhs: *[max_path]f64, n: usize) bool {
    var l: [max_path][max_path]f64 = undefined;
    for (0..n) |i| for (0..n) |j| {
        l[i][j] = 0;
    };
    for (0..n) |i| {
        for (0..i + 1) |j| {
            var sum: f64 = if (j <= i) a[j][i] else a[i][j];
            for (0..j) |k| sum -= l[i][k] * l[j][k];
            if (i == j) {
                if (!(sum > 1e-12)) return false;
                l[i][i] = @sqrt(sum);
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    // Forward, then back.
    for (0..n) |i| {
        var sum = rhs[i];
        for (0..i) |k| sum -= l[i][k] * rhs[k];
        rhs[i] = sum / l[i][i];
    }
    var i = n;
    while (i > 0) {
        i -= 1;
        var sum = rhs[i];
        for (i + 1..n) |k| sum -= l[k][i] * rhs[k];
        rhs[i] = sum / l[i][i];
    }
    return true;
}

/// Slopes for one leaf, appended to `b.lin`. Returns how many were kept.
///
/// Centring each feature on its *hessian-weighted* within-leaf mean is not
/// cosmetic. It makes every cross term between the intercept and a slope
/// vanish, so the intercept is exactly the constant leaf weight already
/// computed and only the slope block has to be solved. A failure there is
/// therefore a fallback to the constant leaf rather than an error.
pub fn fitLinearLeaf(b: *Builder, w: Work) !u8 {
    const n = @min(@as(usize, w.n_path), @as(usize, b.cfg.lin_leaf_max_terms));
    const rows = b.rows[w.start..w.end];
    if (rows.len <= n + 1) return 0;

    var center: [max_path]f64 = undefined;
    var sum_h: f64 = 0;
    for (0..n) |j| center[j] = 0;
    for (rows) |r| {
        const h: f64 = b.g[r].h;
        sum_h += h;
        const rb = b.ds.bins_rm[r * b.ds.n_features ..][0..b.ds.n_features];
        for (0..n) |j| center[j] += h * binValue(b.ds, w.path[j], rb[w.path[j]]);
    }
    if (sum_h <= 0) return 0;
    for (0..n) |j| center[j] /= sum_h;

    // Upper triangle of the slope system, and its right-hand side.
    var a: [max_path][max_path]f64 = undefined;
    var rhs: [max_path]f64 = undefined;
    for (0..n) |j| {
        rhs[j] = 0;
        for (0..n) |k| a[j][k] = 0;
    }
    var x: [max_path]f64 = undefined;
    for (rows) |r| {
        const gp = b.g[r];
        const rb = b.ds.bins_rm[r * b.ds.n_features ..][0..b.ds.n_features];
        for (0..n) |j| x[j] = binValue(b.ds, w.path[j], rb[w.path[j]]) - center[j];
        for (0..n) |j| {
            rhs[j] -= @as(f64, gp.g) * x[j];
            for (j..n) |k| a[j][k] += @as(f64, gp.h) * x[j] * x[k];
        }
    }
    // Standardise before the ridge. `a[j][j]` is `sum_h * var_j`, so a
    // flat additive lambda means something different on a column measured
    // in dollars than on one measured in people -- it is not a ridge at
    // all, it is an arbitrary per-column shrinkage. Scaling each axis by
    // its own weighted sd makes lambda comparable across columns, which
    // is what `linear.zig` does to its design matrix for the same reason.
    var sd: [max_path]f64 = undefined;
    for (0..n) |j| {
        const v = a[j][j] / sum_h;
        sd[j] = if (v > 1e-30) @sqrt(v) else 0;
    }
    for (0..n) |j| {
        if (sd[j] == 0) return 0; // constant within the leaf: no slope to fit
        rhs[j] /= sd[j];
        for (j..n) |k| a[j][k] /= (sd[j] * sd[k]);
    }
    for (0..n) |j| a[j][j] += b.cfg.lin_leaf_lambda * sum_h;

    if (!choleskySolve(&a, &rhs, n)) return 0;
    for (0..n) |j| rhs[j] /= sd[j];

    var kept: u8 = 0;
    for (0..n) |j| {
        const c = rhs[j] * b.cfg.learning_rate;
        if (!std.math.isFinite(c) or c == 0) continue;
        try b.lin.append(b.gpa, .{
            .feature = w.path[j],
            .coef = @floatCast(c),
            .center = @floatCast(center[j]),
        });
        kept += 1;
    }
    return kept;
}
