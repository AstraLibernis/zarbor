// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

const std = @import("std");
const Pool = @import("../pool.zig").Pool;
const data = @import("../data.zig");
const config = @import("../config.zig");
const prof = @import("../prof.zig");
const metric = @import("../metric.zig");
const linear = @import("../linear.zig");
const lin_solve = @import("../lin_solve.zig");
const Problem = lin_solve.Problem;
const projectOrthant = lin_solve.projectOrthant;
const pseudoGrad = lin_solve.pseudoGrad;

const testing = std.testing;

test "pseudoGrad picks the least-magnitude subgradient at a zero coefficient" {
    var theta = [_]f64{ 2.0, -2.0, 0.0, 0.0, 0.0, 0.0 };
    var pr = Problem{
        .pool = undefined,
        .design = undefined,
        .ds = undefined,
        .objective = .logistic,
        .scale_pos_weight = 1,
        .l2 = 0,
        .l1 = 0.5,
        .theta = &theta,
        .wf = &.{},
        .z = &.{},
        .resid = &.{},
        .loss_part = &.{},
        .rsum_part = &.{},
        .chunks = 1,
        .size = 1,
    };
    //              w>0   w<0  w=0,g steep -  w=0,g steep +  w=0,g inside  intercept
    const g = [_]f64{ 1.0, 1.0, -3.0, 3.0, 0.25, 7.0 };
    var out: [6]f64 = undefined;
    pseudoGrad(&pr, &g, &out);

    try testing.expectEqual(@as(f64, 1.5), out[0]); // g + l1, sign of w
    try testing.expectEqual(@as(f64, 0.5), out[1]); // g - l1, sign of w
    try testing.expectEqual(@as(f64, -2.5), out[2]); // left derivative still negative
    try testing.expectEqual(@as(f64, 2.5), out[3]); // right derivative still positive
    // |g| < l1: zero is a minimum along this coordinate, so the subgradient
    // containing zero is the least-magnitude one and the coefficient stays.
    try testing.expectEqual(@as(f64, 0.0), out[4]);
    try testing.expectEqual(@as(f64, 7.0), out[5]); // the intercept is unpenalised

    // With no L1 term it must be the plain gradient, untouched.
    pr.l1 = 0;
    pseudoGrad(&pr, &g, &out);
    try testing.expectEqualSlices(f64, &g, &out);
}

test "projectOrthant zeroes exactly the coefficients that crossed" {
    //                   stays  crossed  stays  crossed  from zero, allowed
    const from = [_]f64{ 2.0, 2.0, -2.0, -2.0, 0.0, 0.0, 99.0 };
    const trial = [_]f64{ 1.0, -1.0, -1.0, 1.0, 0.5, -0.5, -99.0 };
    // At a zero coefficient the orthant is the one the pseudo-gradient points
    // into, i.e. -pg: negative pg admits a positive step and vice versa.
    const pg = [_]f64{ 0, 0, 0, 0, -1.0, -1.0, 0 };
    var t = trial;
    projectOrthant(&t, &from, &pg);
    try testing.expectEqualSlices(f64, &.{
        1.0, // same sign as before: kept
        0.0, // crossed from + to -: clipped
        -1.0, // same sign: kept
        0.0, // crossed from - to +: clipped
        0.5, // left zero in the direction -pg allows
        0.0, // left zero against -pg: clipped back
        -99.0, // the intercept is not in the orthant logic at all
    }, &t);
}
