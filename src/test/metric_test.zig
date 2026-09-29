// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

const std = @import("std");
const metric = @import("../metric.zig");
const auc = metric.auc;

const testing = std.testing;

test "auc is 1 for a perfect ranking" {
    const s = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
    const y = [_]f32{ 0, 0, 1, 1 };
    try testing.expectApproxEqAbs(@as(f64, 1.0), try auc(testing.allocator, &s, &y), 1e-12);
}

test "auc is 0.5 when every score ties" {
    const s = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    const y = [_]f32{ 0, 1, 0, 1 };
    try testing.expectApproxEqAbs(@as(f64, 0.5), try auc(testing.allocator, &s, &y), 1e-12);
}

test "auc is 0 for a perfectly inverted ranking" {
    const s = [_]f32{ 0.9, 0.8, 0.2, 0.1 };
    const y = [_]f32{ 0, 0, 1, 1 };
    try testing.expectApproxEqAbs(@as(f64, 0.0), try auc(testing.allocator, &s, &y), 1e-12);
}
