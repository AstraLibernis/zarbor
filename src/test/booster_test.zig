// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

const std = @import("std");
const Pool = @import("../pool.zig").Pool;
const data = @import("../data.zig");
const hist = @import("../hist.zig");
const tree = @import("../tree.zig");
const config = @import("../config.zig");
const metric = @import("../metric.zig");
const prof = @import("../prof.zig");
const booster = @import("../booster.zig");
const goss = @import("../goss.zig");
const n_radix = goss.n_radix;
const sigmoid = booster.sigmoid;
const gossCut = goss.gossCut;
const sigmoid1 = booster.sigmoid1;
const F8 = booster.F8;
const lanes = booster.lanes;
const sigmoid8 = booster.sigmoid8;

const testing = std.testing;

test "vectorised sigmoid matches the scalar one" {
    // The gradient loop uses an inlined 2^x rather than libm's expf, which is
    // 11.7x faster and approximate. This pins how approximate: anything worse
    // than a few f32 ulp would start moving split decisions around.
    var max_err: f32 = 0;
    var x: f32 = -60.0;
    while (x <= 60.0) : (x += 0.0007) {
        const want = sigmoid(x);
        const got = sigmoid1(x);
        max_err = @max(max_err, @abs(got - want));
    }
    try testing.expect(max_err < 1e-6);

    // Saturation, and no NaN where exp would overflow.
    for ([_]f32{ -1e30, -1000, -89, 0, 89, 1000, 1e30 }) |v| {
        const got = sigmoid1(v);
        try testing.expect(!std.math.isNan(got));
        try testing.expect(got >= 0.0 and got <= 1.0);
        try testing.expect(@abs(got - sigmoid(v)) < 1e-6);
    }
    try testing.expectApproxEqAbs(@as(f32, 0.5), sigmoid1(0), 1e-7);

    // Every lane must agree with lane 0, or the tail of the gradient loop
    // would not match its body.
    const v = F8{ -3.25, -1.0, -0.125, 0, 0.125, 1.0, 3.25, 7.5 };
    const got: [lanes]f32 = sigmoid8(v);
    const src: [lanes]f32 = v;
    for (got, src) |g, sx| try testing.expectEqual(sigmoid1(sx), g);
}

test "gossCut separates exactly the k largest |g|" {
    // A selection step is easy to get subtly wrong — off by one at the
    // threshold, or wrong when every value is identical. Checked against a
    // full sort on random, already-sorted, reverse-sorted, all-equal and
    // sign-mixed inputs, since boosting produces all of them.
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(4);
    const r = prng.random();

    const counts = try gpa.alloc(u32, n_radix);
    defer gpa.free(counts);

    const shapes = enum { random, ascending, descending, all_equal, signed, tiny_spread };
    for (std.enums.values(shapes)) |shape| {
        for ([_]usize{ 1, 2, 7, 64, 1000, 5000 }) |n| {
            const grads = try gpa.alloc(hist.GradPair, n);
            defer gpa.free(grads);
            for (grads, 0..) |*g, i| {
                const v: f32 = switch (shape) {
                    .random => r.floatNorm(f32),
                    .ascending => @floatFromInt(i),
                    .descending => @floatFromInt(n - i),
                    .all_equal => 1.0,
                    // Magnitude is what is ranked, so signs must not matter.
                    .signed => if (i % 2 == 0) -r.float(f32) else r.float(f32),
                    // Values inside one radix bucket, which is what forces the
                    // second pass to do the work.
                    .tiny_spread => 1.0 + @as(f32, @floatFromInt(i % 3)) * 1e-7,
                };
                g.* = .{ .g = v, .h = 1 };
            }

            const mags = try gpa.alloc(f32, n);
            defer gpa.free(mags);
            for (mags, grads) |*m, g| m.* = @abs(g.g);
            std.sort.pdq(f32, mags, {}, std.sort.desc(f32));

            for ([_]usize{ 1, n / 3, n / 2, n - 1, n }) |k| {
                if (k == 0 or k > n) continue;
                const cut = gossCut(grads, counts, k, .gradient);

                // Replay the selection rule the caller uses.
                const chosen = try gpa.alloc(bool, n);
                defer gpa.free(chosen);
                var take = cut.take;
                var n_chosen: usize = 0;
                for (grads, 0..) |g, i| {
                    const b: u32 = @bitCast(@abs(g.g));
                    const keep = b > cut.t or (b == cut.t and take > 0);
                    chosen[i] = keep;
                    if (keep) {
                        if (b == cut.t) take -= 1;
                        n_chosen += 1;
                    }
                }

                // Exactly k rows, and every one of them at least as large as
                // every row left behind. That is the whole contract.
                try testing.expectEqual(k, n_chosen);
                const kth = mags[k - 1];
                for (grads, chosen) |g, c| {
                    if (c) {
                        try testing.expect(@abs(g.g) >= kth);
                    } else {
                        try testing.expect(@abs(g.g) <= kth);
                    }
                }
            }
        }
    }
}
