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
    // much faster and approximate. This pins how approximate: anything worse
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

/// The serial selection `gossSelect` replaced: the reference its parallel passes must reproduce.
fn gossSelectSerial(
    grads: []hist.GradPair,
    counts: []u32,
    others: []u32,
    mask: []u64,
    out: []u32,
    top_rate: f32,
    other_rate: f32,
    rng: std.Random,
) []u32 {
    const n = grads.len;
    var top: usize = @intFromFloat(@round(@as(f32, @floatFromInt(n)) * top_rate));
    top = std.math.clamp(top, 1, n);
    const cut = gossCut(grads, counts, top, .gradient);
    @memset(mask[0 .. (n + 63) / 64], 0);
    var take = cut.take;
    var n_other: usize = 0;
    for (grads, 0..) |g, r| {
        const b: u32 = @bitCast(@abs(g.g));
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
    rand_n = @min(rand_n, @min(n - top, n_other));
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
    var k: usize = 0;
    for (mask[0 .. (n + 63) / 64], 0..) |w0, wi| {
        var w = w0;
        while (w != 0) {
            out[k] = @intCast(wi * 64 + @ctz(w));
            k += 1;
            w &= w - 1;
        }
    }
    return out[0..k];
}

test "parallel gossSelect equals the serial selection at every thread count" {
    const gpa = testing.allocator;
    // Odd length (a partial last mask word, uneven chunks) and gradients drawn from a few values, so
    // the cut lands inside a large tied block that spans many chunks.
    const n = 200_003;
    const base = try gpa.alloc(hist.GradPair, n);
    defer gpa.free(base);
    var prng: std.Random.DefaultPrng = .init(7);
    for (base) |*p| p.* = .{ .g = @as(f32, @floatFromInt(prng.random().uintLessThan(u32, 6))) - 2.5, .h = 1 };

    const counts = try gpa.alloc(u32, n_radix);
    defer gpa.free(counts);
    const want_grads = try gpa.dupe(hist.GradPair, base);
    defer gpa.free(want_grads);
    const got_grads = try gpa.alloc(hist.GradPair, n);
    defer gpa.free(got_grads);
    const others = try gpa.alloc(u32, n);
    defer gpa.free(others);
    const mask = try gpa.alloc(u64, (n + 63) / 64);
    defer gpa.free(mask);
    const want_buf = try gpa.alloc(u32, n);
    defer gpa.free(want_buf);
    const got_buf = try gpa.alloc(u32, n);
    defer gpa.free(got_buf);

    for ([_]f32{ 0.2, 0.37 }) |top_rate| {
        @memcpy(want_grads, base);
        var rng_a: std.Random.DefaultPrng = .init(11);
        const want = gossSelectSerial(want_grads, counts, others, mask, want_buf, top_rate, 0.1, rng_a.random());

        for ([_]u32{ 1, 3, 16 }) |threads| {
            const pool = try Pool.init(gpa, threads);
            defer pool.deinit();
            const chunks = try gpa.alloc(usize, goss.scratchLen(pool));
            defer gpa.free(chunks);
            @memcpy(got_grads, base);
            var rng_b: std.Random.DefaultPrng = .init(11);
            const got = goss.gossSelect(pool, got_grads, counts, chunks, others, mask, got_buf, top_rate, 0.1, .gradient, rng_b.random());
            try testing.expectEqualSlices(u32, want, got);
            try testing.expectEqualSlices(f32, std.mem.bytesAsSlice(f32, std.mem.sliceAsBytes(want_grads)), std.mem.bytesAsSlice(f32, std.mem.sliceAsBytes(got_grads)));
        }
    }
}
