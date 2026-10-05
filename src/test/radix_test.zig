// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

const std = @import("std");
const radix = @import("../radix.zig");

const testing = std.testing;

test "radix order equals std.mem.sort's stable order, signed zeros included" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(3);
    const r = prng.random();
    const n = 50_000;
    const vals = try gpa.alloc(f32, n);
    defer gpa.free(vals);
    const keys = try gpa.alloc(u64, 2 * n);
    defer gpa.free(keys);
    const want = try gpa.alloc(f64, n);
    defer gpa.free(want);

    for (0..3) |shape| {
        for (vals) |*v| v.* = switch (shape) {
            0 => (r.float(f32) - 0.5) * 1e9,
            1 => @floatFromInt(@as(i32, @intCast(r.uintLessThan(u32, 21))) - 10),
            // Many zeros of both signs, so their relative order is visible in the bits.
            else => if (r.uintLessThan(u32, 3) == 0) (if (r.boolean()) -0.0 else 0.0) else r.float(f32) - 0.5,
        };
        for (want, vals) |*w, v| w.* = v;
        std.mem.sort(f64, want, {}, std.sort.asc(f64));
        for (keys[0..n], vals) |*k, v| k.* = @as(u64, radix.f32Key(v)) << 32 | @as(u32, @bitCast(v));
        const got = radix.sortHigh32(keys[0..n], keys[n..]);
        for (want, got) |w, k| {
            const g: f64 = @as(f32, @bitCast(@as(u32, @truncate(k))));
            try testing.expectEqual(@as(u64, @bitCast(w)), @as(u64, @bitCast(g)));
        }
    }
}
