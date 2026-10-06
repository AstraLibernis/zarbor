// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! LSD radix sort on an f32 key carried in the high half of a u64, for the two places that sort
//! hundreds of thousands of floats: the AUC (every logged round) and the column profile (every
//! numeric column at load). Both comparison sorts it replaced were slower (docs/measurements.md).

const std = @import("std");

/// An f32 as a u32 whose unsigned order is the float order. -0 and +0 share a key, as they compare
/// equal under `<` and `==`. NaN has no place in that order: callers either skip it or accept that
/// it groups by bit pattern.
pub fn f32Key(x: f32) u32 {
    const bits: u32 = @bitCast(if (x == 0) @as(f32, 0) else x);
    return if (bits >> 31 == 1) ~bits else bits | 0x8000_0000;
}

/// Sort `src` by its high 32 bits, using `dst` (same length) as the ping-pong buffer; returns
/// whichever of the two holds the result. Stable, so elements with equal keys keep their order,
/// as `std.mem.sort` keeps it. A byte every key shares cannot reorder anything and is skipped.
pub fn sortHigh32(src_in: []u64, dst_in: []u64) []u64 {
    std.debug.assert(src_in.len == dst_in.len);
    const n = src_in.len;
    var src = src_in;
    var dst = dst_in;
    if (n < 2) return src;
    var shift: u6 = 32;
    while (true) : (shift += 8) {
        var count = [_]usize{0} ** 256;
        for (src) |v| count[@as(u8, @truncate(v >> shift))] += 1;
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
    return src;
}
