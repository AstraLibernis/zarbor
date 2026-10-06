// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Histogram construction; the split search is in split.zig. This is the whole cost of boosting.
//! 1. Row-outer: each row's gradient is read once and added to every feature's histogram from the
//!    row-major bins. The gradient read is the costly random access; per row, not per feature.
//! 2. A node over `parallel_threshold` rows is cut into parts fixed by its row count alone (at
//!    most `max_parts`); each part sums into its own private slot, and the slots are added in
//!    part order. No atomics, and every histogram is bit-identical at any thread count.
//! Sums are f64 though gradients are f32: a root adds hundreds of thousands of terms into a few
//! hundred accumulators, and f32 loses real precision over that many additions.

const std = @import("std");
const Pool = @import("pool.zig").Pool;
const data = @import("data.zig");
const Dataset = data.Dataset;
const prof = @import("prof.zig");

/// First-order and second-order derivative of the loss at one row.
pub const GradPair = extern struct {
    g: f32,
    h: f32,
};

/// One histogram cell: gradient sum, hessian sum, row count. Four 32-byte-aligned f64 lanes, not
/// two doubles and a u32: a lane for the count makes the update one aligned vector load-add-store,
/// not three scattered read-modify-writes; the dead lane makes the size a power of two (shift, not
/// multiply). Faster than the 24-byte shape on real bin widths, and the gain composes with the
/// row-major layout at every node size measured; see docs/measurements.md.
pub const Bin = extern struct {
    g: f64 align(32) = 0,
    h: f64 = 0,
    n: f64 = 0,
    _pad: f64 = 0,

    pub const Vec = @Vector(4, f64);

    pub inline fn vec(b: Bin) Vec {
        return @bitCast(b);
    }

    pub inline fn fromVec(x: Vec) Bin {
        return @bitCast(x);
    }

    pub inline fn add(a: Bin, b: Bin) Bin {
        return fromVec(a.vec() + b.vec());
    }

    pub inline fn sub(a: Bin, b: Bin) Bin {
        return fromVec(a.vec() - b.vec());
    }
};

/// Scratch for one tree's histograms. Packed per-feature offsets, not a widest-feature stride:
/// real widths are uneven (a couple of wide numeric columns, many 3-4 bin categoricals), so a
/// widest-feature stride is mostly empty slots. Clear+reduce is fixed per node, dominating the many
/// small bottom nodes, so packing gains most there (see docs/measurements.md); a depth-6 tree has
/// half its internal nodes in the last level. Features start on cache lines, keeping clear/reduce
/// off partial lines.
pub const Bank = struct {
    gpa: std.mem.Allocator,
    n_features: usize,
    /// Feature `f`'s bins: `offsets[f]..offsets[f+1]`. `n_features + 1` long; last = slot length.
    offsets: []u32,
    /// `max_parts * slotLen()`: one partial histogram per part of a parallel build.
    private: []Bin,

    /// Bins per alignment step, so `step * @sizeOf(Bin)` is whole cache lines. Not `cache_line /
    /// @sizeOf(Bin)`: the 24-byte `Bin` this once was gave 128/24 = 5 on a 128-byte line, not the
    /// power of two `alignForward` needs; Debug asserts, ReleaseFast silently overlaps feature slices
    /// and corrupts every split search. gcd gives the smallest valid step for any size: on a
    /// 128-byte line, 16 for 24 bytes and 4 for today's `Bin`.
    const bin_step: usize = @max(1, std.atomic.cache_line / std.math.gcd(@sizeOf(Bin), std.atomic.cache_line));

    inline fn roundUp(n: usize) usize {
        // Multiple-rounding, not alignForward: `bin_step` has no power-of-two guarantee.
        return ((n + bin_step - 1) / bin_step) * bin_step;
    }

    pub fn init(
        gpa: std.mem.Allocator,
        n_features: usize,
        n_bins: []const u16,
    ) !Bank {
        std.debug.assert(n_bins.len == n_features);
        const offsets = try gpa.alloc(u32, n_features + 1);
        errdefer gpa.free(offsets);
        var acc: usize = 0;
        for (0..n_features) |f| {
            offsets[f] = @intCast(acc);
            acc += roundUp(n_bins[f]);
        }
        offsets[n_features] = @intCast(acc);

        const private = try gpa.alloc(Bin, max_parts * acc);
        return .{
            .gpa = gpa,
            .n_features = n_features,
            .offsets = offsets,
            .private = private,
        };
    }

    pub fn deinit(b: *Bank) void {
        b.gpa.free(b.private);
        b.gpa.free(b.offsets);
        b.* = undefined;
    }

    /// Bins in one node's histogram slot.
    pub fn slotLen(b: *const Bank) usize {
        return b.offsets[b.n_features];
    }

    pub inline fn featureSlice(b: *const Bank, hist: []Bin, f: usize) []Bin {
        return hist[b.offsets[f]..b.offsets[f + 1]];
    }

    pub inline fn featureSliceConst(b: *const Bank, hist: []const Bin, f: usize) []const Bin {
        return hist[b.offsets[f]..b.offsets[f + 1]];
    }
};

/// Row-outer accumulation over row-major bins: faster than feature-outer column-major on real data.
/// A row's bins are contiguous, so `rows[i]` and `grads[rows[i]]` are read once per row, not per
/// feature; its updates hit different histograms, so are independent, while feature-outer on a
/// 3-bin categorical collides constantly and serialises on store-to-load forwarding. An earlier
/// benchmark with 256 bins for every feature saw little gain: wide uniform features make collisions
/// rare and spill to L2, but real tables are lopsided (most bins in a couple of columns, the rest a
/// handful each). No prefetch or unroll, both slower: rows come ascending from the stable
/// partition, and more rows in flight only add register pressure. See docs/measurements.md.
const BuildCtx = struct {
    bank: *Bank,
    ds: *const Dataset,
    /// Row ids belonging to this node, contiguous.
    rows: []const u32,
    /// Indexed by original row id (`grads[rows[i]]`), not permuted: keeping it in step would make
    /// the partition move each row's gradients along with its id, which costs far more than this
    /// gather (see docs/measurements.md).
    grads: []const GradPair,
    features: []const u32,
    /// Rows per part; part `c` is `rows[c * size ..]`, accumulated into private slot `c`.
    size: usize,

    /// `begin..end` are part indices: one worker may be handed several, or all of them.
    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *BuildCtx = @ptrCast(@alignCast(ctx));
        const span = self.bank.slotLen();
        for (begin..end) |c| {
            const mine = self.bank.private[c * span ..][0..span];
            @memset(mine, .{});
            const lo = @min(c * self.size, self.rows.len);
            const hi = @min(lo + self.size, self.rows.len);
            accumulate(self.ds, self.bank.offsets, self.rows[lo..hi], self.grads, self.features, mine);
        }
    }
};

/// The accumulate kernel, shared by the pool path and `buildInto` so both add in the same order.
fn accumulate(
    ds: *const Dataset,
    offs: []const u32,
    rows: []const u32,
    grads: []const GradPair,
    features: []const u32,
    dst: []Bin,
) void {
    const nf = ds.n_features;
    const rm = ds.bins_rm;
    for (rows) |r| {
        const row: usize = r;
        const rb = rm[row * nf ..][0..nf];
        const g = grads[row];
        const v = Bin.Vec{ g.g, g.h, 1, 0 };
        for (features) |fid| {
            const cell: *Bin.Vec = @ptrCast(&dst[offs[fid] + rb[fid]]);
            cell.* += v;
        }
    }
}

/// One node of at most `parallel_threshold` rows, accumulated straight into `out` on the calling
/// thread. This is exactly what `build` runs for such a node. It touches no shared state (no
/// `Bank.private` slot, no pool), so several nodes can be built at once from pool tasks.
pub fn buildInto(
    bank: *const Bank,
    ds: *const Dataset,
    rows: []const u32,
    grads: []const GradPair,
    features: []const u32,
    out: []Bin,
) void {
    std.debug.assert(rows.len <= parallel_threshold);
    const span = bank.slotLen();
    @memset(out[0..span], .{});
    accumulate(ds, bank.offsets, rows, grads, features, out[0..span]);
}

/// `subtract` on the calling thread, for pool tasks. Element-wise, so identical at any grouping.
pub fn subtractInto(bank: *const Bank, out: []Bin, parent: []const Bin, sibling: []const Bin) void {
    const span = bank.slotLen();
    for (out[0..span], parent[0..span], sibling[0..span]) |*o, pa, si| o.* = pa.sub(si);
}

/// Reduces the parts' private histograms into `out` over the whole slot. All features, not just
/// selected ones: a flat branch-free pass over one slot (`slotLen()` bins) beats the index
/// arithmetic of skipping. Unselected features hold whatever their private copies did, which the
/// caller already must not read.
const ReduceCtx = struct {
    bank: *Bank,
    /// Parts `0..n_parts` were built; summed in that order.
    n_parts: usize,
    out: []Bin,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *ReduceCtx = @ptrCast(@alignCast(ctx));
        const span = self.bank.slotLen();
        const private = self.bank.private;
        var i = begin;
        while (i < end) : (i += 1) {
            var acc = private[i];
            for (1..self.n_parts) |c| acc = acc.add(private[c * span + i]);
            self.out[i] = acc;
        }
    }
};

/// At or below this many rows a node is built on one thread (`buildInto`). Tuned by a sweep
/// against per-node clear+reduce cost: flat over a wide middle range, worse when parts get too small
/// for the barrier; see docs/measurements.md. It also sets how a node's sum is grouped (one run, or
/// `ceil(rows / parallel_threshold)` parts capped at `max_parts`), so changing it changes histograms
/// in the last bits; the thread count never does. builder.zig's `batch_max_rows` is twice this and
/// must stay below partition.zig's `parallel_partition_min`.
pub const parallel_threshold: usize = 512;

/// Build one node's histogram into `out`, one slot's worth (`bank.slotLen()` bins). Only `features`
/// are populated; the caller must not read the others.
pub fn build(
    pool: *Pool,
    bank: *Bank,
    ds: *const Dataset,
    rows: []const u32,
    grads: []const GradPair,
    features: []const u32,
    out: []Bin,
) void {
    if (rows.len <= parallel_threshold) {
        const t_a = prof.start();
        buildInto(bank, ds, rows, grads, features, out);
        prof.stop(.hist_accum, t_a);
        return;
    }

    // Parts are fixed by the row count alone, never by the thread count or by which worker takes
    // which: each part sums into its own slot and the slots are added in part order, so every
    // histogram is the same to the bit at any `--n_threads` and on every run. Per-worker slots,
    // as before, grouped the additions by whichever chunks a worker happened to grab.
    const n_parts = @min(max_parts, (rows.len + parallel_threshold - 1) / parallel_threshold);
    var bctx = BuildCtx{
        .bank = bank,
        .ds = ds,
        .rows = rows,
        .grads = grads,
        .features = features,
        .size = (rows.len + n_parts - 1) / n_parts,
    };
    const t_a = prof.start();
    pool.parallelFor(n_parts, &bctx, BuildCtx.run, 1);
    prof.stop(.hist_accum, t_a);

    const t_r = prof.start();
    defer prof.stop(.hist_reduce, t_r);
    var rctx = ReduceCtx{ .bank = bank, .n_parts = n_parts, .out = out };
    pool.parallelFor(bank.slotLen(), &rctx, ReduceCtx.run, 512);
}

/// Most partial histograms a parallel build makes. Fixed, so a build's grouping depends only on
/// its row count. Chosen by a sweep: this value matched or beat the old per-worker build, and more
/// parts were slower, the extra partials costing more to clear and reduce than balance gained; see
/// docs/measurements.md.
pub const max_parts: usize = 16;

const SubCtx = struct {
    out: []Bin,
    parent: []const Bin,
    sibling: []const Bin,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *SubCtx = @ptrCast(@alignCast(ctx));
        var i = begin;
        while (i < end) : (i += 1) self.out[i] = self.parent[i].sub(self.sibling[i]);
    }
};

/// A node's histogram is parent minus sibling: one flat pass over the packed slot.
pub fn subtract(pool: *Pool, bank: *Bank, out: []Bin, parent: []const Bin, sibling: []const Bin) void {
    const span = bank.slotLen();
    var ctx = SubCtx{ .out = out[0..span], .parent = parent[0..span], .sibling = sibling[0..span] };
    pool.parallelFor(span, &ctx, SubCtx.run, 1024);
}
