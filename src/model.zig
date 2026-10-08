// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Saving, loading, scoring and blending trained models.
//!
//! A trained model is useless if it cannot outlive the process that made it,
//! and scoring new data needs more than the trees: it needs the exact binning
//! the model was trained under. That is why a bundle carries a `data.Schema`
//! — bin edges *and* categorical level strings — rather than just coefficients.
//!
//! All three model kinds expose one `predict`, on the objective's natural
//! scale (probabilities for logistic, values for regression). That uniformity
//! is what makes blending a few lines rather than a special case per kind,
//! and blending diverse models is what actually wins tabular competitions.

const std = @import("std");
const data = @import("data.zig");
const tree = @import("tree.zig");
const hist = @import("hist.zig");
const booster = @import("booster.zig");
const forest = @import("forest.zig");
const linear = @import("linear.zig");
const Pool = @import("pool.zig").Pool;
const Objective = @import("objective.zig").Objective;
const split = @import("split.zig");

pub const magic = "ZMDL";
/// 2 added the label encoding. A version-1 file still loads; it simply has
/// no class order, which the scoring path rejects rather than guessing at.
/// 3 added categorical subset splits and linear leaves (per-node `n_lin`/`lin_ofs`
/// and a per-tree term table). A version-2 file still loads: it has no mask store,
/// no term table and no node claiming either, which is exactly what an
/// ordinal-split, constant-leaf model is.
/// 4 widened the bin index to `u16` and replaced the categorical bitmask with
/// a sorted list of the level ids on the left. Versions 2 and 3 still load;
/// their thresholds are read a byte at a time and their masks expanded into
/// the list form, so a model saved before this change scores identically.
/// 6 added the class count and each class's starting score (softmax). Older files
/// have one class.
pub const format_version: u32 = 6;

pub const Kind = enum(u8) {
    /// Trees are summed onto `base_score`; logistic needs a sigmoid after.
    gbdt = 0,
    /// Trees are averaged and already hold `mean(y)`; no link to invert.
    forest = 1,
    linear = 2,
};

pub const Bundle = struct {
    gpa: std.mem.Allocator,
    kind: Kind,
    schema: data.Schema,
    objective: Objective,
    base_score: f32 = 0,
    /// Softmax: classes and each one's starting score (owned); tree `i` is class `i % num_class`.
    num_class: u32 = 1,
    class_base: []f32 = &.{},
    trees: []tree.Tree = &.{},
    lin: ?linear.Linear = null,
    /// Target column the model was trained on. Empty when unknown.
    label: []u8 = &.{},
    /// Class order for a string target, `classes[0]` meaning label 0. Empty
    /// for a numeric target, and for models written before format 2.
    classes: [][]u8 = &.{},

    pub fn deinit(b: *Bundle) void {
        for (b.trees) |*t| t.deinit(b.gpa);
        if (b.trees.len != 0) b.gpa.free(b.trees);
        if (b.class_base.len != 0) b.gpa.free(b.class_base);
        if (b.lin) |*l| l.deinit();
        b.schema.deinit();
        if (b.label.len != 0) b.gpa.free(b.label);
        for (b.classes) |c| b.gpa.free(c);
        if (b.classes.len != 0) b.gpa.free(b.classes);
        b.* = undefined;
    }

    /// Record the target column and its encoding, copying both.
    pub fn setLabel(b: *Bundle, name: []const u8, enc: *const data.LabelEncoder) !void {
        const label = try b.gpa.dupe(u8, name);
        errdefer b.gpa.free(label);
        const classes = try b.gpa.alloc([]u8, enc.classes.len);
        var made: usize = 0;
        errdefer {
            for (classes[0..made]) |c| b.gpa.free(c);
            b.gpa.free(classes);
        }
        while (made < enc.classes.len) : (made += 1) classes[made] = try b.gpa.dupe(u8, enc.classes[made]);

        if (b.label.len != 0) b.gpa.free(b.label);
        for (b.classes) |c| b.gpa.free(c);
        if (b.classes.len != 0) b.gpa.free(b.classes);
        b.label = label;
        b.classes = classes;
    }

    /// A borrowing encoder over `classes`. Valid only while the bundle is,
    /// and must never be deinitialised.
    pub fn encoder(b: *const Bundle) data.LabelEncoder {
        return data.LabelEncoder.view(b.gpa, b.classes);
    }

    pub fn nTrees(b: *const Bundle) usize {
        return b.trees.len;
    }

    /// Scores per row: the class count under softmax, else 1. Row-major.
    pub fn width(b: *const Bundle) usize {
        return b.objective.width(b.num_class);
    }

    /// Predictions on the objective's natural scale.
    pub fn predict(b: *const Bundle, pool: *Pool, ds: *const data.Dataset, out: []f32) void {
        std.debug.assert(out.len == ds.n_rows * b.width());
        switch (b.kind) {
            .gbdt, .forest => {
                var ctx = TreeCtx{ .b = b, .ds = ds, .out = out };
                pool.parallelFor(ds.n_rows, &ctx, TreeCtx.run, 2048);
            },
            .linear => b.lin.?.predict(pool, ds, out),
        }
    }
};

const TreeCtx = struct {
    b: *const Bundle,
    ds: *const data.Dataset,
    out: []f32,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *TreeCtx = @ptrCast(@alignCast(ctx));
        const b = self.b;
        const inv: f32 = if (b.trees.len == 0) 0 else 1.0 / @as(f32, @floatFromInt(b.trees.len));
        const k = b.width();
        if (k > 1) {
            // Softmax: each tree adds to its class's column, as `booster.Model` predicts.
            const out = self.out[begin * k .. end * k];
            for (begin..end) |r| @memcpy(out[(r - begin) * k ..][0..k], b.class_base);
            for (b.trees, 0..) |t, i| {
                const c = i % k;
                for (begin..end) |r| out[(r - begin) * k + c] += t.predictBinned(self.ds, r);
            }
            for (begin..end) |r| booster.softmaxRow(out[(r - begin) * k ..][0..k]);
            return;
        }
        const out = self.out[begin..end];

        // Trees outer, rows inner, and the per-row `switch (b.kind)` hoisted
        // out of the hot loop entirely. Bit-exact with the row-major order:
        // every `out[r]` accumulates the same trees in the same sequence,
        // rounding to f32 at each step exactly as a register accumulator did.
        @memset(out, if (b.kind == .gbdt) b.base_score else 0);
        for (b.trees) |t| {
            for (out, begin..) |*o, r| o.* += t.predictBinned(self.ds, r);
        }
        switch (b.kind) {
            .gbdt => if (b.objective == .logistic) {
                for (out) |*o| o.* = booster.sigmoid(o.*);
            },
            .forest => for (out) |*o| {
                o.* *= inv;
            },
            .linear => unreachable,
        }
    }
};

// ------------------------------------------------------------------ writing

const Buf = std.ArrayList(u8);

fn putU8(gpa: std.mem.Allocator, b: *Buf, v: u8) !void {
    try b.append(gpa, v);
}
fn putU16(gpa: std.mem.Allocator, b: *Buf, v: u16) !void {
    try b.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToLittle(u16, v)));
}
fn putU32(gpa: std.mem.Allocator, b: *Buf, v: u32) !void {
    try b.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToLittle(u32, v)));
}
fn putU64(gpa: std.mem.Allocator, b: *Buf, v: u64) !void {
    try b.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToLittle(u64, v)));
}
fn putF32(gpa: std.mem.Allocator, b: *Buf, v: f32) !void {
    try putU32(gpa, b, @bitCast(v));
}
fn putBytes(gpa: std.mem.Allocator, b: *Buf, s: []const u8) !void {
    try putU32(gpa, b, @intCast(s.len));
    try b.appendSlice(gpa, s);
}

/// Cursor over a loaded file. Every read is bounds-checked, so a truncated or
/// corrupt model fails with an error rather than reading past the buffer.
const Reader = struct {
    buf: []const u8,
    pos: usize = 0,

    fn take(r: *Reader, n: usize) ![]const u8 {
        if (r.pos + n > r.buf.len) return error.TruncatedModel;
        defer r.pos += n;
        return r.buf[r.pos..][0..n];
    }
    fn u8v(r: *Reader) !u8 {
        return (try r.take(1))[0];
    }
    fn u16v(r: *Reader) !u16 {
        return std.mem.littleToNative(u16, std.mem.bytesToValue(u16, try r.take(2)));
    }
    fn u32v(r: *Reader) !u32 {
        return std.mem.littleToNative(u32, std.mem.bytesToValue(u32, try r.take(4)));
    }
    fn u64v(r: *Reader) !u64 {
        return std.mem.littleToNative(u64, std.mem.bytesToValue(u64, try r.take(8)));
    }
    fn f32v(r: *Reader) !f32 {
        return @bitCast(try r.u32v());
    }
    /// An element count the rest of the file can hold at `min_bytes` per element.
    /// Callers allocate from the count before reading the elements, so an unchecked
    /// corrupt count (up to 4G) commits gigabytes before any read can fail.
    fn count(r: *Reader, min_bytes: usize) !u32 {
        std.debug.assert(min_bytes > 0);
        const n = try r.u32v();
        if (@as(usize, n) * min_bytes > r.buf.len - r.pos) return error.TruncatedModel;
        return n;
    }
    fn bytes(r: *Reader) ![]const u8 {
        const n = try r.u32v();
        return r.take(n);
    }
};

fn writeSchema(gpa: std.mem.Allocator, b: *Buf, s: *const data.Schema) !void {
    try putU32(gpa, b, @intCast(s.n_features));
    for (0..s.n_features) |f| {
        try putBytes(gpa, b, s.names[f]);
        try putU8(gpa, b, @intFromEnum(s.kinds[f]));
        try putU16(gpa, b, s.n_bins[f]);
        try putU32(gpa, b, @intCast(s.edges[f].len));
        for (s.edges[f]) |e| try putF32(gpa, b, e);
        try putU32(gpa, b, @intCast(s.levels[f].len));
        for (s.levels[f]) |l| try putBytes(gpa, b, l);
    }
}

fn readSchema(gpa: std.mem.Allocator, r: *Reader) !data.Schema {
    const n = try r.count(4 + 1 + 2 + 4 + 4); // name len, kind, bins, edge and level counts
    var s = data.Schema{
        .gpa = gpa,
        .n_features = n,
        .names = try gpa.alloc([]u8, n),
        .kinds = try gpa.alloc(data.ColumnKind, n),
        .n_bins = try gpa.alloc(u16, n),
        .edges = try gpa.alloc([]f32, n),
        .levels = try gpa.alloc([][]u8, n),
    };
    @memset(s.names, &.{});
    @memset(s.edges, &.{});
    @memset(s.levels, &.{});
    @memset(s.kinds, .numeric);
    @memset(s.n_bins, 0);
    errdefer s.deinit();

    for (0..n) |f| {
        s.names[f] = try gpa.dupe(u8, try r.bytes());
        s.kinds[f] = std.enums.fromInt(data.ColumnKind, try r.u8v()) orelse return error.BadModelFile;
        s.n_bins[f] = try r.u16v();
        const ne = try r.count(4);
        const e = try gpa.alloc(f32, ne);
        s.edges[f] = e;
        for (e) |*x| x.* = try r.f32v();
        const nl = try r.count(4); // each level is at least its length prefix
        const ls = try gpa.alloc([]u8, nl);
        @memset(ls, &.{});
        s.levels[f] = ls;
        for (ls) |*l| l.* = try gpa.dupe(u8, try r.bytes());
        // Binning a table counts bins from the edges or levels, but sizes storage from
        // n_bins (`splitByWidth`), so n_bins must cover the highest bin they produce.
        // That also keeps each bin within the u16 that `applySchema` narrows it into.
        const top: usize = switch (s.kinds[f]) {
            .numeric => @as(usize, ne) + 1,
            .categorical => nl,
        };
        if (s.n_bins[f] <= top) return error.BadModelFile;
    }
    return s;
}

fn writeTrees(gpa: std.mem.Allocator, b: *Buf, trees: []const tree.Tree) !void {
    try putU32(gpa, b, @intCast(trees.len));
    for (trees) |t| {
        try putU32(gpa, b, @intCast(t.nodes.len));
        for (t.nodes) |n| {
            try putU32(gpa, b, n.feature);
            try putU32(gpa, b, n.left);
            try putU32(gpa, b, n.right);
            try putF32(gpa, b, n.weight);
            try putU16(gpa, b, n.threshold);
            try putU8(gpa, b, @intFromBool(n.missing_left));
            try putU8(gpa, b, @intFromBool(n.is_leaf));
            try putU8(gpa, b, @intFromBool(n.is_cat));
            try putU8(gpa, b, n.n_cat);
            try putU32(gpa, b, n.cat_ofs);
            try putU8(gpa, b, n.n_lin);
            try putU32(gpa, b, n.lin_ofs);
            try putU8(gpa, b, @intFromEnum(n.kind));
        }
        // After the nodes, the tree's two side tables that nodes index into:
        // the categorical level ids (`cat_ofs`/`n_cat`), then the linear-leaf
        // terms (`lin_ofs`/`n_lin`). `readTrees` branches on the version for
        // the fields and tables older formats lack or encode differently.
        try putU32(gpa, b, @intCast(t.cat_ids.len));
        for (t.cat_ids) |id| try putU16(gpa, b, id);
        try putU32(gpa, b, @intCast(t.lin.len));
        for (t.lin) |term| {
            try putU32(gpa, b, term.feature);
            try putF32(gpa, b, term.coef);
            try putF32(gpa, b, term.center);
        }
        // Version 5: the combination statistics `combo` nodes index.
        try putU32(gpa, b, @intCast(t.combos.len));
        for (t.combos) |c| {
            try putU32(gpa, b, @intCast(c.parts.len));
            for (c.parts) |part| {
                try putU8(gpa, b, @intFromEnum(part.kind));
                try putU32(gpa, b, part.feature);
                try putU16(gpa, b, part.value);
            }
            try putU32(gpa, b, @intCast(c.keys.len));
            for (c.keys, c.buckets) |k, bk| {
                try putU64(gpa, b, k);
                try putU8(gpa, b, bk);
            }
            try putU8(gpa, b, c.unseen);
        }
    }
}

/// Version 3 wrote a 256-bit mask per categorical split and a node pointed at
/// it by word offset. Version 4 stores the ids on the left instead. Expanding
/// on load rather than keeping two prediction paths is what lets the reader be
/// the only place that knows version 3 ever existed.
fn expandV3Masks(
    gpa: std.mem.Allocator,
    r: *Reader,
    n_words: u32,
    nodes: []tree.Node,
    t: *tree.Tree,
) !void {
    const words = try gpa.alloc(u64, n_words);
    defer gpa.free(words);
    for (words) |*w| {
        const lo: u64 = try r.u32v();
        const hi: u64 = try r.u32v();
        w.* = lo | (hi << 32);
    }
    const v3_words: usize = 4; // 256 bins
    var ids: std.ArrayList(data.BinIdx) = .empty;
    errdefer ids.deinit(gpa);
    for (nodes) |*n| {
        if (!n.is_cat) continue;
        if (n.cat_ofs + v3_words > n_words) return error.BadModelFile;
        const m = words[n.cat_ofs..][0..v3_words];
        const start = ids.items.len;
        // Bit 0 was the missing bin in version 3; it is `missing_left` now.
        n.missing_left = (m[0] & 1) != 0;
        var bin: usize = 1;
        while (bin < 256) : (bin += 1) {
            if ((m[bin >> 6] >> @intCast(bin & 63)) & 1 == 0) continue;
            if (ids.items.len - start >= split.max_cat_ids) return error.BadModelFile;
            try ids.append(gpa, @intCast(bin));
        }
        n.cat_ofs = @intCast(start);
        n.n_cat = @intCast(ids.items.len - start);
    }
    t.cat_ids = try ids.toOwnedSlice(gpa);
}

fn readTrees(gpa: std.mem.Allocator, r: *Reader, ver: u32, n_features: usize) ![]tree.Tree {
    // Per node: feature, left, right, weight, two flags, the threshold (u16 from
    // version 4, u8 before), then what each version added.
    const node_bytes: usize = 16 + 2 + @as(usize, if (ver >= 4) 2 else 1) +
        @as(usize, if (ver >= 3) 10 else 0) + @as(usize, if (ver >= 4) 1 else 0) +
        @as(usize, if (ver >= 5) 1 else 0);
    const nt = try r.count(if (ver >= 3) 12 else 4); // node count, then cat-id (v3: mask-word) and term counts
    const trees = try gpa.alloc(tree.Tree, nt);
    var made: usize = 0;
    errdefer {
        for (trees[0..made]) |*t| t.deinit(gpa);
        gpa.free(trees);
    }
    while (made < nt) {
        const nn = try r.count(node_bytes);
        if (nn == 0) return error.BadModelFile;
        const nodes = try gpa.alloc(tree.Node, nn);
        trees[made] = .{ .nodes = nodes };
        // Count it as owned *before* the reads that can fail. Incrementing at
        // the end of the loop instead leaves this tree outside the errdefer's
        // `trees[0..made]` window, so a truncated file leaks it.
        made += 1;
        for (nodes, 0..) |*n, i| {
            // Read into locals rather than into a struct literal. The fields
            // are not declared in wire order, and a literal's initialisers
            // are not guaranteed to run in the order they are written -- so
            // side-effecting reads inside one silently reorder the file.
            const feature = try r.u32v();
            const left = try r.u32v();
            const right = try r.u32v();
            const weight = try r.f32v();
            const threshold: data.BinIdx = if (ver >= 4) try r.u16v() else try r.u8v();
            const missing_left = (try r.u8v()) != 0;
            const is_leaf = (try r.u8v()) != 0;
            const is_cat = if (ver >= 3) (try r.u8v()) != 0 else false;
            const n_cat = if (ver >= 4) try r.u8v() else 0;
            const cat_ofs = if (ver >= 3) try r.u32v() else 0;
            const n_lin = if (ver >= 3) try r.u8v() else 0;
            const lin_ofs = if (ver >= 3) try r.u32v() else 0;
            const kind = if (ver >= 5) std.enums.fromInt(tree.NodeKind, try r.u8v()) orelse return error.BadModelFile else .split;
            n.* = .{
                .feature = feature,
                .left = left,
                .right = right,
                .weight = weight,
                .cat_ofs = cat_ofs,
                .threshold = threshold,
                .missing_left = missing_left,
                .is_leaf = is_leaf,
                .is_cat = is_cat,
                .n_cat = n_cat,
                .n_lin = n_lin,
                .lin_ofs = lin_ofs,
                .kind = kind,
            };
            // A corrupt child or feature index would walk off the node array or the
            // row during prediction, and a child at or above its parent could loop
            // forever. The builder appends children after their parent, so every
            // real child index is past its parent's.
            // A combo node's `feature` indexes the combo table, checked once it is read.
            if (!n.is_leaf and (n.left <= i or n.right <= i or n.left >= nn or n.right >= nn or
                (n.kind == .split and n.feature >= n_features))) return error.BadModelFile;
        }
        if (ver >= 3) {
            const nm = try r.count(if (ver >= 4) 2 else 8); // u16 ids, or u64 mask words
            if (nm != 0) {
                if (ver >= 4) {
                    const ids = try gpa.alloc(data.BinIdx, nm);
                    trees[made - 1].cat_ids = ids;
                    for (ids) |*id| id.* = try r.u16v();
                } else {
                    // Version 3 stored a 256-bit mask per split. Expand each
                    // into the list form so the rest of the loader, and every
                    // prediction path, sees one representation.
                    try expandV3Masks(gpa, r, nm, nodes, &trees[made - 1]);
                }
            }
            const nl = try r.count(12);
            if (nl != 0) {
                const terms = try gpa.alloc(tree.LinTerm, nl);
                trees[made - 1].lin = terms;
                for (terms) |*term| {
                    const feature = try r.u32v();
                    const coef = try r.f32v();
                    const center = try r.f32v();
                    if (feature >= n_features) return error.BadModelFile;
                    term.* = .{ .feature = feature, .coef = coef, .center = center };
                }
            }
            // A node's `cat_ofs` range past `cat_ids`, or `lin_ofs` range past the
            // term table, would read out of bounds at prediction time: the same
            // class of fault as a bad child index, rejected the same way.
            const n_ids = trees[made - 1].cat_ids.len;
            for (nodes) |n| {
                if (n.is_cat and n.cat_ofs + n.n_cat > n_ids) return error.BadModelFile;
                if (n.n_lin != 0 and n.lin_ofs + n.n_lin > nl) return error.BadModelFile;
            }
        }
        if (ver >= 5) try readCombos(gpa, r, n_features, &trees[made - 1]);
        for (nodes) |n| {
            if (!n.is_leaf and n.kind == .combo and n.feature >= trees[made - 1].combos.len) return error.BadModelFile;
        }
    }
    return trees;
}

/// A tree's combination statistics (version 5). Parts must name real columns and keys must be
/// strictly ascending, or a lookup could misread; both are checked here, not at prediction.
fn readCombos(gpa: std.mem.Allocator, r: *Reader, n_features: usize, t: *tree.Tree) !void {
    const nc = try r.count(9); // parts count, keys count, unseen
    if (nc == 0) return;
    const combos = try gpa.alloc(tree.Combo, nc);
    var made: usize = 0;
    // Owned by the tree from here, so the caller's errdefer frees whatever was made.
    t.combos = combos[0..0];
    errdefer {
        for (combos[0..made]) |*c| c.deinit(gpa);
        gpa.free(combos);
        t.combos = &.{};
    }
    while (made < nc) : (made += 1) {
        const np = try r.count(7);
        const parts = try gpa.alloc(tree.Part, np);
        errdefer gpa.free(parts);
        for (parts) |*part| {
            const kind = std.enums.fromInt(tree.PartKind, try r.u8v()) orelse return error.BadModelFile;
            const feature = try r.u32v();
            const value = try r.u16v();
            if (feature >= n_features) return error.BadModelFile;
            part.* = .{ .kind = kind, .feature = feature, .value = value };
        }
        const nk = try r.count(9);
        const keys = try gpa.alloc(u64, nk);
        errdefer gpa.free(keys);
        const buckets = try gpa.alloc(u8, nk);
        errdefer gpa.free(buckets);
        for (keys, buckets, 0..) |*k, *bk, i| {
            k.* = try r.u64v();
            bk.* = try r.u8v();
            if (i != 0 and keys[i - 1] >= k.*) return error.BadModelFile;
        }
        const unseen = try r.u8v();
        combos[made] = .{ .parts = parts, .keys = keys, .buckets = buckets, .unseen = unseen };
    }
    t.combos = combos;
}

/// Serialise to a byte buffer the caller owns.
pub fn serialise(gpa: std.mem.Allocator, b: *const Bundle) ![]u8 {
    var buf: Buf = .empty;
    errdefer buf.deinit(gpa);

    try buf.appendSlice(gpa, magic);
    try putU32(gpa, &buf, format_version);
    try putU8(gpa, &buf, @intFromEnum(b.kind));
    try putU8(gpa, &buf, @intFromEnum(b.objective));
    try putF32(gpa, &buf, b.base_score);
    try putU32(gpa, &buf, b.num_class);
    for (b.class_base) |v| try putF32(gpa, &buf, v);
    try putBytes(gpa, &buf, b.label);
    try putU32(gpa, &buf, @intCast(b.classes.len));
    for (b.classes) |c| try putBytes(gpa, &buf, c);
    try writeSchema(gpa, &buf, &b.schema);

    switch (b.kind) {
        .gbdt, .forest => try writeTrees(gpa, &buf, b.trees),
        .linear => {
            const l = &b.lin.?;
            try putF32(gpa, &buf, l.intercept);
            // Softmax: each class's intercept; `w` then holds one block per class.
            for (l.intercepts) |v| try putF32(gpa, &buf, v);
            try putU32(gpa, &buf, @intCast(l.w.len));
            for (l.w) |c| try putF32(gpa, &buf, c);
            try putU32(gpa, &buf, @intCast(l.design.cols.len));
            for (l.design.cols) |c| {
                try putU32(gpa, &buf, c.feature);
                try putU16(gpa, &buf, c.bin);
                try putF32(gpa, &buf, c.center);
                try putF32(gpa, &buf, c.scale);
            }
            try putU32(gpa, &buf, @intCast(l.design.repr.len));
            for (l.design.repr) |t| {
                try putU32(gpa, &buf, @intCast(t.len));
                for (t) |v| try putF32(gpa, &buf, v);
            }
        },
    }
    return buf.toOwnedSlice(gpa);
}

pub fn deserialise(gpa: std.mem.Allocator, bytes: []const u8) !Bundle {
    var r = Reader{ .buf = bytes };
    if (!std.mem.eql(u8, try r.take(4), magic)) return error.NotAModelFile;
    const ver = try r.u32v();
    if (ver == 0 or ver > format_version) return error.UnsupportedModelVersion;

    const kind = std.enums.fromInt(Kind, try r.u8v()) orelse return error.BadModelFile;
    const obj = std.enums.fromInt(Objective, try r.u8v()) orelse return error.BadModelFile;
    const base = try r.f32v();
    var num_class: u32 = 1;
    var class_base: []f32 = &.{};
    errdefer if (class_base.len != 0) gpa.free(class_base);
    if (ver >= 6) {
        num_class = try r.u32v();
        if ((obj == .softmax) != (num_class >= 2) or num_class == 0) return error.BadModelFile;
        // Boosted trees start each class from a stored score; the linear model keeps its
        // intercepts with its coefficients.
        if (num_class > 1 and kind == .gbdt) {
            if (num_class > r.buf.len) return error.BadModelFile;
            class_base = try gpa.alloc(f32, num_class);
            for (class_base) |*v| v.* = try r.f32v();
        }
    } else if (obj == .softmax) return error.BadModelFile;

    var label: []u8 = &.{};
    errdefer if (label.len != 0) gpa.free(label);
    var classes: [][]u8 = &.{};
    var n_classes: usize = 0;
    errdefer {
        for (classes[0..n_classes]) |c| gpa.free(c);
        if (classes.len != 0) gpa.free(classes);
    }
    if (ver >= 2) {
        label = try gpa.dupe(u8, try r.bytes());
        const nc = try r.u32v();
        // Softmax names every class; otherwise a string target has two.
        if (if (obj == .softmax) nc != 0 and nc != num_class else nc > 2) return error.BadModelFile;
        classes = try gpa.alloc([]u8, nc);
        while (n_classes < nc) : (n_classes += 1) classes[n_classes] = try gpa.dupe(u8, try r.bytes());
    }

    var schema = try readSchema(gpa, &r);
    errdefer schema.deinit();

    var b = Bundle{
        .gpa = gpa,
        .kind = kind,
        .schema = schema,
        .objective = obj,
        .base_score = base,
        .num_class = num_class,
        .class_base = class_base,
        .label = label,
        .classes = classes,
    };

    switch (kind) {
        .gbdt, .forest => {
            b.trees = try readTrees(gpa, &r, ver, b.schema.n_features);
            if (kind != .gbdt and num_class != 1) return error.BadModelFile;
            if (b.trees.len % num_class != 0) return error.BadModelFile;
        },
        .linear => {
            const intercept = try r.f32v();
            const intercepts = try gpa.alloc(f32, if (num_class > 1) num_class else 0);
            errdefer gpa.free(intercepts);
            for (intercepts) |*v| v.* = try r.f32v();
            const nw = try r.count(4);
            const w = try gpa.alloc(f32, nw);
            errdefer gpa.free(w);
            for (w) |*x| x.* = try r.f32v();

            const nc = try r.count(4 + 2 + 4 + 4);
            if (nc * num_class != nw) return error.BadModelFile;
            const cols = try gpa.alloc(linear.Col, nc);
            errdefer gpa.free(cols);
            for (cols) |*c| c.* = .{
                .feature = try r.u32v(),
                .bin = try r.u16v(),
                .center = try r.f32v(),
                .scale = try r.f32v(),
            };

            const nr = try r.count(4);
            const repr = try gpa.alloc([]f32, nr);
            @memset(repr, &.{});
            errdefer {
                for (repr) |t| if (t.len != 0) gpa.free(t);
                gpa.free(repr);
            }
            for (repr) |*t| {
                const n = try r.count(4);
                const tab = try gpa.alloc(f32, n);
                t.* = tab;
                for (tab) |*v| v.* = try r.f32v();
            }
            // Scoring indexes the row by a column's feature and a numeric column's
            // table by bin, unchecked.
            for (cols) |c| {
                if (c.feature >= b.schema.n_features) return error.BadModelFile;
                if (c.bin == linear.numeric_col and
                    (c.feature >= repr.len or repr[c.feature].len < b.schema.n_bins[c.feature])) return error.BadModelFile;
            }

            b.lin = .{
                .gpa = gpa,
                .design = .{ .gpa = gpa, .cols = cols, .repr = repr },
                .w = w,
                .intercept = intercept,
                .objective = obj,
                .n_features = b.schema.n_features,
                .num_class = num_class,
                .intercepts = intercepts,
            };
        },
    }
    return b;
}

pub fn save(gpa: std.mem.Allocator, io: std.Io, path: []const u8, b: *const Bundle) !void {
    const bytes = try serialise(gpa, b);
    defer gpa.free(bytes);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
}

pub fn load(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Bundle {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, std.Io.Limit.limited(1 << 31));
    defer gpa.free(bytes);
    return deserialise(gpa, bytes);
}

// ----------------------------------------------------------------- blending

/// Weighted average of several models' predictions on the same rows.
///
/// Every kind predicts on the natural scale, so probabilities blend with
/// probabilities and values with values, and a forest can be mixed with a
/// booster without either knowing about the other. Weights are normalised, so
/// passing 1,1,1 is a plain mean.
pub fn blend(
    gpa: std.mem.Allocator,
    pool: *Pool,
    bundles: []const *const Bundle,
    weights: []const f32,
    ds: *const data.Dataset,
    out: []f32,
) !void {
    if (bundles.len == 0) return error.NoModels;
    if (weights.len != bundles.len) return error.WeightCountMismatch;

    var total: f64 = 0;
    for (weights) |w| {
        if (w < 0) return error.NegativeWeight;
        total += w;
    }
    if (total <= 0) return error.WeightsSumToZero;

    // Probabilities blend column by column, so every model must predict the same columns.
    const cols = bundles[0].width();
    for (bundles) |b| if (b.width() != cols) return error.BlendWidthMismatch;
    std.debug.assert(out.len == ds.n_rows * cols);
    @memset(out, 0);
    const scratch = try gpa.alloc(f32, ds.n_rows * cols);
    defer gpa.free(scratch);

    for (bundles, weights) |b, w| {
        if (w == 0) continue;
        b.predict(pool, ds, scratch);
        const k: f32 = @floatCast(@as(f64, w) / total);
        for (out, scratch) |*o, s| o.* += k * s;
    }
}

// -------------------------------------------------------------- constructors

pub fn fromBooster(gpa: std.mem.Allocator, m: *const booster.Model, schema: data.Schema) !Bundle {
    const class_base = try gpa.dupe(f32, m.class_base);
    errdefer gpa.free(class_base);
    return .{
        .gpa = gpa,
        .kind = .gbdt,
        .schema = schema,
        .objective = m.objective,
        .base_score = m.base_score,
        .num_class = m.num_class,
        .class_base = class_base,
        .trees = try dupeTrees(gpa, m.trees.items),
    };
}

pub fn fromForest(gpa: std.mem.Allocator, m: *const forest.Forest, schema: data.Schema) !Bundle {
    return .{
        .gpa = gpa,
        .kind = .forest,
        .schema = schema,
        .objective = m.objective,
        .trees = try dupeTrees(gpa, m.trees.items),
    };
}

fn dupeTrees(gpa: std.mem.Allocator, src: []const tree.Tree) ![]tree.Tree {
    const out = try gpa.alloc(tree.Tree, src.len);
    var made: usize = 0;
    errdefer {
        for (out[0..made]) |*t| t.deinit(gpa);
        gpa.free(out);
    }
    while (made < src.len) {
        out[made] = .{ .nodes = try gpa.dupe(tree.Node, src[made].nodes) };
        // Owned before the second allocation, so a failure there frees the
        // nodes rather than leaking them.
        made += 1;
        if (src[made - 1].cat_ids.len != 0)
            out[made - 1].cat_ids = try gpa.dupe(data.BinIdx, src[made - 1].cat_ids);
        if (src[made - 1].lin.len != 0)
            out[made - 1].lin = try gpa.dupe(tree.LinTerm, src[made - 1].lin);
        if (src[made - 1].combos.len != 0) {
            var list: std.ArrayList(tree.Combo) = .empty;
            errdefer {
                for (list.items) |*c| c.deinit(gpa);
                list.deinit(gpa);
            }
            for (src[made - 1].combos) |*c| {
                var copy = try c.dupe(gpa);
                errdefer copy.deinit(gpa);
                try list.append(gpa, copy);
            }
            out[made - 1].combos = try list.toOwnedSlice(gpa);
        }
    }
    return out;
}
