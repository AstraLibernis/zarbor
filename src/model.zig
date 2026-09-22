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
const config = @import("config.zig");
const tree = @import("tree.zig");
const hist = @import("hist.zig");
const booster = @import("booster.zig");
const forest = @import("forest.zig");
const linear = @import("linear.zig");
const Pool = @import("pool.zig").Pool;

pub const magic = "ZMDL";
/// 2 added the label encoding. A version-1 file still loads; it simply has
/// no class order, which the scoring path rejects rather than guessing at.
/// 3 added categorical subset splits. A version-2 file still loads: it has no
/// mask store and no node claiming one, which is exactly what an ordinal-split
/// model is.
pub const format_version: u32 = 3;

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
    objective: config.Objective,
    base_score: f32 = 0,
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

    /// Predictions on the objective's natural scale.
    pub fn predict(b: *const Bundle, pool: *Pool, ds: *const data.Dataset, out: []f32) void {
        std.debug.assert(out.len == ds.n_rows);
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
        var r = begin;
        while (r < end) : (r += 1) {
            var acc: f32 = if (b.kind == .gbdt) b.base_score else 0;
            for (b.trees) |t| acc += t.predictBinned(self.ds, r);
            self.out[r] = switch (b.kind) {
                .gbdt => if (b.objective == .logistic) booster.sigmoid(acc) else acc,
                .forest => acc * inv,
                .linear => unreachable,
            };
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
    fn f32v(r: *Reader) !f32 {
        return @bitCast(try r.u32v());
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
    const n = try r.u32v();
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
        const ne = try r.u32v();
        const e = try gpa.alloc(f32, ne);
        s.edges[f] = e;
        for (e) |*x| x.* = try r.f32v();
        const nl = try r.u32v();
        const ls = try gpa.alloc([]u8, nl);
        @memset(ls, &.{});
        s.levels[f] = ls;
        for (ls) |*l| l.* = try gpa.dupe(u8, try r.bytes());
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
            try putU8(gpa, b, n.threshold);
            try putU8(gpa, b, @intFromBool(n.missing_left));
            try putU8(gpa, b, @intFromBool(n.is_leaf));
            try putU8(gpa, b, @intFromBool(n.is_cat));
            try putU32(gpa, b, n.cat_ofs);
            try putU8(gpa, b, n.n_lin);
            try putU32(gpa, b, n.lin_ofs);
        }
        // Mask store last, so the node loop above stays the same shape as the
        // version-2 one and the two readers differ only in what they skip.
        try putU32(gpa, b, @intCast(t.cat_masks.len));
        for (t.cat_masks) |w| {
            try putU32(gpa, b, @truncate(w));
            try putU32(gpa, b, @truncate(w >> 32));
        }
        try putU32(gpa, b, @intCast(t.lin.len));
        for (t.lin) |term| {
            try putU32(gpa, b, term.feature);
            try putF32(gpa, b, term.coef);
            try putF32(gpa, b, term.center);
        }
    }
}

fn readTrees(gpa: std.mem.Allocator, r: *Reader, ver: u32) ![]tree.Tree {
    const nt = try r.u32v();
    const trees = try gpa.alloc(tree.Tree, nt);
    var made: usize = 0;
    errdefer {
        for (trees[0..made]) |*t| t.deinit(gpa);
        gpa.free(trees);
    }
    while (made < nt) {
        const nn = try r.u32v();
        if (nn == 0) return error.BadModelFile;
        const nodes = try gpa.alloc(tree.Node, nn);
        trees[made] = .{ .nodes = nodes };
        // Count it as owned *before* the reads that can fail. Incrementing at
        // the end of the loop instead leaves this tree outside the errdefer's
        // `trees[0..made]` window, so a truncated file leaks it.
        made += 1;
        for (nodes) |*n| {
            // Read into locals rather than into a struct literal. The fields
            // are not declared in wire order, and a literal's initialisers
            // are not guaranteed to run in the order they are written -- so
            // side-effecting reads inside one silently reorder the file.
            const feature = try r.u32v();
            const left = try r.u32v();
            const right = try r.u32v();
            const weight = try r.f32v();
            const threshold = try r.u8v();
            const missing_left = (try r.u8v()) != 0;
            const is_leaf = (try r.u8v()) != 0;
            const is_cat = if (ver >= 3) (try r.u8v()) != 0 else false;
            const cat_ofs = if (ver >= 3) try r.u32v() else 0;
            const n_lin = if (ver >= 3) try r.u8v() else 0;
            const lin_ofs = if (ver >= 3) try r.u32v() else 0;
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
                .n_lin = n_lin,
                .lin_ofs = lin_ofs,
            };
            // A corrupt child index would walk off the node array during
            // prediction; reject it here instead.
            if (!n.is_leaf and (n.left >= nn or n.right >= nn)) return error.BadModelFile;
        }
        if (ver >= 3) {
            const nm = try r.u32v();
            if (nm != 0) {
                const masks = try gpa.alloc(u64, nm);
                trees[made - 1].cat_masks = masks;
                for (masks) |*w| {
                    const lo: u64 = try r.u32v();
                    const hi: u64 = try r.u32v();
                    w.* = lo | (hi << 32);
                }
            }
            // A mask offset past the store would read out of bounds at
            // prediction time, which is the same class of fault as a bad
            // child index and is rejected the same way.
            const nl = try r.u32v();
            if (nl != 0) {
                const terms = try gpa.alloc(tree.LinTerm, nl);
                trees[made - 1].lin = terms;
                for (terms) |*term| {
                    const feature = try r.u32v();
                    const coef = try r.f32v();
                    const center = try r.f32v();
                    term.* = .{ .feature = feature, .coef = coef, .center = center };
                }
            }
            for (nodes) |n| {
                if (n.is_cat and n.cat_ofs + hist.cat_words > nm) return error.BadModelFile;
                if (n.n_lin != 0 and n.lin_ofs + n.n_lin > nl) return error.BadModelFile;
            }
        }
    }
    return trees;
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
    try putBytes(gpa, &buf, b.label);
    try putU32(gpa, &buf, @intCast(b.classes.len));
    for (b.classes) |c| try putBytes(gpa, &buf, c);
    try writeSchema(gpa, &buf, &b.schema);

    switch (b.kind) {
        .gbdt, .forest => try writeTrees(gpa, &buf, b.trees),
        .linear => {
            const l = &b.lin.?;
            try putF32(gpa, &buf, l.intercept);
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
    const obj = std.enums.fromInt(config.Objective, try r.u8v()) orelse return error.BadModelFile;
    const base = try r.f32v();

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
        if (nc > 2) return error.BadModelFile;
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
        .label = label,
        .classes = classes,
    };

    switch (kind) {
        .gbdt, .forest => b.trees = try readTrees(gpa, &r, ver),
        .linear => {
            const intercept = try r.f32v();
            const nw = try r.u32v();
            const w = try gpa.alloc(f32, nw);
            errdefer gpa.free(w);
            for (w) |*x| x.* = try r.f32v();

            const nc = try r.u32v();
            if (nc != nw) return error.BadModelFile;
            const cols = try gpa.alloc(linear.Col, nc);
            errdefer gpa.free(cols);
            for (cols) |*c| c.* = .{
                .feature = try r.u32v(),
                .bin = try r.u16v(),
                .center = try r.f32v(),
                .scale = try r.f32v(),
            };

            const nr = try r.u32v();
            const repr = try gpa.alloc([]f32, nr);
            @memset(repr, &.{});
            errdefer {
                for (repr) |t| if (t.len != 0) gpa.free(t);
                gpa.free(repr);
            }
            for (repr) |*t| {
                const n = try r.u32v();
                const tab = try gpa.alloc(f32, n);
                t.* = tab;
                for (tab) |*v| v.* = try r.f32v();
            }

            b.lin = .{
                .gpa = gpa,
                .design = .{ .gpa = gpa, .cols = cols, .repr = repr },
                .w = w,
                .intercept = intercept,
                .objective = obj,
                .n_features = b.schema.n_features,
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

    @memset(out, 0);
    const scratch = try gpa.alloc(f32, ds.n_rows);
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
    return .{
        .gpa = gpa,
        .kind = .gbdt,
        .schema = schema,
        .objective = m.objective,
        .base_score = m.base_score,
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
        if (src[made - 1].cat_masks.len != 0)
            out[made - 1].cat_masks = try gpa.dupe(u64, src[made - 1].cat_masks);
        if (src[made - 1].lin.len != 0)
            out[made - 1].lin = try gpa.dupe(tree.LinTerm, src[made - 1].lin);
    }
    return out;
}
