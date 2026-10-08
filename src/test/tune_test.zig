// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

const std = @import("std");
const config = @import("../config.zig");
const data = @import("../data.zig");
const pool_mod = @import("../pool.zig");
const cv = @import("../cv.zig");
const tune = @import("../tune.zig");
const Binner = tune.Binner;
const defaultSpace = tune.defaultSpace;
const parseParam = tune.parseParam;
const Kind = tune.Kind;
const Param = tune.Param;
const Trial = tune.Trial;
const Tpe = tune.Tpe;

const testing = std.testing;

test "a choice spec keeps its values and its order" {
    const p = try parseParam(testing.allocator, "max_depth=4,5,6");
    defer testing.allocator.free(p.choices);
    try testing.expectEqual(Kind.choice, p.kind);
    try testing.expectEqual(@as(usize, 3), p.choices.len);
    try testing.expectEqualStrings("4", p.choices[0]);
    try testing.expectEqualStrings("6", p.choices[2]);
}

test "range specs pick up their int and log suffixes in either order" {
    const a = try parseParam(testing.allocator, "lambda=0.1..50");
    try testing.expectEqual(Kind.uniform, a.kind);
    const b = try parseParam(testing.allocator, "lambda=0.1..50:log");
    try testing.expectEqual(Kind.log_uniform, b.kind);
    const c = try parseParam(testing.allocator, "n_rounds=200..800:int");
    try testing.expectEqual(Kind.int_uniform, c.kind);
    const d = try parseParam(testing.allocator, "n_rounds=200..800:int:log");
    try testing.expectEqual(Kind.int_log, d.kind);
    const e = try parseParam(testing.allocator, "n_rounds=200..800:log:int");
    try testing.expectEqual(Kind.int_log, e.kind);
}

test "a spec naming no config field is rejected" {
    try testing.expectError(error.UnknownConfigField, parseParam(testing.allocator, "not_a_field=1,2"));
    try testing.expectError(error.EmptyParamRange, parseParam(testing.allocator, "lambda=50..1"));
    try testing.expectError(error.LogRangeNeedsPositiveLow, parseParam(testing.allocator, "lambda=0..50:log"));
}

test "a log axis is searched geometrically, not linearly" {
    const p = try parseParam(testing.allocator, "lambda=1..100:log");
    const pts = try p.gridPoints(testing.allocator, 3);
    defer testing.allocator.free(pts);
    var buf: [64]u8 = undefined;
    // Encoded midpoint of log(1)..log(100) is log(10), so the middle lattice
    // point must be 10 -- a linear axis would have put 50.5 there.
    const mid = try p.render(pts[1], &buf);
    try testing.expectEqualStrings("10.000000", mid);
}

test "rendered values round-trip through the config flag parser" {
    var cfg: config.Config = .{};
    const p = try parseParam(testing.allocator, "max_depth=3..9:int");
    var buf: [64]u8 = undefined;
    const s = try p.render(7.4, &buf);
    try testing.expect(try config.applyFlag(&cfg, p.name, s));
    try testing.expectEqual(@as(u32, 7), cfg.gbdt.tree.max_depth);
}

test "grid points span the whole axis inclusively" {
    const p = try parseParam(testing.allocator, "subsample=0.5..1.0");
    const pts = try p.gridPoints(testing.allocator, 5);
    defer testing.allocator.free(pts);
    try testing.expectEqual(@as(usize, 5), pts.len);
    try testing.expectApproxEqAbs(@as(f64, 0.5), pts[0], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), pts[4], 1e-12);
}

test "sampling stays inside the declared bounds" {
    var prng: std.Random.DefaultPrng = .init(4);
    const r = prng.random();
    const specs = [_][]const u8{
        "lambda=0.1..50:log",
        "subsample=0.5..1.0",
        "n_rounds=200..800:int",
        "max_depth=4,5,6",
    };
    for (specs) |s| {
        const p = try parseParam(testing.allocator, s);
        defer if (p.kind == .choice) testing.allocator.free(p.choices);
        const b = p.bounds();
        for (0..200) |_| {
            const v = p.sample(r);
            try testing.expect(v >= b.lo - 1e-9 and v <= b.hi + 1e-9);
        }
    }
}

test "TPE proposes where good outnumbers bad, not merely where it has looked" {
    const gpa = testing.allocator;
    const p = try parseParam(gpa, "subsample=0.0..1.0");
    const space = [_]Param{p};

    // Twenty observations whose score rises with the parameter. A proposal
    // drawn from the good density must land in the upper part of the range;
    // a sampler ignoring the scores would average out near the middle.
    var trials: std.ArrayList(Trial) = .empty;
    defer trials.deinit(gpa);
    var xs: [20][1]f64 = undefined;
    for (&xs, 0..) |*slot, i| {
        slot[0] = @as(f64, @floatFromInt(i)) / 19.0;
        try trials.append(gpa, .{
            .x = slot,
            .score = slot[0],
            .folds = 5,
            .ms = 1,
            .text = "",
        });
    }

    var prng: std.Random.DefaultPrng = .init(9);
    const tpe = Tpe{ .gamma = 0.25, .candidates = 32 };
    var hits: usize = 0;
    var outv: [1]f64 = undefined;
    for (0..40) |_| {
        try tpe.propose(gpa, prng.random(), &space, trials.items, .auc, &outv);
        if (outv[0] > 0.6) hits += 1;
    }
    try testing.expect(hits > 30);
}

test "the default space covers every model" {
    inline for (.{ config.Algo.gbdt, .random_forest, .linear }) |a| {
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        const s = try defaultSpace(arena.allocator(), a);
        try testing.expect(s.len >= 4);
        // Every default axis must name a real field, or `tune` would fail
        // only once someone ran it without --param.
        for (s) |p| try testing.expect(config.hasField(p.name));
    }
}

test "a best value in the outer 5% of a range, or on an end choice, is at an edge" {
    const gpa = testing.allocator;
    const p = try parseParam(gpa, "max_bin=8..16384:int:log");
    const b = p.bounds();
    const span = b.hi - b.lo;
    try testing.expectEqual(.low, p.atEdge(b.lo).?);
    try testing.expectEqual(.low, p.atEdge(b.lo + span * 0.04).?);
    try testing.expect(p.atEdge(b.lo + span * 0.06) == null);
    try testing.expect(p.atEdge(b.lo + span * 0.5) == null);
    try testing.expectEqual(.high, p.atEdge(b.hi - span * 0.04).?);

    const c = try parseParam(gpa, "max_depth=3,4,5");
    defer gpa.free(c.choices);
    try testing.expectEqual(.low, c.atEdge(0).?);
    try testing.expect(c.atEdge(1) == null);
    try testing.expectEqual(.high, c.atEdge(2).?);
    // Non-numeric or two-way choices have no ends to widen.
    const t = try parseParam(gpa, "lin_standardize=true,false");
    defer gpa.free(t.choices);
    try testing.expect(t.atEdge(0) == null);
    const two = try parseParam(gpa, "max_depth=3,4");
    defer gpa.free(two.choices);
    try testing.expect(two.atEdge(0) == null);
}

test "TPE keeps exploring when the good set collapses onto one point" {
    const gpa = testing.allocator;
    const p = try parseParam(gpa, "lambda=0.1..50:log");
    const space = [_]Param{p};

    // Every observation identical, and at the top of the range: the pathology
    // that produced seven repeated trials with lambda pinned to 50. A Parzen
    // estimate with no prior has zero density everywhere else, so the ratio
    // cannot prefer anywhere else and the proposal freezes.
    var trials: std.ArrayList(Trial) = .empty;
    defer trials.deinit(gpa);
    var xs: [24][1]f64 = undefined;
    for (&xs, 0..) |*slot, i| {
        slot[0] = p.bounds().hi;
        try trials.append(gpa, .{
            .x = slot,
            .score = if (i < 6) 0.99 else 0.10,
            .folds = 5,
            .ms = 1,
            .text = "",
        });
    }

    var prng: std.Random.DefaultPrng = .init(21);
    const tpe = Tpe{ .gamma = 0.25, .candidates = 32 };
    var seen: std.ArrayList(f64) = .empty;
    defer seen.deinit(gpa);
    var outv: [1]f64 = undefined;
    for (0..60) |_| {
        try tpe.propose(gpa, prng.random(), &space, trials.items, .auc, &outv);
        try seen.append(gpa, outv[0]);
    }

    var distinct: usize = 0;
    for (seen.items, 0..) |v, i| {
        var dup = false;
        for (seen.items[0..i]) |w| if (@abs(v - w) < 1e-9) {
            dup = true;
        };
        if (!dup) distinct += 1;
    }
    // Before the prior was added this was exactly 1.
    try testing.expect(distinct > 5);
}

test "TPE does not pile proposals onto a boundary" {
    const gpa = testing.allocator;
    const p = try parseParam(gpa, "subsample=0.5..1.0");
    const space = [_]Param{p};

    // Good points sit near the top edge, so jitter pushes past it and gets
    // clamped. Clamping alone would stack proposals exactly on the bound.
    var trials: std.ArrayList(Trial) = .empty;
    defer trials.deinit(gpa);
    var xs: [20][1]f64 = undefined;
    for (&xs, 0..) |*slot, i| {
        slot[0] = if (i < 5) 0.99 else 0.55;
        try trials.append(gpa, .{
            .x = slot,
            .score = if (i < 5) 0.9 else 0.1,
            .folds = 5,
            .ms = 1,
            .text = "",
        });
    }

    var prng: std.Random.DefaultPrng = .init(5);
    const tpe = Tpe{ .gamma = 0.25, .candidates = 32 };
    var on_bound: usize = 0;
    var outv: [1]f64 = undefined;
    for (0..60) |_| {
        try tpe.propose(gpa, prng.random(), &space, trials.items, .auc, &outv);
        if (@abs(outv[0] - 1.0) < 1e-12) on_bound += 1;
    }
    try testing.expect(on_bound < 30);
}

// ----- regression tests

/// Two categorical columns and a numeric target, built directly so the test
/// stays hermetic. `wide` controls how many distinct levels the categorical
/// carries, which is what decides the `max_bin` it can be binned under.
fn testFrame(gpa: std.mem.Allocator, wide: usize, n_rows: usize) !data.Frame {
    var levels_list: std.ArrayList([]u8) = .empty;
    errdefer levels_list.deinit(gpa);
    const cat = try gpa.alloc(f32, n_rows);
    const num = try gpa.alloc(f32, n_rows);
    for (0..wide) |i| {
        var buf: [16]u8 = undefined;
        try levels_list.append(gpa, try gpa.dupe(u8, try std.fmt.bufPrint(&buf, "L{d}", .{i})));
    }
    for (cat, num, 0..) |*c, *y, i| {
        c.* = @floatFromInt(i % wide);
        y.* = @floatFromInt(i % 7);
    }
    const names = try gpa.alloc([]u8, 2);
    names[0] = try gpa.dupe(u8, "cat");
    names[1] = try gpa.dupe(u8, "y");
    const kinds = try gpa.alloc(data.ColumnKind, 2);
    kinds[0] = .categorical;
    kinds[1] = .numeric;
    const vals = try gpa.alloc([]f32, 2);
    vals[0] = cat;
    vals[1] = num;
    const levels = try gpa.alloc([][]u8, 2);
    levels[0] = try levels_list.toOwnedSlice(gpa);
    levels[1] = &.{};
    return .{
        .gpa = gpa,
        .n_rows = n_rows,
        .names = names,
        .kinds = kinds,
        .values = vals,
        .levels = levels,
    };
}

test "Binner: a rebin too narrow for a categorical leaves the cached matrix intact" {
    // A `max_cat_levels` below the widest categorical column cannot be binned
    // at all, and a search space that offers it as a choice will propose one.
    // That has to be a skipped trial, not a failure.
    //
    // This used to be provoked through `max_bin`, which doubled as the
    // categorical width limit. The two are separate now -- `max_bin` cuts
    // numeric columns and has nothing to say about how many levels a
    // categorical has -- so the same path is reached through the knob that
    // actually governs it.
    //
    // It used to be fatal, and the escaping error then ran BOTH
    // `defer binner.ds.deinit()` and a stale `errdefer full.deinit()` over one
    // allocation: a double free, and a use-after-free once a successful rebin
    // had already swapped the matrix out from under the stale handle.
    // ReleaseFast segfaulted inside `free`; Debug aborted there. Only
    // `--search=bandit` surfaced it, because it is the strategy that reliably
    // walks the whole `max_bin` choice set early.
    //
    // `testing.allocator` fails this test on a double free or a leak, so the
    // ownership half is asserted by construction.
    const gpa = testing.allocator;
    const p = try pool_mod.Pool.init(gpa, 2);
    defer p.deinit();

    var frame = try testFrame(gpa, 40, 200);
    defer frame.deinit();

    // A real encoder, not `undefined`. The label column is numeric, so an
    // empty class list is the right one -- and `undefined` here meant the test
    // read uninitialised memory inside `encode`, which happened to be benign
    // until an unrelated allocation shifted underneath it.
    var numeric_enc = data.LabelEncoder{ .gpa = gpa, .classes = &.{} };

    var b = Binner{
        .gpa = gpa,
        .pool = p,
        .frame = &frame,
        .label_col = 1,
        // A real encoder, not `undefined`. The column is numeric, so an empty
        // class list is the right one -- and `undefined` here meant the test
        // was reading uninitialised memory in `encode`, which happened to be
        // benign until an unrelated allocation shifted underneath it.
        .enc = &numeric_enc,
        .drops = &.{},
        .ds = try data.quantise(gpa, p, &frame, .{ .max_bin = 64 }, null, &.{}),
        .bin = .{ .max_bin = 64 },
    };
    defer b.ds.deinit();

    const before = b.ds.n_rows;
    try testing.expect(before == 200);

    // 40 levels. A limit of 32 cannot admit them.
    try testing.expectError(
        error.CategoricalTooWide,
        b.get(config.Config.from(.{ .max_bin = 64, .max_cat_levels = 32 })),
    );

    // The cache must be untouched: same rows, and still the width we loaded.
    try testing.expectEqual(@as(u16, 64), b.bin.max_bin);
    try testing.expectEqual(before, b.ds.n_rows);

    // And a later valid request must still rebin normally.
    const ds = try b.get(config.Config.from(.{ .max_bin = 128 }));
    try testing.expectEqual(before, ds.n_rows);
    try testing.expectEqual(@as(usize, 1), b.rebins);
}

test "Binner: every binning parameter re-bins, including min_data_in_bin" {
    // The binner used to compare a hand-kept list of fields (`max_bin`,
    // `bin_policy`, `max_cat_levels`) and missed `min_data_in_bin`: a search
    // over it under `greedy` printed values it never applied. Each field of
    // `BinParams` is changed alone here and must cost exactly one re-bin.
    const gpa = testing.allocator;
    const p = try pool_mod.Pool.init(gpa, 2);
    defer p.deinit();
    var frame = try testFrame(gpa, 10, 300);
    defer frame.deinit();
    var numeric_enc = data.LabelEncoder{ .gpa = gpa, .classes = &.{} };

    const base: data.BinParams = .{ .bin_policy = .greedy };
    var b = Binner{
        .gpa = gpa,
        .pool = p,
        .frame = &frame,
        .label_col = 1,
        .enc = &numeric_enc,
        .drops = &.{},
        .ds = try data.quantise(gpa, p, &frame, base, null, &.{}),
        .bin = base,
    };
    defer b.ds.deinit();

    // The same parameters: no re-bin.
    _ = try b.get(config.Config.from(.{ .bin_policy = .greedy }));
    try testing.expectEqual(@as(usize, 0), b.rebins);

    _ = try b.get(config.Config.from(.{ .bin_policy = .greedy, .min_data_in_bin = 50 }));
    try testing.expectEqual(@as(usize, 1), b.rebins);
    try testing.expectEqual(@as(u32, 50), b.bin.min_data_in_bin);

    _ = try b.get(config.Config.from(.{ .bin_policy = .greedy, .min_data_in_bin = 50, .max_bin = 32 }));
    try testing.expectEqual(@as(usize, 2), b.rebins);

    _ = try b.get(config.Config.from(.{ .min_data_in_bin = 50, .max_bin = 32 }));
    try testing.expectEqual(@as(usize, 3), b.rebins);

    _ = try b.get(config.Config.from(.{ .min_data_in_bin = 50, .max_bin = 32, .max_cat_levels = 100 }));
    try testing.expectEqual(@as(usize, 4), b.rebins);
}

test "a flag pins its axis out of the default space, and pinning a --param axis is caught" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const full = try defaultSpace(a, .gbdt);
    const kept = try tune.withoutPinned(a, full, &.{ "n_rounds", "max_bin", "not_in_space" });
    try testing.expectEqual(full.len - 2, kept.len);
    for (kept) |p| {
        try testing.expect(!std.mem.eql(u8, p.name, "n_rounds"));
        try testing.expect(!std.mem.eql(u8, p.name, "max_bin"));
    }
    // Order of the rest is kept.
    try testing.expectEqualStrings("learning_rate", kept[0].name);
    try testing.expectEqual(full.len, (try tune.withoutPinned(a, full, &.{})).len);

    const searched = [_]Param{ try parseParam(a, "lambda=0.1..10:log"), try parseParam(a, "max_depth=3,4") };
    try testing.expectEqualStrings("max_depth", tune.pinnedAndSearched(&searched, &.{ "n_rounds", "max_depth" }).?);
    try testing.expect(tune.pinnedAndSearched(&searched, &.{"n_rounds"}) == null);
}

test "cat_split joins the default space only for categorical data, and parses" {
    try testing.expect(tune.hasCategorical(&.{ .numeric, .categorical }));
    try testing.expect(!tune.hasCategorical(&.{ .numeric, .numeric }));
    try testing.expect(!tune.hasCategorical(&.{}));
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const p = try parseParam(arena.allocator(), tune.cat_split_axis);
    try testing.expect(config.hasField(p.name));
    try testing.expectEqual(@as(usize, 2), p.choices.len);
}
