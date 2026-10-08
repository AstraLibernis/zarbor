// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Multiclass (softmax) boosting. Tree-for-tree parity with XGBoost `multi:softprob` and LightGBM
//! `multiclass` is checked outside the unit tests (`bench/parity/parity.py`); these pin what a
//! wrong change would break silently: which class a tree adds to, whole rounds under early
//! stopping, thread independence, the file round trip, label encoding and fold stratification.

const std = @import("std");
const data = @import("../data.zig");
const config = @import("../config.zig");
const booster = @import("../booster.zig");
const metric = @import("../metric.zig");
const model_mod = @import("../model.zig");
const cv = @import("../cv.zig");
const csv = @import("../csv.zig");
const Pool = @import("../pool.zig").Pool;
const fromBins = @import("split_test.zig").fromBins;

const testing = std.testing;

const n_rows = 2400;
const n_class = 4;

/// Four 8-level features and four overlapping classes of unequal size: each class is likeliest
/// where its own feature is high, plus noise.
fn fixture(gpa: std.mem.Allocator, seed: u64) !data.Dataset {
    var cols: [4][n_rows]u8 = undefined;
    var y: [n_rows]f32 = undefined;
    var prng: std.Random.DefaultPrng = .init(seed);
    const r = prng.random();
    for (0..n_rows) |i| {
        var best: usize = 0;
        var best_s: f32 = -std.math.inf(f32);
        for (&cols, 0..) |*c, f| {
            c[i] = r.intRangeAtMost(u8, 1, 8);
            const s = @as(f32, @floatFromInt(c[i])) * (0.3 + 0.1 * @as(f32, @floatFromInt(f))) + r.floatNorm(f32) * 1.2;
            if (s > best_s) {
                best_s = s;
                best = f;
            }
        }
        y[i] = @floatFromInt(best);
    }
    return fromBins(gpa, &.{ &cols[0], &cols[1], &cols[2], &cols[3] }, &.{ 9, 9, 9, 9 }, &y);
}

fn params(opts: anytype) booster.Params {
    var cfg = config.Config.from(.{ .objective = .softmax, .verbose_eval = 0 }).gbdt;
    cfg.num_class = n_class;
    inline for (@typeInfo(@TypeOf(opts)).@"struct".fields) |f| {
        if (@hasField(booster.Params, f.name)) @field(cfg, f.name) = @field(opts, f.name) else @field(cfg.tree, f.name) = @field(opts, f.name);
    }
    return cfg;
}

test "softmax: the same model to the bit at 1, 3 and 16 threads, and probabilities sum to one" {
    const gpa = testing.allocator;
    var ds = try fixture(gpa, 1);
    defer ds.deinit();
    const p = params(.{ .n_rounds = 15, .max_depth = 4, .subsample = 0.8, .colsample_bytree = 0.75, .seed = 9 });

    var want: [n_rows * n_class]f32 = undefined;
    for ([_]u32{ 1, 3, 16 }) |threads| {
        const pool = try Pool.init(gpa, threads);
        defer pool.deinit();
        var res = try booster.train(gpa, pool, &ds, null, p, null);
        defer res.model.deinit();
        var raw: [n_rows * n_class]f32 = undefined;
        res.model.predictRaw(pool, &ds, &raw);
        if (threads == 1) want = raw else try testing.expectEqualSlices(f32, &want, &raw);

        var prob: [n_rows * n_class]f32 = undefined;
        res.model.predict(pool, &ds, &prob);
        for (0..n_rows) |r| {
            var sum: f32 = 0;
            for (prob[r * n_class ..][0..n_class]) |v| {
                try testing.expect(v > 0 and v < 1);
                sum += v;
            }
            try testing.expectApproxEqAbs(@as(f32, 1), sum, 1e-5);
        }
    }
}

test "softmax: tree i adds only to class i mod K, from that class's starting score" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();
    var ds = try fixture(gpa, 2);
    defer ds.deinit();
    var res = try booster.train(gpa, pool, &ds, null, params(.{ .n_rounds = 6, .max_depth = 3 }), null);
    defer res.model.deinit();
    const m = &res.model;
    try testing.expectEqual(@as(usize, 6 * n_class), m.trees.items.len);
    try testing.expectEqual(@as(u32, 6), res.n_rounds);

    var raw: [n_rows * n_class]f32 = undefined;
    m.predictRaw(pool, &ds, &raw);
    for (0..n_rows) |r| for (0..n_class) |c| {
        var want = m.class_base[c];
        for (m.trees.items, 0..) |t, i| if (i % n_class == c) {
            want += t.predictBinned(&ds, r);
        };
        try testing.expectEqual(want, raw[r * n_class + c]);
    };

    // The starting scores are the centred log class shares (XGBoost's `InitEstimation`).
    var counts = [_]f32{0} ** n_class;
    for (ds.labels) |y| counts[@intFromFloat(y)] += 1;
    var logs: [n_class]f32 = undefined;
    var mean: f32 = 0;
    for (&logs, counts) |*l, cnt| {
        l.* = @log(cnt / @as(f32, n_rows));
        mean += l.*;
    }
    mean /= n_class;
    for (logs, m.class_base) |l, b| try testing.expectApproxEqAbs(l - mean, b, 1e-6);
}

test "softmax: the two hessian forms agree at two classes and differ above" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();
    var ds = try fixture(gpa, 3);
    defer ds.deinit();
    var raws: [2][n_rows * n_class]f32 = undefined;
    for ([_]booster.SoftmaxHessian{ .xgboost, .lightgbm }, &raws) |h, *out| {
        var res = try booster.train(gpa, pool, &ds, null, params(.{ .n_rounds = 4, .max_depth = 3, .softmax_hessian = h }), null);
        defer res.model.deinit();
        res.model.predictRaw(pool, &ds, out);
    }
    try testing.expect(!std.mem.eql(f32, &raws[0], &raws[1]));

    // Two classes: K / (K - 1) = 2 equals XGBoost's factor; only f32 vs f64 arithmetic differs.
    for (ds.labels) |*y| y.* = if (y.* >= 2) 1 else 0;
    var two: [2][n_rows * 2]f32 = undefined;
    for ([_]booster.SoftmaxHessian{ .xgboost, .lightgbm }, &two) |h, *out| {
        var p = params(.{ .n_rounds = 4, .max_depth = 3, .softmax_hessian = h });
        p.num_class = 2;
        var res = try booster.train(gpa, pool, &ds, null, p, null);
        defer res.model.deinit();
        res.model.predictRaw(pool, &ds, out);
    }
    for (two[0], two[1]) |a, b| try testing.expectApproxEqAbs(a, b, 1e-4);
}

test "softmax: early stopping keeps whole rounds, all classes of the best one" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();
    var ds = try fixture(gpa, 4);
    defer ds.deinit();
    var valid = try fixture(gpa, 5);
    defer valid.deinit();
    // A high learning rate overfits quickly, so the stop fires well before the cap.
    var res = try booster.train(gpa, pool, &ds, &valid, params(.{ .n_rounds = 300, .max_depth = 6, .learning_rate = 0.8, .early_stopping_rounds = 5 }), null);
    defer res.model.deinit();
    try testing.expect(res.rounds_run < 300);
    try testing.expectEqual(@as(usize, res.n_rounds) * n_class, res.model.trees.items.len);
    try testing.expect(res.n_rounds + 5 == res.rounds_run);

    // The kept model scores the best log loss it reported.
    var raw: [n_rows * n_class]f32 = undefined;
    res.model.predictRaw(pool, &valid, &raw);
    try testing.expectApproxEqAbs(res.best_score, metric.mlogloss(&raw, valid.labels, n_class), 1e-9);
}

test "softmax: a saved model predicts the same to the bit after loading" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();
    var ds = try fixture(gpa, 6);
    defer ds.deinit();
    var res = try booster.train(gpa, pool, &ds, null, params(.{ .n_rounds = 5, .max_depth = 4 }), null);
    defer res.model.deinit();

    var schema = try data.Schema.fromDataset(gpa, &ds);
    errdefer schema.deinit();
    var b = try model_mod.fromBooster(gpa, &res.model, schema);
    defer b.deinit();
    try testing.expectEqual(@as(usize, n_class), b.width());

    var before: [n_rows * n_class]f32 = undefined;
    b.predict(pool, &ds, &before);
    var direct: [n_rows * n_class]f32 = undefined;
    res.model.predict(pool, &ds, &direct);
    try testing.expectEqualSlices(f32, &direct, &before);

    const bytes = try model_mod.serialise(gpa, &b);
    defer gpa.free(bytes);
    var loaded = try model_mod.deserialise(gpa, bytes);
    defer loaded.deinit();
    try testing.expectEqual(@as(u32, n_class), loaded.num_class);
    var after: [n_rows * n_class]f32 = undefined;
    loaded.predict(pool, &ds, &after);
    try testing.expectEqualSlices(f32, &before, &after);

    // A softmax file claiming one class, or a binary one claiming four, is corrupt.
    // The class count follows magic (4), version (4), kind (1), objective (1) and base score (4).
    const at = 14;
    try testing.expectEqual(@as(u32, n_class), std.mem.readInt(u32, bytes[at..][0..4], .little));
    const bad = try gpa.dupe(u8, bytes);
    defer gpa.free(bad);
    @memcpy(bad[at..][0..4], &std.mem.toBytes(std.mem.nativeToLittle(u32, 1)));
    try testing.expectError(error.BadModelFile, model_mod.deserialise(gpa, bad));
}

test "softmax is refused where it is not built yet" {
    const ok = params(.{});
    try ok.validate();
    var p = ok;
    p.num_class = 1;
    try testing.expectError(error.SoftmaxNeedsClasses, p.validate());
    p = ok;
    p.tree.grow_policy = .symmetric;
    try testing.expectError(error.SoftmaxSymmetricUnsupported, p.validate());
    p = ok;
    p.sampling = .goss;
    try testing.expectError(error.SoftmaxGossUnsupported, p.validate());
    p = ok;
    p.tree.linear_leaves = true;
    try testing.expectError(error.SoftmaxLinearLeavesUnsupported, p.validate());
    p = ok;
    p.scale_pos_weight = 2;
    try testing.expectError(error.ScalePosWeightNeedsBinary, p.validate());
    for ([_]config.Algo{ .random_forest, .linear }) |algo| {
        var c = config.Config.from(.{ .algo = algo, .objective = .softmax });
        c.gbdt.num_class = 3;
        try testing.expectError(error.SoftmaxNeedsGbdt, c.validate());
    }
}

/// A frame of one label column: numbers, or strings when `levels` is given (values are ids).
fn labelFrame(gpa: std.mem.Allocator, values: []const f32, levels: []const []const u8) !csv.Frame {
    const names = try gpa.alloc([]u8, 1);
    names[0] = try gpa.dupe(u8, "y");
    const kinds = try gpa.alloc(data.ColumnKind, 1);
    kinds[0] = if (levels.len == 0) .numeric else .categorical;
    const vals = try gpa.alloc([]f32, 1);
    vals[0] = try gpa.dupe(f32, values);
    const lv = try gpa.alloc([][]u8, 1);
    lv[0] = try gpa.alloc([]u8, levels.len);
    for (lv[0], levels) |*d, s| d.* = try gpa.dupe(u8, s);
    return .{ .gpa = gpa, .n_rows = values.len, .names = names, .kinds = kinds, .values = vals, .levels = lv };
}

test "softmax labels: classes sorted, integers numerically, and unusable targets refused" {
    const gpa = testing.allocator;
    {
        // Integers order as numbers ("10" after "2"); every value is a class.
        var f = try labelFrame(gpa, &.{ 10, 2, 7, 2, 10 }, &.{});
        defer f.deinit();
        var enc = try data.LabelEncoder.forObjective(gpa, &f, 0, null, .softmax);
        defer enc.deinit();
        try testing.expectEqual(@as(usize, 3), enc.classes.len);
        try testing.expectEqualStrings("2", enc.classes[0]);
        try testing.expectEqualStrings("10", enc.classes[2]);
        const y = try enc.encode(gpa, &f, 0);
        defer gpa.free(y);
        try testing.expectEqualSlices(f32, &.{ 2, 0, 1, 0, 2 }, y);
        try enc.validate(y, .softmax);
        try testing.expectError(error.LabelOutOfRange, enc.validate(&.{ 0, 3 }, .softmax));
    }
    {
        // Strings sort; ids in the frame follow first appearance and do not matter.
        var f = try labelFrame(gpa, &.{ 0, 1, 2, 0 }, &.{ "virginica", "setosa", "versicolor" });
        defer f.deinit();
        var enc = try data.LabelEncoder.forObjective(gpa, &f, 0, null, .softmax);
        defer enc.deinit();
        const y = try enc.encode(gpa, &f, 0);
        defer gpa.free(y);
        try testing.expectEqualSlices(f32, &.{ 2, 0, 1, 2 }, y);
        try testing.expectError(error.PosLabelWithSoftmax, data.LabelEncoder.forObjective(gpa, &f, 0, "setosa", .softmax));
        // A holdout class the model never saw is an error, not a quiet zero.
        var g = try labelFrame(gpa, &.{0}, &.{"iris"});
        defer g.deinit();
        try testing.expectError(error.UnseenLabelClass, enc.encode(gpa, &g, 0));
    }
    {
        var f = try labelFrame(gpa, &.{ 0, 1.5, 2 }, &.{});
        defer f.deinit();
        try testing.expectError(error.NonIntegerClass, data.LabelEncoder.forObjective(gpa, &f, 0, null, .softmax));
        var one = try labelFrame(gpa, &.{ 3, 3, 3 }, &.{});
        defer one.deinit();
        try testing.expectError(error.SingleClassTarget, data.LabelEncoder.forObjective(gpa, &one, 0, null, .softmax));
    }
}

test "stratified folds hold every class in near-equal shares" {
    const gpa = testing.allocator;
    var ds = try fixture(gpa, 7);
    defer ds.deinit();
    const folds = 5;
    const fold_of = try cv.assignFolds(gpa, ds.labels, folds, 3, true);
    defer gpa.free(fold_of);
    var count = [_][n_class]u32{[_]u32{0} ** n_class} ** folds;
    var total = [_]u32{0} ** n_class;
    for (fold_of, ds.labels) |f, y| {
        count[f][@intFromFloat(y)] += 1;
        total[@intFromFloat(y)] += 1;
    }
    for (0..n_class) |c| {
        var lo: u32 = std.math.maxInt(u32);
        var hi: u32 = 0;
        for (count) |row| {
            lo = @min(lo, row[c]);
            hi = @max(hi, row[c]);
        }
        try testing.expect(total[c] > 50);
        try testing.expect(hi - lo <= 1);
    }
}

test "multiclass metrics: log loss from scores and from probabilities agree; accuracy" {
    const raw = [_]f32{ 2, 0, -1, 0.5, 0.5, 3, -2, 1, 0 };
    const y = [_]f32{ 0, 2, 1 };
    var prob = raw;
    for (0..3) |r| booster.softmaxRow(prob[r * 3 ..][0..3]);
    try testing.expectApproxEqAbs(metric.mlogloss(&raw, &y, 3), metric.mloglossProb(&prob, &y, 3), 1e-6);
    // Rows 0 and 1 are right; row 2's best is class 1, its label: 3 of 3. A tie takes the first.
    try testing.expectEqual(@as(f64, 1), metric.accuracy(&raw, &y, 3));
    try testing.expectEqual(@as(f64, 0), metric.accuracy(&.{ 1, 1 }, &.{1}, 2));
}
