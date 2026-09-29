// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! End-to-end regression tests for the three models.
//!
//! These exist because two silent correctness bugs shipped: a histogram stride
//! that only misbehaved where `cache_line / @sizeOf(Bin)` is not a power of two
//! (so it depended on the CPU), and a sibling-subtraction identity that broke
//! whenever per-level or per-node feature sampling was on. Both produced
//! plausible-looking runs rather than crashes, and neither was covered. Every
//! test below therefore asserts on *achieved accuracy*, not merely on "it ran".

const std = @import("std");
const data = @import("../data.zig");
const config = @import("../config.zig");
const booster = @import("../booster.zig");
const forest = @import("../forest.zig");
const linear = @import("../linear.zig");
const hist = @import("../hist.zig");
const tree = @import("../tree.zig");
const Pool = @import("../pool.zig").Pool;
const metric = @import("../metric.zig");

const testing = std.testing;

/// A binned dataset with a learnable signal, built directly rather than
/// through the CSV path so the tests stay hermetic.
fn synth(gpa: std.mem.Allocator, n_rows: usize, seed: u64) !data.Dataset {
    const n_features: usize = 6;
    const n_bin: u16 = 17; // bin 0 is missing; 1..16 are real

    const bins = try gpa.alloc(u8, n_features * n_rows);
    errdefer gpa.free(bins);
    const labels = try gpa.alloc(f32, n_rows);
    errdefer gpa.free(labels);

    var prng: std.Random.DefaultPrng = .init(seed);
    const r = prng.random();

    for (0..n_rows) |row| {
        var b: [6]u8 = undefined;
        for (&b) |*v| v.* = r.intRangeAtMost(u8, 1, 16);
        for (0..n_features) |f| bins[f * n_rows + row] = b[f];
        // Signal in three features; the other three are pure noise, which is
        // what makes feature sampling worth testing.
        var score: f32 = 0;
        if (b[0] > 8) score += 1.0;
        if (b[1] > 12) score += 1.0;
        if (b[2] < 4) score -= 1.0;
        score += r.floatNorm(f32) * 0.35;
        labels[row] = if (score > 0.5) 1.0 else 0.0;
    }

    const n_bins = try gpa.alloc(u16, n_features);
    errdefer gpa.free(n_bins);
    @memset(n_bins, n_bin);

    const kinds = try gpa.alloc(data.ColumnKind, n_features);
    errdefer gpa.free(kinds);
    @memset(kinds, .numeric);

    const levels = try gpa.alloc([][]u8, n_features);
    errdefer gpa.free(levels);
    @memset(levels, &.{});

    const edges = try gpa.alloc([]f32, n_features);
    errdefer gpa.free(edges);
    const names = try gpa.alloc([]u8, n_features);
    errdefer gpa.free(names);
    for (0..n_features) |f| {
        const e = try gpa.alloc(f32, n_bin - 2);
        for (e, 0..) |*v, i| v.* = @floatFromInt(i + 1);
        edges[f] = e;
        names[f] = try std.fmt.allocPrint(gpa, "f{d}", .{f});
    }

    const means = try gpa.alloc([]f32, n_features);
    errdefer gpa.free(means);
    @memset(means, &.{});

    // Mirror of `bins`, since the histogram kernel reads row-major.
    // Every fixture column is narrow, so the wide overrides are all empty.
    const wide_cols = try gpa.alloc([]data.BinIdx, n_features);
    errdefer gpa.free(wide_cols);
    @memset(wide_cols, &.{});

    const bins_rm = try gpa.alloc(data.BinIdx, n_features * n_rows);
    errdefer gpa.free(bins_rm);
    for (0..n_rows) |ri| for (0..n_features) |f| {
        bins_rm[ri * n_features + f] = bins[f * n_rows + ri];
    };

    return .{
        .gpa = gpa,
        .n_rows = n_rows,
        .n_features = n_features,
        .bins = bins,
        .wide_cols = wide_cols,
        .bins_rm = bins_rm,
        .n_bins = n_bins,
        .edges = edges,
        // Left empty on purpose: `buildDesign` must still work from edges
        // alone, which is the path a hand-built dataset takes.
        .means = means,
        .kinds = kinds,
        .names = names,
        .levels = levels,
        .labels = labels,
    };
}

fn aucOf(gpa: std.mem.Allocator, pred: []const f32, labels: []const f32) !f64 {
    return metric.auc(gpa, pred, labels);
}

test "packed offsets never truncate a feature and stay cache-line aligned" {
    // The original code sized every feature by `alignForward(max_bins,
    // cache_line / @sizeOf(Bin))`, which requires a power-of-two alignment it
    // does not have on every CPU. On a 128-byte line with a 24-byte Bin that
    // argument is 5, and the stride came out *below* max_bins, overlapping
    // adjacent features' histograms.
    //
    // It also sized *every* feature by the widest one. Real tables are uneven
    // — 234 bins for a continuous column next to 3 for a boolean — so that
    // wasted 5.6x the slots on the dataset this was measured against, and the
    // per-node clear and reduce paid for all of it.
    const gpa = testing.allocator;
    const widths = [_]u16{ 2, 3, 17, 47, 202, 234, 256, 257 };

    var bank = try hist.Bank.init(gpa, 2, widths.len, &widths);
    defer bank.deinit();

    var prev: u32 = 0;
    for (widths, 0..) |w, f| {
        const lo = bank.offsets[f];
        const hi = bank.offsets[f + 1];
        try testing.expectEqual(prev, lo); // packed: no gaps between features
        try testing.expect(hi - lo >= w); // never truncates
        // Each feature starts on a cache line, so the clear and the reduce
        // never straddle one.
        try testing.expectEqual(@as(usize, 0), (@as(usize, lo) * @sizeOf(hist.Bin)) % std.atomic.cache_line);
        prev = hi;
    }
    try testing.expectEqual(prev, bank.offsets[widths.len]);
    try testing.expectEqual(@as(usize, prev), bank.slotLen());

    // And the whole point, on the shape real tables actually have: a few wide
    // continuous columns beside several tiny categorical ones. These are the
    // measured bin counts of the dataset this was tuned against.
    const real = [_]u16{ 47, 234, 202, 6, 17, 22, 7, 4, 4, 5, 3, 3, 4 };
    var rb = try hist.Bank.init(gpa, 1, real.len, &real);
    defer rb.deinit();

    var maxw: usize = 0;
    for (real) |w| maxw = @max(maxw, w);
    const step = std.atomic.cache_line / std.math.gcd(@sizeOf(hist.Bin), std.atomic.cache_line);
    const uniform = real.len * ((maxw + step - 1) / step * step);
    // 688 against 3120 when this was written; assert a clear majority saved
    // rather than the exact figure, which depends on the cache line size.
    try testing.expect(rb.slotLen() * 3 < uniform);
}

test "gbdt and random forest both learn the signal" {
    // Against the shared fixture rather than two fresh fits. The assertion is
    // that the model found the structure `synth` put there, and that does not
    // need a model nobody else has looked at.
    const gpa = testing.allocator;
    const f = try shared();
    const pred = try gpa.alloc(f32, f.ds.n_rows);
    defer gpa.free(pred);

    f.gbdt.predict(f.pool, &f.ds, pred);
    try testing.expect(try aucOf(gpa, pred, f.ds.labels) > 0.90);

    f.forest.predict(f.pool, &f.ds, pred);
    try testing.expect(try aucOf(gpa, pred, f.ds.labels) > 0.90);
    // A forest averages votes, so its output is already a probability.
    for (pred) |p| try testing.expect(p >= 0.0 and p <= 1.0);
}

test "per-node feature sampling does not corrupt sibling subtraction" {
    // Regression test. With colsample_bynode < 1 the child histograms used to
    // be built over a different feature set than the parent they were
    // subtracted from, which drove AUC to ~0.73 and logloss into the hundreds.
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();

    var ds = try synth(gpa, 4000, 11);
    defer ds.deinit();

    var res = try booster.train(gpa, pool, &ds, null, .{
        .n_rounds = 60,
        .max_depth = 4,
        .colsample_bynode = 0.34,
        .colsample_bylevel = 0.5,
        .verbose_eval = 0,
    }, null);
    defer res.model.deinit();

    const pred = try gpa.alloc(f32, ds.n_rows);
    defer gpa.free(pred);
    res.model.predictRaw(pool, &ds, pred);

    try testing.expect(try aucOf(gpa, pred, ds.labels) > 0.90);
    // The real tell was the leaf weights, not the ranking.
    try testing.expect(metric.logloss(pred, ds.labels) < 1.0);
}

test "goss trains and keeps the raw scores of unsampled rows current" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();

    var ds = try synth(gpa, 6000, 13);
    defer ds.deinit();

    var res = try booster.train(gpa, pool, &ds, null, .{
        .n_rounds = 60,
        .max_depth = 4,
        .sampling = .goss,
        .top_rate = 0.2,
        .other_rate = 0.1,
        .verbose_eval = 0,
    }, null);
    defer res.model.deinit();

    const pred = try gpa.alloc(f32, ds.n_rows);
    defer gpa.free(pred);
    res.model.predictRaw(pool, &ds, pred);
    try testing.expect(try aucOf(gpa, pred, ds.labels) > 0.90);
}

test "the forest defaults pick sqrt(p) features and the fixture honours them" {
    // What this uniquely covered was `applyForestFeatureDefault`, not that a
    // forest can learn -- the fixture asserts that. Keeping the defaults check
    // and dropping the second fit of the same thing.
    var cfg: config.Config = .{ .algo = .random_forest, .n_rounds = 40, .verbose_eval = 0 };
    cfg.applyAlgoDefaults(&.{});
    cfg.applyForestFeatureDefault(9, &.{});
    try testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), cfg.colsample_bynode, 1e-6);
    // A forest averages unshrunk trees, so shrinkage must be off.
    try testing.expectEqual(@as(f32, 1.0), cfg.learning_rate);
}

test "linear model learns the signal and L1 drives coefficients to zero" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();

    var ds = try synth(gpa, 4000, 23);
    defer ds.deinit();

    var res = try linear.train(gpa, pool, &ds, null, .{
        .algo = .linear,
        .lin_epochs = 400,
        .verbose_eval = 0,
    }, null);
    defer res.model.deinit();

    const pred = try gpa.alloc(f32, ds.n_rows);
    defer gpa.free(pred);
    res.model.predict(pool, &ds, pred);
    try testing.expect(try aucOf(gpa, pred, ds.labels) > 0.80);

    // Heavy L1 should zero out the three noise features.
    var sparse = try linear.train(gpa, pool, &ds, null, .{
        .algo = .linear,
        .lin_epochs = 400,
        .alpha = 5000.0,
        .verbose_eval = 0,
    }, null);
    defer sparse.model.deinit();
    try testing.expect(sparse.model.nZero() > 0);
}

// --------------------------------------------------------- save / load / blend

const model_mod = @import("../model.zig");

/// Builds a Frame by hand so a test can control the dictionary order, which is
/// the whole point of the schema.
/// A two-column numeric frame: the column under test and a label.
fn numericFrame(gpa: std.mem.Allocator, col: []const f32, lab: []const f32) !data.Frame {
    const names = try gpa.alloc([]u8, 2);
    names[0] = try gpa.dupe(u8, "x");
    names[1] = try gpa.dupe(u8, "y");
    const kinds = try gpa.alloc(data.ColumnKind, 2);
    kinds[0] = .numeric;
    kinds[1] = .numeric;
    const values = try gpa.alloc([]f32, 2);
    values[0] = try gpa.dupe(f32, col);
    values[1] = try gpa.dupe(f32, lab);
    const levels = try gpa.alloc([][]u8, 2);
    levels[0] = &.{};
    levels[1] = &.{};
    return .{ .gpa = gpa, .n_rows = col.len, .names = names, .kinds = kinds, .values = values, .levels = levels };
}

fn frameWith(gpa: std.mem.Allocator, colours: []const []const u8, nums: []const f32) !data.Frame {
    const n = colours.len;
    var levels_list: std.ArrayList([]u8) = .empty;
    errdefer levels_list.deinit(gpa);
    const cat_vals = try gpa.alloc(f32, n);
    for (colours, cat_vals) |c, *v| {
        var id: ?usize = null;
        for (levels_list.items, 0..) |l, i| if (std.mem.eql(u8, l, c)) {
            id = i;
        };
        if (id == null) {
            try levels_list.append(gpa, try gpa.dupe(u8, c));
            id = levels_list.items.len - 1;
        }
        v.* = @floatFromInt(id.?);
    }

    const names = try gpa.alloc([]u8, 2);
    names[0] = try gpa.dupe(u8, "colour");
    names[1] = try gpa.dupe(u8, "num");
    const kinds = try gpa.alloc(data.ColumnKind, 2);
    kinds[0] = .categorical;
    kinds[1] = .numeric;
    const values = try gpa.alloc([]f32, 2);
    values[0] = cat_vals;
    values[1] = try gpa.dupe(f32, nums);
    const levels = try gpa.alloc([][]u8, 2);
    levels[0] = try levels_list.toOwnedSlice(gpa);
    levels[1] = &.{};

    return .{
        .gpa = gpa,
        .n_rows = n,
        .names = names,
        .kinds = kinds,
        .values = values,
        .levels = levels,
    };
}

test "applySchema maps categories by string, not by the new file's own ids" {
    // The trap this exists to catch: a categorical bin *is* its dictionary id,
    // and ids are assigned in order of first appearance. Here the second file
    // introduces the same two levels in the opposite order, so binning it on
    // its own would map "red" and "blue" to each other's bins and silently
    // score a different model input.
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();

    var train = try frameWith(gpa, &.{ "red", "blue", "red", "green" }, &.{ 1, 2, 3, 4 });
    defer train.deinit();
    var train_ds = try data.quantise(gpa, pool, &train, .{}, null, &.{});
    defer train_ds.deinit();

    var schema = try data.Schema.fromDataset(gpa, &train_ds);
    defer schema.deinit();

    // Opposite order of first appearance.
    var test_f = try frameWith(gpa, &.{ "green", "blue", "red", "mauve" }, &.{ 4, 3, 2, 1 });
    defer test_f.deinit();
    var test_ds = try data.applySchema(gpa, pool, &test_f, &schema, null);
    defer test_ds.deinit();

    const cat = test_ds.columnNarrow(0);
    const train_cat = train_ds.columnNarrow(0);
    // red is bin 1 and blue bin 2 in training; the test file must agree.
    try testing.expectEqual(train_cat[0], cat[2]); // red
    try testing.expectEqual(train_cat[1], cat[1]); // blue
    try testing.expectEqual(train_cat[3], cat[0]); // green
    // A level never seen in training falls into the missing bin.
    try testing.expectEqual(@as(u8, 0), cat[3]); // mauve
}

// ---------------------------------------------------------------- fixtures
//
// One dataset and one model of each kind, trained once for the whole file.
//
// Every test used to build its own pool, its own data and its own model: 22
// trainings across 25 tests, in a Debug build, which is where nearly all of
// the suite's wall clock went. Most of those tests do not care how the model
// was fitted -- they serialise it, blend it, corrupt its bytes, or read its
// nodes. Those want *a* model, not a fresh one.
//
// Held in `std.heap.page_allocator` rather than `testing.allocator` on
// purpose: the fixture outlives every test, and `testing.allocator` would
// correctly report that as a leak. Nothing here is ever freed, which is what
// a process-lifetime fixture is.
const Fixture = struct {
    pool: *Pool,
    ds: data.Dataset,
    gbdt: model_mod.Bundle,
    forest: model_mod.Bundle,
    linear: model_mod.Bundle,
    lin_raw: linear.Linear,
};

var fixture: ?Fixture = null;

fn shared() !*const Fixture {
    if (fixture) |*f| return f;
    const gpa = std.heap.page_allocator;
    const pool = try Pool.init(gpa, 2);
    var ds = try synth(gpa, 3000, 31);

    var gb = try booster.train(gpa, pool, &ds, null, .{ .n_rounds = 25, .max_depth = 4, .verbose_eval = 0 }, null);
    const a = try model_mod.fromBooster(gpa, &gb.model, try data.Schema.fromDataset(gpa, &ds));
    gb.model.deinit();

    var cfg: config.Config = .{ .algo = .random_forest, .n_rounds = 15, .verbose_eval = 0 };
    cfg.applyAlgoDefaults(&.{});
    var rf = try forest.train(gpa, pool, &ds, null, cfg, null);
    const b = try model_mod.fromForest(gpa, &rf.model, try data.Schema.fromDataset(gpa, &ds));
    rf.model.deinit();

    const lr = try linear.train(gpa, pool, &ds, null, .{ .algo = .linear, .lin_epochs = 60, .verbose_eval = 0 }, null);
    const c = model_mod.Bundle{
        .gpa = gpa,
        .kind = .linear,
        .schema = try data.Schema.fromDataset(gpa, &ds),
        .objective = .logistic,
        .lin = lr.model,
    };

    fixture = .{ .pool = pool, .ds = ds, .gbdt = a, .forest = b, .linear = c, .lin_raw = lr.model };
    return &fixture.?;
}

fn roundTrip(kind: config.Algo) !void {
    const gpa = testing.allocator;
    const f = try shared();
    const bundle: *const model_mod.Bundle = switch (kind) {
        .gbdt => &f.gbdt,
        .random_forest => &f.forest,
        .linear => &f.linear,
    };

    const before = try gpa.alloc(f32, f.ds.n_rows);
    defer gpa.free(before);
    bundle.predict(f.pool, &f.ds, before);

    const bytes = try model_mod.serialise(gpa, bundle);
    defer gpa.free(bytes);
    var loaded = try model_mod.deserialise(gpa, bytes);
    defer loaded.deinit();

    const after = try gpa.alloc(f32, f.ds.n_rows);
    defer gpa.free(after);
    loaded.predict(f.pool, &f.ds, after);

    // Bit-identical, not approximately equal: a saved model that drifts is a
    // model whose submitted predictions do not match the ones you validated.
    for (before, after) |x, y| try testing.expectEqual(x, y);
    try testing.expectEqual(bundle.schema.n_features, loaded.schema.n_features);
}

/// `synth` with two of its columns relabelled categorical, so the subset-split
/// path has something to act on. The bins are unchanged -- a categorical bin
/// *is* its dictionary id, so relabelling is all it takes.
fn synthWithCats(gpa: std.mem.Allocator, n_rows: usize, seed: u64) !data.Dataset {
    var ds = try synth(gpa, n_rows, seed);
    ds.kinds[3] = .categorical;
    ds.kinds[4] = .categorical;
    return ds;
}

fn countCatNodes(b: *const model_mod.Bundle) usize {
    var n: usize = 0;
    for (b.trees) |t| for (t.nodes) |nd| {
        if (!nd.is_leaf and nd.is_cat) n += 1;
    };
    return n;
}

test "cat_l2 penalises the children and not the parent" {
    // The asymmetry is the whole point and it is easy to get wrong: adding
    // cat_l2 to the parent score as well reads as the self-consistent choice,
    // and cost 36% of the gain the categorical columns carry until LightGBM's
    // source settled it. See docs/vs-lightgbm.md.
    //
    // Pinned by consequence rather than by a magic number. With the penalty on
    // the children alone, a huge cat_l2 drives every categorical child score
    // to nothing while the parent's stays put, so no categorical split can
    // ever show a gain. Applied to both, the two move together and splits
    // survive.
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();
    var ds = try synthWithCats(gpa, 3000, 29);
    defer ds.deinit();

    var counts: [2]usize = undefined;
    for ([2]f32{ 0.0, 1e9 }, 0..) |l2, i| {
        var schema = try data.Schema.fromDataset(gpa, &ds);
        errdefer schema.deinit();
        var res = try booster.train(gpa, pool, &ds, null, .{
            .n_rounds = 15,
            .max_depth = 4,
            .cat_split = .optimal,
            .cat_l2 = l2,
            .verbose_eval = 0,
        }, null);
        defer res.model.deinit();
        var bundle = try model_mod.fromBooster(gpa, &res.model, schema);
        defer bundle.deinit();
        counts[i] = countCatNodes(&bundle);
    }
    try testing.expect(counts[0] > 0);
    try testing.expectEqual(@as(usize, 0), counts[1]);
}

test "catContains agrees with a linear scan, on every set it can hold" {
    // The binary search a categorical split uses at prediction time. Nothing
    // else in the suite could see it: making it return false unconditionally
    // left every test passing, because a round trip compares a broken model
    // against itself and a trained model simply routes every row right.
    var prng: std.Random.DefaultPrng = .init(5);
    const r = prng.random();
    var ids: [hist.max_cat_ids]data.BinIdx = undefined;
    for (0..200) |_| {
        var present = [_]bool{false} ** 512;
        const n = r.uintLessThan(usize, hist.max_cat_ids) + 1;
        var k: usize = 0;
        while (k < n) : (k += 1) present[r.uintLessThan(u32, 512)] = true;
        var m: usize = 0;
        for (present, 0..) |p, v| {
            if (!p or m == hist.max_cat_ids) continue;
            ids[m] = @intCast(v);
            m += 1;
        }
        if (m == 0) continue;
        for (0..512) |v| {
            const bin: data.BinIdx = @intCast(v);
            var want = false;
            for (ids[0..m]) |x| want = want or x == bin;
            try testing.expectEqual(want, hist.catContains(ids[0..m], bin));
        }
    }
}

test "greedy binning gives a dominant value its own bin; quantile does not" {
    // The failure this guards is silent and expensive: on a column that is
    // mostly one value a quantile rule spends its budget inside that mass and
    // leaves the tail a handful of bins. On adult's capital-gain that was
    // 0.0136 AUC against LightGBM, and no test could see it.
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 1);
    defer pool.deinit();

    const n = 4000;
    const vals = try gpa.alloc(f32, n);
    defer gpa.free(vals);
    const lab = try gpa.alloc(f32, n);
    defer gpa.free(lab);
    var prng: std.Random.DefaultPrng = .init(9);
    const r = prng.random();
    for (vals, lab) |*v, *l| {
        v.* = if (r.float(f32) < 0.9) 0.0 else r.float(f32) * 1000.0;
        l.* = if (v.* > 500) 1.0 else 0.0;
    }

    var used: [2]usize = undefined;
    for ([2]data.BinPolicy{ .quantile, .greedy }, 0..) |policy, i| {
        var f = try numericFrame(gpa, vals, lab);
        defer f.deinit();
        var ds = try data.quantise(gpa, pool, &f, .{ .bin_policy = policy, .max_bin = 256 }, null, &.{});
        defer ds.deinit();
        var seen = [_]bool{false} ** 300;
        var c: usize = 0;
        for (ds.columnNarrow(0), vals) |b, v| {
            if (v > 0 and !seen[b]) {
                seen[b] = true;
                c += 1;
            }
        }
        used[i] = c;
    }
    // Quantile collapses the tail onto a few bins; greedy spends the budget there.
    try testing.expect(used[1] > used[0] * 4);
    try testing.expect(used[1] > 100);
}

test "optimal categorical splits actually fire, and ordinal ones never do" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();
    var ds = try synthWithCats(gpa, 3000, 7);
    defer ds.deinit();

    for ([2]tree.CatSplit{ .ordinal, .optimal }) |mode| {
        var schema = try data.Schema.fromDataset(gpa, &ds);
        errdefer schema.deinit();
        var res = try booster.train(gpa, pool, &ds, null, .{
            .n_rounds = 30,
            .max_depth = 4,
            .cat_split = mode,
            .verbose_eval = 0,
        }, null);
        defer res.model.deinit();
        var bundle = try model_mod.fromBooster(gpa, &res.model, schema);
        defer bundle.deinit();
        const n = countCatNodes(&bundle);
        switch (mode) {
            // Without this the round-trip test below would pass vacuously.
            .optimal => try testing.expect(n > 0),
            .ordinal => try testing.expectEqual(@as(usize, 0), n),
        }
    }
}

test "a model carrying categorical subset splits round trips exactly" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();
    var ds = try synthWithCats(gpa, 3000, 11);
    defer ds.deinit();
    var schema = try data.Schema.fromDataset(gpa, &ds);
    errdefer schema.deinit();

    var res = try booster.train(gpa, pool, &ds, null, .{
        .n_rounds = 30,
        .max_depth = 4,
        .cat_split = .optimal,
        .verbose_eval = 0,
    }, null);
    defer res.model.deinit();
    var bundle = try model_mod.fromBooster(gpa, &res.model, schema);
    defer bundle.deinit();
    try testing.expect(countCatNodes(&bundle) > 0);

    const before = try gpa.alloc(f32, ds.n_rows);
    defer gpa.free(before);
    bundle.predict(pool, &ds, before);

    const bytes = try model_mod.serialise(gpa, &bundle);
    defer gpa.free(bytes);
    var loaded = try model_mod.deserialise(gpa, bytes);
    defer loaded.deinit();
    try testing.expectEqual(countCatNodes(&bundle), countCatNodes(&loaded));

    const after = try gpa.alloc(f32, ds.n_rows);
    defer gpa.free(after);
    loaded.predict(pool, &ds, after);
    for (before, after) |x, y| try testing.expectEqual(x, y);
}

fn countLinLeaves(b: *const model_mod.Bundle) usize {
    var n: usize = 0;
    for (b.trees) |t| for (t.nodes) |nd| {
        if (nd.is_leaf and nd.n_lin != 0) n += 1;
    };
    return n;
}

test "linear leaves fire, stay off by default, and round trip exactly" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();
    var ds = try synth(gpa, 3000, 13);
    defer ds.deinit();

    // Off by default: not one leaf carries a slope.
    {
        var schema = try data.Schema.fromDataset(gpa, &ds);
        errdefer schema.deinit();
        var res = try booster.train(gpa, pool, &ds, null, .{ .n_rounds = 20, .max_depth = 4, .verbose_eval = 0 }, null);
        defer res.model.deinit();
        var bundle = try model_mod.fromBooster(gpa, &res.model, schema);
        defer bundle.deinit();
        try testing.expectEqual(@as(usize, 0), countLinLeaves(&bundle));
    }

    var schema = try data.Schema.fromDataset(gpa, &ds);
    errdefer schema.deinit();
    var res = try booster.train(gpa, pool, &ds, null, .{
        .n_rounds = 20,
        .max_depth = 4,
        .linear_leaves = true,
        .verbose_eval = 0,
    }, null);
    defer res.model.deinit();
    var bundle = try model_mod.fromBooster(gpa, &res.model, schema);
    defer bundle.deinit();
    // Without this the round trip below would pass vacuously.
    try testing.expect(countLinLeaves(&bundle) > 0);

    const before = try gpa.alloc(f32, ds.n_rows);
    defer gpa.free(before);
    bundle.predict(pool, &ds, before);

    const bytes = try model_mod.serialise(gpa, &bundle);
    defer gpa.free(bytes);
    var loaded = try model_mod.deserialise(gpa, bytes);
    defer loaded.deinit();
    try testing.expectEqual(countLinLeaves(&bundle), countLinLeaves(&loaded));

    const after = try gpa.alloc(f32, ds.n_rows);
    defer gpa.free(after);
    loaded.predict(pool, &ds, after);
    for (before, after) |x, y| try testing.expectEqual(x, y);
}

test "a linear leaf beats its own constant leaf on the training objective" {
    // The affine fit contains the constant fit as beta = 0, so it can only
    // reduce training loss. If it does not, the solve is returning something
    // that is not the optimum.
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();
    var ds = try synth(gpa, 4000, 17);
    defer ds.deinit();

    var loss: [2]f64 = undefined;
    for ([2]bool{ false, true }, 0..) |lin, i| {
        var res = try booster.train(gpa, pool, &ds, null, .{
            .n_rounds = 40,
            .max_depth = 4,
            .linear_leaves = lin,
            .verbose_eval = 0,
        }, null);
        defer res.model.deinit();
        const out = try gpa.alloc(f32, ds.n_rows);
        defer gpa.free(out);
        res.model.predict(pool, &ds, out);
        var acc: f64 = 0;
        for (out, ds.labels) |p, y| {
            const q = @min(@max(p, 1e-7), 1 - 1e-7);
            acc -= @as(f64, y) * @log(q) + (1 - @as(f64, y)) * @log(1 - q);
        }
        loss[i] = acc / @as(f64, @floatFromInt(ds.n_rows));
    }
    try testing.expect(loss[1] < loss[0]);
}

test "a linear leaf is invariant to rescaling a column" {
    // Multiplying a numeric column by 1000 changes no bin -- binning is by
    // rank -- so the model must not move. It is the sharpest available check
    // that the ridge is applied to standardised axes and the coefficients are
    // brought back out of them: skip either step and the fitted slopes come
    // out scaled by 1000 and the predictions change.
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();

    var out: [2][]f32 = undefined;
    for ([2]f32{ 1.0, 1000.0 }, 0..) |scale, i| {
        var ds = try synth(gpa, 3000, 23);
        defer ds.deinit();
        for (ds.edges[0]) |*e| e.* *= scale;

        var res = try booster.train(gpa, pool, &ds, null, .{
            .n_rounds = 30,
            .max_depth = 4,
            .linear_leaves = true,
            .verbose_eval = 0,
        }, null);
        defer res.model.deinit();
        out[i] = try gpa.alloc(f32, ds.n_rows);
        res.model.predict(pool, &ds, out[i]);
    }
    defer for (out) |o| gpa.free(o);

    for (out[0], out[1]) |x, y| try testing.expectApproxEqAbs(x, y, 1e-4);
}

test "gbdt survives a save/load round trip exactly" {
    try roundTrip(.gbdt);
}
test "random forest survives a save/load round trip exactly" {
    try roundTrip(.random_forest);
}
test "linear survives a save/load round trip exactly" {
    try roundTrip(.linear);
}

test "blend weights behave, and mixing model kinds works" {
    const gpa = testing.allocator;
    const f = try shared();
    const a = &f.gbdt;
    const b = &f.forest;
    const n = f.ds.n_rows;

    const pa = try gpa.alloc(f32, n);
    defer gpa.free(pa);
    const pb = try gpa.alloc(f32, n);
    defer gpa.free(pb);
    const mix = try gpa.alloc(f32, n);
    defer gpa.free(mix);
    a.predict(f.pool, &f.ds, pa);
    b.predict(f.pool, &f.ds, pb);

    const refs = [_]*const model_mod.Bundle{ a, b };

    // All the weight on one model reproduces that model.
    try model_mod.blend(gpa, f.pool, &refs, &.{ 1, 0 }, &f.ds, mix);
    for (pa, mix) |x, y| try testing.expectApproxEqAbs(x, y, 1e-6);

    // Equal weights give the mean of a booster's probabilities and a forest's
    // — different kinds, one scale, which is the point of the interface.
    try model_mod.blend(gpa, f.pool, &refs, &.{ 1, 1 }, &f.ds, mix);
    for (pa, pb, mix) |x, y, m| try testing.expectApproxEqAbs((x + y) / 2.0, m, 1e-6);

    // Weights are normalised, so 3:1 and 30:10 agree.
    const m2 = try gpa.alloc(f32, n);
    defer gpa.free(m2);
    try model_mod.blend(gpa, f.pool, &refs, &.{ 3, 1 }, &f.ds, mix);
    try model_mod.blend(gpa, f.pool, &refs, &.{ 30, 10 }, &f.ds, m2);
    for (mix, m2) |x, y| try testing.expectApproxEqAbs(x, y, 1e-6);

    try testing.expectError(error.WeightCountMismatch, model_mod.blend(gpa, f.pool, &refs, &.{1}, &f.ds, mix));
    try testing.expectError(error.WeightsSumToZero, model_mod.blend(gpa, f.pool, &refs, &.{ 0, 0 }, &f.ds, mix));
}

test "a corrupt or truncated model file is rejected, not read past" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 1);
    defer pool.deinit();
    var ds = try synth(gpa, 800, 41);
    defer ds.deinit();
    var res = try booster.train(gpa, pool, &ds, null, .{ .n_rounds = 5, .max_depth = 3, .verbose_eval = 0 }, null);
    defer res.model.deinit();
    var bundle = try model_mod.fromBooster(gpa, &res.model, try data.Schema.fromDataset(gpa, &ds));
    defer bundle.deinit();

    const bytes = try model_mod.serialise(gpa, &bundle);
    defer gpa.free(bytes);

    try testing.expectError(error.NotAModelFile, model_mod.deserialise(gpa, "nope"));
    try testing.expectError(error.NotAModelFile, model_mod.deserialise(gpa, bytes[4..]));

    // Every truncation must fail cleanly rather than read out of bounds.
    var cut: usize = 4;
    while (cut < bytes.len) : (cut += @max(1, bytes.len / 37)) {
        if (model_mod.deserialise(gpa, bytes[0..cut])) |*m| {
            var mm = m.*;
            mm.deinit();
            return error.TruncationAccepted;
        } else |_| {}
    }

    // A wrong version is refused rather than misparsed.
    const bad = try gpa.dupe(u8, bytes);
    defer gpa.free(bad);
    bad[4] = 99;
    try testing.expectError(error.UnsupportedModelVersion, model_mod.deserialise(gpa, bad));
}

// ---------------------------------------------------------- label encoding
//
// The trap here is the sibling of the one above, and it went unnoticed for
// longer: a *target* column of strings also becomes a dictionary whose ids are
// assigned by first appearance, and that id was used directly as the label. A
// file whose first row holds the positive class trained the opposite model,
// and a holdout whose first row disagreed with the training file's scored an
// inverted AUC -- which still looks like a number.

test "label encoding does not depend on row order" {
    const gpa = testing.allocator;

    var a = try frameWith(gpa, &.{ "No", "Yes", "No", "Yes" }, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    var b = try frameWith(gpa, &.{ "Yes", "No", "Yes", "No" }, &.{ 1, 2, 3, 4 });
    defer b.deinit();

    // The two frames build opposite dictionaries...
    try testing.expect(a.values[0][0] == b.values[0][0]);
    try testing.expectEqualStrings("No", a.levels[0][@intFromFloat(a.values[0][0])]);
    try testing.expectEqualStrings("Yes", b.levels[0][@intFromFloat(b.values[0][0])]);

    // ...and the encoder makes them agree, because it orders by string.
    var enc_a = try data.LabelEncoder.fromColumn(gpa, &a, 0, null);
    defer enc_a.deinit();
    var enc_b = try data.LabelEncoder.fromColumn(gpa, &b, 0, null);
    defer enc_b.deinit();

    const ya = try enc_a.encode(gpa, &a, 0);
    defer gpa.free(ya);
    const yb = try enc_b.encode(gpa, &b, 0);
    defer gpa.free(yb);

    try testing.expectEqualSlices(f32, &.{ 0, 1, 0, 1 }, ya);
    try testing.expectEqualSlices(f32, &.{ 1, 0, 1, 0 }, yb);
    // Same row content, same number, whichever file it came from.
    for (0..4) |i| {
        const sa = a.levels[0][@as(usize, @intFromFloat(a.values[0][i]))];
        const sb = b.levels[0][@as(usize, @intFromFloat(b.values[0][i]))];
        if (std.mem.eql(u8, sa, sb)) try testing.expectEqual(ya[i], yb[i]);
    }
}

test "--pos-label overrides the sorted order" {
    const gpa = testing.allocator;
    var f = try frameWith(gpa, &.{ "abnormal", "normal", "abnormal" }, &.{ 1, 2, 3 });
    defer f.deinit();

    // Sorted puts the clinically interesting class at 0, which is backwards.
    var plain = try data.LabelEncoder.fromColumn(gpa, &f, 0, null);
    defer plain.deinit();
    try testing.expectEqual(@as(?usize, 1), plain.classIndex("normal"));

    var forced = try data.LabelEncoder.fromColumn(gpa, &f, 0, "abnormal");
    defer forced.deinit();
    try testing.expectEqual(@as(?usize, 1), forced.classIndex("abnormal"));

    const y = try forced.encode(gpa, &f, 0);
    defer gpa.free(y);
    try testing.expectEqualSlices(f32, &.{ 1, 0, 1 }, y);

    try testing.expectError(
        error.PosLabelNotFound,
        data.LabelEncoder.fromColumn(gpa, &f, 0, "Abnormal"),
    );
    // A numeric target has no classes to name, so the flag is a mistake
    // rather than a no-op.
    try testing.expectError(
        error.PosLabelOnNumericTarget,
        data.LabelEncoder.fromColumn(gpa, &f, 1, "abnormal"),
    );
}

test "targets the models cannot represent are rejected" {
    const gpa = testing.allocator;

    // Three classes would encode as 0,1,2 and be fitted as if those were
    // magnitudes.
    var three = try frameWith(gpa, &.{ "a", "b", "c", "a" }, &.{ 1, 2, 3, 4 });
    defer three.deinit();
    try testing.expectError(
        error.MulticlassNotSupported,
        data.LabelEncoder.fromColumn(gpa, &three, 0, null),
    );

    var one = try frameWith(gpa, &.{ "a", "a", "a" }, &.{ 1, 2, 3 });
    defer one.deinit();
    try testing.expectError(
        error.SingleClassTarget,
        data.LabelEncoder.fromColumn(gpa, &one, 0, null),
    );
}

test "a missing or unknown label is an error, never a quiet zero" {
    const gpa = testing.allocator;

    var f = try frameWith(gpa, &.{ "No", "Yes", "No" }, &.{ 1, 2, 3 });
    defer f.deinit();
    var enc = try data.LabelEncoder.fromColumn(gpa, &f, 0, null);
    defer enc.deinit();

    // A hole in the target: class 0 is a legitimate value, so it cannot also
    // mean "absent".
    var holed = try frameWith(gpa, &.{ "No", "Yes", "No" }, &.{ 1, 2, 3 });
    defer holed.deinit();
    holed.values[0][1] = std.math.nan(f32);
    try testing.expectError(error.MissingLabelValue, enc.encode(gpa, &holed, 0));
    // Same rule on the numeric path, where NaN otherwise flows straight
    // through into the gradient.
    var num_enc = data.LabelEncoder{ .gpa = gpa };
    holed.values[1][2] = std.math.nan(f32);
    try testing.expectError(error.MissingLabelValue, num_enc.encode(gpa, &holed, 1));

    // A class the model never saw cannot be scored against.
    var typo = try frameWith(gpa, &.{ "No", "Yes", "Yse" }, &.{ 1, 2, 3 });
    defer typo.deinit();
    try testing.expectError(error.UnseenLabelClass, enc.encode(gpa, &typo, 0));

    // Numbers where strings were expected, and the reverse.
    try testing.expectError(error.LabelKindMismatch, enc.encode(gpa, &f, 1));
    try testing.expectError(error.LabelKindMismatch, num_enc.encode(gpa, &f, 0));
}

test "logistic rejects a numeric target outside [0,1]" {
    const gpa = testing.allocator;
    var f = try frameWith(gpa, &.{ "No", "Yes", "No" }, &.{ 1, 2, 1 });
    defer f.deinit();

    var enc = try data.LabelEncoder.fromColumn(gpa, &f, 1, null);
    defer enc.deinit();
    const y = try enc.encode(gpa, &f, 1);
    defer gpa.free(y);

    // A {1,2}-coded target is the classic mistake; squared error is happy
    // with it, logistic is not.
    try testing.expectError(error.LabelOutOfRange, enc.validate(y, .logistic));
    try enc.validate(y, .squared_error);
    try enc.validate(&.{ 0, 1, 0.5 }, .logistic);
    try testing.expectError(error.LabelOutOfRange, enc.validate(&.{std.math.nan(f32)}, .logistic));
}

test "a saved model decodes a holdout with its own class order" {
    // The end-to-end version: train-time classes travel in the .zm file and
    // are applied to a holdout that orders its target the other way round.
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();

    var train = try frameWith(gpa, &.{ "No", "Yes", "No", "Yes" }, &.{ 1, 2, 3, 4 });
    defer train.deinit();
    var enc = try data.LabelEncoder.fromColumn(gpa, &train, 0, null);
    defer enc.deinit();

    var train_ds = try data.quantise(gpa, pool, &train, .{}, .{ .col = 0, .enc = &enc }, &.{});
    defer train_ds.deinit();
    try testing.expectEqualSlices(f32, &.{ 0, 1, 0, 1 }, train_ds.labels);

    var bundle = model_mod.Bundle{
        .gpa = gpa,
        .kind = .gbdt,
        .schema = try data.Schema.fromDataset(gpa, &train_ds),
        .objective = .logistic,
    };
    defer bundle.deinit();
    try bundle.setLabel("colour", &enc);

    const bytes = try model_mod.serialise(gpa, &bundle);
    defer gpa.free(bytes);
    var back = try model_mod.deserialise(gpa, bytes);
    defer back.deinit();
    try testing.expectEqualStrings("colour", back.label);
    try testing.expectEqual(@as(usize, 2), back.classes.len);
    try testing.expectEqualStrings("No", back.classes[0]);
    try testing.expectEqualStrings("Yes", back.classes[1]);

    // Holdout with the opposite first appearance. Read as raw ids this would
    // give {0,1,0,1} -- the exact inversion of the truth.
    var holdout = try frameWith(gpa, &.{ "Yes", "No", "Yes", "No" }, &.{ 4, 3, 2, 1 });
    defer holdout.deinit();
    const view = back.encoder();
    var ds = try data.applySchema(gpa, pool, &holdout, &back.schema, .{ .col = 0, .enc = &view });
    defer ds.deinit();
    try testing.expectEqualSlices(f32, &.{ 1, 0, 1, 0 }, ds.labels);
}

test "the categorical width limit is a policy now, and still holds by default" {
    // This limit used to be the bin type: bins were u8, bin 0 was missing, so
    // 255 levels was all a column could carry. The index is u16 now and the
    // limit is `max_cat_levels`, which *defaults* to the same 255 so that
    // widening the index changes no existing run on its own.
    //
    // Both sides of the boundary, because an off-by-one here once truncated a
    // level into the missing bin in ReleaseFast and trained on it silently.
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();

    var names: [260][8]u8 = undefined;
    var levels: [260][]const u8 = undefined;
    for (&names, &levels, 0..) |*buf, *lvl, i| {
        lvl.* = try std.fmt.bufPrint(buf, "c{d}", .{i});
    }

    {
        var f = try frameWith(gpa, levels[0..255], &[_]f32{0} ** 255);
        defer f.deinit();
        var ds = try data.quantise(gpa, pool, &f, .{}, null, &.{});
        defer ds.deinit();
        // 255 levels -> bins 1..255, plus the missing bin.
        try testing.expectEqual(@as(u16, 256), ds.n_bins[0]);
        try testing.expect(!ds.isWide(0));
        var max_bin: data.BinIdx = 0;
        for (ds.columnNarrow(0)) |b| max_bin = @max(max_bin, b);
        try testing.expectEqual(@as(data.BinIdx, 255), max_bin);
    }

    {
        // And the capability the widening exists for: raise the policy and a
        // column past the old ceiling bins correctly, into bins the u8 index
        // could not have addressed.
        var f = try frameWith(gpa, levels[0..260], &[_]f32{0} ** 260);
        defer f.deinit();
        var ds = try data.quantise(gpa, pool, &f, .{ .max_cat_levels = 4096 }, null, &.{});
        defer ds.deinit();
        try testing.expectEqual(@as(u16, 261), ds.n_bins[0]);
        // Past 256 bins the column moves out of the byte mirror and into the
        // wide store, which is the whole point of the split.
        try testing.expect(ds.isWide(0));
        var max_bin: data.BinIdx = 0;
        for (ds.columnWide(0)) |b| max_bin = @max(max_bin, b);
        try testing.expectEqual(@as(data.BinIdx, 260), max_bin);
        // The row-major mirror is uniform and still carries the real bin.
        var max_rm: data.BinIdx = 0;
        for (0..ds.n_rows) |r| max_rm = @max(max_rm, ds.bins_rm[r * ds.n_features]);
        try testing.expectEqual(@as(data.BinIdx, 260), max_rm);
    }

    {
        var f = try frameWith(gpa, levels[0..256], &[_]f32{0} ** 256);
        defer f.deinit();
        // `quantise` checks width up front, on the columns that survived the
        // drop list, so the error names the real cause rather than the
        // BinningFailed a worker would have produced later.
        try testing.expectError(
            error.CategoricalTooWide,
            data.quantise(gpa, pool, &f, .{}, null, &.{}),
        );
    }

    {
        // And the whole point of checking after drops: a column too wide to
        // bin must not stop a run that excluded it. This used to fail during
        // the CSV read, before `quantise` had seen the drop list at all.
        var f = try frameWith(gpa, levels[0..256], &[_]f32{0} ** 256);
        defer f.deinit();
        try testing.expectError(
            error.NoFeatures,
            data.quantise(gpa, pool, &f, .{}, null, &.{ "colour", "num" }),
        );
    }
}

test "a numeric bin is represented by the mean of its values, not its midpoint" {
    // The linear model has to pick one number to stand for a whole bin. The
    // midpoint of the bin's edges is the obvious choice and a biased one: the
    // top bin of a quantile split is unbounded above, so its midpoint is the
    // cut itself no matter how far the values inside run. On the EV set that
    // bias cost 0.0003 AUC against scikit-learn -- the entire remaining gap
    // once the solver was fixed.
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();

    // Six zeros then a right tail. With two real bins the single cut lands on
    // 0, so bin 1 is the zeros and bin 2 is {1, 2, 3, 400}.
    const nums = [_]f32{ 0, 0, 0, 0, 0, 0, 1, 2, 3, 400 };
    const colours = [_][]const u8{"a"} ** nums.len;
    var f = try frameWith(gpa, &colours, &nums);
    defer f.deinit();

    var ds = try data.quantise(gpa, pool, &f, .{ .max_bin = 3 }, null, &.{});
    defer ds.deinit();

    try testing.expectEqual(@as(u16, 3), ds.n_bins[1]);
    try testing.expectEqualSlices(f32, &.{0}, ds.edges[1]);
    try testing.expectEqual(@as(usize, 3), ds.means[1].len);

    // Slot 0 is the missing bin, which holds no values of its own and takes
    // the column mean so a missing entry sits at the centre.
    try testing.expectApproxEqAbs(@as(f32, 40.6), ds.means[1][0], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 0.0), ds.means[1][1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 101.5), ds.means[1][2], 1e-3);

    // What the midpoint rule would have said for that top bin, for contrast:
    // the cut itself, off by two orders of magnitude.
    try testing.expectApproxEqAbs(@as(f32, 0.0), data.binMidpoint(ds.edges[1], 1), 1e-6);

    // A categorical bin *is* its level, so there is nothing to average.
    try testing.expectEqual(@as(usize, 0), ds.means[0].len);

    // A row subset carries the representatives with it; a train/valid split
    // that dropped them would fit one half on a different design matrix.
    const rows = [_]u32{ 0, 6, 9 };
    var part = try data.subset(gpa, &ds, &rows);
    defer part.deinit();
    try testing.expectEqualSlices(f32, ds.means[1], part.means[1]);

    // And the linear model must actually use them. Same column, now as the
    // only feature under a labelled frame, so the fitted design's bin-to-value
    // table can be compared against the means directly. Without this the
    // wiring could be cut and every other test would still pass, since the
    // hand-built fixtures all take the midpoint fallback.
    const labels = [_][]const u8{ "No", "Yes" } ** (nums.len / 2);
    var lf = try frameWith(gpa, &labels, &nums);
    defer lf.deinit();
    var enc = try data.LabelEncoder.fromColumn(gpa, &lf, 0, null);
    defer enc.deinit();
    var lds = try data.quantise(gpa, pool, &lf, .{ .max_bin = 3 }, .{ .col = 0, .enc = &enc }, &.{});
    defer lds.deinit();

    var res = try linear.train(gpa, pool, &lds, null, .{
        .algo = .linear,
        .lin_epochs = 5,
        .lin_standardize = false,
        .verbose_eval = 0,
    }, null);
    defer res.model.deinit();
    try testing.expectEqualSlices(f32, lds.means[0], res.model.design.repr[0]);
}
