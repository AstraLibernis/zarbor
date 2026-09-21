//! End-to-end regression tests for the three models.
//!
//! These exist because two silent correctness bugs shipped: a histogram stride
//! that only misbehaved where `cache_line / @sizeOf(Bin)` is not a power of two
//! (so it depended on the CPU), and a sibling-subtraction identity that broke
//! whenever per-level or per-node feature sampling was on. Both produced
//! plausible-looking runs rather than crashes, and neither was covered. Every
//! test below therefore asserts on *achieved accuracy*, not merely on "it ran".

const std = @import("std");
const data = @import("data.zig");
const config = @import("config.zig");
const booster = @import("booster.zig");
const forest = @import("forest.zig");
const linear = @import("linear.zig");
const hist = @import("hist.zig");
const Pool = @import("pool.zig").Pool;
const metric = @import("metric.zig");

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

    return .{
        .gpa = gpa,
        .n_rows = n_rows,
        .n_features = n_features,
        .bins = bins,
        .n_bins = n_bins,
        .edges = edges,
        .kinds = kinds,
        .names = names,
        .labels = labels,
    };
}

fn aucOf(gpa: std.mem.Allocator, pred: []const f32, labels: []const f32) !f64 {
    return metric.auc(gpa, pred, labels);
}

test "histogram stride is a whole number of cache lines and never truncates" {
    // The original code used `alignForward(max_bins, cache_line / @sizeOf(Bin))`,
    // which requires a power-of-two alignment it does not have on every CPU.
    // On a 128-byte line with a 24-byte Bin that argument is 5, and the stride
    // came out *below* max_bins, overlapping adjacent features' histograms.
    const gpa = testing.allocator;
    for ([_]usize{ 2, 3, 17, 33, 64, 255, 256 }) |max_bins| {
        var bank = try hist.Bank.init(gpa, 2, 3, max_bins);
        defer bank.deinit();
        try testing.expect(bank.stride >= max_bins);
        try testing.expectEqual(@as(usize, 0), (bank.stride * @sizeOf(hist.Bin)) % std.atomic.cache_line);
    }
}

test "gbdt learns the signal" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();

    var ds = try synth(gpa, 4000, 7);
    defer ds.deinit();

    var res = try booster.train(gpa, pool, &ds, null, .{
        .n_rounds = 60,
        .max_depth = 4,
        .verbose_eval = 0,
    }, null);
    defer res.model.deinit();

    const pred = try gpa.alloc(f32, ds.n_rows);
    defer gpa.free(pred);
    res.model.predictRaw(pool, &ds, pred);
    try testing.expect(try aucOf(gpa, pred, ds.labels) > 0.90);
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

test "random forest learns the signal and predicts probabilities" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();

    var ds = try synth(gpa, 4000, 17);
    defer ds.deinit();

    var cfg: config.Config = .{ .algo = .random_forest, .n_rounds = 40, .verbose_eval = 0 };
    cfg.applyAlgoDefaults(&.{});
    cfg.applyForestFeatureDefault(ds.n_features, &.{});

    var res = try forest.train(gpa, pool, &ds, null, cfg, null);
    defer res.model.deinit();

    const pred = try gpa.alloc(f32, ds.n_rows);
    defer gpa.free(pred);
    res.model.predict(pool, &ds, pred);

    try testing.expect(try aucOf(gpa, pred, ds.labels) > 0.88);
    // A forest averages leaf means, so its output is already a probability.
    for (pred) |v| try testing.expect(v >= 0.0 and v <= 1.0);
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
