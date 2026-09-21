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

    // Mirror of `bins`, since the histogram kernel reads row-major.
    const bins_rm = try gpa.alloc(u8, n_features * n_rows);
    errdefer gpa.free(bins_rm);
    for (0..n_rows) |ri| for (0..n_features) |f| {
        bins_rm[ri * n_features + f] = bins[f * n_rows + ri];
    };

    return .{
        .gpa = gpa,
        .n_rows = n_rows,
        .n_features = n_features,
        .bins = bins,
        .bins_rm = bins_rm,
        .n_bins = n_bins,
        .edges = edges,
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

// --------------------------------------------------------- save / load / blend

const model_mod = @import("model.zig");

/// Builds a Frame by hand so a test can control the dictionary order, which is
/// the whole point of the schema.
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

    const cat = test_ds.column(0);
    const train_cat = train_ds.column(0);
    // red is bin 1 and blue bin 2 in training; the test file must agree.
    try testing.expectEqual(train_cat[0], cat[2]); // red
    try testing.expectEqual(train_cat[1], cat[1]); // blue
    try testing.expectEqual(train_cat[3], cat[0]); // green
    // A level never seen in training falls into the missing bin.
    try testing.expectEqual(@as(u8, 0), cat[3]); // mauve
}

fn roundTrip(kind: config.Algo) !void {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();

    var ds = try synth(gpa, 3000, 31);
    defer ds.deinit();
    var schema = try data.Schema.fromDataset(gpa, &ds);
    errdefer schema.deinit();

    var bundle: model_mod.Bundle = undefined;
    var keep_linear: ?linear.Linear = null;
    switch (kind) {
        .gbdt => {
            var res = try booster.train(gpa, pool, &ds, null, .{ .n_rounds = 25, .max_depth = 4, .verbose_eval = 0 }, null);
            defer res.model.deinit();
            bundle = try model_mod.fromBooster(gpa, &res.model, schema);
        },
        .random_forest => {
            var cfg: config.Config = .{ .algo = .random_forest, .n_rounds = 15, .verbose_eval = 0 };
            cfg.applyAlgoDefaults(&.{});
            var res = try forest.train(gpa, pool, &ds, null, cfg, null);
            defer res.model.deinit();
            bundle = try model_mod.fromForest(gpa, &res.model, schema);
        },
        .linear => {
            const res = try linear.train(gpa, pool, &ds, null, .{ .algo = .linear, .lin_epochs = 60, .verbose_eval = 0 }, null);
            keep_linear = res.model;
            bundle = .{ .gpa = gpa, .kind = .linear, .schema = schema, .objective = .logistic, .lin = res.model };
        },
    }
    defer {
        if (keep_linear) |*l| {
            l.deinit();
            bundle.schema.deinit();
        } else bundle.deinit();
    }

    const before = try gpa.alloc(f32, ds.n_rows);
    defer gpa.free(before);
    bundle.predict(pool, &ds, before);

    const bytes = try model_mod.serialise(gpa, &bundle);
    defer gpa.free(bytes);
    var loaded = try model_mod.deserialise(gpa, bytes);
    defer loaded.deinit();

    const after = try gpa.alloc(f32, ds.n_rows);
    defer gpa.free(after);
    loaded.predict(pool, &ds, after);

    // Bit-identical, not approximately equal: a saved model that drifts is a
    // model whose submitted predictions do not match the ones you validated.
    for (before, after) |x, y| try testing.expectEqual(x, y);
    try testing.expectEqual(bundle.schema.n_features, loaded.schema.n_features);
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
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();

    var ds = try synth(gpa, 2500, 37);
    defer ds.deinit();

    var gb = try booster.train(gpa, pool, &ds, null, .{ .n_rounds = 20, .max_depth = 4, .verbose_eval = 0 }, null);
    defer gb.model.deinit();
    var a = try model_mod.fromBooster(gpa, &gb.model, try data.Schema.fromDataset(gpa, &ds));
    defer a.deinit();

    var cfg: config.Config = .{ .algo = .random_forest, .n_rounds = 15, .verbose_eval = 0 };
    cfg.applyAlgoDefaults(&.{});
    var rf = try forest.train(gpa, pool, &ds, null, cfg, null);
    defer rf.model.deinit();
    var b = try model_mod.fromForest(gpa, &rf.model, try data.Schema.fromDataset(gpa, &ds));
    defer b.deinit();

    const pa = try gpa.alloc(f32, ds.n_rows);
    defer gpa.free(pa);
    const pb = try gpa.alloc(f32, ds.n_rows);
    defer gpa.free(pb);
    const mix = try gpa.alloc(f32, ds.n_rows);
    defer gpa.free(mix);
    a.predict(pool, &ds, pa);
    b.predict(pool, &ds, pb);

    const refs = [_]*const model_mod.Bundle{ &a, &b };

    // All the weight on one model reproduces that model.
    try model_mod.blend(gpa, pool, &refs, &.{ 1, 0 }, &ds, mix);
    for (pa, mix) |x, y| try testing.expectApproxEqAbs(x, y, 1e-6);

    // Equal weights give the mean of a booster's probabilities and a forest's
    // — different kinds, one scale, which is the point of the interface.
    try model_mod.blend(gpa, pool, &refs, &.{ 1, 1 }, &ds, mix);
    for (pa, pb, mix) |x, y, m| try testing.expectApproxEqAbs((x + y) / 2.0, m, 1e-6);

    // Weights are normalised, so 3:1 and 30:10 agree.
    const m2 = try gpa.alloc(f32, ds.n_rows);
    defer gpa.free(m2);
    try model_mod.blend(gpa, pool, &refs, &.{ 3, 1 }, &ds, mix);
    try model_mod.blend(gpa, pool, &refs, &.{ 30, 10 }, &ds, m2);
    for (mix, m2) |x, y| try testing.expectApproxEqAbs(x, y, 1e-6);

    try testing.expectError(error.WeightCountMismatch, model_mod.blend(gpa, pool, &refs, &.{1}, &ds, mix));
    try testing.expectError(error.WeightsSumToZero, model_mod.blend(gpa, pool, &refs, &.{ 0, 0 }, &ds, mix));
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
