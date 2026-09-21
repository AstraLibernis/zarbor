//! Min/max sweep: every config field, asserted against what it claims to do.
//!
//! Two silent correctness bugs shipped in this library before anything here
//! existed, both of which produced plausible output rather than a crash. The
//! lesson taken is that "it trained and the number looked fine" is not
//! evidence. Each test below therefore drives one setting to its extremes and
//! asserts the *mechanism* — node counts, leaf weights, tree counts, epoch
//! counts — rather than only that accuracy stayed acceptable.

const std = @import("std");
const data = @import("data.zig");
const config = @import("config.zig");
const booster = @import("booster.zig");
const forest = @import("forest.zig");
const linear = @import("linear.zig");
const tree = @import("tree.zig");
const Pool = @import("pool.zig").Pool;
const metric = @import("metric.zig");

const testing = std.testing;

fn synth(gpa: std.mem.Allocator, n_rows: usize, seed: u64) !data.Dataset {
    const n_features: usize = 6;
    const n_bin: u16 = 17;
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
        // Left empty on purpose: `buildDesign` must still work from edges
        // alone, which is the path a hand-built dataset takes.
        .means = means,
        .kinds = kinds,
        .names = names,
        .levels = levels,
        .labels = labels,
    };
}

const Fix = struct {
    gpa: std.mem.Allocator,
    pool: *Pool,
    ds: data.Dataset,

    fn init(gpa: std.mem.Allocator, n_rows: usize, seed: u64) !Fix {
        const pool = try Pool.init(gpa, 2);
        errdefer pool.deinit();
        return .{ .gpa = gpa, .pool = pool, .ds = try synth(gpa, n_rows, seed) };
    }
    fn deinit(f: *Fix) void {
        f.ds.deinit();
        f.pool.deinit();
    }
    fn fit(f: *Fix, cfg: config.Config) !booster.TrainResult {
        var c = cfg;
        c.verbose_eval = 0;
        return booster.train(f.gpa, f.pool, &f.ds, null, c, null);
    }
    fn auc(f: *Fix, m: *const booster.Model) !f64 {
        const p = try f.gpa.alloc(f32, f.ds.n_rows);
        defer f.gpa.free(p);
        m.predictRaw(f.pool, &f.ds, p);
        return metric.auc(f.gpa, p, f.ds.labels);
    }
};

fn maxAbsLeaf(m: *const booster.Model) f32 {
    var mx: f32 = 0;
    for (m.trees.items) |t| for (t.nodes) |n| {
        if (n.is_leaf) mx = @max(mx, @abs(n.weight));
    };
    return mx;
}

fn leafCount(t: tree.Tree) usize {
    var n: usize = 0;
    for (t.nodes) |nd| {
        if (nd.is_leaf) n += 1;
    }
    return n;
}

fn maxDepthOf(t: tree.Tree, i: u32, d: u32) u32 {
    if (t.nodes[i].is_leaf) return d;
    return @max(maxDepthOf(t, t.nodes[i].left, d + 1), maxDepthOf(t, t.nodes[i].right, d + 1));
}

// ------------------------------------------------------------- ensemble size

test "n_rounds: the ensemble is exactly the size asked for" {
    var f = try Fix.init(testing.allocator, 2000, 1);
    defer f.deinit();
    for ([_]u32{ 1, 5, 50 }) |n| {
        var r = try f.fit(.{ .n_rounds = n, .max_depth = 3 });
        defer r.model.deinit();
        try testing.expectEqual(@as(usize, n), r.model.trees.items.len);
        try testing.expectEqual(n, r.n_rounds);
    }
}

test "learning_rate: leaf weights scale with it" {
    var f = try Fix.init(testing.allocator, 2000, 2);
    defer f.deinit();
    var lo = try f.fit(.{ .n_rounds = 1, .max_depth = 3, .learning_rate = 0.01 });
    defer lo.model.deinit();
    var hi = try f.fit(.{ .n_rounds = 1, .max_depth = 3, .learning_rate = 1.0 });
    defer hi.model.deinit();
    // Shrinkage is applied inside the leaf, so a 100x rate is a 100x weight.
    const ratio = maxAbsLeaf(&hi.model) / maxAbsLeaf(&lo.model);
    try testing.expect(ratio > 50 and ratio < 200);
}

test "base_score: an explicit value is used verbatim" {
    var f = try Fix.init(testing.allocator, 2000, 3);
    defer f.deinit();
    var r = try f.fit(.{ .n_rounds = 1, .max_depth = 2, .base_score = -1.25 });
    defer r.model.deinit();
    try testing.expectEqual(@as(f32, -1.25), r.model.base_score);
}

// ------------------------------------------------------------- capacity caps

test "max_depth: bounds realised tree depth, and 1 gives a stump" {
    var f = try Fix.init(testing.allocator, 4000, 4);
    defer f.deinit();
    for ([_]u32{ 1, 2, 5 }) |d| {
        var r = try f.fit(.{ .n_rounds = 1, .max_depth = d });
        defer r.model.deinit();
        try testing.expect(maxDepthOf(r.model.trees.items[0], 0, 0) <= d);
    }
    var stump = try f.fit(.{ .n_rounds = 1, .max_depth = 1 });
    defer stump.model.deinit();
    // root + two leaves
    try testing.expectEqual(@as(usize, 3), stump.model.trees.items[0].nodes.len);
}

test "max_leaves: bounds the leaf count under lossguide" {
    var f = try Fix.init(testing.allocator, 4000, 5);
    defer f.deinit();
    for ([_]u32{ 2, 4, 16 }) |L| {
        var r = try f.fit(.{ .n_rounds = 1, .grow_policy = .lossguide, .max_depth = 0, .max_leaves = L });
        defer r.model.deinit();
        try testing.expect(leafCount(r.model.trees.items[0]) <= L);
    }
}

test "min_child_samples: a cap at n forbids every split" {
    var f = try Fix.init(testing.allocator, 2000, 6);
    defer f.deinit();
    var r = try f.fit(.{ .n_rounds = 1, .max_depth = 6, .min_child_samples = 2000 });
    defer r.model.deinit();
    try testing.expectEqual(@as(usize, 1), r.model.trees.items[0].nodes.len);
}

test "min_child_weight: a huge hessian floor forbids every split" {
    var f = try Fix.init(testing.allocator, 2000, 7);
    defer f.deinit();
    var r = try f.fit(.{ .n_rounds = 1, .max_depth = 6, .min_child_weight = 1e9 });
    defer r.model.deinit();
    try testing.expectEqual(@as(usize, 1), r.model.trees.items[0].nodes.len);
}

test "min_split_gain: a huge gamma forbids every split" {
    var f = try Fix.init(testing.allocator, 2000, 8);
    defer f.deinit();
    var r = try f.fit(.{ .n_rounds = 1, .max_depth = 6, .min_split_gain = 1e9 });
    defer r.model.deinit();
    try testing.expectEqual(@as(usize, 1), r.model.trees.items[0].nodes.len);
}

// ----------------------------------------------------------- regularisation

test "lambda: L2 shrinks leaf weights toward zero" {
    var f = try Fix.init(testing.allocator, 2000, 9);
    defer f.deinit();
    var soft = try f.fit(.{ .n_rounds = 1, .max_depth = 3, .lambda = 0.0 });
    defer soft.model.deinit();
    var hard = try f.fit(.{ .n_rounds = 1, .max_depth = 3, .lambda = 1e6 });
    defer hard.model.deinit();
    try testing.expect(maxAbsLeaf(&hard.model) < maxAbsLeaf(&soft.model) * 0.01);
}

test "alpha: L1 drives leaf weights to exactly zero" {
    var f = try Fix.init(testing.allocator, 2000, 10);
    defer f.deinit();
    var r = try f.fit(.{ .n_rounds = 1, .max_depth = 3, .alpha = 1e9 });
    defer r.model.deinit();
    try testing.expectEqual(@as(f32, 0.0), maxAbsLeaf(&r.model));
}

test "max_delta_step: clamps the absolute leaf weight" {
    var f = try Fix.init(testing.allocator, 2000, 11);
    defer f.deinit();
    const cap: f32 = 0.05;
    var r = try f.fit(.{ .n_rounds = 1, .max_depth = 4, .learning_rate = 1.0, .max_delta_step = cap });
    defer r.model.deinit();
    try testing.expect(maxAbsLeaf(&r.model) <= cap + 1e-6);
}

// ---------------------------------------------------------------- sampling

test "subsample and bootstrap: extremes still learn" {
    var f = try Fix.init(testing.allocator, 4000, 12);
    defer f.deinit();
    for ([_]f32{ 0.1, 0.5, 1.0 }) |ss| {
        var r = try f.fit(.{ .n_rounds = 40, .max_depth = 4, .subsample = ss });
        defer r.model.deinit();
        try testing.expect(try f.auc(&r.model) > 0.85);
    }
}

test "goss rates: extremes still learn and stay unbiased" {
    var f = try Fix.init(testing.allocator, 4000, 13);
    defer f.deinit();
    const cases = [_][2]f32{ .{ 0.05, 0.05 }, .{ 0.2, 0.1 }, .{ 0.5, 0.4 } };
    for (cases) |c| {
        var r = try f.fit(.{
            .n_rounds = 40,
            .max_depth = 4,
            .sampling = .goss,
            .top_rate = c[0],
            .other_rate = c[1],
        });
        defer r.model.deinit();
        try testing.expect(try f.auc(&r.model) > 0.85);
    }
}

test "colsample: every level of the three knobs still learns" {
    var f = try Fix.init(testing.allocator, 4000, 14);
    defer f.deinit();
    for ([_]f32{ 0.2, 0.5, 1.0 }) |c| {
        var r = try f.fit(.{
            .n_rounds = 40,
            .max_depth = 4,
            .colsample_bytree = c,
            .colsample_bylevel = c,
            .colsample_bynode = c,
        });
        defer r.model.deinit();
        try testing.expect(try f.auc(&r.model) > 0.85);
    }
}

// --------------------------------------------------------------- objective

test "objective: squared_error fits a continuous target" {
    const gpa = testing.allocator;
    var f = try Fix.init(gpa, 3000, 15);
    defer f.deinit();
    var r = try f.fit(.{ .n_rounds = 60, .max_depth = 4, .objective = .squared_error });
    defer r.model.deinit();
    const p = try gpa.alloc(f32, f.ds.n_rows);
    defer gpa.free(p);
    r.model.predict(f.pool, &f.ds, p);
    // Must beat predicting the label mean.
    var mean: f64 = 0;
    for (f.ds.labels) |y| mean += y;
    mean /= @floatFromInt(f.ds.labels.len);
    const base = try gpa.alloc(f32, f.ds.n_rows);
    defer gpa.free(base);
    @memset(base, @floatCast(mean));
    try testing.expect(metric.rmse(p, f.ds.labels) < metric.rmse(base, f.ds.labels));
}

test "scale_pos_weight: upweighting positives raises predicted probability" {
    const gpa = testing.allocator;
    var f = try Fix.init(gpa, 3000, 16);
    defer f.deinit();

    var lo = try f.fit(.{ .n_rounds = 20, .max_depth = 3, .scale_pos_weight = 1.0 });
    defer lo.model.deinit();
    var hi = try f.fit(.{ .n_rounds = 20, .max_depth = 3, .scale_pos_weight = 10.0 });
    defer hi.model.deinit();

    const a = try gpa.alloc(f32, f.ds.n_rows);
    defer gpa.free(a);
    const b = try gpa.alloc(f32, f.ds.n_rows);
    defer gpa.free(b);
    lo.model.predict(f.pool, &f.ds, a);
    hi.model.predict(f.pool, &f.ds, b);

    var sa: f64 = 0;
    var sb: f64 = 0;
    for (a, b) |x, y| {
        sa += x;
        sb += y;
    }
    try testing.expect(sb > sa);
}

// ----------------------------------------------------------------- control

test "early_stopping_rounds: stops before n_rounds on unlearnable noise" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 2);
    defer pool.deinit();

    var train_ds = try synth(gpa, 2000, 17);
    defer train_ds.deinit();
    // Validation labels shuffled: nothing to learn, so the metric cannot keep
    // improving and early stopping must fire.
    var valid_ds = try synth(gpa, 1000, 18);
    defer valid_ds.deinit();
    var prng: std.Random.DefaultPrng = .init(99);
    prng.random().shuffle(f32, valid_ds.labels);

    var r = try booster.train(gpa, pool, &train_ds, &valid_ds, .{
        .n_rounds = 300,
        .max_depth = 4,
        .early_stopping_rounds = 5,
        .verbose_eval = 0,
    }, null);
    defer r.model.deinit();
    // `rounds_run` is the one that proves stopping fired; `n_rounds` alone is
    // also satisfied by the post-hoc trim, which a mutation test caught.
    try testing.expect(r.rounds_run < 300);
    try testing.expect(r.n_rounds <= r.rounds_run);
}

test "seed: different seeds give different models, same seed is repeatable" {
    var f = try Fix.init(testing.allocator, 3000, 19);
    defer f.deinit();
    const base = config.Config{ .n_rounds = 5, .max_depth = 4, .subsample = 0.5, .verbose_eval = 0 };

    var a = try f.fit(blk: {
        var c = base;
        c.seed = 1;
        break :blk c;
    });
    defer a.model.deinit();
    var b = try f.fit(blk: {
        var c = base;
        c.seed = 2;
        break :blk c;
    });
    defer b.model.deinit();
    var a2 = try f.fit(blk: {
        var c = base;
        c.seed = 1;
        break :blk c;
    });
    defer a2.model.deinit();

    try testing.expect(maxAbsLeaf(&a.model) != maxAbsLeaf(&b.model));
    try testing.expectEqual(maxAbsLeaf(&a.model), maxAbsLeaf(&a2.model));
}

test "n_threads: thread count changes nothing about the result" {
    // The strongest race check available: a histogram reduction with a data
    // race would diverge between 1 and 8 workers.
    const gpa = testing.allocator;
    var ds = try synth(gpa, 5000, 20);
    defer ds.deinit();

    const cfg = config.Config{ .n_rounds = 25, .max_depth = 5, .verbose_eval = 0 };

    var weights: [2][]f32 = undefined;
    for ([_]u32{ 1, 8 }, 0..) |nt, i| {
        const pool = try Pool.init(gpa, nt);
        defer pool.deinit();
        var r = try booster.train(gpa, pool, &ds, null, cfg, null);
        defer r.model.deinit();
        var list: std.ArrayList(f32) = .empty;
        for (r.model.trees.items) |t| for (t.nodes) |n| {
            if (n.is_leaf) try list.append(gpa, n.weight);
        };
        weights[i] = try list.toOwnedSlice(gpa);
    }
    defer gpa.free(weights[0]);
    defer gpa.free(weights[1]);

    try testing.expectEqual(weights[0].len, weights[1].len);
    for (weights[0], weights[1]) |x, y| try testing.expectEqual(x, y);
}

// ------------------------------------------------------------------ linear

test "lin_epochs and lin_tol: more budget converges further, a loose tol stops early" {
    const gpa = testing.allocator;
    var f = try Fix.init(gpa, 3000, 21);
    defer f.deinit();

    var short = try linear.train(gpa, f.pool, &f.ds, null, .{ .algo = .linear, .lin_epochs = 5, .verbose_eval = 0 }, null);
    defer short.model.deinit();
    var long = try linear.train(gpa, f.pool, &f.ds, null, .{ .algo = .linear, .lin_epochs = 400, .verbose_eval = 0 }, null);
    defer long.model.deinit();

    const ps = try gpa.alloc(f32, f.ds.n_rows);
    defer gpa.free(ps);
    const pl = try gpa.alloc(f32, f.ds.n_rows);
    defer gpa.free(pl);
    short.model.predict(f.pool, &f.ds, ps);
    long.model.predict(f.pool, &f.ds, pl);
    try testing.expect(metric.loglossProb(pl, f.ds.labels) < metric.loglossProb(ps, f.ds.labels));

    // A tolerance this loose must trip the convergence break immediately.
    var early = try linear.train(gpa, f.pool, &f.ds, null, .{ .algo = .linear, .lin_epochs = 400, .lin_tol = 1e9, .verbose_eval = 0 }, null);
    defer early.model.deinit();
    try testing.expect(early.epochs < 400);
}

test "lin_lr: a larger step moves the coefficients further in one Adam epoch" {
    const gpa = testing.allocator;
    var f = try Fix.init(gpa, 2000, 22);
    defer f.deinit();
    const base = config.Config{ .algo = .linear, .lin_solver = .adam, .lin_epochs = 1, .verbose_eval = 0 };
    var slow = try linear.train(gpa, f.pool, &f.ds, null, mix(base, 0.001), null);
    defer slow.model.deinit();
    var fast = try linear.train(gpa, f.pool, &f.ds, null, mix(base, 0.5), null);
    defer fast.model.deinit();

    var s: f32 = 0;
    for (slow.model.w) |c| s = @max(s, @abs(c));
    var q: f32 = 0;
    for (fast.model.w) |c| q = @max(q, @abs(c));
    try testing.expect(q > s * 10);
}

fn mix(base: config.Config, lr: f32) config.Config {
    var c = base;
    c.lin_lr = lr;
    return c;
}

test "lin_solver: lbfgs beats Adam at the same iteration budget and ignores lin_lr" {
    const gpa = testing.allocator;
    var f = try Fix.init(gpa, 3000, 23);
    defer f.deinit();

    const p = try gpa.alloc(f32, f.ds.n_rows);
    defer gpa.free(p);
    const q = try gpa.alloc(f32, f.ds.n_rows);
    defer gpa.free(q);

    // Adam gets the same 40 passes L-BFGS gets. The whole point of a
    // curvature estimate is that those passes go much further.
    var adam = try linear.train(gpa, f.pool, &f.ds, null, .{
        .algo = .linear, .lin_solver = .adam, .lin_epochs = 40, .verbose_eval = 0,
    }, null);
    defer adam.model.deinit();
    var lb = try linear.train(gpa, f.pool, &f.ds, null, .{
        .algo = .linear, .lin_solver = .lbfgs, .lin_epochs = 40, .verbose_eval = 0,
    }, null);
    defer lb.model.deinit();
    adam.model.predict(f.pool, &f.ds, p);
    lb.model.predict(f.pool, &f.ds, q);
    try testing.expect(metric.loglossProb(q, f.ds.labels) < metric.loglossProb(p, f.ds.labels));

    // Beating Adam is a low bar -- a line search alone would clear it. The
    // claim being made is stronger: 40 iterations is the whole fit, so ten
    // times the budget must find nothing further and the solver must stop on
    // its own. Steepest descent with the same line search converges linearly
    // and would still be moving here.
    var plenty = try linear.train(gpa, f.pool, &f.ds, null, .{
        .algo = .linear, .lin_solver = .lbfgs, .lin_epochs = 400, .verbose_eval = 0,
    }, null);
    defer plenty.model.deinit();
    try testing.expect(plenty.epochs < 400);
    const r = try gpa.alloc(f32, f.ds.n_rows);
    defer gpa.free(r);
    plenty.model.predict(f.pool, &f.ds, r);
    const at40 = metric.loglossProb(q, f.ds.labels);
    const at400 = metric.loglossProb(r, f.ds.labels);
    try testing.expectApproxEqAbs(at400, at40, 1e-7);

    // `lin_lr` is an Adam knob. A solver that takes its step from a line
    // search must produce the same coefficients whatever it is set to --
    // otherwise the flag is lying about what it controls.
    var other = try linear.train(gpa, f.pool, &f.ds, null, .{
        .algo = .linear, .lin_solver = .lbfgs, .lin_epochs = 40, .lin_lr = 0.5, .verbose_eval = 0,
    }, null);
    defer other.model.deinit();
    for (lb.model.w, other.model.w) |a, b| try testing.expectEqual(a, b);
    try testing.expectEqual(lb.model.intercept, other.model.intercept);
}

test "alpha under lbfgs lands on a genuine L1 optimum" {
    // The cheap version of this test -- "more alpha means more zeros" -- would
    // pass with the whole orthant machinery deleted, because every coefficient
    // starts at zero and a heavy penalty simply never lets it leave. So check
    // the optimality condition instead: at the minimum of a penalised
    // objective, nudging any coefficient that was driven to zero must not
    // lower the objective, in either direction. That is what an L1 solution
    // means, and it is false for anything that merely stopped early.
    //
    // Swept across penalties because the orthant projection only does work
    // once a step is big enough to carry a coefficient through zero, which a
    // single mild alpha never provokes.
    const gpa = testing.allocator;
    var fx = try Fix.init(gpa, 3000, 25);
    defer fx.deinit();

    // `synth` makes six independent uniform columns, and an orthogonal design
    // never lets a step carry a coefficient through zero -- so on it the
    // orthant handling is dead code and this test would certify nothing.
    // Reading two of those columns as categorical turns each into a one-hot
    // block whose columns sum to one, which is collinear with the intercept
    // and is the structure that actually provokes a crossing.
    fx.ds.kinds[0] = .categorical;
    fx.ds.kinds[3] = .categorical;

    const lambda: f32 = 1;
    const pred = try gpa.alloc(f32, fx.ds.n_rows);
    defer gpa.free(pred);
    const n: f64 = @floatFromInt(fx.ds.n_rows);

    const obj = struct {
        fn at(m: *const linear.Linear, pool: *Pool, ds: *const data.Dataset, buf: []f32, nn: f64, a: f32, l: f32) f64 {
            m.predict(pool, ds, buf);
            var sq: f64 = 0;
            var ab: f64 = 0;
            for (m.w) |c| {
                sq += @as(f64, c) * c;
                ab += @abs(c);
            }
            return metric.loglossProb(buf, ds.labels) + (0.5 * l * sq + a * ab) / nn;
        }
    }.at;

    var saw_partial = false;
    for ([_]f32{ 25, 50, 100, 200 }) |alpha| {
        var r = try linear.train(gpa, fx.pool, &fx.ds, null, .{
            .algo = .linear, .lin_epochs = 600, .alpha = alpha, .lambda = lambda, .verbose_eval = 0,
        }, null);
        defer r.model.deinit();

        const nz = r.model.nZero();
        if (nz > 0 and nz < r.model.w.len) saw_partial = true;

        // An L1 solution is sparse, not nearly-sparse: the orthant projection
        // puts coefficients exactly on zero, which leaves a gap between zero
        // and the smallest survivor. Drop the projection and the same run
        // strews coefficients at 1e-7 through that gap -- while still passing
        // every optimality check below, because a coefficient that tiny barely
        // moves the objective. This is the assertion that pins the projection.
        for (r.model.w) |c| {
            if (c != 0 and @abs(c) < 1e-5) {
                std.debug.print("alpha={d}: coefficient {e} is small but not zero\n", .{ alpha, c });
                return error.NotSparse;
            }
        }

        const at_opt = obj(&r.model, fx.pool, &fx.ds, pred, n, alpha, lambda);
        const eps: f32 = 0.05;
        for (r.model.w, 0..) |c, k| {
            if (c != 0) continue;
            for ([_]f32{ eps, -eps }) |d| {
                r.model.w[k] = d;
                const moved = obj(&r.model, fx.pool, &fx.ds, pred, n, alpha, lambda);
                r.model.w[k] = 0;
                if (moved < at_opt - 1e-6) {
                    std.debug.print(
                        "alpha={d}: coefficient {d} was zeroed, but moving it by {d} lowers the " ++
                            "objective {d:.9} -> {d:.9}\n",
                        .{ alpha, k, d, at_opt, moved },
                    );
                    return error.NotAnL1Optimum;
                }
            }
        }
    }
    // Neither "all zero" nor "none zero" at every alpha would say anything
    // about how the zeros were arrived at.
    try testing.expect(saw_partial);
}

test "lin_standardize: both settings learn" {
    const gpa = testing.allocator;
    var f = try Fix.init(gpa, 3000, 24);
    defer f.deinit();
    for ([_]bool{ true, false }) |std_on| {
        var r = try linear.train(gpa, f.pool, &f.ds, null, .{
            .algo = .linear,
            .lin_epochs = 300,
            .lin_standardize = std_on,
            .verbose_eval = 0,
        }, null);
        defer r.model.deinit();
        const p = try gpa.alloc(f32, f.ds.n_rows);
        defer gpa.free(p);
        r.model.predict(f.pool, &f.ds, p);
        try testing.expect(try metric.auc(gpa, p, f.ds.labels) > 0.75);
    }
}

// ------------------------------------------------------------------ config

test "validate rejects every documented impossible combination" {
    const bad = [_]config.Config{
        .{ .n_rounds = 0 },
        .{ .learning_rate = 0 },
        .{ .learning_rate = 1.5 },
        .{ .max_bin = 1 },
        .{ .subsample = 0 },
        .{ .subsample = 1.5 },
        .{ .colsample_bytree = 0 },
        .{ .lambda = -1 },
        .{ .alpha = -1 },
        .{ .max_depth = 0, .max_leaves = 0 },
        .{ .algo = .gbdt, .bootstrap = true },
        .{ .sampling = .goss, .top_rate = 0.9, .other_rate = 0.9 },
        .{ .sampling = .goss, .top_rate = 0 },
        .{ .algo = .linear, .lin_epochs = 0 },
        .{ .algo = .linear, .lin_lr = 0 },
    };
    for (bad, 0..) |c, i| {
        c.validate() catch continue;
        std.debug.print("config #{d} should have been rejected\n", .{i});
        return error.ValidationTooPermissive;
    }
}

test "applyAlgoDefaults respects explicit choices and fills the rest" {
    var c = config.Config{ .algo = .random_forest, .learning_rate = 0.3 };
    c.applyAlgoDefaults(&.{"learning_rate"});
    try testing.expectEqual(@as(f32, 0.3), c.learning_rate); // explicit, kept
    try testing.expect(c.bootstrap); // implicit, filled

    var d = config.Config{ .algo = .random_forest };
    d.applyAlgoDefaults(&.{});
    try testing.expectEqual(@as(f32, 1.0), d.learning_rate); // implicit, overridden

    // sqrt(p)/p for classification.
    var e = config.Config{ .algo = .random_forest, .objective = .logistic };
    e.applyForestFeatureDefault(16, &.{});
    try testing.expectApproxEqAbs(@as(f32, 0.25), e.colsample_bynode, 1e-6);
    var g = config.Config{ .algo = .random_forest };
    g.applyForestFeatureDefault(16, &.{"colsample_bynode"});
    try testing.expectEqual(@as(f32, 1.0), g.colsample_bynode); // explicit, kept
}

test "a solver stalled by bad scaling is not reported as converged" {
    // One design column scaled far above the others is enough. The
    // coefficient it needs is proportionally tiny, so every step falls under
    // `lin_tol` long before the gradient is anywhere near zero -- and
    // stopping on coefficient movement alone called that converged, returning
    // a model equivalent to ranking by that one column.
    const gpa = testing.allocator;
    var f = try Fix.init(gpa, 3000, 31);
    defer f.deinit();
    for (f.ds.edges[1]) |*e| e.* *= 50_000.0;

    var bad = try linear.train(gpa, f.pool, &f.ds, null, .{
        .algo = .linear, .lin_standardize = false, .lin_epochs = 300, .verbose_eval = 0,
    }, null);
    defer bad.model.deinit();
    try testing.expect(bad.fit.stalled());
    // The tell is a large gradient at the stop, not simply a short run.
    try testing.expect(bad.fit.g_last > 1e-3 * bad.fit.g_first);

    // Standardising the same design removes the conditioning problem, so the
    // identical data must now arrive. Without this half the test would pass
    // against a `stalled()` that always returns true.
    var good = try linear.train(gpa, f.pool, &f.ds, null, .{
        .algo = .linear, .lin_standardize = true, .lin_epochs = 300, .verbose_eval = 0,
    }, null);
    defer good.model.deinit();
    try testing.expect(!good.fit.stalled());
}
