//! K-fold cross-validation in a single process.
//!
//! `train` fits one model against one split, so cross-validating it from
//! outside means re-reading and re-binning the same CSV once per fold -- ten
//! times per configuration, counting the `predict` calls, for data that never
//! changes between them. A hyperparameter search pays that on every trial.
//!
//! This reads and bins once, then loops the folds over the already-binned
//! matrix. What it prints is the out-of-fold score: every row predicted by the
//! one model that did not train on it, pooled and scored as a single vector.
//! That is a stricter number than the mean of the per-fold scores and the one
//! worth tuning against, because it is computed on every row exactly once.
//!
//! Predictions are pooled on the natural scale (probabilities), not raw
//! log-odds. Each fold is a different model with its own base score, and
//! AUC over a pooled vector compares rows *across* folds -- so the scale has
//! to mean the same thing in all of them.

const std = @import("std");
const builtin = @import("builtin");
const config = @import("config.zig");
const data = @import("data.zig");
const pool_mod = @import("pool.zig");
const booster = @import("booster.zig");
const forest = @import("forest.zig");
const linear = @import("linear.zig");
const metric = @import("metric.zig");

pub const usage =
    \\usage: zgbdt cv <train.csv> --label=<column> [options]
    \\
    \\  --folds=N           number of folds (default 5)
    \\  --fold-seed=N       seed for the fold assignment (default 1)
    \\  --label=NAME        target column (required)
    \\  --pos-label=NAME    class of a string target to encode as 1
    \\  --drop=NAME         exclude a column; repeatable
    \\  --oof=FILE          write the out-of-fold predictions as CSV
    \\  --max-bytes=N       CSV size cap in bytes (default 1<<31)
    \\  --quiet=1           print only the summary line
    \\
    \\Every Config flag works here too, so a search can vary the model
    \\without changing anything else:
    \\  zgbdt cv train.csv --label=y --n_rounds=500 --max_depth=4 --lambda=45
    \\
    \\Folds are stratified on the label for a classification objective, so
    \\each fold holds the same class balance as the whole file.
    \\
;

/// Assign every row a fold. For logistic the assignment is stratified: the
/// two classes are shuffled and dealt out separately, so a fold cannot draw
/// an unrepresentative share of a rare positive class. Dealing round-robin
/// from a shuffled list also keeps the folds within one row of equal size.
pub fn assignFolds(
    gpa: std.mem.Allocator,
    labels: []const f32,
    n_folds: u32,
    seed: u64,
    stratify: bool,
) ![]u32 {
    const n = labels.len;
    const fold = try gpa.alloc(u32, n);
    errdefer gpa.free(fold);

    var order = try gpa.alloc(u32, n);
    defer gpa.free(order);

    var head: usize = 0;
    if (stratify) {
        // Positives first, then negatives; each group is shuffled on its own
        // and dealt from the same rotating counter, which interleaves them.
        for (labels, 0..) |y, i| if (y >= 0.5) {
            order[head] = @intCast(i);
            head += 1;
        };
        var tail = head;
        for (labels, 0..) |y, i| if (y < 0.5) {
            order[tail] = @intCast(i);
            tail += 1;
        };
    } else {
        for (order, 0..) |*p, i| p.* = @intCast(i);
        head = n;
    }

    var prng: std.Random.DefaultPrng = .init(seed);
    const r = prng.random();
    shuffle(r, order[0..head]);
    if (head < n) shuffle(r, order[head..]);

    var k: u32 = 0;
    for (order) |row| {
        fold[row] = k;
        k += 1;
        if (k == n_folds) k = 0;
    }
    return fold;
}

fn shuffle(r: std.Random, xs: []u32) void {
    var i: usize = xs.len;
    while (i > 1) {
        i -= 1;
        const j = r.uintLessThan(usize, i + 1);
        std.mem.swap(u32, &xs[i], &xs[j]);
    }
}


/// What one cross-validation produced. `pooled` is the out-of-fold score over
/// every evaluated row at once; `mean`/`sd` describe the spread between folds.
/// They are different numbers and answer different questions: tune against
/// `pooled`, judge whether a difference is real against `sd`.
pub const Outcome = struct {
    pooled: f64,
    mean: f64,
    sd: f64,
    fit_ms: i64,
    folds_run: u32,
};

pub const Opts = struct {
    /// Evaluate only the first N folds. A search uses this as a cheap
    /// approximation: a hopeless configuration reveals itself on two folds
    /// and need not be charged for five. 0 means all of them.
    use_folds: u32 = 0,
    /// Filled with the out-of-fold prediction for every evaluated row.
    oof: ?[]f32 = null,
    /// One line per fold while it runs.
    progress: ?*std.Io.Writer = null,
};

/// Higher is better for AUC, lower is better for RMSE. Everything that ranks
/// configurations goes through here so the comparison cannot drift from the
/// objective.
pub fn better(obj: config.Objective, a: f64, b: f64) bool {
    return switch (obj) {
        .logistic => a > b,
        .squared_error => a < b,
    };
}

/// Cross-validate one configuration over an already-binned dataset.
///
/// The dataset and the fold assignment are inputs rather than things this
/// computes, which is the whole point: a search binds them once and then pays
/// only for fitting, instead of re-reading a CSV per configuration.
pub fn crossValidate(
    gpa: std.mem.Allocator,
    io: std.Io,
    pool: *pool_mod.Pool,
    full: *const data.Dataset,
    cfg: config.Config,
    fold_of: []const u32,
    n_folds: u32,
    opts: Opts,
) !Outcome {
    const run_folds = if (opts.use_folds == 0) n_folds else @min(opts.use_folds, n_folds);

    const perm = try gpa.alloc(u32, full.n_rows);
    defer gpa.free(perm);
    const per_fold = try gpa.alloc(f64, run_folds);
    defer gpa.free(per_fold);

    // Only the rows belonging to a fold that actually ran are scored, so a
    // partial-fidelity result is an honest score on a subset rather than a
    // full-length vector padded with zeros.
    var scored: std.ArrayList(u32) = .empty;
    defer scored.deinit(gpa);

    const t0 = std.Io.Timestamp.now(io, .awake).toNanoseconds();
    for (0..run_folds) |k| {
        var head: usize = 0;
        var tail: usize = full.n_rows;
        for (fold_of, 0..) |f, i| {
            if (f == k) {
                tail -= 1;
                perm[tail] = @intCast(i);
            } else {
                perm[head] = @intCast(i);
                head += 1;
            }
        }
        std.mem.reverse(u32, perm[head..]);
        if (head == 0) return error.FoldLeftNoTrainingRows;

        var train_ds = try data.subset(gpa, full, perm[0..head]);
        defer train_ds.deinit();
        var valid_ds = try data.subset(gpa, full, perm[head..]);
        defer valid_ds.deinit();

        const scores = try gpa.alloc(f32, valid_ds.n_rows);
        defer gpa.free(scores);

        switch (cfg.algo) {
            .gbdt => {
                var res = try booster.train(gpa, pool, &train_ds, null, cfg, opts.progress);
                defer res.model.deinit();
                res.model.predict(pool, &valid_ds, scores);
            },
            .random_forest => {
                var res = try forest.train(gpa, pool, &train_ds, null, cfg, opts.progress);
                defer res.model.deinit();
                res.model.predict(pool, &valid_ds, scores);
            },
            .linear => {
                var res = try linear.train(gpa, pool, &train_ds, null, cfg, opts.progress);
                defer res.model.deinit();
                res.model.predict(pool, &valid_ds, scores);
            },
        }

        if (opts.oof) |o| for (perm[head..], scores) |row, s| {
            o[row] = s;
        };
        try scored.appendSlice(gpa, perm[head..]);

        per_fold[k] = switch (cfg.objective) {
            .logistic => try metric.auc(gpa, scores, valid_ds.labels),
            .squared_error => metric.rmse(scores, valid_ds.labels),
        };
        if (opts.progress) |w| {
            try w.print("fold {d}  {d} train / {d} valid   {s}={d:.6}\n", .{
                k,                 head,
                valid_ds.n_rows,   if (cfg.objective == .logistic) "auc" else "rmse",
                per_fold[k],
            });
            try w.flush();
        }
    }
    const fit_ms: i64 = @intCast(@divTrunc(
        std.Io.Timestamp.now(io, .awake).toNanoseconds() - t0,
        1_000_000,
    ));

    var mean: f64 = 0;
    for (per_fold) |v| mean += v;
    mean /= @floatFromInt(run_folds);
    var sd: f64 = 0;
    for (per_fold) |v| sd += (v - mean) * (v - mean);
    sd = @sqrt(sd / @as(f64, @floatFromInt(run_folds)));

    // Pool over exactly the rows that were predicted.
    const ps = try gpa.alloc(f32, scored.items.len);
    defer gpa.free(ps);
    const ys = try gpa.alloc(f32, scored.items.len);
    defer gpa.free(ys);
    const src = opts.oof orelse return error.PooledScoreNeedsOofBuffer;
    for (scored.items, 0..) |row, i| {
        ps[i] = src[row];
        ys[i] = full.labels[row];
    }
    const pooled = switch (cfg.objective) {
        .logistic => try metric.auc(gpa, ps, ys),
        .squared_error => metric.rmse(ps, ys),
    };

    return .{ .pooled = pooled, .mean = mean, .sd = sd, .fit_ms = fit_ms, .folds_run = run_folds };
}

pub fn run(init: std.process.Init, gpa: std.mem.Allocator, out: *std.Io.Writer) !void {
    const io = init.io;

    var cfg: config.Config = .{};
    var csv_path: ?[]const u8 = null;
    var label: ?[]const u8 = null;
    var pos_label: ?[]const u8 = null;
    var oof_path: ?[]const u8 = null;
    var max_bytes: usize = 1 << 31;
    var n_folds: u32 = 5;
    var fold_seed: u64 = 1;
    var quiet = false;

    var drops: std.ArrayList([]const u8) = .empty;
    defer drops.deinit(gpa);
    var explicit: std.ArrayList([]const u8) = .empty;
    defer explicit.deinit(gpa);

    var it = std.process.Args.Iterator.init(init.minimal.args);
    _ = it.skip();
    _ = it.skip(); // "cv"
    while (it.next()) |arg| {
        if (!std.mem.startsWith(u8, arg, "--")) {
            csv_path = arg;
            continue;
        }
        const body = arg[2..];
        const eq = std.mem.indexOfScalar(u8, body, '=') orelse return error.FlagNeedsValue;
        const key = body[0..eq];
        const val = body[eq + 1 ..];
        if (std.mem.eql(u8, key, "label")) {
            label = val;
        } else if (std.mem.eql(u8, key, "pos-label")) {
            pos_label = val;
        } else if (std.mem.eql(u8, key, "drop")) {
            try drops.append(gpa, val);
        } else if (std.mem.eql(u8, key, "folds")) {
            n_folds = try std.fmt.parseInt(u32, val, 10);
        } else if (std.mem.eql(u8, key, "fold-seed")) {
            fold_seed = try std.fmt.parseInt(u64, val, 10);
        } else if (std.mem.eql(u8, key, "oof")) {
            oof_path = val;
        } else if (std.mem.eql(u8, key, "max-bytes")) {
            max_bytes = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "quiet")) {
            quiet = std.mem.eql(u8, val, "1") or std.mem.eql(u8, val, "true");
        } else if (try config.applyFlag(&cfg, key, val)) {
            try explicit.append(gpa, key);
        } else {
            try out.print("unknown flag: --{s}\n\n", .{key});
            try out.writeAll(usage);
            try out.flush();
            return error.UnknownFlag;
        }
    }

    const path = csv_path orelse {
        try out.writeAll(usage);
        try out.flush();
        return error.NoInput;
    };
    const target = label orelse {
        try out.writeAll(usage);
        try out.flush();
        return error.NoLabel;
    };
    if (n_folds < 2) return error.TooFewFolds;

    cfg.applyAlgoDefaults(explicit.items);
    // Per-round validation output would be one block per fold and says
    // nothing the fold summary does not.
    cfg.verbose_eval = 0;

    const pool = try pool_mod.Pool.init(gpa, cfg.n_threads);
    defer pool.deinit();

    const t0 = std.Io.Timestamp.now(io, .awake).toNanoseconds();
    var frame = try data.readCsv(gpa, io, pool, path, max_bytes);
    defer frame.deinit();

    const label_col = frame.columnIndex(target) orelse return error.LabelColumnNotFound;
    var enc = try data.LabelEncoder.fromColumn(gpa, &frame, label_col, pos_label);
    defer enc.deinit();

    // Bin once. Every fold is a `subset` of this matrix, which also means all
    // folds share one set of bin edges -- the edges are derived from feature
    // values only, never the label, so this leaks nothing.
    var full = try data.quantise(gpa, pool, &frame, cfg, .{ .col = label_col, .enc = &enc }, drops.items);
    defer full.deinit();
    try enc.validate(full.labels, cfg.objective);
    cfg.applyForestFeatureDefault(full.n_features, explicit.items);
    const t_bin = std.Io.Timestamp.now(io, .awake).toNanoseconds();

    if (!quiet) {
        try out.print(
            \\data    {s}
            \\rows    {d}
            \\feats   {d}  (label "{s}")
            \\folds   {d}  (seed {d}{s})
            \\threads {d}
            \\build   {s}
            \\prep    {d} ms  (read + bin, paid once for all folds)
            \\
            \\
        , .{
            path,                          full.n_rows,
            full.n_features,               target,
            n_folds,                       fold_seed,
            if (cfg.objective == .logistic) ", stratified" else "",
            pool.workerCount(),            @tagName(builtin.mode),
            @divTrunc(t_bin - t0, 1_000_000),
        });
        try out.flush();
    }

    const fold_of = try assignFolds(gpa, full.labels, n_folds, fold_seed, cfg.objective == .logistic);
    defer gpa.free(fold_of);

    const oof = try gpa.alloc(f32, full.n_rows);
    defer gpa.free(oof);

    const r = try crossValidate(gpa, io, pool, &full, cfg, fold_of, n_folds, .{
        .oof = oof,
        .progress = if (quiet) null else out,
    });

    if (!quiet) try out.writeAll("\n");
    try out.print("oof     {d:.6}   mean {d:.6}   sd {d:.6}   {d} ms\n", .{
        r.pooled, r.mean, r.sd, r.fit_ms,
    });

    if (oof_path) |op| {
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(gpa);
        try buf.appendSlice(gpa, "fold,label,prediction\n");
        var line: [128]u8 = undefined;
        for (oof, full.labels, fold_of) |p, y, k|
            try buf.appendSlice(gpa, try std.fmt.bufPrint(
                &line,
                "{d},{d},{d:.8}\n",
                .{ k, y, p },
            ));
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = op, .data = buf.items });
        try out.print("wrote   {s}\n", .{op});
    }
    try out.flush();
}


// ----- tests

const testing = std.testing;

test "folds are equally sized and cover every row exactly once" {
    const gpa = testing.allocator;
    const n = 1000;
    const labels = try gpa.alloc(f32, n);
    defer gpa.free(labels);
    for (labels, 0..) |*y, i| y.* = if (i % 5 == 0) 1 else 0;

    const fold = try assignFolds(gpa, labels, 5, 7, true);
    defer gpa.free(fold);

    var count = [_]usize{0} ** 5;
    for (fold) |k| {
        try testing.expect(k < 5);
        count[k] += 1;
    }
    for (count) |c| try testing.expectEqual(@as(usize, 200), c);
}

test "stratification keeps each fold's class balance" {
    const gpa = testing.allocator;
    const n = 10_000;
    const labels = try gpa.alloc(f32, n);
    defer gpa.free(labels);
    // 10% positive, and deliberately contiguous: an unshuffled or
    // unstratified split would hand whole folds a wildly wrong balance.
    for (labels, 0..) |*y, i| y.* = if (i < n / 10) 1 else 0;

    const fold = try assignFolds(gpa, labels, 5, 3, true);
    defer gpa.free(fold);

    var pos = [_]usize{0} ** 5;
    var tot = [_]usize{0} ** 5;
    for (fold, labels) |k, y| {
        tot[k] += 1;
        if (y >= 0.5) pos[k] += 1;
    }
    // Exactly 1000 positives over 5 folds: dealing them round-robin from
    // their own shuffled list puts 200 in each, not merely "about" 200.
    for (pos) |p| try testing.expectEqual(@as(usize, 200), p);
    for (tot) |t| try testing.expectEqual(@as(usize, 2000), t);
}

test "the fold seed changes the partition, and repeating it reproduces one" {
    const gpa = testing.allocator;
    const labels = try gpa.alloc(f32, 500);
    defer gpa.free(labels);
    for (labels, 0..) |*y, i| y.* = if (i % 3 == 0) 1 else 0;

    const a = try assignFolds(gpa, labels, 5, 1, true);
    defer gpa.free(a);
    const b = try assignFolds(gpa, labels, 5, 1, true);
    defer gpa.free(b);
    const c = try assignFolds(gpa, labels, 5, 2, true);
    defer gpa.free(c);

    try testing.expectEqualSlices(u32, a, b);
    var same: usize = 0;
    for (a, c) |x, y| same += @intFromBool(x == y);
    // Two independent 5-way partitions agree on ~1/5 of rows by chance;
    // anything near total agreement would mean the seed is not being used.
    try testing.expect(same < labels.len / 2);
}

test "unstratified assignment still partitions completely" {
    const gpa = testing.allocator;
    const labels = try gpa.alloc(f32, 333);
    defer gpa.free(labels);
    for (labels) |*y| y.* = 0.5;

    const fold = try assignFolds(gpa, labels, 4, 11, false);
    defer gpa.free(fold);

    var count = [_]usize{0} ** 4;
    for (fold) |k| count[k] += 1;
    var total: usize = 0;
    for (count) |c| {
        try testing.expect(c == 83 or c == 84); // 333 does not divide by 4
        total += c;
    }
    try testing.expectEqual(@as(usize, 333), total);
}
