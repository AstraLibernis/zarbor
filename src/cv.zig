// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! K-fold cross-validation in one process (the CLI is src/cli/cv.zig).
//! Driving `train` per fold from outside re-reads and re-bins the same CSV twice
//! per fold per config (`train`, then `predict`); a search pays that every trial. Here the
//! caller bins once and folds loop over that matrix. The headline is the
//! out-of-fold score: each row predicted by the model that did not train on
//! it, pooled into one vector. Stricter than the per-fold mean, and the one to
//! tune against: every row counts exactly once.
//! Pooled on the natural scale (probabilities), not log-odds: each fold has its
//! own base score and pooled AUC compares rows *across* folds.

const std = @import("std");
const config = @import("config.zig");
const data = @import("data.zig");
const pool_mod = @import("pool.zig");
const metric = @import("metric.zig");
const Objective = @import("objective.zig").Objective;
const Fitted = @import("fitted.zig").Fitted;

/// Assign every row a fold. `stratify` (for logistic) shuffles and deals each
/// class separately, so no fold gets a skewed share of a rare positive class.
/// Round-robin dealing keeps folds within one row of equal size.
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
        // Positives then negatives, each shuffled alone, dealt from one counter.
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

/// Assign folds by *group*: rows sharing a group id share a fold. Without it,
/// panel-data CV lies: a ZIP code in May and June split by row scores memory,
/// not generalisation ("new month, known place" vs "unseen place" can differ
/// by several RMSE). Groups go largest-first into the smallest fold, since
/// uneven sizes under round-robin leave folds lopsided.
pub fn assignGroupFolds(
    gpa: std.mem.Allocator,
    groups: []const f32,
    n_folds: u32,
    seed: u64,
) ![]u32 {
    const n = groups.len;
    const fold = try gpa.alloc(u32, n);
    errdefer gpa.free(fold);

    // Distinct group ids, and how many rows each holds.
    var counts: std.array_hash_map.Auto(u64, u32) = .empty;
    defer counts.deinit(gpa);
    for (groups) |g| {
        const key: u64 = @bitCast(@as(f64, g));
        const e = try counts.getOrPut(gpa, key);
        e.value_ptr.* = if (e.found_existing) e.value_ptr.* + 1 else 1;
    }
    const n_groups = counts.count();
    if (n_groups < n_folds) return error.FewerGroupsThanFolds;

    const order = try gpa.alloc(u32, n_groups);
    defer gpa.free(order);
    for (order, 0..) |*v, i| v.* = @intCast(i);

    // Shuffle, then sort by size: ties keep shuffled, not appearance, order.
    var prng: std.Random.DefaultPrng = .init(seed);
    shuffle(prng.random(), order);
    const sizes = counts.values();
    const By = struct {
        s: []const u32,
        fn gt(c: @This(), a: u32, b: u32) bool {
            return c.s[a] > c.s[b];
        }
    };
    std.sort.pdq(u32, order, By{ .s = sizes }, By.gt);

    const load = try gpa.alloc(u32, n_folds);
    defer gpa.free(load);
    @memset(load, 0);

    const of = try gpa.alloc(u32, n_groups);
    defer gpa.free(of);
    for (order) |gi| {
        var best: u32 = 0;
        for (1..n_folds) |k| if (load[k] < load[best]) {
            best = @intCast(k);
        };
        of[gi] = best;
        load[best] += sizes[gi];
    }

    for (groups, 0..) |g, i| {
        const key: u64 = @bitCast(@as(f64, g));
        // getIndex, not a scan: a linear lookup would be rows x groups.
        fold[i] = of[counts.getIndex(key).?];
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

/// `pooled`: out-of-fold score over all evaluated rows; `mean`/`sd`: spread
/// between folds. Tune against `pooled`; judge if a difference is real by `sd`.
pub const Outcome = struct {
    pooled: f64,
    mean: f64,
    sd: f64,
    fit_ms: i64,
    folds_run: u32,
    /// Mean over folds of the rounds (trees, epochs) each fold's model kept: under early
    /// stopping, the round count a final `train` should use.
    steps: f64 = 0,
    /// Whether the folds stopped early on a nested slice.
    stopped_early: bool = false,
};

/// Parts of the inner assignment; part 0 is the early-stopping slice, a tenth of each fold's
/// training rows.
pub const early_stop_parts: u32 = 10;

/// Every row's inner part, dealt like the folds themselves (by group when `groups` is given,
/// stratified for logistic), so a group never straddles the slice and the scored fold.
/// Caller frees.
pub fn assignEarlyStop(
    gpa: std.mem.Allocator,
    labels: []const f32,
    groups: ?[]const f32,
    seed: u64,
    stratify: bool,
) ![]u32 {
    const s = seed ^ 0x5EED_E5_0001;
    return if (groups) |g|
        assignGroupFolds(gpa, g, early_stop_parts, s)
    else
        assignFolds(gpa, labels, early_stop_parts, s, stratify);
}

pub const Opts = struct {
    /// Evaluate only the first N folds (0 = all): a cheap search fidelity, a
    /// hopeless config shows on two folds without paying for all `n_folds`.
    use_folds: u32 = 0,
    /// Filled with the out-of-fold prediction for every evaluated row; length
    /// `full.n_rows`. Optional in type only: `crossValidate` pools from it and
    /// returns `error.PooledScoreNeedsOofBuffer` (after fitting every fold) if null.
    oof: ?[]f32 = null,
    /// One line per fold while it runs.
    progress: ?*std.Io.Writer = null,
    /// The rows' groups, when folds are grouped: the early-stopping slice is then grouped too.
    groups: ?[]const f32 = null,
    /// Seed of the early-stopping slice; vary it with the fold seed.
    early_stop_seed: u64 = 0,
};

/// One policy's score in `chooseBinPolicy`.
pub const PolicyScore = struct { policy: data.BinPolicy, score: f64 };

/// Resolve `bin_policy = auto`: bin `frame` under each concrete policy, score each by a 3-fold
/// CV of `cfg` (grouped when `groups` is given, stratified for logistic, fold seed 1), and
/// return the best, ties to the earlier policy. Fills `scores` in `data.concrete_policies`
/// order. One measurement per dataset, made once before the run that uses it.
pub fn chooseBinPolicy(
    gpa: std.mem.Allocator,
    io: std.Io,
    pool: *pool_mod.Pool,
    frame: *const data.Frame,
    cfg: config.Config,
    label: data.LabelSpec,
    skip: []const []const u8,
    groups: ?[]const f32,
    /// Score on these rows only (`train` passes its training rows); null for every row.
    rows: ?[]const u32,
    scores: *[data.concrete_policies.len]PolicyScore,
) !data.BinPolicy {
    // `groups` is indexed by every row of `frame`; a row subset would need it subset too.
    if (groups != null and rows != null) return error.GroupsWithRowSubset;
    const folds: u32 = 3;
    var best: usize = 0;
    for (data.concrete_policies, scores, 0..) |p, *sc, i| {
        var c = cfg;
        c.bin.bin_policy = p;
        var all = try data.quantise(gpa, pool, frame, c.bin, label, skip);
        defer all.deinit();
        var part: ?data.Dataset = if (rows) |r| try data.subset(gpa, &all, r) else null;
        defer if (part) |*d| d.deinit();
        const ds = if (part) |*d| d else &all;
        const fold_of = if (groups) |g|
            try assignGroupFolds(gpa, g, folds, 1)
        else
            try assignFolds(gpa, ds.labels, folds, 1, cfg.objective() == .logistic);
        defer gpa.free(fold_of);
        const oof = try gpa.alloc(f32, ds.n_rows);
        defer gpa.free(oof);
        const o = try crossValidate(gpa, io, pool, ds, c, fold_of, folds, .{ .oof = oof, .groups = groups });
        sc.* = .{ .policy = p, .score = o.pooled };
        if (better(cfg.objective(), o.pooled, scores[best].score)) best = i;
    }
    return scores[best].policy;
}

/// Prints `chooseBinPolicy`'s table.
pub fn writePolicyScores(out: *std.Io.Writer, scores: []const PolicyScore, chosen: data.BinPolicy) !void {
    try out.writeAll("bin policy auto, 3-fold cv:\n");
    for (scores) |s| try out.print("  {s:<10} {d:.6}{s}\n", .{
        @tagName(s.policy), s.score, if (s.policy == chosen) "   <- chosen" else "",
    });
    try out.flush();
}

/// Higher AUC, lower RMSE is better. All config ranking goes through here so
/// the comparison cannot drift from the objective.
pub fn better(obj: Objective, a: f64, b: f64) bool {
    return switch (obj) {
        .logistic => a > b,
        .squared_error => a < b,
    };
}

/// Cross-validate one config over an already-binned dataset. Data and folds
/// are inputs so a search binds them once and pays only for fitting.
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
    try cfg.validate();
    const run_folds = if (opts.use_folds == 0) n_folds else @min(opts.use_folds, n_folds);

    const perm = try gpa.alloc(u32, full.n_rows);
    defer gpa.free(perm);

    // Early stopping is nested: each fold stops on a slice of its own training rows, never on
    // the rows it is scored on, which would let the score pick its own stopping point.
    const stop_early = cfg.algo == .gbdt and cfg.gbdt.early_stopping_rounds != 0;
    const es_of: ?[]u32 = if (stop_early)
        try assignEarlyStop(gpa, full.labels, opts.groups, opts.early_stop_seed, cfg.objective() == .logistic)
    else
        null;
    defer if (es_of) |e| gpa.free(e);
    const fit_rows = try gpa.alloc(u32, if (stop_early) full.n_rows else 0);
    defer gpa.free(fit_rows);
    var steps_sum: f64 = 0;
    const per_fold = try gpa.alloc(f64, run_folds);
    defer gpa.free(per_fold);

    // Only rows of folds that ran are scored: a partial-fidelity result is an
    // honest subset score, not a full vector padded with zeros.
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

        // Under early stopping the training rows split, in order, into fitting rows and the
        // slice (inner part 0) that picks the round.
        var n_fit: usize = head;
        if (es_of) |e| {
            n_fit = 0;
            for (perm[0..head]) |row| if (e[row] != 0) {
                fit_rows[n_fit] = row;
                n_fit += 1;
            };
            var at = n_fit;
            for (perm[0..head]) |row| if (e[row] == 0) {
                fit_rows[at] = row;
                at += 1;
            };
            if (n_fit == 0 or n_fit == head) return error.EarlyStopSliceEmpty;
        }
        const train_rows = if (es_of != null) fit_rows[0..n_fit] else perm[0..head];

        var train_ds = try data.subset(gpa, full, train_rows);
        defer train_ds.deinit();
        var valid_ds = try data.subset(gpa, full, perm[head..]);
        defer valid_ds.deinit();
        var stop_ds: ?data.Dataset = if (es_of != null) try data.subset(gpa, full, fit_rows[n_fit..head]) else null;
        defer if (stop_ds) |*d| d.deinit();

        const scores = try gpa.alloc(f32, valid_ds.n_rows);
        defer gpa.free(scores);

        var res = try Fitted.train(gpa, pool, &train_ds, if (stop_ds) |*d| d else null, cfg, opts.progress);
        defer res.model.deinit();
        steps_sum += @floatFromInt(res.steps);
        res.model.predict(pool, &valid_ds, scores);

        if (opts.oof) |o| for (perm[head..], scores) |row, s| {
            o[row] = s;
        };
        try scored.appendSlice(gpa, perm[head..]);

        per_fold[k] = switch (cfg.objective()) {
            .logistic => try metric.auc(gpa, scores, valid_ds.labels),
            .squared_error => metric.rmse(scores, valid_ds.labels),
        };
        if (opts.progress) |w| {
            try w.print("fold {d}  {d} train / {d} valid   {s}={d:.6}", .{
                k,               n_fit,
                valid_ds.n_rows, if (cfg.objective() == .logistic) "auc" else "rmse",
                per_fold[k],
            });
            if (stop_early) try w.print("   stopped at {d} rounds on {d} rows", .{ res.steps, head - n_fit });
            try w.writeAll("\n");
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
    const pooled = switch (cfg.objective()) {
        .logistic => try metric.auc(gpa, ps, ys),
        .squared_error => metric.rmse(ps, ys),
    };

    return .{
        .pooled = pooled,
        .mean = mean,
        .sd = sd,
        .fit_ms = fit_ms,
        .folds_run = run_folds,
        .steps = steps_sum / @as(f64, @floatFromInt(run_folds)),
        .stopped_early = stop_early,
    };
}
