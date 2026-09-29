// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

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
const config = @import("config.zig");
const data = @import("data.zig");
const pool_mod = @import("pool.zig");
const metric = @import("metric.zig");
const Objective = @import("objective.zig").Objective;
const Fitted = @import("fitted.zig").Fitted;

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

/// Assign folds by *group* rather than by row: every row sharing a group id
/// lands in the same fold.
///
/// Without this, cross-validation on panel data silently lies. If a ZIP code
/// appears in May and June and the split is by row, one month trains while
/// the other validates, and the score measures memory rather than
/// generalisation. Grouping is the difference between "how well does this
/// predict a new month for a place I know" and "how well does it predict a
/// place I have never seen" -- which can be several RMSE apart.
///
/// Groups are dealt largest-first into whichever fold is currently smallest,
/// because group sizes are usually uneven and a round-robin over a shuffled
/// list would leave the folds lopsided.
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

    // Shuffle first so equal-sized groups are not ordered by appearance, then
    // sort by size; ties keep the shuffled order.
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
        // getIndex, not a scan: the map already knows where the group is, and
        // a linear lookup here would be rows x groups.
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
pub fn better(obj: Objective, a: f64, b: f64) bool {
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
    try cfg.validate();
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

        var res = try Fitted.train(gpa, pool, &train_ds, null, cfg, opts.progress);
        defer res.model.deinit();
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
            try w.print("fold {d}  {d} train / {d} valid   {s}={d:.6}\n", .{
                k,               head,
                valid_ds.n_rows, if (cfg.objective() == .logistic) "auc" else "rmse",
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
    const pooled = switch (cfg.objective()) {
        .logistic => try metric.auc(gpa, ps, ys),
        .squared_error => metric.rmse(ps, ys),
    };

    return .{ .pooled = pooled, .mean = mean, .sd = sd, .fit_ms = fit_ms, .folds_run = run_folds };
}
