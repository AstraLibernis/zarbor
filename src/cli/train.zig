// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! `zarbor <train.csv>`: fit one model against one train/valid split.

const std = @import("std");
const builtin = @import("builtin");
const zarbor = @import("zarbor");
const args = @import("args.zig");
const config = zarbor.config;
const data = zarbor.data;
const pool_mod = zarbor.pool;
const booster = zarbor.booster;
const forest = zarbor.forest;
const linear = zarbor.linear;
const metric = zarbor.metric;
const prof = zarbor.prof;
const model_mod = zarbor.model;
const csv = zarbor.csv;
const common = @import("common.zig");
const explainLabel = common.explainLabel;
const usage = @import("main.zig").usage;

pub fn run(init: std.process.Init, gpa: std.mem.Allocator, out: *std.Io.Writer) !void {
    const io = init.io;

    var cfg: config.Config = .{};
    var csv_path: ?[]const u8 = null;
    var label: ?[]const u8 = null;
    var valid_frac: f32 = 0.2;
    var split_seed: u64 = 1;
    var max_bytes: usize = 1 << 31;
    var split_col: ?[]const u8 = null;
    var save_path: ?[]const u8 = null;
    var pos_label: ?[]const u8 = null;

    var drops: std.ArrayList([]const u8) = .empty;
    defer drops.deinit(gpa);

    // Which config fields the user named. Each algo fills in the rest with
    // defaults that suit it, and must not overwrite an explicit choice.
    var explicit: std.ArrayList([]const u8) = .empty;
    defer explicit.deinit(gpa);

    var it = args.Iterator.init(init.minimal.args, 0);
    while (it.next()) |arg| {
        const flag = switch (arg) {
            .positional => |p| {
                csv_path = p;
                continue;
            },
            .bare => {
                try out.writeAll(usage);
                try out.flush();
                return error.FlagNeedsValue;
            },
            .flag => |f| f,
        };
        const key, const val = .{ flag.key, flag.val };

        if (std.mem.eql(u8, key, "label")) {
            label = val;
        } else if (std.mem.eql(u8, key, "pos-label")) {
            pos_label = val;
        } else if (std.mem.eql(u8, key, "drop")) {
            try drops.append(gpa, val);
        } else if (std.mem.eql(u8, key, "valid-frac")) {
            valid_frac = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, key, "split-seed")) {
            split_seed = try std.fmt.parseInt(u64, val, 10);
        } else if (std.mem.eql(u8, key, "save")) {
            save_path = val;
        } else if (std.mem.eql(u8, key, "profile")) {
            prof.enabled = std.mem.eql(u8, val, "1") or std.mem.eql(u8, val, "true");
        } else if (std.mem.eql(u8, key, "split-col")) {
            split_col = val;
        } else if (std.mem.eql(u8, key, "max-bytes")) {
            max_bytes = try std.fmt.parseInt(usize, val, 10);
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

    const pool = try pool_mod.Pool.init(gpa, cfg.n_threads);
    defer pool.deinit();

    const t0 = std.Io.Timestamp.now(io, .awake).toNanoseconds();

    var frame = try data.readCsv(gpa, io, pool, path, max_bytes);
    defer frame.deinit();
    const t_read = std.Io.Timestamp.now(io, .awake).toNanoseconds();

    {
        // Cheap enough to do unconditionally, and it is the line that catches
        // a file read wrong -- a column silently all-NaN, a categorical that
        // was meant to be numeric -- before a score is blamed on the model.
        const stats = try csv.profile(gpa, &frame);
        defer gpa.free(stats);
        try csv.writeSummary(out, &frame, stats);
    }

    const label_col = frame.columnIndex(target) orelse return error.LabelColumnNotFound;

    // An explicit split column assigns rows to train/validation by value
    // instead of by a random draw. It exists so an external tool can hand this
    // program exactly the same partition it used itself, which is the only way
    // to compare against another implementation without the split being a
    // confound. It is a label, never a feature.
    var split_idx: ?usize = null;
    if (split_col) |name| {
        split_idx = frame.columnIndex(name) orelse return error.SplitColumnNotFound;
        try drops.append(gpa, name);
    }

    // Fix the class order before anything reads the target. Left to the
    // dictionary, "1" would mean whichever class the first row happened to
    // hold, so the same data in a different row order would train the
    // opposite model.
    var enc = data.LabelEncoder.fromColumn(gpa, &frame, label_col, pos_label) catch |err| {
        try explainLabel(out, err, target);
        return err;
    };
    defer enc.deinit();

    var full = data.quantise(gpa, pool, &frame, cfg.bin, .{ .col = label_col, .enc = &enc }, drops.items) catch |err| {
        if (err == error.CategoricalTooWide) try data.explainWidth(out, &frame, cfg.bin.max_bin, drops.items);
        try explainLabel(out, err, target);
        return err;
    };
    defer full.deinit();
    enc.validate(full.labels, cfg.objective()) catch |err| {
        try explainLabel(out, err, target);
        return err;
    };
    const t_bin = std.Io.Timestamp.now(io, .awake).toNanoseconds();
    cfg.applyForestFeatureDefault(full.n_features, explicit.items);

    try out.print(
        \\data    {s}
        \\rows    {d}
        \\feats   {d}  (label "{s}")
        \\model   {s}
        \\threads {d}
        \\build   {s}
        \\read    {d} ms
        \\bin     {d} ms
        \\
    , .{
        path,
        full.n_rows,
        full.n_features,
        target,
        cfg.modelName(),
        pool.workerCount(),
        // `zig build` defaults to Debug, and a Debug binary is ~8x slower
        // here. Printing the mode means a benchmark can never quietly measure
        // the wrong build -- which is exactly what happened once.
        @tagName(builtin.mode),
        @divTrunc(t_read - t0, 1_000_000),
        @divTrunc(t_bin - t_read, 1_000_000),
    });
    // Print the encoding rather than leaving it implicit: which class is 1
    // decides the sign of every prediction, and it is the one thing a reader
    // cannot check from the numbers alone.
    try printEncoding(out, &enc);
    try out.flush();

    // --- train / validation split ---
    const perm = try gpa.alloc(u32, full.n_rows);
    defer gpa.free(perm);

    var n_train: usize = undefined;
    var n_valid: usize = undefined;

    if (split_idx) |sc| {
        // Train rows first, validation rows after, so the same `subset` calls
        // below work unchanged. A non-zero value means validation.
        const col = frame.values[sc];
        var head: usize = 0;
        var tail: usize = full.n_rows;
        for (col, 0..) |v, i| {
            if (v >= 0.5) {
                tail -= 1;
                perm[tail] = @intCast(i);
            } else {
                perm[head] = @intCast(i);
                head += 1;
            }
        }
        n_train = head;
        n_valid = full.n_rows - head;
        if (n_train == 0) return error.SplitColumnLeftNoTrainingRows;
    } else {
        for (perm, 0..) |*p, i| p.* = @intCast(i);
        var prng: std.Random.DefaultPrng = .init(split_seed);
        const r = prng.random();
        var i: usize = full.n_rows;
        while (i > 1) {
            i -= 1;
            const j = r.uintLessThan(usize, i + 1);
            std.mem.swap(u32, &perm[i], &perm[j]);
        }
        n_valid = @intFromFloat(@round(@as(f32, @floatFromInt(full.n_rows)) * valid_frac));
        n_train = full.n_rows - n_valid;
    }

    var train_ds = try data.subset(gpa, &full, perm[0..n_train]);
    defer train_ds.deinit();

    var valid_ds: ?data.Dataset = null;
    if (n_valid != 0) valid_ds = try data.subset(gpa, &full, perm[n_train..]);
    defer if (valid_ds) |*v| v.deinit();

    try out.print("train   {d} rows / valid {d} rows\n\n", .{ n_train, n_valid });
    try out.flush();

    const t_train0 = std.Io.Timestamp.now(io, .awake).toNanoseconds();
    const valid_ptr: ?*const data.Dataset = if (valid_ds) |*v| v else null;

    try cfg.validate();
    switch (cfg.algo) {
        .gbdt => {
            var res = try booster.train(gpa, pool, &train_ds, valid_ptr, cfg.gbdt, out);
            defer res.model.deinit();
            try printTiming(out, io, t_train0, "tree", res.n_rounds, res.valid_ns);
            if (save_path) |sp| {
                var b = try model_mod.fromBooster(gpa, &res.model, try data.Schema.fromDataset(gpa, &full));
                defer b.deinit();
                try b.setLabel(target, &enc);
                try model_mod.save(gpa, io, sp, &b);
                try out.print("saved   {s}\n", .{sp});
            }
            if (valid_ds) |*v| {
                const scores = try gpa.alloc(f32, v.n_rows);
                defer gpa.free(scores);
                // Raw log-odds: AUC is rank-based so the link does not matter,
                // and logloss wants the raw scale anyway.
                res.model.predictRaw(pool, v, scores);
                try report(gpa, out, cfg.objective(), scores, v.labels, .raw);
            }
        },
        .random_forest => {
            var res = try forest.train(gpa, pool, &train_ds, valid_ptr, cfg.random_forest, out);
            defer res.model.deinit();
            try printTiming(out, io, t_train0, "tree", res.n_trees, res.valid_ns);
            if (save_path) |sp| {
                var b = try model_mod.fromForest(gpa, &res.model, try data.Schema.fromDataset(gpa, &full));
                defer b.deinit();
                try b.setLabel(target, &enc);
                try model_mod.save(gpa, io, sp, &b);
                try out.print("saved   {s}\n", .{sp});
            }
            if (valid_ds) |*v| {
                const scores = try gpa.alloc(f32, v.n_rows);
                defer gpa.free(scores);
                res.model.predict(pool, v, scores);
                try report(gpa, out, cfg.objective(), scores, v.labels, .natural);
            }
        },
        .linear => {
            var res = try linear.train(gpa, pool, &train_ds, valid_ptr, cfg.linear, out);
            defer res.model.deinit();
            try printTiming(out, io, t_train0, "epoch", res.epochs, res.valid_ns);
            try out.print("coefs   {d} ({d} zero)\n", .{ res.model.w.len, res.model.nZero() });
            {
                if (res.fit.stalled()) {
                    // Silence here used to mean "fitted". It did not: with
                    // `--lin_standardize=false` on a column reaching 188,000,
                    // the coefficient steps fall under `lin_tol` after four
                    // iterations while the gradient is still enormous, and the
                    // result ranks by that one column and nothing else.
                    try out.print(
                        \\STALLED the solver stopped moving while the gradient was still
                        \\        large ({e:.2}, from {e:.2} at the start). These
                        \\        coefficients are not a fitted model.
                        \\
                    , .{ res.fit.g_last, res.fit.g_first });
                    if (!cfg.linear.lin_standardize)
                        try out.writeAll(
                            \\        The usual cause is unscaled columns. Try
                            \\        --lin_standardize=true.
                            \\
                        );
                    // Adam has no line search, so it cannot report "could not
                    // move"; it just runs out of schedule. A gradient still
                    // this large after the budget is as likely to mean the
                    // budget was short as that the problem is ill-scaled.
                    if (cfg.linear.lin_solver == .adam)
                        try out.writeAll(
                            \\        adam needs far more epochs than lbfgs. Try
                            \\        --lin_epochs=30000, or --lin_solver=lbfgs.
                            \\
                        );
                } else {
                    try out.print("fit     converged, |g|max {e:.2} from {e:.2}\n", .{
                        res.fit.g_last, res.fit.g_first,
                    });
                }
            }
            if (save_path) |sp| {
                // The bundle borrows the fitted model's design and weights
                // rather than copying, so it must not free them.
                var b = model_mod.Bundle{
                    .gpa = gpa,
                    .kind = .linear,
                    .schema = try data.Schema.fromDataset(gpa, &full),
                    .objective = cfg.objective(),
                    .lin = res.model,
                };
                // Dropping the borrowed model first lets the normal deinit
                // clean up everything the bundle does own.
                defer {
                    b.lin = null;
                    b.deinit();
                }
                try b.setLabel(target, &enc);
                try model_mod.save(gpa, io, sp, &b);
                try out.print("saved   {s}\n", .{sp});
            }
            if (valid_ds) |*v| {
                const scores = try gpa.alloc(f32, v.n_rows);
                defer gpa.free(scores);
                res.model.predict(pool, v, scores);
                try report(gpa, out, cfg.objective(), scores, v.labels, .natural);
            }
        },
    }
    try prof.report(out);
    try out.flush();
}

/// `valid_ns` is time spent predicting and scoring the validation set. It is
/// reported separately rather than folded in, because it is not part of
/// fitting: a benchmark against a library called with no eval set would
/// otherwise charge us for work that library never did.
fn printTiming(
    out: *std.Io.Writer,
    io: std.Io,
    t0: i128,
    comptime unit: []const u8,
    n: u32,
    valid_ns: u64,
) !void {
    const ms = @divTrunc(std.Io.Timestamp.now(io, .awake).toNanoseconds() - t0, 1_000_000);
    const vms = valid_ns / 1_000_000;
    const fit: i128 = ms - @as(i128, @intCast(vms));
    try out.print(
        \\
        \\rounds  {d}
        \\train   {d} ms  ({d:.2} ms/
    ++ unit ++ ")\n", .{
        n,
        ms,
        @as(f64, @floatFromInt(ms)) / @as(f64, @floatFromInt(@max(n, 1))),
    });
    if (vms != 0) try out.print("fit     {d} ms  (+{d} ms validating)\n", .{ fit, vms });
}

/// Which scale the predictions are on. The booster reports raw log-odds; the
/// forest and the linear model already apply their own link.
const Scale = enum { raw, natural };

fn report(
    gpa: std.mem.Allocator,
    out: *std.Io.Writer,
    obj: zarbor.objective.Objective,
    pred: []const f32,
    labels: []const f32,
    scale: Scale,
) !void {
    switch (obj) {
        .logistic => {
            const a = try metric.auc(gpa, pred, labels);
            const ll = switch (scale) {
                .raw => metric.logloss(pred, labels),
                .natural => metric.loglossProb(pred, labels),
            };
            try out.print("valid   auc={d:.6}  logloss={d:.6}\n", .{ a, ll });
        },
        .squared_error => {
            try out.print("valid   rmse={d:.6}\n", .{metric.rmse(pred, labels)});
        },
    }
}

/// Show which class became which number, or that the target was already
/// numeric and untouched.
fn printEncoding(out: *std.Io.Writer, enc: *const data.LabelEncoder) !void {
    if (enc.classes.len == 0) {
        try out.writeAll("encode  numeric target, used as-is\n");
        return;
    }
    try out.writeAll("encode  ");
    for (enc.classes, 0..) |c, i| {
        if (i != 0) try out.writeAll("  ");
        try out.print("\"{s}\"={d}", .{ c, i });
    }
    try out.writeAll("\n");
}
