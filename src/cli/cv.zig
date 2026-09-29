// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! `zarbor cv`: k-fold cross-validation from the command line (see zarbor.cv).

const std = @import("std");
const builtin = @import("builtin");
const zarbor = @import("zarbor");
const args = @import("args.zig");
const config = zarbor.config;
const data = zarbor.data;
const pool_mod = zarbor.pool;
const csv = zarbor.csv;
const cv = zarbor.cv;
const assignGroupFolds = cv.assignGroupFolds;
const assignFolds = cv.assignFolds;
const crossValidate = cv.crossValidate;

const usage =
    \\usage: zarbor cv <train.csv> --label=<column> [options]
    \\
    \\  --folds=N           number of folds (default 5)
    \\  --fold-seed=N       seed for the fold assignment (default 1)
    \\  --repeats=N         re-run with N consecutive fold seeds and report
    \\                      the spread (default 1). One fold assignment on a
    \\                      small table is mostly noise; this is how you find
    \\                      out whether a difference survives it.
    \\  --group-col=NAME    keep rows sharing this column's value in the
    \\                      same fold, and drop it as a feature. Required
    \\                      whenever a unit appears more than once (a panel,
    \\                      repeated measures) or CV measures memory, not
    \\                      generalisation.
    \\  --label=NAME        target column (required)
    \\  --pos-label=NAME    class of a string target to encode as 1
    \\  --drop=NAME         exclude a column; repeatable
    \\  --oof=FILE          write the out-of-fold predictions as CSV
    \\  --max-bytes=N       CSV size cap in bytes (default 1<<31)
    \\  --quiet=1           print only the summary line
    \\
    \\Every Config flag works here too, so a search can vary the model
    \\without changing anything else:
    \\  zarbor cv train.csv --label=y --n_rounds=500 --max_depth=4 --lambda=45
    \\
    \\Folds are stratified on the label for a classification objective, so
    \\each fold holds the same class balance as the whole file.
    \\
;

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
    var repeats: u32 = 1;
    var group_col: ?[]const u8 = null;
    var quiet = false;

    var drops: std.ArrayList([]const u8) = .empty;
    defer drops.deinit(gpa);
    var explicit: std.ArrayList([]const u8) = .empty;
    defer explicit.deinit(gpa);

    var it = args.Iterator.init(init.minimal.args, 1);
    while (it.next()) |arg| {
        const flag = switch (arg) {
            .positional => |p| {
                csv_path = p;
                continue;
            },
            .bare => return error.FlagNeedsValue,
            .flag => |f| f,
        };
        const key, const val = .{ flag.key, flag.val };
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
        } else if (std.mem.eql(u8, key, "group-col")) {
            group_col = val;
        } else if (std.mem.eql(u8, key, "oof")) {
            oof_path = val;
        } else if (std.mem.eql(u8, key, "max-bytes")) {
            max_bytes = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "repeats")) {
            repeats = try std.fmt.parseInt(u32, val, 10);
            if (repeats == 0) return error.RepeatsMustBePositive;
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

    if (!quiet) {
        const stats = try csv.profile(gpa, &frame);
        defer gpa.free(stats);
        try csv.writeSummary(out, &frame, stats);
    }

    const label_col = frame.columnIndex(target) orelse return error.LabelColumnNotFound;

    // The grouping column is a label on the rows, never a feature: leaving a
    // ZIP or subject id in the matrix invites the model to memorise it.
    var group_idx: ?usize = null;
    if (group_col) |name| {
        group_idx = frame.columnIndex(name) orelse return error.GroupColumnNotFound;
        try drops.append(gpa, name);
    }

    var enc = try data.LabelEncoder.fromColumn(gpa, &frame, label_col, pos_label);
    defer enc.deinit();

    // Bin once. Every fold is a `subset` of this matrix, which also means all
    // folds share one set of bin edges -- the edges are derived from feature
    // values only, never the label, so this leaks nothing.
    var full = data.quantise(gpa, pool, &frame, cfg, .{ .col = label_col, .enc = &enc }, drops.items) catch |err| {
        if (err == error.CategoricalTooWide) try data.explainWidth(out, &frame, cfg.max_bin, drops.items);
        return err;
    };
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
            path,                                                                                           full.n_rows,
            full.n_features,                                                                                target,
            n_folds,                                                                                        fold_seed,
            if (group_col != null) ", grouped" else if (cfg.objective == .logistic) ", stratified" else "", pool.workerCount(),
            @tagName(builtin.mode),                                                                         @divTrunc(t_bin - t0, 1_000_000),
        });
        try out.flush();
    }

    const oof = try gpa.alloc(f32, full.n_rows);
    defer gpa.free(oof);

    // Repeats loop *outside* the read and the bin, which is the whole point:
    // a fold assignment is a permutation of already-binned rows, so re-running
    // one costs a fit and nothing else. Doing it by relaunching the process
    // per seed -- which is what a shell loop does -- re-reads the CSV, rebuilds
    // every dictionary and re-quantises every column, N times over, for a
    // partition that never touched any of it.
    var pooled_buf: std.ArrayList(f64) = .empty;
    defer pooled_buf.deinit(gpa);
    // `--oof` names one partition, so it has to be one repeat's, and the
    // predictions and the fold ids must come from the *same* one. The `oof`
    // buffer is overwritten by every repeat, so keeping only the fold
    // assignment would silently pair the last repeat's predictions with the
    // first repeat's folds.
    var keep_fold: []u32 = &.{};
    defer if (keep_fold.len != 0) gpa.free(keep_fold);
    var keep_oof: []f32 = &.{};
    defer if (keep_oof.len != 0) gpa.free(keep_oof);
    var total_fit_ms: i64 = 0;

    for (0..repeats) |rep| {
        const seed = fold_seed + rep;
        const fold_of = if (group_idx) |gi|
            try assignGroupFolds(gpa, frame.values[gi], n_folds, seed)
        else
            try assignFolds(gpa, full.labels, n_folds, seed, cfg.objective == .logistic);
        var keep_this = false;
        defer if (!keep_this) gpa.free(fold_of);

        const r = try crossValidate(gpa, io, pool, &full, cfg, fold_of, n_folds, .{
            .oof = oof,
            // Per-fold progress on every repeat is 5N lines nobody reads.
            .progress = if (quiet or repeats > 1) null else out,
        });
        try pooled_buf.append(gpa, r.pooled);
        total_fit_ms += r.fit_ms;

        if (repeats == 1) {
            if (!quiet) try out.writeAll("\n");
            try out.print("oof     {d:.6}   mean {d:.6}   sd {d:.6}   {d} ms\n", .{
                r.pooled, r.mean, r.sd, r.fit_ms,
            });
        } else if (!quiet) {
            try out.print("seed {d: <4} oof {d:.6}   folds mean {d:.6} sd {d:.6}\n", .{
                seed, r.pooled, r.mean, r.sd,
            });
        }

        // `--oof` describes one partition, so it can only mean the first.
        if (rep == 0 and oof_path != null) {
            keep_fold = fold_of;
            keep_this = true;
            keep_oof = try gpa.dupe(f32, oof);
        }
    }

    if (repeats > 1) {
        const pooled = pooled_buf.items;
        var sum: f64 = 0;
        var lo = pooled[0];
        var hi = pooled[0];
        for (pooled) |v| {
            sum += v;
            lo = @min(lo, v);
            hi = @max(hi, v);
        }
        const mean = sum / @as(f64, @floatFromInt(pooled.len));
        var ss: f64 = 0;
        for (pooled) |v| ss += (v - mean) * (v - mean);
        const sd = @sqrt(ss / @as(f64, @floatFromInt(pooled.len)));
        if (!quiet) try out.writeAll("\n");
        try out.print(
            "repeats {d}   oof {d:.6}   sd {d:.6}   best {d:.6}   worst {d:.6}   {d} ms\n",
            .{ pooled.len, mean, sd, lo, hi, total_fit_ms },
        );
    }

    if (oof_path) |op| {
        // Both are the first repeat's, snapshotted together above.
        const fold_of = keep_fold;
        const oof_w = keep_oof;
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(gpa);
        try buf.appendSlice(gpa, "fold,label,prediction\n");
        var line: [128]u8 = undefined;
        if (repeats > 1) try out.writeAll("note    --oof is the first repeat only\n");
        for (oof_w, full.labels, fold_of) |p, y, k|
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
