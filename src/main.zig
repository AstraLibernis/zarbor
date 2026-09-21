//! zgbdt — train a gradient-boosted tree ensemble on a CSV.
//!
//! Every field of `Config` is exposed as `--field=value` by comptime
//! reflection, so the flag surface and the struct can never drift apart.

const std = @import("std");
const config = @import("config.zig");
const data = @import("data.zig");
const pool_mod = @import("pool.zig");
const booster = @import("booster.zig");
const forest = @import("forest.zig");
const linear = @import("linear.zig");
const metric = @import("metric.zig");

const usage =
    \\usage: zgbdt <train.csv> --label=<column> [options]
    \\
    \\  --algo=NAME         gbdt | random_forest | linear   (default gbdt)
    \\  --label=NAME        target column (required)
    \\  --drop=NAME         exclude a column; repeatable
    \\  --valid-frac=F      fraction held out for validation (default 0.2)
    \\  --split-seed=N      seed for the validation split (default 1)
    \\  --max-bytes=N       CSV size cap in bytes (default 1<<31)
    \\  --split-col=NAME    column assigning rows to train(0)/valid(nonzero);
    \\                      overrides --valid-frac, and is dropped as a feature
    \\
    \\Any Config field is also a flag, e.g.:
    \\  --n_rounds=800 --learning_rate=0.05 --max_depth=7 --lambda=2.0
    \\  --grow_policy=lossguide --max_leaves=64 --subsample=0.8
    \\  --colsample_bytree=0.8 --early_stopping_rounds=50 --n_threads=16
    \\
    \\XGBoost-style is the default; LightGBM-style is:
    \\  --grow_policy=lossguide --max_leaves=64 --sampling=goss
    \\
    \\Each algo sets its own defaults for anything you do not pass:
    \\  --algo=random_forest   bagged, unshrunk, 1024-leaf trees, sqrt(p)/split
    \\  --algo=linear          Adam + L1/L2 on the binned design matrix
    \\
;

fn parseInto(comptime T: type, val: []const u8) !T {
    return switch (@typeInfo(T)) {
        .int => try std.fmt.parseInt(T, val, 10),
        .float => try std.fmt.parseFloat(T, val),
        .bool => std.mem.eql(u8, val, "true") or std.mem.eql(u8, val, "1"),
        .@"enum" => std.meta.stringToEnum(T, val) orelse error.UnknownEnumValue,
        .optional => |o| try parseInto(o.child, val),
        else => @compileError("config field type not parseable: " ++ @typeName(T)),
    };
}

/// Returns false when `key` names no config field, letting the caller fall
/// through to its own flags.
fn applyConfigFlag(cfg: *config.Config, key: []const u8, val: []const u8) !bool {
    inline for (@typeInfo(config.Config).@"struct".fields) |f| {
        if (std.mem.eql(u8, f.name, key)) {
            @field(cfg, f.name) = try parseInto(f.type, val);
            return true;
        }
    }
    return false;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var out_buf: [16 * 1024]u8 = undefined;
    var fw = std.Io.File.stdout().writer(io, &out_buf);
    const out = &fw.interface;

    var cfg: config.Config = .{};
    var csv_path: ?[]const u8 = null;
    var label: ?[]const u8 = null;
    var valid_frac: f32 = 0.2;
    var split_seed: u64 = 1;
    var max_bytes: usize = 1 << 31;
    var split_col: ?[]const u8 = null;

    var drops: std.ArrayList([]const u8) = .empty;
    defer drops.deinit(gpa);

    // Which config fields the user named. Each algo fills in the rest with
    // defaults that suit it, and must not overwrite an explicit choice.
    var explicit: std.ArrayList([]const u8) = .empty;
    defer explicit.deinit(gpa);

    var it = std.process.Args.Iterator.init(init.minimal.args);
    _ = it.skip();
    while (it.next()) |arg| {
        if (!std.mem.startsWith(u8, arg, "--")) {
            csv_path = arg;
            continue;
        }
        const body = arg[2..];
        const eq = std.mem.indexOfScalar(u8, body, '=') orelse {
            try out.writeAll(usage);
            try out.flush();
            return error.FlagNeedsValue;
        };
        const key = body[0..eq];
        const val = body[eq + 1 ..];

        if (std.mem.eql(u8, key, "label")) {
            label = val;
        } else if (std.mem.eql(u8, key, "drop")) {
            try drops.append(gpa, val);
        } else if (std.mem.eql(u8, key, "valid-frac")) {
            valid_frac = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, key, "split-seed")) {
            split_seed = try std.fmt.parseInt(u64, val, 10);
        } else if (std.mem.eql(u8, key, "split-col")) {
            split_col = val;
        } else if (std.mem.eql(u8, key, "max-bytes")) {
            max_bytes = try std.fmt.parseInt(usize, val, 10);
        } else if (try applyConfigFlag(&cfg, key, val)) {
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

    cfg.applyAlgoDefaults(explicit.items);

    const pool = try pool_mod.Pool.init(gpa, cfg.n_threads);
    defer pool.deinit();

    const t0 = std.Io.Timestamp.now(io, .awake).toNanoseconds();

    var frame = try data.readCsv(gpa, io, pool, path, max_bytes);
    defer frame.deinit();
    const t_read = std.Io.Timestamp.now(io, .awake).toNanoseconds();

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

    var full = try data.quantise(gpa, pool, &frame, cfg, label_col, drops.items);
    defer full.deinit();
    const t_bin = std.Io.Timestamp.now(io, .awake).toNanoseconds();
    cfg.applyForestFeatureDefault(full.n_features, explicit.items);

    try out.print(
        \\data    {s}
        \\rows    {d}
        \\feats   {d}  (label "{s}")
        \\threads {d}
        \\read    {d} ms
        \\bin     {d} ms
        \\
    , .{
        path,
        full.n_rows,
        full.n_features,
        target,
        pool.workerCount(),
        @divTrunc(t_read - t0, 1_000_000),
        @divTrunc(t_bin - t_read, 1_000_000),
    });
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

    switch (cfg.algo) {
        .gbdt => {
            var res = try booster.train(gpa, pool, &train_ds, valid_ptr, cfg, out);
            defer res.model.deinit();
            try printTiming(out, io, t_train0, "tree", res.n_rounds);
            if (valid_ds) |*v| {
                const scores = try gpa.alloc(f32, v.n_rows);
                defer gpa.free(scores);
                // Raw log-odds: AUC is rank-based so the link does not matter,
                // and logloss wants the raw scale anyway.
                res.model.predictRaw(pool, v, scores);
                try report(gpa, out, cfg.objective, scores, v.labels, .raw);
            }
        },
        .random_forest => {
            var res = try forest.train(gpa, pool, &train_ds, valid_ptr, cfg, out);
            defer res.model.deinit();
            try printTiming(out, io, t_train0, "tree", res.n_trees);
            if (valid_ds) |*v| {
                const scores = try gpa.alloc(f32, v.n_rows);
                defer gpa.free(scores);
                res.model.predict(pool, v, scores);
                try report(gpa, out, cfg.objective, scores, v.labels, .natural);
            }
        },
        .linear => {
            var res = try linear.train(gpa, pool, &train_ds, valid_ptr, cfg, out);
            defer res.model.deinit();
            try printTiming(out, io, t_train0, "epoch", res.epochs);
            try out.print("coefs   {d} ({d} zero)\n", .{ res.model.w.len, res.model.nZero() });
            if (valid_ds) |*v| {
                const scores = try gpa.alloc(f32, v.n_rows);
                defer gpa.free(scores);
                res.model.predict(pool, v, scores);
                try report(gpa, out, cfg.objective, scores, v.labels, .natural);
            }
        },
    }
    try out.flush();
}

fn printTiming(
    out: *std.Io.Writer,
    io: std.Io,
    t0: i128,
    comptime unit: []const u8,
    n: u32,
) !void {
    const ms = @divTrunc(std.Io.Timestamp.now(io, .awake).toNanoseconds() - t0, 1_000_000);
    try out.print(
        \\
        \\rounds  {d}
        \\train   {d} ms  ({d:.2} ms/
    ++ unit ++ ")\n", .{
        n,
        ms,
        @as(f64, @floatFromInt(ms)) / @as(f64, @floatFromInt(@max(n, 1))),
    });
}

/// Which scale the predictions are on. The booster reports raw log-odds; the
/// forest and the linear model already apply their own link.
const Scale = enum { raw, natural };

fn report(
    gpa: std.mem.Allocator,
    out: *std.Io.Writer,
    obj: config.Objective,
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
