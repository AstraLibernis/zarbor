//! zgbdt — train a gradient-boosted tree ensemble on a CSV.
//!
//! Every field of `Config` is exposed as `--field=value` by comptime
//! reflection, so the flag surface and the struct can never drift apart.

const std = @import("std");
const builtin = @import("builtin");
const config = @import("config.zig");
const data = @import("data.zig");
const pool_mod = @import("pool.zig");
const booster = @import("booster.zig");
const forest = @import("forest.zig");
const linear = @import("linear.zig");
const metric = @import("metric.zig");
const prof = @import("prof.zig");
const model_mod = @import("model.zig");
const cv_mod = @import("cv.zig");

const usage =
    \\usage: zgbdt <train.csv> --label=<column> [options]
    \\
    \\  --algo=NAME         gbdt | random_forest | linear   (default gbdt)
    \\  --label=NAME        target column (required)
    \\  --pos-label=NAME    class of a string target to encode as 1
    \\                      (default: classes sorted, so "No"<"Yes" -> Yes=1)
    \\  --drop=NAME         exclude a column; repeatable
    \\  --valid-frac=F      fraction held out for validation (default 0.2)
    \\  --split-seed=N      seed for the validation split (default 1)
    \\  --max-bytes=N       CSV size cap in bytes (default 1<<31)
    \\  --split-col=NAME    column assigning rows to train(0)/valid(nonzero);
    \\                      overrides --valid-frac, and is dropped as a feature
    \\  --save=FILE         write the trained model to FILE
    \\
    \\other commands:
    \\  zgbdt predict <data.csv> --model=M.zm [--out=P.csv] [--id-col=id]
    \\  zgbdt blend   <data.csv> --models=A.zm,B.zm [--weights=1,2] [--out=P.csv]
    \\  zgbdt info    --model=M.zm
    \\  zgbdt cv      <train.csv> --label=<column> [--folds=5]
    \\
    \\predict and blend bin the new data with the schema stored in the model,
    \\so categorical levels map to the same bins they did in training. Pass
    \\--label=NAME as well to score against a labelled holdout; it is decoded
    \\with the model's own class order, not the holdout file's.
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
    \\  --algo=linear          L-BFGS + L1/L2 on the binned design matrix
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
    var save_path: ?[]const u8 = null;
    var pos_label: ?[]const u8 = null;

    var drops: std.ArrayList([]const u8) = .empty;
    defer drops.deinit(gpa);

    // Which config fields the user named. Each algo fills in the rest with
    // defaults that suit it, and must not overwrite an explicit choice.
    var explicit: std.ArrayList([]const u8) = .empty;
    defer explicit.deinit(gpa);

    {
        // Subcommand dispatch. A bare CSV path still means "train", so every
        // existing invocation keeps working.
        var probe = std.process.Args.Iterator.init(init.minimal.args);
        _ = probe.skip();
        if (probe.next()) |first| {
            if (std.mem.eql(u8, first, "predict")) return score(init, gpa, out, .predict);
            if (std.mem.eql(u8, first, "blend")) return score(init, gpa, out, .blend);
            if (std.mem.eql(u8, first, "info")) return info(init, gpa, out);
            if (std.mem.eql(u8, first, "cv")) return cv_mod.run(init, gpa, out);
        }
    }

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

    // Fix the class order before anything reads the target. Left to the
    // dictionary, "1" would mean whichever class the first row happened to
    // hold, so the same data in a different row order would train the
    // opposite model.
    var enc = data.LabelEncoder.fromColumn(gpa, &frame, label_col, pos_label) catch |err| {
        try explainLabel(out, err, target);
        return err;
    };
    defer enc.deinit();

    var full = data.quantise(gpa, pool, &frame, cfg, .{ .col = label_col, .enc = &enc }, drops.items) catch |err| {
        try explainLabel(out, err, target);
        return err;
    };
    defer full.deinit();
    enc.validate(full.labels, cfg.objective) catch |err| {
        try explainLabel(out, err, target);
        return err;
    };
    const t_bin = std.Io.Timestamp.now(io, .awake).toNanoseconds();
    cfg.applyForestFeatureDefault(full.n_features, explicit.items);

    try out.print(
        \\data    {s}
        \\rows    {d}
        \\feats   {d}  (label "{s}")
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

    switch (cfg.algo) {
        .gbdt => {
            var res = try booster.train(gpa, pool, &train_ds, valid_ptr, cfg, out);
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
                try report(gpa, out, cfg.objective, scores, v.labels, .raw);
            }
        },
        .random_forest => {
            var res = try forest.train(gpa, pool, &train_ds, valid_ptr, cfg, out);
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
                try report(gpa, out, cfg.objective, scores, v.labels, .natural);
            }
        },
        .linear => {
            var res = try linear.train(gpa, pool, &train_ds, valid_ptr, cfg, out);
            defer res.model.deinit();
            try printTiming(out, io, t_train0, "epoch", res.epochs, res.valid_ns);
            try out.print("coefs   {d} ({d} zero)\n", .{ res.model.w.len, res.model.nZero() });
            if (save_path) |sp| {
                // The bundle borrows the fitted model's design and weights
                // rather than copying, so it must not free them.
                var b = model_mod.Bundle{
                    .gpa = gpa,
                    .kind = .linear,
                    .schema = try data.Schema.fromDataset(gpa, &full),
                    .objective = cfg.objective,
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
                try report(gpa, out, cfg.objective, scores, v.labels, .natural);
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

// ------------------------------------------------------------ scoring modes

const ScoreMode = enum { predict, blend };

/// `predict` and `blend` share everything but how many models they load, so
/// they share an implementation.
fn score(init: std.process.Init, gpa: std.mem.Allocator, out: *std.Io.Writer, mode: ScoreMode) !void {
    const io = init.io;

    var csv_path: ?[]const u8 = null;
    var model_arg: ?[]const u8 = null;
    var weights_arg: ?[]const u8 = null;
    var out_path: ?[]const u8 = null;
    var id_col: ?[]const u8 = null;
    var label: ?[]const u8 = null;
    var pred_col: []const u8 = "prediction";
    var n_threads: u32 = 0;
    var max_bytes: usize = 1 << 31;

    var it = std.process.Args.Iterator.init(init.minimal.args);
    _ = it.skip();
    _ = it.skip(); // the subcommand itself
    while (it.next()) |arg| {
        if (!std.mem.startsWith(u8, arg, "--")) {
            csv_path = arg;
            continue;
        }
        const body = arg[2..];
        const eq = std.mem.indexOfScalar(u8, body, '=') orelse return error.FlagNeedsValue;
        const key = body[0..eq];
        const val = body[eq + 1 ..];
        if (std.mem.eql(u8, key, "model") or std.mem.eql(u8, key, "models")) {
            model_arg = val;
        } else if (std.mem.eql(u8, key, "weights")) {
            weights_arg = val;
        } else if (std.mem.eql(u8, key, "out")) {
            out_path = val;
        } else if (std.mem.eql(u8, key, "id-col")) {
            id_col = val;
        } else if (std.mem.eql(u8, key, "pred-col")) {
            pred_col = val;
        } else if (std.mem.eql(u8, key, "label")) {
            label = val;
        } else if (std.mem.eql(u8, key, "n_threads")) {
            n_threads = try std.fmt.parseInt(u32, val, 10);
        } else if (std.mem.eql(u8, key, "max-bytes")) {
            max_bytes = try std.fmt.parseInt(usize, val, 10);
        } else {
            try out.print("unknown flag: --{s}\n", .{key});
            try out.flush();
            return error.UnknownFlag;
        }
    }

    const path = csv_path orelse return error.NoInput;
    const models = model_arg orelse return error.NoModel;

    const pool = try pool_mod.Pool.init(gpa, n_threads);
    defer pool.deinit();

    // --- load the models ---
    var bundles: std.ArrayList(model_mod.Bundle) = .empty;
    defer {
        for (bundles.items) |*b| b.deinit();
        bundles.deinit(gpa);
    }
    var names = std.mem.splitScalar(u8, models, ',');
    while (names.next()) |name| {
        const trimmed = std.mem.trim(u8, name, " ");
        if (trimmed.len == 0) continue;
        try bundles.append(gpa, try model_mod.load(gpa, io, trimmed));
    }
    if (bundles.items.len == 0) return error.NoModel;
    if (mode == .predict and bundles.items.len != 1) return error.PredictTakesOneModel;

    // --- weights ---
    const weights = try gpa.alloc(f32, bundles.items.len);
    defer gpa.free(weights);
    @memset(weights, 1.0);
    if (weights_arg) |w| {
        var parts = std.mem.splitScalar(u8, w, ',');
        var i: usize = 0;
        while (parts.next()) |ptxt| : (i += 1) {
            if (i >= weights.len) return error.TooManyWeights;
            weights[i] = try std.fmt.parseFloat(f32, std.mem.trim(u8, ptxt, " "));
        }
        if (i != weights.len) return error.WeightCountMismatch;
    }

    // --- bin the new data under the first model's schema ---
    var frame = try data.readCsv(gpa, io, pool, path, max_bytes);
    defer frame.deinit();

    // Decode the holdout's target with the *model's* class order. This file
    // built its own dictionary by first appearance, so reading its raw ids
    // would silently invert the target whenever the two files disagree on
    // which class appears first -- and an inverted AUC of 0.06 still looks
    // like a number, not a bug.
    var enc = bundles.items[0].encoder();
    var label_spec: ?data.LabelSpec = null;
    if (label) |name| {
        const lc = frame.columnIndex(name) orelse return error.LabelColumnNotFound;
        if (frame.kinds[lc] == .categorical and enc.classes.len == 0) {
            try out.writeAll(
                \\this model stores no label encoding, so a string target cannot
                \\be decoded safely. Retrain to write one, or score without
                \\--label.
                \\
            );
            try out.flush();
            std.process.exit(1);
        }
        label_spec = .{ .col = lc, .enc = &enc };
    }

    var ds = data.applySchema(gpa, pool, &frame, &bundles.items[0].schema, label_spec) catch |err| {
        if (label) |name| try explainLabel(out, err, name);
        return err;
    };
    defer ds.deinit();

    // Blending models trained on different schemas would silently score the
    // same column against different bin edges.
    for (bundles.items[1..]) |*b| {
        if (b.schema.n_features != bundles.items[0].schema.n_features) return error.SchemaMismatch;
        for (0..b.schema.n_features) |f| {
            if (!std.mem.eql(u8, b.schema.names[f], bundles.items[0].schema.names[f])) return error.SchemaMismatch;
            if (b.schema.n_bins[f] != bundles.items[0].schema.n_bins[f]) return error.SchemaMismatch;
        }
    }

    const preds = try gpa.alloc(f32, ds.n_rows);
    defer gpa.free(preds);

    if (mode == .predict) {
        bundles.items[0].predict(pool, &ds, preds);
    } else {
        const refs = try gpa.alloc(*const model_mod.Bundle, bundles.items.len);
        defer gpa.free(refs);
        for (bundles.items, refs) |*b, *r| r.* = b;
        try model_mod.blend(gpa, pool, refs, weights, &ds, preds);
    }

    try out.print("rows    {d}\nmodels  {d}\n", .{ ds.n_rows, bundles.items.len });

    if (ds.labels.len == ds.n_rows and ds.labels.len != 0) {
        switch (bundles.items[0].objective) {
            .logistic => try out.print("score   auc={d:.6}  logloss={d:.6}\n", .{
                try metric.auc(gpa, preds, ds.labels),
                metric.loglossProb(preds, ds.labels),
            }),
            .squared_error => try out.print("score   rmse={d:.6}\n", .{metric.rmse(preds, ds.labels)}),
        }
    }

    if (out_path) |op| {
        try writePredictions(gpa, io, op, &frame, id_col, pred_col, preds);
        try out.print("wrote   {s}\n", .{op});
    } else {
        // No destination: show the head so the command is still useful alone.
        const show = @min(preds.len, 10);
        try out.print("\nfirst {d} predictions\n", .{show});
        for (preds[0..show]) |p| try out.print("  {d:.6}\n", .{p});
    }
    try out.flush();
}

/// Writes `id,prediction` when an id column is named, else one column.
fn writePredictions(
    gpa: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    frame: *const data.Frame,
    id_col: ?[]const u8,
    pred_col: []const u8,
    preds: []const f32,
) !void {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    // std.ArrayList has no writer in this Zig; format each field into a small
    // stack buffer and append. One row never exceeds this.
    var line: [512]u8 = undefined;

    const idx: ?usize = if (id_col) |name| frame.columnIndex(name) orelse return error.IdColumnNotFound else null;

    if (idx) |i| {
        try buf.appendSlice(gpa, try std.fmt.bufPrint(&line, "{s},{s}\n", .{ frame.names[i], pred_col }));
    } else {
        try buf.appendSlice(gpa, try std.fmt.bufPrint(&line, "{s}\n", .{pred_col}));
    }

    for (preds, 0..) |p, row| {
        if (idx) |i| {
            const v = frame.values[i][row];
            const txt = if (frame.kinds[i] == .categorical) blk: {
                // Ids read as text keep their original spelling.
                const lid: usize = @intFromFloat(v);
                break :blk try std.fmt.bufPrint(&line, "{s},", .{frame.levels[i][lid]});
            } else if (v == @trunc(v) and @abs(v) < 16_777_216)
                // Integral and exactly representable: print without a decimal
                // point, which is what an id column almost always wants.
                try std.fmt.bufPrint(&line, "{d},", .{@as(i64, @intFromFloat(v))})
            else
                try std.fmt.bufPrint(&line, "{d},", .{v});
            try buf.appendSlice(gpa, txt);
        }
        try buf.appendSlice(gpa, try std.fmt.bufPrint(&line, "{d:.6}\n", .{p}));
    }
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = buf.items });
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

/// Turn a label-encoding error into a line that says what to do about it,
/// then exit. These are the errors a user hits with a well-formed CSV and a
/// wrong flag, so the bare error name is not enough -- and a stack trace
/// through the binner is worse than nothing. Unrecognised errors are left to
/// propagate, trace and all.
fn explainLabel(out: *std.Io.Writer, err: anyerror, target: []const u8) !void {
    const hint: []const u8 = switch (err) {
        error.MulticlassNotSupported =>
            \\has more than two distinct values. zmodels fits binary and
            \\regression targets only; a multiclass column would otherwise be
            \\encoded 0,1,2,... and fitted as if those were magnitudes.
        ,
        error.SingleClassTarget =>
            \\has only one distinct value, so there is nothing to learn.
        ,
        error.EmptyTarget => "is empty.",
        error.PosLabelNotFound =>
            \\does not contain the class named by --pos-label. Run without it
            \\to see the classes as parsed.
        ,
        error.PosLabelOnNumericTarget =>
            \\is numeric, so --pos-label has nothing to name. Its values are
            \\used as-is.
        ,
        error.MissingLabelValue =>
            \\has missing values. A missing target cannot be guessed, and
            \\treating it as the negative class would bias the fit.
        ,
        error.LabelOutOfRange =>
            \\has values outside [0,1], which the logistic objective cannot
            \\represent. Use --objective=squared_error, or recode the target.
        ,
        error.UnseenLabelClass =>
            \\contains a class the model was not trained on.
        ,
        error.LabelKindMismatch =>
            \\is numeric here but was a string in training, or the reverse.
        ,
        else => return,
    };
    try out.print("\nlabel column \"{s}\" {s}\n", .{ target, hint });
    try out.flush();
    std.process.exit(1);
}

fn info(init: std.process.Init, gpa: std.mem.Allocator, out: *std.Io.Writer) !void {
    var path: ?[]const u8 = null;
    var it = std.process.Args.Iterator.init(init.minimal.args);
    _ = it.skip();
    _ = it.skip();
    while (it.next()) |arg| {
        if (std.mem.startsWith(u8, arg, "--model=")) path = arg["--model=".len..];
    }
    const p = path orelse return error.NoModel;

    var b = try model_mod.load(gpa, init.io, p);
    defer b.deinit();

    try out.print(
        \\file      {s}
        \\kind      {s}
        \\objective {s}
        \\features  {d}
        \\
    , .{ p, @tagName(b.kind), @tagName(b.objective), b.schema.n_features });

    if (b.label.len != 0) {
        try out.print("label     {s}", .{b.label});
        for (b.classes, 0..) |c, i| try out.print("{s}\"{s}\"={d}", .{ if (i == 0) "  " else ", ", c, i });
        if (b.classes.len == 0) try out.writeAll("  (numeric)");
        try out.writeAll("\n");
    }

    switch (b.kind) {
        .gbdt, .forest => {
            var nodes: usize = 0;
            var leaves: usize = 0;
            for (b.trees) |t| {
                nodes += t.nodes.len;
                for (t.nodes) |n| {
                    if (n.is_leaf) leaves += 1;
                }
            }
            try out.print("trees     {d} ({d} nodes, {d} leaves)\n", .{ b.trees.len, nodes, leaves });
            if (b.kind == .gbdt) try out.print("base      {d:.6}\n", .{b.base_score});
        },
        .linear => {
            var zero: usize = 0;
            for (b.lin.?.w) |c| {
                if (c == 0) zero += 1;
            }
            try out.print("coefs     {d} ({d} zero)\nintercept {d:.6}\n", .{ b.lin.?.w.len, zero, b.lin.?.intercept });
        },
    }

    try out.print("\nschema\n", .{});
    for (0..b.schema.n_features) |f| {
        try out.print("  {s:<32} {s:<12} {d:>4} bins\n", .{
            b.schema.names[f],
            @tagName(b.schema.kinds[f]),
            b.schema.n_bins[f],
        });
    }
    try out.flush();
}
