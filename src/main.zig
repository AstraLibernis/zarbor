//! zgbdt — train a gradient-boosted tree ensemble on a CSV.
//!
//! Every field of `Config` is exposed as `--field=value` by comptime
//! reflection, so the flag surface and the struct can never drift apart.

const std = @import("std");
const config = @import("config.zig");
const data = @import("data.zig");
const pool_mod = @import("pool.zig");
const booster = @import("booster.zig");
const metric = @import("metric.zig");

const usage =
    \\usage: zgbdt <train.csv> --label=<column> [options]
    \\
    \\  --label=NAME        target column (required)
    \\  --drop=NAME         exclude a column; repeatable
    \\  --valid-frac=F      fraction held out for validation (default 0.2)
    \\  --split-seed=N      seed for the validation split (default 1)
    \\  --max-bytes=N       CSV size cap in bytes (default 1<<31)
    \\
    \\Any Config field is also a flag, e.g.:
    \\  --n_rounds=800 --learning_rate=0.05 --max_depth=7 --lambda=2.0
    \\  --grow_policy=lossguide --max_leaves=64 --subsample=0.8
    \\  --colsample_bytree=0.8 --early_stopping_rounds=50 --n_threads=16
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

    var drops: std.ArrayList([]const u8) = .empty;
    defer drops.deinit(gpa);

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
        } else if (std.mem.eql(u8, key, "max-bytes")) {
            max_bytes = try std.fmt.parseInt(usize, val, 10);
        } else if (!try applyConfigFlag(&cfg, key, val)) {
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

    const label_col = frame.columnIndex(target) orelse return error.LabelColumnNotFound;

    var full = try data.quantise(gpa, pool, &frame, cfg, label_col, drops.items);
    defer full.deinit();
    const t_bin = std.Io.Timestamp.now(io, .awake).toNanoseconds();

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
    for (perm, 0..) |*p, i| p.* = @intCast(i);
    {
        var prng: std.Random.DefaultPrng = .init(split_seed);
        const r = prng.random();
        var i: usize = full.n_rows;
        while (i > 1) {
            i -= 1;
            const j = r.uintLessThan(usize, i + 1);
            std.mem.swap(u32, &perm[i], &perm[j]);
        }
    }
    const n_valid: usize = @intFromFloat(@round(@as(f32, @floatFromInt(full.n_rows)) * valid_frac));
    const n_train = full.n_rows - n_valid;

    var train_ds = try data.subset(gpa, &full, perm[0..n_train]);
    defer train_ds.deinit();

    var valid_ds: ?data.Dataset = null;
    if (n_valid != 0) valid_ds = try data.subset(gpa, &full, perm[n_train..]);
    defer if (valid_ds) |*v| v.deinit();

    try out.print("train   {d} rows / valid {d} rows\n\n", .{ n_train, n_valid });
    try out.flush();

    const t_train0 = std.Io.Timestamp.now(io, .awake).toNanoseconds();
    var res = try booster.train(
        gpa,
        pool,
        &train_ds,
        if (valid_ds) |*v| v else null,
        cfg,
        out,
    );
    defer res.model.deinit();
    const t_train1 = std.Io.Timestamp.now(io, .awake).toNanoseconds();

    const ms = @divTrunc(t_train1 - t_train0, 1_000_000);
    try out.print(
        \\
        \\rounds  {d}
        \\train   {d} ms  ({d:.2} ms/tree)
        \\
    , .{
        res.n_rounds,
        ms,
        @as(f64, @floatFromInt(ms)) / @as(f64, @floatFromInt(@max(res.n_rounds, 1))),
    });

    if (valid_ds) |*v| {
        const scores = try gpa.alloc(f32, v.n_rows);
        defer gpa.free(scores);
        res.model.predictRaw(pool, v, scores);
        const a = try metric.auc(gpa, scores, v.labels);
        const ll = metric.logloss(scores, v.labels);
        try out.print("valid   auc={d:.6}  logloss={d:.6}\n", .{ a, ll });
    }
    try out.flush();
}
