// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! `zarbor explain`: what a saved model uses. Without data, feature importance from the trees'
//! training statistics. With data, SHAP values per row: their mean size per feature (the usual
//! global ranking), a check that each row's values sum to its raw score, and `--shap=FILE` for
//! the values themselves.

const std = @import("std");
const zarbor = @import("zarbor");
const args = @import("args.zig");
const data = zarbor.data;
const pool_mod = zarbor.pool;
const model_mod = zarbor.model;
const csv = zarbor.csv;
const explain = zarbor.explain;

const usage =
    \\usage: zarbor explain [data.csv] --model=M.zm [options]
    \\
    \\  --cover=hessian|count  what weights the paths a row does not take
    \\                         (default hessian, XGBoost's; count is LightGBM's
    \\                         and CatBoost's)
    \\  --shap=FILE            write each row's SHAP values as CSV
    \\  --id-col=NAME          carry this column into --shap's file
    \\  --top=N                rows of the tables (default 20, 0 = all)
    \\  --n_threads=N          worker threads (default all)
    \\
    \\Importance needs a model saved by this version or newer (format 7).
    \\SHAP values and the bias column sum to the raw score (log-odds,
    \\class score, value) the model predicts for that row.
    \\
;

pub fn run(init: std.process.Init, gpa: std.mem.Allocator, out: *std.Io.Writer) !void {
    const io = init.io;
    var csv_path: ?[]const u8 = null;
    var model_path: ?[]const u8 = null;
    var shap_path: ?[]const u8 = null;
    var id_col: ?[]const u8 = null;
    var cover: explain.Cover = .hessian;
    var top: usize = 20;
    var n_threads: u32 = 0;

    var it = args.Iterator.init(init.minimal.args, 1);
    while (it.next()) |arg| {
        const flag = switch (arg) {
            .positional => |p| {
                csv_path = p;
                continue;
            },
            .bare => |b| {
                if (std.mem.eql(u8, b, "help")) {
                    try out.writeAll(usage);
                    try out.flush();
                    return;
                }
                return error.FlagNeedsValue;
            },
            .flag => |f| f,
        };
        const key, const val = .{ flag.key, flag.val };
        if (std.mem.eql(u8, key, "model")) {
            model_path = val;
        } else if (std.mem.eql(u8, key, "shap")) {
            shap_path = val;
        } else if (std.mem.eql(u8, key, "id-col")) {
            id_col = val;
        } else if (std.mem.eql(u8, key, "cover")) {
            cover = std.meta.stringToEnum(explain.Cover, val) orelse return error.UnknownEnumValue;
        } else if (std.mem.eql(u8, key, "top")) {
            top = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "n_threads")) {
            n_threads = try std.fmt.parseInt(u32, val, 10);
        } else {
            try out.print("unknown flag: --{s}\n{s}", .{ key, usage });
            try out.flush();
            return error.UnknownFlag;
        }
    }
    const mp = model_path orelse {
        try out.writeAll(usage);
        try out.flush();
        return error.NoModel;
    };

    const pool = try pool_mod.Pool.init(gpa, n_threads);
    defer pool.deinit();
    var b = try model_mod.load(gpa, io, mp);
    defer b.deinit();
    const sch = &b.schema;
    const nf = sch.n_features;

    try out.print("model   {s}  ({s}, {s}, {d} features)\n", .{ mp, @tagName(b.kind), @tagName(b.objective), nf });

    // --- importance from the stored node statistics ---
    if (b.kind != .linear) {
        const imp = try gpa.alloc(explain.Importance, nf);
        defer gpa.free(imp);
        explain.importance(b.trees, cover, imp) catch |err| {
            if (err == error.ModelHasNoNodeStats) {
                try out.writeAll("this model was saved before node statistics were kept (format < 7); retrain it to explain it\n");
                try out.flush();
            }
            return err;
        };
        var total: f64 = 0;
        for (imp) |x| total += x.gain;
        const order = try rank(gpa, nf, imp, struct {
            fn key(xs: []const explain.Importance, i: usize) f64 {
                return xs[i].gain;
            }
        }.key);
        defer gpa.free(order);
        try out.print("\nimportance (summed over {d} trees; cover = {s})\n", .{ b.trees.len, @tagName(cover) });
        try out.print("  {s:<32} {s:>12} {s:>7} {s:>12} {s:>7}\n", .{ "feature", "gain", "gain%", "cover", "splits" });
        for (order[0..limit(top, nf)]) |f| {
            const x = imp[f];
            if (x.splits == 0) break;
            try out.print("  {s:<32} {d:>12.4} {d:>6.2}% {d:>12.1} {d:>7}\n", .{
                sch.names[f], x.gain, if (total > 0) 100 * x.gain / total else 0, x.cover, x.splits,
            });
        }
        var unused: usize = 0;
        for (imp) |x| unused += @intFromBool(x.splits == 0);
        if (unused != 0) try out.print("  ({d} features never split on)\n", .{unused});
    }

    const path = csv_path orelse {
        if (b.kind == .linear) try out.writeAll("\nthe linear model's importance is per row: pass a data file for SHAP values\n");
        try out.flush();
        return;
    };

    // --- SHAP on the data, binned with the model's own schema ---
    var frame = try csv.readCsvHinted(gpa, io, pool, path, 1 << 31, .{ .names = sch.names, .kinds = sch.kinds });
    defer frame.deinit();
    var ds = try data.applySchema(gpa, pool, &frame, sch, null);
    defer ds.deinit();
    const k = b.width();
    const per = nf + 1;
    const phi = try gpa.alloc(f64, ds.n_rows * k * per);
    defer gpa.free(phi);
    explain.bundleShap(gpa, pool, &b, &ds, cover, phi) catch |err| {
        switch (err) {
            error.ShapLinearLeavesUnsupported => try out.writeAll("SHAP is not defined here for linear leaves\n"),
            error.ShapCombinationUnsupported => try out.writeAll("SHAP is not available for trees with categorical combinations (--max_ctr_complexity=1 avoids them)\n"),
            error.ShapNeedsStandardizedLinear => try out.writeAll("linear SHAP needs the training means, kept only with --lin_standardize=true (the default)\n"),
            error.ModelHasNoNodeStats => try out.writeAll("this model was saved before node statistics were kept (format < 7); retrain it to explain it\n"),
            else => {},
        }
        try out.flush();
        return err;
    };

    // Each row's values and bias must sum to the raw score the model predicts: the check that
    // makes the numbers trustworthy without another library.
    const raw = try gpa.alloc(f32, ds.n_rows * k);
    defer gpa.free(raw);
    b.predictRaw(pool, &ds, raw);
    var worst: f64 = 0;
    for (0..ds.n_rows * k) |i| {
        var sum: f64 = 0;
        for (phi[i * per ..][0..per]) |v| sum += v;
        worst = @max(worst, @abs(sum - raw[i]) / @max(1, @abs(@as(f64, raw[i]))));
    }

    // Global ranking: mean |SHAP| per feature, over rows (and classes).
    const mean_abs = try gpa.alloc(explain.Importance, nf);
    defer gpa.free(mean_abs);
    @memset(mean_abs, .{});
    for (0..ds.n_rows * k) |i| {
        for (phi[i * per ..][0..nf], mean_abs) |v, *m| m.gain += @abs(v);
    }
    const nk: f64 = @floatFromInt(ds.n_rows * k);
    for (mean_abs) |*m| m.gain /= nk;
    const order = try rank(gpa, nf, mean_abs, struct {
        fn key(xs: []const explain.Importance, i: usize) f64 {
            return xs[i].gain;
        }
    }.key);
    defer gpa.free(order);
    try out.print("\nmean |SHAP| over {d} rows{s}\n", .{ ds.n_rows, if (k > 1) " and every class" else "" });
    for (order[0..limit(top, nf)]) |f| {
        if (mean_abs[f].gain == 0) break;
        try out.print("  {s:<32} {d:>12.6}\n", .{ sch.names[f], mean_abs[f].gain });
    }
    try out.print("check   values + bias = raw score, largest relative difference {e:.2}\n", .{worst});

    if (shap_path) |sp| {
        try writeShap(gpa, io, sp, &frame, sch, b.classes, k, id_col, phi);
        try out.print("wrote   {s}\n", .{sp});
    }
    try out.flush();
}

fn limit(top: usize, n: usize) usize {
    return if (top == 0) n else @min(top, n);
}

/// Feature indices by descending `key`, ties by index.
fn rank(gpa: std.mem.Allocator, n: usize, xs: []const explain.Importance, comptime key: fn ([]const explain.Importance, usize) f64) ![]usize {
    const order = try gpa.alloc(usize, n);
    for (order, 0..) |*o, i| o.* = i;
    std.sort.pdq(usize, order, xs, struct {
        fn gt(c: []const explain.Importance, a: usize, b2: usize) bool {
            const ka = key(c, a);
            const kb = key(c, b2);
            return if (ka != kb) ka > kb else a < b2;
        }
    }.gt);
    return order;
}

/// One row per input row: the id if asked, then per output `shap_<feature>` and `shap_bias`
/// (prefixed `<class>_` under softmax).
fn writeShap(
    gpa: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    frame: *const data.Frame,
    sch: *const data.Schema,
    classes: []const []const u8,
    k: usize,
    id_col: ?[]const u8,
    phi: []const f64,
) !void {
    const idx: ?usize = if (id_col) |name| frame.columnIndex(name) orelse return error.IdColumnNotFound else null;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    var line: [512]u8 = undefined;
    const nf = sch.n_features;
    const per = nf + 1;
    var first = true;
    if (idx) |i| {
        try buf.appendSlice(gpa, frame.names[i]);
        first = false;
    }
    for (0..k) |c| {
        const prefix = if (k > 1) (if (c < classes.len) classes[c] else "") else "";
        for (0..per) |f| {
            const name = if (f < nf) sch.names[f] else "bias";
            try buf.appendSlice(gpa, try std.fmt.bufPrint(&line, "{s}{s}{s}shap_{s}", .{ if (first) "" else ",", prefix, if (k > 1) "_" else "", name }));
            first = false;
        }
    }
    try buf.append(gpa, '\n');
    const n_rows = phi.len / (k * per);
    for (0..n_rows) |r| {
        first = true;
        if (idx) |i| {
            const v = frame.values[i][r];
            const txt = if (std.math.isNan(v))
                ""
            else if (frame.kinds[i] == .categorical)
                frame.levels[i][@intFromFloat(v)]
            else
                try std.fmt.bufPrint(&line, "{d}", .{v});
            try buf.appendSlice(gpa, txt);
            first = false;
        }
        for (phi[r * k * per ..][0 .. k * per]) |v| {
            try buf.appendSlice(gpa, try std.fmt.bufPrint(&line, "{s}{d:.8}", .{ if (first) "" else ",", v }));
            first = false;
        }
        try buf.append(gpa, '\n');
    }
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = buf.items });
}
