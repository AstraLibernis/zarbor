// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! `zarbor predict` and `zarbor blend`: score a CSV with one or more saved models.

const std = @import("std");
const zarbor = @import("zarbor");
const args = @import("args.zig");
const data = zarbor.data;
const pool_mod = zarbor.pool;
const metric = zarbor.metric;
const model_mod = zarbor.model;
const csv = zarbor.csv;
const common = @import("common.zig");
const explainLabel = common.explainLabel;

const ScoreMode = enum { predict, blend };

/// `predict` and `blend` share everything but how many models they load, so
/// they share an implementation.
pub fn run(init: std.process.Init, gpa: std.mem.Allocator, out: *std.Io.Writer, mode: ScoreMode) !void {
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
    // The schema pins column kinds through the parse as well as the binning.
    // Sniffing them again from this file lets a slice that happens to hold no
    // usable values for a column -- every entry missing, say -- disagree with
    // the model about what the column is.
    const sch = &bundles.items[0].schema;
    const t0 = std.Io.Timestamp.now(io, .awake).toNanoseconds();
    var frame = try csv.readCsvHinted(gpa, io, pool, path, max_bytes, .{
        .names = sch.names,
        .kinds = sch.kinds,
    });
    defer frame.deinit();
    const t_read = std.Io.Timestamp.now(io, .awake).toNanoseconds();

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
    const t_bin = std.Io.Timestamp.now(io, .awake).toNanoseconds();

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
    const t_pred = std.Io.Timestamp.now(io, .awake).toNanoseconds();

    // Broken out because scoring an implementation against another one means
    // knowing which phase a difference is in. Reading the CSV is not the
    // model's work and should not be charged to it; see docs/arena.md.
    try out.print("rows    {d}\nmodels  {d}\nread    {d} ms\nbin     {d} ms\npredict {d} ms\n", .{
        ds.n_rows,
        bundles.items.len,
        @divTrunc(t_read - t0, 1_000_000),
        @divTrunc(t_bin - t_read, 1_000_000),
        @divTrunc(t_pred - t_bin, 1_000_000),
    });

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
