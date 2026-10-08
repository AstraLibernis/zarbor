// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Target encoding: which class of a string label becomes 1.

const std = @import("std");
const csv = @import("csv.zig");
const Objective = @import("objective.zig").Objective;
const Frame = csv.Frame;

fn lessThanStr(_: void, a: []u8, b: []u8) bool {
    return std.mem.lessThan(u8, a, b);
}

/// Largest integer class a numeric target may use: exact in f32.
const max_int_class: f32 = 1 << 24;

fn indexOfStr(haystack: []const []const u8, needle: []const u8) ?usize {
    for (haystack, 0..) |h, i| if (std.mem.eql(u8, h, needle)) return i;
    return null;
}

/// Raw target -> fitted numbers. Dictionary ids follow first appearance, so
/// as labels a positive first row silently inverts the target (plausible AUC,
/// no error). This fixes the class order, travels with the model, and maps by
/// *string*, independent of row order.
pub const LabelEncoder = struct {
    gpa: std.mem.Allocator,
    /// `classes[i]` encodes to i. Empty for a numeric target (passed through).
    classes: [][]u8 = &.{},

    pub fn deinit(e: *LabelEncoder) void {
        for (e.classes) |c| e.gpa.free(c);
        if (e.classes.len != 0) e.gpa.free(e.classes);
        e.* = undefined;
    }

    /// Classes sorted lexicographically (row-order independent, matches
    /// scikit-learn's `LabelEncoder`); negative-first for "No"/"Yes",
    /// "false"/"true", "neg"/"pos". `pos_label` overrides ("abnormal" < "normal").
    pub fn fromColumn(
        gpa: std.mem.Allocator,
        src: *const Frame,
        col: usize,
        pos_label: ?[]const u8,
    ) !LabelEncoder {
        if (src.kinds[col] == .numeric) {
            // Numeric: reject --pos-label so a label-column typo is not a no-op.
            if (pos_label != null) return error.PosLabelOnNumericTarget;
            return .{ .gpa = gpa };
        }

        const levels = src.levels[col];
        if (levels.len == 0) return error.EmptyTarget;
        // Binary-only: 0/1/2 would train and score silently, the failure this type removes.
        if (levels.len == 1) return error.SingleClassTarget;
        if (levels.len > 2) return error.MulticlassNotSupported;

        const classes = try gpa.alloc([]u8, levels.len);
        var made: usize = 0;
        errdefer {
            for (classes[0..made]) |c| gpa.free(c);
            gpa.free(classes);
        }
        while (made < levels.len) : (made += 1) classes[made] = try gpa.dupe(u8, levels[made]);
        std.mem.sort([]u8, classes, {}, lessThanStr);

        if (pos_label) |p| {
            const at = indexOfStr(classes, p) orelse return error.PosLabelNotFound;
            const last = classes.len - 1;
            if (at != last) std.mem.swap([]u8, &classes[at], &classes[last]);
        }
        return .{ .gpa = gpa, .classes = classes };
    }

    /// The encoder `obj` needs: under softmax every distinct value is a class, a numeric target's
    /// integers included ("0" < "1" < ... numerically, written as integers); otherwise as
    /// `fromColumn`.
    pub fn forObjective(
        gpa: std.mem.Allocator,
        src: *const Frame,
        col: usize,
        pos_label: ?[]const u8,
        obj: Objective,
    ) !LabelEncoder {
        if (obj != .softmax) return fromColumn(gpa, src, col, pos_label);
        // Which class is "positive" means nothing with more than two.
        if (pos_label != null) return error.PosLabelWithSoftmax;
        if (src.kinds[col] == .categorical) {
            const levels = src.levels[col];
            if (levels.len == 0) return error.EmptyTarget;
            if (levels.len == 1) return error.SingleClassTarget;
            const classes = try gpa.alloc([]u8, levels.len);
            var made: usize = 0;
            errdefer {
                for (classes[0..made]) |c| gpa.free(c);
                gpa.free(classes);
            }
            while (made < levels.len) : (made += 1) classes[made] = try gpa.dupe(u8, levels[made]);
            std.mem.sort([]u8, classes, {}, lessThanStr);
            return .{ .gpa = gpa, .classes = classes };
        }
        // Numeric: the distinct values, which must be integers, in numeric order.
        var seen: std.array_hash_map.Auto(i64, void) = .empty;
        defer seen.deinit(gpa);
        for (src.values[col]) |v| {
            if (std.math.isNan(v)) return error.MissingLabelValue;
            if (v != @trunc(v) or @abs(v) > max_int_class) return error.NonIntegerClass;
            try seen.put(gpa, @intFromFloat(v), {});
        }
        if (seen.count() == 0) return error.EmptyTarget;
        if (seen.count() == 1) return error.SingleClassTarget;
        const vals = try gpa.dupe(i64, seen.keys());
        defer gpa.free(vals);
        std.mem.sort(i64, vals, {}, std.sort.asc(i64));
        const classes = try gpa.alloc([]u8, vals.len);
        var made: usize = 0;
        errdefer {
            for (classes[0..made]) |c| gpa.free(c);
            gpa.free(classes);
        }
        while (made < vals.len) : (made += 1) classes[made] = try std.fmt.allocPrint(gpa, "{d}", .{vals[made]});
        return .{ .gpa = gpa, .classes = classes };
    }

    /// A non-owning view over borrowed class strings. Never `deinit` one.
    pub fn view(gpa: std.mem.Allocator, classes: [][]u8) LabelEncoder {
        return .{ .gpa = gpa, .classes = classes };
    }

    pub fn classIndex(e: *const LabelEncoder, name: []const u8) ?usize {
        return indexOfStr(e.classes, name);
    }

    /// Map `src`'s target onto labels. Missing or unseen classes fail: they
    /// would land on 0, indistinguishable from a real negative.
    pub fn encode(
        e: *const LabelEncoder,
        gpa: std.mem.Allocator,
        src: *const Frame,
        col: usize,
    ) ![]f32 {
        const raw = src.values[col];
        const out = try gpa.alloc(f32, raw.len);
        errdefer gpa.free(out);

        if (e.classes.len != 0 and src.kinds[col] == .numeric) {
            // Integer classes (softmax on a numeric target), matched by their written form. Any
            // other class names never meet a number: a string target's encoder on a numeric
            // column is a kind mismatch.
            for (e.classes) |c| _ = std.fmt.parseInt(i64, c, 10) catch return error.LabelKindMismatch;
            for (raw, out) |v, *o| {
                if (std.math.isNan(v)) return error.MissingLabelValue;
                if (v != @trunc(v) or @abs(v) > max_int_class) return error.NonIntegerClass;
                var buf: [32]u8 = undefined;
                const name = std.fmt.bufPrint(&buf, "{d}", .{@as(i64, @intFromFloat(v))}) catch unreachable; // zsnag:ok 32 bytes hold any i64
                o.* = @floatFromInt(indexOfStr(e.classes, name) orelse return error.UnseenLabelClass);
            }
            return out;
        }
        if (e.classes.len == 0) {
            if (src.kinds[col] != .numeric) return error.LabelKindMismatch;
            for (raw, out) |v, *o| {
                if (std.math.isNan(v)) return error.MissingLabelValue;
                o.* = v;
            }
            return out;
        }

        if (src.kinds[col] != .categorical) return error.LabelKindMismatch;
        const levels = src.levels[col];
        // Resolve id -> string -> class once per level rather than per row.
        const map = try gpa.alloc(f32, levels.len);
        defer gpa.free(map);
        for (levels, map) |lvl, *m| {
            const at = indexOfStr(e.classes, lvl) orelse return error.UnseenLabelClass;
            m.* = @floatFromInt(at);
        }
        for (raw, out) |v, *o| {
            if (std.math.isNan(v)) return error.MissingLabelValue;
            const id: usize = @intFromFloat(v);
            if (id >= map.len) return error.BadLabelValue;
            o.* = map[id];
        }
        return out;
    }

    /// Reject labels the objective cannot represent, e.g. {1,2} under logistic loss.
    pub fn validate(e: *const LabelEncoder, labels: []const f32, obj: Objective) !void {
        if (obj == .softmax) {
            if (e.classes.len < 2) return error.SoftmaxNeedsClasses;
            const k: f32 = @floatFromInt(e.classes.len);
            for (labels) |y| if (!(y >= 0 and y < k and y == @trunc(y))) return error.LabelOutOfRange;
            return;
        }
        if (obj == .poisson) {
            // Counts and rates: a negative label has no Poisson likelihood.
            for (labels) |y| if (!(y >= 0)) return error.LabelOutOfRange;
            return;
        }
        if (obj != .logistic) return;
        // Written so NaN fails too.
        for (labels) |y| if (!(y >= 0 and y <= 1)) return error.LabelOutOfRange;
    }
};

/// Which column carries the target, and how to read it.
pub const LabelSpec = struct {
    col: usize,
    enc: *const LabelEncoder,
    /// A column of per-row weights (`--weight-col`), read with the labels.
    weight_col: ?usize = null,
    /// Columns of per-row starting raw scores (`--init-col`): one, or one per class.
    init_cols: []const usize = &.{},
};
