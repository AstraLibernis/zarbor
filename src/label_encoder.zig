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

fn indexOfStr(haystack: []const []const u8, needle: []const u8) ?usize {
    for (haystack, 0..) |h, i| if (std.mem.eql(u8, h, needle)) return i;
    return null;
}

/// How a raw target column becomes the numbers a model actually fits.
///
/// A categorical column's dictionary id is assigned by order of *first
/// appearance*, so using that id directly as the label makes the meaning of
/// "1" depend on which row happened to come first: train on a file whose
/// first row holds the positive class and the target is silently inverted,
/// with no error and a perfectly plausible AUC. The encoder fixes an explicit
/// class order, travels with the model, and is applied to every later file by
/// *string*, so a category always means the same number no matter how the
/// rows are ordered.
pub const LabelEncoder = struct {
    gpa: std.mem.Allocator,
    /// Class strings in label order: `classes[0]` encodes to 0, `classes[1]`
    /// to 1. Empty for a numeric target, which is passed through unchanged.
    classes: [][]u8 = &.{},

    pub fn deinit(e: *LabelEncoder) void {
        for (e.classes) |c| e.gpa.free(c);
        if (e.classes.len != 0) e.gpa.free(e.classes);
        e.* = undefined;
    }

    /// Derive the encoding from a target column.
    ///
    /// Classes are ordered lexicographically rather than by appearance, which
    /// is what makes the result independent of row order, and matches
    /// scikit-learn's `LabelEncoder` so a reference implementation agrees. For
    /// the usual binary spellings that also puts the negative class first
    /// ("No" < "Yes", "false" < "true", "neg" < "pos"); `pos_label` overrides
    /// it for the ones where it does not ("abnormal" < "normal").
    pub fn fromColumn(
        gpa: std.mem.Allocator,
        src: *const Frame,
        col: usize,
        pos_label: ?[]const u8,
    ) !LabelEncoder {
        if (src.kinds[col] == .numeric) {
            // Nothing to name: the column is already numbers. Rejecting
            // --pos-label here rather than ignoring it keeps a typo in the
            // label column from passing as a no-op.
            if (pos_label != null) return error.PosLabelOnNumericTarget;
            return .{ .gpa = gpa };
        }

        const levels = src.levels[col];
        if (levels.len == 0) return error.EmptyTarget;
        // zarbor is binary-only. Three classes encoded as 0/1/2 would train
        // and score without complaint, which is precisely the failure mode
        // this type exists to remove.
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

    /// A non-owning view over borrowed class strings. Never `deinit` one.
    pub fn view(gpa: std.mem.Allocator, classes: [][]u8) LabelEncoder {
        return .{ .gpa = gpa, .classes = classes };
    }

    pub fn classIndex(e: *const LabelEncoder, name: []const u8) ?usize {
        return indexOfStr(e.classes, name);
    }

    /// Map `src`'s target column onto model labels.
    ///
    /// Fails loudly on a missing value or a class the encoder has never seen:
    /// both would otherwise land on class 0 and be indistinguishable from a
    /// legitimate negative.
    pub fn encode(
        e: *const LabelEncoder,
        gpa: std.mem.Allocator,
        src: *const Frame,
        col: usize,
    ) ![]f32 {
        const raw = src.values[col];
        const out = try gpa.alloc(f32, raw.len);
        errdefer gpa.free(out);

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

    /// Reject labels the objective cannot represent, such as a {1,2}-coded
    /// numeric target handed to logistic loss.
    pub fn validate(_: *const LabelEncoder, labels: []const f32, obj: Objective) !void {
        if (obj != .logistic) return;
        // Written so NaN fails too.
        for (labels) |y| if (!(y >= 0 and y <= 1)) return error.LabelOutOfRange;
    }
};

/// Which column carries the target, and how to read it.
pub const LabelSpec = struct {
    col: usize,
    enc: *const LabelEncoder,
};
