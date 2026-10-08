// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Messages shared by the train and score commands.

const std = @import("std");
const zarbor = @import("zarbor");

/// Turn a label-encoding error into a line that says what to do about it,
/// then exit. These are the errors a user hits with a well-formed CSV and a
/// wrong flag, so the bare error name is not enough -- and a stack trace
/// through the binner is worse than nothing. Unrecognised errors are left to
/// propagate, trace and all.
pub fn explainLabel(out: *std.Io.Writer, err: anyerror, target: []const u8) !void {
    const hint: []const u8 = switch (err) {
        error.MulticlassNotSupported =>
        \\has more than two distinct values. zarbor fits binary and
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

/// Refuse a `--drop` name that matches no column. Ignoring it silently trains on a column the
/// user meant to remove, and the usual cause is a comma list where the flag takes one name.
pub fn checkDrops(out: *std.Io.Writer, frame: *const zarbor.data.Frame, drops: []const []const u8) !void {
    for (drops) |d| if (frame.columnIndex(d) == null) {
        try out.print("error: --drop={s}: no column has that name. --drop takes one column; repeat it for more (--drop=a --drop=b).\n", .{d});
        try out.flush();
        return error.DropColumnNotFound;
    };
}
