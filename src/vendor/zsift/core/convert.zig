// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! On-demand conversion of a borrowed field's bytes into a typed value. The bottom
//! of the zsift layering: imports only `std`, operates on `[]const u8` (never on
//! `Field`), so there is no import cycle with `types.zig`. Every converter here is
//! opt-in — a caller who never types a field pays nothing, since Zig only compiles
//! reachable functions, and `as` resolves its dispatch entirely at comptime.

const std = @import("std");

pub const ConvertError = error{
    /// Bytes were not a valid base-10 integer for the requested type.
    InvalidInt,
    /// Bytes were not a valid float for the requested type.
    InvalidFloat,
    /// Bytes did not match any accepted boolean token (see `asBool`).
    InvalidBool,
    /// Bytes did not name a tag of the requested enum (exact, case-sensitive).
    InvalidEnum,
};

pub fn asInt(bytes: []const u8, comptime T: type) ConvertError!T {
    return std.fmt.parseInt(T, bytes, 10) catch ConvertError.InvalidInt;
}

pub fn asFloat(bytes: []const u8, comptime T: type) ConvertError!T {
    return std.fmt.parseFloat(T, bytes) catch ConvertError.InvalidFloat;
}

/// Accepts a fixed, documented token set; anything else fails loud rather than
/// defaulting silently. Truthy: `true True TRUE 1 t T`; falsy: `false False FALSE 0 f F`.
pub fn asBool(bytes: []const u8) ConvertError!bool {
    const trues = [_][]const u8{ "true", "True", "TRUE", "1", "t", "T" };
    const falses = [_][]const u8{ "false", "False", "FALSE", "0", "f", "F" };
    for (trues) |s| if (std.mem.eql(u8, bytes, s)) return true;
    for (falses) |s| if (std.mem.eql(u8, bytes, s)) return false;
    return ConvertError.InvalidBool;
}

/// Exact, case-sensitive match against the enum's tag names; the CSV value must equal
/// a Zig tag identifier (so values like `"N/A"` cannot be enum tags — use `[]const u8`).
pub fn asEnum(bytes: []const u8, comptime T: type) ConvertError!T {
    return std.meta.stringToEnum(T, bytes) orelse ConvertError.InvalidEnum;
}

/// Comptime-dispatched conversion. The whole switch is resolved at comptime, so only
/// the one arm for `T` is compiled into a caller's binary. `T` may be an integer,
/// float, bool, enum, `[]const u8` (identity/borrow — never errors), or an optional of
/// any of those (an empty field becomes `null`). Any other type is a compile error that
/// names the offending type and the accepted set, so misuse fails loud at compile time.
///
/// The return type is uniformly `ConvertError!T` even for the never-erroring arms
/// (`[]const u8`, the non-null optional path), so a typed reader's `inline for` can call
/// `try as(...)` uniformly; the never-erroring path optimizes to a plain move.
pub fn as(bytes: []const u8, comptime T: type) ConvertError!T {
    return switch (@typeInfo(T)) {
        .int, .comptime_int => asInt(bytes, T),
        .float, .comptime_float => asFloat(bytes, T),
        .bool => asBool(bytes),
        .@"enum" => asEnum(bytes, T),
        .optional => |o| if (bytes.len == 0) null else @as(T, try as(bytes, o.child)),
        .pointer => |p| if (p.size == .slice and p.child == u8 and p.is_const)
            bytes // []const u8 — identity/borrow
        else
            @compileError("zsift: cannot convert a field to " ++ @typeName(T) ++
                " (the only supported pointer type is []const u8)"),
        else => @compileError("zsift: unsupported field type " ++ @typeName(T) ++
            " — want an int, float, bool, enum, []const u8, or an optional of those"),
    };
}
