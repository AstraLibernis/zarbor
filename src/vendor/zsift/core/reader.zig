// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Typed row reader: deserialize each record straight into a caller struct `T`.
//!
//! This is an opt-in layer built entirely on the unchanged `SimdParser.nextRecord` —
//! it adds no scanner code. `reader(T)` monomorphizes per struct; the per-record fill
//! is an `inline for` over `T`'s fields, so each column converts through exactly the
//! one converter its field type needs (see `convert.as`). An unused `reader(T)`
//! compiles to nothing.
//!
//! Column mapping is positional by default (struct field `i` ← column `i`); call
//! `withHeader` to instead match struct field *names* against the first record and
//! remember the permutation as indices.
//!
//! Lifetime: a returned `T` that contains `[]const u8` fields *borrows* the parser's
//! input (or scratch, for an escaped `""`) exactly like a `Field` — it is valid only
//! until the next `next()` call. Copy the bytes if you need them to outlive that.
//! Value-typed fields (ints/floats/bools/enums) are independent copies and always safe.

const std = @import("std");
const types = @import("types.zig");
const convert = @import("convert.zig");
const SimdParser = @import("simd.zig").SimdParser;

const Options = types.Options;
const CoreError = types.Error;

pub const ReaderError = CoreError || convert.ConvertError || error{
    /// `withHeader` was called but the input had no header record.
    EmptyHeader,
    /// A struct field's name was not found among the header columns.
    MissingHeaderColumn,
    /// A record was too short to supply a (non-optional) struct field's column.
    MissingColumn,
};

/// A typed reader for struct `T`, sized for exactly `T`'s field count (strict
/// positional: a record with more columns than `T` has fields is a `TooManyFields`
/// error). For a headered CSV whose rows are wider than `T`, use `ReaderWide`.
pub fn Reader(comptime T: type) type {
    return ReaderCap(T, structFieldCount(T));
}

/// Like `Reader(T)` but tolerating up to `max_cols` columns per record (must be >=
/// `T`'s field count). Use for headered CSVs whose rows carry columns your struct
/// does not name.
pub fn ReaderWide(comptime T: type, comptime max_cols: usize) type {
    return ReaderCap(T, max_cols);
}

fn structFieldCount(comptime T: type) usize {
    const info = @typeInfo(T);
    if (info != .@"struct")
        @compileError("zsift.reader expects a struct type, got " ++ @typeName(T));
    return info.@"struct".fields.len;
}

fn ReaderCap(comptime T: type, comptime max_cols: usize) type {
    const fields = @typeInfo(T).@"struct".fields;
    const N = fields.len;
    if (max_cols < N)
        @compileError("zsift.reader: max_cols (" ++ std.fmt.comptimePrint("{d}", .{max_cols}) ++
            ") must be >= " ++ @typeName(T) ++ "'s field count (" ++ std.fmt.comptimePrint("{d}", .{N}) ++ ")");

    return struct {
        parser: SimdParser,
        /// struct field `i` reads CSV column `perm[i]` (identity until `withHeader`).
        perm: [N]usize,
        /// Internal row buffer for `nextRecord`; holds borrowed byte-slices.
        row: [max_cols][]const u8,

        const Self = @This();
        pub const field_count = N;
        /// The row type (for `parallel.forEachRow`).
        pub const Row = T;

        pub fn init(input: []const u8, scratch: []u8, opts: Options) CoreError!Self {
            var self: Self = .{
                .parser = try SimdParser.init(input, scratch, opts),
                .perm = undefined,
                .row = undefined,
            };
            inline for (0..N) |i| self.perm[i] = i; // positional default
            return self;
        }

        /// Consume the first record as a header row and record, for each struct field,
        /// which column carries its name. Only the header bytes are read here; `perm`
        /// stores indices, so nothing borrows the (reused) row buffer afterward.
        pub fn withHeader(self: *Self) ReaderError!void {
            const hrec = (try self.parser.nextRecord(&self.row)) orelse return error.EmptyHeader;
            inline for (fields, 0..) |field, i| {
                self.perm[i] = indexOfName(hrec, field.name) orelse return error.MissingHeaderColumn;
            }
        }

        /// The next record as a `T`, or null at end of input. Zero-alloc; see the
        /// lifetime note at the top of this file for `[]const u8` fields.
        pub fn next(self: *Self) ReaderError!?T {
            const rec = (try self.parser.nextRecord(&self.row)) orelse return null;
            var out: T = undefined;
            inline for (fields, 0..) |field, i| {
                const c = self.perm[i];
                if (c >= rec.len) {
                    if (@typeInfo(field.type) == .optional) {
                        @field(out, field.name) = null;
                    } else {
                        return error.MissingColumn;
                    }
                } else {
                    @field(out, field.name) = try convert.as(rec[c], field.type);
                }
            }
            return out;
        }
    };
}

fn indexOfName(record: []const []const u8, name: []const u8) ?usize {
    for (record, 0..) |c, i| if (std.mem.eql(u8, c, name)) return i;
    return null;
}
