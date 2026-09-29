// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! An ergonomic, borrowed view over one record's fields — positional (`at`/`as`) and,
//! when a `Header` is attached, by name (`get`). Backed by the `[]const []const u8`
//! that `SimdParser.nextRecord` already produces, so it adds no scanner code and no
//! allocation; `at(i)` reconstructs a `Field` (with the correct `last_in_record`) on
//! demand.
//!
//! Lifetime: `fields` borrows whatever `nextRecord` filled — valid until the next read
//! reuses that buffer (and, for escaped `""`, until the next `resetScratch`). Same
//! contract as `Field`.

const std = @import("std");
const types = @import("types.zig");
const convert = @import("convert.zig");
const Field = types.Field;
const Header = @import("header.zig").Header;

pub const Record = struct {
    fields: []const []const u8,
    /// Optional attached header enabling `get(name)`. Must outlive the record view.
    header: ?*const Header = null,

    pub fn init(fields: []const []const u8) Record {
        return .{ .fields = fields };
    }

    pub fn withHeader(fields: []const []const u8, header: *const Header) Record {
        return .{ .fields = fields, .header = header };
    }

    pub fn count(self: Record) usize {
        return self.fields.len;
    }

    /// The `i`th field. Asserts `i` is in range (a programming error, not input error).
    pub fn at(self: Record, i: usize) Field {
        std.debug.assert(i < self.fields.len);
        return .{ .bytes = self.fields[i], .last_in_record = i + 1 == self.fields.len };
    }

    /// The field under column `name`, or null if there is no attached header, the name
    /// is unknown, or the record is too short to have that column.
    pub fn get(self: Record, name: []const u8) ?Field {
        const h = self.header orelse return null;
        return h.col(self.fields, name);
    }

    /// Convert the `i`th field to `T`. Asserts `i` is in range; see `convert.as`.
    pub fn as(self: Record, i: usize, comptime T: type) convert.ConvertError!T {
        std.debug.assert(i < self.fields.len);
        return convert.as(self.fields[i], T);
    }
};
