// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! A zero-allocation view over a record's column names, so fields can be reached by
//! name instead of position. Built entirely on top of the borrowed slices a parser
//! already produces — no scanner code, no allocation.
//!
//! Lifetime: `Header` borrows the name slices; it does not own their bytes. Because
//! `SimdParser.nextRecord` reuses its destination buffer on every call, a `Header`
//! built directly from the first record is clobbered by the next read — use `capture`
//! to copy the *slice pointers* into a caller-owned array that outlives later reads.
//! `capture` copies pointers, not bytes: an unquoted header name points into the input
//! (durable), but a quoted/escaped name points into `scratch` and dies at the next
//! `resetScratch`. So the standalone Header/Record path wants clean (unquoted) header
//! names, or copy the bytes yourself. (`reader(T)` sidesteps this by storing indices.)

const std = @import("std");
const types = @import("types.zig");
const Field = types.Field;

pub const Header = struct {
    names: []const []const u8,

    pub fn init(names: []const []const u8) Header {
        return .{ .names = names };
    }

    pub fn len(self: Header) usize {
        return self.names.len;
    }

    /// Column index of `name`, or null if absent. Linear scan (column counts are small).
    pub fn index(self: Header, name: []const u8) ?usize {
        for (self.names, 0..) |n, i| if (std.mem.eql(u8, n, name)) return i;
        return null;
    }

    /// The field in `record` under column `name`, or null if the name is unknown or the
    /// record is too short to have that column.
    pub fn col(self: Header, record: []const []const u8, name: []const u8) ?Field {
        const i = self.index(name) orelse return null;
        if (i >= record.len) return null;
        return .{ .bytes = record[i], .last_in_record = i + 1 == record.len };
    }

    /// Copy `record`'s slice pointers into caller-owned `slots` so the resulting header
    /// survives later `nextRecord` calls (which reuse the row buffer). The bytes are not
    /// copied — see the lifetime note at the top of this file. `slots` must have room.
    pub fn capture(record: []const []const u8, slots: [][]const u8) Header {
        std.debug.assert(slots.len >= record.len);
        @memcpy(slots[0..record.len], record);
        return .{ .names = slots[0..record.len] };
    }
};
