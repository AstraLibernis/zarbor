// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Types shared by the scalar (`csv.Parser`) and SIMD (`csv.SimdParser`) parsers.

const std = @import("std");
const convert = @import("convert.zig");

/// Re-exported so a single `@import("types.zig")` reaches both `Field` and the
/// conversion error set its methods return.
pub const ConvertError = convert.ConvertError;

pub const Options = struct {
    /// Byte that separates fields within a record.
    delimiter: u8 = ',',
    /// Byte used to quote fields that contain the delimiter, a newline, or a quote.
    quote: u8 = '"',

    /// Reject configurations the parsers cannot represent unambiguously, so misuse
    /// fails loud instead of emitting silently-wrong fields.
    ///
    /// The SIMD classifier ORs the delimiter, `\n`, and `\r` bitmasks into a single
    /// "separator" mask and then masks *out* quoted regions using the quote bitmask.
    /// If the delimiter equalled the quote, or either equalled a record terminator
    /// (`\n`/`\r`), those masks would overlap and the same byte would be classified
    /// two ways at once — the parser would split or swallow fields wrongly with no
    /// error. Rather than trust the caller, we check up front. `init` calls this for
    /// you; call it yourself before the push/stream entry points if you build
    /// `Options` from untrusted input.
    pub fn validate(o: Options) Error!void {
        if (o.delimiter == o.quote) return Error.InvalidOptions;
        if (o.delimiter == '\n' or o.delimiter == '\r') return Error.InvalidOptions;
        if (o.quote == '\n' or o.quote == '\r') return Error.InvalidOptions;
    }
};

pub const Error = error{
    /// A quoted field was opened but the input ended before the closing quote.
    UnterminatedQuote,
    /// A closing quote was followed by something other than a delimiter,
    /// a newline, or end-of-input (e.g. `"ab"c`).
    InvalidQuote,
    /// A quoted field contained an escaped quote and needed rewriting, but the
    /// caller-supplied scratch buffer was too small to hold the result.
    ScratchTooSmall,
    /// `nextRecord` was given a destination slice with fewer slots than the
    /// record has fields.
    TooManyFields,
    /// `Options.delimiter`/`quote` are unusable: they collide with each other or
    /// with a record terminator (`\n`/`\r`). See `Options.validate`.
    InvalidOptions,
    /// `parallel.forEachField` was given more workers than `parallel.max_workers`,
    /// or a different number of scratch buffers than sinks.
    BadWorkerCount,
};

/// One field of a record. `last_in_record` is true when this field is the final
/// one of its record, i.e. the next field (if any) belongs to a new record.
///
/// `bytes` is *borrowed*, never owned: it points either directly into the parser's
/// input (the zero-copy common case) or into the caller-supplied `scratch` (only
/// when a quoted field held an escaped `""` that had to be collapsed). A scratch-
/// backed slice is invalidated by the next `resetScratch()` (which `nextRecord`
/// calls per record), and every push/stream field is valid only for the duration
/// of the callback. Copy the bytes if you need to outlive that. Debug builds poison
/// reclaimed scratch (see `resetScratch`) so a stale retain shows up loudly in tests.
pub const Field = struct {
    bytes: []const u8,
    last_in_record: bool,

    // Opt-in typed access over the borrowed bytes. These delegate to `convert.zig`
    // and never touch the parser's hot path (adding methods leaves `Field`'s layout
    // unchanged); an unused converter compiles to nothing. See `convert.as`.

    /// Convert the field to `T` (int/float/bool/enum/`[]const u8`/optional-of-those).
    pub fn as(self: Field, comptime T: type) convert.ConvertError!T {
        return convert.as(self.bytes, T);
    }
    pub fn asInt(self: Field, comptime T: type) convert.ConvertError!T {
        return convert.asInt(self.bytes, T);
    }
    pub fn asFloat(self: Field, comptime T: type) convert.ConvertError!T {
        return convert.asFloat(self.bytes, T);
    }
    pub fn asBool(self: Field) convert.ConvertError!bool {
        return convert.asBool(self.bytes);
    }
    pub fn asEnum(self: Field, comptime T: type) convert.ConvertError!T {
        return convert.asEnum(self.bytes, T);
    }
    /// An empty field is `null`; otherwise convert to `T`.
    pub fn asOptional(self: Field, comptime T: type) convert.ConvertError!?T {
        return if (self.bytes.len == 0) null else try convert.as(self.bytes, T);
    }
    pub fn isEmpty(self: Field) bool {
        return self.bytes.len == 0;
    }
    pub fn eql(self: Field, s: []const u8) bool {
        return std.mem.eql(u8, self.bytes, s);
    }
    /// Zero-copy: a sub-slice with surrounding ASCII whitespace stripped. Still borrows.
    pub fn trimmed(self: Field) Field {
        return .{ .bytes = std.mem.trim(u8, self.bytes, " \t\r\n"), .last_in_record = self.last_in_record };
    }
};
