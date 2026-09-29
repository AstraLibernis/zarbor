// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Scalar (byte-at-a-time) CSV parser — the lenient, in-memory reference.
//!
//! Operates on a `[]const u8` and never allocates. It hands back one `Field` at a
//! time; a field's bytes are usually a slice pointing *directly into the input*
//! (zero-copy). The only exception is a quoted field containing an escaped quote
//! (`""`), which is rewritten into the caller-provided scratch buffer.
//!
//! Lifetime of `Field.bytes`:
//!   * Fields that point into the input are valid as long as the input is.
//!   * Unescaped fields point into `scratch`; they stay valid until
//!     `resetScratch()`. `nextRecord` resets scratch at each record boundary, so
//!     per-record collection is always safe.
//!
//! RFC 4180 with two pragmatic relaxations (this parser only — the SIMD parsers
//! are strict):
//!   * A quote opens a quoted field only as a field's *first* byte; a quote
//!     elsewhere in an unquoted field is literal (`3" pipe` → `3" pipe`).
//!   * `\n`, `\r\n`, and a lone `\r` all terminate a record.

const builtin = @import("builtin");
const types = @import("types.zig");

const Options = types.Options;
const Error = types.Error;
const Field = types.Field;

pub const Parser = struct {
    input: []const u8,
    pos: usize,
    scratch: []u8,
    scratch_used: usize,
    opts: Options,
    /// True when the previous field ended on a delimiter, so another field is
    /// owed even at end of input (a trailing delimiter means a final empty
    /// field: `"a,"` is `["a", ""]`, matching RFC 4180 / Go / Python).
    pending: bool,

    /// `scratch` is only ever written when a quoted field contains an escaped
    /// quote (`""`). If your data has none, an empty scratch (`&.{}`) is fine.
    pub fn init(input: []const u8, scratch: []u8, opts: Options) Error!Parser {
        try opts.validate();
        return .{
            .input = input,
            .pos = 0,
            .scratch = scratch,
            .scratch_used = 0,
            .opts = opts,
            .pending = false,
        };
    }

    /// Reclaim the scratch buffer. Invalidates any previously returned field
    /// that pointed into scratch (an unescaped field); fields that point into
    /// the input are unaffected.
    pub fn resetScratch(self: *Parser) void {
        // Debug-only: poison the region being reclaimed so any `Field.bytes` still
        // pointing into scratch (a retained unescaped field) reads obvious garbage
        // in tests instead of silently-valid stale bytes. Zero cost in release.
        if (builtin.mode == .Debug) @memset(self.scratch[0..self.scratch_used], 0xAA);
        self.scratch_used = 0;
    }

    /// Returns the next field, or `null` at end of input.
    pub fn next(self: *Parser) Error!?Field {
        if (self.pos >= self.input.len) {
            // A trailing delimiter leaves one final empty field owed.
            if (self.pending) {
                self.pending = false;
                return .{ .bytes = self.input[self.input.len..], .last_in_record = true };
            }
            return null;
        }
        self.pending = false;
        if (self.input[self.pos] == self.opts.quote) {
            return try self.quotedField();
        }
        return self.unquotedField();
    }

    /// Read a whole record into `dst`, returning the fields actually written
    /// (a sub-slice of `dst`), or `null` at end of input. Scratch is reset at
    /// entry, so every field in the returned slice is simultaneously valid.
    pub fn nextRecord(self: *Parser, dst: [][]const u8) Error!?[]const []const u8 {
        self.resetScratch();
        var n: usize = 0;
        while (true) {
            const f = try self.next() orelse {
                if (n == 0) return null;
                break; // input ended mid-record (no trailing newline)
            };
            if (n >= dst.len) return Error.TooManyFields;
            dst[n] = f.bytes;
            n += 1;
            if (f.last_in_record) break;
        }
        return dst[0..n];
    }

    /// Scan an unquoted field: data runs until a delimiter, a newline, or the
    /// end of input. Quotes are literal here (see the module relaxations).
    fn unquotedField(self: *Parser) Field {
        const start = self.pos;
        var i = self.pos;
        const input = self.input;
        const delim = self.opts.delimiter;
        while (i < input.len) : (i += 1) {
            const c = input[i];
            if (c == delim or c == '\n' or c == '\r') break;
        }
        const bytes = input[start..i];
        self.pos = i;
        return .{ .bytes = bytes, .last_in_record = self.consumeTerminator() };
    }

    /// Scan a quoted field. `self.pos` is on the opening quote.
    fn quotedField(self: *Parser) Error!Field {
        const input = self.input;
        const quote = self.opts.quote;
        self.pos += 1; // consume opening quote
        const content_start = self.pos;

        var i = self.pos;
        var needs_unescape = false;
        const close = while (i < input.len) : (i += 1) {
            if (input[i] != quote) continue;
            // A quote: either an escaped pair `""` or the closing quote.
            if (i + 1 < input.len and input[i + 1] == quote) {
                needs_unescape = true;
                i += 1; // skip the second quote of the pair; loop's += 1 skips none extra
                continue;
            }
            break i; // closing quote position
        } else return Error.UnterminatedQuote;

        const raw = input[content_start..close];
        self.pos = close + 1; // move past the closing quote

        const value = if (!needs_unescape) raw else blk: {
            // Collapse every `""` down to a single quote into scratch. The
            // result is never longer than `raw`, so one length check suffices.
            const dst = self.scratch[self.scratch_used..];
            if (dst.len < raw.len) return Error.ScratchTooSmall;
            var w: usize = 0;
            var j: usize = 0;
            while (j < raw.len) {
                if (raw[j] == quote) {
                    // Inside `raw`, every quote is the first of an escaped pair.
                    dst[w] = quote;
                    w += 1;
                    j += 2;
                } else {
                    dst[w] = raw[j];
                    w += 1;
                    j += 1;
                }
            }
            self.scratch_used += w;
            break :blk dst[0..w];
        };

        // After a closing quote only a delimiter / newline / EOF may follow.
        if (self.pos < input.len) {
            const c = input[self.pos];
            if (c != self.opts.delimiter and c != '\n' and c != '\r') {
                return Error.InvalidQuote;
            }
        }
        return .{ .bytes = value, .last_in_record = self.consumeTerminator() };
    }

    /// Consume the byte(s) that end a field. Returns true if the field ended a
    /// record (newline or EOF), false if a delimiter (more fields follow).
    fn consumeTerminator(self: *Parser) bool {
        if (self.pos >= self.input.len) return true;
        const c = self.input[self.pos];
        if (c == self.opts.delimiter) {
            self.pos += 1;
            self.pending = true; // another field follows, even if input ends here
            return false;
        }
        if (c == '\n') {
            self.pos += 1;
            return true;
        }
        if (c == '\r') {
            self.pos += 1;
            if (self.pos < self.input.len and self.input[self.pos] == '\n') {
                self.pos += 1; // CRLF
            }
            return true;
        }
        unreachable; // callers only stop on delimiter / newline / EOF
    }
};
