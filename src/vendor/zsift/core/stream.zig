// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Streaming CSV over a `std.Io.Reader`, plus an auto-selecting facade that
//! picks between slurp-into-memory and streaming based on input size.
//!
//! Streaming model: the 0.16 `Reader` already maintains a sliding window
//! (`buffer` / `seek` / `end`). We process the largest prefix of the window that
//! consists of *whole records*, in one batched `forEachField` pass (so SIMD
//! amortizes over many records, not one), then `toss` that prefix and let the
//! reader refill. Because we only ever consume up to a record boundary, every
//! refill begins outside any quoted region — so no quote state has to cross a
//! refill, only a re-scan of the small leftover partial record.
//!
//! Field lifetime: fields handed to the callback are valid only for the duration
//! of that call (the reader buffer is reused on the next refill). This matches
//! every streaming CSV reader (Go's `ReuseRecord`, etc.).
//!
//! Like the SIMD parser, streaming assumes RFC 4180-strict quoting. The window
//! must be larger than the longest record; a record that fills the entire window
//! with no terminator yields `error.RecordTooLong`.

const std = @import("std");
const types = @import("types.zig");
const simd = @import("simd.zig");
const classify = simd.classify; // classification primitives folded into simd.zig

const Reader = std.Io.Reader;

pub const Options = types.Options;
pub const Error = types.Error;

pub const StreamError = Error || error{
    /// The underlying reader failed.
    ReadFailed,
    /// A single record is larger than the reader's buffer (window).
    RecordTooLong,
};

/// Length of the largest prefix of `window` that consists of whole records:
/// the index one past the last record terminator lying outside a quoted region.
/// Returns 0 when the window holds no complete record.
///
/// Vectorized: `classify.terminatorsAt` gives, per 64-byte chunk, the `\n`/`\r` bits
/// that lie outside quoted regions (escaped `""` self-cancels in the quote mask).
/// We take the highest such bit per chunk — later chunks hold later boundaries —
/// resolving `\r\n` and deferring only a lone `\r` at the very end of the window
/// (it may be a split `\r\n` completed by the next refill).
pub fn completeRecordsLen(window: []const u8, opts: Options) usize {
    var carry: u64 = 0;
    var last: usize = 0;
    var base: usize = 0;
    while (base < window.len) : (base += classify.chunk_len) {
        var t = classify.terminatorsAt(window, base, opts, &carry);
        while (t != 0) {
            // `t != 0` ⇒ `@clz(t)` ∈ [0, 63] ⇒ `hi` ∈ [0, 63], so the u6 shift-amount
            // cast below is always in range.
            const hi: usize = 63 - @clz(t);
            std.debug.assert(hi < 64);
            if (resolveTerminator(window, base + hi)) |boundary| {
                last = boundary; // highest definitive boundary in this chunk
                break;
            }
            t &= ~(@as(u64, 1) << @intCast(hi)); // lone trailing '\r': try the next-highest
        }
    }
    return last;
}

/// Resolve a record terminator at absolute offset `at` to the index one past it,
/// or `null` to defer (a lone `\r` at the very end of the window).
fn resolveTerminator(window: []const u8, at: usize) ?usize {
    if (window[at] == '\n') return at + 1;
    // '\r': consume a following '\n' as CRLF; defer if it is the window's last byte.
    if (at + 1 < window.len) return if (window[at + 1] == '\n') at + 2 else at + 1;
    return null;
}

/// Push every field of the CSV stream to `onField(ctx, bytes, last_in_record)`,
/// reading from `r` in windows. `scratch` holds unescaped (`""`) field bytes and
/// must be at least as large as the longest field; sizing it to the reader's
/// buffer length is always safe.
pub fn streamReader(
    r: *Reader,
    scratch: []u8,
    opts: Options,
    ctx: anytype,
    comptime onField: fn (@TypeOf(ctx), bytes: []const u8, last_in_record: bool) void,
) StreamError!void {
    try opts.validate();
    // Set after a record was terminated by a `\r` sitting at the exact window end:
    // if the next window opens with `\n`, that byte is the CRLF's second half and
    // must be swallowed rather than read as an empty leading record.
    var swallow_lf = false;
    while (true) {
        var window = r.buffered();
        if (swallow_lf) {
            if (window.len == 0) {
                // Need one byte to decide; EndOfStream means the `\r` was the last
                // byte of input — the record is already emitted, so we are done.
                r.fillMore() catch |err| switch (err) {
                    error.EndOfStream => return,
                    error.ReadFailed => return StreamError.ReadFailed,
                };
                window = r.buffered();
            }
            if (window[0] == '\n') r.toss(1); // CRLF second half; a lone `\r` had none
            swallow_lf = false;
            continue;
        }
        const boundary = completeRecordsLen(window, opts);
        if (boundary > 0) {
            try simd.forEachField(window[0..boundary], scratch, opts, ctx, onField);
            r.toss(boundary);
            continue;
        }
        // No complete record buffered.
        if (r.bufferedLen() == r.buffer.len) {
            // A full window whose only would-be terminator is a `\r` at the very
            // end read as "no complete record" solely because `completeRecordsLen`
            // defers a trailing `\r` (it might begin a CRLF). But the `\r` *does*
            // terminate the record — parse the window, then swallow a following
            // `\n` next round. This is a complete record, not an over-long one.
            if (window.len > 0 and window[window.len - 1] == '\r') {
                try simd.forEachField(window, scratch, opts, ctx, onField);
                r.toss(window.len);
                swallow_lf = true;
                continue;
            }
            // Otherwise the buffer is genuinely full with no terminator. Probe one
            // byte beyond it (into our own buffer, leaving the window untouched) to
            // tell a truly over-long record from a final record that fills the
            // window exactly (EOF here).
            var probe: [1]u8 = undefined;
            var bufs: [1][]u8 = .{&probe};
            const got = r.vtable.readVec(r, &bufs) catch |err| switch (err) {
                error.EndOfStream => {
                    try simd.forEachField(window, scratch, opts, ctx, onField);
                    return;
                },
                error.ReadFailed => return StreamError.ReadFailed,
            };
            // Any byte beyond a full window means the record does not fit.
            _ = got;
            return StreamError.RecordTooLong;
        }
        r.fillMore() catch |err| switch (err) {
            error.EndOfStream => {
                // End of input: any leftover is the final record (no trailing
                // newline). It is shorter than the buffer, so it is complete.
                const rem = r.buffered();
                if (rem.len > 0) try simd.forEachField(rem, scratch, opts, ctx, onField);
                return;
            },
            error.ReadFailed => return StreamError.ReadFailed,
        };
    }
}

// ---------------------------------------------------------------------------
// Auto-selecting facade
// ---------------------------------------------------------------------------

pub const Strategy = enum { in_memory, streaming };

pub const AutoOptions = struct {
    csv: Options = .{},
    /// Inputs whose size is known and at most this are slurped into memory and
    /// parsed with the fastest in-memory push path. Larger or unknown-size
    /// inputs are streamed with bounded memory. (Size is the right axis here:
    /// it decides memory footprint, not which parser is faster.)
    in_memory_threshold: usize = 4 << 20, // 4 MiB
    /// Safety cap on the in-memory slurp.
    max_in_memory: usize = 1 << 30, // 1 GiB
};

/// Decide a strategy from a size hint. `null` (unknown size) streams.
pub fn decide(size_hint: ?u64, threshold: usize) Strategy {
    const sz = size_hint orelse return .streaming;
    return if (sz <= threshold) .in_memory else .streaming;
}

/// Parse a CSV stream, auto-selecting between an in-memory slurp (small/known
/// size) and bounded streaming (large/unknown size). The streaming window is the
/// reader's own buffer, so size it for your largest record. Returns the chosen
/// strategy. This is the allocating convenience layer over the zero-alloc core.
pub fn parseReader(
    gpa: std.mem.Allocator,
    r: *Reader,
    size_hint: ?u64,
    ao: AutoOptions,
    ctx: anytype,
    comptime onField: fn (@TypeOf(ctx), bytes: []const u8, last_in_record: bool) void,
) !Strategy {
    try ao.csv.validate(); // fail loud before we allocate, not mid-parse
    const strategy = decide(size_hint, ao.in_memory_threshold);
    switch (strategy) {
        .in_memory => {
            const buf = try r.allocRemaining(gpa, .limited(ao.max_in_memory));
            defer gpa.free(buf);
            const scratch = try gpa.alloc(u8, @max(buf.len, 64));
            defer gpa.free(scratch);
            try simd.forEachField(buf, scratch, ao.csv, ctx, onField);
        },
        .streaming => {
            const scratch = try gpa.alloc(u8, @max(r.buffer.len, 64));
            defer gpa.free(scratch);
            try streamReader(r, scratch, ao.csv, ctx, onField);
        },
    }
    return strategy;
}
