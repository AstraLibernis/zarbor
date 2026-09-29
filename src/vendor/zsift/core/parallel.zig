// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Multi-core parsing: split one input at exact record starts, parse ranges in parallel.
//!
//! Under strict RFC 4180 quoting (enforced by the SIMD paths), a position is inside a
//! quoted field iff an odd number of quotes precede it (`""` adds two). So the split is
//! exact, without speculation, in two parallel phases: `countQuotes` per range, then a
//! prefix parity per cut and `recordStartAfter` from each cut. `splitRecords` is the
//! serial reference; `forEachField` runs the phases and the range parses on the caller's
//! `std.Io`. Invalid quoting is reported by each range's parser.

const std = @import("std");
const types = @import("types.zig");
const simd = @import("simd.zig");
const assert = std.debug.assert;

const Options = types.Options;
const Error = types.Error;
const classify = simd.classify;
const chunk_len = classify.chunk_len;
const Vec = @Vector(chunk_len, u8);

/// Number of `quote` bytes in `bytes` (SIMD compare + popcount per 64-byte chunk).
pub fn countQuotes(bytes: []const u8, quote: u8) u64 {
    var total: u64 = 0;
    var i: usize = 0;
    while (i + chunk_len <= bytes.len) : (i += chunk_len) {
        const v: Vec = bytes[i..][0..chunk_len].*;
        total += @popCount(@as(u64, @bitCast(v == @as(Vec, @splat(quote)))));
    }
    while (i < bytes.len) : (i += 1) total += @intFromBool(bytes[i] == quote);
    return total;
}

/// First record start at or after `from` (`input.len` if none), given whether `from`
/// is inside a quoted field. A cut inside a CRLF moves past the `\n`.
pub fn recordStartAfter(input: []const u8, from: usize, in_quote: bool, opts: Options) usize {
    assert(from <= input.len);
    if (from == 0) return 0;
    if (from == input.len) return input.len;
    if (!in_quote) {
        // Already a record start if the previous byte ended a record.
        const prev = input[from - 1];
        if (prev == '\n' or (prev == '\r' and input[from] != '\n')) return from;
    }
    var carry: u64 = if (in_quote) ~@as(u64, 0) else 0;
    var base = from;
    while (base < input.len) : (base += chunk_len) {
        const t = classify.terminatorsAt(input, base, opts, &carry);
        if (t != 0) {
            const at = base + @ctz(t);
            if (input[at] == '\r' and at + 1 < input.len and input[at + 1] == '\n') return at + 2;
            return at + 1;
        }
    }
    return input.len;
}

/// Split `input` into `bounds.len - 1` ranges near equal offsets, each starting at a
/// record start (non-decreasing; empty when one record spans a cut). Serial reference
/// of the two phases. An odd quote total is `UnterminatedQuote`.
pub fn splitRecords(input: []const u8, opts: Options, bounds: []usize) Error!void {
    try opts.validate();
    assert(bounds.len >= 2);
    const n = bounds.len - 1;
    var quotes_before: u64 = 0;
    var prev_cut: usize = 0;
    bounds[0] = 0;
    for (1..n) |i| {
        const cut = rawCut(input.len, n, i);
        quotes_before += countQuotes(input[prev_cut..cut], opts.quote);
        prev_cut = cut;
        const start = recordStartAfter(input, cut, quotes_before % 2 == 1, opts);
        bounds[i] = @max(start, bounds[i - 1]);
    }
    quotes_before += countQuotes(input[prev_cut..], opts.quote);
    if (quotes_before % 2 == 1) return Error.UnterminatedQuote;
    bounds[n] = input.len;
}

/// Byte offset of raw cut `i` of `n` (before snapping to a record start).
pub fn rawCut(len: usize, n: usize, i: usize) usize {
    return @intCast(@as(u128, len) * i / n);
}

/// Most workers `forEachField` accepts (its bookkeeping lives on the stack).
pub const max_workers = 256;

/// Parallel push parse on exactly `ctxs.len` workers (see `forEachField` for size-based
/// selection). Range `i` goes to `onField(ctxs[i], …)` with `scratches[i]`; the sinks'
/// output concatenated in index order is the serial field sequence. Concurrency comes
/// from `io`; zsift starts no threads and allocates nothing. Keep each sink on its own
/// cache line: packed sinks (false sharing) measured slower than serial. Errors: the
/// earliest failing range's, which equals the serial error; sinks may hold partial output.
pub fn forEachFieldExact(
    io: std.Io,
    input: []const u8,
    opts: Options,
    scratches: []const []u8,
    ctxs: anytype,
    comptime onField: fn (std.meta.Elem(@TypeOf(ctxs)), bytes: []const u8, last_in_record: bool) void,
) (Error || std.Io.Cancelable)!void {
    try opts.validate();
    const n = ctxs.len;
    if (n == 0 or n > max_workers or scratches.len != n) return Error.BadWorkerCount;
    if (n == 1) return simd.forEachField(input, scratches[0], opts, ctxs[0], onField);

    const Ctx = std.meta.Elem(@TypeOf(ctxs));
    const Task = struct {
        fn parse(out: *?Error, bytes: []const u8, scratch: []u8, o: Options, ctx: Ctx) void {
            simd.forEachField(bytes, scratch, o, ctx, onField) catch |e| {
                out.* = e;
            };
        }
    };

    var bounds: [max_workers + 1]usize = undefined;
    try splitParallel(io, input, opts, n, &bounds);
    // Phase 3: parse every range.
    var errs: [max_workers]?Error = @splat(null);
    {
        var g: std.Io.Group = .init;
        for (0..n) |i| g.async(io, Task.parse, .{ &errs[i], input[bounds[i]..bounds[i + 1]], scratches[i], opts, ctxs[i] });
        try g.await(io);
    }
    for (errs[0..n]) |e| if (e) |err| return err;
}

/// Phases 1 and 2 on `io`: `bounds[0..n + 1]` = record-aligned range starts.
fn splitParallel(io: std.Io, input: []const u8, opts: Options, n: usize, bounds: *[max_workers + 1]usize) std.Io.Cancelable!void {
    const Task = struct {
        fn count(out: *u64, bytes: []const u8, quote: u8) void {
            out.* = countQuotes(bytes, quote);
        }
        fn start(out: *usize, text: []const u8, from: usize, in_quote: bool, o: Options) void {
            out.* = recordStartAfter(text, from, in_quote, o);
        }
    };
    var counts: [max_workers]u64 = undefined;
    {
        var g: std.Io.Group = .init;
        for (0..n) |i| g.async(io, Task.count, .{ &counts[i], input[rawCut(input.len, n, i)..rawCut(input.len, n, i + 1)], opts.quote });
        try g.await(io);
    }
    // Prefix parity per cut, then its record start. An odd total is left to the range
    // holding the open quote, so an earlier defect keeps its place in error order.
    var g: std.Io.Group = .init;
    var quotes_before: u64 = 0;
    bounds[0] = 0;
    for (1..n) |i| {
        quotes_before += counts[i - 1];
        g.async(io, Task.start, .{ &bounds[i], input, rawCut(input.len, n, i), quotes_before % 2 == 1, opts });
    }
    try g.await(io);
    for (1..n) |i| bounds[i] = @max(bounds[i], bounds[i - 1]);
    bounds[n] = input.len;
}

/// `forEachFieldExact` on as many of the given workers as the input keeps busy
/// (`workersFor`): small inputs run serially on `ctxs[0]`; unused sinks get nothing.
pub fn forEachField(
    io: std.Io,
    input: []const u8,
    opts: Options,
    scratches: []const []u8,
    ctxs: anytype,
    comptime onField: fn (std.meta.Elem(@TypeOf(ctxs)), bytes: []const u8, last_in_record: bool) void,
) (Error || std.Io.Cancelable)!void {
    if (ctxs.len == 0 or ctxs.len > max_workers or scratches.len != ctxs.len) return Error.BadWorkerCount;
    const k = workersFor(input.len, ctxs.len);
    return forEachFieldExact(io, input, opts, scratches[0..k], ctxs[0..k], onField);
}

/// Workers worth using for `len` bytes, at most `available`: serial (1) below
/// `min_parallel_bytes`, else one per `min_bytes_per_worker`.
pub fn workersFor(len: usize, available: usize) usize {
    if (len < min_parallel_bytes) return 1;
    return @max(1, @min(available, len / min_bytes_per_worker));
}

/// Below this, workers cost more than they save (`zig build sweep`; see ROADMAP M4).
pub const min_parallel_bytes: usize = 2 << 20;

/// Smallest range worth giving a worker (same sweep).
pub const min_bytes_per_worker: usize = 384 << 10;

const stream = @import("stream.zig");

/// `parseReader` options. Unlike `stream.AutoOptions`, known-size inputs up to 1 GiB
/// load into memory by default: the parallel win needs them there.
pub const ReaderOptions = struct {
    csv: Options = .{},
    /// Known-size inputs up to this load into memory; others stream into `ctxs[0]`.
    in_memory_threshold: usize = 1 << 30,
    /// Safety cap on the in-memory read.
    max_in_memory: usize = 1 << 30,
};

/// Reader facade: load a known-size input and `forEachField` it, or stream it serially
/// into `ctxs[0]` (`scratches[0]` must hold the reader's window). Returns the strategy.
pub fn parseReader(
    io: std.Io,
    gpa: std.mem.Allocator,
    r: *std.Io.Reader,
    size_hint: ?u64,
    ro: ReaderOptions,
    scratches: []const []u8,
    ctxs: anytype,
    comptime onField: fn (std.meta.Elem(@TypeOf(ctxs)), bytes: []const u8, last_in_record: bool) void,
) !stream.Strategy {
    try ro.csv.validate();
    if (ctxs.len == 0 or ctxs.len > max_workers or scratches.len != ctxs.len) return Error.BadWorkerCount;
    const strategy = stream.decide(size_hint, ro.in_memory_threshold);
    switch (strategy) {
        .in_memory => {
            const buf = try r.allocRemaining(gpa, .limited(ro.max_in_memory));
            defer gpa.free(buf);
            try forEachField(io, buf, ro.csv, scratches, ctxs, onField);
        },
        .streaming => try stream.streamReader(r, scratches[0], ro.csv, ctxs[0], onField),
    }
    return strategy;
}

const reader_mod = @import("reader.zig");
const Header = @import("header.zig").Header;
const Record = @import("record.zig").Record;

/// Capture the first record of `input` as a `Header` (names in `slots`, escaped names
/// unescaped into `scratch`, which must outlive the header) and return it with the rest
/// of the input, which starts at a record start: pass `data` to `forEachRecord`.
pub fn splitHeader(input: []const u8, opts: Options, slots: [][]const u8, scratch: []u8) Error!struct { header: Header, data: []const u8 } {
    var p = try simd.SimdParser.init(input, scratch, opts);
    const rec = (try p.nextRecord(slots)) orelse return .{ .header = Header.init(slots[0..0]), .data = input[input.len..] };
    return .{ .header = Header.init(rec), .data = input[@min(p.field_start, input.len)..] };
}

/// Parallel records: each record of `data` reaches `onRecord(ctxs[i], rec)` as a
/// `Record` view (with `header` attached for `get(name)` when given), at most
/// `max_cols` fields per record. Worker count by size, ordering and errors as
/// `forEachField`. The record borrows its range and is valid only during the call.
pub fn forEachRecord(
    io: std.Io,
    data: []const u8,
    opts: Options,
    header: ?*const Header,
    comptime max_cols: usize,
    scratches: []const []u8,
    ctxs: anytype,
    comptime onRecord: fn (std.meta.Elem(@TypeOf(ctxs)), rec: Record) void,
) (Error || std.Io.Cancelable)!void {
    try opts.validate();
    if (ctxs.len == 0 or ctxs.len > max_workers or scratches.len != ctxs.len) return Error.BadWorkerCount;
    const n = workersFor(data.len, ctxs.len);
    const Ctx = std.meta.Elem(@TypeOf(ctxs));
    const Task = struct {
        fn run(out: *?Error, bytes: []const u8, scratch: []u8, o: Options, h: ?*const Header, ctx: Ctx) void {
            var p = simd.SimdParser.init(bytes, scratch, o) catch |e| {
                out.* = e;
                return;
            };
            var row: [max_cols][]const u8 = undefined;
            while (p.nextRecord(&row) catch |e| {
                out.* = e;
                return;
            }) |fields| onRecord(ctx, if (h) |hh| Record.withHeader(fields, hh) else Record.init(fields));
        }
    };
    var bounds: [max_workers + 1]usize = undefined;
    if (n > 1) try splitParallel(io, data, opts, n, &bounds) else {
        bounds[0] = 0;
        bounds[1] = data.len;
    }
    var errs: [max_workers]?Error = @splat(null);
    var g: std.Io.Group = .init;
    for (0..n) |i| g.async(io, Task.run, .{ &errs[i], data[bounds[i]..bounds[i + 1]], scratches[i], opts, header, ctxs[i] });
    try g.await(io);
    for (errs[0..n]) |e| if (e) |err| return err;
}

/// Parallel typed rows: `R` is `zsift.reader(T)` or `zsift.readerWide(T, n)`; every
/// record becomes an `R.Row` passed to `onRow(ctxs[i], row)`. With `header`, the first
/// record is read once, serially, to map struct fields to columns by name (as
/// `withHeader`), and that mapping is shared by every worker. Worker count by size,
/// ordering and errors as `forEachField`; `[]const u8` row fields borrow the input.
pub fn forEachRow(
    comptime R: type,
    io: std.Io,
    input: []const u8,
    opts: Options,
    header: bool,
    scratches: []const []u8,
    ctxs: anytype,
    comptime onRow: fn (std.meta.Elem(@TypeOf(ctxs)), row: R.Row) void,
) (reader_mod.ReaderError || std.Io.Cancelable)!void {
    try opts.validate();
    if (ctxs.len == 0 or ctxs.len > max_workers or scratches.len != ctxs.len) return Error.BadWorkerCount;
    var head = try R.init(input, scratches[0], opts);
    var data = input;
    if (header) {
        try head.withHeader();
        data = input[@min(head.parser.field_start, input.len)..];
    }
    const perm = head.perm;
    const n = workersFor(data.len, ctxs.len);
    const Ctx = std.meta.Elem(@TypeOf(ctxs));
    const Task = struct {
        fn run(out: *?reader_mod.ReaderError, bytes: []const u8, scratch: []u8, o: Options, p: @TypeOf(perm), ctx: Ctx) void {
            var r = R.init(bytes, scratch, o) catch |e| {
                out.* = e;
                return;
            };
            r.perm = p;
            while (r.next() catch |e| {
                out.* = e;
                return;
            }) |row| onRow(ctx, row);
        }
    };
    var bounds: [max_workers + 1]usize = undefined;
    if (n > 1) try splitParallel(io, data, opts, n, &bounds) else {
        bounds[0] = 0;
        bounds[1] = data.len;
    }
    var errs: [max_workers]?reader_mod.ReaderError = @splat(null);
    var g: std.Io.Group = .init;
    for (0..n) |i| g.async(io, Task.run, .{ &errs[i], data[bounds[i]..bounds[i + 1]], scratches[i], opts, perm, ctxs[i] });
    try g.await(io);
    for (errs[0..n]) |e| if (e) |err| return err;
}
