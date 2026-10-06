// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! bench/tools.zig — zarbor's benchmark glue, in Zig. Replaces run.py, summarise.py,
//! controls.sh, arena/make_figure.py and arena/solver_stress.py; the scripts that run
//! a reference library (LightGBM, XGBoost, scikit-learn) stay Python, as baselines.
//!
//!   zig build tools -- summarise <base.tsv> <new.tsv>    pair two grid runs (seeds)
//!   zig build tools -- figure [scoreboard.json] [out.html]
//!   zig build tools -- grid <zarbor> <tag> [flags...]     the PROTOCOL.md grid, TSV out
//!   zig build tools -- controls                          PROTOCOL.md controls C3-C5
//!   zig build tools -- solver-stress                     do lbfgs and adam meet?
//!
//! Paths: `$ZARBOR_BENCH_DATA` overrides `<repo>/bench/data`; the repo root is injected
//! by build.zig. `controls` needs `$ZARBOR_OLD` (a previous zarbor build): no default.
//! Scratch files go under `<repo>/.zig-cache/tools/` and are removed afterwards.

const std = @import("std");
const zarbor = @import("zarbor");
const paths = @import("tools_paths");

const Allocator = std.mem.Allocator;
const Io = std.Io;
const print = std.debug.print;

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const args = try init.minimal.args.toSlice(gpa);
    const cmd = if (args.len > 1) args[1] else "";
    const rest = if (args.len > 2) args[2..] else args[0..0];
    var out_buf: [1 << 16]u8 = undefined;
    var ow = std.Io.File.stdout().writerStreaming(init.io, &out_buf);
    const out = &ow.interface;
    defer out.flush() catch |e| print("stdout: {s}\n", .{@errorName(e)});
    if (std.mem.eql(u8, cmd, "summarise")) return summarise(init, gpa, out, rest);
    if (std.mem.eql(u8, cmd, "figure")) return figure(init, gpa, rest);
    if (std.mem.eql(u8, cmd, "grid")) return grid(init, gpa, out, rest);
    if (std.mem.eql(u8, cmd, "controls")) return controls(init, gpa, out);
    if (std.mem.eql(u8, cmd, "solver-stress")) return solverStress(init, gpa, out);
    print("usage: tools summarise|figure|grid|controls|solver-stress ... (see bench/tools.zig)\n", .{});
    return error.BadArgs;
}

fn envOr(init: std.process.Init, name: []const u8, default: []const u8) []const u8 {
    return init.environ_map.get(name) orelse default;
}

fn dataDir(init: std.process.Init, gpa: Allocator) ![]const u8 {
    if (init.environ_map.get("ZARBOR_BENCH_DATA")) |d| return d;
    return std.fs.path.join(gpa, &.{ paths.root, "bench", "data" });
}

fn nanoTime() u64 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// A fresh scratch directory under `<repo>/.zig-cache/tools/`.
fn scratchDir(io: Io, gpa: Allocator) ![]const u8 {
    const d = try std.fmt.allocPrint(gpa, "{s}/.zig-cache/tools/{d}", .{ paths.root, nanoTime() });
    try Io.Dir.cwd().createDirPath(io, d);
    return d;
}

const Run = struct { code: u8, text: []const u8 };

/// Run `argv`, return its exit code and stdout followed by stderr.
fn run(init: std.process.Init, gpa: Allocator, argv: []const []const u8) !Run {
    const r = try std.process.run(gpa, init.io, .{ .argv = argv, .stdout_limit = .limited(64 << 20), .stderr_limit = .limited(64 << 20) });
    const code: u8 = switch (r.term) {
        .exited => |c| c,
        else => 255,
    };
    return .{ .code = code, .text = try std.mem.concat(gpa, u8, &.{ r.stdout, r.stderr }) };
}

fn sameFile(io: Io, gpa: Allocator, a: []const u8, b: []const u8) bool {
    const x = Io.Dir.cwd().readFileAlloc(io, a, gpa, .unlimited) catch return false;
    const y = Io.Dir.cwd().readFileAlloc(io, b, gpa, .unlimited) catch return false;
    return x.len > 0 and std.mem.eql(u8, x, y);
}

// ---------------------------------------------------------------- summarise

const lower_is_better = [_]struct { []const u8, bool }{
    .{ "california", true }, .{ "adult", false }, .{ "bank", false }, .{ "ames", true }, .{ "housing", true },
};

const Cell = struct { oof: f64, ms: f64 };
const Seeds = std.array_hash_map.Auto(i64, Cell);

/// dataset -> seed -> (oof, ms) from a grid TSV, datasets in order of first appearance.
fn loadTsv(io: Io, gpa: Allocator, path: []const u8) !std.array_hash_map.String(Seeds) {
    const text = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
    var d: std.array_hash_map.String(Seeds) = .empty;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        var f: [5][]const u8 = undefined;
        var n: usize = 0;
        var it = std.mem.splitScalar(u8, line, '\t');
        while (it.next()) |x| : (n += 1) {
            if (n < 5) f[n] = x else break;
        }
        if (n < 5) continue;
        const oof_at = std.mem.find(u8, f[4], "oof") orelse continue;
        const oof = numberAfter(f[4][oof_at + 3 ..]) orelse continue;
        const gop = try d.getOrPut(gpa, f[1]);
        if (!gop.found_existing) gop.value_ptr.* = .empty;
        try gop.value_ptr.put(gpa, try std.fmt.parseInt(i64, f[2], 10), .{ .oof = oof, .ms = msIn(f[4]) });
    }
    return d;
}

/// The number after optional whitespace (summarise.py's `oof\s+([0-9.]+)`).
fn numberAfter(s: []const u8) ?f64 {
    const t = std.mem.trimStart(u8, s, " \t");
    if (t.len == s.len) return null;
    var e: usize = 0;
    while (e < t.len and (std.ascii.isDigit(t[e]) or t[e] == '.')) e += 1;
    return std.fmt.parseFloat(f64, t[0..e]) catch null;
}

/// The first `<digits> ms` in `s`, or 0.
fn msIn(s: []const u8) f64 {
    var i: usize = 0;
    while (std.mem.findPos(u8, s, i, " ms")) |at| : (i = at + 1) {
        var b = at;
        while (b > 0 and std.ascii.isDigit(s[b - 1])) b -= 1;
        if (b < at) return std.fmt.parseFloat(f64, s[b..at]) catch 0;
    }
    return 0;
}

/// `v` to `p` decimals, rounded from its exact binary value with ties to even — as
/// Python's `{:.pf}` does. (Zig's `{d:.p}` rounds the shortest decimal form instead:
/// 5.145, stored as 5.14499…, came out 5.15 where Python says 5.14.) Exact because a
/// 53-bit mantissa times 10^p (p <= 6) fits f128's 113 bits.
fn fixedDec(buf: []u8, v: f64, comptime p: u8) ![]const u8 {
    comptime std.debug.assert(p <= 6);
    if (std.math.isNan(v)) return std.fmt.bufPrint(buf, "nan", .{});
    const scaled = @as(f128, @abs(v)) * @as(f128, @floatFromInt(comptime std.math.pow(u64, 10, p)));
    var whole = @floor(scaled);
    const frac = scaled - whole;
    if (frac > 0.5 or (frac == 0.5 and @mod(whole, 2) == 1)) whole += 1;
    const n: u128 = @intFromFloat(whole);
    const unit = comptime std.math.pow(u128, 10, p);
    const neg = std.math.signbit(v) and n != 0;
    if (p == 0) return std.fmt.bufPrint(buf, "{s}{d}", .{ if (neg) "-" else "", n });
    return std.fmt.bufPrint(buf, "{s}{d}.{d:0>" ++ std.fmt.comptimePrint("{d}", .{p}) ++ "}", .{ if (neg) "-" else "", n / unit, n % unit });
}

/// `{f}` wrapper printing `v` with `fixedDec`: `w.print("{f}", .{dec(2, x)})`.
fn Dec(comptime p: u8) type {
    return struct {
        v: f64,
        pub fn format(self: @This(), w: *std.Io.Writer) std.Io.Writer.Error!void {
            var b: [64]u8 = undefined;
            try w.writeAll(fixedDec(&b, self.v, p) catch return error.WriteFailed);
        }
    };
}

fn dec(comptime p: u8, v: f64) Dec(p) {
    return .{ .v = v };
}

fn padLeft(w: *std.Io.Writer, width: usize, s: []const u8) !void {
    if (s.len < width) try w.splatByteAll(' ', width - s.len);
    try w.writeAll(s);
}

fn summarise(init: std.process.Init, gpa: Allocator, w: *std.Io.Writer, args: []const []const u8) !void {
    if (args.len != 2) {
        print("usage: tools summarise <base.tsv> <new.tsv>\n", .{});
        return error.BadArgs;
    }
    const a = try loadTsv(init.io, gpa, args[0]);
    const b = try loadTsv(init.io, gpa, args[1]);
    try w.print("{s:<11} {s:>13} {s:>13} {s:>11} {s:>9} {s:>7}  {s:>10}  verdict\n", .{ "dataset", "base", "new", "delta", "sd", "|d|/sd", "ms" });
    for (a.keys(), a.values()) |k, sa| {
        const sb = b.get(k) orelse continue;
        var seeds: std.ArrayList(i64) = .empty;
        for (sa.keys()) |s| if (sb.contains(s)) try seeds.append(gpa, s);
        std.mem.sort(i64, seeds.items, {}, std.sort.asc(i64));
        const n: f64 = @floatFromInt(seeds.items.len);
        var mu: f64 = 0;
        var base: f64 = 0;
        var new: f64 = 0;
        var tms: f64 = 0;
        var nms: f64 = 0;
        for (seeds.items) |s| {
            const x = sa.get(s).?;
            const y = sb.get(s).?;
            mu += y.oof - x.oof;
            base += x.oof;
            new += y.oof;
            tms += x.ms;
            nms += y.ms;
        }
        mu /= n;
        base /= n;
        new /= n;
        tms /= n;
        nms /= n;
        var sd: f64 = std.math.nan(f64);
        if (seeds.items.len > 1) {
            var ss: f64 = 0;
            for (seeds.items) |s| {
                const dlt = sb.get(s).?.oof - sa.get(s).?.oof;
                ss += (dlt - mu) * (dlt - mu);
            }
            sd = @sqrt(ss / (n - 1));
        }
        const lower = for (lower_is_better) |e| {
            if (std.mem.eql(u8, e[0], k)) break e[1];
        } else {
            print("error: no direction known for dataset '{s}'\n", .{k});
            return error.UnknownDataset;
        };
        const good = if (lower) mu < 0 else mu > 0;
        const sig = sd > 0 and @abs(mu) > 2 * sd;
        const v = if (sig) (if (good) "HELPS" else "HURTS") else if (mu == 0) "identical" else "no effect";
        var buf: [64]u8 = undefined;
        try w.print("{s:<11} ", .{k});
        try padLeft(w, 13, try std.fmt.bufPrint(&buf, "{f}", .{dec(6, base)}));
        try w.writeAll(" ");
        try padLeft(w, 13, try std.fmt.bufPrint(&buf, "{f}", .{dec(6, new)}));
        try w.writeAll(" ");
        try padLeft(w, 11, try std.fmt.bufPrint(&buf, "{s}{f}", .{ if (std.math.signbit(mu)) "-" else "+", dec(6, @abs(mu)) }));
        try w.writeAll(" ");
        try padLeft(w, 9, if (std.math.isNan(sd)) "nan" else try std.fmt.bufPrint(&buf, "{f}", .{dec(6, sd)}));
        try w.writeAll(" ");
        try padLeft(w, 7, try std.fmt.bufPrint(&buf, "{f}", .{dec(2, if (sd > 0) @abs(mu) / sd else 0)}));
        try w.writeAll("  ");
        try padLeft(w, 9, try std.fmt.bufPrint(&buf, "{f}", .{dec(2, nms / tms)}));
        try w.print("x  {s}  (n={d})\n", .{ v, seeds.items.len });
    }
}

// ---------------------------------------------------------------- figure

const fig_w: i64 = 1060;
const left: i64 = 240;
const gapw: i64 = 250;
const right: i64 = 28;
const track: i64 = fig_w - left - gapw - right - 40;
const rowh: i64 = 58;
const top: i64 = 104;
const gx0: i64 = left + track + 40;
const t_min = 130.0;
const t_max = 8000.0;
const g_min = 1e-7;
const g_max = 1e-2;

fn tx(ms: f64) f64 {
    return @as(f64, left) + (std.math.log10(ms) - std.math.log10(t_min)) / (std.math.log10(t_max) - std.math.log10(t_min)) * @as(f64, track);
}

fn gx(v: f64) f64 {
    const c = @max(v, g_min);
    return @as(f64, gx0) + (std.math.log10(c) - std.math.log10(g_min)) / (std.math.log10(g_max) - std.math.log10(g_min)) * @as(f64, gapw);
}

fn num(v: std.json.Value) f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => std.math.nan(f64),
    };
}

/// make_figure.py's fmt_ms.
fn fmtMs(buf: []u8, v: f64) ![]const u8 {
    return if (v >= 1000) std.fmt.bufPrint(buf, "{f} s", .{dec(2, v / 1000)}) else std.fmt.bufPrint(buf, "{f} ms", .{dec(0, v)});
}

/// make_figure.py's fmt_score: `{:,.0f}` for an RMSE above 100, else `{:.6f}`.
fn fmtScore(buf: []u8, v: f64, metric: []const u8) ![]const u8 {
    if (!(std.mem.eql(u8, metric, "rmse") and v > 100)) return std.fmt.bufPrint(buf, "{f}", .{dec(6, v)});
    var digits: [48]u8 = undefined;
    const d = try std.fmt.bufPrint(&digits, "{f}", .{dec(0, v)});
    var n: usize = 0;
    for (d, 0..) |c, i| {
        if (i > 0 and (d.len - i) % 3 == 0) {
            buf[n] = ',';
            n += 1;
        }
        buf[n] = c;
        n += 1;
    }
    return buf[0..n];
}

/// Python's `{:.1e}`: one decimal, exponent signed and at least two digits.
fn fmtE1(buf: []u8, v: f64) ![]const u8 {
    var raw: [48]u8 = undefined;
    const s = try std.fmt.bufPrint(&raw, "{e:.1}", .{v});
    const e = std.mem.findScalar(u8, s, 'e').?;
    const exp = try std.fmt.parseInt(i32, s[e + 1 ..], 10);
    return std.fmt.bufPrint(buf, "{s}e{s}{d:0>2}", .{ s[0..e], if (exp < 0) "-" else "+", @abs(exp) });
}

/// Python's `repr` of a float (json.dumps): shortest round-trip digits, `.0` on an
/// integral value, scientific form below 1e-4 or from 1e16.
fn pyFloat(w: *std.Io.Writer, v: f64) !void {
    const a = @abs(v);
    if (a != 0 and (a < 1e-4 or a >= 1e16)) {
        var raw: [64]u8 = undefined;
        const s = try std.fmt.bufPrint(&raw, "{e}", .{v});
        const e = std.mem.findScalar(u8, s, 'e').?;
        const exp = try std.fmt.parseInt(i32, s[e + 1 ..], 10);
        return w.print("{s}e{s}{d:0>2}", .{ s[0..e], if (exp < 0) "-" else "+", @abs(exp) });
    }
    if (@floor(v) == v) return w.print("{f}", .{dec(1, v)});
    try w.print("{d}", .{v});
}

/// Python's json.dumps for the values this page embeds (separators ", " and ": ").
fn pyJson(w: *std.Io.Writer, v: std.json.Value) !void {
    switch (v) {
        .integer => |i| try w.print("{d}", .{i}),
        .float => |f| try pyFloat(w, f),
        .string => |s| try w.print("{f}", .{std.json.fmt(s, .{})}),
        else => return error.UnsupportedJson,
    }
}

const Page = struct { height: i64, axis: []const u8, body: []const u8, tbody: []const u8, data: []const u8 };

fn figure(init: std.process.Init, gpa: Allocator, args: []const []const u8) !void {
    const arena_dir = try std.fs.path.join(gpa, &.{ paths.root, "bench", "arena" });
    const in_path = if (args.len > 0) args[0] else try std.fs.path.join(gpa, &.{ arena_dir, "scoreboard.json" });
    const out_path = if (args.len > 1) args[1] else try std.fs.path.join(gpa, &.{ arena_dir, "scoreboard.html" });
    const text = try Io.Dir.cwd().readFileAlloc(init.io, in_path, gpa, .unlimited);
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, gpa, text, .{});
    const rows = parsed.array.items;

    var body_a: std.Io.Writer.Allocating = .init(gpa);
    var tbody_a: std.Io.Writer.Allocating = .init(gpa);
    var data_a: std.Io.Writer.Allocating = .init(gpa);
    var axis_a: std.Io.Writer.Allocating = .init(gpa);
    const body = &body_a.writer;
    const tbody = &tbody_a.writer;
    const data = &data_a.writer;
    const axis = &axis_a.writer;
    var b1: [64]u8 = undefined;
    var b2: [64]u8 = undefined;
    var b3: [64]u8 = undefined;
    var b4: [64]u8 = undefined;

    try data.writeAll("[");
    for (rows, 0..) |row, i| {
        const r = row.object;
        const zar = r.get("zarbor").?.object;
        const ref = r.get("ref").?.object;
        const key = r.get("key").?.string;
        const reference = r.get("reference").?.string;
        const metric = r.get("metric").?.string;
        const y = @as(f64, @floatFromInt(top + @as(i64, @intCast(i)) * rowh)) + @as(f64, @floatFromInt(rowh)) / 2;
        const z = num(zar.get("model_total").?);
        const s = num(ref.get("model_total").?);
        const zs = num(r.get("zarbor_score").?);
        const ss = num(r.get("ref_score").?);
        const ratio = s / z;
        const better = if (std.mem.eql(u8, metric, "auc")) zs > ss else zs < ss;
        const gap = @abs(zs - ss) / @abs(ss);
        const rh: f64 = @floatFromInt(rowh);

        try body.print("<g class=\"row\" data-i=\"{d}\">", .{i});
        try body.print("<rect class=\"hit\" x=\"0\" y=\"{f}\" width=\"{d}\" height=\"{d}\"/>", .{ dec(0, y - rh / 2), fig_w, rowh });
        try body.print("<text class=\"rl\" x=\"{d}\" y=\"{f}\">{s}</text>", .{ left - 18, dec(0, y - 3), key });
        try body.print("<text class=\"rs\" x=\"{d}\" y=\"{f}\">vs {s}</text>", .{ left - 18, dec(0, y + 13), reference });
        try body.print("<line class=\"conn\" x1=\"{f}\" y1=\"{f}\" x2=\"{f}\" y2=\"{f}\"/>", .{ dec(1, tx(@min(z, s))), dec(0, y), dec(1, tx(@max(z, s))), dec(0, y) });
        try body.print("<circle class=\"dot ref\" cx=\"{f}\" cy=\"{f}\" r=\"6\"/>", .{ dec(1, tx(s)), dec(0, y) });
        try body.print("<circle class=\"dot zar\" cx=\"{f}\" cy=\"{f}\" r=\"6\"/>", .{ dec(1, tx(z)), dec(0, y) });
        const verdict = if (ratio >= 1) try std.fmt.bufPrint(&b1, "{f}× faster", .{dec(2, ratio)}) else try std.fmt.bufPrint(&b1, "{f}× slower", .{dec(2, 1 / ratio)});
        // The ratio label goes past the slower end unless it would run into the gap
        // panel; then left of the faster end. Width estimated per character, as in
        // make_figure.py (`×` is one character there).
        const est_w = @as(f64, @floatFromInt(try std.unicode.utf8CountCodepoints(verdict))) * 7.2;
        var lx = tx(@max(z, s)) + 14;
        var anchor: []const u8 = "start";
        if (lx + est_w > @as(f64, @floatFromInt(gx0 - 24))) {
            lx = tx(@min(z, s)) - 14;
            anchor = "end";
        }
        try body.print("<text class=\"ratio {s}\" x=\"{f}\" y=\"{f}\" text-anchor=\"{s}\">{s}</text>", .{ if (ratio >= 1) "win" else "loss", dec(1, lx), dec(0, y + 4), anchor, verdict });
        try body.print("<line class=\"gline\" x1=\"{d}\" y1=\"{f}\" x2=\"{f}\" y2=\"{f}\"/>", .{ gx0, dec(0, y), dec(1, gx(gap)), dec(0, y) });
        try body.print("<circle class=\"gdot {s}\" cx=\"{f}\" cy=\"{f}\" r=\"5\"/>", .{ if (better) "zar" else "ref", dec(1, gx(gap)), dec(0, y) });
        // make_figure.py's `.rstrip("0")` runs on a string ending in `%`, so it strips
        // nothing; kept that way so the page is unchanged.
        const pct = if (gap >= 1e-6) try std.fmt.bufPrint(&b2, "{f}%", .{dec(4, gap * 100)}) else blk: {
            const e = try fmtE1(&b3, gap * 100);
            break :blk try std.fmt.bufPrint(&b2, "{s}%", .{e});
        };
        const who = if (better) "zarbor ahead" else "reference ahead";
        try body.print("<text class=\"gl\" x=\"{f}\" y=\"{f}\">{s}</text>", .{ dec(1, gx(gap) + 12), dec(0, y + 1), pct });
        try body.print("<text class=\"gw\" x=\"{f}\" y=\"{f}\">{s}</text>", .{ dec(1, gx(gap) + 12), dec(0, y + 13), who });
        try body.writeAll("</g>");

        try tbody.print("<tr><th>{s}</th><td>{s}</td><td>{s}</td>", .{ key, reference, try fmtScore(&b3, zs, metric) });
        try tbody.print("<td>{s}</td><td>", .{try fmtScore(&b3, ss, metric)});
        for (metric) |c| try tbody.writeByte(std.ascii.toUpper(c));
        try tbody.print("</td><td>{s}</td><td>{s}</td>", .{ pct, if (better) "zarbor" else "reference" });
        try tbody.print("<td>{s}</td><td>{s}</td><td>{s}</td></tr>", .{ try fmtMs(&b3, z), try fmtMs(&b4, s), verdict });

        if (i > 0) try data.writeAll(", ");
        const fields = [_]struct { []const u8, std.json.Value }{
            .{ "k", r.get("key").? },            .{ "r", r.get("reference").? },  .{ "zs", r.get("zarbor_score").? },
            .{ "ss", r.get("ref_score").? },     .{ "m", r.get("metric").? },     .{ "zp", zar.get("prepare").? },
            .{ "zf", zar.get("fit").? },         .{ "zd", zar.get("predict").? }, .{ "rp", ref.get("prepare").? },
            .{ "rf", ref.get("fit").? },         .{ "rd", ref.get("predict").? }, .{ "zt", zar.get("model_total").? },
            .{ "rt", ref.get("model_total").? },
        };
        try data.writeAll("{");
        for (fields, 0..) |f, fi| {
            if (fi > 0) try data.writeAll(", ");
            try data.print("\"{s}\": ", .{f[0]});
            try pyJson(data, f[1]);
        }
        try data.writeAll("}");
    }
    try data.writeAll("]");

    const n_rows: i64 = @intCast(rows.len);
    for ([_]f64{ 200, 500, 1000, 2000, 5000 }) |t| {
        try axis.print("<line class=\"grid\" x1=\"{f}\" y1=\"{d}\" x2=\"{f}\" y2=\"{d}\"/>", .{ dec(1, tx(t)), top - 16, dec(1, tx(t)), top + rowh * n_rows });
        try axis.print("<text class=\"tick\" x=\"{f}\" y=\"{d}\">{s}</text>", .{ dec(1, tx(t)), top - 24, try fmtMs(&b1, t) });
    }
    for ([_]struct { f64, []const u8 }{ .{ 1e-6, "0.0001%" }, .{ 1e-4, "0.01%" }, .{ 1e-2, "1%" } }) |g| {
        try axis.print("<line class=\"grid\" x1=\"{f}\" y1=\"{d}\" x2=\"{f}\" y2=\"{d}\"/>", .{ dec(1, gx(g[0])), top - 16, dec(1, gx(g[0])), top + rowh * n_rows });
        try axis.print("<text class=\"tick\" x=\"{f}\" y=\"{d}\">{s}</text>", .{ dec(1, gx(g[0])), top - 24, g[1] });
    }

    var page: std.Io.Writer.Allocating = .init(gpa);
    try writePage(&page.writer, .{
        .height = top + rowh * n_rows + 26,
        .axis = axis_a.written(),
        .body = body_a.written(),
        .tbody = tbody_a.written(),
        .data = data_a.written(),
    });
    try Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out_path, .data = page.written() });
    print("wrote {s}\n", .{out_path});
}

// The page, generated from make_figure.py's HTML template: literal text copied verbatim,
// placeholders in the same places.
fn writePage(w: *std.Io.Writer, x: Page) !void {
    try w.writeAll(
        \\<!doctype html><html lang="en"><head><meta charset="utf-8">
        \\<title>zarbor vs standard implementations</title>
        \\<body data-palette="#2a78d6,#eb6834">
        \\<style>
        \\  .viz-root { color-scheme: light;
        \\    --surface-1:#fcfcfb; --text-primary:#0b0b0b; --text-secondary:#52514e;
        \\    --text-muted:#7a7873; --grid:#e8e7e3;
        \\    --series-1:#2a78d6; --series-2:#eb6834; --neutral:#b9b7b1; }
        \\  @media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) .viz-root {
        \\    color-scheme: dark;
        \\    --surface-1:#1a1a19; --text-primary:#ffffff; --text-secondary:#c3c2b7;
        \\    --text-muted:#8e8d84; --grid:#2e2e2c;
        \\    --series-1:#3987e5; --series-2:#d95926; --neutral:#5a5954; } }
        \\  :root[data-theme="dark"] .viz-root { color-scheme: dark;
        \\    --surface-1:#1a1a19; --text-primary:#ffffff; --text-secondary:#c3c2b7;
        \\    --text-muted:#8e8d84; --grid:#2e2e2c;
        \\    --series-1:#3987e5; --series-2:#d95926; --neutral:#5a5954; }
        \\  body { margin:0; background:var(--surface-1,#fcfcfb); }
        \\  .viz-root { background:var(--surface-1); color:var(--text-primary);
        \\    font:14px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
        \\    padding:28px 32px 36px; max-width:
    );
    try w.print("{d}", .{fig_w + 64});
    try w.writeAll(
        \\px; margin:0 auto; }
        \\  h1 { font-size:19px; margin:0 0 4px; letter-spacing:-.01em; }
        \\  .sub { color:var(--text-secondary); margin:0 0 4px; font-size:13px; }
        \\  .meta { color:var(--text-muted); margin:0 0 18px; font-size:12px; }
        \\  .legend { display:flex; gap:20px; align-items:center; margin:0 0 6px; font-size:13px;
        \\    color:var(--text-secondary); }
        \\  .legend i { width:11px; height:11px; border-radius:50%; display:inline-block;
        \\    margin-right:7px; vertical-align:-1px; }
        \\  svg { display:block; overflow:visible; }
        \\  .rl { fill:var(--text-primary); font-size:13.5px; font-weight:600; text-anchor:end; }
        \\  .rs { fill:var(--text-muted); font-size:11.5px; text-anchor:end; }
        \\  .conn { stroke:var(--neutral); stroke-width:2; stroke-linecap:round; }
        \\  .dot { stroke:var(--surface-1); stroke-width:2; }
        \\  .zar { fill:var(--series-1); } .ref { fill:var(--series-2); }
        \\  .ratio { font-size:12.5px; font-weight:600; fill:var(--text-secondary); }
        \\  .ratio.loss { fill:var(--text-muted); }
        \\  .gline { stroke:var(--grid); stroke-width:2; stroke-linecap:round; }
        \\  .gdot { stroke:var(--surface-1); stroke-width:2; }
        \\  .gdot.zar { fill:var(--series-1); } .gdot.ref { fill:var(--series-2); }
        \\  .gl { font-size:11.5px; fill:var(--text-secondary); }
        \\  .gw { font-size:10.5px; fill:var(--text-muted); }
        \\  .grid { stroke:var(--grid); stroke-width:1; }
        \\  .tick { fill:var(--text-muted); font-size:11px; text-anchor:middle; }
        \\  .ptitle { fill:var(--text-secondary); font-size:11.5px; font-weight:600;
        \\    letter-spacing:.04em; text-transform:uppercase; }
        \\  .hit { fill:transparent; }
        \\  .row:hover .hit { fill:var(--grid); opacity:.45; }
        \\  table { border-collapse:collapse; margin-top:26px; font-size:12.5px; width:100%; }
        \\  th,td { text-align:right; padding:6px 10px; border-bottom:1px solid var(--grid);
        \\    color:var(--text-secondary); white-space:nowrap; }
        \\  thead th { color:var(--text-muted); font-weight:600; font-size:11px;
        \\    text-transform:uppercase; letter-spacing:.04em; }
        \\  tbody th { text-align:left; color:var(--text-primary); font-weight:600; }
        \\  td:nth-child(2) { text-align:left; }
        \\  .foot { color:var(--text-muted); font-size:11.5px; margin:14px 0 0; max-width:1000px; line-height:1.55; }
        \\  .foot b { color:var(--text-secondary); font-weight:600; }
        \\  details { margin-top:8px; } summary { cursor:pointer; color:var(--text-muted);
        \\    font-size:12px; }
        \\  #tip { position:fixed; pointer-events:none; opacity:0; transition:opacity .08s;
        \\    background:var(--surface-1); color:var(--text-primary); border:1px solid var(--grid);
        \\    border-radius:7px; padding:9px 11px; font-size:12px; box-shadow:0 4px 14px #0002;
        \\    z-index:9; line-height:1.5; }
        \\  #tip b { font-weight:600; } #tip .k { color:var(--text-muted); }
        \\</style>
        \\<div class="viz-root">
        \\  <h1>zarbor against the standard implementation of each model</h1>
        \\  <p class="sub">Kaggle Playground S6E9 &mdash; 534,932 train / 133,733 validation rows, 13 features.
        \\     Time is <b>prepare + fit + predict</b>, so binning is charged to whichever side does it.</p>
        \\  <p class="meta">Ryzen 7 9800X3D, 16 threads &middot; median of 5 &middot;
        \\     XGBoost 3.4.1, LightGBM 4.7.0, scikit-learn 1.9.1 &middot; CSV parsing measured separately and charged to neither</p>
        \\  <div class="legend">
        \\    <span><i style="background:var(--series-1)"></i>zarbor</span>
        \\    <span><i style="background:var(--series-2)"></i>standard implementation</span>
        \\  </div>
        \\  <svg viewBox="0 0 
    );
    try w.print("{d}", .{fig_w});
    try w.writeAll(
        \\ 
    );
    try w.print("{d}", .{x.height});
    try w.writeAll(
        \\" width="
    );
    try w.print("{d}", .{fig_w});
    try w.writeAll(
        \\" height="
    );
    try w.print("{d}", .{x.height});
    try w.writeAll(
        \\" role="img"
        \\       aria-label="Dumbbell chart comparing zarbor to XGBoost, LightGBM and scikit-learn on model time, with an accuracy gap panel">
        \\    <text class="ptitle" x="
    );
    try w.print("{d}", .{left});
    try w.writeAll(
        \\" y="
    );
    try w.print("{d}", .{top - 48});
    try w.writeAll(
        \\">Model time (log scale)</text>
        \\    <text class="ptitle" x="
    );
    try w.print("{d}", .{gx0});
    try w.writeAll(
        \\" y="
    );
    try w.print("{d}", .{top - 48});
    try w.writeAll(
        \\">Accuracy gap &mdash; and which side is ahead</text>
        \\    
    );
    try w.writeAll(x.axis);
    try w.writeAll(
        \\
        \\    
    );
    try w.writeAll(x.body);
    try w.writeAll(
        \\
        \\  </svg>
        \\  <p class="foot">Every row is the same dataset and machine, so the time column is comparable down the page.
        \\     <b>GBDT&nbsp;+&nbsp;GOSS is the one row where the gap exceeds its own noise floor</b> (1.42&times; the
        \\     reference's max_bin envelope) &mdash; zarbor scores higher, but that is an unexplained disagreement,
        \\     not a result. Random forest and linear regression compare deliberately different constructions,
        \\     so their gaps are design differences rather than defects. See docs/arena.md.</p>
        \\  <details open><summary>Table view &mdash; every number in the figure</summary>
        \\  <table><thead><tr><th>Model</th><th>Reference</th><th>zarbor</th><th>reference</th>
        \\  <th>Metric</th><th>Rel. gap</th><th>Closer to truth</th>
        \\  <th>zarbor time</th><th>ref time</th><th>Speed</th></tr></thead>
        \\  <tbody>
    );
    try w.writeAll(x.tbody);
    try w.writeAll(
        \\</tbody></table></details>
        \\</div>
        \\<div id="tip"></div>
        \\<script>
        \\const DATA = 
    );
    try w.writeAll(x.data);
    try w.writeAll(
        \\;
        \\const tip=document.getElementById('tip');
        \\const ms=v=>v>=1000?(v/1000).toFixed(2)+' s':Math.round(v)+' ms';
        \\const sc=(v,m)=>m==='rmse'&&v>100?v.toLocaleString(undefined,{maximumFractionDigits:2}):v.toFixed(6);
        \\document.querySelectorAll('.row').forEach(g=>{
        \\  g.addEventListener('pointermove',e=>{
        \\    const d=DATA[+g.dataset.i];
        \\    tip.innerHTML=`<b>${d.k}</b><br><span class="k">metric</span> ${d.m.toUpperCase()} &mdash; `+
        \\      `zarbor <b>${sc(d.zs,d.m)}</b> vs ${d.r} <b>${sc(d.ss,d.m)}</b>`+
        \\      `<br><span class="k">zarbor</span> prepare ${ms(d.zp)} + fit ${ms(d.zf)} + predict ${ms(d.zd)} = <b>${ms(d.zt)}</b>`+
        \\      `<br><span class="k">${d.r}</span> prepare ${ms(d.rp)} + fit ${ms(d.rf)} + predict ${ms(d.rd)} = <b>${ms(d.rt)}</b>`;
        \\    tip.style.opacity=1;
        \\    const r=tip.getBoundingClientRect();
        \\    tip.style.left=Math.min(e.clientX+16,innerWidth-r.width-12)+'px';
        \\    tip.style.top=Math.min(e.clientY+16,innerHeight-r.height-12)+'px';
        \\  });
        \\  g.addEventListener('pointerleave',()=>tip.style.opacity=0);
        \\});
        \\</script>
        \\</body></html>
    );
}

// ---------------------------------------------------------------- grid

const fixed_grid = [_][]const u8{ "--n_rounds=500", "--learning_rate=0.05", "--max_depth=6", "--min_child_samples=20", "--lambda=1.0", "--max_bin=256" };
const Dataset = struct { name: []const u8, label: []const u8, objective: []const u8, extra: []const []const u8 = &.{} };
const grid_sets = [_]Dataset{
    .{ .name = "california", .label = "y", .objective = "squared_error" },
    .{ .name = "adult", .label = "y", .objective = "logistic" },
    .{ .name = "bank", .label = "y", .objective = "logistic" },
    .{ .name = "ames", .label = "y", .objective = "squared_error" },
    .{ .name = "housing", .label = "AffordabilityPercentageTrue", .objective = "squared_error", .extra = &.{ "--group-col=Zip", "--drop=City", "--drop=Metro", "--drop=Zip3" } },
};

/// The pre-registered grid of docs/PROTOCOL.md (was run.py): 5-fold cv per dataset and
/// seed, one TSV line each: tag, dataset, seed, seconds, the run's output on one line.
/// `$ONLY=a,b` limits datasets; `$SEEDS` defaults to 0,1,2.
fn grid(init: std.process.Init, gpa: Allocator, w: *std.Io.Writer, args: []const []const u8) !void {
    if (args.len < 2) {
        print("usage: tools grid <zarbor> <tag> [extra flags...]\n", .{});
        return error.BadArgs;
    }
    const binary = args[0];
    const tag = args[1];
    const only = init.environ_map.get("ONLY");
    const data = try dataDir(init, gpa);
    for (grid_sets) |ds| {
        if (only) |o| {
            var it = std.mem.splitScalar(u8, o, ',');
            const hit = while (it.next()) |name| {
                if (std.mem.eql(u8, name, ds.name)) break true;
            } else false;
            if (!hit) continue;
        }
        var seeds = std.mem.splitScalar(u8, envOr(init, "SEEDS", "0,1,2"), ',');
        while (seeds.next()) |seed_s| {
            const seed = try std.fmt.parseInt(i64, seed_s, 10);
            var argv: std.ArrayList([]const u8) = .empty;
            try argv.appendSlice(gpa, &.{ binary, "cv", try std.fmt.allocPrint(gpa, "{s}/{s}.csv", .{ data, ds.name }), try std.fmt.allocPrint(gpa, "--label={s}", .{ds.label}), "--folds=5", try std.fmt.allocPrint(gpa, "--fold-seed={d}", .{seed}), try std.fmt.allocPrint(gpa, "--objective={s}", .{ds.objective}) });
            try argv.appendSlice(gpa, &fixed_grid);
            try argv.appendSlice(gpa, ds.extra);
            try argv.appendSlice(gpa, args[2..]);
            try argv.append(gpa, "--quiet=1");
            const t0 = nanoTime();
            const r = try run(init, gpa, argv.items);
            const dt = @as(f64, @floatFromInt(nanoTime() - t0)) / 1e9;
            const line = try std.mem.replaceOwned(u8, gpa, std.mem.trim(u8, r.text, " \t\r\n\x0b\x0c"), "\n", " | ");
            var buf: [32]u8 = undefined;
            try w.print("{s}\t{s}\t{d}\t", .{ tag, ds.name, seed });
            try padLeft(w, 8, try std.fmt.bufPrint(&buf, "{f}", .{dec(2, dt)}));
            try w.print("\t{s}\n", .{line});
            try w.flush();
        }
    }
}

// ---------------------------------------------------------------- controls

/// Controls C3-C5 of docs/PROTOCOL.md (was controls.sh; C1 and C2 are not run here, and
/// C3 here only checks that two predictions from the same saved file agree; the exact
/// in-process vs reloaded comparison is in model_test.zig's round-trip tests). NEW is this checkout's
/// zig-out/bin/zarbor; OLD is `$ZARBOR_OLD` (required); the data is `adult.csv` in the
/// data directory, label `y`. Exit status 1 when any control fails.
fn controls(init: std.process.Init, gpa: Allocator, w: *std.Io.Writer) !void {
    const new = try std.fs.path.join(gpa, &.{ paths.root, "zig-out", "bin", "zarbor" });
    const old = init.environ_map.get("ZARBOR_OLD") orelse {
        print("error: controls needs $ZARBOR_OLD, a zarbor built from the previous version\n", .{});
        return error.MissingOld;
    };
    const t = try scratchDir(init.io, gpa);
    defer Io.Dir.cwd().deleteTree(init.io, t) catch |e| print("note: could not remove {s}: {s}\n", .{ t, @errorName(e) });
    var c: Ctl = .{ .init = init, .gpa = gpa, .w = w, .t = t, .data = try std.fmt.allocPrint(gpa, "{s}/adult.csv", .{try dataDir(init, gpa)}) };

    try w.writeAll("--- C3: save -> load -> predict twice from the saved file gives identical predictions\n");
    try c.train(new, "m.zm", &.{"--cat_split=optimal"});
    try c.predict(new, "m.zm", "p1.csv");
    try c.predict(new, "m.zm", "p2.csv");
    try c.verdict(c.same("p1.csv", "p2.csv"), "  reload stable: PASS\n", "  reload stable: FAIL\n");
    const info = try run(init, gpa, &.{ new, "info", try c.flag("--model=", "m.zm") });
    try w.print("  info on the saved cat-split model exited {d}\n", .{info.code});

    try w.writeAll("--- C4: bit-identical across --n_threads, with F1 on\n");
    try c.threads(new, "--cat_split=optimal");

    try w.writeAll("--- C5: a model written by OLD loads and scores identically under NEW\n");
    try c.train(old, "old.zm", &.{});
    try c.predict(old, "old.zm", "old_pred.csv");
    try c.predict(new, "old.zm", "new_pred.csv");
    try c.verdict(c.same("old_pred.csv", "new_pred.csv"), "  OLD-written model, OLD vs NEW predictions: PASS\n", "  FAIL\n");

    try w.writeAll("--- C3c: linear leaves: two predictions from the saved file agree\n");
    try c.train(new, "lin.zm", &.{"--linear_leaves=1"});
    try c.predict(new, "lin.zm", "l1.csv");
    try c.predict(new, "lin.zm", "l2.csv");
    try c.verdict(c.same("l1.csv", "l2.csv"), "  PASS\n", "  FAIL\n");

    try w.writeAll("--- C4b: bit-identical across --n_threads, with F2 on\n");
    try c.threads(new, "--linear_leaves=1");

    try w.writeAll("--- C2b: F2 changes nothing when no split is on a numeric feature\n");
    try w.writeAll("  (not applicable: every dataset here has numeric columns; C2 itself, for F1, is not run here)\n");

    try w.writeAll("--- C3b: round trip of a NEW model through OLD must be refused, not misread\n");
    const refuse = try run(init, gpa, &.{ old, "predict", c.data, try c.flag("--model=", "m.zm"), try c.flag("--out=", "x.csv") });
    const lower = try std.ascii.allocLowerString(gpa, refuse.text);
    if (std.mem.find(u8, lower, "unsupported") != null or std.mem.find(u8, lower, "version") != null) {
        try w.writeAll("  OLD rejects a NEW-format file: PASS\n");
    } else {
        try w.print("  OLD did not reject a NEW-format file: FAIL\n{s}", .{refuse.text});
        c.fail = true;
    }
    try w.flush();
    if (c.fail) std.process.exit(1);
}

const Ctl = struct {
    init: std.process.Init,
    gpa: Allocator,
    w: *std.Io.Writer,
    t: []const u8,
    data: []const u8,
    fail: bool = false,
    const fixed = [_][]const u8{ "--n_rounds=60", "--learning_rate=0.1", "--max_depth=6", "--objective=logistic", "--label=y" };

    fn path(c: *Ctl, name: []const u8) ![]const u8 {
        return std.fs.path.join(c.gpa, &.{ c.t, name });
    }
    fn flag(c: *Ctl, comptime prefix: []const u8, name: []const u8) ![]const u8 {
        return std.fmt.allocPrint(c.gpa, prefix ++ "{s}", .{try c.path(name)});
    }
    fn train(c: *Ctl, bin: []const u8, model: []const u8, extra: []const []const u8) !void {
        var argv: std.ArrayList([]const u8) = .empty;
        try argv.appendSlice(c.gpa, &.{ bin, c.data });
        try argv.appendSlice(c.gpa, &fixed);
        try argv.appendSlice(c.gpa, extra);
        try argv.appendSlice(c.gpa, &.{ try c.flag("--save=", model), "--valid-frac=0.2" });
        _ = try run(c.init, c.gpa, argv.items);
    }
    fn predict(c: *Ctl, bin: []const u8, model: []const u8, out: []const u8) !void {
        _ = try run(c.init, c.gpa, &.{ bin, "predict", c.data, try c.flag("--model=", model), try c.flag("--out=", out) });
    }
    fn same(c: *Ctl, a: []const u8, b: []const u8) bool {
        return sameFile(c.init.io, c.gpa, c.path(a) catch return false, c.path(b) catch return false);
    }
    fn verdict(c: *Ctl, ok: bool, pass: []const u8, failed: []const u8) !void {
        try c.w.writeAll(if (ok) pass else failed);
        if (!ok) c.fail = true;
    }
    /// cv at 1, 4 and 16 threads must print the same thing once timings are removed.
    fn threads(c: *Ctl, bin: []const u8, feature: []const u8) !void {
        var outs: [3][]const u8 = undefined;
        for ([_][]const u8{ "1", "4", "16" }, 0..) |th, k| {
            var argv: std.ArrayList([]const u8) = .empty;
            try argv.appendSlice(c.gpa, &.{ bin, "cv", c.data });
            try argv.appendSlice(c.gpa, &fixed);
            try argv.appendSlice(c.gpa, &.{ "--folds=3", feature, try std.fmt.allocPrint(c.gpa, "--n_threads={s}", .{th}), "--quiet=1" });
            outs[k] = try stripMs(c.gpa, (try run(c.init, c.gpa, argv.items)).text);
        }
        if (std.mem.eql(u8, outs[0], outs[1]) and std.mem.eql(u8, outs[0], outs[2])) {
            try c.w.print("  1 == 4 == 16 threads: PASS  ({s})\n", .{std.mem.trimEnd(u8, outs[0], "\n")});
        } else {
            try c.w.writeAll("  FAIL\n");
            for (outs) |o| try c.w.writeAll(o);
            c.fail = true;
        }
    }
};

/// controls.sh's `sed 's/[0-9]* ms//'`: drop every `<digits> ms` (timings vary by run).
fn stripMs(gpa: Allocator, s: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < s.len) {
        var j = i;
        while (j < s.len and std.ascii.isDigit(s[j])) j += 1;
        if (std.mem.startsWith(u8, s[j..], " ms")) {
            i = j + 3;
            continue;
        }
        if (j > i) {
            try out.appendSlice(gpa, s[i..j]);
            i = j;
            continue;
        }
        try out.append(gpa, s[i]);
        i += 1;
    }
    return out.toOwnedSlice(gpa);
}

// ---------------------------------------------------------------- solver-stress

const adam_epochs = 20000;
const agree = 1e-3;

const Solved = struct { auc: ?f64, state: []const u8, ratio: ?f64 };

/// Do zarbor's two linear solvers meet? (was arena/solver_stress.py) L-BFGS/OWL-QN and
/// Adam minimise the same convex objective, so where both claim success they must agree;
/// a solver that is wrong and reports success is the trap this looks for. Data:
/// ev_small.csv / ev_small_valid.csv in bench/arena (label Will_Buy_EV, positive "Yes").
fn solverStress(init: std.process.Init, gpa: Allocator, w: *std.Io.Writer) !void {
    const arena_dir = try std.fs.path.join(gpa, &.{ paths.root, "bench", "arena" });
    const train = try std.fs.path.join(gpa, &.{ arena_dir, "ev_small.csv" });
    const valid = try std.fs.path.join(gpa, &.{ arena_dir, "ev_small_valid.csv" });
    const t = try scratchDir(init.io, gpa);
    defer Io.Dir.cwd().deleteTree(init.io, t) catch |e| print("note: could not remove {s}: {s}\n", .{ t, @errorName(e) });

    var pool = try zarbor.pool.Pool.init(gpa, 0);
    defer pool.deinit();
    var vf = try zarbor.csv.readCsv(gpa, init.io, pool, valid, 1 << 30);
    defer vf.deinit();
    const lc = vf.columnIndex("Will_Buy_EV") orelse return error.NoLabel;
    const yes_id: ?usize = for (vf.levels[lc], 0..) |l, k| {
        if (std.mem.eql(u8, l, "Yes")) break k;
    } else null;
    const y = try gpa.alloc(f32, vf.n_rows);
    for (y, vf.values[lc]) |*o, v| o.* = if (yes_id != null and !std.math.isNan(v) and @as(usize, @intFromFloat(v)) == yes_id.?) 1 else 0;

    try w.print("{s:>5} {s:>7} {s:>7} | {s:>10} {s:>8} {s:>9} | {s:>10} {s:>8} {s:>9} | verdict\n", .{ "std", "alpha", "lambda", "lbfgs auc", "state", "g_ratio", "adam auc", "state", "g_ratio" });
    try w.splatByteAll('-', 108);
    try w.writeAll("\n");
    var bad: usize = 0;
    var n: usize = 0;
    for ([_]bool{ true, false }) |std_on| for ([_]f64{ 0.0, 0.001, 0.1 }) |alpha| for ([_]f64{ 0.0, 1.0, 100.0 }) |lam| {
        n += 1;
        const l = try solve(init, gpa, pool, train, valid, t, y, "lbfgs", std_on, alpha, lam);
        const a = try solve(init, gpa, pool, train, valid, t, y, "adam", std_on, alpha, lam);
        var vbuf: [64]u8 = undefined;
        var verdict: []const u8 = undefined;
        if (l.auc == null or a.auc == null) {
            verdict = "RUN FAILED";
        } else if (std.mem.eql(u8, l.state, "ok") and std.mem.eql(u8, a.state, "ok")) {
            const d = @abs(l.auc.? - a.auc.?);
            verdict = if (d < agree) "meet" else try std.fmt.bufPrint(&vbuf, "DISAGREE {f}", .{dec(5, d)});
            if (d >= agree) bad += 1;
        } else if (std.mem.eql(u8, l.state, "ok")) {
            verdict = "adam flagged itself";
        } else if (std.mem.eql(u8, a.state, "STALLED")) {
            verdict = "both flagged";
        } else {
            verdict = try std.fmt.bufPrint(&vbuf, "lbfgs {s}, adam {s}{s}", .{ l.state, a.state, if (std.mem.eql(u8, l.state, "STALLED") and std.mem.eql(u8, a.state, "ok")) "  <-- SILENT" else "" });
            if (std.mem.eql(u8, l.state, "STALLED") and std.mem.eql(u8, a.state, "ok")) bad += 1;
        }
        // One buffer per printed value: they are all formatted before the print.
        var b_alpha: [32]u8 = undefined;
        var b_lam: [32]u8 = undefined;
        var b_lauc: [32]u8 = undefined;
        var b_lratio: [32]u8 = undefined;
        var b_aauc: [32]u8 = undefined;
        var b_aratio: [32]u8 = undefined;
        try w.print("{s:>5} {s:>7} {s:>7} | {s:>10} {s:>8} {s:>9} | {s:>10} {s:>8} {s:>9} | {s}\n", .{
            if (std_on) "True" else "False", try pyNum(&b_alpha, alpha), try pyNum(&b_lam, lam),
            try optF6(&b_lauc, l.auc),       l.state,                    try optE2(&b_lratio, l.ratio),
            try optF6(&b_aauc, a.auc),       a.state,                    try optE2(&b_aratio, a.ratio),
            verdict,
        });
    };
    try w.splatByteAll('-', 108);
    try w.print("\n{d} configurations, {d} where the two do not meet with both claiming success\n", .{ n, bad });
    try w.flush();
    if (bad > 0) std.process.exit(1);
}

fn pyNum(buf: []u8, v: f64) ![]const u8 {
    return if (@floor(v) == v) std.fmt.bufPrint(buf, "{f}", .{dec(1, v)}) else std.fmt.bufPrint(buf, "{d}", .{v});
}

fn optF6(buf: []u8, v: ?f64) ![]const u8 {
    return if (v) |x| std.fmt.bufPrint(buf, "{f}", .{dec(6, x)}) else "  --  ";
}

fn optE2(buf: []u8, v: ?f64) ![]const u8 {
    const x = v orelse return "  --  ";
    var raw: [48]u8 = undefined;
    const s = try std.fmt.bufPrint(&raw, "{e:.2}", .{x});
    const e = std.mem.findScalar(u8, s, 'e').?;
    const exp = try std.fmt.parseInt(i32, s[e + 1 ..], 10);
    return std.fmt.bufPrint(buf, "{s}e{s}{d:0>2}", .{ s[0..e], if (exp < 0) "-" else "+", @abs(exp) });
}

/// The two numbers of the first `<prefix><a> from <b>` in `text`, as a/b.
fn ratioAfter(text: []const u8, prefix: []const u8) ?f64 {
    const at = std.mem.find(u8, text, prefix) orelse return null;
    var it = std.mem.tokenizeAny(u8, text[at + prefix.len ..], " ,)");
    const a = std.fmt.parseFloat(f64, it.next() orelse return null) catch return null;
    if (!std.mem.eql(u8, it.next() orelse return null, "from")) return null;
    const b = std.fmt.parseFloat(f64, it.next() orelse return null) catch return null;
    return a / b;
}

fn solve(init: std.process.Init, gpa: Allocator, pool: *zarbor.pool.Pool, train: []const u8, valid: []const u8, dir: []const u8, y: []const f32, solver: []const u8, std_on: bool, alpha: f64, lam: f64) !Solved {
    const exe_path = try std.fs.path.join(gpa, &.{ paths.root, "zig-out", "bin", "zarbor" });
    const model = try std.fmt.allocPrint(gpa, "{s}/ss_{s}.zm", .{ dir, solver });
    const pred = try std.fmt.allocPrint(gpa, "{s}/ss.csv", .{dir});
    var argv: std.ArrayList([]const u8) = .empty;
    try argv.appendSlice(gpa, &.{ exe_path, train, "--label=Will_Buy_EV", "--valid-frac=0", "--algo=linear", "--verbose_eval=0", "--n_threads=16" });
    try argv.append(gpa, try std.fmt.allocPrint(gpa, "--lin_solver={s}", .{solver}));
    try argv.append(gpa, try std.fmt.allocPrint(gpa, "--lin_standardize={s}", .{if (std_on) "true" else "false"}));
    var b: [32]u8 = undefined;
    try argv.append(gpa, try std.fmt.allocPrint(gpa, "--alpha={s}", .{try pyNum(&b, alpha)}));
    try argv.append(gpa, try std.fmt.allocPrint(gpa, "--lambda={s}", .{try pyNum(&b, lam)}));
    try argv.append(gpa, try std.fmt.allocPrint(gpa, "--save={s}", .{model}));
    if (std.mem.eql(u8, solver, "adam")) try argv.append(gpa, try std.fmt.allocPrint(gpa, "--lin_epochs={d}", .{adam_epochs}));
    const r = try run(init, gpa, argv.items);
    if (r.code != 0) return .{ .auc = null, .state = "ERROR", .ratio = null };
    const stalled = std.mem.find(u8, r.text, "STALLED") != null;
    const ratio = ratioAfter(r.text, "|g|max ") orelse ratioAfter(r.text, "large (");
    const p = try run(init, gpa, &.{ exe_path, "predict", valid, try std.fmt.allocPrint(gpa, "--model={s}", .{model}), try std.fmt.allocPrint(gpa, "--out={s}", .{pred}), "--n_threads=16" });
    if (p.code != 0) return .{ .auc = null, .state = "PREDFAIL", .ratio = ratio };
    var pf = try zarbor.csv.readCsv(gpa, init.io, pool, pred, 1 << 30);
    defer pf.deinit();
    const scores = pf.values[pf.names.len - 1];
    return .{ .auc = try zarbor.metric.auc(gpa, scores, y), .state = if (stalled) "STALLED" else "ok", .ratio = ratio };
}
