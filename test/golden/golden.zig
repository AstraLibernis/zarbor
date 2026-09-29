// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! test/golden/golden.zig — end-to-end regression fence for refactors.
//!
//!   zig build golden               run every case, compare with expected/
//!   zig build golden -- --update   rewrite expected/ (a deliberate re-baseline)
//!
//! Each case runs the CLI on titanic3.csv inside a fresh scratch dir and records exit
//! code, stdout, stderr and a SHA-256 of every file it wrote. Timings and the build
//! mode are masked; everything else must match byte for byte.

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;

const data_file = "titanic3.csv";
const drops = [_][]const u8{ "--drop=boat", "--drop=body", "--drop=name", "--drop=ticket", "--drop=home.dest" };

const Case = struct {
    name: []const u8,
    args: []const []const u8,
    /// Pass the standard label/drop/thread flags after `args`.
    train_flags: bool = true,
    files: []const []const u8 = &.{},
};

// Order matters: predict/blend/info read models saved by earlier cases.
const cases = [_]Case{
    .{ .name = "train-gbdt", .args = &.{ data_file, "--label=survived", "--save=g.zm" }, .files = &.{"g.zm"} },
    .{ .name = "train-gbdt-lossguide", .args = &.{ data_file, "--label=survived", "--grow_policy=lossguide", "--max_leaves=16", "--sampling=goss", "--colsample_bytree=0.7", "--cat_split=optimal", "--n_rounds=150" } },
    .{ .name = "train-gbdt-linear-leaves", .args = &.{ data_file, "--label=survived", "--linear_leaves=true", "--n_rounds=80", "--subsample=0.8", "--seed=7" } },
    .{ .name = "train-gbdt-regression", .args = &.{ data_file, "--label=pclass", "--objective=squared_error", "--n_rounds=150", "--early_stopping_rounds=20" } },
    .{ .name = "error-missing-label", .args = &.{ data_file, "--label=fare", "--objective=squared_error" } },
    .{ .name = "train-string-label", .args = &.{ data_file, "--label=sex", "--pos-label=female", "--n_rounds=60", "--bin_policy=greedy" } },
    .{ .name = "train-forest", .args = &.{ data_file, "--label=survived", "--algo=random_forest", "--n_rounds=40", "--save=f.zm" }, .files = &.{"f.zm"} },
    .{ .name = "train-linear", .args = &.{ data_file, "--label=survived", "--algo=linear", "--save=l.zm" }, .files = &.{"l.zm"} },
    .{ .name = "train-linear-adam", .args = &.{ data_file, "--label=survived", "--algo=linear", "--lin_solver=adam", "--lin_epochs=100", "--alpha=0.01" } },
    .{ .name = "predict", .train_flags = false, .args = &.{ "predict", data_file, "--model=g.zm", "--label=survived", "--out=p.csv" }, .files = &.{"p.csv"} },
    .{ .name = "blend", .train_flags = false, .args = &.{ "blend", data_file, "--models=g.zm,f.zm,l.zm", "--weights=2,1,1", "--out=b.csv" }, .files = &.{"b.csv"} },
    .{ .name = "info-gbdt", .train_flags = false, .args = &.{ "info", "--model=g.zm" } },
    .{ .name = "info-linear", .train_flags = false, .args = &.{ "info", "--model=l.zm" } },
    .{ .name = "profile", .train_flags = false, .args = &.{ "profile", data_file } },
    .{ .name = "cv", .args = &.{ "cv", data_file, "--label=survived", "--folds=3", "--n_rounds=100", "--oof=oof.csv" }, .files = &.{"oof.csv"} },
    .{ .name = "cv-forest", .args = &.{ "cv", data_file, "--label=survived", "--algo=random_forest", "--folds=3", "--n_rounds=30" } },
    .{ .name = "tune-random", .args = &.{ "tune", data_file, "--label=survived", "--trials=6", "--folds=2", "--out=t.csv" }, .files = &.{"t.csv"} },
    .{ .name = "tune-bayes", .args = &.{ "tune", data_file, "--label=survived", "--search=bayes", "--trials=6", "--warmup=3", "--folds=2" } },
    .{ .name = "tune-bandit", .args = &.{ "tune", data_file, "--label=survived", "--search=bandit", "--trials=6", "--folds=2" } },
    .{ .name = "tune-grid-linear", .args = &.{ "tune", data_file, "--label=survived", "--algo=linear", "--search=grid", "--grid-steps=2", "--folds=2" } },
    .{ .name = "train-split-col", .args = &.{ data_file, "--label=survived", "--split-col=parch", "--n_rounds=40", "--profile=0" } },
    .{ .name = "train-valid-frac", .args = &.{ data_file, "--label=survived", "--valid-frac=0.35", "--split-seed=9", "--n_rounds=40", "--max-bytes=200000" } },
    .{ .name = "predict-ids", .train_flags = false, .args = &.{ "predict", data_file, "--model=f.zm", "--id-col=name", "--pred-col=p_survive", "--n_threads=2", "--out=pi.csv" }, .files = &.{"pi.csv"} },
    .{ .name = "predict-stdout", .train_flags = false, .args = &.{ "predict", data_file, "--model=l.zm" } },
    .{ .name = "profile-max-bytes", .train_flags = false, .args = &.{ "profile", data_file, "--max-bytes=150000" } },
    .{ .name = "cv-repeats-quiet", .args = &.{ "cv", data_file, "--label=survived", "--folds=3", "--repeats=3", "--quiet=1", "--n_rounds=40", "--oof=oofr.csv" }, .files = &.{"oofr.csv"} },
    .{ .name = "cv-group", .args = &.{ "cv", data_file, "--label=survived", "--folds=3", "--group-col=sibsp", "--fold-seed=5", "--algo=linear" } },
    .{ .name = "tune-params", .args = &.{ "tune", data_file, "--label=survived", "--search=random", "--trials=4", "--folds=2", "--seed=3", "--param=max_depth=2,4", "--param=lambda=0.5..8:log", "--param=n_rounds=20..60:int", "--confirm=2", "--confirm-seeds=2" } },
    .{ .name = "error-too-wide", .train_flags = false, .args = &.{ data_file, "--label=survived" } },
    .{ .name = "error-unknown-flag", .args = &.{ data_file, "--label=survived", "--no_such_flag=1" } },
    .{ .name = "error-flag-no-value", .args = &.{ data_file, "--label" } },
    .{ .name = "error-cv-unknown-flag", .args = &.{ "cv", data_file, "--label=survived", "--nope=2" } },
    .{ .name = "error-tune-no-input", .train_flags = false, .args = &.{"tune"} },
    .{ .name = "error-predict-no-model", .train_flags = false, .args = &.{ "predict", data_file } },
    .{ .name = "error-info-no-model", .train_flags = false, .args = &.{ "info", "g.zm" } },
    .{ .name = "error-profile-no-path", .train_flags = false, .args = &.{"profile"} },
    .{ .name = "error-no-input", .train_flags = false, .args = &.{} },
    .{ .name = "error-cat-levels", .args = &.{ data_file, "--label=survived", "--max_cat_levels=100" } },
    .{ .name = "error-cv-cat-levels", .args = &.{ "cv", data_file, "--label=survived", "--max_cat_levels=150", "--folds=3" } },
    .{ .name = "train-max-bin-64", .args = &.{ data_file, "--label=survived", "--max_bin=64", "--n_rounds=30" } },
    .{ .name = "error-bad-max-bin", .args = &.{ data_file, "--label=survived", "--max_bin=300" } },
    .{ .name = "error-bad-subsample", .args = &.{ data_file, "--label=survived", "--subsample=0" } },
    .{ .name = "error-unbounded-tree", .args = &.{ data_file, "--label=survived", "--max_depth=0" } },
    .{ .name = "error-unbounded-lossguide", .args = &.{ data_file, "--label=survived", "--grow_policy=lossguide", "--max_depth=0" } },
    .{ .name = "error-goss-bootstrap", .args = &.{ data_file, "--label=survived", "--sampling=goss", "--bootstrap=true" } },
    .{ .name = "error-boost-bootstrap", .args = &.{ data_file, "--label=survived", "--bootstrap=true" } },
    .{ .name = "error-forest-no-rounds", .args = &.{ data_file, "--label=survived", "--algo=random_forest", "--n_rounds=0" } },
    .{ .name = "error-linear-no-epochs", .args = &.{ data_file, "--label=survived", "--algo=linear", "--lin_epochs=0" } },
    .{ .name = "error-linear-neg-lambda", .args = &.{ "cv", data_file, "--label=survived", "--algo=linear", "--lambda=-1" } },
    .{ .name = "forest-explicit-overrides", .args = &.{ data_file, "--label=survived", "--algo=random_forest", "--n_rounds=20", "--max_depth=5", "--learning_rate=0.3", "--colsample_bynode=0.5", "--lambda=2" } },
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    if (args.len < 4) {
        std.log.err("usage: golden <zarbor> <golden-dir> <scratch-dir> [--update]", .{});
        return error.BadArgs;
    }
    const golden_dir, const scratch = .{ args[2], args[3] };
    const update = args.len > 4 and std.mem.eql(u8, args[4], "--update");

    const cwd = Io.Dir.cwd();
    // The build passes a cwd-relative path; cases run inside `scratch`.
    const exe = try cwd.realPathFileAlloc(io, args[1], gpa);
    const src_data = try std.fs.path.join(gpa, &.{ golden_dir, data_file });
    const dst_data = try std.fs.path.join(gpa, &.{ scratch, data_file });
    try cwd.copyFile(src_data, cwd, dst_data, io, .{});
    const expected_dir = try std.fs.path.join(gpa, &.{ golden_dir, "expected" });
    if (update) try cwd.createDirPath(io, expected_dir);

    var failed: usize = 0;
    for (cases) |case| {
        const got = try runCase(io, gpa, exe, scratch, case);
        const path = try std.fmt.allocPrint(gpa, "{s}/{s}.txt", .{ expected_dir, case.name });
        if (update) {
            try cwd.writeFile(io, .{ .sub_path = path, .data = got });
            continue;
        }
        const want = cwd.readFileAlloc(io, path, gpa, .unlimited) catch |e| {
            std.log.err("{s}: no baseline ({s}); run with --update", .{ case.name, @errorName(e) });
            failed += 1;
            continue;
        };
        if (!std.mem.eql(u8, want, got)) {
            failed += 1;
            reportDiff(case.name, want, got);
        }
    }
    if (update) {
        std.log.info("golden: wrote {d} baselines to {s}", .{ cases.len, expected_dir });
    } else if (failed != 0) {
        std.log.err("golden: {d}/{d} cases differ", .{ failed, cases.len });
        return error.GoldenMismatch;
    } else {
        std.log.info("golden: {d}/{d} cases match", .{ cases.len, cases.len });
    }
}

/// Run one case and render its record: command, exit, masked stdout/stderr, file hashes.
fn runCase(io: Io, gpa: Allocator, exe: []const u8, scratch: []const u8, case: Case) ![]const u8 {
    var argv: std.ArrayList([]const u8) = .empty;
    try argv.append(gpa, exe);
    try argv.appendSlice(gpa, case.args);
    if (case.train_flags) {
        try argv.appendSlice(gpa, &drops);
        try argv.append(gpa, "--n_threads=4");
    }
    const r = try std.process.run(gpa, io, .{
        .argv = argv.items,
        .cwd = .{ .path = scratch },
        .stdout_limit = .limited(16 << 20),
        .stderr_limit = .limited(16 << 20),
    });
    var rec: Io.Writer.Allocating = .init(gpa);
    const w = &rec.writer;
    try w.writeAll("$ zarbor");
    for (argv.items[1..]) |a| try w.print(" {s}", .{a});
    switch (r.term) {
        .exited => |c| try w.print("\nexit {d}\n", .{c}),
        else => try w.print("\nterm {t}\n", .{r.term}),
    }
    try w.writeAll("--- stdout\n");
    try mask(w, r.stdout);
    try w.writeAll("--- stderr\n");
    try mask(w, r.stderr);
    try w.writeAll("--- files\n");
    for (case.files) |f| {
        const p = try std.fs.path.join(gpa, &.{ scratch, f });
        const bytes = try Io.Dir.cwd().readFileAlloc(io, p, gpa, .unlimited);
        var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
        const masked = try maskMsColumn(gpa, bytes);
        std.crypto.hash.sha2.Sha256.hash(masked, &digest, .{});
        try w.print("{s}  {d} bytes  sha256 {s}\n", .{ f, masked.len, std.fmt.bytesToHex(digest, .lower) });
    }
    return rec.written();
}

/// Lines whose presence depends on timing: `build` (optimize mode), and the
/// `fit     N ms  (+M ms validating)` line, which prints only when validation
/// took at least 1 ms. The linear model's `fit     converged` line stays.
const dropped_prefixes = [_][]const u8{ "build " };

fn isTimingFitLine(line: []const u8) bool {
    return std.mem.startsWith(u8, line, "fit ") and std.mem.find(u8, line, " ms  (+") != null;
}

/// Copy `text` without the dropped lines, replacing every duration (a number
/// followed by `ms`, optionally after one space) and its padding with ` #`.
fn mask(w: *Io.Writer, text: []const u8) !void {
    var buf: [4096]u8 = undefined;
    var lines = std.mem.splitScalar(u8, text, '\n');
    var first = true;
    next_line: while (lines.next()) |line| {
        for (dropped_prefixes) |p| if (std.mem.startsWith(u8, line, p)) continue :next_line;
        if (isTimingFitLine(line)) continue;
        // Debug/ReleaseSafe append an error return trace; the error line above it stays.
        if (line.len > 0 and line[0] == '/' and std.mem.find(u8, line, ": 0x") != null) {
            if (!first) try w.writeByte('\n');
            break;
        }
        if (!first) try w.writeByte('\n');
        first = false;
        if (line.len * 2 > buf.len) { // " #" can outgrow a 1-digit duration
            try w.writeAll(line);
            continue;
        }
        var n: usize = 0;
        var i: usize = 0;
        while (i < line.len) {
            const at_token = i == 0 or !isNumChar(line[i - 1]);
            if (at_token and std.ascii.isDigit(line[i])) {
                var j = i;
                while (j < line.len and isNumChar(line[j])) j += 1;
                const k = if (j < line.len and line[j] == ' ') j + 1 else j;
                if (std.mem.startsWith(u8, line[k..], "ms")) {
                    // Right-aligned durations pad by width: fold the padding too.
                    while (n > 0 and buf[n - 1] == ' ') n -= 1;
                    @memcpy(buf[n..][0..2], " #");
                    n += 2;
                } else {
                    @memcpy(buf[n..][0 .. j - i], line[i..j]);
                    n += j - i;
                }
                i = j;
                continue;
            }
            buf[n] = line[i];
            n += 1;
            i += 1;
        }
        try w.writeAll(buf[0..n]);
    }
}

fn isNumChar(c: u8) bool {
    return std.ascii.isDigit(c) or c == '.';
}

/// tune's `--out` CSV has an `ms` column (4th): blank it so the hash is stable.
fn maskMsColumn(gpa: Allocator, bytes: []const u8) ![]const u8 {
    if (!std.mem.startsWith(u8, bytes, "rank,score,folds,ms,")) return bytes;
    var out: std.ArrayList(u8) = .empty;
    var lines = std.mem.splitScalar(u8, bytes, '\n');
    while (lines.next()) |line| {
        var commas: usize = 0;
        for (line) |c| {
            if (c == ',') commas += 1;
            if (commas == 3 and c != ',') continue;
            try out.append(gpa, c);
        }
        try out.append(gpa, '\n');
    }
    return out.items;
}

fn reportDiff(name: []const u8, want: []const u8, got: []const u8) void {
    var wl = std.mem.splitScalar(u8, want, '\n');
    var gl = std.mem.splitScalar(u8, got, '\n');
    var n: usize = 1;
    while (true) : (n += 1) {
        const a = wl.next();
        const b = gl.next();
        if (a == null and b == null) break;
        if (a == null or b == null or !std.mem.eql(u8, a.?, b.?)) {
            std.log.err("{s}: line {d}\n  want: {s}\n  got:  {s}", .{ name, n, a orelse "<eof>", b orelse "<eof>" });
            return;
        }
    }
}
