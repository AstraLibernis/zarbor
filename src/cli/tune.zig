// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! `zgbdt tune`: hyperparameter search from the command line (see zarbor.tune).

const std = @import("std");
const builtin = @import("builtin");
const zarbor = @import("zarbor");
const args = @import("args.zig");
const config = zarbor.config;
const data = zarbor.data;
const pool_mod = zarbor.pool;
const linear = zarbor.linear;
const csv = zarbor.csv;
const tune = zarbor.tune;
const Scored = tune.Scored;
const Tpe = tune.Tpe;
const Trial = tune.Trial;
const Evaluator = tune.Evaluator;
const Search = tune.Search;
const defaultSpace = tune.defaultSpace;
const Param = tune.Param;
const parseParam = tune.parseParam;
const Binner = tune.Binner;
const cv = zarbor.cv;

const usage =
    \\usage: zgbdt tune <train.csv> --label=<column> [options]
    \\
    \\  --search=NAME       grid | random | bayes | bandit   (default random)
    \\  --trials=N          evaluations, or the starting population for
    \\                      bandit (default 40)
    \\  --param=NAME=SPEC   what to vary; repeatable. SPEC is either a
    \\                      comma-separated list of values, or LO..HI with
    \\                      optional :int and :log suffixes, e.g.
    \\                        --param=max_depth=4,5,6,7
    \\                        --param=lambda=0.1..50:log
    \\                        --param=n_rounds=200..800:int
    \\                      With no --param the default space for the chosen
    \\                      --algo is used.
    \\  --folds=N           folds per evaluation (default 5)
    \\  --fold-seed=N       seed for the fold assignment (default 1)
    \\  --group-col=NAME    keep rows sharing this column's value in the
    \\                      same fold, and drop it as a feature. Required
    \\                      whenever a unit appears more than once (a panel,
    \\                      repeated measures) or CV measures memory, not
    \\                      generalisation.
    \\  --seed=N            seed for the search itself (default 1)
    \\  --confirm=N         re-score the top N on unseen fold seeds
    \\  --confirm-seeds=N   how many unseen seeds to use (default 2)
    \\  --out=FILE          write every trial as CSV
    \\  --grid-steps=N      points per continuous axis for grid (default 5)
    \\  --warmup=N          random trials before bayes starts modelling (12)
    \\  --candidates=N      proposals bayes scores per trial (default 24)
    \\  --gamma=F           best fraction bayes treats as "good" (default 0.25)
    \\  --eta=N             bandit's cull factor per rung (default 3)
    \\  --min-folds=N       bandit's cheapest rung (default 2)
    \\
    \\Any Config flag pins a value for the whole search:
    \\  zgbdt tune t.csv --label=y --algo=linear --search=bayes --trials=80
    \\
;

pub fn run(init: std.process.Init, gpa: std.mem.Allocator, out: *std.Io.Writer) !void {
    const io = init.io;

    var cfg: config.Config = .{};
    var csv_path: ?[]const u8 = null;
    var label: ?[]const u8 = null;
    var pos_label: ?[]const u8 = null;
    var out_path: ?[]const u8 = null;
    var max_bytes: usize = 1 << 31;
    var search: Search = .random;
    var trials_want: usize = 40;
    var n_folds: u32 = 5;
    var fold_seed: u64 = 1;
    var group_col: ?[]const u8 = null;
    var seed: u64 = 1;
    var grid_steps: usize = 5;
    var warmup: usize = 12;
    var candidates: usize = 24;
    var gamma: f64 = 0.25;
    var eta: usize = 3;
    var min_folds: u32 = 2;
    var confirm: usize = 0;
    var confirm_seeds: usize = 2;

    var drops: std.ArrayList([]const u8) = .empty;
    defer drops.deinit(gpa);
    var explicit: std.ArrayList([]const u8) = .empty;
    defer explicit.deinit(gpa);
    var specs: std.ArrayList([]const u8) = .empty;
    defer specs.deinit(gpa);

    var it = args.Iterator.init(init.minimal.args, 1);
    while (it.next()) |arg| {
        const flag = switch (arg) {
            .positional => |p| {
                csv_path = p;
                continue;
            },
            .bare => return error.FlagNeedsValue,
            .flag => |f| f,
        };
        const key, const val = .{ flag.key, flag.val };
        if (std.mem.eql(u8, key, "label")) {
            label = val;
        } else if (std.mem.eql(u8, key, "pos-label")) {
            pos_label = val;
        } else if (std.mem.eql(u8, key, "drop")) {
            try drops.append(gpa, val);
        } else if (std.mem.eql(u8, key, "param")) {
            try specs.append(gpa, val);
        } else if (std.mem.eql(u8, key, "search")) {
            search = std.meta.stringToEnum(Search, val) orelse return error.UnknownSearch;
        } else if (std.mem.eql(u8, key, "trials")) {
            trials_want = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "folds")) {
            n_folds = try std.fmt.parseInt(u32, val, 10);
        } else if (std.mem.eql(u8, key, "fold-seed")) {
            fold_seed = try std.fmt.parseInt(u64, val, 10);
        } else if (std.mem.eql(u8, key, "group-col")) {
            group_col = val;
        } else if (std.mem.eql(u8, key, "seed")) {
            seed = try std.fmt.parseInt(u64, val, 10);
        } else if (std.mem.eql(u8, key, "grid-steps")) {
            grid_steps = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "warmup")) {
            warmup = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "candidates")) {
            candidates = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "gamma")) {
            gamma = try std.fmt.parseFloat(f64, val);
        } else if (std.mem.eql(u8, key, "eta")) {
            eta = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "min-folds")) {
            min_folds = try std.fmt.parseInt(u32, val, 10);
        } else if (std.mem.eql(u8, key, "confirm")) {
            confirm = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "confirm-seeds")) {
            confirm_seeds = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "out")) {
            out_path = val;
        } else if (std.mem.eql(u8, key, "max-bytes")) {
            max_bytes = try std.fmt.parseInt(usize, val, 10);
        } else if (try config.applyFlag(&cfg, key, val)) {
            try explicit.append(gpa, key);
        } else {
            try out.print("unknown flag: --{s}\n\n", .{key});
            try out.writeAll(usage);
            try out.flush();
            return error.UnknownFlag;
        }
    }

    const path = csv_path orelse {
        try out.writeAll(usage);
        try out.flush();
        return error.NoInput;
    };
    const target = label orelse {
        try out.writeAll(usage);
        try out.flush();
        return error.NoLabel;
    };
    if (n_folds < 2) return error.TooFewFolds;
    if (eta < 2) return error.EtaTooSmall;

    cfg.applyAlgoDefaults(explicit.items);
    cfg.verbose_eval = 0;

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const space = if (specs.items.len == 0)
        try defaultSpace(arena, cfg.algo)
    else blk: {
        const ps = try arena.alloc(Param, specs.items.len);
        for (ps, specs.items) |*p, s| p.* = try parseParam(arena, s);
        break :blk ps;
    };

    const pool = try pool_mod.Pool.init(gpa, cfg.n_threads);
    defer pool.deinit();

    const t0 = std.Io.Timestamp.now(io, .awake).toNanoseconds();
    var frame = try data.readCsv(gpa, io, pool, path, max_bytes);
    defer frame.deinit();
    const label_col = frame.columnIndex(target) orelse return error.LabelColumnNotFound;
    var group_idx: ?usize = null;
    if (group_col) |name| {
        group_idx = frame.columnIndex(name) orelse return error.GroupColumnNotFound;
        try drops.append(gpa, name);
    }
    var enc = try data.LabelEncoder.fromColumn(gpa, &frame, label_col, pos_label);
    defer enc.deinit();
    // The binner owns the binned matrix from the moment it exists, and frees
    // the old one on every re-bin. It is constructed here, before anything
    // fallible, precisely so there is never a second owner: an
    // `errdefer full.deinit()` alongside `defer binner.ds.deinit()` double-freed
    // on every error path, and freed already-freed memory once a re-bin had
    // swapped the matrix out from under the stale handle.
    var binner = Binner{
        .gpa = gpa,
        .pool = pool,
        .frame = &frame,
        .label_col = label_col,
        .enc = &enc,
        .drops = drops.items,
        .ds = try data.quantise(gpa, pool, &frame, cfg, .{ .col = label_col, .enc = &enc }, drops.items),
        .max_bin = cfg.max_bin,
        .policy = cfg.bin_policy,
    };
    defer binner.ds.deinit();
    // Row order, labels and feature count do not change with binning, so a
    // view taken now stays correct across every re-bin.
    const full = &binner.ds;
    try enc.validate(full.labels, cfg.objective);
    cfg.applyForestFeatureDefault(full.n_features, explicit.items);
    const prep_ms = @divTrunc(
        std.Io.Timestamp.now(io, .awake).toNanoseconds() - t0,
        1_000_000,
    );

    // Row order and labels do not change with binning, so a fold assignment
    // and an out-of-fold buffer built now stay valid across every re-bin.
    const fold_of = if (group_idx) |gi|
        try cv.assignGroupFolds(gpa, frame.values[gi], n_folds, fold_seed)
    else
        try cv.assignFolds(gpa, full.labels, n_folds, fold_seed, cfg.objective == .logistic);
    defer gpa.free(fold_of);
    const oof = try gpa.alloc(f32, full.n_rows);
    defer gpa.free(oof);

    const ev = Evaluator{
        .gpa = gpa,
        .io = io,
        .pool = pool,
        .binner = &binner,
        .base = cfg,
        .space = space,
        .fold_of = fold_of,
        .n_folds = n_folds,
        .oof = oof,
    };

    try out.print(
        \\data     {s}
        \\rows     {d}   feats {d}   (label "{s}")
        \\algo     {s}
        \\search   {s}
        \\space    {d} params
        \\folds    {d}  (seed {d})
        \\threads  {d}
        \\build    {s}
        \\prep     {d} ms  (read + bin, paid once for the whole search)
        \\
        \\
    , .{
        path,                   full.n_rows,
        full.n_features,        target,
        @tagName(cfg.algo),     @tagName(search),
        space.len,              n_folds,
        fold_seed,              pool.workerCount(),
        @tagName(builtin.mode), prep_ms,
    });
    for (space) |p| {
        try out.print("  {s:<20} {s}", .{ p.name, @tagName(p.kind) });
        if (p.isCategorical()) {
            try out.writeAll("  [");
            for (p.choices, 0..) |c, i| {
                if (i != 0) try out.writeAll(", ");
                try out.writeAll(c);
            }
            try out.writeAll("]\n");
        } else {
            try out.print("  {d} .. {d}\n", .{ p.lo, p.hi });
        }
    }
    try out.writeAll("\n");
    try out.flush();

    var prng: std.Random.DefaultPrng = .init(seed);
    const r = prng.random();

    var results: std.ArrayList(Trial) = .empty;
    defer results.deinit(gpa);
    var best: ?usize = null;

    const record = struct {
        fn go(
            a: std.mem.Allocator,
            w: *std.Io.Writer,
            e: Evaluator,
            list: *std.ArrayList(Trial),
            best_i: *?usize,
            x: []const f64,
            o: cv.Outcome,
            obj: config.Objective,
        ) !void {
            const xs = try a.alloc(f64, x.len);
            @memcpy(xs, x);
            const text = try e.describe(a, x);
            try list.append(e.gpa, .{
                .x = xs,
                .score = o.pooled,
                .folds = o.folds_run,
                .ms = o.fit_ms,
                .text = text,
            });
            const i = list.items.len - 1;
            var mark: []const u8 = "";
            if (best_i.* == null or cv.better(obj, o.pooled, list.items[best_i.*.?].score)) {
                best_i.* = i;
                mark = "  <- best";
            }
            try w.print("{d:>4}  {d:.6}  {d}f {d:>6}ms  {s}{s}\n", .{
                i + 1,       o.pooled,
                o.folds_run, @as(u64, @intCast(@max(o.fit_ms, 0))),
                text,        mark,
            });
            try w.flush();
        }
    }.go;

    const x = try gpa.alloc(f64, space.len);
    defer gpa.free(x);

    switch (search) {
        .grid => {
            // Enumerate the lattice by odometer. The product is reported
            // before any fitting, because "grid over 10 axes" is usually a
            // mistake the user wants to hear about rather than discover.
            const axes = try arena.alloc([]f64, space.len);
            var total: usize = 1;
            for (axes, space) |*ax, p| {
                ax.* = try p.gridPoints(arena, grid_steps);
                total *|= ax.len;
            }
            try out.print("grid     {d} combinations", .{total});
            if (total > trials_want) {
                try out.print(", capped at --trials={d}\n\n", .{trials_want});
            } else {
                try out.writeAll("\n\n");
            }
            try out.flush();

            const digit = try arena.alloc(usize, space.len);
            @memset(digit, 0);
            var done: usize = 0;
            while (done < @min(total, trials_want)) : (done += 1) {
                for (x, axes, digit) |*v, ax, d| v.* = ax[d];
                if (try ev.run(x, 0)) |o|
                    try record(arena, out, ev, &results, &best, x, o, cfg.objective);
                var d: usize = 0;
                while (d < digit.len) : (d += 1) {
                    digit[d] += 1;
                    if (digit[d] < axes[d].len) break;
                    digit[d] = 0;
                }
                if (d == digit.len) break; // odometer wrapped: lattice exhausted
            }
        },
        .random => {
            for (0..trials_want) |_| {
                for (x, space) |*v, p| v.* = p.sample(r);
                if (try ev.run(x, 0)) |o|
                    try record(arena, out, ev, &results, &best, x, o, cfg.objective);
            }
        },
        .bayes => {
            const tpe = Tpe{ .gamma = gamma, .candidates = candidates };
            for (0..trials_want) |i| {
                if (i < warmup or results.items.len < 4) {
                    for (x, space) |*v, p| v.* = p.sample(r);
                } else {
                    try tpe.propose(gpa, r, space, results.items, cfg.objective, x);
                }
                if (try ev.run(x, 0)) |o|
                    try record(arena, out, ev, &results, &best, x, o, cfg.objective);
            }
        },
        .bandit => {
            // Successive halving over fold count. Every rung re-scores its
            // survivors from scratch at the higher fidelity rather than
            // reusing the cheap score, so what finally ranks them is a real
            // full-fidelity number and not an average of different budgets.
            var alive: std.ArrayList([]f64) = .empty;
            defer alive.deinit(gpa);
            for (0..trials_want) |_| {
                const xs = try arena.alloc(f64, space.len);
                for (xs, space) |*v, p| v.* = p.sample(r);
                try alive.append(gpa, xs);
            }
            var rung_folds = @min(min_folds, n_folds);
            var rung: usize = 0;
            while (true) : (rung += 1) {
                try out.print("rung {d}  {d} configs at {d} folds\n", .{
                    rung, alive.items.len, rung_folds,
                });
                try out.flush();

                var scored: std.ArrayList(Scored) = .empty;
                defer scored.deinit(gpa);
                for (alive.items) |xs| {
                    if (try ev.run(xs, rung_folds)) |o| {
                        try scored.append(gpa, .{ .x = xs, .s = o.pooled });
                        // Only a full-fidelity result is comparable with the
                        // other strategies, so only those are recorded.
                        if (o.folds_run == n_folds)
                            try record(arena, out, ev, &results, &best, xs, o, cfg.objective);
                    }
                }
                if (scored.items.len == 0) break;

                const S = struct {
                    obj: config.Objective,
                    fn lessThan(c: @This(), a: Scored, b: Scored) bool {
                        return cv.better(c.obj, a.s, b.s);
                    }
                };
                std.sort.pdq(Scored, scored.items, S{ .obj = cfg.objective }, S.lessThan);

                if (rung_folds >= n_folds) break;
                const keep = @max(@as(usize, 1), scored.items.len / eta);
                alive.clearRetainingCapacity();
                for (scored.items[0..keep]) |e| try alive.append(gpa, e.x);
                rung_folds = @min(n_folds, @max(rung_folds + 1, rung_folds * @as(u32, @intCast(eta))));
                if (keep == 1) rung_folds = n_folds;
            }
        },
    }

    if (results.items.len == 0) {
        try out.writeAll("\nno valid configuration was found\n");
        try out.flush();
        return error.NoValidTrials;
    }

    const order = try gpa.alloc(u32, results.items.len);
    defer gpa.free(order);
    for (order, 0..) |*v, i| v.* = @intCast(i);
    const Ctx = struct {
        t: []const Trial,
        obj: config.Objective,
        fn lessThan(c: @This(), a: u32, b: u32) bool {
            return cv.better(c.obj, c.t[a].score, c.t[b].score);
        }
    };
    std.sort.pdq(u32, order, Ctx{ .t = results.items, .obj = cfg.objective }, Ctx.lessThan);

    const win = results.items[order[0]];
    try out.print("\nbest     {d:.6}   {d} trials", .{ win.score, results.items.len });
    if (binner.rebins != 0) try out.print("   ({d} re-bins)", .{binner.rebins});
    try out.print("\n  {s}\n", .{win.text});

    // Re-run the winner from scratch on the folds it was searched on. A
    // reported configuration that does not reproduce means the search applied
    // something it did not print, or printed something it did not apply --
    // which is how `max_bin` was caught being ignored while sixty trials
    // claimed to be varying it. One evaluation is a cheap price for knowing.
    if (try ev.run(win.x, 0)) |again| {
        const delta = again.pooled - win.score;
        if (@abs(delta) > 1e-9) {
            try out.print(
                "WARNING  re-run gives {d:.6}, searched {d:.6}, difference {d:.6}" ++
                    " -- the reported config does not reproduce\n",
                .{ again.pooled, win.score, delta },
            );
        } else {
            try out.print("verified {d:.6}   (winner re-ran identically)\n", .{again.pooled});
        }
    }

    if (confirm > 0) {
        // The winner is the maximum of many trials on one split, so part of
        // its margin is whatever suited that split. Re-scoring on seeds it has
        // never seen is the only way to tell that part from the real one.
        try out.print("\nconfirming the top {d} on {d} unseen fold seed(s)\n\n", .{
            @min(confirm, results.items.len), confirm_seeds,
        });
        try out.print("{s:>5}  {s:>9}", .{ "rank", "searched" });
        for (0..confirm_seeds) |s| try out.print("  {s}{d:<7}", .{ "seed", fold_seed + 1 + s });
        try out.print("  {s:>9}\n", .{"mean new"});
        try out.flush();

        for (order[0..@min(confirm, results.items.len)], 0..) |ti, rank| {
            const t = results.items[ti];
            try out.print("{d:>5}  {d:9.6}", .{ rank + 1, t.score });
            var sum: f64 = 0;
            var n_ok: usize = 0;
            for (0..confirm_seeds) |s| {
                const fs = fold_seed + 1 + s;
                const folds2 = if (group_idx) |gi|
                    try cv.assignGroupFolds(gpa, frame.values[gi], n_folds, fs)
                else
                    try cv.assignFolds(gpa, full.labels, n_folds, fs, cfg.objective == .logistic);
                defer gpa.free(folds2);
                var ev2 = ev;
                ev2.fold_of = folds2;
                if (try ev2.run(t.x, 0)) |o| {
                    try out.print("  {d:11.6}", .{o.pooled});
                    sum += o.pooled;
                    n_ok += 1;
                } else try out.writeAll("            -");
                try out.flush();
            }
            if (n_ok != 0) {
                try out.print("  {d:9.6}\n", .{sum / @as(f64, @floatFromInt(n_ok))});
            } else try out.writeAll("          -\n");
            try out.flush();
        }
    }

    if (out_path) |op| {
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(gpa);
        try buf.appendSlice(gpa, "rank,score,folds,ms,params\n");
        var line: [256]u8 = undefined;
        for (order, 0..) |ti, rank| {
            const t = results.items[ti];
            try buf.appendSlice(gpa, try std.fmt.bufPrint(&line, "{d},{d:.8},{d},{d},\"", .{
                rank + 1, t.score, t.folds, t.ms,
            }));
            try buf.appendSlice(gpa, t.text);
            try buf.appendSlice(gpa, "\"\n");
        }
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = op, .data = buf.items });
        try out.print("\nwrote    {s}\n", .{op});
    }
    try out.flush();
}
