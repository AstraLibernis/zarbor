//! Hyperparameter search, with the strategy selected the way the model is.
//!
//! `--search` picks between four families that differ in what they assume:
//!
//!   grid    Every combination of a discrete lattice. Assumes nothing, and
//!           pays for it: cost is the product of the axes, so it re-tests the
//!           same value of an important knob once per combination of the
//!           unimportant ones. Exhaustive and reproducible, which is what it
//!           is for -- not for finding an optimum in ten dimensions.
//!   random  Independent draws per parameter. The honest baseline: with only
//!           a few parameters mattering, every trial gives each of them a
//!           fresh value, which is exactly what a grid fails to do.
//!   bayes   Tree-structured Parzen Estimator. Models the density of good
//!           configurations against the density of bad ones and proposes
//!           where their ratio is highest. Spends its budget near what has
//!           worked, at the risk of committing early to a local basin.
//!   bandit  Successive halving. Starts many configurations at a cheap
//!           fidelity, keeps the best fraction, and re-runs the survivors at
//!           a higher one. Assumes a config's cheap score ranks roughly like
//!           its expensive score -- usually true, and false exactly when a
//!           config needs its full budget to show its worth.
//!
//! All four share one evaluator (`cv.crossValidate`) over one binned dataset,
//! so a strategy difference is never confounded by a data difference, and the
//! CSV is read and binned once for the entire search rather than once per
//! trial.
//!
//! The search reports against the fold split it searched on. That number is
//! the maximum of many trials and so is optimistic by construction -- the
//! winner is partly whatever suited this split. `--confirm=N` re-scores the
//! leaders on fold seeds they have never seen, which is the number to believe.

const std = @import("std");
const builtin = @import("builtin");
const config = @import("config.zig");
const data = @import("data.zig");
const pool_mod = @import("pool.zig");
const cv = @import("cv.zig");

pub const Search = enum { grid, random, bayes, bandit };

pub const usage =
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

// ----- the search space

const Kind = enum { choice, uniform, log_uniform, int_uniform, int_log };

/// One axis. Values are carried as f64 so every strategy can treat the space
/// uniformly; `render` turns a coordinate back into the string the config
/// flag parser expects, which is what keeps this generic over any field.
const Param = struct {
    name: []const u8,
    kind: Kind,
    choices: []const []const u8 = &.{},
    lo: f64 = 0,
    hi: f64 = 0,

    fn isCategorical(p: Param) bool {
        return p.kind == .choice;
    }

    /// Bounds in the *encoded* space -- log-scaled axes are searched in logs,
    /// so that a step near 0.1 counts for as much as a step near 50.
    fn bounds(p: Param) struct { lo: f64, hi: f64 } {
        return switch (p.kind) {
            .choice => .{ .lo = 0, .hi = @floatFromInt(p.choices.len - 1) },
            .uniform, .int_uniform => .{ .lo = p.lo, .hi = p.hi },
            .log_uniform, .int_log => .{ .lo = @log(p.lo), .hi = @log(p.hi) },
        };
    }

    fn sample(p: Param, r: std.Random) f64 {
        const b = p.bounds();
        return switch (p.kind) {
            .choice => @floatFromInt(r.uintLessThan(usize, p.choices.len)),
            else => b.lo + r.float(f64) * (b.hi - b.lo),
        };
    }

    fn clamp(p: Param, x: f64) f64 {
        const b = p.bounds();
        return std.math.clamp(x, b.lo, b.hi);
    }

    fn render(p: Param, x: f64, buf: []u8) ![]const u8 {
        return switch (p.kind) {
            .choice => blk: {
                const i: usize = @intFromFloat(@round(std.math.clamp(
                    x,
                    0,
                    @as(f64, @floatFromInt(p.choices.len - 1)),
                )));
                break :blk p.choices[i];
            },
            .uniform => try std.fmt.bufPrint(buf, "{d:.6}", .{x}),
            .log_uniform => try std.fmt.bufPrint(buf, "{d:.6}", .{@exp(x)}),
            .int_uniform => try std.fmt.bufPrint(buf, "{d}", .{@as(i64, @intFromFloat(@round(x)))}),
            .int_log => try std.fmt.bufPrint(buf, "{d}", .{@as(i64, @intFromFloat(@round(@exp(x))))}),
        };
    }

    /// Lattice points for `grid`. A continuous axis has no natural set, so it
    /// is cut into `steps` points from end to end -- in the encoded space, so
    /// a log axis is cut geometrically.
    fn gridPoints(p: Param, gpa: std.mem.Allocator, steps: usize) ![]f64 {
        if (p.kind == .choice) {
            const xs = try gpa.alloc(f64, p.choices.len);
            for (xs, 0..) |*v, i| v.* = @floatFromInt(i);
            return xs;
        }
        const n = @max(steps, 2);
        const b = p.bounds();
        const xs = try gpa.alloc(f64, n);
        for (xs, 0..) |*v, i|
            v.* = b.lo + (b.hi - b.lo) * @as(f64, @floatFromInt(i)) /
                @as(f64, @floatFromInt(n - 1));
        return xs;
    }
};

/// `name=spec`, where spec is `a,b,c` or `lo..hi` with `:int` / `:log`.
fn parseParam(gpa: std.mem.Allocator, text: []const u8) !Param {
    const eq = std.mem.indexOfScalar(u8, text, '=') orelse return error.ParamNeedsValue;
    const name = text[0..eq];
    if (!config.hasField(name)) return error.UnknownConfigField;
    var spec = text[eq + 1 ..];
    if (spec.len == 0) return error.EmptyParamSpec;

    if (std.mem.indexOf(u8, spec, "..")) |dots| {
        var is_int = false;
        var is_log = false;
        // Suffixes are order-independent so `:int:log` and `:log:int` agree.
        while (std.mem.lastIndexOfScalar(u8, spec, ':')) |c| {
            const suffix = spec[c + 1 ..];
            if (std.mem.eql(u8, suffix, "int")) {
                is_int = true;
            } else if (std.mem.eql(u8, suffix, "log")) {
                is_log = true;
            } else break;
            spec = spec[0..c];
        }
        const lo = try std.fmt.parseFloat(f64, spec[0..dots]);
        const hi = try std.fmt.parseFloat(f64, spec[dots + 2 ..]);
        if (!(hi > lo)) return error.EmptyParamRange;
        if (is_log and lo <= 0) return error.LogRangeNeedsPositiveLow;
        return .{
            .name = name,
            .kind = if (is_int and is_log) .int_log else if (is_int)
                .int_uniform
            else if (is_log) .log_uniform else .uniform,
            .lo = lo,
            .hi = hi,
        };
    }

    var list: std.ArrayList([]const u8) = .empty;
    errdefer list.deinit(gpa);
    var it = std.mem.splitScalar(u8, spec, ',');
    while (it.next()) |v| if (v.len != 0) try list.append(gpa, v);
    if (list.items.len == 0) return error.EmptyParamSpec;
    return .{ .name = name, .kind = .choice, .choices = try list.toOwnedSlice(gpa) };
}

/// A space worth searching for each model, so `tune` is useful with no
/// `--param` at all. Ranges bracket the shipped defaults rather than centring
/// on them, so a search can move in either direction.
fn defaultSpace(gpa: std.mem.Allocator, algo: config.Algo) ![]Param {
    const specs: []const []const u8 = switch (algo) {
        .gbdt => &.{
            "n_rounds=200,300,500,800",
            "learning_rate=0.02..0.2:log",
            "max_depth=3,4,5,6,7,8",
            "lambda=0.1..50:log",
            "min_child_weight=0.1..100:log",
            "subsample=0.5..1.0",
            "colsample_bytree=0.5..1.0",
            "max_bin=64,128,256",
        },
        .random_forest => &.{
            "n_rounds=100,200,300",
            "max_leaves=128,256,512,1024,2048",
            "min_child_samples=1,5,20",
            "colsample_bynode=0.2..1.0",
            "max_bin=64,128,256",
        },
        .linear => &.{
            "lambda=0.01..1000:log",
            "alpha=0.0,0.1,1.0,10.0,100.0",
            "lin_epochs=100,300,600",
            "lin_standardize=true,false",
        },
    };
    const ps = try gpa.alloc(Param, specs.len);
    for (ps, specs) |*p, s| p.* = try parseParam(gpa, s);
    return ps;
}

// ----- trials

const Trial = struct {
    x: []f64,
    score: f64,
    folds: u32,
    ms: i64,
    /// Rendered `name=value` pairs, for printing and for the CSV.
    text: []const u8,
};

/// A configuration and the score it earned at the current rung. Named rather
/// than anonymous because the sort needs a concrete type to specialise on.
const Scored = struct { x: []f64, s: f64 };

const Evaluator = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    pool: *pool_mod.Pool,
    full: *const data.Dataset,
    base: config.Config,
    space: []const Param,
    fold_of: []const u32,
    n_folds: u32,
    oof: []f32,

    fn apply(e: Evaluator, x: []const f64) !config.Config {
        var cfg = e.base;
        var buf: [64]u8 = undefined;
        for (e.space, x) |p, v| {
            const rendered = try p.render(v, &buf);
            _ = try config.applyFlag(&cfg, p.name, rendered);
        }
        return cfg;
    }

    fn describe(e: Evaluator, gpa: std.mem.Allocator, x: []const f64) ![]const u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        var buf: [64]u8 = undefined;
        for (e.space, x, 0..) |p, v, i| {
            if (i != 0) try out.append(gpa, ' ');
            try out.appendSlice(gpa, p.name);
            try out.append(gpa, '=');
            try out.appendSlice(gpa, try p.render(v, &buf));
        }
        return out.toOwnedSlice(gpa);
    }

    /// Returns null when the configuration is invalid -- a search space can
    /// always propose a combination `Config.validate` rejects, and that is a
    /// skipped trial, not a failure.
    fn run(e: Evaluator, x: []const f64, use_folds: u32) !?cv.Outcome {
        const cfg = try e.apply(x);
        cfg.validate() catch return null;
        return cv.crossValidate(e.gpa, e.io, e.pool, e.full, cfg, e.fold_of, e.n_folds, .{
            .use_folds = use_folds,
            .oof = e.oof,
        }) catch |err| switch (err) {
            error.FoldLeftNoTrainingRows => return null,
            else => return err,
        };
    }
};

// ----- strategies

/// TPE. Splits what has been seen into a good set and a bad set, builds a
/// Parzen density over each, and proposes the candidate maximising the ratio.
/// Modelling each axis independently is the "tree-structured" simplification:
/// it cannot represent "a deep tree wants a bigger lambda", but it costs
/// nothing and is what makes the estimator usable at this trial count.
const Tpe = struct {
    gamma: f64,
    candidates: usize,

    fn bandwidth(p: Param, n: usize) f64 {
        const b = p.bounds();
        const span = b.hi - b.lo;
        // Silverman's rule, floored so a handful of observations cannot
        // collapse the kernel onto its own points and stop exploring.
        const h = 1.06 * span * std.math.pow(f64, @floatFromInt(@max(n, 1)), -0.2);
        return @max(h, span * 0.05);
    }

    fn logDensity(p: Param, xs: []const f64, at: f64, h: f64) f64 {
        if (p.isCategorical()) {
            const k: f64 = @floatFromInt(p.choices.len);
            var hits: f64 = 0;
            for (xs) |v| if (@round(v) == @round(at)) {
                hits += 1;
            };
            // Laplace smoothing: an unobserved level keeps a real probability,
            // so the ratio cannot be infinite on a level nobody tried yet.
            return @log((hits + 1) / (@as(f64, @floatFromInt(xs.len)) + k));
        }
        var acc: f64 = 0;
        for (xs) |v| {
            const z = (at - v) / h;
            acc += @exp(-0.5 * z * z);
        }
        const norm = @as(f64, @floatFromInt(@max(xs.len, 1))) * h * @sqrt(2.0 * std.math.pi);
        return @log(@max(acc, 1e-300) / norm);
    }

    fn propose(
        t: Tpe,
        gpa: std.mem.Allocator,
        r: std.Random,
        space: []const Param,
        trials: []const Trial,
        obj: config.Objective,
        out: []f64,
    ) !void {
        const n = trials.len;
        const order = try gpa.alloc(u32, n);
        defer gpa.free(order);
        for (order, 0..) |*v, i| v.* = @intCast(i);
        const Ctx = struct {
            t: []const Trial,
            obj: config.Objective,
            fn lessThan(c: @This(), a: u32, b: u32) bool {
                return cv.better(c.obj, c.t[a].score, c.t[b].score);
            }
        };
        std.sort.pdq(u32, order, Ctx{ .t = trials, .obj = obj }, Ctx.lessThan);

        const n_good = @max(@as(usize, 2), @min(n - 1, @as(usize, @intFromFloat(
            @floor(t.gamma * @as(f64, @floatFromInt(n))),
        ))));

        const good = try gpa.alloc(f64, n_good);
        defer gpa.free(good);
        const bad = try gpa.alloc(f64, n - n_good);
        defer gpa.free(bad);

        for (space, 0..) |p, d| {
            for (good, 0..) |*v, i| v.* = trials[order[i]].x[d];
            for (bad, 0..) |*v, i| v.* = trials[order[n_good + i]].x[d];
            const hg = bandwidth(p, n_good);
            const hb = bandwidth(p, bad.len);

            var best_x: f64 = good[0];
            var best_ratio: f64 = -std.math.inf(f64);
            for (0..t.candidates) |_| {
                // Draw from the good density itself: pick one of its points
                // and jitter by the kernel width.
                const pick = good[r.uintLessThan(usize, good.len)];
                const cand = if (p.isCategorical())
                    pick
                else
                    p.clamp(pick + r.floatNorm(f64) * hg);
                const ratio = logDensity(p, good, cand, hg) - logDensity(p, bad, cand, hb);
                if (ratio > best_ratio) {
                    best_ratio = ratio;
                    best_x = cand;
                }
            }
            out[d] = best_x;
        }
    }
};

// ----- the command

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

    var it = std.process.Args.Iterator.init(init.minimal.args);
    _ = it.skip();
    _ = it.skip(); // "tune"
    while (it.next()) |arg| {
        if (!std.mem.startsWith(u8, arg, "--")) {
            csv_path = arg;
            continue;
        }
        const body = arg[2..];
        const eq = std.mem.indexOfScalar(u8, body, '=') orelse return error.FlagNeedsValue;
        const key = body[0..eq];
        const val = body[eq + 1 ..];
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
    var enc = try data.LabelEncoder.fromColumn(gpa, &frame, label_col, pos_label);
    defer enc.deinit();
    var full = try data.quantise(gpa, pool, &frame, cfg, .{ .col = label_col, .enc = &enc }, drops.items);
    defer full.deinit();
    try enc.validate(full.labels, cfg.objective);
    cfg.applyForestFeatureDefault(full.n_features, explicit.items);
    const prep_ms = @divTrunc(
        std.Io.Timestamp.now(io, .awake).toNanoseconds() - t0,
        1_000_000,
    );

    const fold_of = try cv.assignFolds(gpa, full.labels, n_folds, fold_seed, cfg.objective == .logistic);
    defer gpa.free(fold_of);
    const oof = try gpa.alloc(f32, full.n_rows);
    defer gpa.free(oof);

    const ev = Evaluator{
        .gpa = gpa,
        .io = io,
        .pool = pool,
        .full = &full,
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
        path,              full.n_rows,
        full.n_features,   target,
        @tagName(cfg.algo), @tagName(search),
        space.len,         n_folds,
        fold_seed,         pool.workerCount(),
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
                i + 1,          o.pooled,
                o.folds_run,    @as(u64, @intCast(@max(o.fit_ms, 0))),
                text,           mark,
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
    try out.print("\nbest     {d:.6}   {d} trials\n  {s}\n", .{
        win.score, results.items.len, win.text,
    });

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
                const folds2 = try cv.assignFolds(gpa, full.labels, n_folds, fs, cfg.objective == .logistic);
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

// ----- tests

const testing = std.testing;

test "a choice spec keeps its values and its order" {
    const p = try parseParam(testing.allocator, "max_depth=4,5,6");
    defer testing.allocator.free(p.choices);
    try testing.expectEqual(Kind.choice, p.kind);
    try testing.expectEqual(@as(usize, 3), p.choices.len);
    try testing.expectEqualStrings("4", p.choices[0]);
    try testing.expectEqualStrings("6", p.choices[2]);
}

test "range specs pick up their int and log suffixes in either order" {
    const a = try parseParam(testing.allocator, "lambda=0.1..50");
    try testing.expectEqual(Kind.uniform, a.kind);
    const b = try parseParam(testing.allocator, "lambda=0.1..50:log");
    try testing.expectEqual(Kind.log_uniform, b.kind);
    const c = try parseParam(testing.allocator, "n_rounds=200..800:int");
    try testing.expectEqual(Kind.int_uniform, c.kind);
    const d = try parseParam(testing.allocator, "n_rounds=200..800:int:log");
    try testing.expectEqual(Kind.int_log, d.kind);
    const e = try parseParam(testing.allocator, "n_rounds=200..800:log:int");
    try testing.expectEqual(Kind.int_log, e.kind);
}

test "a spec naming no config field is rejected" {
    try testing.expectError(error.UnknownConfigField, parseParam(testing.allocator, "not_a_field=1,2"));
    try testing.expectError(error.EmptyParamRange, parseParam(testing.allocator, "lambda=50..1"));
    try testing.expectError(error.LogRangeNeedsPositiveLow, parseParam(testing.allocator, "lambda=0..50:log"));
}

test "a log axis is searched geometrically, not linearly" {
    const p = try parseParam(testing.allocator, "lambda=1..100:log");
    const pts = try p.gridPoints(testing.allocator, 3);
    defer testing.allocator.free(pts);
    var buf: [64]u8 = undefined;
    // Encoded midpoint of log(1)..log(100) is log(10), so the middle lattice
    // point must be 10 -- a linear axis would have put 50.5 there.
    const mid = try p.render(pts[1], &buf);
    try testing.expectEqualStrings("10.000000", mid);
}

test "rendered values round-trip through the config flag parser" {
    var cfg: config.Config = .{};
    const p = try parseParam(testing.allocator, "max_depth=3..9:int");
    var buf: [64]u8 = undefined;
    const s = try p.render(7.4, &buf);
    try testing.expect(try config.applyFlag(&cfg, p.name, s));
    try testing.expectEqual(@as(u32, 7), cfg.max_depth);
}

test "grid points span the whole axis inclusively" {
    const p = try parseParam(testing.allocator, "subsample=0.5..1.0");
    const pts = try p.gridPoints(testing.allocator, 5);
    defer testing.allocator.free(pts);
    try testing.expectEqual(@as(usize, 5), pts.len);
    try testing.expectApproxEqAbs(@as(f64, 0.5), pts[0], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), pts[4], 1e-12);
}

test "sampling stays inside the declared bounds" {
    var prng: std.Random.DefaultPrng = .init(4);
    const r = prng.random();
    const specs = [_][]const u8{
        "lambda=0.1..50:log",
        "subsample=0.5..1.0",
        "n_rounds=200..800:int",
        "max_depth=4,5,6",
    };
    for (specs) |s| {
        const p = try parseParam(testing.allocator, s);
        defer if (p.kind == .choice) testing.allocator.free(p.choices);
        const b = p.bounds();
        for (0..200) |_| {
            const v = p.sample(r);
            try testing.expect(v >= b.lo - 1e-9 and v <= b.hi + 1e-9);
        }
    }
}

test "TPE proposes where good outnumbers bad, not merely where it has looked" {
    const gpa = testing.allocator;
    const p = try parseParam(gpa, "subsample=0.0..1.0");
    const space = [_]Param{p};

    // Twenty observations whose score rises with the parameter. A proposal
    // drawn from the good density must land in the upper part of the range;
    // a sampler ignoring the scores would average out near the middle.
    var trials: std.ArrayList(Trial) = .empty;
    defer trials.deinit(gpa);
    var xs: [20][1]f64 = undefined;
    for (&xs, 0..) |*slot, i| {
        slot[0] = @as(f64, @floatFromInt(i)) / 19.0;
        try trials.append(gpa, .{
            .x = slot,
            .score = slot[0],
            .folds = 5,
            .ms = 1,
            .text = "",
        });
    }

    var prng: std.Random.DefaultPrng = .init(9);
    const tpe = Tpe{ .gamma = 0.25, .candidates = 32 };
    var hits: usize = 0;
    var outv: [1]f64 = undefined;
    for (0..40) |_| {
        try tpe.propose(gpa, prng.random(), &space, trials.items, .logistic, &outv);
        if (outv[0] > 0.6) hits += 1;
    }
    try testing.expect(hits > 30);
}

test "the default space covers every model" {
    inline for (.{ config.Algo.gbdt, .random_forest, .linear }) |a| {
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        const s = try defaultSpace(arena.allocator(), a);
        try testing.expect(s.len >= 4);
        // Every default axis must name a real field, or `tune` would fail
        // only once someone ran it without --param.
        for (s) |p| try testing.expect(config.hasField(p.name));
    }
}
