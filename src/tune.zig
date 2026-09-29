// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

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
const config = @import("config.zig");
const data = @import("data.zig");
const pool_mod = @import("pool.zig");
const cv = @import("cv.zig");

pub const Search = enum { grid, random, bayes, bandit };

// ----- the search space

pub const Kind = enum { choice, uniform, log_uniform, int_uniform, int_log };

/// One axis. Values are carried as f64 so every strategy can treat the space
/// uniformly; `render` turns a coordinate back into the string the config
/// flag parser expects, which is what keeps this generic over any field.
pub const Param = struct {
    name: []const u8,
    kind: Kind,
    choices: []const []const u8 = &.{},
    lo: f64 = 0,
    hi: f64 = 0,

    pub fn isCategorical(p: Param) bool {
        return p.kind == .choice;
    }

    /// Bounds in the *encoded* space -- log-scaled axes are searched in logs,
    /// so that a step near 0.1 counts for as much as a step near 50.
    pub fn bounds(p: Param) struct { lo: f64, hi: f64 } {
        return switch (p.kind) {
            .choice => .{ .lo = 0, .hi = @floatFromInt(p.choices.len - 1) },
            .uniform, .int_uniform => .{ .lo = p.lo, .hi = p.hi },
            .log_uniform, .int_log => .{ .lo = @log(p.lo), .hi = @log(p.hi) },
        };
    }

    pub fn sample(p: Param, r: std.Random) f64 {
        const b = p.bounds();
        return switch (p.kind) {
            .choice => @floatFromInt(r.uintLessThan(usize, p.choices.len)),
            else => b.lo + r.float(f64) * (b.hi - b.lo),
        };
    }

    /// Fold an out-of-range value back inside instead of clamping it.
    ///
    /// Clamping sends every overshoot to the same endpoint, so a kernel
    /// centred near a bound stacks a point mass exactly on it -- proposals
    /// pile up on `lambda=50.000000` and the axis stops being searched.
    /// Reflection puts that probability back into the interior, where it
    /// belongs.
    fn reflect(p: Param, x: f64) f64 {
        const b = p.bounds();
        if (!(b.hi > b.lo)) return b.lo;
        var v = x;
        var guard: usize = 0;
        while ((v < b.lo or v > b.hi) and guard < 16) : (guard += 1) {
            if (v < b.lo) v = b.lo + (b.lo - v);
            if (v > b.hi) v = b.hi - (v - b.hi);
        }
        return std.math.clamp(v, b.lo, b.hi);
    }

    pub fn render(p: Param, x: f64, buf: []u8) ![]const u8 {
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
    pub fn gridPoints(p: Param, gpa: std.mem.Allocator, steps: usize) ![]f64 {
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
pub fn parseParam(gpa: std.mem.Allocator, text: []const u8) !Param {
    const eq = std.mem.findScalar(u8, text, '=') orelse return error.ParamNeedsValue;
    const name = text[0..eq];
    if (!config.hasField(name)) return error.UnknownConfigField;
    var spec = text[eq + 1 ..];
    if (spec.len == 0) return error.EmptyParamSpec;

    if (std.mem.find(u8, spec, "..")) |dots| {
        var is_int = false;
        var is_log = false;
        // Suffixes are order-independent so `:int:log` and `:log:int` agree.
        while (std.mem.findScalarLast(u8, spec, ':')) |c| {
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
pub fn defaultSpace(gpa: std.mem.Allocator, algo: config.Algo) ![]Param {
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

pub const Trial = struct {
    x: []f64,
    score: f64,
    folds: u32,
    ms: i64,
    /// Rendered `name=value` pairs, for printing and for the CSV.
    text: []const u8,
};

/// Whether a Config field decides how the data is *binned* rather than how a
/// model is fitted to it.
///
/// This distinction is not cosmetic. Binning happens once, before any trial,
/// so a search that varies one of these and does nothing about it will report
/// a value it never actually used -- which is exactly what happened: sixty
/// trials printed `max_bin=64` while every one of them fitted the 256-bin
/// matrix, and the winning config did not reproduce when run through `cv`.
pub fn affectsBinning(name: []const u8) bool {
    return std.mem.eql(u8, name, "max_bin") or
        std.mem.eql(u8, name, "bin_policy") or
        std.mem.eql(u8, name, "max_cat_levels");
}

/// Holds the binned matrix, and rebuilds it when a trial asks for binning that
/// differs from what is loaded. Re-binning costs ~200 ms against seconds of
/// fitting, so doing it on change is cheap; doing it never was wrong.
pub const Binner = struct {
    gpa: std.mem.Allocator,
    pool: *pool_mod.Pool,
    frame: *data.Frame,
    label_col: usize,
    enc: *data.LabelEncoder,
    drops: []const []const u8,
    ds: data.Dataset,
    max_bin: u16,
    /// Tracked like `max_bin`, because it changes the binning and a cached
    /// matrix built under a different value is the wrong matrix.
    max_cat_levels: u32 = (config.Config{}).max_cat_levels,
    policy: config.BinPolicy,
    rebins: usize = 0,

    pub fn get(b: *Binner, cfg: config.Config) !*const data.Dataset {
        if (cfg.max_bin != b.max_bin or cfg.bin_policy != b.policy or
            cfg.max_cat_levels != b.max_cat_levels)
        {
            const next = try data.quantise(
                b.gpa,
                b.pool,
                b.frame,
                cfg,
                .{ .col = b.label_col, .enc = b.enc },
                b.drops,
            );
            b.ds.deinit();
            b.ds = next;
            b.max_bin = cfg.max_bin;
            b.policy = cfg.bin_policy;
            b.max_cat_levels = cfg.max_cat_levels;
            b.rebins += 1;
        }
        return &b.ds;
    }
};

/// A configuration and the score it earned at the current rung. Named rather
/// than anonymous because the sort needs a concrete type to specialise on.
pub const Scored = struct { x: []f64, s: f64 };

pub const Evaluator = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    pool: *pool_mod.Pool,
    binner: *Binner,
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

    pub fn describe(e: Evaluator, gpa: std.mem.Allocator, x: []const f64) ![]const u8 {
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
    pub fn run(e: Evaluator, x: []const f64, use_folds: u32) !?cv.Outcome {
        const cfg = try e.apply(x);
        cfg.validate() catch return null;
        // Binning first: a trial that changes `max_bin` needs a different
        // matrix, not just different flags. A `max_bin` below the widest
        // categorical column cannot be binned at all -- that is the search
        // space proposing something invalid, exactly like a rejected
        // `validate`, so it is a skipped trial and not a failure. `get` fails
        // before it frees anything, so the cached matrix is still intact.
        const ds = e.binner.get(cfg) catch |err| switch (err) {
            error.CategoricalTooWide => return null,
            else => return err,
        };
        return cv.crossValidate(e.gpa, e.io, e.pool, ds, cfg, e.fold_of, e.n_folds, .{
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
pub const Tpe = struct {
    gamma: f64,
    candidates: usize,

    fn bandwidth(p: Param, n: usize) f64 {
        const b = p.bounds();
        const span = b.hi - b.lo;
        // Silverman's rule, floored so a handful of observations cannot
        // collapse the kernel onto its own points and stop exploring.
        const h = 1.06 * span * std.math.pow(f64, @floatFromInt(@max(n, 1)), -0.2);
        // Floored so a handful of observations cannot collapse the kernel onto
        // its own points, and capped because Silverman's rule over-smooths
        // badly at small n -- at n=5 it asks for a kernel three quarters as
        // wide as the entire axis.
        return std.math.clamp(h, span * 0.05, span * 0.35);
    }

    /// Density of `at` under the observations `xs`, plus one pseudo-observation
    /// covering the whole axis.
    ///
    /// That prior term is not decoration. Without it the estimate is a sum of
    /// kernels sitting only where the search has already been, so as the good
    /// set concentrates the density outside it goes to zero, the ratio stops
    /// distinguishing anything, and the proposal locks onto a single point --
    /// which is exactly what this did before the prior was added: seven
    /// identical trials in a row with two parameters pinned to their bounds.
    fn logDensity(p: Param, xs: []const f64, at: f64, h: f64) f64 {
        if (p.isCategorical()) {
            const k: f64 = @floatFromInt(p.choices.len);
            var hits: f64 = 0;
            for (xs) |v| if (@round(v) == @round(at)) {
                hits += 1;
            };
            // Laplace smoothing is the categorical form of the same prior: an
            // unobserved level keeps a real probability, so the ratio cannot
            // be infinite on a level nobody has tried yet.
            return @log((hits + 1) / (@as(f64, @floatFromInt(xs.len)) + k));
        }
        const b = p.bounds();
        var acc: f64 = 0;
        for (xs) |v| {
            const z = (at - v) / h;
            acc += @exp(-0.5 * z * z) / h;
        }
        // The prior: one kernel at mid-range, as wide as the range itself.
        const mid = 0.5 * (b.lo + b.hi);
        const hp = @max(0.5 * (b.hi - b.lo), 1e-12);
        const zp = (at - mid) / hp;
        acc += @exp(-0.5 * zp * zp) / hp;

        const n_eff = @as(f64, @floatFromInt(xs.len)) + 1;
        return @log(@max(acc, 1e-300) / (n_eff * @sqrt(2.0 * std.math.pi)));
    }

    pub fn propose(
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
            const b = p.bounds();
            for (0..t.candidates) |_| {
                // Draw from the good density -- which includes its prior
                // component, so one candidate in (n_good + 1) comes from the
                // whole axis rather than from somewhere already visited. That
                // is what keeps the proposal able to leave a basin.
                const from_prior = r.uintLessThan(usize, good.len + 1) == 0;
                const pick = good[r.uintLessThan(usize, good.len)];
                const cand = if (p.isCategorical())
                    if (from_prior)
                        @as(f64, @floatFromInt(r.uintLessThan(usize, p.choices.len)))
                    else
                        pick
                else if (from_prior)
                    b.lo + r.float(f64) * (b.hi - b.lo)
                else
                    p.reflect(pick + r.floatNorm(f64) * hg);
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
