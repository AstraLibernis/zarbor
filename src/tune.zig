// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Hyperparameter search: space, evaluator and strategies (flags and printing
//! live in src/cli/tune.zig). Four `Search` families, differing in assumptions:
//!   grid    every lattice combination; assumes nothing, costs the product of
//!           the axes (re-tests a key knob per combo of unimportant ones).
//!           Exhaustive and reproducible, not an optimiser in ten dimensions.
//!   random  independent draws; the honest baseline: every trial gives each
//!           of the few parameters that matter a fresh value, unlike a grid.
//!   bayes   Tree-structured Parzen Estimator (TPE): proposes where good/bad
//!           density ratio peaks; may commit early to a local basin.
//!   bandit  successive halving: many configs at cheap fidelity, survivors
//!           re-run higher. Assumes cheap score ranks like full score; false
//!           exactly when a config needs its full budget to show its worth.
//! All share one evaluator (`cv.crossValidate`) over one binned dataset: no
//! strategy/data confound. The CSV is read once per search, and binned again
//! only when a trial's binning parameters differ (`Binner`).
//! The best score on the searched split is optimistic (max of many trials);
//! re-scoring leaders on unseen fold seeds (CLI `--confirm`) is the one to believe.

const std = @import("std");
const config = @import("config.zig");
const data = @import("data.zig");
const pool_mod = @import("pool.zig");
const cv = @import("cv.zig");
const Objective = @import("objective.zig").Objective;

pub const Search = enum { grid, random, bayes, bandit };

// ----- the search space

pub const Kind = enum { choice, uniform, log_uniform, int_uniform, int_log };

/// One axis, carried as f64 so every strategy treats the space uniformly;
/// `render` gives back the config flag string, keeping this generic per field.
pub const Param = struct {
    name: []const u8,
    kind: Kind,
    choices: []const []const u8 = &.{},
    lo: f64 = 0,
    hi: f64 = 0,

    pub fn isCategorical(p: Param) bool {
        return p.kind == .choice;
    }

    /// Bounds in the *encoded* space: log axes are searched in logs, so a step
    /// near 0.1 counts as much as one near 50.
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

    /// Reflect an out-of-range value inside, not clamp: clamping stacks a
    /// point mass on the bound (proposals piled on `lambda=50.000000`) and
    /// the axis stops being searched.
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

    /// Whether `x` (encoded) sits in the outer 5% of a range, or is the first
    /// or last of three or more numeric choices: a best value there may be
    /// held back by the range, not found inside it.
    pub fn atEdge(p: Param, x: f64) ?enum { low, high } {
        if (p.kind == .choice) {
            if (p.choices.len < 3) return null;
            for (p.choices) |c| _ = std.fmt.parseFloat(f64, c) catch return null;
            const i: usize = @intFromFloat(@round(std.math.clamp(x, 0, @as(f64, @floatFromInt(p.choices.len - 1)))));
            if (i == 0) return .low;
            if (i == p.choices.len - 1) return .high;
            return null;
        }
        const b = p.bounds();
        const margin = (b.hi - b.lo) * 0.05;
        if (!(margin > 0)) return null;
        if (x <= b.lo + margin) return .low;
        if (x >= b.hi - margin) return .high;
        return null;
    }

    /// Lattice points for `grid`: a continuous axis is cut into `steps` points
    /// end to end in the encoded space (a log axis geometrically).
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

/// Default space per model, so a search needs no explicit params. Ranges
/// include the shipped defaults; where a default sits at a range's edge
/// (e.g. `subsample` 1.0), the search can only move one way. `max_bin` stops at 1024: past
/// it, a tuned model gained nothing and every trial got slower (docs/archive/binning.md); an `edge`
/// line says when a winner presses on it, and `--param` searches further.
pub fn defaultSpace(gpa: std.mem.Allocator, algo: config.Algo) ![]Param {
    const specs: []const []const u8 = switch (algo) {
        .gbdt => &.{
            "n_rounds=200,300,500,800",
            "learning_rate=0.02..0.2:log",
            "max_depth=3,4,5,6,7,8",
            "lambda=0.1..300:log",
            "min_child_weight=0.01..300:log",
            "subsample=0.5..1.0",
            "colsample_bytree=0.1..1.0",
            "max_bin=8..1024:int:log",
            "min_data_in_bin=1..300:int:log",
        },
        .random_forest => &.{
            "n_rounds=100,200,300",
            "max_leaves=128,256,512,1024,2048",
            "min_child_samples=1,5,20",
            "colsample_bynode=0.2..1.0",
            "max_bin=8..1024:int:log",
            "min_data_in_bin=1..300:int:log",
        },
        .linear => &.{
            "lambda=0.0001..1000:log",
            "alpha=0.0,0.1,1.0,10.0,100.0",
            "lin_epochs=100,300,600",
            "lin_standardize=true,false",
        },
    };
    const ps = try gpa.alloc(Param, specs.len);
    for (ps, specs) |*p, s| p.* = try parseParam(gpa, s);
    return ps;
}

/// The default gbdt axis for categorical data: `optimal` closed the whole gap to LightGBM on
/// House Prices, and a search that never tries it cannot find that. Added only when a feature is
/// categorical, so numeric files pay nothing.
pub const cat_split_axis = "cat_split=ordinal,optimal";

/// Whether the binned data has a categorical feature.
pub fn hasCategorical(kinds: []const data.ColumnKind) bool {
    for (kinds) |k| if (k == .categorical) return true;
    return false;
}

/// The default space minus every axis the command line pinned: a flag like `--n_rounds=100`
/// is a promise the search will not vary it, and searching it anyway once silently overrode
/// the pin. Returned in `gpa`, reusing `space`'s entries.
pub fn withoutPinned(gpa: std.mem.Allocator, space: []const Param, pinned: []const []const u8) ![]Param {
    var kept: std.ArrayList(Param) = .empty;
    errdefer kept.deinit(gpa);
    outer: for (space) |p| {
        for (pinned) |name| if (std.mem.eql(u8, p.name, name)) continue :outer;
        try kept.append(gpa, p);
    }
    return kept.toOwnedSlice(gpa);
}

/// The first `--param` axis that a flag also pins, if any: one of the two would be ignored.
pub fn pinnedAndSearched(space: []const Param, pinned: []const []const u8) ?[]const u8 {
    for (space) |p| for (pinned) |name| if (std.mem.eql(u8, p.name, name)) return name;
    return null;
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

/// Holds the binned matrix; rebuilds it when a trial's binning differs.
/// Binning happens before trials, so a binning knob varied unnoticed reports a
/// value never used: sixty trials once printed `max_bin=64` but fitted the
/// 256-bin matrix, and the winner did not reproduce through `cv`. So the whole
/// `data.BinParams` is compared, never a hand-kept list of its fields: a list
/// once missed `min_data_in_bin`, and a knob added later would be missed again.
/// Re-binning costs far less than fitting; never doing it is wrong.
pub const Binner = struct {
    gpa: std.mem.Allocator,
    pool: *pool_mod.Pool,
    frame: *data.Frame,
    label_col: usize,
    enc: *data.LabelEncoder,
    drops: []const []const u8,
    ds: data.Dataset,
    /// The parameters `ds` was binned under; any difference from a trial's re-bins.
    bin: data.BinParams,
    rebins: usize = 0,

    pub fn get(b: *Binner, cfg: config.Config) !*const data.Dataset {
        if (!std.meta.eql(cfg.bin, b.bin)) {
            const next = try data.quantise(
                b.gpa,
                b.pool,
                b.frame,
                cfg.bin,
                .{ .col = b.label_col, .enc = b.enc },
                b.drops,
            );
            b.ds.deinit();
            b.ds = next;
            b.bin = cfg.bin;
            b.rebins += 1;
        }
        return &b.ds;
    }
};

/// A config and its score at the current rung; named so the sort can specialise.
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

    /// Null for an invalid config (one `Config.validate` rejects): a skipped
    /// trial, not a failure.
    pub fn run(e: Evaluator, x: []const f64, use_folds: u32) !?cv.Outcome {
        const cfg = try e.apply(x);
        cfg.validate() catch return null;
        // Binning first: a new `max_bin` needs a new matrix. One below the
        // widest categorical is invalid like a rejected `validate`: skipped.
        // `get` fails before freeing, so the cached matrix stays intact.
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

/// TPE: Parzen densities over good and bad trials, propose the max ratio.
/// Axes are modelled independently ("tree-structured"): cannot express "deep
/// trees want bigger lambda", but free and usable at this trial count.
pub const Tpe = struct {
    gamma: f64,
    candidates: usize,

    fn bandwidth(p: Param, n: usize) f64 {
        const b = p.bounds();
        const span = b.hi - b.lo;
        // Silverman's rule, floored so few observations cannot collapse the
        // kernel onto its own points and stop exploring; capped since it
        // over-smooths at small n (n=5 asks for 3/4 of the axis).
        const h = 1.06 * span * std.math.pow(f64, @floatFromInt(@max(n, 1)), -0.2);
        return std.math.clamp(h, span * 0.05, span * 0.35);
    }

    /// Density of `at` under `xs` plus one whole-axis pseudo-observation.
    /// Without that prior, density outside the visited set goes to zero, the
    /// ratio stops discriminating and proposals lock on one point (seen: seven
    /// identical trials, two parameters pinned to their bounds).
    fn logDensity(p: Param, xs: []const f64, at: f64, h: f64) f64 {
        if (p.isCategorical()) {
            const k: f64 = @floatFromInt(p.choices.len);
            var hits: f64 = 0;
            for (xs) |v| if (@round(v) == @round(at)) {
                hits += 1;
            };
            // Laplace smoothing, the categorical prior: an untried level keeps
            // real probability, so the ratio cannot be infinite on it.
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
        obj: Objective,
        out: []f64,
    ) !void {
        const n = trials.len;
        const order = try gpa.alloc(u32, n);
        defer gpa.free(order);
        for (order, 0..) |*v, i| v.* = @intCast(i);
        const Ctx = struct {
            t: []const Trial,
            obj: Objective,
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
                // Draw from the good density incl. its prior: one candidate in
                // (n_good + 1) spans the whole axis, so proposals can leave a basin.
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
