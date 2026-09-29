// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Regularised linear and logistic regression — the simple baseline.
//!
//! The design matrix is derived from the same binned dataset the trees use,
//! which is what keeps one data path for the whole library: a numeric feature
//! contributes its bin's representative value, a categorical one contributes a
//! one-hot block, and the missing bin encodes as the column mean (zero once
//! standardised). Nothing is materialised — columns are read straight out of
//! the u8 bin matrix on each pass, so memory stays at the size of the bins.
//!
//! Binning a numeric feature before fitting a linear model is a small loss of
//! resolution and a real gain in robustness: quantile edges bound the influence
//! of outliers, which plain OLS on raw columns handles badly.
//!
//! Fitting is L-BFGS by default: it estimates the problem's curvature from
//! recent steps, which is what lets it reach the optimum in a few dozen passes
//! instead of the few thousand a fixed-step first-order method needs. An L1
//! term switches it to OWL-QN, the orthant-wise variant, so `alpha` still
//! produces exact zeros rather than merely small coefficients.
//! `--lin_solver=adam` keeps the first-order fitter available.

const std = @import("std");
const Pool = @import("pool.zig").Pool;
const data = @import("data.zig");
const Dataset = data.Dataset;
const prof = @import("prof.zig");
const metric = @import("metric.zig");
const Objective = @import("objective.zig").Objective;
const lin_solve = @import("lin_solve.zig");
pub const Fit = lin_solve.Fit;
const reduce_chunks = lin_solve.reduce_chunks;
const reduce_parallel_min = lin_solve.reduce_parallel_min;
const Problem = lin_solve.Problem;
const fitLbfgs = lin_solve.fitLbfgs;
const fitAdam = lin_solve.fitAdam;

/// How the linear model's coefficients are fitted. Both minimise the *same*
/// convex objective, so wherever both arrive they must agree -- measured, and
/// the places they do not are in docs/linear-solvers.md.
pub const LinSolver = enum {
    /// **L-BFGS** (limited-memory Broyden-Fletcher-Goldfarb-Shanno), with
    /// **OWL-QN** (orthant-wise limited-memory quasi-Newton) when `alpha` > 0.
    /// Builds a curvature estimate from recent steps, so it reaches the
    /// optimum in tens of passes where a first-order method needs thousands.
    /// This is also what scikit-learn's LogisticRegression defaults to, which
    /// makes the two directly comparable.
    lbfgs,
    /// **Adam** (adaptive moment estimation), full batch, with an L1
    /// proximal step. Needs no objective
    /// evaluation and so no line search, which makes each pass cheaper — but
    /// it takes far more of them to reach the same coefficients.
    adam,
};

/// The linear model's settings. `objective` picks the model: logistic
/// regression or linear regression (see objective.zig).
pub const Params = struct {
    objective: Objective = .logistic,
    /// Multiplier on positive-class gradients. >1 upweights the minority
    /// class for imbalanced binary problems.
    scale_pos_weight: f32 = 1.0,
    /// L2 penalty on the coefficients (ridge). The intercept is unpenalised.
    lambda: f32 = 1.0,
    /// L1 penalty on the coefficients (lasso). The intercept is unpenalised.
    alpha: f32 = 0.0,
    /// Which optimiser fits the coefficients. Linear only.
    lin_solver: LinSolver = .lbfgs,
    /// Maximum full-batch iterations. `lbfgs` normally converges in a few
    /// dozen and stops on `lin_tol` well short of this; `adam` usually needs
    /// all of them. Linear only.
    lin_epochs: u32 = 300,
    /// Adam step size. Ignored by `lbfgs`, which gets its step length from a
    /// line search rather than from a constant. Linear only.
    lin_lr: f32 = 0.05,
    /// Stop when the largest absolute coefficient change in an iteration
    /// falls below this. Same units for both solvers.
    lin_tol: f32 = 1e-7,
    /// Standardise each design column to zero mean and unit variance before
    /// fitting. Off makes the penalties scale-dependent and is rarely right.
    lin_standardize: bool = true,
    /// Print per-round metrics every N rounds. 0 silences training.
    verbose_eval: u32 = 10,

    pub fn validate(p: Params) !void {
        if (p.lin_epochs == 0) return error.NoEpochs;
        if (p.lin_lr <= 0) return error.BadLinearLr;
        if (p.lambda < 0 or p.alpha < 0) return error.NegativeRegularisation;
    }
};

/// Ceiling on one-hot expansion. A categorical with thousands of levels would
/// silently turn a small table into a huge design matrix; fail loudly instead.
pub const max_design_cols: usize = 1 << 16;

const numeric_col: u16 = std.math.maxInt(u16);

/// One column of the design matrix. Public because saving a linear model
/// means writing these out.
pub const Col = struct {
    feature: u32,
    /// `numeric_col`, or the bin this one-hot column indicates.
    bin: u16,
    center: f32,
    /// Reciprocal of the standard deviation; 0 for a constant column, which
    /// zeroes the column out rather than dividing by ~0.
    scale: f32,
};

/// Bin-to-value tables plus the standardisation the fit was done under.
/// Owned by the model because prediction needs exactly the same mapping.
pub const Design = struct {
    gpa: std.mem.Allocator,
    cols: []Col,
    /// `repr[f][b]` is the value bin `b` of feature `f` stands for. Only
    /// populated for numeric features.
    repr: [][]f32,

    pub fn deinit(d: *Design) void {
        for (d.repr) |r| if (r.len != 0) d.gpa.free(r);
        d.gpa.free(d.repr);
        d.gpa.free(d.cols);
        d.* = undefined;
    }

    pub inline fn raw(d: *const Design, c: Col, ds: *const Dataset, row: usize) f32 {
        const b = ds.bins[@as(usize, c.feature) * ds.n_rows + row];
        if (c.bin == numeric_col) return d.repr[c.feature][b];
        return if (b == c.bin) 1.0 else 0.0;
    }

    pub inline fn value(d: *const Design, c: Col, ds: *const Dataset, row: usize) f32 {
        return (d.raw(c, ds, row) - c.center) * c.scale;
    }
};

fn buildDesign(
    gpa: std.mem.Allocator,
    ds: *const Dataset,
    standardize: bool,
) !Design {
    const repr = try gpa.alloc([]f32, ds.n_features);
    errdefer gpa.free(repr);
    @memset(repr, &.{});

    var cols: std.ArrayList(Col) = .empty;
    errdefer cols.deinit(gpa);

    for (0..ds.n_features) |f| {
        const nb = ds.n_bins[f];
        if (nb <= 1) continue; // nothing but the missing bin: no information
        switch (ds.kinds[f]) {
            .numeric => {
                const table = try gpa.alloc(f32, nb);
                repr[f] = table;
                // Binning computed the mean of the values that landed in each
                // bin, which is what a bin should stand for. Fall back to the
                // midpoint of the bin's edges for a dataset assembled without
                // going through `quantise`; that is a biased stand-in under a
                // skewed within-bin distribution, and on the EV set it costs
                // 0.0003 AUC against the means.
                if (ds.means[f].len == nb) {
                    @memcpy(table, ds.means[f]);
                } else {
                    var sum: f64 = 0;
                    for (1..nb) |b| {
                        table[b] = data.binMidpoint(ds.edges[f], b - 1);
                        sum += table[b];
                    }
                    // Bin 0 is missing; give it the feature's mean so a
                    // missing entry sits at the centre and contributes
                    // nothing once standardised.
                    table[0] = if (nb <= 1) 0 else @floatCast(sum / @as(f64, @floatFromInt(nb - 1)));
                }
                try cols.append(gpa, .{ .feature = @intCast(f), .bin = numeric_col, .center = 0, .scale = 1 });
            },
            .categorical => {
                if (cols.items.len + nb - 1 > max_design_cols) return error.DesignMatrixTooWide;
                for (1..nb) |b| {
                    try cols.append(gpa, .{
                        .feature = @intCast(f),
                        .bin = @intCast(b),
                        .center = 0,
                        .scale = 1,
                    });
                }
            },
        }
        if (cols.items.len > max_design_cols) return error.DesignMatrixTooWide;
    }

    if (cols.items.len == 0) return error.NoUsableFeatures;

    var d = Design{ .gpa = gpa, .cols = try cols.toOwnedSlice(gpa), .repr = repr };
    errdefer d.deinit();

    if (standardize) {
        const n: f64 = @floatFromInt(ds.n_rows);
        for (d.cols) |*c| {
            var sum: f64 = 0;
            var sq: f64 = 0;
            for (0..ds.n_rows) |r| {
                const v = d.raw(c.*, ds, r);
                sum += v;
                sq += @as(f64, v) * v;
            }
            const mean = sum / n;
            const varr = @max(sq / n - mean * mean, 0);
            const sd = @sqrt(varr);
            c.center = @floatCast(mean);
            c.scale = if (sd > 1e-12) @floatCast(1.0 / sd) else 0.0;
        }
    }
    return d;
}

// ------------------------------------------------------------------- model

pub const Linear = struct {
    gpa: std.mem.Allocator,
    design: Design,
    w: []f32,
    intercept: f32,
    objective: Objective,
    n_features: usize,

    pub fn deinit(m: *Linear) void {
        m.design.deinit();
        m.gpa.free(m.w);
        m.* = undefined;
    }

    /// Linear predictor before the link function.
    pub fn predictRaw(m: *const Linear, pool: *Pool, ds: *const Dataset, out: []f32) void {
        std.debug.assert(out.len == ds.n_rows);
        var ctx = ScoreCtx{ .m = m, .ds = ds, .out = out };
        pool.parallelFor(ds.n_rows, &ctx, ScoreCtx.run, 1024);
    }

    /// Probabilities for logistic, values for regression.
    pub fn predict(m: *const Linear, pool: *Pool, ds: *const Dataset, out: []f32) void {
        m.predictRaw(pool, ds, out);
        if (m.objective == .logistic) for (out) |*v| {
            v.* = 1.0 / (1.0 + @exp(-v.*));
        };
    }

    /// Number of coefficients L1 drove exactly to zero.
    pub fn nZero(m: *const Linear) usize {
        var n: usize = 0;
        for (m.w) |c| {
            if (c == 0) n += 1;
        }
        return n;
    }
};

const ScoreCtx = struct {
    m: *const Linear,
    ds: *const Dataset,
    out: []f32,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *ScoreCtx = @ptrCast(@alignCast(ctx));
        const d = &self.m.design;
        var r = begin;
        while (r < end) : (r += 1) {
            var acc: f32 = self.m.intercept;
            for (d.cols, self.m.w) |c, coef| {
                if (coef != 0) acc += coef * d.value(c, self.ds, r);
            }
            self.out[r] = acc;
        }
    }
};

// ---------------------------------------------------------------- driver

pub const TrainResult = struct {
    model: Linear,
    epochs: u32,
    /// How the fit ended. A caller that prints nothing else should still
    /// check `fit.stalled()`: a stalled solve returns coefficients that look
    /// ordinary and are not a solution to anything.
    fit: Fit = .{ .iters = 0 },
    score: f64,
    /// Nanoseconds spent scoring the validation set; not part of fitting.
    /// See `booster.TrainResult.valid_ns`.
    valid_ns: u64,
};

pub fn train(
    gpa: std.mem.Allocator,
    pool: *Pool,
    ds: *const Dataset,
    valid: ?*const Dataset,
    cfg: Params,
    log: ?*std.Io.Writer,
) !TrainResult {
    try cfg.validate();
    if (ds.labels.len == 0) return error.NoLabels;

    var design = try buildDesign(gpa, ds, cfg.lin_standardize);
    errdefer design.deinit();
    const p = design.cols.len;

    const theta = try gpa.alloc(f64, p + 1);
    defer gpa.free(theta);
    @memset(theta, 0);

    const wf = try gpa.alloc(f32, p);
    defer gpa.free(wf);
    const z = try gpa.alloc(f32, ds.n_rows);
    defer gpa.free(z);
    const resid = try gpa.alloc(f32, ds.n_rows);
    defer gpa.free(resid);
    const loss_part = try gpa.alloc(f64, reduce_chunks);
    defer gpa.free(loss_part);
    const rsum_part = try gpa.alloc(f64, reduce_chunks);
    defer gpa.free(rsum_part);

    // Start the intercept at the base rate so the first steps do not have to
    // travel there; for logistic that is the log-odds of the label mean.
    var sum: f64 = 0;
    for (ds.labels) |y| sum += y;
    const mean = sum / @as(f64, @floatFromInt(ds.labels.len));
    theta[p] = switch (cfg.objective) {
        .logistic => blk: {
            const q = std.math.clamp(mean, 1e-6, 1 - 1e-6);
            break :blk @log(q / (1 - q));
        },
        .squared_error => mean,
    };

    const n_f: f64 = @floatFromInt(ds.n_rows);
    const chunks: usize = if (ds.n_rows >= reduce_parallel_min) reduce_chunks else 1;
    var pr = Problem{
        .pool = pool,
        .design = &design,
        .ds = ds,
        .objective = cfg.objective,
        .scale_pos_weight = cfg.scale_pos_weight,
        // L1/L2 are stated against the summed loss, so divide by n to match
        // the mean-loss gradient the steps are taken against.
        .l2 = @as(f64, cfg.lambda) / n_f,
        .l1 = @as(f64, cfg.alpha) / n_f,
        .theta = theta,
        .wf = wf,
        .z = z,
        .resid = resid,
        .loss_part = loss_part,
        .rsum_part = rsum_part,
        .chunks = chunks,
        .size = (ds.n_rows + chunks - 1) / chunks,
    };

    const fit: Fit = switch (cfg.lin_solver) {
        .lbfgs => try fitLbfgs(gpa, &pr, cfg, log),
        // Adam has no line search and so no "the search could not move"
        // failure, which is why `converged` stays true for it. But "did the
        // gradient actually come down" is answerable for any solver, and
        // leaving it unanswered meant adam reported success while returning
        // a coin-flip model on a badly scaled design.
        .adam => try fitAdam(gpa, &pr, cfg, log),
    };
    const epochs = fit.iters;

    const w = try gpa.alloc(f32, p);
    errdefer gpa.free(w);
    for (w, theta[0..p]) |*d, s| d.* = @floatCast(s);

    var model = Linear{
        .gpa = gpa,
        .design = design,
        .w = w,
        .intercept = @floatCast(theta[p]),
        .objective = cfg.objective,
        .n_features = ds.n_features,
    };
    errdefer model.deinit();

    var score: f64 = std.math.nan(f64);
    var valid_ns: u64 = 0;
    if (valid) |v| {
        const wall0 = prof.now();
        const pred = try gpa.alloc(f32, v.n_rows);
        defer gpa.free(pred);
        model.predict(pool, v, pred);
        score = switch (cfg.objective) {
            .logistic => try metric.auc(gpa, pred, v.labels),
            .squared_error => metric.rmse(pred, v.labels),
        };
        valid_ns = prof.now() - wall0;
    }

    return .{
        .model = model,
        .epochs = epochs,
        .fit = fit,
        .score = score,
        .valid_ns = valid_ns,
    };
}
