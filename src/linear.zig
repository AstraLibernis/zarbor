// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Regularised linear and logistic regression: the simple baseline.
//! The design matrix derives from the trees' binned dataset, keeping one data path:
//! a numeric feature gives its bin's representative value, a categorical a one-hot
//! block, the missing bin the column mean (zero once standardised). Nothing is
//! materialised: columns are read from the bin matrix each pass, so memory stays
//! at the bins' size. Binning costs a little resolution, but quantile edges bound
//! outlier influence, which plain OLS on raw columns handles badly.
//! Default fit is L-BFGS: a curvature estimate from recent steps reaches the optimum
//! in a few dozen passes, not the thousands a fixed-step first-order method needs.
//! L1 switches to OWL-QN (orthant-wise) so `alpha` gives exact zeros, not small
//! coefficients. `--lin_solver=adam` keeps the first-order fitter.

const std = @import("std");
const Pool = @import("pool.zig").Pool;
const data = @import("data.zig");
const Dataset = data.Dataset;
const prof = @import("prof.zig");
const metric = @import("metric.zig");
const Objective = @import("objective.zig").Objective;
const booster = @import("booster.zig");
const lin_solve = @import("lin_solve.zig");
pub const Fit = lin_solve.Fit;
const reduce_chunks = lin_solve.reduce_chunks;
const reduce_parallel_min = lin_solve.reduce_parallel_min;
const Problem = lin_solve.Problem;
const fitLbfgs = lin_solve.fitLbfgs;
const fitAdam = lin_solve.fitAdam;

/// How coefficients are fitted. Both minimise the *same* convex objective, so where
/// both arrive they must agree; measured, with the exceptions, in docs/archive/linear-solvers.md.
pub const LinSolver = enum {
    /// **L-BFGS** (limited-memory Broyden-Fletcher-Goldfarb-Shanno), **OWL-QN**
    /// (orthant-wise limited-memory quasi-Newton) when `alpha` > 0. Tens of passes where
    /// first-order needs thousands; scikit-learn LogisticRegression's default, so comparable.
    lbfgs,
    /// **Adam** (adaptive moment estimation), full batch, L1 proximal step. No objective
    /// evaluation, so no line search: cheaper passes, but far more of them.
    adam,
};

/// Linear model settings; `objective` picks logistic or linear regression (objective.zig).
pub const Params = struct {
    objective: Objective = .logistic,
    /// Multiplier on positive-class gradients; >1 upweights an imbalanced minority.
    scale_pos_weight: f32 = 1.0,
    /// Classes under `objective = softmax` (multinomial logistic regression); set from the
    /// label's classes, not by hand.
    num_class: u32 = 0,
    /// L2 penalty on the coefficients (ridge). The intercept is unpenalised.
    lambda: f32 = 1.0,
    /// L1 penalty on the coefficients (lasso). The intercept is unpenalised.
    alpha: f32 = 0.0,
    /// Which optimiser fits the coefficients. Linear only.
    lin_solver: LinSolver = .lbfgs,
    /// Max full-batch iterations. `lbfgs` usually stops on `lin_tol` within a few
    /// dozen; `adam` usually needs all. Linear only.
    lin_epochs: u32 = 300,
    /// Adam step size; `lbfgs` ignores it (line search sets its step). Linear only.
    lin_lr: f32 = 0.05,
    /// Stop when an iteration's largest coefficient change is below this (both solvers).
    lin_tol: f32 = 1e-7,
    /// Standardise columns to zero mean, unit variance. Off (scale-dependent penalties) is rarely right.
    lin_standardize: bool = true,
    /// Log solver progress (objective and gradient norm; `adam` logs the largest
    /// coefficient step instead of the objective) every N iterations for `lbfgs`,
    /// every 10*N epochs for `adam`, plus the last. 0 silences training.
    verbose_eval: u32 = 10,

    pub fn validate(p: Params) !void {
        if (p.lin_epochs == 0) return error.NoEpochs;
        if (p.lin_lr <= 0) return error.BadLinearLr;
        if (p.lambda < 0 or p.alpha < 0) return error.NegativeRegularisation;
        if (p.objective == .softmax) {
            if (p.num_class < 2) return error.SoftmaxNeedsClasses;
            if (p.scale_pos_weight != 1) return error.ScalePosWeightNeedsBinary;
        }
    }
};

/// One-hot ceiling: fail loudly rather than silently build a huge design matrix.
pub const max_design_cols: usize = 1 << 16;

pub const numeric_col: u16 = std.math.maxInt(u16);

/// One design-matrix column; public because saving a linear model writes these.
pub const Col = struct {
    feature: u32,
    /// `numeric_col`, or the bin this one-hot column indicates.
    bin: u16,
    center: f32,
    /// 1/stddev; 0 for a constant column, zeroing it rather than dividing by ~0.
    scale: f32,
};

/// Bin-to-value tables plus the standardisation the fit was done under.
/// Owned by the model because prediction needs exactly the same mapping.
pub const Design = struct {
    gpa: std.mem.Allocator,
    cols: []Col,
    /// `repr[f][b]`: the value bin `b` of feature `f` stands for. Numeric only.
    repr: [][]f32,

    pub fn deinit(d: *Design) void {
        for (d.repr) |r| if (r.len != 0) d.gpa.free(r);
        d.gpa.free(d.repr);
        d.gpa.free(d.cols);
        d.* = undefined;
    }

    pub inline fn raw(d: *const Design, c: Col, ds: *const Dataset, row: usize) f32 {
        // A column with more than 256 bins lives in `wide_cols`; its narrow slot is zeros.
        const b: data.BinIdx = if (ds.isWide(c.feature)) ds.columnWide(c.feature)[row] else ds.columnNarrow(c.feature)[row];
        if (c.bin == numeric_col) return d.repr[c.feature][b];
        return if (b == c.bin) 1.0 else 0.0;
    }

    pub inline fn value(d: *const Design, c: Col, ds: *const Dataset, row: usize) f32 {
        return (d.raw(c, ds, row) - c.center) * c.scale;
    }

    /// Per column, `value` for every bin of its feature, computed by the same f32 expression.
    /// The solver's loops index these instead of recomputing per row and re-checking each
    /// read for a wide column. Caller frees with `freeTables`.
    pub fn valueTables(d: *const Design, gpa: std.mem.Allocator, ds: *const Dataset) ![][]f32 {
        const t = try gpa.alloc([]f32, d.cols.len);
        var made: usize = 0;
        errdefer {
            for (t[0..made]) |x| gpa.free(x);
            gpa.free(t);
        }
        for (d.cols, t) |c, *tab| {
            tab.* = try gpa.alloc(f32, ds.n_bins[c.feature]);
            made += 1;
            for (tab.*, 0..) |*v, b| {
                const rv: f32 = if (c.bin == numeric_col) d.repr[c.feature][b] else if (b == c.bin) 1.0 else 0.0;
                v.* = (rv - c.center) * c.scale;
            }
        }
        return t;
    }

    pub fn freeTables(gpa: std.mem.Allocator, t: [][]f32) void {
        for (t) |x| gpa.free(x);
        gpa.free(t);
    }
};

pub fn buildDesign(
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
                // Binning's per-bin value means. Without `quantise`, fall back to edge
                // midpoints: biased under skew, and measurably less accurate.
                if (ds.means[f].len == nb) {
                    @memcpy(table, ds.means[f]);
                } else {
                    var sum: f64 = 0;
                    for (1..nb) |b| {
                        table[b] = data.binMidpoint(ds.edges[f], b - 1);
                        sum += table[b];
                    }
                    // Bin 0 (missing) gets the unweighted average of the bin midpoints.
                    // Standardisation centres on the row mean, so it is not zero after it.
                    table[0] = @floatCast(sum / @as(f64, @floatFromInt(nb - 1))); // nb >= 2: checked above
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
    /// Coefficients; under softmax `num_class` blocks of one per design column, class-major.
    w: []f32,
    intercept: f32,
    objective: Objective,
    n_features: usize,
    /// Softmax: classes and each class's intercept (owned; `intercept` is then unused).
    num_class: u32 = 1,
    intercepts: []f32 = &.{},

    pub fn deinit(m: *Linear) void {
        m.design.deinit();
        m.gpa.free(m.w);
        if (m.intercepts.len != 0) m.gpa.free(m.intercepts);
        m.* = undefined;
    }

    /// Scores per row: the class count under softmax, else 1. Row-major.
    pub fn width(m: *const Linear) usize {
        return m.objective.width(m.num_class);
    }

    /// Linear predictor before the link function (one per class under softmax).
    pub fn predictRaw(m: *const Linear, pool: *Pool, ds: *const Dataset, out: []f32) void {
        std.debug.assert(out.len == ds.n_rows * m.width());
        var ctx = ScoreCtx{ .m = m, .ds = ds, .out = out };
        pool.parallelFor(ds.n_rows, &ctx, ScoreCtx.run, 1024);
    }

    /// Probabilities for logistic and softmax, values for regression.
    pub fn predict(m: *const Linear, pool: *Pool, ds: *const Dataset, out: []f32) void {
        m.predictRaw(pool, ds, out);
        switch (m.objective) {
            .logistic => for (out) |*v| {
                v.* = 1.0 / (1.0 + @exp(-v.*));
            },
            .softmax => {
                const k = m.width();
                for (0..ds.n_rows) |r| booster.softmaxRow(out[r * k ..][0..k]);
            },
            .squared_error => {},
        }
    }

    /// Number of coefficients exactly zero: those L1 drove there, plus those of
    /// constant columns (scale 0 when standardised), which never leave their zero start.
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
        const k = self.m.width();
        if (k > 1) {
            const p = d.cols.len;
            for (begin..end) |r| for (0..k) |cls| {
                var acc: f32 = self.m.intercepts[cls];
                for (d.cols, self.m.w[cls * p ..][0..p]) |c, coef| {
                    if (coef != 0) acc += coef * d.value(c, self.ds, r);
                }
                self.out[r * k + cls] = acc;
            };
            return;
        }
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
    /// How the fit ended. Always check `fit.stalled()`: a stalled solve returns
    /// ordinary-looking coefficients that solve nothing.
    fit: Fit = .{ .iters = 0 },
    score: f64,
    /// Nanoseconds scoring validation, not fitting; see `booster.TrainResult.valid_ns`.
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
    // Softmax fits one block of coefficients and an intercept per class (the full,
    // over-parameterised multinomial form scikit-learn fits; the penalty makes it unique).
    const k = cfg.objective.width(cfg.num_class);

    const theta = try gpa.alloc(f64, (p + 1) * k);
    defer gpa.free(theta);
    @memset(theta, 0);

    const wf = try gpa.alloc(f32, p * k);
    defer gpa.free(wf);
    const z = try gpa.alloc(f32, ds.n_rows * k);
    defer gpa.free(z);
    const resid = try gpa.alloc(f32, ds.n_rows * k);
    defer gpa.free(resid);
    const loss_part = try gpa.alloc(f64, reduce_chunks);
    defer gpa.free(loss_part);
    const rsum_part = try gpa.alloc(f64, reduce_chunks * k);
    defer gpa.free(rsum_part);

    // Intercepts start at the base rate: log-odds of the label mean, the mean, or under softmax
    // each class's centred log share (as the boosted model starts).
    if (k > 1) {
        const priors = try gpa.alloc(f32, k);
        defer gpa.free(priors);
        booster.softmaxBaseW(ds.labels, ds.weights, priors[0..k]);
        for (0..k) |cls| theta[cls * (p + 1) + p] = priors[cls];
    } else {
        var sum: f64 = 0;
        var wsum: f64 = 0;
        for (ds.labels, 0..) |y, r| {
            const w: f64 = if (ds.weights.len != 0) ds.weights[r] else 1;
            sum += w * y;
            wsum += w;
        }
        const mean = sum / wsum;
        theta[p] = switch (cfg.objective) {
            .logistic => blk: {
                const q = std.math.clamp(mean, 1e-6, 1 - 1e-6);
                break :blk @log(q / (1 - q));
            },
            .squared_error => mean,
            .softmax => unreachable,
        };
    }

    const n_f: f64 = @floatFromInt(ds.n_rows);
    const chunks: usize = if (ds.n_rows >= reduce_parallel_min) reduce_chunks else 1;
    const tables = try design.valueTables(gpa, ds);
    defer Design.freeTables(gpa, tables);
    var pr = Problem{
        .pool = pool,
        .design = &design,
        .tables = tables,
        .ds = ds,
        .objective = cfg.objective,
        .scale_pos_weight = cfg.scale_pos_weight,
        // L1/L2 are stated for the summed loss; divide by n for the mean-loss gradient.
        .l2 = @as(f64, cfg.lambda) / n_f,
        .l1 = @as(f64, cfg.alpha) / n_f,
        .theta = theta,
        .k = k,
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
        // No line search, so no "search could not move" failure: `converged` stays
        // true. `stalled()` still checks the gradient came down; without that, adam
        // reported success on a coin-flip model with a badly scaled design.
        .adam => try fitAdam(gpa, &pr, cfg, log),
    };
    const epochs = fit.iters;

    const w = try gpa.alloc(f32, p * k);
    errdefer gpa.free(w);
    for (0..k) |cls| {
        for (w[cls * p ..][0..p], theta[cls * (p + 1) ..][0..p]) |*d, s| d.* = @floatCast(s);
    }
    const intercepts = try gpa.alloc(f32, if (k > 1) k else 0);
    errdefer gpa.free(intercepts);
    for (intercepts, 0..) |*b, cls| b.* = @floatCast(theta[cls * (p + 1) + p]);

    var model = Linear{
        .gpa = gpa,
        .design = design,
        .w = w,
        .intercept = if (k > 1) 0 else @floatCast(theta[p]),
        .objective = cfg.objective,
        .n_features = ds.n_features,
        .num_class = if (k > 1) cfg.num_class else 1,
        .intercepts = intercepts,
    };
    errdefer model.deinit();

    var score: f64 = std.math.nan(f64);
    var valid_ns: u64 = 0;
    if (valid) |v| {
        const wall0 = prof.now();
        const pred = try gpa.alloc(f32, v.n_rows * k);
        defer gpa.free(pred);
        model.predict(pool, v, pred);
        score = if (v.weights.len != 0) switch (cfg.objective) {
            .logistic => try metric.aucW(gpa, pred, v.labels, v.weights),
            .squared_error => metric.rmseW(pred, v.labels, v.weights),
            .softmax => metric.mloglossProbW(pred, v.labels, k, v.weights),
        } else switch (cfg.objective) {
            .logistic => try metric.auc(gpa, pred, v.labels),
            .squared_error => metric.rmse(pred, v.labels),
            .softmax => metric.mloglossProb(pred, v.labels, k),
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
