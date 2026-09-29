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
const config = @import("config.zig");
const prof = @import("prof.zig");
const metric = @import("metric.zig");
const Objective = @import("objective.zig").Objective;

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

// ----------------------------------------------------------------- fitting

/// Fixed chunk count for the reductions that feed the line search. Fixed, not
/// one per thread: a backtracking step is accepted or rejected on a
/// comparison between two objective values, so the sum has to come out
/// bit-identical however the pool happens to hand the chunks out.
const reduce_chunks: usize = 64;
/// Below this row count the reduction is cheaper inline than across the pool.
const reduce_parallel_min: usize = 8192;

/// Per-row loss, its derivative with respect to the linear predictor, and the
/// intercept's share of the gradient — one pass, because the line search runs
/// this far more often than it runs anything else.
///
/// `begin`/`end` index chunks, not rows, so a chunk always covers the same
/// rows and the partial sums always combine in the same order.
const EvalCtx = struct {
    z: []const f32,
    labels: []const f32,
    resid: []f32,
    loss: []f64,
    rsum: []f64,
    objective: Objective,
    scale_pos_weight: f32,
    want_loss: bool,
    size: usize,
    n: usize,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *EvalCtx = @ptrCast(@alignCast(ctx));
        var c = begin;
        while (c < end) : (c += 1) {
            const lo = @min(c * self.size, self.n);
            const hi = @min(lo + self.size, self.n);
            var l: f64 = 0;
            var rs: f64 = 0;
            var i = lo;
            while (i < hi) : (i += 1) {
                const y: f64 = self.labels[i];
                const zi: f64 = self.z[i];
                var r: f64 = undefined;
                switch (self.objective) {
                    .logistic => {
                        const wt: f64 = if (y > 0.5) self.scale_pos_weight else 1.0;
                        r = wt * (1.0 / (1.0 + @exp(-zi)) - y);
                        if (self.want_loss) {
                            // log(1+e^z) - y*z, pivoted on the sign of z so
                            // neither exponential can overflow.
                            const sp = if (zi > 0)
                                zi + @log(1.0 + @exp(-zi))
                            else
                                @log(1.0 + @exp(zi));
                            l += wt * (sp - y * zi);
                        }
                    },
                    .squared_error => {
                        r = zi - y;
                        if (self.want_loss) l += 0.5 * r * r;
                    },
                }
                self.resid[i] = @floatCast(r);
                rs += r;
            }
            self.loss[c] = l;
            self.rsum[c] = rs;
        }
    }
};

/// One dot product per design column. Column-major bins make this sequential.
const GradCtx = struct {
    design: *const Design,
    ds: *const Dataset,
    resid: []const f32,
    grad: []f64,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *GradCtx = @ptrCast(@alignCast(ctx));
        var c = begin;
        while (c < end) : (c += 1) {
            const col = self.design.cols[c];
            var acc: f64 = 0;
            for (self.resid, 0..) |r, row| acc += @as(f64, r) * self.design.value(col, self.ds, row);
            self.grad[c] = acc;
        }
    }
};

const ScoreAllCtx = struct {
    design: *const Design,
    ds: *const Dataset,
    w: []const f32,
    intercept: f32,
    z: []f32,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *ScoreAllCtx = @ptrCast(@alignCast(ctx));
        var r = begin;
        while (r < end) : (r += 1) {
            var acc: f32 = self.intercept;
            for (self.design.cols, self.w) |c, coef| {
                if (coef != 0) acc += coef * self.design.value(c, self.ds, r);
            }
            self.z[r] = acc;
        }
    }
};

/// The objective both solvers minimise:
///
///     f(w, b) = (1/n) Σ lossᵢ  +  ½·l2·‖w‖²  +  l1·‖w‖₁
///
/// with `l2 = lambda/n` and `l1 = alpha/n`, so `lambda` and `alpha` are stated
/// against the *summed* loss — the same convention the tree models use, and
/// the same one as scikit-learn's `C = 1/lambda`. The intercept is the last
/// slot of the parameter vector and is in neither penalty.
///
/// Parameters are f64 even though the model stores f32: L-BFGS measures
/// progress by differencing successive parameter vectors, and in f32 those
/// differences are rounding noise long before the solver is done.
pub const Problem = struct {
    pool: *Pool,
    design: *const Design,
    ds: *const Dataset,
    objective: Objective,
    scale_pos_weight: f32,
    l2: f64,
    l1: f64,
    /// `p` coefficients, then the intercept.
    theta: []f64,
    /// f32 mirror of `theta[0..p]`, which is what the scoring kernel reads.
    wf: []f32,
    z: []f32,
    resid: []f32,
    loss_part: []f64,
    rsum_part: []f64,
    chunks: usize,
    size: usize,

    inline fn p(pr: *const Problem) usize {
        return pr.theta.len - 1;
    }

    /// Smooth part of the objective at `th`. Leaves `resid` and the
    /// intercept's gradient share behind, so a following `grad` call costs
    /// only the column pass.
    fn value(pr: *Problem, th: []const f64, want_loss: bool) f64 {
        const np = pr.p();
        for (pr.wf, th[0..np]) |*d, s| d.* = @floatCast(s);

        var sctx = ScoreAllCtx{
            .design = pr.design,
            .ds = pr.ds,
            .w = pr.wf,
            .intercept = @floatCast(th[np]),
            .z = pr.z,
        };
        pr.pool.parallelFor(pr.ds.n_rows, &sctx, ScoreAllCtx.run, 1024);

        var ectx = EvalCtx{
            .z = pr.z,
            .labels = pr.ds.labels,
            .resid = pr.resid,
            .loss = pr.loss_part,
            .rsum = pr.rsum_part,
            .objective = pr.objective,
            .scale_pos_weight = pr.scale_pos_weight,
            .want_loss = want_loss,
            .size = pr.size,
            .n = pr.ds.n_rows,
        };
        pr.pool.parallelFor(pr.chunks, &ectx, EvalCtx.run, 1);

        if (!want_loss) return 0;
        var l: f64 = 0;
        for (pr.loss_part[0..pr.chunks]) |x| l += x;
        l /= @floatFromInt(pr.ds.n_rows);
        var sq: f64 = 0;
        for (th[0..np]) |c| sq += c * c;
        return l + 0.5 * pr.l2 * sq;
    }

    /// Gradient of the smooth part. Reads the `resid` the last `value` call
    /// left behind, so the two must be called in that order on the same point.
    fn grad(pr: *Problem, th: []const f64, out: []f64) void {
        const np = pr.p();
        var gctx = GradCtx{ .design = pr.design, .ds = pr.ds, .resid = pr.resid, .grad = out[0..np] };
        pr.pool.parallelFor(np, &gctx, GradCtx.run, 1);

        const n: f64 = @floatFromInt(pr.ds.n_rows);
        for (out[0..np], th[0..np]) |*d, c| d.* = d.* / n + pr.l2 * c;
        var rs: f64 = 0;
        for (pr.rsum_part[0..pr.chunks]) |x| rs += x;
        out[np] = rs / n;
    }

    fn l1norm(pr: *const Problem, th: []const f64) f64 {
        if (pr.l1 == 0) return 0;
        var s: f64 = 0;
        for (th[0..pr.p()]) |c| s += @abs(c);
        return pr.l1 * s;
    }
};

fn dot(a: []const f64, b: []const f64) f64 {
    var s: f64 = 0;
    for (a, b) |x, y| s += x * y;
    return s;
}

/// y += a·x
fn axpy(y: []f64, x: []const f64, a: f64) void {
    for (y, x) |*d, s| d.* += a * s;
}

// --------------------------------------------------------------- L-BFGS

/// Curvature pairs kept. Ten is the usual choice: the extra pairs buy little
/// and each costs two vectors the length of the parameter vector.
const lbfgs_history: usize = 10;
const ls_max: usize = 40;
const armijo_c1: f64 = 1e-4;
/// A gradient this small is below what an f32-scored design can resolve, so
/// treat it as converged rather than let a tight `lin_tol` spin on rounding.
const grad_floor: f64 = 1e-12;
/// Reject a curvature pair whose sᵀy is this small: it would make ρ enormous
/// and the next direction meaningless. Skipping keeps the older pairs.
const curvature_floor: f64 = 1e-12;

/// OWL-QN's choice of subgradient — the one of least magnitude, which at a
/// coefficient sitting on zero is zero unless the smooth gradient is steep
/// enough to push it off. Equals the plain gradient when there is no L1 term.
pub fn pseudoGrad(pr: *const Problem, g: []const f64, out: []f64) void {
    if (pr.l1 == 0) {
        @memcpy(out, g);
        return;
    }
    const np = pr.p();
    for (out[0..np], g[0..np], pr.theta[0..np]) |*o, gv, c| {
        o.* = if (c > 0)
            gv + pr.l1
        else if (c < 0)
            gv - pr.l1
        else if (gv + pr.l1 < 0)
            gv + pr.l1
        else if (gv - pr.l1 > 0)
            gv - pr.l1
        else
            0;
    }
    out[np] = g[np]; // the intercept is unpenalised
}

/// Clip a trial point back into the orthant the step started from, so the L1
/// term stays differentiable along the whole step. A coefficient that tried to
/// cross zero lands exactly on zero, which is how OWL-QN delivers the sparsity
/// an L1 penalty is asked for.
pub fn projectOrthant(t: []f64, from: []const f64, pg: []const f64) void {
    const np = t.len - 1;
    for (t[0..np], from[0..np], pg[0..np]) |*v, o, s| {
        const orthant = if (o != 0) o else -s;
        if (v.* * orthant <= 0) v.* = 0;
    }
}

/// Limited-memory BFGS: approximate the inverse Hessian from the last few
/// steps and gradient differences, so the step direction already carries the
/// problem's curvature and no learning rate has to be guessed. `alpha` > 0
/// switches on OWL-QN, which is the same recursion with the subgradient and
/// orthant projection above.
/// How a fit ended. `converged` means the coefficients stopped moving *and*
/// the gradient actually came down; a fit that stops moving with a large
/// gradient has stalled, which is a different thing and used to be reported
/// as success.
pub const Fit = struct {
    iters: u32,
    /// Pseudo-gradient infinity-norm at the first iteration and the last.
    g_first: f64 = 0,
    g_last: f64 = 0,
    converged: bool = true,

    /// The gradient vanishes at an optimum whatever the variables are scaled
    /// by, so its fall from where it started is the scale-free way to ask
    /// whether a fit arrived. Measured here: a healthy run ends four to five
    /// orders down (1.4e-5 of its starting value); a run stalled by bad
    /// conditioning ends three (3e-3). 1e-3 sits between them.
    pub fn stalled(f: Fit) bool {
        return !f.converged or f.g_last > 1e-3 * f.g_first;
    }
};

fn fitLbfgs(
    gpa: std.mem.Allocator,
    pr: *Problem,
    cfg: config.Config,
    log: ?*std.Io.Writer,
) !Fit {
    var fit: Fit = .{ .iters = 0 };
    const n = pr.theta.len;
    const m = @min(lbfgs_history, cfg.lin_epochs);

    const g = try gpa.alloc(f64, n);
    defer gpa.free(g);
    const pg = try gpa.alloc(f64, n);
    defer gpa.free(pg);
    const dir = try gpa.alloc(f64, n);
    defer gpa.free(dir);
    const trial = try gpa.alloc(f64, n);
    defer gpa.free(trial);
    const hs = try gpa.alloc(f64, m * n);
    defer gpa.free(hs);
    const hy = try gpa.alloc(f64, m * n);
    defer gpa.free(hy);
    const rho = try gpa.alloc(f64, m);
    defer gpa.free(rho);
    const alph = try gpa.alloc(f64, m);
    defer gpa.free(alph);

    var f = pr.value(pr.theta, true) + pr.l1norm(pr.theta);
    pr.grad(pr.theta, g);

    var stored: usize = 0; // curvature pairs held
    var head: usize = 0; // slot the next pair goes in
    var iter: u32 = 0;
    while (iter < cfg.lin_epochs) : (iter += 1) {
        pseudoGrad(pr, g, pg);
        var gmax: f64 = 0;
        for (pg) |v| gmax = @max(gmax, @abs(v));
        if (iter == 0) fit.g_first = gmax;
        fit.g_last = gmax;
        if (gmax <= grad_floor) break;

        // Two-loop recursion: dir ← H·pg, newest pair first on the way down
        // and oldest first on the way back up.
        @memcpy(dir, pg);
        for (0..stored) |i| {
            const k = (head + m - 1 - i) % m;
            alph[k] = rho[k] * dot(hs[k * n ..][0..n], dir);
            axpy(dir, hy[k * n ..][0..n], -alph[k]);
        }
        if (stored != 0) {
            // Scale the identity the recursion starts from by the last pair's
            // curvature; without this the first step of every iteration is
            // the wrong size by orders of magnitude.
            const k = (head + m - 1) % m;
            const y = hy[k * n ..][0..n];
            const yy = dot(y, y);
            if (yy > 0) {
                const gamma = dot(hs[k * n ..][0..n], y) / yy;
                for (dir) |*v| v.* *= gamma;
            }
        }
        for (0..stored) |i| {
            const k = (head + m - stored + i) % m;
            const beta = rho[k] * dot(hy[k * n ..][0..n], dir);
            axpy(dir, hs[k * n ..][0..n], alph[k] - beta);
        }
        for (dir) |*v| v.* = -v.*;

        // The quasi-Newton step may leave the current orthant; the components
        // that would are dropped rather than followed. Untested on its own,
        // and not for want of trying: `projectOrthant` clips the trial point
        // back regardless, so removing this changes how fast the solver gets
        // there and not where it lands. Kept because it is the formulation
        // OWL-QN is stated in and the projection alone is a weaker guarantee.
        if (pr.l1 != 0) {
            for (dir[0 .. n - 1], pg[0 .. n - 1]) |*dv, s| {
                if (dv.* * -s <= 0) dv.* = 0;
            }
        }

        var dg = dot(dir, pg);
        if (!(dg < 0)) {
            // A stale history, or every component clipped away. Fall back to
            // steepest descent and start the history over.
            for (dir, pg) |*dv, s| dv.* = -s;
            dg = dot(dir, pg);
            stored = 0;
            head = 0;
            if (!(dg < 0)) break;
        }

        // With no curvature yet, unit length is arbitrary; scale by the
        // gradient so the first trial point is not absurdly far away.
        var step: f64 = if (stored == 0) @min(1.0, 1.0 / gmax) else 1.0;
        var f_new: f64 = f;
        var accepted = false;
        for (0..ls_max) |_| {
            for (trial, pr.theta, dir) |*t, th, dv| t.* = th + step * dv;
            if (pr.l1 != 0) projectOrthant(trial, pr.theta, pg);
            f_new = pr.value(trial, true) + pr.l1norm(trial);
            // After projection the step actually taken is not `step * dir`,
            // so Armijo has to be measured against the realised move.
            var expected = step * dg;
            if (pr.l1 != 0) {
                expected = 0;
                for (pg, trial, pr.theta) |s, t, th| expected += s * (t - th);
            }
            if (f_new <= f + armijo_c1 * expected) {
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        // No improvement anywhere along the ray. On a well-scaled problem
        // that means f32 scoring noise now dominates the objective; on a
        // badly-scaled one it means the line search cannot move at all.
        // `stalled` tells those apart by looking at the gradient.
        if (!accepted) {
            fit.converged = false;
            break;
        }

        var dmax: f64 = 0;
        const s_slot = hs[head * n ..][0..n];
        for (s_slot, trial, pr.theta) |*sv, t, th| {
            sv.* = t - th;
            dmax = @max(dmax, @abs(sv.*));
        }
        @memcpy(pr.theta, trial);
        f = f_new;

        // `value` left `resid` at the accepted point, so this is its gradient.
        const y_slot = hy[head * n ..][0..n];
        @memcpy(y_slot, g);
        pr.grad(pr.theta, g);
        for (y_slot, g) |*yv, gv| yv.* = gv - yv.*;

        const sy = dot(s_slot, y_slot);
        if (sy > curvature_floor) {
            rho[head] = 1.0 / sy;
            head = (head + 1) % m;
            if (stored < m) stored += 1;
        }

        if (log) |wr| {
            if (cfg.verbose_eval != 0 and
                (iter % cfg.verbose_eval == 0 or iter + 1 == cfg.lin_epochs))
            {
                try wr.print("[{d:>4}] f={d:.8} |g|max={e:.3}\n", .{ iter, f, gmax });
                try wr.flush();
            }
        }
        if (dmax < cfg.lin_tol) {
            iter += 1;
            break;
        }
    }
    fit.iters = iter;
    return fit;
}

// ----------------------------------------------------------------- Adam

inline fn softThreshold(v: f64, t: f64) f64 {
    if (t == 0) return v;
    if (v > t) return v - t;
    if (v < -t) return v + t;
    return 0;
}

/// Full-batch Adam with the L1 term applied as a proximal soft-threshold after
/// each step, so `alpha` genuinely produces zeros rather than merely small
/// coefficients. Needs no objective evaluation and so no line search, at the
/// cost of an order of magnitude more passes than `lbfgs` for the same
/// accuracy.
fn fitAdam(
    gpa: std.mem.Allocator,
    pr: *Problem,
    cfg: config.Config,
    log: ?*std.Io.Writer,
) !Fit {
    const n = pr.theta.len;
    const np = n - 1;

    const g = try gpa.alloc(f64, n);
    defer gpa.free(g);
    // Same quantity `fitLbfgs` records, so the 1e-3 ratio in `Fit.stalled()`
    // is calibrated for both. Adam takes a proximal step rather than using
    // the pseudo-gradient to move, but "how far did the gradient fall" is a
    // question about the problem, not about the step rule.
    const pg = try gpa.alloc(f64, n);
    defer gpa.free(pg);
    var fit: Fit = .{ .iters = 0 };
    const m1 = try gpa.alloc(f64, np);
    defer gpa.free(m1);
    const v1 = try gpa.alloc(f64, np);
    defer gpa.free(v1);
    @memset(m1, 0);
    @memset(v1, 0);

    const beta1: f64 = 0.9;
    const beta2: f64 = 0.999;
    const eps: f64 = 1e-8;
    const lr: f64 = cfg.lin_lr;

    var epoch: u32 = 0;
    while (epoch < cfg.lin_epochs) : (epoch += 1) {
        _ = pr.value(pr.theta, false);
        pr.grad(pr.theta, g);

        pseudoGrad(pr, g, pg);
        var gmax: f64 = 0;
        for (pg) |v| gmax = @max(gmax, @abs(v));
        if (epoch == 0) fit.g_first = gmax;
        fit.g_last = gmax;

        const t: f64 = @floatFromInt(epoch + 1);
        const bc1 = 1.0 - std.math.pow(f64, beta1, t);
        const bc2 = 1.0 - std.math.pow(f64, beta2, t);

        var max_delta: f64 = 0;
        for (pr.theta[0..np], g[0..np], m1, v1) |*coef, gv, *mm, *vv| {
            mm.* = beta1 * mm.* + (1 - beta1) * gv;
            vv.* = beta2 * vv.* + (1 - beta2) * gv * gv;
            const step = lr * (mm.* / bc1) / (@sqrt(vv.* / bc2) + eps);
            const before = coef.*;
            coef.* = softThreshold(coef.* - step, lr * pr.l1);
            max_delta = @max(max_delta, @abs(coef.* - before));
        }
        // The intercept is deliberately unpenalised, and takes a plain step.
        pr.theta[np] -= lr * g[np];

        if (log) |wr| {
            if (cfg.verbose_eval != 0 and
                (epoch % (cfg.verbose_eval * 10) == 0 or epoch + 1 == cfg.lin_epochs))
            {
                try wr.print("[{d:>4}] |dw|max={e:.3}  |g|max={e:.3}\n", .{ epoch, max_delta, gmax });
                try wr.flush();
            }
        }
        if (max_delta < cfg.lin_tol) {
            epoch += 1;
            break;
        }
    }
    fit.iters = epoch;
    return fit;
}

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
    cfg: config.Config,
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
