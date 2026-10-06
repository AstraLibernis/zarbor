// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Fitting the linear model: the penalised objective, L-BFGS (OWL-QN under L1) and Adam.

const std = @import("std");
const Pool = @import("pool.zig").Pool;
const data = @import("data.zig");
const Dataset = data.Dataset;
const Objective = @import("objective.zig").Objective;
const linear = @import("linear.zig");
const Design = linear.Design;
const Params = linear.Params;

// ----------------------------------------------------------------- fitting

/// Fixed chunk count for line-search reductions, not one per thread: a step is accepted
/// by comparing two objective sums, so they must be bit-identical however chunks are handed out.
pub const reduce_chunks: usize = 64;
/// Below this row count the reduction is cheaper inline than across the pool.
pub const reduce_parallel_min: usize = 8192;

/// Per-row loss, its derivative w.r.t. the linear predictor, and the intercept's gradient
/// share in one pass: the line search runs this most. `begin`/`end` index chunks, not
/// rows, so a chunk always covers the same rows and partial sums combine in the same order.
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
                            // log(1+e^z) - y*z, pivoted on sign(z) so no exp overflows.
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

/// Objective both solvers minimise: `f(w, b) = (1/n) Σ lossᵢ + ½·l2·‖w‖² + l1·‖w‖₁`,
/// `l2 = lambda/n`, `l1 = alpha/n`: `lambda`/`alpha` are stated against the *summed*
/// loss, like the tree models and scikit-learn's `C = 1/lambda`. The intercept is the
/// last parameter slot, in neither penalty. Parameters are f64 though the model stores
/// f32: L-BFGS differences successive parameter vectors, which in f32 are rounding
/// noise long before the solver is done.
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

    /// Smooth part of the objective at `th`. Leaves `resid` and the intercept's
    /// gradient share behind, so a following `grad` costs only the column pass.
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

/// Curvature pairs kept. Ten is usual: more buy little, each costs two parameter-length vectors.
const lbfgs_history: usize = 10;
const ls_max: usize = 40;
const armijo_c1: f64 = 1e-4;
/// A gradient this small is below what an f32-scored design can resolve, so
/// treat it as converged rather than let a tight `lin_tol` spin on rounding.
const grad_floor: f64 = 1e-12;
/// Reject a curvature pair whose sᵀy is this small: it would make ρ enormous
/// and the next direction meaningless. Skipping keeps the older pairs.
const curvature_floor: f64 = 1e-12;

/// OWL-QN's least-magnitude subgradient: zero at a coefficient on zero unless the
/// smooth gradient is steep enough to push it off. The plain gradient without L1.
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

/// Clip a trial point into the step's starting orthant so L1 stays differentiable along
/// it; a coefficient crossing zero lands exactly on zero, which is OWL-QN's sparsity.
pub fn projectOrthant(t: []f64, from: []const f64, pg: []const f64) void {
    const np = t.len - 1;
    for (t[0..np], from[0..np], pg[0..np]) |*v, o, s| {
        const orthant = if (o != 0) o else -s;
        if (v.* * orthant <= 0) v.* = 0;
    }
}

/// How a fit ended. `converged` is false only when L-BFGS's line search found no
/// improvement; stopping on `lin_tol`, the gradient floor or the `lin_epochs` cap
/// leaves it true. Whether the gradient came down is `stalled()`'s test: stopping
/// with a large gradient is a stall, not success.
pub const Fit = struct {
    iters: u32,
    /// Pseudo-gradient infinity-norm at the first iteration and the last.
    g_first: f64 = 0,
    g_last: f64 = 0,
    converged: bool = true,

    /// The gradient vanishes at an optimum under any scaling, so its fall is the
    /// scale-free arrival test. The 1e-3 ratio sits between where healthy runs end
    /// and where runs stalled by bad conditioning end.
    pub fn stalled(f: Fit) bool {
        return !f.converged or f.g_last > 1e-3 * f.g_first;
    }
};

/// Limited-memory BFGS: inverse Hessian from recent steps and gradient differences,
/// so the direction carries curvature and no learning rate is guessed. `alpha` > 0
/// is OWL-QN: same recursion with the subgradient and orthant projection above.
pub fn fitLbfgs(
    gpa: std.mem.Allocator,
    pr: *Problem,
    cfg: Params,
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

        // Two-loop recursion: dir ← H·pg, newest pair first down, oldest first back up.
        @memcpy(dir, pg);
        for (0..stored) |i| {
            const k = (head + m - 1 - i) % m;
            alph[k] = rho[k] * dot(hs[k * n ..][0..n], dir);
            axpy(dir, hy[k * n ..][0..n], -alph[k]);
        }
        if (stored != 0) {
            // Scale the starting identity by the last pair's curvature; without it
            // every iteration's first step is wrong by orders of magnitude.
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

        // Drop direction components that would leave the orthant. Not testable alone:
        // `projectOrthant` clips anyway, so removing this changes speed, not result.
        // Kept as OWL-QN's stated formulation; projection alone is a weaker guarantee.
        if (pr.l1 != 0) {
            for (dir[0 .. n - 1], pg[0 .. n - 1]) |*dv, s| {
                if (dv.* * -s <= 0) dv.* = 0;
            }
        }

        var dg = dot(dir, pg);
        if (!(dg < 0)) {
            // Stale history or all components clipped: steepest descent, reset history.
            for (dir, pg) |*dv, s| dv.* = -s;
            dg = dot(dir, pg);
            stored = 0;
            head = 0;
            if (!(dg < 0)) break;
        }

        // No curvature yet: scale by the gradient so the first trial is not absurdly far.
        var step: f64 = if (stored == 0) @min(1.0, 1.0 / gmax) else 1.0;
        var f_new: f64 = f;
        var accepted = false;
        for (0..ls_max) |_| {
            for (trial, pr.theta, dir) |*t, th, dv| t.* = th + step * dv;
            if (pr.l1 != 0) projectOrthant(trial, pr.theta, pg);
            f_new = pr.value(trial, true) + pr.l1norm(trial);
            // After projection the move is not `step * dir`; Armijo uses the realised one.
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
        // No improvement along the ray: f32 scoring noise dominates (well scaled) or the
        // search cannot move (badly scaled). `stalled` tells them apart by the gradient.
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

/// Full-batch Adam, L1 as a proximal soft-threshold after each step so `alpha` gives
/// exact zeros. No objective evaluation, so no line search, but an order of magnitude
/// more passes than `lbfgs` for the same accuracy.
pub fn fitAdam(
    gpa: std.mem.Allocator,
    pr: *Problem,
    cfg: Params,
    log: ?*std.Io.Writer,
) !Fit {
    const n = pr.theta.len;
    const np = n - 1;

    const g = try gpa.alloc(f64, n);
    defer gpa.free(g);
    // Same quantity `fitLbfgs` records, so `Fit.stalled()`'s 1e-3 ratio fits both;
    // gradient fall is a property of the problem, not of Adam's proximal step rule.
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
