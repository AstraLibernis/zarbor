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
//! Fitting is full-batch Adam with an L2 term folded into the gradient and L1
//! applied as a proximal soft-threshold after each step, so `alpha` genuinely
//! produces zeros rather than merely small coefficients.

const std = @import("std");
const Pool = @import("pool.zig").Pool;
const data = @import("data.zig");
const Dataset = data.Dataset;
const config = @import("config.zig");
const prof = @import("prof.zig");
const metric = @import("metric.zig");

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

/// The value a numeric bin stands for: the midpoint of its edges, or the one
/// finite edge for the two unbounded end bins.
fn binValue(edges: []const f32, real_bin: usize) f32 {
    if (edges.len == 0) return 0;
    if (real_bin == 0) return edges[0];
    if (real_bin >= edges.len) return edges[edges.len - 1];
    return 0.5 * (edges[real_bin - 1] + edges[real_bin]);
}

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
                // Bin 0 is missing; give it the feature's mean so a missing
                // entry sits at the centre and contributes nothing once
                // standardised.
                const table = try gpa.alloc(f32, nb);
                repr[f] = table;
                var sum: f64 = 0;
                var seen: usize = 0;
                for (1..nb) |b| {
                    table[b] = binValue(ds.edges[f], b - 1);
                    sum += table[b];
                    seen += 1;
                }
                table[0] = if (seen == 0) 0 else @floatCast(sum / @as(f64, @floatFromInt(seen)));
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
    objective: config.Objective,
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

/// Per-row derivative of the loss with respect to the linear predictor.
const ResidCtx = struct {
    z: []const f32,
    labels: []const f32,
    resid: []f32,
    objective: config.Objective,
    scale_pos_weight: f32,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *ResidCtx = @ptrCast(@alignCast(ctx));
        var i = begin;
        while (i < end) : (i += 1) {
            const y = self.labels[i];
            switch (self.objective) {
                .logistic => {
                    const p = 1.0 / (1.0 + @exp(-self.z[i]));
                    const w: f32 = if (y > 0.5) self.scale_pos_weight else 1.0;
                    self.resid[i] = w * (p - y);
                },
                .squared_error => self.resid[i] = self.z[i] - y,
            }
        }
    }
};

/// One dot product per design column. Column-major bins make this sequential.
const GradCtx = struct {
    design: *const Design,
    ds: *const Dataset,
    resid: []const f32,
    grad: []f32,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *GradCtx = @ptrCast(@alignCast(ctx));
        var c = begin;
        while (c < end) : (c += 1) {
            const col = self.design.cols[c];
            var acc: f64 = 0;
            for (self.resid, 0..) |r, row| acc += @as(f64, r) * self.design.value(col, self.ds, row);
            self.grad[c] = @floatCast(acc);
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

pub const TrainResult = struct {
    model: Linear,
    epochs: u32,
    score: f64,
    /// Nanoseconds spent scoring the validation set; not part of fitting.
    /// See `booster.TrainResult.valid_ns`.
    valid_ns: u64,
};

inline fn softThreshold(v: f32, t: f32) f32 {
    if (t == 0) return v;
    if (v > t) return v - t;
    if (v < -t) return v + t;
    return 0;
}

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

    const w = try gpa.alloc(f32, p);
    errdefer gpa.free(w);
    @memset(w, 0);

    const grad = try gpa.alloc(f32, p);
    defer gpa.free(grad);
    const m1 = try gpa.alloc(f32, p);
    defer gpa.free(m1);
    const v1 = try gpa.alloc(f32, p);
    defer gpa.free(v1);
    @memset(m1, 0);
    @memset(v1, 0);

    const z = try gpa.alloc(f32, ds.n_rows);
    defer gpa.free(z);
    const resid = try gpa.alloc(f32, ds.n_rows);
    defer gpa.free(resid);

    // Start the intercept at the base rate so the first steps do not have to
    // travel there; for logistic that is the log-odds of the label mean.
    var sum: f64 = 0;
    for (ds.labels) |y| sum += y;
    const mean = sum / @as(f64, @floatFromInt(ds.labels.len));
    var intercept: f32 = switch (cfg.objective) {
        .logistic => blk: {
            const q = std.math.clamp(mean, 1e-6, 1 - 1e-6);
            break :blk @floatCast(@log(q / (1 - q)));
        },
        .squared_error => @floatCast(mean),
    };

    const n_f: f32 = @floatFromInt(ds.n_rows);
    const beta1: f32 = 0.9;
    const beta2: f32 = 0.999;
    const eps: f32 = 1e-8;
    // L1/L2 are specified per the whole objective, so divide by n to match the
    // mean-loss gradient the steps are taken against.
    const l2: f32 = cfg.lambda / n_f;
    const l1: f32 = cfg.alpha / n_f;

    var epoch: u32 = 0;
    while (epoch < cfg.lin_epochs) : (epoch += 1) {
        var sctx = ScoreAllCtx{ .design = &design, .ds = ds, .w = w, .intercept = intercept, .z = z };
        pool.parallelFor(ds.n_rows, &sctx, ScoreAllCtx.run, 1024);

        var rctx = ResidCtx{
            .z = z,
            .labels = ds.labels,
            .resid = resid,
            .objective = cfg.objective,
            .scale_pos_weight = cfg.scale_pos_weight,
        };
        pool.parallelFor(ds.n_rows, &rctx, ResidCtx.run, 8192);

        var gctx = GradCtx{ .design = &design, .ds = ds, .resid = resid, .grad = grad };
        pool.parallelFor(p, &gctx, GradCtx.run, 1);

        const t: f32 = @floatFromInt(epoch + 1);
        const bc1 = 1.0 - std.math.pow(f32, beta1, t);
        const bc2 = 1.0 - std.math.pow(f32, beta2, t);

        var max_delta: f32 = 0;
        for (w, grad, m1, v1) |*coef, g_raw, *mm, *vv| {
            // L2 belongs in the gradient; L1 is applied as a prox step below
            // so it can land exactly on zero.
            const g = g_raw / n_f + l2 * coef.*;
            mm.* = beta1 * mm.* + (1 - beta1) * g;
            vv.* = beta2 * vv.* + (1 - beta2) * g * g;
            const step = cfg.lin_lr * (mm.* / bc1) / (@sqrt(vv.* / bc2) + eps);
            const before = coef.*;
            coef.* = softThreshold(coef.* - step, cfg.lin_lr * l1);
            max_delta = @max(max_delta, @abs(coef.* - before));
        }

        // The intercept is deliberately unpenalised.
        var rsum: f64 = 0;
        for (resid) |r| rsum += r;
        intercept -= cfg.lin_lr * @as(f32, @floatCast(rsum / @as(f64, n_f)));

        if (log) |wr| {
            if (cfg.verbose_eval != 0 and
                (epoch % (cfg.verbose_eval * 10) == 0 or epoch + 1 == cfg.lin_epochs))
            {
                try wr.print("[{d:>4}] |dw|max={e:.3}\n", .{ epoch, max_delta });
                try wr.flush();
            }
        }
        if (max_delta < cfg.lin_tol) {
            epoch += 1;
            break;
        }
    }

    var model = Linear{
        .gpa = gpa,
        .design = design,
        .w = w,
        .intercept = intercept,
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

    return .{ .model = model, .epochs = epoch, .score = score, .valid_ns = valid_ns };
}
