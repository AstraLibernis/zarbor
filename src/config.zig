//! Hyperparameter surface for every model in zmodels.
//!
//! Names follow XGBoost where an equivalent exists, so published tunings
//! transfer without a translation table. LightGBM-only knobs keep LightGBM's
//! names for the same reason.

const std = @import("std");

/// Which model to fit. All three share the binning, missing-value and
/// categorical handling in `data.zig`; they differ in what they do with it.
pub const Algo = enum {
    /// Gradient-boosted trees. `grow_policy` selects XGBoost- or
    /// LightGBM-style growth.
    gbdt,
    /// Bagged unshrunk trees, averaged. No boosting.
    random_forest,
    /// Regularised linear or logistic regression on the binned design matrix.
    linear,
};

/// Loss function. Dispatched at comptime so the boosting loop inlines the
/// exact gradient/hessian pair with no indirect call in the hot path.
pub const Objective = enum {
    /// Binary logistic. Raw scores are log-odds; predictions are sigmoid(raw).
    logistic,
    /// Squared error for regression. Hessian is constant 1.
    squared_error,
};

/// How a tree is expanded once its root histogram exists.
pub const GrowPolicy = enum {
    /// Expand all nodes at depth d before any at depth d+1 (XGBoost default).
    depthwise,
    /// Always split the leaf with the highest gain (LightGBM-style). Usually
    /// stronger per tree, and needs `max_leaves` rather than `max_depth` to
    /// control capacity.
    lossguide,
};

/// Strategy for choosing bin edges when quantising a numeric column.
pub const BinPolicy = enum {
    /// Equal-count bins from the empirical distribution. Robust to skew.
    quantile,
    /// Equal-width bins between min and max. Cheaper, worse on skewed data.
    uniform,
};

/// How rows are chosen for each tree.
pub const Sampling = enum {
    /// Uniform random subset of size `subsample`, without replacement.
    uniform,
    /// LightGBM's Gradient-based One-Side Sampling: keep every large-gradient
    /// row, sample the rest, and amplify the survivors so the gradient sum
    /// stays unbiased. Ignores `subsample`.
    goss,
};

/// How the linear model's coefficients are fitted.
pub const LinSolver = enum {
    /// Limited-memory BFGS, with OWL-QN's orthant handling when `alpha` > 0.
    /// Builds a curvature estimate from recent steps, so it reaches the
    /// optimum in tens of passes where a first-order method needs thousands.
    /// This is also what scikit-learn's LogisticRegression defaults to, which
    /// makes the two directly comparable.
    lbfgs,
    /// Full-batch Adam with an L1 proximal step. Needs no objective
    /// evaluation and so no line search, which makes each pass cheaper — but
    /// it takes far more of them to reach the same coefficients.
    adam,
};

pub const Config = struct {
    // ---- which model -------------------------------------------------
    algo: Algo = .gbdt,

    // ---- ensemble ----------------------------------------------------
    /// Number of boosting rounds (trees) to fit. For `random_forest` this is
    /// the number of bagged trees, fitted independently.
    n_rounds: u32 = 500,
    /// Shrinkage applied to each tree's leaf values. XGBoost's `eta`.
    /// Forced to 1.0 for `random_forest`, which averages instead of shrinking.
    learning_rate: f32 = 0.1,
    /// Initial raw score for every row. For logistic this is a log-odds.
    /// Null means "derive from the training label mean", which is what you
    /// almost always want.
    base_score: ?f32 = null,

    // ---- tree shape --------------------------------------------------
    grow_policy: GrowPolicy = .depthwise,
    /// Hard depth cap. 0 disables the cap, which is only safe when
    /// `max_leaves` is set — otherwise the histogram budget is unbounded.
    max_depth: u32 = 6,
    /// Leaf cap. 0 disables.
    max_leaves: u32 = 0,
    /// Minimum number of training rows that must land in a leaf.
    min_child_samples: u32 = 20,
    /// Minimum summed hessian in a leaf. For logistic this is a measure of
    /// confidence mass, not row count, and is the more principled of the two.
    min_child_weight: f32 = 1.0,

    // ---- regularisation ----------------------------------------------
    /// L2 penalty on leaf weights. Appears as lambda in the gain formula.
    lambda: f32 = 1.0,
    /// L1 penalty on leaf weights, applied by soft-thresholding the gradient.
    alpha: f32 = 0.0,
    /// Minimum gain required to accept a split. XGBoost's `gamma`.
    min_split_gain: f32 = 0.0,
    /// Cap on the absolute leaf weight. 0 disables. Stabilises logistic loss
    /// on heavily imbalanced data.
    max_delta_step: f32 = 0.0,

    // ---- sampling ----------------------------------------------------
    /// Row-selection strategy. `goss` is LightGBM's; `uniform` is XGBoost's.
    sampling: Sampling = .uniform,
    /// Fraction of rows sampled per tree. Without replacement unless
    /// `bootstrap` is set. Ignored when `sampling = .goss`.
    subsample: f32 = 1.0,
    /// Sample rows *with* replacement. This is what makes a bagged ensemble a
    /// random forest rather than a subsample ensemble; on by default for
    /// `random_forest` and off for everything else.
    bootstrap: bool = false,
    /// GOSS: fraction of rows kept for having the largest |gradient|.
    top_rate: f32 = 0.2,
    /// GOSS: fraction of the *remaining* rows sampled uniformly.
    other_rate: f32 = 0.1,
    /// Fraction of features sampled once per tree.
    colsample_bytree: f32 = 1.0,
    /// Fraction of the per-tree features sampled again at each depth level.
    colsample_bylevel: f32 = 1.0,
    /// Fraction of the per-level features sampled again at each split.
    colsample_bynode: f32 = 1.0,

    // ---- binning -----------------------------------------------------
    bin_policy: BinPolicy = .quantile,
    /// Bins per feature. Capped at 256 because bins are stored as u8, which
    /// is what keeps the feature matrix inside L3.
    max_bin: u16 = 256,

    // ---- objective ---------------------------------------------------
    objective: Objective = .logistic,
    /// Multiplier on positive-class gradients. >1 upweights the minority
    /// class for imbalanced binary problems.
    scale_pos_weight: f32 = 1.0,

    // ---- linear model ------------------------------------------------
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

    // ---- control -----------------------------------------------------
    /// Stop when the validation metric has not improved for this many rounds.
    /// 0 disables early stopping. Not meaningful for `random_forest`, whose
    /// trees are independent, or for `linear`.
    early_stopping_rounds: u32 = 0,
    seed: u64 = 0,
    /// Worker threads. 0 means "one per logical core".
    n_threads: u32 = 0,
    /// Print per-round metrics every N rounds. 0 silences training.
    verbose_eval: u32 = 10,

    /// Apply the defaults that define each algorithm, for any field the user
    /// left at the shared default. Call once, before `validate`.
    pub fn applyAlgoDefaults(c: *Config, explicit: []const []const u8) void {
        const set = struct {
            fn has(ex: []const []const u8, name: []const u8) bool {
                for (ex) |e| if (std.mem.eql(u8, e, name)) return true;
                return false;
            }
        };
        switch (c.algo) {
            .gbdt, .linear => {},
            .random_forest => {
                // A forest averages full-strength trees; shrinking them would
                // make it a very slow boosted model with no boosting.
                if (!set.has(explicit, "learning_rate")) c.learning_rate = 1.0;
                if (!set.has(explicit, "bootstrap")) c.bootstrap = true;
                // Forests want deep, low-bias trees; the ensemble average is
                // what controls variance. Leaf-capped so the histogram budget
                // stays bounded.
                if (!set.has(explicit, "max_depth")) c.max_depth = 0;
                if (!set.has(explicit, "max_leaves")) c.max_leaves = 1024;
                if (!set.has(explicit, "min_child_samples")) c.min_child_samples = 1;
                if (!set.has(explicit, "min_child_weight")) c.min_child_weight = 0.0;
                if (!set.has(explicit, "lambda")) c.lambda = 0.0;
                // sqrt(p) features per split is the classic Breiman default;
                // applied per node by the caller, which knows p.
                if (!set.has(explicit, "n_rounds")) c.n_rounds = 300;
            },
        }
    }

    /// Breiman's per-split feature default, which needs the feature count and
    /// so cannot be set until the data is binned. sqrt(p) for classification,
    /// p/3 for regression.
    pub fn applyForestFeatureDefault(
        c: *Config,
        n_features: usize,
        explicit: []const []const u8,
    ) void {
        if (c.algo != .random_forest) return;
        for (explicit) |e| if (std.mem.eql(u8, e, "colsample_bynode")) return;
        if (n_features == 0) return;
        const p_f: f32 = @floatFromInt(n_features);
        const want: f32 = switch (c.objective) {
            .logistic => @sqrt(p_f),
            .squared_error => p_f / 3.0,
        };
        c.colsample_bynode = std.math.clamp(want / p_f, 1.0 / p_f, 1.0);
    }

    pub fn validate(c: Config) !void {
        if (c.n_rounds == 0) return error.NoRounds;
        if (c.max_bin < 2 or c.max_bin > 256) return error.BadMaxBin;

        if (c.algo == .linear) {
            if (c.lin_epochs == 0) return error.NoEpochs;
            if (c.lin_lr <= 0) return error.BadLinearLr;
            if (c.lambda < 0 or c.alpha < 0) return error.NegativeRegularisation;
            return;
        }

        if (c.learning_rate <= 0 or c.learning_rate > 1) return error.BadLearningRate;
        if (c.subsample <= 0 or c.subsample > 1) return error.BadSubsample;
        if (c.colsample_bytree <= 0 or c.colsample_bytree > 1) return error.BadColsample;
        if (c.colsample_bylevel <= 0 or c.colsample_bylevel > 1) return error.BadColsample;
        if (c.colsample_bynode <= 0 or c.colsample_bynode > 1) return error.BadColsample;
        if (c.lambda < 0 or c.alpha < 0) return error.NegativeRegularisation;

        // Bagging with replacement makes a row's gradient contribute more
        // than once, which is meaningless when the next round's gradient is
        // computed from a single accumulated score per row.
        if (c.bootstrap and c.algo == .gbdt) return error.BootstrapWithBoosting;

        if (c.sampling == .goss) {
            if (c.top_rate <= 0 or c.top_rate >= 1) return error.BadTopRate;
            if (c.other_rate <= 0 or c.other_rate >= 1) return error.BadOtherRate;
            if (c.top_rate + c.other_rate > 1) return error.GossRatesExceedOne;
            if (c.bootstrap) return error.GossWithBootstrap;
        }

        // An unbounded tree is only safe if *something* caps its leaves.
        if (c.max_depth == 0 and c.max_leaves == 0) return error.UnboundedTree;
        if (c.grow_policy == .lossguide and c.max_leaves == 0 and c.max_depth == 0)
            return error.UnboundedLossguide;
    }

    /// Upper bound on leaves for allocation sizing.
    pub fn leafBudget(c: Config) u32 {
        if (c.max_leaves != 0) return c.max_leaves;
        // depthwise with a depth cap: 2^depth leaves, clamped to something sane.
        const d = @min(c.max_depth, 20);
        return @as(u32, 1) << @intCast(d);
    }
};
