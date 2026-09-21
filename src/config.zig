//! Hyperparameter surface for the gradient-boosted tree booster.
//!
//! Names follow XGBoost where an equivalent exists, so published tunings
//! transfer without a translation table.

const std = @import("std");

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

pub const Config = struct {
    // ---- ensemble ----------------------------------------------------
    /// Number of boosting rounds (trees) to fit.
    n_rounds: u32 = 500,
    /// Shrinkage applied to each tree's leaf values. XGBoost's `eta`.
    learning_rate: f32 = 0.1,
    /// Initial raw score for every row. For logistic this is a log-odds.
    /// Null means "derive from the training label mean", which is what you
    /// almost always want.
    base_score: ?f32 = null,

    // ---- tree shape --------------------------------------------------
    grow_policy: GrowPolicy = .depthwise,
    /// Hard depth cap. 0 disables the cap (only meaningful for lossguide).
    max_depth: u32 = 6,
    /// Leaf cap for lossguide. 0 disables.
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
    /// Fraction of rows sampled per tree, without replacement.
    subsample: f32 = 1.0,
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

    // ---- control -----------------------------------------------------
    /// Stop when the validation metric has not improved for this many rounds.
    /// 0 disables early stopping.
    early_stopping_rounds: u32 = 0,
    seed: u64 = 0,
    /// Worker threads. 0 means "one per logical core".
    n_threads: u32 = 0,
    /// Print per-round metrics every N rounds. 0 silences training.
    verbose_eval: u32 = 10,

    pub fn validate(c: Config) !void {
        if (c.n_rounds == 0) return error.NoRounds;
        if (c.learning_rate <= 0 or c.learning_rate > 1) return error.BadLearningRate;
        if (c.max_bin < 2 or c.max_bin > 256) return error.BadMaxBin;
        if (c.subsample <= 0 or c.subsample > 1) return error.BadSubsample;
        if (c.colsample_bytree <= 0 or c.colsample_bytree > 1) return error.BadColsample;
        if (c.colsample_bylevel <= 0 or c.colsample_bylevel > 1) return error.BadColsample;
        if (c.colsample_bynode <= 0 or c.colsample_bynode > 1) return error.BadColsample;
        if (c.lambda < 0 or c.alpha < 0) return error.NegativeRegularisation;
        if (c.grow_policy == .depthwise and c.max_depth == 0) return error.UnboundedDepthwise;
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
