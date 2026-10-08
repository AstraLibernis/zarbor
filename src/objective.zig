// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! The loss every model minimises.

/// Loss function, dispatched at comptime so the boosting loop inlines the
/// gradient/hessian pair with no indirect call in the hot path.
/// With `--algo=linear` this picks the model: `logistic` = LOGISTIC
/// REGRESSION (binary classifier), `squared_error` = LINEAR REGRESSION (ridge
/// at the default `linear.Params.lambda`; OLS only with `lambda` and `alpha` 0).
/// Same machinery (weighted feature sum), differing in loss and sigmoid link.
/// `Config.modelName` prints the name at train time; see docs/archive/linear-solvers.md.
pub const Objective = enum {
    /// Binary logistic / cross-entropy; raw = log-odds, prediction = sigmoid(raw).
    /// With `--algo=linear`: scikit-learn's `LogisticRegression`.
    logistic,
    /// Squared error. Hessian is constant 1, so `adam` converges in far fewer
    /// epochs than logistic needs (docs/archive/linear-solvers.md).
    /// With `--algo=linear`: scikit-learn's `Ridge` while `lambda > 0` (the
    /// default), `Lasso` once `alpha > 0`, elastic net with both,
    /// `LinearRegression` with neither.
    squared_error,
    /// Absolute error (L1), LightGBM's `regression_l1`: the gradient is the residual's sign and,
    /// after each tree, every leaf's value is the weighted median of its rows' residuals. Starts at
    /// the label median.
    absolute_error,
    /// Pinball (quantile) loss at `quantile_alpha`, LightGBM's `quantile`: as `absolute_error` with
    /// the alpha-percentile in place of the median. Predicts that percentile of the target.
    quantile,
    /// Pseudo-Huber loss with `huber_slope`, XGBoost's `reg:pseudohubererror`: squared error near 0,
    /// absolute error far out, smooth throughout.
    pseudo_huber,
    /// Poisson log-likelihood with a log link, XGBoost's `count:poisson`: for counts and rates.
    /// Raw score = log of the predicted mean; prediction = exp(raw). Labels must be >= 0.
    poisson,
    /// Multiclass softmax / cross-entropy over `num_class` classes (XGBoost `multi:softprob`,
    /// LightGBM `multiclass`). Each row has one raw score per class; prediction = softmax.
    /// Labels are class indices 0..num_class-1. Boosting grows one tree per class per round.
    softmax,

    /// Raw scores per row: one per class under softmax, else one.
    pub fn width(o: Objective, num_class: u32) usize {
        return if (o == .softmax) num_class else 1;
    }

    /// Whether each leaf's value is replaced after the tree is grown (by a residual percentile).
    pub fn renewsLeaves(o: Objective) bool {
        return o == .absolute_error or o == .quantile;
    }

    /// Whether predictions apply a link to the raw score (sigmoid, softmax, exp).
    pub fn hasLink(o: Objective) bool {
        return o == .logistic or o == .softmax or o == .poisson;
    }

    /// Whether the target is classes (stratified folds, class weights).
    pub fn classifies(o: Objective) bool {
        return o == .logistic or o == .softmax;
    }
};
