// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! The loss every model minimises.

/// Loss function, dispatched at comptime so the boosting loop inlines the
/// gradient/hessian pair with no indirect call in the hot path.
/// With `--algo=linear` this picks the model: `logistic` = LOGISTIC
/// REGRESSION (binary classifier), `squared_error` = LINEAR REGRESSION (ridge
/// at the default `linear.Params.lambda`; OLS only with `lambda` and `alpha` 0).
/// Same machinery (weighted feature sum), differing in loss and sigmoid link.
/// `Config.modelName` prints the name at train time; see docs/linear-solvers.md.
pub const Objective = enum {
    /// Binary logistic / cross-entropy; raw = log-odds, prediction = sigmoid(raw).
    /// With `--algo=linear`: scikit-learn's `LogisticRegression`.
    logistic,
    /// Squared error. Hessian is constant 1, so `adam` converges in far fewer
    /// epochs than logistic needs (docs/linear-solvers.md).
    /// With `--algo=linear`: scikit-learn's `Ridge` while `lambda > 0` (the
    /// default), `Lasso` once `alpha > 0`, elastic net with both,
    /// `LinearRegression` with neither.
    squared_error,
};
