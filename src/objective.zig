// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! The loss every model minimises.

/// Loss function. Dispatched at comptime so the boosting loop inlines the
/// exact gradient/hessian pair with no indirect call in the hot path.
///
/// With `--algo=linear` this choice *is* the choice of model, and the two it
/// selects are the two standard simple models every library ships:
///
///     --objective=logistic        LOGISTIC REGRESSION   (binary classifier)
///     --objective=squared_error   LINEAR REGRESSION     (ordinary least squares)
///
/// They are the same machinery -- a weighted sum of features -- differing in
/// the loss and in whether a sigmoid link is applied to the output. See
/// `Config.modelName`, which prints the name at train time, and
/// docs/linear-solvers.md.
pub const Objective = enum {
    /// Binary logistic / cross-entropy. Raw scores are log-odds; predictions
    /// are sigmoid(raw).
    ///
    /// With `--algo=linear`: **logistic regression**, i.e. scikit-learn's
    /// `LogisticRegression`.
    logistic,
    /// Squared error for regression. Hessian is constant 1, which is why
    /// `adam` converges on it in a few hundred epochs where logistic needs
    /// tens of thousands.
    ///
    /// With `--algo=linear`: **linear regression** (ordinary least squares),
    /// i.e. scikit-learn's `LinearRegression` -- becoming `Ridge` once
    /// `lambda > 0`, `Lasso` once `alpha > 0`, and elastic net with both.
    squared_error,
};
