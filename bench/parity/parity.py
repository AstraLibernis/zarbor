#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 AstraLibernis

"""Exact-tree parity: zarbor against XGBoost, LightGBM and CatBoost, on synthetic data.

Every feature takes at most six integer values, so every library gives each value its own bin,
and every partition is reachable on every side. With sampling off and the same starting score,
the same arithmetic must grow the same trees, so predictions must agree row for row. A
difference is a calculation difference, not a binning or tuning one.

The data is generated here from a fixed seed, so anyone can rerun it (the original harness ran
on competition data that cannot be redistributed; docs/archive/parity.md). Each case prints
the largest prediction difference and passes within 1e-6 times max(1, |prediction|): zarbor
writes 6 decimals and accumulates scores in f32, so a regression prediction near 2.4 can be
1.1e-6 off from rounding alone. Exit status 1 if any case fails.

    python bench/parity/parity.py              # every case
    python bench/parity/parity.py xgb_500 cat_ctr

Needs xgboost, lightgbm and catboost, and a ReleaseFast zarbor (`zig build -Doptimize=ReleaseFast`).
Results: docs/correctness.md section 1.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import catboost
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
Z = str(ROOT / "zig-out" / "bin" / "zarbor")
WORK = Path(tempfile.mkdtemp(prefix="zarbor-parity-"))
TOL = 1e-6
N = 100_000


def make_data():
    """13 correlated columns of ratings 0-5 and a binary target with interactions, so trees grow
    deep and every split matters; a regression target from the same columns; and three
    categoricals (2, 3 and 40 levels) for CatBoost's target statistics."""
    rng = np.random.default_rng(20261007)
    latent = rng.normal(size=(N, 3))
    load = rng.normal(size=(3, 13))
    raw = latent @ load + rng.normal(scale=1.2, size=(N, 13))
    X = pd.DataFrame(np.clip(np.round(raw + 2.5), 0, 5).astype(int), columns=[f"r{i}" for i in range(13)])
    f = (0.6 * X.r0 - 0.4 * X.r3 + 0.3 * (X.r1 > 3) * X.r7 - 0.5 * (X.r5 < 2) + 0.2 * X.r9 * (X.r11 > 2)
         - 2.0 + rng.normal(scale=0.8, size=N))
    y = (rng.random(N) < 1 / (1 + np.exp(-f))).astype(int)
    yreg = 0.4 * X.r0 - 0.25 * X.r5 + 0.15 * X.r8 * (X.r2 > 2) + rng.normal(scale=1.0, size=N)
    cats = pd.DataFrame({
        "c2": np.where(rng.random(N) < 0.4, "a", "b"),
        "c3": rng.choice(["x", "y", "z"], N, p=[0.5, 0.3, 0.2]),
        "c40": [f"k{v}" for v in rng.integers(0, 40, N)],
    })
    # The categoricals carry signal, so their statistics are chosen as splits.
    fc = f + 0.8 * (cats.c3 == "z") + 0.05 * cats.c40.str[1:].astype(int) - 1.0
    ycat = (rng.random(N) < 1 / (1 + np.exp(-fc))).astype(int)
    # Four classes, from four noisy scores of the same ratings: unequal sizes, overlapping.
    s4 = np.stack([0.5 * X.r0 - 0.3 * X.r3, 0.4 * X.r5 + 0.2 * X.r1 * (X.r7 > 2), 0.3 * X.r9 - 0.2 * X.r2 + 0.4,
                   0.25 * X.r11 + 0.2 * (X.r4 < 2)], axis=1) + rng.gumbel(scale=0.7, size=(N, 4))
    ymc = s4.argmax(axis=1)
    return X, y, yreg.to_numpy(), cats, ycat, ymc


X, y, yreg, CATS, ycat, ymc = make_data()
pd.concat([X, pd.Series(y, name="y")], axis=1).to_csv(WORK / "disc.csv", index=False)
pd.concat([X, pd.Series(yreg, name="yreg")], axis=1).to_csv(WORK / "reg.csv", index=False)
XC = pd.concat([X.iloc[:, :4], CATS], axis=1)
pd.concat([XC, pd.Series(ycat, name="y")], axis=1).to_csv(WORK / "cat.csv", index=False)
pd.concat([X, pd.Series(ymc, name="ymc")], axis=1).to_csv(WORK / "multi.csv", index=False)
# Row weights from 0.1 to 5, some exactly 0, for the weighted cases.
W = np.where(np.random.default_rng(5).random(N) < 0.05, 0.0, np.random.default_rng(6).gamma(2.0, 0.6, N).clip(0.1, 5))
pd.concat([X, pd.Series(W, name="w"), pd.Series(y, name="y")], axis=1).to_csv(WORK / "wdisc.csv", index=False)
pd.concat([X, pd.Series(W, name="w"), pd.Series(ymc, name="ymc")], axis=1).to_csv(WORK / "wmulti.csv", index=False)
MEAN = y.mean()
BASE_MARGIN = float(np.log(MEAN / (1 - MEAN)))


def zarbor(csv, label, flags, cols=1, predict_flags=()):
    model, out = WORK / "z.zm", WORK / "z.csv"
    for cmd in ([Z, str(WORK / csv), f"--label={label}", "--valid-frac=0", "--verbose_eval=0", f"--save={model}", *flags],
                [Z, "predict", str(WORK / csv), f"--model={model}", f"--out={out}", *predict_flags]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd[:3])} failed:\n{r.stdout[-600:]}{r.stderr[-600:]}")
    pred = pd.read_csv(out)
    return pred.iloc[:, -1].to_numpy() if cols == 1 else pred.iloc[:, -cols:].to_numpy()


FAILED = []


def report(name, z, ref, tol=TOL):
    d = np.abs(z - ref)
    ok = (d <= tol * np.maximum(1.0, np.abs(ref))).all()
    if not ok:
        FAILED.append(name)
    print(f"{'pass' if ok else 'FAIL'}  {name:<44} max|diff| {d.max():.1e}  rows within 6e-7 {(d <= 6e-7).mean() * 100:6.2f}%",
          flush=True)


CASES = {}


def case(f):
    CASES[f.__name__] = f
    return f


# ---- XGBoost, depthwise: no per-bin floor, no row-count floor ---------------------------------
XZ = ["--min_data_in_bin=1", "--min_child_samples=0"]


def xgb_pred(params, rounds, label=y, objective="binary:logistic", base=None):
    p = dict(objective=objective, tree_method="hist", max_bin=256, nthread=16, seed=0,
             base_score=float(np.mean(label)) if base is None else base, **params)
    return xgb.train(p, xgb.DMatrix(X, label), rounds).predict(xgb.DMatrix(X))


@case
def xgb_stump():
    z = zarbor("disc.csv", "y", XZ + ["--n_rounds=1", "--max_depth=1", "--learning_rate=1"])
    report("xgb 1 tree depth 1", z, xgb_pred(dict(eta=1.0, max_depth=1), 1))


@case
def xgb_regularised():
    z = zarbor("disc.csv", "y", XZ + ["--n_rounds=1", "--max_depth=6", "--learning_rate=1", "--lambda=5", "--alpha=2",
                                     "--min_split_gain=1", "--min_child_weight=20"])
    report("xgb 1 tree, L1 + L2 + gamma + min_child_weight", z,
           xgb_pred(dict(eta=1.0, max_depth=6, reg_lambda=5.0, reg_alpha=2.0, gamma=1.0, min_child_weight=20), 1))


@case
def xgb_500():
    z = zarbor("disc.csv", "y", XZ + ["--n_rounds=500", "--max_depth=6", "--learning_rate=0.1"])
    report("xgb 500 trees depth 6", z, xgb_pred(dict(eta=0.1, max_depth=6), 500))


@case
def xgb_regression():
    z = zarbor("reg.csv", "yreg", XZ + ["--objective=squared_error", "--n_rounds=200", "--max_depth=6", "--learning_rate=0.1"])
    report("xgb regression 200 trees", z, xgb_pred(dict(eta=0.1, max_depth=6), 200, yreg, "reg:squarederror"))


@case
def xgb_max_delta_step():
    z = zarbor("disc.csv", "y", XZ + ["--n_rounds=100", "--max_depth=6", "--learning_rate=0.3", "--lambda=0.1",
                                     "--max_delta_step=0.7"])
    report("xgb max_delta_step 0.7, 100 trees", z,
           xgb_pred(dict(eta=0.3, max_depth=6, reg_lambda=0.1, max_delta_step=0.7), 100))


@case
def xgb_scale_pos_weight():
    # The starting score differs by design (zarbor follows LightGBM), so it is pinned on both sides.
    z = zarbor("disc.csv", "y", XZ + ["--n_rounds=100", "--max_depth=6", "--learning_rate=0.1", "--scale_pos_weight=3",
                                     f"--base_score={BASE_MARGIN}"])
    report("xgb scale_pos_weight 3, 100 trees", z, xgb_pred(dict(eta=0.1, max_depth=6, scale_pos_weight=3), 100, base=MEAN))


# Multiclass: one tree per class per round, XGBoost's `2 p (1 - p)` hessian and its centred
# per-class starting scores (estimated by XGBoost itself, as zarbor does).
def xgb_multi(params, rounds):
    p = dict(objective="multi:softprob", num_class=4, tree_method="hist", max_bin=256, nthread=16, seed=0, **params)
    return xgb.train(p, xgb.DMatrix(X, ymc), rounds).predict(xgb.DMatrix(X))


@case
def xgb_multiclass():
    z = zarbor("multi.csv", "ymc", XZ + ["--objective=softmax", "--n_rounds=100", "--max_depth=6", "--learning_rate=0.1"], cols=4)
    report("xgb multiclass, 4 classes, 100 rounds", z.ravel(), xgb_multi(dict(eta=0.1, max_depth=6), 100).ravel())


@case
def xgb_multiclass_regularised():
    z = zarbor("multi.csv", "ymc", XZ + ["--objective=softmax", "--n_rounds=40", "--max_depth=4", "--learning_rate=0.3",
                                        "--lambda=3", "--alpha=0.5", "--min_child_weight=5", "--min_split_gain=0.5"], cols=4)
    report("xgb multiclass, L1 + L2 + gamma + mcw, 40 rounds", z.ravel(),
           xgb_multi(dict(eta=0.3, max_depth=4, reg_lambda=3.0, reg_alpha=0.5, min_child_weight=5, gamma=0.5), 40).ravel())


# ---- LightGBM, leaf-wise: its binning, no row-count floor (it estimates row counts) ------------
LZ = ["--bin_policy=greedy", "--grow_policy=lossguide", "--max_depth=0", "--max_leaves=31", "--min_child_samples=0"]


def lgb_pred(params, rounds, label=y, objective="binary"):
    p = dict(objective=objective, max_bin=255, num_threads=16, seed=0, verbose=-1, max_depth=-1, num_leaves=31,
             min_data_in_leaf=0, min_sum_hessian_in_leaf=1.0, lambda_l2=1.0)
    p.update(params)
    return lgb.train(p, lgb.Dataset(X, label), rounds).predict(X)


# LightGBM skips, in every descendant, a feature that had no positive-gain cut at a node, and
# gain is not monotone down a branch, so in bigger trees it can miss the best split that zarbor
# takes (docs/correctness.md section 1). On this data the first such miss is tree 34 at 31
# leaves: LightGBM splits a leaf of gain 4.731 because the better leaf's best split (4.761, on
# r11) is on a feature it pruned there. These cases stop short of that, so any difference is
# an arithmetic one.
@case
def lgb_31_leaves():
    z = zarbor("disc.csv", "y", LZ + ["--n_rounds=30", "--learning_rate=0.1"])
    report("lgb 30 trees 31 leaves", z, lgb_pred(dict(learning_rate=0.1), 30))


@case
def lgb_15_leaves():
    lz = [f for f in LZ if not f.startswith("--max_leaves")] + ["--max_leaves=15"]
    z = zarbor("disc.csv", "y", lz + ["--n_rounds=50", "--learning_rate=0.1"])
    report("lgb 50 trees 15 leaves", z, lgb_pred(dict(learning_rate=0.1, num_leaves=15), 50))


# LightGBM's multiclass hessian is `K / (K - 1) p (1 - p)`; its starting scores are the class
# log-priors without XGBoost's centring, which changes no probability.
@case
def lgb_multiclass():
    z = zarbor("multi.csv", "ymc", LZ + ["--objective=softmax", "--softmax_hessian=lightgbm", "--n_rounds=10",
                                        "--learning_rate=0.1"], cols=4)
    p = dict(objective="multiclass", num_class=4, max_bin=255, num_threads=16, seed=0, verbose=-1, max_depth=-1,
             num_leaves=31, min_data_in_leaf=0, min_sum_hessian_in_leaf=1.0, lambda_l2=1.0, learning_rate=0.1)
    ref = lgb.train(p, lgb.Dataset(X, ymc), 10).predict(X)
    report("lgb multiclass, 4 classes, 10 rounds", z.ravel(), ref.ravel())


# ---- CatBoost, symmetric trees: plain, file order, no sampling or noise -----------------------
CZ = ["--grow_policy=symmetric", "--has_time=true", "--min_data_in_bin=1", "--base_score=0", "--lambda=3",
      "--bin_policy=logsum", "--max_depth=6"]


def cat_pred(data, label, **kw):
    p = dict(depth=6, l2_leaf_reg=3, boosting_type="Plain", bootstrap_type="No", random_strength=0,
             leaf_estimation_method="Newton", leaf_estimation_iterations=1, leaf_estimation_backtracking="No",
             border_count=254, feature_border_type="GreedyLogSum", thread_count=16, verbose=0,
             allow_writing_files=False, has_time=True)
    p.update(kw)
    fit_kw = {}
    if "cat_features" in p:
        fit_kw["cat_features"] = p.pop("cat_features")
    m = catboost.CatBoostClassifier(**p)
    m.fit(data, label, **fit_kw)
    return m.predict_proba(data)[:, 1]


@case
def cat_300():
    z = zarbor("disc.csv", "y", CZ + ["--n_rounds=300", "--learning_rate=0.1"])
    report("catboost 300 trees, Cosine", z, cat_pred(X, y, iterations=300, learning_rate=0.1))


@case
def cat_newton10():
    z = zarbor("disc.csv", "y", CZ + ["--n_rounds=50", "--learning_rate=0.1", "--leaf_estimation_iterations=10"])
    report("catboost 50 trees, 10 Newton steps", z, cat_pred(X, y, iterations=50, learning_rate=0.1,
                                                              leaf_estimation_iterations=10))


@case
def cat_ordered():
    z = zarbor("disc.csv", "y", CZ + ["--n_rounds=50", "--learning_rate=0.1", "--boosting_type=ordered"])
    report("catboost ordered boosting, 50 trees", z, cat_pred(X, y, iterations=50, learning_rate=0.1,
                                                               boosting_type="Ordered"))


@case
def cat_ctr():
    ctr = ["--cat_split=ctr", "--one_hot_max_size=2"]
    for cx in (1, 4):
        z = zarbor("cat.csv", "y", CZ + ctr + ["--n_rounds=50", "--learning_rate=0.1", f"--max_ctr_complexity={cx}"])
        report(f"catboost target statistics, complexity {cx}, 50 trees", z,
               cat_pred(XC, ycat, iterations=50, learning_rate=0.1, one_hot_max_size=2, max_ctr_complexity=cx,
                        cat_features=["c2", "c3", "c40"]))


# ---- sample weights: each row's gradient and hessian times its weight -------------------------
WMEAN = float(np.sum(W * y) / np.sum(W))


@case
def xgb_weights():
    z = zarbor("wdisc.csv", "y", XZ + ["--weight-col=w", "--n_rounds=100", "--max_depth=6", "--learning_rate=0.1"])
    p = dict(objective="binary:logistic", tree_method="hist", max_bin=256, nthread=16, seed=0, base_score=WMEAN,
             eta=0.1, max_depth=6)
    ref = xgb.train(p, xgb.DMatrix(X, y, weight=W), 100).predict(xgb.DMatrix(X))
    report("xgb weighted, 100 trees", z, ref)


@case
def xgb_weights_multiclass():
    z = zarbor("wmulti.csv", "ymc", XZ + ["--weight-col=w", "--objective=softmax", "--n_rounds=40", "--max_depth=6",
                                         "--learning_rate=0.1"], cols=4)
    p = dict(objective="multi:softprob", num_class=4, tree_method="hist", max_bin=256, nthread=16, seed=0, eta=0.1,
             max_depth=6)
    ref = xgb.train(p, xgb.DMatrix(X, ymc, weight=W), 40).predict(xgb.DMatrix(X))
    report("xgb weighted multiclass, 40 rounds", z.ravel(), ref.ravel())


@case
def lgb_weights():
    z = zarbor("wdisc.csv", "y", LZ + ["--weight-col=w", "--n_rounds=30", "--learning_rate=0.1"])
    p = dict(objective="binary", max_bin=255, num_threads=16, seed=0, verbose=-1, max_depth=-1, num_leaves=31,
             min_data_in_leaf=0, min_sum_hessian_in_leaf=1.0, lambda_l2=1.0, learning_rate=0.1)
    ref = lgb.train(p, lgb.Dataset(X, y, weight=W), 30).predict(X)
    report("lgb weighted, 30 trees", z, ref)


# CatBoost: weights multiply the derivatives, replace the row count in split scoring, and scale
# l2_leaf_reg by the mean weight; target statistics ignore them.
pd.concat([XC, pd.Series(W, name="w"), pd.Series(ycat, name="y")], axis=1).to_csv(WORK / "wcat.csv", index=False)


def cat_pred_w(data, label, **kw):
    p = dict(depth=6, l2_leaf_reg=3, boosting_type="Plain", bootstrap_type="No", random_strength=0,
             leaf_estimation_method="Newton", leaf_estimation_iterations=1, leaf_estimation_backtracking="No",
             border_count=254, feature_border_type="GreedyLogSum", thread_count=16, verbose=0,
             allow_writing_files=False, has_time=True)
    p.update(kw)
    fit_kw = {"sample_weight": W}
    if "cat_features" in p:
        fit_kw["cat_features"] = p.pop("cat_features")
    m = catboost.CatBoostClassifier(**p)
    m.fit(data, label, **fit_kw)
    return m.predict_proba(data)[:, 1]


@case
def cat_weights():
    z = zarbor("wdisc.csv", "y", CZ + ["--weight-col=w", "--n_rounds=100", "--learning_rate=0.1",
                                      "--leaf_estimation_iterations=3"])
    report("catboost weighted, 100 trees, 3 Newton steps", z, cat_pred_w(X, y, iterations=100, learning_rate=0.1,
                                                                        leaf_estimation_iterations=3))
    z = zarbor("wdisc.csv", "y", CZ + ["--weight-col=w", "--n_rounds=50", "--learning_rate=0.1", "--boosting_type=ordered"])
    report("catboost weighted ordered boosting, 50 trees", z, cat_pred_w(X, y, iterations=50, learning_rate=0.1,
                                                                        boosting_type="Ordered"))
    z = zarbor("wcat.csv", "y", CZ + ["--weight-col=w", "--cat_split=ctr", "--one_hot_max_size=2", "--n_rounds=50",
                                      "--learning_rate=0.1", "--max_ctr_complexity=4"])
    report("catboost weighted, target statistics cx 4, 50 trees", z,
           cat_pred_w(XC, ycat, iterations=50, learning_rate=0.1, one_hot_max_size=2, max_ctr_complexity=4,
                      cat_features=["c2", "c3", "c40"]))


# ---- starting scores: XGBoost base_margin, LightGBM init_score, CatBoost baseline -------------
M = (0.3 * (X.r2 - 2.5) + 0.2 * np.random.default_rng(7).normal(size=N)).to_numpy()
M4 = np.stack([0.2 * (X.r2 - 2.5), -0.1 * X.r6, 0.15 * (X.r10 - 2), np.zeros(N)], axis=1)
pd.concat([X, pd.Series(M, name="m"), pd.Series(y, name="y")], axis=1).to_csv(WORK / "mdisc.csv", index=False)
pd.concat([X, pd.DataFrame(M4, columns=["m0", "m1", "m2", "m3"]), pd.Series(ymc, name="ymc")], axis=1).to_csv(
    WORK / "mmulti.csv", index=False)


@case
def xgb_margin():
    z = zarbor("mdisc.csv", "y", XZ + ["--init-col=m", "--n_rounds=100", "--max_depth=6", "--learning_rate=0.1"],
               predict_flags=["--init-col=m"])
    p = dict(objective="binary:logistic", tree_method="hist", max_bin=256, nthread=16, seed=0, eta=0.1, max_depth=6)
    ref = xgb.train(p, xgb.DMatrix(X, y, base_margin=M), 100).predict(xgb.DMatrix(X, base_margin=M))
    report("xgb base_margin, 100 trees", z, ref)
    z = zarbor("mmulti.csv", "ymc", XZ + ["--init-col=m0,m1,m2,m3", "--objective=softmax", "--n_rounds=40",
                                         "--max_depth=6", "--learning_rate=0.1"], cols=4,
               predict_flags=["--init-col=m0,m1,m2,m3"])
    p = dict(objective="multi:softprob", num_class=4, tree_method="hist", max_bin=256, nthread=16, seed=0, eta=0.1,
             max_depth=6)
    ref = xgb.train(p, xgb.DMatrix(X, ymc, base_margin=M4), 40).predict(xgb.DMatrix(X, base_margin=M4))
    report("xgb base_margin multiclass, 40 rounds", z.ravel(), ref.ravel())


@case
def lgb_init_score():
    z = zarbor("mdisc.csv", "y", LZ + ["--init-col=m", "--n_rounds=30", "--learning_rate=0.1"],
               predict_flags=["--init-col=m"])
    p = dict(objective="binary", max_bin=255, num_threads=16, seed=0, verbose=-1, max_depth=-1, num_leaves=31,
             min_data_in_leaf=0, min_sum_hessian_in_leaf=1.0, lambda_l2=1.0, learning_rate=0.1)
    b = lgb.train(p, lgb.Dataset(X, y, init_score=M), 30)
    ref = 1 / (1 + np.exp(-(b.predict(X, raw_score=True) + M)))
    report("lgb init_score, 30 trees", z, ref)


@case
def cat_baseline():
    z = zarbor("mdisc.csv", "y", CZ + ["--init-col=m", "--n_rounds=50", "--learning_rate=0.1"],
               predict_flags=["--init-col=m"])
    p = dict(depth=6, l2_leaf_reg=3, boosting_type="Plain", bootstrap_type="No", random_strength=0,
             leaf_estimation_method="Newton", leaf_estimation_iterations=1, leaf_estimation_backtracking="No",
             border_count=254, feature_border_type="GreedyLogSum", thread_count=16, verbose=0,
             allow_writing_files=False, has_time=True, iterations=50, learning_rate=0.1)
    m = catboost.CatBoostClassifier(**p).fit(catboost.Pool(X, y, baseline=M))
    ref = 1 / (1 + np.exp(-(m.predict(catboost.Pool(X), prediction_type="RawFormulaVal") + M)))
    report("catboost baseline, 50 trees", z, ref)


# ---- SHAP: on the same trees, the per-row attributions must agree too ------------------------
def zarbor_shap(csv, label, flags, cover):
    """Train like `zarbor`, then `zarbor explain --shap`: one column per feature, then the bias."""
    model, out = WORK / "s.zm", WORK / "s.csv"
    for cmd in ([Z, str(WORK / csv), f"--label={label}", "--valid-frac=0", "--verbose_eval=0", f"--save={model}", *flags],
                [Z, "explain", str(WORK / csv), f"--model={model}", f"--shap={out}", f"--cover={cover}"]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd[:3])} failed:\n{r.stdout[-600:]}{r.stderr[-600:]}")
    return pd.read_csv(out).to_numpy()


@case
def xgb_shap():
    z = zarbor_shap("disc.csv", "y", XZ + ["--n_rounds=100", "--max_depth=6", "--learning_rate=0.1"], "hessian")
    p = dict(objective="binary:logistic", tree_method="hist", max_bin=256, nthread=16, seed=0, base_score=MEAN,
             eta=0.1, max_depth=6)
    b = xgb.train(p, xgb.DMatrix(X, y), 100)
    ref = b.predict(xgb.DMatrix(X), pred_contribs=True)
    # XGBoost sums SHAP in f32: its own values miss its own margin by up to ~4e-6 here, so that,
    # not the print rounding, is the tolerance (zarbor's sums match the margin to 1e-6).
    own = np.abs(ref.sum(axis=1) - b.predict(xgb.DMatrix(X), output_margin=True)).max()
    report(f"SHAP vs xgb pred_contribs, 100 trees (xgb self-error {own:.1e})", z.ravel(), ref.ravel(), tol=max(TOL, own))


@case
def lgb_shap():
    z = zarbor_shap("disc.csv", "y", LZ + ["--n_rounds=30", "--learning_rate=0.1"], "count")
    p = dict(objective="binary", max_bin=255, num_threads=16, seed=0, verbose=-1, max_depth=-1, num_leaves=31,
             min_data_in_leaf=0, min_sum_hessian_in_leaf=1.0, lambda_l2=1.0, learning_rate=0.1)
    ref = lgb.train(p, lgb.Dataset(X, y), 30).predict(X, pred_contrib=True)
    report("SHAP vs lgb pred_contrib, 30 trees", z.ravel(), ref.ravel())


@case
def cat_shap():
    z = zarbor_shap("disc.csv", "y", CZ + ["--n_rounds=50", "--learning_rate=0.1"], "count")
    p = dict(depth=6, l2_leaf_reg=3, boosting_type="Plain", bootstrap_type="No", random_strength=0,
             leaf_estimation_method="Newton", leaf_estimation_iterations=1, leaf_estimation_backtracking="No",
             border_count=254, feature_border_type="GreedyLogSum", thread_count=16, verbose=0,
             allow_writing_files=False, has_time=True, iterations=50, learning_rate=0.1)
    m = catboost.CatBoostClassifier(**p).fit(X, y)
    ref = m.get_feature_importance(catboost.Pool(X, y), type="ShapValues")
    report("SHAP vs catboost ShapValues, 50 trees", z.ravel(), ref.ravel())


for name in sys.argv[1:] or CASES:
    CASES[name]()
print(f"\n{len(FAILED)} of {len(sys.argv[1:] or CASES)} cases failed" + (f": {', '.join(FAILED)}" if FAILED else ""))
sys.exit(1 if FAILED else 0)
