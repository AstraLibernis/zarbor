#!/usr/bin/env python
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 AstraLibernis

"""Does zarbor agree with LightGBM, or has it been fitted to one dataset?

The question this answers is not "does zarbor score well". It is "does zarbor
compute the same thing an independent implementation computes, on data it was
never tuned against". Those come apart exactly when a library has been shaped
around one competition frame, which is the failure mode worth guarding.

Three things make the comparison mean something:

  * **One split, shared.** A `__split` column is written into a copy of the CSV
    and handed to `zarbor --split-col`; the Python side masks on the same
    column. Different splits would put more noise on the comparison than the
    effect being measured.

  * **A noise floor, not a tolerance.** "Within 0.002 AUC" is a number someone
    chose. The first version of this ran LightGBM against itself across
    `random_state` values, which measures nothing: with no row or feature
    sampling LightGBM is deterministic and the spread came out exactly zero.
    The yardstick used instead is LightGBM's spread over `max_bin` in
    {63, 127, 255} -- how much the score moves when a defensible binning choice
    changes, which is the largest known difference between the two
    implementations. A gap inside that envelope is implementation variation.

  * **Categoricals isolated.** Every dataset is also run with its string
    columns dropped on both sides. If the gap is the same with and without
    them, the difference is in the base learner; if it opens up only when they
    are present, it is the categorical path. Those need different fixes.

  * **A negative control.** zarbor is also run in a configuration that is
    deliberately *not* what LightGBM is doing -- ordinal categorical splits
    against LightGBM's optimal ones. If that does not show a larger gap, the
    harness cannot detect a disagreement and none of its passes mean anything.

Exact agreement is not on offer and is not the goal: the two bin numeric
columns differently (zarbor's quantile edges against LightGBM's sketch), break
ties differently, and start from different base scores.
"""

import argparse, json, re, subprocess, sys, tempfile, os
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
ZARBOR = ROOT / "zig-out" / "bin" / "zarbor"

# Matched deliberately, not left at either side's defaults, which differ.
# min_child_weight is the one that bites: LightGBM's min_sum_hessian_in_leaf
# defaults to 1e-3 where zarbor's min_child_weight defaults to 1.0.
COMMON = dict(n_rounds=300, learning_rate=0.05, max_leaves=31, max_depth=0,
              min_child_samples=20, min_child_weight=1e-3, lam=0.0, max_bin=255)

DATASETS = [
    # name, label, objective, categorical columns to let both sides use
    ("california", "y", "l2", []),
    ("adult", "y", "binary", "auto"),
    ("bank", "y", "binary", "auto"),
    ("ames", "y", "l2", "auto"),
    ("housing", "AffordabilityPercentageTrue", "l2", "auto"),
]


def load(name, label):
    df = pd.read_csv(ROOT / "bench" / "data" / f"{name}.csv", low_memory=False)
    if name == "housing":
        df = df.drop(columns=["Zip"])  # an id, not a feature; zarbor drops it too
    rng = np.random.default_rng(12345)
    df["__split"] = (rng.random(len(df)) < 0.2).astype(int)
    return df, label


def run_zarbor(df, label, objective, extra):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "d.csv"
        df.to_csv(p, index=False)
        cmd = [str(ZARBOR), str(p), f"--label={label}", "--split-col=__split",
               f"--objective={'logistic' if objective == 'binary' else 'squared_error'}",
               f"--n_rounds={COMMON['n_rounds']}",
               f"--learning_rate={COMMON['learning_rate']}",
               f"--max_leaves={COMMON['max_leaves']}", f"--max_depth={COMMON['max_depth']}",
               f"--min_child_samples={COMMON['min_child_samples']}",
               f"--min_child_weight={COMMON['min_child_weight']}",
               f"--lambda={COMMON['lam']}", f"--max_bin={COMMON['max_bin']}",
               "--grow_policy=lossguide", *extra]
        r = subprocess.run(cmd, capture_output=True, text=True)
        out = r.stdout + r.stderr
        m = re.search(r"valid\s+(auc|rmse)=([0-9.]+)", out)
        if not m:
            return None, out.strip().splitlines()[-1] if out.strip() else "no output"
        return float(m.group(2)), None


def drop_flags(df, label):
    out = []
    for c in df.columns:
        if c in (label, "__split"):
            continue
        if df[c].dtype == object or str(df[c].dtype) == "str":
            out.append(f"--drop={c}")
    return out


def run_lightgbm(df, label, objective, cats, seed, use_cats=True, max_bin=None):
    tr, va = df.__split == 0, df.__split != 0
    X = df.drop(columns=[label, "__split"]).copy()
    y = df[label].values
    cat_cols = []
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "str":
            X[c] = pd.Categorical(X[c].fillna("?")).codes.astype(np.int32)
            cat_cols.append(c)
    if not use_cats:
        X = X.drop(columns=cat_cols)
        cat_cols = []
    M = lgb.LGBMClassifier if objective == "binary" else lgb.LGBMRegressor
    m = M(n_estimators=COMMON["n_rounds"], learning_rate=COMMON["learning_rate"],
          num_leaves=COMMON["max_leaves"], max_depth=-1,
          min_child_samples=COMMON["min_child_samples"],
          min_child_weight=COMMON["min_child_weight"],
          reg_lambda=COMMON["lam"], max_bin=max_bin or COMMON["max_bin"],
          min_split_gain=0.0, subsample=1.0, colsample_bytree=1.0,
          cat_smooth=10, cat_l2=10, max_cat_threshold=32,
          verbose=-1, random_state=seed)
    m.fit(X[tr], y[tr], categorical_feature=cat_cols)
    if objective == "binary":
        return roc_auc_score(y[va], m.predict_proba(X[va])[:, 1])
    p = m.predict(X[va])
    return float(np.sqrt(np.mean((p - y[va]) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    print(f"{'dataset':22s} {'metric':6s} {'lightgbm':>10s} {'zarbor':>10s} "
          f"{'gap':>9s} {'binning':>9s} {'gap/bin':>8s}  verdict")
    for name, label, objective, cats in DATASETS:
        if args.only and name not in args.only.split(","):
            continue
        df, label = load(name, label)
        metric = "auc" if objective == "binary" else "rmse"

        for keep_cats in (True, False):
            tag = "" if keep_cats else " (numeric only)"
            lg = [run_lightgbm(df, label, objective, cats, 1, use_cats=keep_cats, max_bin=mb)
                  for mb in (63, 127, 255)]
            lg_ref = lg[-1]                     # max_bin=255, the matched setting
            envelope = float(max(lg) - min(lg))  # what a binning choice is worth

            extra = ["--cat_split=optimal", "--max_cat_levels=4096",
                     "--bin_policy=greedy", "--min_data_in_bin=3"]
            if not keep_cats:
                extra += drop_flags(df, label)
            za, err = run_zarbor(df, label, objective, extra)
            if za is None:
                print(f"{name + tag:22s} {metric:6s} {lg_ref:10.6f} {'-':>10s}  FAILED: {err}")
                continue
            gap = abs(za - lg_ref)
            ratio = gap / envelope if envelope > 0 else float("inf")
            verdict = "agrees" if ratio <= 1 else ("close" if ratio <= 3 else "DIFFERS")
            print(f"{name + tag:22s} {metric:6s} {lg_ref:10.6f} {za:10.6f} {gap:9.6f} "
                  f"{envelope:9.6f} {ratio:8.2f}  {verdict}")

            # Negative control, only where there are categoricals to get wrong.
            if keep_cats and cats == "auto":
                zo, _ = run_zarbor(df, label, objective,
                                   ["--cat_split=ordinal", "--bin_policy=greedy"])
                if zo is not None:
                    g2 = abs(zo - lg_ref)
                    detect = "detects it" if g2 > gap else "NO SIGNAL"
                    print(f"{'  control: ordinal':22s} {'':6s} {'':>10s} {zo:10.6f} "
                          f"{g2:9.6f} {envelope:9.6f} {g2 / envelope if envelope else 0:8.2f}"
                          f"  {detect}")


main()
