#!/usr/bin/env python
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 AstraLibernis

"""Compare zarbor against the reference Python implementations, 1:1.

Fairness rests on three things:

  * One split, shared. The script writes a copy of the CSV with a `__split`
    column and hands it to `zarbor --split-col=__split`, so both sides train on
    exactly the same rows and score exactly the same rows. A random split per
    implementation would put ~0.001 of AUC noise on every comparison.
  * Matched hyperparameters, named explicitly below rather than left at each
    library's defaults, which differ from each other.
  * The same unit of work on both sides: in-memory numeric data to trained
    model. That means zarbor is charged for binning as well as training, since
    the reference libraries bin inside fit(). CSV parsing is excluded from
    both, pandas having already paid it for the Python side.
  * Best-of-N, not one shot. A single fit on a loaded box scatters by 10-20%,
    which is the same size as the differences worth chasing. `--repeat`
    controls N; the minimum is reported, being the run least contaminated by
    whatever else the machine was doing.
  * A release build, checked rather than assumed. `zig build` with no flags
    produces a *Debug* binary, which is ~8x slower here; benchmarking one
    silently turned a 2x gap into an 8x gap and sent the conclusion badly
    wrong. This script builds ReleaseFast itself and refuses to time anything
    that does not report a release mode.

What it cannot control for: binning is not identical (zarbor uses quantile
edges over u8 bins; XGBoost and LightGBM have their own sketching), and
categorical handling differs. Read the AUC column as "same ballpark or not",
not as a claim of algorithmic equivalence.
"""

import argparse, json, re, subprocess, sys, time
import tempfile
TMP = tempfile.mkdtemp(prefix="zarbor-bench-")  # private scratch, honours $TMPDIR
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
import lightgbm as lgb

# Hyperparameters held common across every pair.
N_ROUNDS = 200
LR = 0.1
MAX_DEPTH = 6
MAX_LEAVES = 64
L2 = 1.0
MAX_BIN = 256
RF_TREES = 100
RF_LEAVES = 1024


def load(csv, label, drop, valid_frac, seed):
    df = pd.read_csv(csv)
    for c in drop:
        if c in df.columns:
            df = df.drop(columns=c)
    # zarbor encodes a string target by sorted class order, which is what
    # sklearn's LabelEncoder does, so the two sides agree without a
    # translation table. (It did once use the dictionary's first-appearance
    # ids, which made the target depend on row order; both sides are on the
    # sorted convention now.)
    ycol = df[label]
    if ycol.dtype == object or str(ycol.dtype) == "str":
        uniques = np.sort(ycol.unique())
        if len(uniques) != 2:
            raise SystemExit(f"label {label!r} has {len(uniques)} classes; "
                             "zarbor fits binary targets only")
        y = (ycol.to_numpy() == uniques[1]).astype(np.float32)
        print(f"label {label!r} encoded by sorted class order: "
              + ", ".join(f"{v!r}->{i}" for i, v in enumerate(uniques)))
    else:
        y = ycol.astype(np.float32).to_numpy()
    X = df.drop(columns=label)

    rng = np.random.default_rng(seed)
    is_valid = np.zeros(len(df), dtype=np.int8)
    is_valid[rng.choice(len(df), size=int(round(len(df) * valid_frac)), replace=False)] = 1
    return df, X, y, is_valid, label


def write_split_csv(df, is_valid, path):
    out = df.copy()
    out["__split"] = is_valid
    out.to_csv(path, index=False)
    return path


def encode(X):
    """Ordinal codes for tree models (which is how zarbor splits categoricals)
    and a one-hot matrix for the linear model (which is how zarbor builds its
    design matrix)."""
    cat = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) == "str"]
    num = [c for c in X.columns if c not in cat]

    ordinal = X.copy()
    for c in cat:
        # factorize, not Categorical: first-appearance ids are what zarbor
        # assigns, and for a tree splitting on `bin <= threshold` the id order
        # changes which partitions are even reachable.
        ordinal[c] = pd.factorize(ordinal[c])[0].astype(np.float32)
    ordinal = ordinal.astype(np.float32).to_numpy()

    parts = []
    if num:
        parts.append(StandardScaler().fit_transform(X[num].astype(np.float64).fillna(0)))
    if cat:
        parts.append(OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                     .fit_transform(X[cat].astype(str)))
    onehot = np.hstack(parts) if parts else np.zeros((len(X), 0))
    return ordinal, onehot


def build_release(root):
    """Build the binary the benchmark is about to time. Not optional: the
    default `zig build` is Debug."""
    r = subprocess.run(["zig", "build", "-Doptimize=ReleaseFast"],
                       cwd=root, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("zig build -Doptimize=ReleaseFast failed:\n" + r.stderr)


def check_release(binary, csv, label):
    """Read the mode the binary reports and refuse a Debug one."""
    r = subprocess.run([str(binary), str(csv), f"--label={label}",
                        "--n_rounds=1", "--verbose_eval=0", "--n_threads=1"],
                       capture_output=True, text=True)
    m = re.search(r"^build\s+(\w+)", r.stdout, re.M)
    if not m:
        raise SystemExit(f"{binary} does not report a build mode; it predates "
                         "this check. Rebuild with -Doptimize=ReleaseFast.")
    if m.group(1) == "Debug":
        raise SystemExit(f"{binary} is a Debug build (~8x slower). "
                         "Rebuild with: zig build -Doptimize=ReleaseFast")
    return m.group(1)


def run_zarbor_once(binary, csv, label, extra):
    cmd = [str(binary), str(csv), f"--label={label}", "--split-col=__split",
           "--verbose_eval=0"] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None, None, None, r.stderr.strip()[:200] or r.stdout.strip()[:200]
    auc = re.search(r"auc=([0-9.]+)", r.stdout)
    if not auc:
        return None, None, None, "no auc in output"
    # Charge zarbor for binning as well as training. The reference libraries bin
    # inside fit() (xgboost ~87 ms, lightgbm ~46 ms on this data), so timing
    # only zarbor's `train` line compared a kernel against a kernel-plus-prep
    # and quietly flattered zarbor. The fair unit is "in-memory numeric data ->
    # trained model", which for zarbor is bin + train and excludes CSV parsing,
    # since pandas has already paid that on the Python side.
    # Prefer the `fit` line, which excludes predicting and scoring the
    # validation set. The reference libraries are called without an eval set,
    # so charging zarbor for work they never did is not a comparison.
    fit = re.search(r"^fit\s+(\d+) ms", r.stdout, re.M)
    ms = fit or re.search(r"^train\s+(\d+) ms", r.stdout, re.M)
    bin_ms = re.search(r"^bin\s+(\d+) ms", r.stdout, re.M)
    if not ms:
        return None, None, None, "no train time in output"
    bin_s = (int(bin_ms.group(1)) if bin_ms else 0) / 1000
    return float(auc.group(1)), int(ms.group(1)) / 1000, bin_s, None


def run_zarbor(binary, csv, label, extra, repeat):
    best, auc, bin_s, err = None, None, None, None
    for _ in range(repeat):
        a, t, b, e = run_zarbor_once(binary, csv, label, extra)
        if e:
            return None, None, None, e
        auc, err = a, None
        if best is None or t + b < best:
            best, bin_s = t + b, b
    return auc, best, bin_s, err


def timed_fit(fit, repeat):
    """Best of `repeat` fits. Returns the last model and the fastest time."""
    best, out = None, None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fit()
        el = time.perf_counter() - t0
        if best is None or el < best:
            best = el
    return out, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--label", required=True)
    ap.add_argument("--drop", action="append", default=[])
    ap.add_argument("--valid-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--binary", default=str(Path(__file__).parent.parent / "zig-out/bin/zarbor"))
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--repeat", type=int, default=3,
                    help="fits per implementation; the fastest is reported")
    ap.add_argument("--only", action="append", default=[],
                    help="run a subset: xgb, lgb, rf, linear (repeatable)")
    ap.add_argument("--no-build", action="store_true",
                    help="time the existing binary instead of rebuilding it")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    want = set(a.only) or {"xgb", "lgb", "rf", "linear"}
    unknown = want - {"xgb", "lgb", "rf", "linear"}
    if unknown:
        raise SystemExit(f"unknown --only value(s): {', '.join(sorted(unknown))}")

    if not a.no_build:
        build_release(Path(__file__).parent.parent)

    df, X, y, is_valid, label = load(a.csv, a.label, a.drop, a.valid_frac, a.seed)
    split_csv = Path(TMP) / "zbench_split.csv"
    write_split_csv(df, is_valid, split_csv)
    mode = check_release(a.binary, split_csv, label)

    ordinal, onehot = encode(X)
    tr, va = is_valid == 0, is_valid == 1
    ytr, yva = y[tr], y[va]
    print(f"rows {len(df)}  feats {X.shape[1]}  train {tr.sum()}  valid {va.sum()}\n")

    print(f"best of {a.repeat} fit(s) per implementation, {a.threads} threads, "
          f"zarbor built {mode}\n")

    rows = []

    def binnote(bin_s):
        """Show how much of zarbor's time was binning, since the reference
        libraries hide that cost inside fit() and it is the part that does not
        shrink with better tree code."""
        return f"incl {bin_s:.2f}s bin" if bin_s else ""

    def add(name, impl, auc, fit_s, note=""):
        rows.append(dict(model=name, impl=impl, auc=auc, fit_s=fit_s, note=note))

    if "xgb" in want:
        # ---- XGBoost-style boosting -----------------------------------------
        auc, fit_s, bin_s, err = run_zarbor(a.binary, split_csv, label, [
            f"--n_rounds={N_ROUNDS}", f"--learning_rate={LR}", f"--max_depth={MAX_DEPTH}",
            f"--lambda={L2}", f"--max_bin={MAX_BIN}", f"--n_threads={a.threads}",
            # xgboost has no min_child_samples at all, so turn ours off rather
            # than leave it at 20 against xgboost's hessian-only constraint.
            "--min_child_samples=1", "--min_child_weight=1",
            "--grow_policy=depthwise"], a.repeat)
        add("GBDT (level-wise)", "zarbor", auc, fit_s, err or binnote(bin_s))

        m = xgb.XGBClassifier(n_estimators=N_ROUNDS, learning_rate=LR, max_depth=MAX_DEPTH,
                              reg_lambda=L2, max_bin=MAX_BIN, tree_method="hist",
                              grow_policy="depthwise", n_jobs=a.threads,
                              eval_metric="auc", min_child_weight=1)
        _, s = timed_fit(lambda: m.fit(ordinal[tr], ytr), a.repeat)
        add("GBDT (level-wise)", "xgboost", roc_auc_score(yva, m.predict_proba(ordinal[va])[:, 1]), s)

    if "lgb" in want:
        # ---- LightGBM-style boosting ----------------------------------------
        auc, fit_s, bin_s, err = run_zarbor(a.binary, split_csv, label, [
            f"--n_rounds={N_ROUNDS}", f"--learning_rate={LR}", "--max_depth=0",
            f"--max_leaves={MAX_LEAVES}", f"--lambda={L2}", f"--max_bin={MAX_BIN}",
            f"--n_threads={a.threads}", "--grow_policy=lossguide", "--sampling=goss"], a.repeat)
        add("GBDT (leaf-wise + GOSS)", "zarbor", auc, fit_s, err or binnote(bin_s))

        m = lgb.LGBMClassifier(n_estimators=N_ROUNDS, learning_rate=LR, num_leaves=MAX_LEAVES,
                               max_depth=-1, reg_lambda=L2, max_bin=MAX_BIN,
                               data_sample_strategy="goss", top_rate=0.2, other_rate=0.1,
                               n_jobs=a.threads, verbose=-1, min_child_samples=20)
        _, s = timed_fit(lambda: m.fit(ordinal[tr], ytr), a.repeat)
        add("GBDT (leaf-wise + GOSS)", "lightgbm", roc_auc_score(yva, m.predict_proba(ordinal[va])[:, 1]), s)

    if "rf" in want:
        # ---- Random forest ---------------------------------------------------
        auc, fit_s, bin_s, err = run_zarbor(a.binary, split_csv, label, [
            "--algo=random_forest", f"--n_rounds={RF_TREES}", f"--max_leaves={RF_LEAVES}",
            f"--max_bin={MAX_BIN}", f"--n_threads={a.threads}"], a.repeat)
        add("RandomForest", "zarbor", auc, fit_s, err or binnote(bin_s))

        m = RandomForestClassifier(n_estimators=RF_TREES, max_leaf_nodes=RF_LEAVES,
                                   max_features="sqrt", bootstrap=True, n_jobs=a.threads,
                                   random_state=a.seed)
        _, s = timed_fit(lambda: m.fit(ordinal[tr], ytr), a.repeat)
        add("RandomForest", "sklearn", roc_auc_score(yva, m.predict_proba(ordinal[va])[:, 1]), s)

    if "linear" in want:
        # ---- Linear ----------------------------------------------------------
        auc, fit_s, bin_s, err = run_zarbor(a.binary, split_csv, label, [
            "--algo=linear", "--lin_epochs=300", f"--lambda={L2}", f"--n_threads={a.threads}"], a.repeat)
        add("Linear (logistic)", "zarbor", auc, fit_s, err or binnote(bin_s))

        m = LogisticRegression(max_iter=1000, C=1.0)  # n_jobs is a no-op since 1.8
        _, s = timed_fit(lambda: m.fit(onehot[tr], ytr), a.repeat)
        add("Linear (logistic)", "sklearn", roc_auc_score(yva, m.predict_proba(onehot[va])[:, 1]), s)

    # ---- report ----------------------------------------------------------
    w = max(len(r["model"]) for r in rows) + 2
    print(f"{'model':<{w}} {'impl':<10} {'AUC':>9} {'fit(s)':>9}   note")
    print("-" * (w + 45))
    for i in range(0, len(rows), 2):
        for r in rows[i:i + 2]:
            auc = f"{r['auc']:.6f}" if r["auc"] is not None else "FAILED"
            fit = f"{r['fit_s']:.2f}" if r["fit_s"] is not None else "-"
            print(f"{r['model']:<{w}} {r['impl']:<10} {auc:>9} {fit:>9}   {r['note']}")
        z, p = rows[i], rows[i + 1]
        if z["auc"] is not None and p["auc"] is not None:
            d = z["auc"] - p["auc"]
            sp = p["fit_s"] / z["fit_s"] if z["fit_s"] else float("nan")
            print(f"{'':<{w}} {'->':<10} {d:>+9.6f} {sp:>8.2f}x   AUC delta, zarbor speed vs reference")
        print()

    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    sys.exit(main())
