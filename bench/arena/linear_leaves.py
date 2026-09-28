#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 AstraLibernis

"""zarbor's linear leaves against LightGBM's `linear_tree`.

Both fit an affine function of the root-to-leaf path's numeric features in
each leaf instead of emitting a constant. docs/RESULTS.md measured zarbor's
against its own constant-leaf baseline; this measures it against the other
implementation of the same idea, which is the check that was missing.

Both sides are swept over their leaf-ridge constant rather than compared at
one value, because the two constants are not on a known common scale and
picking one for the reference would be handicapping it.

    python bench/arena/linear_leaves.py
"""
import re, subprocess, time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
ARENA = ROOT / "bench" / "arena"
ZGBDT = ROOT / "zig-out" / "bin" / "zgbdt"
TRAIN, VALID = ARENA / "cal_train.csv", ARENA / "cal_valid.csv"
LAMBDAS = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

COMMON = dict(n_estimators=200, learning_rate=0.1, num_leaves=31, max_depth=-1,
              max_bin=255, reg_lambda=1.0, min_sum_hessian_in_leaf=1.0,
              min_child_samples=20, n_jobs=16, random_state=0, verbose=-1)
ZFLAGS = ["--objective=squared_error", "--n_rounds=200", "--learning_rate=0.1",
          "--grow_policy=lossguide", "--max_leaves=31", "--max_depth=0",
          "--lambda=1.0", "--min_child_weight=1.0", "--min_child_samples=20",
          "--max_bin=256", "--valid-frac=0", "--verbose_eval=0", "--n_threads=16"]


def zarbor(extra, repeats=3):
    fits = []
    for i in range(repeats):
        cmd = [str(ZGBDT), str(TRAIN), "--label=y", *ZFLAGS, *extra]
        if i == 0:
            cmd.append("--save=/tmp/llb.zm")
        o = subprocess.run(cmd, capture_output=True, text=True)
        fits.append(int(re.search(r"^train\s+(\d+) ms", o.stdout, re.M).group(1)))
    p = subprocess.run([str(ZGBDT), "predict", str(VALID), "--model=/tmp/llb.zm",
                        "--label=y", "--n_threads=16"], capture_output=True, text=True)
    return float(re.search(r"rmse=([0-9.]+)", p.stdout).group(1)), int(np.median(fits))


def lightgbm(tr, ytr, va, yva, kw, repeats=3):
    fits = []
    for _ in range(repeats):
        m = lgb.LGBMRegressor(**COMMON, **kw)
        t0 = time.perf_counter(); m.fit(tr, ytr); fits.append((time.perf_counter() - t0) * 1000)
    return float(np.sqrt(mean_squared_error(yva, m.predict(va)))), int(np.median(fits))


def main():
    tr = pd.read_csv(TRAIN); va = pd.read_csv(VALID)
    ytr = tr.pop("y").values; yva = va.pop("y").values

    zb, zbt = zarbor([])
    lb, lbt = lightgbm(tr, ytr, va, yva, {})
    print("constant leaves (the baseline each must beat to be worth anything)")
    print(f"  zarbor    rmse {zb:.6f}   fit {zbt} ms")
    print(f"  lightgbm  rmse {lb:.6f}   fit {lbt} ms\n")

    print(f"{'leaf ridge':>12} | {'zarbor':>10} {'fit':>6} | {'lightgbm':>10} {'fit':>6}")
    print("-" * 56)
    zs, ls = [], []
    for v in LAMBDAS:
        zr, zt = zarbor(["--linear_leaves=1", f"--lin_leaf_lambda={v}"])
        lr, lt = lightgbm(tr, ytr, va, yva, dict(linear_tree=True, linear_lambda=v))
        zs.append(zr); ls.append(lr)
        print(f"{v:>12} | {zr:>10.6f} {zt:>5}ms | {lr:>10.6f} {lt:>5}ms")
    print("-" * 56)
    print(f"  best   zarbor {min(zs):.6f} (vs its baseline {zb:.6f}, "
          f"{'better' if min(zs) < zb else 'worse'} by {abs(min(zs)-zb):.6f})")
    print(f"       lightgbm {min(ls):.6f} (vs its baseline {lb:.6f}, "
          f"{'better' if min(ls) < lb else 'worse'} by {abs(min(ls)-lb):.6f})")
    # How far the score moves as the leaf ridge is swept. A well-conditioned
    # solve should respond to a regularisation constant gently; an
    # ill-conditioned one blows up somewhere in the middle. Counting sign
    # changes does not separate them (both are 2 -- zarbor is U-shaped with a
    # tail wiggle), but the size of the excursion does.
    zr, lr = max(zs) - min(zs), max(ls) - min(ls)
    print(f"  sweep range: zarbor {zr:.6f}  lightgbm {lr:.6f}  ({lr/zr:.0f}x wider)")
    print(f"  worst cell as a multiple of own baseline: "
          f"zarbor {max(zs)/zb:.2f}x, lightgbm {max(ls)/lb:.2f}x")


if __name__ == "__main__":
    main()
