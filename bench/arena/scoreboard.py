#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 AstraLibernis

"""Merge every model-vs-counterpart comparison into one scoreboard.

Rows 1-5 come from bench/arena/results.json (written by arena.py). Row 6 --
linear regression -- is measured here, because arena.py's dataset is binary.

Every row is the SAME dataset (EV purchases, 534,932 x 13) on the same
machine, which is the only way the time column is comparable down the page.
Linear leaves is deliberately excluded: it is measured on california, and a
16k-row fit sitting beside a 535k-row fit would make the bars lie.
"""
import json, re, statistics, subprocess, time
import tempfile
TMP = tempfile.mkdtemp(prefix="zarbor-bench-")  # private scratch, honours $TMPDIR
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
ARENA = ROOT / "bench" / "arena"
ZARBOR = ROOT / "zig-out" / "bin" / "zarbor"
R = 5


def med(x):
    return statistics.median(x)


def ms(o, k):
    m = re.search(rf"^{k}\s+(\d+) ms", o, re.M)
    return int(m.group(1)) if m else None


def linear_regression_row():
    tr = ARENA / "ev_reg_train.csv"; va = ARENA / "ev_reg_valid.csv"
    lab = "Annual_Income_USD"
    fits, preps, preds = [], [], []
    for i in range(R):
        cmd = [str(ZARBOR), str(tr), f"--label={lab}", "--valid-frac=0",
               "--algo=linear", "--objective=squared_error", "--lambda=0",
               "--alpha=0", "--verbose_eval=0", "--n_threads=16"]
        if i == 0:
            cmd.append(f"--save={TMP}/sb.zm")
        o = subprocess.run(cmd, capture_output=True, text=True)
        out = o.stdout + o.stderr
        fits.append(int(re.search(r"^train\s+(\d+) ms", out, re.M).group(1)))
        preps.append(ms(out, "bin"))
    pp = []
    for _ in range(R):
        o = subprocess.run([str(ZARBOR), "predict", str(va), f"--model={TMP}/sb.zm",
                            f"--out={TMP}/sb.csv", f"--label={lab}", "--n_threads=16"],
                           capture_output=True, text=True)
        out = o.stdout + o.stderr
        pp.append(ms(out, "predict")); preps.append(ms(out, "bin"))
    dtr = pd.read_csv(tr); dva = pd.read_csv(va)
    ytr = dtr.pop(lab).values.astype(float); yva = dva.pop(lab).values.astype(float)
    cats = [c for c in dtr.columns if dtr[c].dtype == object or str(dtr[c].dtype) == "str"]
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(dtr[cats])
    num = [c for c in dtr.columns if c not in cats]
    f = lambda d: np.hstack([d[num].to_numpy(float), oh.transform(d[cats])])
    Xtr, Xva = f(dtr), f(dva)
    sfit, sprep, spred = [], [], []
    for _ in range(R):
        t0 = time.perf_counter(); sc = StandardScaler().fit(Xtr)
        A, B = sc.transform(Xtr), sc.transform(Xva); sprep.append((time.perf_counter()-t0)*1000)
        m = LinearRegression()
        t0 = time.perf_counter(); m.fit(A, ytr); sfit.append((time.perf_counter()-t0)*1000)
        t0 = time.perf_counter(); p = m.predict(B); spred.append((time.perf_counter()-t0)*1000)
    zp = pd.read_csv(f"{TMP}/sb.csv").iloc[:, -1].values
    return dict(
        key="linear regression", reference="sklearn LinearRegression",
        kind="family", metric="rmse",
        zarbor_score=float(np.sqrt(mean_squared_error(yva, zp))),
        ref_score=float(np.sqrt(mean_squared_error(yva, p))),
        envelope=None,
        zarbor=dict(prepare=med(preps), fit=med(fits), predict=med(pp),
                    model_total=med(preps)+med(fits)+med(pp)),
        ref=dict(prepare=med(sprep), fit=med(sfit), predict=med(spred),
                 model_total=med(sprep)+med(sfit)+med(spred)))


NAMES = {
    "A gbdt-depthwise": ("GBDT, depthwise", "XGBoost", "auc"),
    "B gbdt-leafwise": ("GBDT, leafwise", "LightGBM", "auc"),
    "C gbdt-goss": ("GBDT + GOSS", "LightGBM GOSS", "auc"),
    "D forest (1024-leaf cap)": ("Random forest", "scikit-learn RF", "auc"),
    "E linear": ("Logistic regression", "scikit-learn LogReg", "auc"),
}


def main():
    src = json.loads((ARENA / "results.json").read_text())
    rows = []
    for k, (name, ref, metric) in NAMES.items():
        v = src[k]
        rows.append(dict(key=name, reference=ref, kind=v["kind"], metric=metric,
                         zarbor_score=v["zarbor_auc"], ref_score=v["ref_auc"],
                         envelope=v["envelope"],
                         zarbor=v["zarbor"], ref=v["ref"]))
    rows.append(linear_regression_row())
    (ARENA / "scoreboard.json").write_text(json.dumps(rows, indent=2))
    print(f"{'model':22} {'reference':24} {'zarbor':>10} {'ref':>12} "
          f"{'z ms':>7} {'ref ms':>7}  ratio")
    for r in rows:
        zt, rt = r["zarbor"]["model_total"], r["ref"]["model_total"]
        print(f"{r['key']:22} {r['reference']:24} {r['zarbor_score']:>10.6f} "
              f"{r['ref_score']:>12.6f} {zt:>7.0f} {rt:>7.0f}  {rt/zt:>5.2f}x")
    print(f"\nwrote {ARENA/'scoreboard.json'}")


if __name__ == "__main__":
    main()
