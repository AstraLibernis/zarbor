#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 AstraLibernis

"""The two standard simple models, against their standard implementations.

  logistic regression   --algo=linear --objective=logistic         vs sklearn LogisticRegression
  linear regression     --algo=linear --objective=squared_error    vs sklearn LinearRegression

Both of zarbor's solvers (`lbfgs`, `adam`) are run for each, because they
minimise the same convex objective and so must agree wherever both arrive --
see docs/linear-solvers.md.

Accounting follows docs/PROTOCOL-arena.md: prepare + fit + predict on both
sides (binning is paid by whoever does it), parsing measured separately and
charged to neither.

    python bench/arena/linear_models.py
"""
import re, statistics, subprocess, time
import tempfile
TMP = tempfile.mkdtemp(prefix="zarbor-bench-")  # private scratch, honours $TMPDIR
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
ARENA = ROOT / "bench" / "arena"
ZGBDT = ROOT / "zig-out" / "bin" / "zgbdt"
THREADS = 16
REPEATS = 5

# logistic is compared at lambda=1 / C=1, a pairing already validated to
# 1.6e-5 AUC. regression is compared unregularised, so that no penalty-scaling
# convention has to be assumed to hold between two libraries.
TASKS = [
    dict(key="logistic regression", model="logistic",
         train=ARENA / "ev_train.csv", valid=ARENA / "ev_valid_only.csv",
         label="Will_Buy_EV", metric="auc", lam=1.0,
         note="EV purchases, 534,932 x 13, binary"),
    dict(key="linear regression (standard set)", model="squared_error",
         train=ARENA / "cal_train.csv", valid=ARENA / "cal_valid.csv",
         label="y", metric="rmse", lam=0.0,
         note="california housing, 16,512 x 8"),
    dict(key="linear regression (at scale)", model="squared_error",
         train=ARENA / "ev_reg_train.csv", valid=ARENA / "ev_reg_valid.csv",
         label="Annual_Income_USD", metric="rmse", lam=0.0,
         note="EV income, 534,932 x 13"),
]
# adam is given whatever it needs to arrive, since handicapping it to lbfgs's
# budget would only re-measure that it is slower. Squared error needs far less
# than logistic: its Hessian is constant, so adam's per-coordinate adaptation
# has little to correct for.
ADAM_EPOCHS = dict(logistic=30000, squared_error=5000)


def med(x):
    return statistics.median(x)


def ms(out, key):
    m = re.search(rf"^{key}\s+(\d+) ms", out, re.M)
    return int(m.group(1)) if m else None


def zarbor(task, solver):
    mp = f"{TMP}/lm_{solver}.zm"
    fits, reads, bins, state = [], [], [], "ok"
    for i in range(REPEATS):
        cmd = [str(ZGBDT), str(task["train"]), f"--label={task['label']}",
               "--valid-frac=0", "--algo=linear", f"--objective={task['model']}",
               f"--lambda={task['lam']}", "--alpha=0", "--verbose_eval=0",
               f"--lin_solver={solver}", f"--n_threads={THREADS}"]
        if solver == "adam":
            cmd.append(f"--lin_epochs={ADAM_EPOCHS[task['model']]}")
        if i == 0:
            cmd.append(f"--save={mp}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        o = r.stdout + r.stderr
        if r.returncode != 0:
            raise RuntimeError(f"{task['key']}/{solver}:\n{o[-600:]}")
        if "STALLED" in o:
            state = "STALLED"
        reads.append(ms(o, "read")); bins.append(ms(o, "bin"))
        fits.append(int(re.search(r"^train\s+(\d+) ms", o, re.M).group(1)))

    preads, pbins, ppred = [], [], []
    for _ in range(REPEATS):
        r = subprocess.run([str(ZGBDT), "predict", str(task["valid"]),
                            f"--model={mp}", f"--out={TMP}/lm.csv",
                            f"--n_threads={THREADS}"], capture_output=True, text=True)
        o = r.stdout + r.stderr
        preads.append(ms(o, "read")); pbins.append(ms(o, "bin")); ppred.append(ms(o, "predict"))
    p = pd.read_csv(f"{TMP}/lm.csv").iloc[:, -1].values
    return dict(read=med(reads) + med(preads), prepare=med(bins) + med(pbins),
                fit=med(fits), predict=med(ppred), state=state, pred=p)


def sklearn_side(task, Xtr, ytr, Xva):
    fits, preps, preds = [], [], []
    p = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        sc = StandardScaler().fit(Xtr); A = sc.transform(Xtr); B = sc.transform(Xva)
        preps.append((time.perf_counter() - t0) * 1000)
        m = (LogisticRegression(C=1.0, solver="lbfgs", max_iter=300, tol=1e-7)
             if task["model"] == "logistic" else LinearRegression())
        t0 = time.perf_counter(); m.fit(A, ytr); fits.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        p = m.predict_proba(B)[:, 1] if task["model"] == "logistic" else m.predict(B)
        preds.append((time.perf_counter() - t0) * 1000)
    return dict(prepare=med(preps), fit=med(fits), predict=med(preds), state="ok", pred=p)


def main():
    print(f"n={REPEATS}, {THREADS} threads. "
          f"model time = prepare + fit + predict; parsing shown separately.\n")
    for task in TASKS:
        dtr = pd.read_csv(task["train"]); dva = pd.read_csv(task["valid"])
        lab = task["label"]
        if task["metric"] == "auc":
            ytr = (dtr.pop(lab) == "Yes").astype(int).values
            yva = (dva.pop(lab) == "Yes").astype(int).values
        else:
            ytr = dtr.pop(lab).values.astype(float)
            yva = dva.pop(lab).values.astype(float)
        cats = [c for c in dtr.columns if dtr[c].dtype == object or str(dtr[c].dtype) == "str"]
        if cats:
            oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(dtr[cats])
            num = [c for c in dtr.columns if c not in cats]
            f = lambda d: np.hstack([d[num].to_numpy(float), oh.transform(d[cats])])
        else:
            f = lambda d: d.to_numpy(float)
        Xtr, Xva = f(dtr), f(dva)

        t0 = time.perf_counter()
        pd.read_csv(task["train"]); pd.read_csv(task["valid"])
        pandas_read = (time.perf_counter() - t0) * 1000

        def score(p):
            return (roc_auc_score(yva, p) if task["metric"] == "auc"
                    else float(np.sqrt(mean_squared_error(yva, p))))

        rows = [("zarbor lbfgs", zarbor(task, "lbfgs")),
                ("zarbor adam", zarbor(task, "adam")),
                ("sklearn", sklearn_side(task, Xtr, ytr, Xva))]

        print(f"### {task['key']}  -- {task['note']}")
        print(f"    parse: zarbor {rows[0][1]['read']:.0f} ms | pandas {pandas_read:.0f} ms")
        print(f"    {'':14} {task['metric']:>10} {'prepare':>8} {'fit':>8} "
              f"{'predict':>8} {'model':>8}  state")
        base = None
        for name, r in rows:
            s = score(r["pred"])
            tot = r["prepare"] + r["fit"] + r["predict"]
            if base is None:
                base = s
            print(f"    {name:14} {s:10.6f} {r['prepare']:8.0f} {r['fit']:8.0f} "
                  f"{r['predict']:8.0f} {tot:8.0f}  {r['state']}")
        ls, as_, sk = [score(r["pred"]) for _, r in rows]
        print(f"    solvers meet: |lbfgs - adam| = {abs(ls - as_):.2e}   "
              f"vs sklearn: {abs(ls - sk):.2e}\n")


if __name__ == "__main__":
    main()
