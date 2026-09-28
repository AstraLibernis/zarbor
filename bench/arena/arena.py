#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 AstraLibernis

"""zarbor against the reference implementation of each model it ships,
measured phase by phase so a difference can be attributed to a phase.

Protocol: docs/PROTOCOL-arena.md.  Results: docs/arena.md.

The accounting that makes this fair
-----------------------------------
XGBoost and LightGBM bin *inside* `fit()`. zarbor bins before it, and reports
the two separately. Comparing zarbor's `fit` against their `fit` therefore
charges them for work zarbor was not charged for -- an earlier version of this
file did exactly that and overstated zarbor's fit advantage. The model column
below is **prepare + fit + predict** on both sides, so binning is paid by
whoever does it, whenever they do it.

Parsing is not model work. Every model pays it before seeing a bin, so it is
measured once, on its own, against pandas -- not folded into any model's time.

zarbor trains with `--valid-frac=0 --verbose_eval=0`, so it does no per-round
validation prediction or scoring. The reference side is fitted with no
`eval_set`. Neither side is charged for work the other is not doing.

    python bench/arena/arena.py            # full, n=5
    python bench/arena/arena.py --repeats 1 --skip-envelope --only A
"""
import argparse, json, re, statistics, subprocess, time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
ARENA = ROOT / "bench" / "arena"
ZGBDT = ROOT / "zig-out" / "bin" / "zgbdt"
TRAIN_CSV = ARENA / "ev_train.csv"
VALID_CSV = ARENA / "ev_valid_only.csv"
LABEL = "Will_Buy_EV"
THREADS = 16

P = dict(n_rounds=200, lr=0.1, depth=6, lam=1.0, mcw=1.0, mcs=20, max_bin=256)


def med(xs):
    return statistics.median(xs)


# ------------------------------------------------------------ build guard --
def check_release():
    """Refuse to time a Debug binary (~8x slower); this repo has already sent
    a conclusion wrong that way. Same guard as bench/compare.py."""
    r = subprocess.run([str(ZGBDT), str(TRAIN_CSV), f"--label={LABEL}",
                        "--n_rounds=1", "--verbose_eval=0", "--n_threads=1"],
                       capture_output=True, text=True)
    m = re.search(r"^build\s+(\w+)", r.stdout, re.M)
    if not m:
        raise SystemExit(f"{ZGBDT} reports no build mode; rebuild ReleaseFast")
    if m.group(1) == "Debug":
        raise SystemExit(f"{ZGBDT} is Debug (~8x slower); rebuild ReleaseFast")
    return m.group(1)


# -------------------------------------------------------------- zarbor ----
def _ms(out, key):
    m = re.search(rf"^{key}\s+(\d+) ms", out, re.M)
    if m is None:
        raise RuntimeError(f"no '{key}' timing in:\n{out[-800:]}")
    return int(m.group(1))


def zarbor(flags, tag, repeats):
    """Phase timings for train and predict, as zarbor reports them itself."""
    model = ARENA / f"{tag}.zm"
    pred = ARENA / f"{tag}.pred.csv"
    read_tr, bin_tr, fit = [], [], []
    for i in range(repeats):
        cmd = [str(ZGBDT), str(TRAIN_CSV), f"--label={LABEL}",
               "--valid-frac=0", "--verbose_eval=0", f"--n_threads={THREADS}",
               *flags]
        if i == 0:
            cmd.append(f"--save={model}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        out = r.stdout + r.stderr
        if r.returncode != 0:
            raise RuntimeError(f"{tag} train failed:\n{out[-800:]}")
        read_tr.append(_ms(out, "read"))
        bin_tr.append(_ms(out, "bin"))
        fit.append(int(re.search(r"^train\s+(\d+) ms", out, re.M).group(1)))

    read_va, bin_va, pr = [], [], []
    for _ in range(repeats):
        # --label makes zarbor score the holdout itself, so its number can be
        # checked against the referee's on the very same predictions. Without
        # it a metric bug on either side has nowhere to show up.
        r = subprocess.run([str(ZGBDT), "predict", str(VALID_CSV),
                            f"--model={model}", f"--out={pred}",
                            f"--label={LABEL}", f"--n_threads={THREADS}"],
                           capture_output=True, text=True)
        out = r.stdout + r.stderr
        if r.returncode != 0:
            raise RuntimeError(f"{tag} predict failed:\n{out[-800:]}")
        self_auc = re.search(r"score\s+auc=([0-9.]+)", out)
        read_va.append(_ms(out, "read"))
        bin_va.append(_ms(out, "bin"))
        pr.append(_ms(out, "predict"))

    p = pd.read_csv(pred)
    return dict(read=med(read_tr) + med(read_va),
                prepare=med(bin_tr) + med(bin_va),
                fit=med(fit), predict=med(pr),
                fit_range=(min(fit), max(fit)),
                self_auc=float(self_auc.group(1)) if self_auc else None,
                pred=p.iloc[:, -1].values)


# ---------------------------------------------------------- python side ----
def py_phases(prep_tr, prep_va, make, ytr, repeats):
    """prepare / fit / predict, each timed on its own. Binning inside fit()
    stays inside fit(), which is the point of charging zarbor for its bin."""
    prep, fits, preds = [], [], []
    p = None
    for _ in range(repeats):
        t0 = time.perf_counter(); Xtr = prep_tr(); Xva = prep_va()
        prep.append((time.perf_counter() - t0) * 1000)
        m = make()
        t0 = time.perf_counter(); m.fit(Xtr, ytr)
        fits.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter(); p = m.predict_proba(Xva)[:, 1]
        preds.append((time.perf_counter() - t0) * 1000)
    return dict(prepare=med(prep), fit=med(fits), predict=med(preds),
                fit_range=(min(fits), max(fits)), pred=p)


def xgb_model(max_bin=None, **kw):
    return xgb.XGBClassifier(
        n_estimators=P["n_rounds"], learning_rate=P["lr"], max_depth=P["depth"],
        tree_method="hist", grow_policy="depthwise", max_bin=max_bin or P["max_bin"],
        reg_lambda=P["lam"], reg_alpha=0.0, min_child_weight=P["mcw"],
        subsample=1.0, colsample_bytree=1.0, n_jobs=THREADS,
        enable_categorical=True, max_cat_to_onehot=1, eval_metric="auc",
        random_state=0, verbosity=0, **kw)


def lgb_model(max_bin=None, goss=False, **kw):
    # LightGBM's max_bin excludes its missing bin; zarbor's and XGBoost's
    # include it, so 255 here is 256 there.
    return lgb.LGBMClassifier(
        n_estimators=P["n_rounds"], learning_rate=P["lr"], num_leaves=31,
        max_depth=-1, max_bin=(max_bin or P["max_bin"]) - 1,
        reg_lambda=P["lam"], reg_alpha=0.0,
        min_sum_hessian_in_leaf=P["mcw"], min_child_samples=P["mcs"],
        subsample=1.0, colsample_bytree=1.0,
        boosting_type="goss" if goss else "gbdt", top_rate=0.2, other_rate=0.1,
        n_jobs=THREADS, random_state=0, verbose=-1,
        cat_smooth=10.0, cat_l2=10.0, min_data_per_group=P["mcs"], **kw)


def envelope(make, Xtr, ytr, Xva, yva):
    """Reference's own spread over max_bin {63,127,255}: the noise floor."""
    s = []
    for mb in (63, 127, 255):
        m = make(max_bin=mb); m.fit(Xtr, ytr)
        s.append(roc_auc_score(yva, m.predict_proba(Xva)[:, 1]))
    return max(s) - min(s)


# ---------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--skip-envelope", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    print(f"zgbdt build: {check_release()}", flush=True)

    # ---- the parser, measured on its own ----------------------------
    # dtype=category, not a plain read then .astype(). zarbor's parser builds
    # its categorical level dictionaries *during* the parse, so pandas must be
    # allowed to do the same or the comparison charges one side for work the
    # other hid in another phase. It is also faster this way (136 vs 180 ms):
    # storing category codes beats storing object strings.
    CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
            "Subsidy_Available", "Range_Anxiety_Level", LABEL]
    DT = {c: "category" for c in CATS}

    def read_both():
        return pd.read_csv(TRAIN_CSV, dtype=DT), pd.read_csv(VALID_CSV, dtype=DT)
    rt = []
    for _ in range(a.repeats):
        t0 = time.perf_counter(); dtr, dva = read_both()
        rt.append((time.perf_counter() - t0) * 1000)
    pandas_read = med(rt)
    zr = []
    for _ in range(a.repeats):
        o = subprocess.run([str(ZGBDT), str(TRAIN_CSV), f"--label={LABEL}",
                            "--valid-frac=0", "--n_rounds=1", "--verbose_eval=0",
                            f"--n_threads={THREADS}"],
                           capture_output=True, text=True).stdout
        o2 = subprocess.run([str(ZGBDT), str(VALID_CSV), f"--label={LABEL}",
                             "--valid-frac=0", "--n_rounds=1", "--verbose_eval=0",
                             f"--n_threads={THREADS}"],
                            capture_output=True, text=True).stdout
        zr.append(_ms(o, "read") + _ms(o2, "read"))
    zarbor_read = med(zr)
    print(f"\nparser (both CSVs, 40 MB / 668,665 rows), n={a.repeats}")
    print(f"  zarbor csv.zig   {zarbor_read:7.0f} ms")
    print(f"  pandas read_csv  {pandas_read:7.0f} ms   "
          f"({pandas_read/zarbor_read:.2f}x zarbor)\n", flush=True)

    ytr = (dtr.pop(LABEL).astype(str) == "Yes").astype(int).values
    yva = (dva.pop(LABEL).astype(str) == "Yes").astype(int).values
    cats = [c for c in CATS if c != LABEL]

    # Already category dtype from the read; align valid's categories to
    # train's so the codes mean the same thing on both sides.
    for c in cats:
        dva[c] = dva[c].cat.set_categories(dtr[c].cat.categories)

    # No-ops: the categorical work happened in read_csv, as zarbor's does in
    # its parser. xgb/lgb bin inside fit(), so their prepare is genuinely nil.
    def cat_tr():
        return dtr

    def cat_va():
        return dva

    # sklearn needs an explicit encoding; fitted on train only, never combined.
    _oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(dtr[cats])
    _num = [c for c in dtr.columns if c not in cats]

    def oh(d):
        return np.hstack([d[_num].to_numpy(np.float64), _oh.transform(d[cats])])

    _sc = StandardScaler().fit(oh(dtr))
    results = {}

    def record(key, z, r, refname, kind, env=None):
        za, ra = roc_auc_score(yva, z["pred"]), roc_auc_score(yva, r["pred"])
        zmodel = z["prepare"] + z["fit"] + z["predict"]
        rmodel = r["prepare"] + r["fit"] + r["predict"]
        ztot, rtot = zmodel + z["read"], rmodel + pandas_read
        results[key] = dict(
            kind=kind, reference=refname, zarbor_auc=za,
            zarbor_self_auc=z["self_auc"], ref_auc=ra,
            gap=abs(za - ra), envelope=env,
            zarbor=dict(read=z["read"], prepare=z["prepare"], fit=z["fit"],
                        predict=z["predict"], model_total=zmodel, total=ztot),
            ref=dict(read=pandas_read, prepare=r["prepare"], fit=r["fit"],
                     predict=r["predict"], model_total=rmodel, total=rtot))
        e = f"  env {env:.6f}" if env else ""
        print(f"  {key}")
        agree = (z["self_auc"] is not None
                 and abs(z["self_auc"] - za) < 1e-6)
        print(f"    auc    zarbor {za:.6f} | {refname} {ra:.6f} | "
              f"gap {abs(za-ra):.6f}{e}  [self=referee: {agree}]")
        print(f"    prepare{z['prepare']:7.0f} |{r['prepare']:8.0f}   "
              f"fit{z['fit']:7.0f} |{r['fit']:8.0f}   "
              f"predict{z['predict']:6.0f} |{r['predict']:7.0f}")
        print(f"    MODEL  {zmodel:7.0f} |{rmodel:8.0f} ms   "
              f"({rmodel/zmodel:.2f}x zarbor)")
        print(f"    TOTAL  {ztot:7.0f} |{rtot:8.0f} ms   "
              f"({rtot/ztot:.2f}x zarbor, incl. parse)", flush=True)

    cz = [f"--n_rounds={P['n_rounds']}", f"--learning_rate={P['lr']}",
          f"--lambda={P['lam']}", f"--min_child_weight={P['mcw']}",
          f"--min_child_samples={P['mcs']}", f"--max_bin={P['max_bin']}"]

    if not only or "A" in only:
        print("A  depthwise GBDT  (parity, vs xgboost)", flush=True)
        z = zarbor(cz + [f"--max_depth={P['depth']}", "--grow_policy=depthwise"], "A", a.repeats)
        r = py_phases(cat_tr, cat_va, xgb_model, ytr, a.repeats)
        env = None if a.skip_envelope else envelope(
            lambda max_bin: xgb_model(max_bin=max_bin), cat_tr(), ytr, cat_va(), yva)
        record("A gbdt-depthwise", z, r, "xgboost", "parity", env)

    if not only or "B" in only:
        print("B  leafwise GBDT  (parity, vs lightgbm)", flush=True)
        zf = cz + ["--grow_policy=lossguide", "--max_leaves=31", "--max_depth=0",
                   "--cat_split=optimal", "--bin_policy=greedy"]
        z = zarbor(zf, "B", a.repeats)
        r = py_phases(cat_tr, cat_va, lgb_model, ytr, a.repeats)
        env = None if a.skip_envelope else envelope(
            lambda max_bin: lgb_model(max_bin=max_bin), cat_tr(), ytr, cat_va(), yva)
        record("B gbdt-leafwise", z, r, "lightgbm", "parity", env)

    if not only or "C" in only:
        print("C  leafwise GBDT + GOSS  (parity, vs lightgbm)", flush=True)
        zf = cz + ["--grow_policy=lossguide", "--max_leaves=31", "--max_depth=0",
                   "--cat_split=optimal", "--bin_policy=greedy", "--sampling=goss"]
        z = zarbor(zf, "C", a.repeats)
        r = py_phases(cat_tr, cat_va, lambda: lgb_model(goss=True), ytr, a.repeats)
        env = None if a.skip_envelope else envelope(
            lambda max_bin: lgb_model(max_bin=max_bin, goss=True), cat_tr(), ytr, cat_va(), yva)
        record("C gbdt-goss", z, r, "lightgbm-goss", "parity", env)

    if not only or "D" in only:
        print("D  random forest  (family, vs sklearn)", flush=True)
        z = zarbor(["--algo=random_forest", "--n_rounds=300",
                    f"--max_bin={P['max_bin']}"], "D", a.repeats)
        mk = lambda cap: (lambda: RandomForestClassifier(
            n_estimators=300, max_features="sqrt", bootstrap=True,
            min_samples_leaf=1, max_leaf_nodes=cap, n_jobs=THREADS, random_state=0))
        r = py_phases(lambda: oh(dtr), lambda: oh(dva), mk(1024), ytr, a.repeats)
        record("D forest (1024-leaf cap)", z, r, "sklearn-rf-capped", "family")
        r = py_phases(lambda: oh(dtr), lambda: oh(dva), mk(None), ytr, a.repeats)
        record("D forest (uncapped ref)", z, r, "sklearn-rf-default", "family")

    if not only or "E" in only:
        print("E  regularised logistic  (family, vs sklearn)", flush=True)
        z = zarbor(["--algo=linear"], "E", a.repeats)
        r = py_phases(lambda: _sc.transform(oh(dtr)), lambda: _sc.transform(oh(dva)),
                      lambda: LogisticRegression(C=1.0, solver="lbfgs",
                                                 max_iter=300, tol=1e-7),
                      ytr, a.repeats)
        record("E linear", z, r, "sklearn-logreg", "family")

    results["_parser"] = dict(zarbor_ms=zarbor_read, pandas_ms=pandas_read,
                              ratio=pandas_read / zarbor_read)
    (ARENA / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {ARENA/'results.json'}")


if __name__ == "__main__":
    main()
