#!/usr/bin/env python3
"""zarbor against the reference implementation of each model it ships.

Protocol: docs/PROTOCOL-arena.md (pre-registered before any run).
Results:  docs/arena.md

    python bench/arena/arena.py            # full, n=5
    python bench/arena/arena.py --repeats 1 --skip-envelope   # smoke
"""
import argparse, json, re, subprocess, statistics, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
ARENA = ROOT / "bench" / "arena"
ZGBDT = ROOT / "zig-out" / "bin" / "zgbdt"
CSV = ARENA / "ev.csv"
VALID_CSV = ARENA / "ev_valid.csv"
LABEL = "Will_Buy_EV"
THREADS = 16

# Matched deliberately. Defaults differ across all four libraries, so no
# default is used for anything both sides have.
P = dict(n_rounds=200, lr=0.1, depth=6, lam=1.0, mcw=1.0, mcs=20, max_bin=256)


# ------------------------------------------------------------ build guard --
def check_release():
    """Refuse to time a Debug binary. `zig build` with no flags produces one,
    and it is ~8x slower here -- this repo has already sent a conclusion badly
    wrong that way. Same check as bench/compare.py, for the same reason."""
    r = subprocess.run([str(ZGBDT), str(CSV), f"--label={LABEL}", "--n_rounds=1",
                        "--verbose_eval=0", "--n_threads=1"],
                       capture_output=True, text=True)
    m = re.search(r"^build\s+(\w+)", r.stdout, re.M)
    if not m:
        raise SystemExit(f"{ZGBDT} reports no build mode. Rebuild with "
                         "zig build -Doptimize=ReleaseFast")
    if m.group(1) == "Debug":
        raise SystemExit(f"{ZGBDT} is a Debug build (~8x slower). Rebuild with "
                         "zig build -Doptimize=ReleaseFast")
    return m.group(1)


# ---------------------------------------------------------------- data ----
def load():
    df = pd.read_csv(CSV)
    y = (df.pop(LABEL) == "Yes").astype(int).values
    sp = df.pop("__split").values
    cats = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "str"]
    return df, y, sp, cats


def write_valid(df, y, sp):
    """Valid rows only, for `zgbdt predict` to score independently."""
    if VALID_CSV.exists():
        return
    full = pd.read_csv(CSV)
    full[full.__split != 0].to_csv(VALID_CSV, index=False)


# -------------------------------------------------------------- zarbor ----
FIT_RE = re.compile(r"^fit\s+(\d+) ms", re.M)
AUC_RE = re.compile(r"valid\s+auc=([0-9.]+)")


def zarbor(flags, tag, repeats):
    """Returns (fit_ms median, wall_s median, self_auc, referee_auc, ranges)."""
    model = ARENA / f"{tag}.zm"
    fits, walls = [], []
    self_auc = None
    for i in range(repeats):
        cmd = [str(ZGBDT), str(CSV), f"--label={LABEL}", "--split-col=__split",
               f"--n_threads={THREADS}", *flags]
        if i == 0:
            cmd.append(f"--save={model}")
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        walls.append(time.perf_counter() - t0)
        out = r.stdout + r.stderr
        m = FIT_RE.search(out)
        if not m:
            raise RuntimeError(f"{tag}: no fit line\n{out[-800:]}")
        fits.append(int(m.group(1)))
        a = AUC_RE.search(out)
        if a:
            self_auc = float(a.group(1))

    # Independent scoring path: reload the saved model, predict, referee it.
    pred = ARENA / f"{tag}.pred.csv"
    r = subprocess.run([str(ZGBDT), "predict", str(VALID_CSV), f"--model={model}",
                        f"--out={pred}", f"--n_threads={THREADS}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{tag}: predict failed\n{(r.stdout+r.stderr)[-800:]}")
    p = pd.read_csv(pred)
    col = [c for c in p.columns if c.lower() in ("pred", "prediction", "p", "score")]
    p = p[col[0]].values if col else p.iloc[:, -1].values
    return dict(fit_ms=statistics.median(fits), wall_s=statistics.median(walls),
                self_auc=self_auc, pred=p,
                fit_range=(min(fits), max(fits)),
                wall_range=(min(walls), max(walls)))


# ---------------------------------------------------------- python side ----
def timed_fit(make, X, ytr, Xva, repeats, proba=True):
    """fit_ms is the fit call alone; work_s is fit + predict, to which the
    caller adds the shared load+encode cost for a true end-to-end."""
    fits, works = [], []
    p = None
    for _ in range(repeats):
        m = make()
        t0 = time.perf_counter()
        m.fit(X, ytr)
        fits.append((time.perf_counter() - t0) * 1000)
        p = m.predict_proba(Xva)[:, 1] if proba else m.predict(Xva)
        works.append(time.perf_counter() - t0)
    return dict(fit_ms=statistics.median(fits), pred=p,
                work_s=statistics.median(works),
                work_range=(min(works), max(works)),
                fit_range=(min(fits), max(fits)))


def xgb_model(max_bin=None, **kw):
    return xgb.XGBClassifier(
        n_estimators=P["n_rounds"], learning_rate=P["lr"], max_depth=P["depth"],
        tree_method="hist", grow_policy="depthwise", max_bin=max_bin or P["max_bin"],
        reg_lambda=P["lam"], reg_alpha=0.0, min_child_weight=P["mcw"],
        subsample=1.0, colsample_bytree=1.0, n_jobs=THREADS,
        enable_categorical=True, max_cat_to_onehot=1, eval_metric="auc",
        random_state=0, verbosity=0, **kw)


def lgb_model(max_bin=None, goss=False, **kw):
    # LightGBM's max_bin excludes its missing bin where zarbor's and XGBoost's
    # include it, so 255 here is 256 there. Recorded in the protocol.
    return lgb.LGBMClassifier(
        n_estimators=P["n_rounds"], learning_rate=P["lr"], num_leaves=31,
        max_depth=-1, max_bin=(max_bin or P["max_bin"]) - 1,
        reg_lambda=P["lam"], reg_alpha=0.0,
        min_sum_hessian_in_leaf=P["mcw"], min_child_samples=P["mcs"],
        subsample=1.0, colsample_bytree=1.0,
        boosting_type="goss" if goss else "gbdt",
        top_rate=0.2, other_rate=0.1,
        n_jobs=THREADS, random_state=0, verbose=-1,
        cat_smooth=10.0, cat_l2=10.0, min_data_per_group=P["mcs"], **kw)


# ------------------------------------------------------------ envelope ----
def envelope(make, X, ytr, Xva, yva):
    """Reference's own spread over max_bin in {63,127,255}: the noise floor."""
    scores = []
    for mb in (63, 127, 255):
        m = make(max_bin=mb)
        m.fit(X, ytr)
        scores.append(roc_auc_score(yva, m.predict_proba(Xva)[:, 1]))
    return max(scores) - min(scores), scores


# ---------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--skip-envelope", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    print(f"zgbdt build: {check_release()}", flush=True)

    load_times = []
    for _ in range(a.repeats):
        t0 = time.perf_counter()
        _d, _y, _s, _c = load()
        for c in _c:
            _d[c] = _d[c].astype("category")
        load_times.append(time.perf_counter() - t0)
    LOAD_S = statistics.median(load_times)
    print(f"python load+encode: {LOAD_S:.2f} s  (n={a.repeats}, "
          f"range {min(load_times):.2f}-{max(load_times):.2f})", flush=True)

    df, y, sp, cats = load()
    write_valid(df, y, sp)
    tr, va = sp == 0, sp != 0
    yva = y[va]

    # Native categorical dtypes for xgb/lgb; one-hot + scaled for sklearn.
    Xcat = df.copy()
    for c in cats:
        Xcat[c] = Xcat[c].astype("category")
    Xoh = pd.get_dummies(df, columns=cats, drop_first=False).astype(np.float64)
    sc = StandardScaler().fit(Xoh.values[tr])
    Xsc = sc.transform(Xoh.values)

    results = {}

    def record(key, z, ref, refname, kind, env=None):
        z_ref_auc = roc_auc_score(yva, z["pred"])
        r_auc = roc_auc_score(yva, ref["pred"])
        results[key] = dict(
            kind=kind, reference=refname,
            zarbor_auc=z_ref_auc, zarbor_self_auc=z["self_auc"],
            ref_auc=r_auc, gap=abs(z_ref_auc - r_auc),
            envelope=env,
            zarbor_fit_ms=z["fit_ms"], ref_fit_ms=ref["fit_ms"],
            zarbor_fit_range=z["fit_range"], ref_fit_range=ref["fit_range"],
            zarbor_wall_s=z["wall_s"], zarbor_wall_range=z["wall_range"],
            ref_work_s=ref["work_s"], ref_work_range=ref["work_range"],
            python_load_s=LOAD_S, ref_end_to_end_s=ref["work_s"] + LOAD_S)
        e = f" env {env:.6f}" if env else ""
        print(f"  {key:22s} zarbor auc {z_ref_auc:.6f} ({z['fit_ms']:.0f} ms) | "
              f"{refname} {r_auc:.6f} ({ref['fit_ms']:.0f} ms) | "
              f"gap {abs(z_ref_auc-r_auc):.6f}{e}\n"
              f"  {'':22s} end-to-end: zarbor {z['wall_s']:.2f} s | "
              f"{refname} {ref['work_s']+LOAD_S:.2f} s", flush=True)

    common_z = [f"--n_rounds={P['n_rounds']}", f"--learning_rate={P['lr']}",
                f"--lambda={P['lam']}", f"--min_child_weight={P['mcw']}",
                f"--min_child_samples={P['mcs']}", f"--max_bin={P['max_bin']}"]

    # ---- A: depthwise GBDT vs XGBoost --------------------------------
    if not only or "A" in only:
        print("A  depthwise GBDT  (parity)", flush=True)
        z = zarbor(common_z + [f"--max_depth={P['depth']}", "--grow_policy=depthwise"],
                   "A_zarbor", a.repeats)
        r = timed_fit(lambda: xgb_model(), Xcat[tr], y[tr], Xcat[va], a.repeats)
        env = None if a.skip_envelope else envelope(
            lambda max_bin: xgb_model(max_bin=max_bin), Xcat[tr], y[tr], Xcat[va], yva)[0]
        record("A gbdt-depthwise", z, r, "xgboost", "parity", env)

    # ---- B: leafwise GBDT vs LightGBM --------------------------------
    if not only or "B" in only:
        print("B  leafwise GBDT  (parity)", flush=True)
        zflags = common_z + ["--grow_policy=lossguide", "--max_leaves=31",
                             "--max_depth=0", "--cat_split=optimal",
                             "--bin_policy=greedy"]
        z = zarbor(zflags, "B_zarbor", a.repeats)
        r = timed_fit(lambda: lgb_model(), Xcat[tr], y[tr], Xcat[va], a.repeats)
        env = None if a.skip_envelope else envelope(
            lambda max_bin: lgb_model(max_bin=max_bin), Xcat[tr], y[tr], Xcat[va], yva)[0]
        record("B gbdt-leafwise", z, r, "lightgbm", "parity", env)

    # ---- C: leafwise + GOSS ------------------------------------------
    if not only or "C" in only:
        print("C  leafwise GBDT + GOSS  (parity)", flush=True)
        zflags = common_z + ["--grow_policy=lossguide", "--max_leaves=31",
                             "--max_depth=0", "--cat_split=optimal",
                             "--bin_policy=greedy", "--sampling=goss"]
        z = zarbor(zflags, "C_zarbor", a.repeats)
        r = timed_fit(lambda: lgb_model(goss=True), Xcat[tr], y[tr], Xcat[va], a.repeats)
        env = None if a.skip_envelope else envelope(
            lambda max_bin: lgb_model(max_bin=max_bin, goss=True),
            Xcat[tr], y[tr], Xcat[va], yva)[0]
        record("C gbdt-goss", z, r, "lightgbm-goss", "parity", env)

    # ---- D: random forest vs sklearn ---------------------------------
    if not only or "D" in only:
        print("D  random forest  (family)", flush=True)
        z = zarbor(["--algo=random_forest", "--n_rounds=300",
                    f"--max_bin={P['max_bin']}"], "D_zarbor", a.repeats)
        mk = lambda cap: (lambda: RandomForestClassifier(
            n_estimators=300, max_features="sqrt", bootstrap=True,
            min_samples_leaf=1, max_leaf_nodes=cap, n_jobs=THREADS,
            random_state=0))
        r_cap = timed_fit(mk(1024), Xoh.values[tr], y[tr], Xoh.values[va], a.repeats)
        record("D forest (1024-leaf cap)", z, r_cap, "sklearn-rf-capped", "family")
        r_unc = timed_fit(mk(None), Xoh.values[tr], y[tr], Xoh.values[va], a.repeats)
        record("D forest (uncapped ref)", z, r_unc, "sklearn-rf-default", "family")

    # ---- E: linear vs sklearn ----------------------------------------
    if not only or "E" in only:
        print("E  regularised logistic  (family)", flush=True)
        z = zarbor(["--algo=linear"], "E_zarbor", a.repeats)
        mk = lambda: LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                        max_iter=300, tol=1e-7, n_jobs=THREADS)
        r_raw = timed_fit(mk, Xsc[tr], y[tr], Xsc[va], a.repeats)
        record("E linear (raw scaled)", z, r_raw, "sklearn-logreg-raw", "family")

    (ARENA / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {ARENA/'results.json'}")


if __name__ == "__main__":
    main()
