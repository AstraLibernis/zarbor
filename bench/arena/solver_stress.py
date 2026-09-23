#!/usr/bin/env python3
"""Do zarbor's two linear solvers meet?

L-BFGS/OWL-QN and Adam minimise the *same convex objective*. A convex problem
has one optimum, so on any configuration where both arrive they must agree --
on the coefficients, and therefore on any score computed from them. Where they
disagree, at least one did not arrive, and the interesting question is whether
it said so.

That last part is the whole point. A solver that is merely slow is a cost. A
solver that is wrong *and reports success* is a trap, and adam was one: it
returned `Fit{.iters}` only, leaving g_first = g_last = 0, so `Fit.stalled()`
evaluated 0 > 1e-3*0 and could never fire.

    python bench/arena/solver_stress.py
"""
import itertools, re, subprocess, sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
ARENA = ROOT / "bench" / "arena"
ZGBDT = ROOT / "zig-out" / "bin" / "zgbdt"
TRAIN = ARENA / "ev_small.csv"
VALID = ARENA / "ev_small_valid.csv"
LABEL = "Will_Buy_EV"

# adam needs orders of magnitude more passes than lbfgs for the same answer,
# so it is given them. Handicapping it to lbfgs's budget would prove nothing
# except that it is slower, which is already known and documented.
ADAM_EPOCHS = 20000
AGREE = 1e-3          # AUC; both solving the same convex problem

yva = (pd.read_csv(VALID)[LABEL] == "Yes").astype(int).values


def run(solver, standardize, alpha, lam):
    tag = f"/tmp/ss_{solver}.zm"
    cmd = [str(ZGBDT), str(TRAIN), f"--label={LABEL}", "--valid-frac=0",
           "--algo=linear", "--verbose_eval=0", "--n_threads=16",
           f"--lin_solver={solver}", f"--lin_standardize={str(standardize).lower()}",
           f"--alpha={alpha}", f"--lambda={lam}", f"--save={tag}"]
    if solver == "adam":
        cmd.append(f"--lin_epochs={ADAM_EPOCHS}")
    out = subprocess.run(cmd, capture_output=True, text=True)
    txt = out.stdout + out.stderr
    if out.returncode != 0:
        return None, "ERROR", None
    stalled = "STALLED" in txt
    g = re.search(r"\|g\|max ([0-9.e+-]+) from ([0-9.e+-]+)", txt)
    ratio = (float(g.group(1)) / float(g.group(2))) if g else None
    if not ratio:
        g2 = re.search(r"large \(([0-9.e+-]+), from ([0-9.e+-]+)", txt)
        ratio = (float(g2.group(1)) / float(g2.group(2))) if g2 else None

    pr = subprocess.run([str(ZGBDT), "predict", str(VALID), f"--model={tag}",
                         "--out=/tmp/ss.csv", "--n_threads=16"],
                        capture_output=True, text=True)
    if pr.returncode != 0:
        return None, "PREDFAIL", ratio
    auc = roc_auc_score(yva, pd.read_csv("/tmp/ss.csv").iloc[:, -1].values)
    return auc, ("STALLED" if stalled else "ok"), ratio


def main():
    grid = list(itertools.product([True, False], [0.0, 0.001, 0.1], [0.0, 1.0, 100.0]))
    print(f"{'std':>5} {'alpha':>7} {'lambda':>7} | "
          f"{'lbfgs auc':>10} {'state':>8} {'g_ratio':>9} | "
          f"{'adam auc':>10} {'state':>8} {'g_ratio':>9} | verdict")
    print("-" * 108)
    bad = 0
    for std, alpha, lam in grid:
        la, ls, lr_ = run("lbfgs", std, alpha, lam)
        aa, as_, ar = run("adam", std, alpha, lam)
        if la is None or aa is None:
            verdict = "RUN FAILED"
        elif ls == "ok" and as_ == "ok":
            d = abs(la - aa)
            verdict = "meet" if d < AGREE else f"DISAGREE {d:.5f}"
            if d >= AGREE:
                bad += 1
        elif ls == "ok" and as_ == "STALLED":
            verdict = "adam flagged itself"
        elif ls == "STALLED" and as_ == "STALLED":
            verdict = "both flagged"
        else:
            verdict = f"lbfgs {ls}, adam {as_}"
            # lbfgs stalled while adam claims ok: the dangerous direction
            if ls == "STALLED" and as_ == "ok":
                verdict += "  <-- SILENT"
                bad += 1
        f = lambda v: f"{v:.6f}" if v is not None else "  --  "
        g = lambda v: f"{v:.2e}" if v is not None else "  --  "
        print(f"{str(std):>5} {alpha:>7} {lam:>7} | {f(la):>10} {ls:>8} {g(lr_):>9} | "
              f"{f(aa):>10} {as_:>8} {g(ar):>9} | {verdict}")
    print("-" * 108)
    print(f"{len(grid)} configurations, {bad} where the two do not meet "
          f"with both claiming success")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
