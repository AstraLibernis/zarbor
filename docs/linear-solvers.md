# Two solvers, one objective: do they meet?

`--algo=linear` ships two fitters. They minimise the **same convex objective**,
and a convex problem has one optimum, so wherever both arrive they must agree —
on the coefficients, and therefore on anything computed from them.

| `--lin_solver` | what it is |
|---|---|
| `lbfgs` (default) | L-BFGS, with OWL-QN's orthant handling when `alpha > 0`. Line search per step. |
| `adam` | Full-batch Adam with an L1 proximal step. No objective evaluation, no line search. |

Stress test: `bench/arena/solver_stress.py`, 18 configurations (standardise
on/off x alpha in {0, 0.001, 0.1} x lambda in {0, 1, 100}) on a 60k-row
subsample, scored on 40k held-out rows.

## Where they meet

**On a standardised design, they meet.** Eight of nine configurations agree to
better than 2e-5 AUC — 0.937040 against 0.937040 with no regularisation, and
the same to five decimals across every L1/L2 setting. That is the convex
guarantee showing up as a measurement.

**Adam pays about 500x for it.** On the full 535k-row training set, reaching
the same answer took adam 30,000 epochs and 83 s against L-BFGS's 31
iterations and 156 ms, and it still stopped further out — `|g|max` 8.31e-5
against 3.07e-7. At the default 300 epochs adam is not close: 0.937066 against
0.937713.

## Where they do not, and what used to happen then

Turn standardisation off and the design is badly conditioned: one column
reaches 188,000 while others sit near 1. Nine of nine configurations break.

The point is not that they break. It is what each one *says* about it.

    lbfgs   AUC 0.6659   STALLED, every configuration, with the cause named
    adam    AUC 0.50-0.87, wildly varying with the regularisation, silent

Adam's models here ranged from 0.500000 — an exact coin flip — to 0.867930,
moving non-monotonically with `lambda`, which is the signature of a diverging
optimiser rather than a regularised one. Every one of them was saved with
`coefs 24 (0 zero)` and no warning.

### The cause was structural, not a tuning miss

`fitAdam` returned a bare `u32` epoch count, and the caller built its result
as `Fit{ .iters = ... }`. Every other field took its default: `converged =
true`, `g_first = 0`, `g_last = 0`. `Fit.stalled()` is

    !converged or g_last > 1e-3 * g_first

which for adam evaluated `!true or 0 > 0` — **false, unconditionally**. Adam
could not be reported as stalled whatever it did. The reporting site in
`main.zig` was additionally gated on `cfg.lin_solver == .lbfgs`, so the check
was disabled twice over.

This is precisely the defect `c3ebed5` fixed for L-BFGS ("a stalled solve is
no longer reported as a converged one"), left live in the other solver.

### The fix

`fitAdam` now computes the pseudo-gradient infinity norm each epoch — the same
quantity `fitLbfgs` records, so the threshold is calibrated on the same scale —
and returns a full `Fit`. `converged` stays `true` for adam, which has no line
search and therefore no "could not move" failure; the gradient ratio does the
work. The `== .lbfgs` gate is gone.

It catches eight of the nine broken configurations, where it previously caught
none.

## The honest limit: the statistic does not fully separate for adam

It does not catch all nine, and tightening the constant will not fix that.
Measured `g_last / g_first` across the grid:

| | min | max |
|---|---:|---:|
| adam, standardised (healthy) | 1.32e-06 | **2.80e-03** |
| adam, unstandardised (broken) | **5.40e-04** | 1.08e+01 |

**The ranges overlap.** A healthy adam run reaches 2.80e-03; a broken one gets
as low as 5.40e-04. No threshold on this statistic separates them, so the
1e-3 line inherited from L-BFGS both misses one broken configuration
(`alpha=0.001, lambda=0`, which returns AUC 0.826 against a healthy 0.937 and
reports `ok`) and flags one healthy one (`alpha=0.1, lambda=0`, off by 1.7e-05
in AUC and harmless).

Why it separates cleanly for L-BFGS and not for adam: L-BFGS stops when its
line search fails, which on a well-scaled problem means it is at the optimum
and on a badly-scaled one means it is stuck — two states far apart in gradient
terms. Adam stops when its fixed schedule runs out, wherever that happens to
be. Its adaptive per-coordinate steps shrink the gradient substantially
without reaching the optimum, so a partially-converged run and a healthy one
look alike by this measure.

Tuning the 1e-3 constant until this grid comes out clean would be fitting the
diagnostic to one dataset, which is the trap `docs/vs-lightgbm.md` exists to
warn about. Named as a limit, not patched over.

**What would actually close it** is a criterion adam does not currently have:
a relative objective change across epochs, or a bound on the optimality gap.
Both need the objective value, which `fitAdam` deliberately skips computing —
that is what makes its epochs cheap. Not attempted.

## Practical guidance

Use `lbfgs`. It is 500x faster to the same answer, its convergence report
separates cleanly, and it is what `--algo=linear` defaults to. `adam` is a
first-order fallback worth keeping for problems where the line search is the
difficulty, and it should now be read with `STALLED` taken seriously and
silence *not* taken as proof of success.

Keep `--lin_standardize=true`, which is the default. Every failure in this
document is downstream of turning it off.
