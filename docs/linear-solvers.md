# The two simple models, their solvers, and whether any of it agrees

`--algo=linear` fits **the two standard simple models**, selected by
`--objective`. They are the same machinery -- a weighted sum of features --
differing in the loss and in whether a sigmoid link is applied:

| flag | model | scikit-learn equivalent |
|---|---|---|
| `--objective=logistic` | **logistic regression** | `LogisticRegression` |
| `--objective=squared_error` | **linear regression** (ordinary least squares) | `LinearRegression`, `Ridge` with `lambda>0`, `Lasso` with `alpha>0` |

`Config.modelName()` prints this at train time, so a run says
`model   logistic regression (L2 / ridge)` rather than leaving it to be
reassembled from flags.

## Against the standard implementations

`bench/arena/linear_models.py`, n=5, 16 threads. Times in ms, accounted as in
`docs/PROTOCOL-arena.md`: prepare + fit + predict on both sides, parsing
separate. Logistic is compared at `lambda=1` / `C=1`; regression is compared
unregularised so no penalty-scaling convention has to be assumed to carry
between two libraries.

### logistic regression -- EV purchases, 534,932 x 13, AUC

| | AUC | prepare | fit | predict | model |
|---|---:|---:|---:|---:|---:|
| zarbor lbfgs | 0.937713 | 45 | **154** | 1 | **200** |
| zarbor adam | 0.937711 | 45 | 83225 | 1 | 83271 |
| sklearn | 0.937729 | 76 | 885 | 8 | 969 |

Agreement with sklearn: **1.56e-05 AUC**. zarbor is **4.8x faster** on model
work. Parsing, charged to neither: zarbor 141 ms against pandas 225 ms.

### linear regression -- california housing, 16,512 x 8, RMSE

| | RMSE |
|---|---:|
| zarbor lbfgs | **0.721291** |
| zarbor adam | 0.721291 |
| sklearn `LinearRegression` | 0.742607 |

Every fit here is 1-9 ms, too small to time meaningfully. zarbor is **0.0213
better**, and that is not noise -- see "binning is doing the work" below.

### linear regression at scale -- EV income, 534,932 x 13, RMSE

| | RMSE | prepare | fit | predict | model |
|---|---:|---:|---:|---:|---:|
| zarbor lbfgs | 27922.135986 | 35 | 199 | 0 | 234 |
| zarbor adam | 28536.595745 | 34 | 12118 | 0 | 12152 (**STALLED**) |
| sklearn | 27922.154109 | 71 | **107** | 4 | **182** |

Agreement with sklearn: 1.81e-02 on a scale of 27,922 -- **6.5e-07
relative**. The two are computing the same thing.

**sklearn is 1.29x faster here, and the reason is structural.** Ordinary least
squares has a closed-form solution; `LinearRegression` takes a direct
least-squares solve. zarbor runs L-BFGS at it, iterating toward an answer that
could be obtained in one factorisation. That is the right call for logistic,
which has no closed form, and unnecessary work for squared error. Recorded as
a lead, not fixed: the normal equations on the binned design are exactly what
`tree.zig` already solves by Cholesky for linear leaves.

## Binning is doing the work on california

The 0.0213 RMSE zarbor gains there is the binning, and sweeping the resolution
shows it directly:

| `max_bin` | 2 | 4 | 16 | 64 | 256 (default) | sklearn (raw) |
|---|---:|---:|---:|---:|---:|---:|
| RMSE | 1.157936 | 0.856906 | **0.693906** | 0.700796 | 0.721291 | 0.742607 |

RMSE is U-shaped in the bin count and approaches sklearn's raw-feature result
monotonically from `max_bin=16` upward -- which is what "more bins means less
binning" should look like. california's features are extremely skewed
(`AveOccup` skew 97.6, `AveBedrms` 31.3, `AveRooms` 20.7), so quantile edges
bound the influence of a handful of extreme rows that ordinary least squares
fits at everyone else's expense.

`linear.zig`'s header claims binning is "a small loss of resolution and a real
gain in robustness". On the EV data that trade was worth 1.6e-05 AUC -- free.
Here it is worth 0.0213 RMSE at the default and 0.0487 at `max_bin=16`, and it
is a gain rather than a cost. The claim holds in both directions it was
measured.

## Two solvers, one objective: do they meet?

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

**The objective decides how hard adam's job is.** Squared error has a constant
Hessian, so adam's per-coordinate adaptation has almost nothing to correct
for and it converges in a few hundred epochs; on california it lands on
lbfgs's answer to **6.04e-08** RMSE. Logistic's Hessian varies with the
current fit, and there adam needs 30,000 epochs and **540x** the wall time
(83.2 s against 154 ms) to reach the same answer to 2.54e-06 AUC. The 540x
figure quoted below is a logistic-regression number and does not transfer.

At scale even squared error defeats it: on the 535k-row regression adam was
still **STALLED** after 5,000 epochs and 12.1 s, 614 RMSE short. It said so,
which is the entire point of the fix below -- before it, that run reported
success.

**On a standardised design, they meet.** Eight of nine configurations agree to
better than 2e-5 AUC — 0.937040 against 0.937040 with no regularisation, and
the same to five decimals across every L1/L2 setting. That is the convex
guarantee showing up as a measurement.

**Adam pays about 540x for it on logistic.** On the full 535k-row training
set, reaching the same answer took adam 30,000 epochs and 83.2 s against
L-BFGS's 31 iterations and 154 ms, and it still stopped further out — `|g|max` 8.31e-5
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
Both need the objective value, which `fitAdam` does not ask for: it calls
`pr.value(theta, false)`, where the `false` gates the loss accumulation.

An earlier version of this paragraph added "that is what makes its epochs
cheap". That was wrong. `want_loss` gates only a per-row accumulation *inside
a pass that already runs* -- scoring every row and forming the residuals
happens either way, because `grad` reads the residuals `value` leaves behind.
Measured on 2,000 logistic epochs over 534,932 rows, turning it on costs
**26%** (5.47-5.56 s against 6.94-6.96 s); for squared error it gates a single
multiply-add and is nearer free. What makes adam's epochs cheap is the absence
of a line search, not the absence of the loss.

The obstacle is therefore a real 26% on logistic rather than a structural one.
Still not attempted, but for want of doing it rather than because it cannot be
afforded.

## Practical guidance

Use `lbfgs`. On logistic it is 540x faster to the same answer; on squared
error the gap is smaller but it is still the only one of the two that
arrived on every problem tried here. Its convergence report separates
cleanly, and it is what `--algo=linear` defaults to. `adam` is a
first-order fallback worth keeping for problems where the line search is the
difficulty, and it should now be read with `STALLED` taken seriously and
silence *not* taken as proof of success.

Keep `--lin_standardize=true`, which is the default. Every failure in this
document is downstream of turning it off.
