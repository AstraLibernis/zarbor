# zarbor against the reference implementation of each model it ships

Protocol pre-registered in `docs/PROTOCOL-arena.md` before any number here was
produced. Harness: `bench/arena/arena.py`. Raw output: `bench/arena/results.json`.

Dataset: Kaggle Playground S6E9 (EV purchases), 668,665 x 13, binary, ROC-AUC.
534,932 train / 133,733 valid on one shared stratified split, seed 20260923.
Zero nulls, zero duplicate rows. Chosen for cleanliness over `kaggle-housing`;
the four reasons are in the protocol.

Hardware: Ryzen 7 9800X3D, 16 threads, all runs pinned to 16.
Versions: XGBoost 3.4.1, LightGBM 4.7.0, scikit-learn 1.9.1.
n=5 repeats throughout; ranges given, medians reported.

## Accuracy

`parity` rows are the same algorithm in two implementations, judged against
the reference's own max_bin envelope. `family` rows are deliberately different
constructions and get no envelope — see the protocol for why the distinction
governs how these are read.

| | kind | reference | zarbor | reference | gap | envelope | gap/env | |
|---|---|---|---:|---:|---:|---:|---:|---|
| A gbdt depthwise | parity | xgboost | 0.941323 | 0.941441 | 0.000118 | 0.000918 | 0.13 | agrees |
| B gbdt leafwise | parity | lightgbm | 0.941358 | 0.941427 | 0.000068 | 0.001103 | 0.06 | agrees |
| C gbdt + GOSS | parity | lightgbm-goss | 0.941296 | 0.940247 | 0.001050 | 0.000757 | **1.39** | **differs** |
| D random forest | family | sklearn, 1024-leaf cap | 0.939104 | 0.934829 | 0.004275 | — | — | zarbor ahead |
| D random forest | family | sklearn, uncapped default | 0.939104 | 0.933515 | 0.005588 | — | — | zarbor ahead |
| E logistic | family | sklearn lbfgs | 0.937715 | 0.937729 | 0.000014 | — | — | agrees |

**The metric path is clean.** On all six rows zarbor's self-reported AUC
agrees with the referee's, where the referee reloaded the saved `.zm`,
re-predicted, and scored with `sklearn.metrics.roc_auc_score`. The residual is
1e-8 to 5e-7, which is exactly the rounding from zarbor printing `auc={:.6f}`
-- so the check confirms agreement to within +/-5e-7 and cannot resolve
finer, that being the precision zarbor emits rather than a disagreement. A
metric bug on either side would have shown up here at a scale far above this
and did not.

Every model lands at or below the ~0.9420 ceiling this dataset is known to
have, which is what a correct implementation is supposed to do.

## Speed

Two numbers, because they answer different questions and reporting one would
be a choice about who wins.

**fit** — the training call alone, both sides.

| | zarbor | range | reference | range | zarbor is |
|---|---:|---|---:|---|---:|
| A gbdt depthwise | 503 ms | 498-510 | 609 ms | 599-623 | 1.21x faster |
| B gbdt leafwise | 449 ms | 444-464 | 474 ms | 471-489 | 1.06x faster |
| C gbdt + GOSS | 596 ms | 593-603 | 625 ms | 622-629 | 1.05x faster |
| D forest (capped ref) | 5029 ms | 4985-5102 | 6194 ms | 6189-6213 | 1.23x faster |
| D forest (uncapped ref) | 5029 ms | 4985-5102 | 8998 ms | 8836-9082 | 1.79x faster |
| E logistic | 203 ms | 203-210 | 949 ms | 823-1084 | 4.67x faster |

zarbor fits faster in all six, and **no pair of ranges overlaps**, so each of
these is a measured difference rather than noise. The protocol committed to
reporting an overlap as "no difference measured"; none occurred.

**end-to-end** — CSV in, predictions out. zarbor: process wall clock.
Python: `pd.read_csv` + categorical encode (0.35 s, n=5, range 0.34-0.36)
+ fit + predict.

| | zarbor | reference | |
|---|---:|---:|---|
| A gbdt depthwise | 1.00 s | 1.00 s | tie |
| B gbdt leafwise | 0.93 s | 0.87 s | **reference faster** |
| C gbdt + GOSS | 1.08 s | 1.03 s | **reference faster** |
| D forest (capped ref) | 5.99 s | 6.63 s | zarbor faster |
| D forest (uncapped ref) | 5.99 s | 9.76 s | zarbor faster |
| E logistic | 0.45 s | 1.31 s | zarbor faster |

**This is the finding worth keeping.** zarbor's fit is faster every time, and
on the two cheapest models that advantage is entirely consumed before the fit
starts. Where fitting dominates (D, E) it wins end-to-end; where fitting is
half a second, it does not.

One confound, named because it is large enough to matter: zarbor scores the
full validation set every round for its progress log, and the Python side was
not asked to do the equivalent (no `eval_set`). zarbor reports this cost
itself:

    A +226 ms   B +216 ms   C +218 ms   D +456 ms   E none

Subtracting it — comparing like with like — zarbor leads end-to-end on every
row (A 0.77 vs 1.00, B 0.71 vs 0.87, C 0.86 vs 1.03, D 5.53 vs 6.63). Both
versions are given because the first is what a user actually experiences at
the default settings and the second is what the implementations cost.

Either way the ingestion path, not the learner, is what to optimise next: on a
41 MB CSV, parse + bin + startup is ~0.28 s against pandas' 0.35 s for read +
encode, so zarbor's data path is *competitive with pandas and not better*,
while its fit is 1.05-4.7x better. That asymmetry is where the remaining
end-to-end time is.

## C: the one parity row that differs, characterised

GOSS costs zarbor 0.000062 AUC and costs LightGBM 0.001180 at the same rates.
Both implementations are doing real work — this is not one of them ignoring
the flag.

**zarbor's GOSS is verified live, at the extremes.** Driving the rates down
degrades it exactly as a real subsampler must:

| top/other | rows kept | zarbor AUC | fit |
|---|---:|---:|---:|
| none | 100% | 0.941358 | 446 ms |
| 0.2/0.1 | 30% | 0.941296 | 590 ms |
| 0.01/0.01 | 2% | 0.930397 | 335 ms |
| 0.001/0.001 | 0.2% | 0.581802 | 419 ms |

**LightGBM's GOSS is verified applied**, and identically through both the
legacy `boosting_type='goss'` and the current `data_sample_strategy='goss'`
(0.940041 either way), so the harness is not comparing against a silent
no-op.

The disagreement closes monotonically as sampling gets less aggressive:

| top/other | rows kept | zarbor | lightgbm | gap |
|---|---:|---:|---:|---:|
| 0.1/0.05 | 15% | 0.940315 | 0.937465 | 0.002850 |
| 0.2/0.1 | 30% | 0.941296 | 0.940247 | 0.001050 |
| 0.4/0.2 | 60% | 0.941471 | 0.941424 | 0.000047 |

At 60% both converge on the no-sampling score and agree well inside the
envelope. So the two agree on the boosting and disagree only on **how much
accuracy is lost per row discarded**, in proportion to how many are discarded.
zarbor degrades more gracefully.

Scoring higher is not a pass. The protocol calls an out-of-envelope parity row
a failure regardless of direction, and this one stays a failure until the
mechanism is identified. The cause is **not diagnosed** — pinning it means
reading LightGBM's `goss.hpp`, which this exercise did not do. Suspects, named
as suspects: whether the amplification factor `(1-top_rate)/other_rate` is
applied to the hessian as well as the gradient on both sides, and whether the
gradient threshold is an exact top-k (zarbor's is, pinned by the `gossCut`
test in `booster.zig`) or an approximation on either side.

A separate observation that is *not* a defect in either: at 30% sampling GOSS
is **slower** than no sampling in both implementations (zarbor 590 vs 446 ms,
LightGBM 636 vs 484 ms). Selecting and amplifying 535k gradients per tree
costs more than the histogram build it saves at this size. GOSS only becomes
faster once the rate is low enough to matter (335 ms at 2%).

## D: where the forest gap comes from, and where it does not

zarbor's forest beats sklearn's by 0.0043 with the leaf cap matched and 0.0056
with sklearn at its own uncapped default. The cap is therefore **not** the
source: removing it makes sklearn *worse*, so unlimited depth is overfitting
here rather than being held back.

That leaves binning as the leading candidate — zarbor splits on 256-bin
quantised features, which bounds where a split can be placed and acts as a
regulariser sklearn's exact splitter does not have. Consistent with the
measurement, not established by it: no experiment here isolates binning from
the other construction differences, so this is a direction, not a conclusion.

## E: the result that says binning is free

zarbor fits the binned, one-hot design matrix. sklearn fits raw standardised
columns. Different design matrices, different code, and they agree to
**0.000014 AUC** — while zarbor is 4.67x faster in fit and 2.9x end-to-end.

`linear.zig`'s header argues binning a numeric feature before a linear fit is
"a small loss of resolution and a real gain in robustness". On this dataset
the loss of resolution is 1.4e-5 AUC, which is the strongest evidence in this
document for the one-shared-data-path design: the binning built for the trees
costs the linear model essentially nothing.

## Summary

- Four of five model families match their reference implementation: two parity
  rows inside their noise floor, one family row agreeing to 1.4e-5, one family
  row ahead of its reference.
- One parity row (GOSS) is outside its envelope by 1.39x, characterised but
  not diagnosed.
- zarbor's fit is faster than every reference, 1.05x to 4.67x, with no
  overlapping ranges.
- End-to-end it wins only where fitting dominates. The data ingestion path is
  the bottleneck worth attacking next, not the learners.
