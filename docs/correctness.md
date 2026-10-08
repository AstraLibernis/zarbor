# Is zarbor right, and is it fast?

zarbor reimplements XGBoost, LightGBM, CatBoost and scikit-learn's forest and linear models.
This page shows how closely each one matches its reference, how fast it is, and how the code
is kept honest. Every claim names its evidence. Full method and history are in
[archive/](archive/README.md); timing numbers that code comments rely on are in
[measurements.md](measurements.md).

References: XGBoost 3.4.1 (`hist`), LightGBM 4.7.0, CatBoost 1.2.10, scikit-learn 1.9.1.
Machine: Ryzen 7 9800X3D, 16 threads, Fedora 44, ReleaseFast builds.

## In short

- **Same trees, not just similar scores.** Given the same data and settings, zarbor grows the
  same trees as XGBoost (500 rounds), LightGBM (50 rounds, until its pruning shortcut
  applies) and CatBoost (300 rounds plain, 156 ordered). Predictions agree to the 6-decimal
  print precision. Its linear models reach the same optimum as scikit-learn's.
- **Where randomness differs, the distributions agree.** Every sampling option sits within
  two standard errors of its reference over 10 seeds, with one exception: CatBoost's
  Bernoulli bootstrap (below).
- **Faster than XGBoost, LightGBM and scikit-learn on the benchmark dataset** (1.16x
  LightGBM, 1.39x XGBoost, 2.0x scikit-learn's forest, 3.8-6.6x its logistic regression and
  1.7x its linear regression on model work; the CSV parser is 4.7x pandas), **but slower than
  CatBoost**: 1.4x on plain symmetric
  trees, 1.75x on ordered boosting and 9.7x with CatBoost's default settings.
- **It processes messy real data.** 16 datasets from 891 rows to 1M rows ran through every
  command and setting extreme without a crash. Bad input is refused with a reason.

## 1. Exact parity: the same trees

**Rerun it:** `python bench/parity/parity.py` (42 s; exits non-zero on any failure) runs the
core cases on seeded synthetic data. All 15 pass: XGBoost to 500 trees, LightGBM to 30 trees
at 31 leaves and 50 at 15, CatBoost to 300 trees, with ordered boosting and target statistics,
and multiclass: 4 classes against XGBoost `multi:softprob` (100 rounds, 400 trees, max
difference 7.1e-7; and 40 rounds with L1, L2, gamma and `min_child_weight`) and LightGBM
`multiclass` (10 rounds with `--softmax_hessian=lightgbm`, 6.0e-7). Using the wrong hessian
form against XGBoost misses by 0.32, so the cases do discriminate. Multinomial logistic
regression against scikit-learn's `LogisticRegression` (2026-10-08, 20,000 rows, 8 integer
columns so the binned design holds the raw values, standardisation off on both sides): the
penalised objective at the optimum agrees to 4.8e-9 (L2, `lambda` 1) and 4.7e-10 (`lambda`
100), and to 1.8e-8 with L1 (`alpha` 50, against SAGA); probabilities within 1.1e-4 (L2) and
3.6e-4 (L1), the f32 flat-direction differences the binary model shows too.
The tables below come from the original harness on competition data.

**Method.** Compare predictions row for row on data where binning cannot differ: 100,000
rows of 13 integer columns with values 0–5 (Kaggle S6E10 ratings, binary target). Every
library gives every value its own bin, nothing is random, and both sides start from the same
base score. Rows count as equal within 6e-7, the print rounding. When predictions diverge,
the harness bisects to the first different tree and recomputes every candidate split at the
first different node in exact f64, to say which side chose correctly. The harness lives in
`~/workspace/research/gbdt-refs/parity/`, outside the repo, because the data cannot be
redistributed.

| zarbor | against | trees | max difference |
|---|---|---:|---|
| depthwise, depth 6, lr 0.1 | XGBoost | 500 | 7.7e-7 (99.8% of rows identical) |
| depthwise with L1 + L2 + gamma + `min_child_weight` | XGBoost | 1 | 5.1e-7 |
| depthwise, squared error | XGBoost | 200 | 7.8e-7 |
| `scale_pos_weight` 3 (base score pinned) | XGBoost | 100 | 7.1e-7 |
| `max_delta_step` 0.7 | XGBoost | 100 | 7.6e-7 |
| lossguide, 31 leaves | LightGBM | 50 | 6.3e-7 |
| symmetric, Cosine score | CatBoost | 300 | 8.6e-7 |
| symmetric, CTRs on categoricals | CatBoost | 200 | 7.3e-7, train and held-out |
| symmetric, combinations (complexity 4) | CatBoost | 200 | 6.6e-7 |
| ordered boosting | CatBoost | 156 | identical; first difference at tree 157 |

XGBoost parity needs `--min_data_in_bin=1` (XGBoost has no per-bin floor) and
`--min_child_samples=0`. CatBoost parity needs `--has_time=true` and the flags in
[guide.md](guide.md) section 7.

**Linear models** against scikit-learn, on 30,000 rows with a six-level categorical:
logistic L2, L1 and elastic net, and least squares L2, L1 and elastic net, all reach the same
optimum. Objectives are equal to 4e-9 relative or better, with the same non-zero
coefficients under L1. Prediction differences of 1e-5 to 2e-4 are f32 rounding along nearly
flat directions; AUC and RMSE agree to six decimals.

**Binning** matches each reference's own cut rule. On 150,000 rows of S6E10 at 255 bins:
`greedy` gives LightGBM's partitions exactly, and `logsum` gives CatBoost's exactly.
`quantile` gives XGBoost's whenever XGBoost's summary is exact (up to `8 x max_bin` distinct
values). Above that XGBoost's sketch approximates, and zarbor deliberately does not copy the
approximation.

### Where the references cut corners

Two LightGBM speed shortcuts explain every difference from it. zarbor does not copy them,
because they make LightGBM less exact.

- **Row counts are estimated.** `min_data_in_leaf` is checked against `hessian x n / H`. A
  child with 23 real rows was estimated at 15 and rejected under a floor of 20; zarbor counts
  rows.
- **A feature unsplittable at a node is skipped in all its descendants.** Gain is not
  monotone down a branch: one column had best gain −0.18 at depth 5 and 17.8 at depth 11,
  where LightGBM could no longer see it. With squared error, LightGBM's 45th tree splits a
  leaf with gain 2.78 instead of one with 4.12 for this reason. On the synthetic parity data
  the first miss is tree 34 at 31 leaves: LightGBM splits a leaf of gain 4.731 because the
  better leaf's best split (4.761) is on a feature it pruned there.

### Known differences, by design

| | zarbor | references |
|---|---|---|
| `max_bin` | 256 = 255 real bins + the missing bin | XGBoost 256 real; LightGBM 255 including a NaN bin |
| `colsample_*` count | `floor(n x rate)`, at least 1 | XGBoost the same; LightGBM rounds |
| `subsample` | exactly `round(n x rate)` rows | XGBoost: a coin flip per row |
| missing direction when a node saw no missing rows | the larger child | XGBoost right, LightGBM left |
| unseen value between training values | the nearer side (midpoint edges) | XGBoost the lower; others midpoint |
| GOSS ranking | `\|g\|` (the paper's); `--goss_rank=gradient_hessian` for LightGBM's `\|g*h\|` | LightGBM `\|g*h\|` |
| base score with `scale_pos_weight` | logit of the label mean | LightGBM the same; XGBoost one Newton step |
| sigmoid in gradients | polynomial, max error 8.1e-7 | `exp` |
| score accumulation | f32 | XGBoost f32, LightGBM and CatBoost f64 |

None of these moves a tree under the parity conditions above.

## 2. Statistical parity: where randomness differs

Random draws cannot be matched between libraries, so these compare hold-out AUC over 10 seeds
(20 for Bernoulli), 200 trees. In every block the no-sampling row is identical, so any gap
comes from the sampling.

| option | reference | zarbor | difference / s.e. |
|---|---|---|---:|
| XGBoost `subsample` 0.7 | 0.951085 | 0.951102 | 0.20 |
| XGBoost `colsample_bytree/bylevel/bynode` 0.54 | | | 1.88 / 0.45 / 1.11 |
| LightGBM bagging 0.7 | 0.951431 | 0.951300 | −1.35 |
| LightGBM `feature_fraction` 0.54 | 0.951477 | 0.951425 | −0.67 |
| LightGBM GOSS 0.2/0.1, LightGBM's key and warm-up | 0.951488 | 0.951356 | −1.51 |
| CatBoost MVS 0.8 | 0.950964 | 0.950962 | −0.05 |
| CatBoost Bayesian, temperature 1 | 0.950751 | 0.950758 | 0.15 |
| CatBoost split noise, strength 1 | 0.951015 | 0.950967 | −0.88 |
| CatBoost defaults (MVS + noise) | 0.950966 | 0.950941 | −0.48 |
| CatBoost permutations, CTRs | 0.945326 | 0.945267 | −0.56 |
| CatBoost permutations, ordered | 0.944993 | 0.945024 | 0.19 |
| CatBoost defaults with combinations | 0.945536 | 0.945493 | −0.38 |
| **CatBoost Bernoulli 0.5** | 0.950820 | 0.950959 | **+4.25** |

The ordered row was 0.945087 (0.55) before a48ebd7: ordered scoring now subtracts histograms,
so its sums round differently; every other row reproduced to the digit.

The Bernoulli outlier: Bernoulli costs CatBoost about twice the AUC it costs zarbor. Single
trees agree, and the cause is not found. zarbor's result is the better of the two.

## 3. Speed and accuracy on one benchmark

**[MEASURED] 2026-10-07** at 23999ba. `bench/arena/arena.py`, raw `bench/arena/results.json`.
Kaggle S6E9 (EV purchases): 534,932 training rows x 13 columns, binary, ROC-AUC on 133,733
held-out rows. Medians of 5 runs, all at 16 threads, 200 trees, depth 6, 256 bins.

Model time is prepare + fit + predict. XGBoost and LightGBM bin inside `fit`, while zarbor
bins before it, so binning is charged to whoever does it. Parsing is measured separately.
"Envelope" is the reference's own spread across `max_bin` 63/127/255: a gap inside it is
implementation noise. A referee reloads zarbor's saved model and rescores it, so neither
side's metric is trusted blind.

| model | reference | zarbor AUC | reference AUC | gap / envelope | zarbor model ms | reference ms | speed |
|---|---|---:|---:|---:|---:|---:|---:|
| gbdt depthwise | XGBoost | 0.941425 | 0.941441 | 0.02 | 477 | 661 | 1.39x faster |
| gbdt leafwise | LightGBM | 0.941418 | 0.941427 | 0.01 | 456 | 528 | 1.16x faster |
| gbdt + GOSS | LightGBM | 0.941313 | 0.940247 | 1.41, zarbor ahead | 591 | 678 | 1.15x faster |
| random forest | sklearn, 1024-leaf cap | **0.938148** | 0.934829 | different algorithm | 3,184 | 6,479 | 2.03x faster |
| random forest | sklearn, default | **0.938148** | 0.933515 | different algorithm | 3,184 | 9,497 | 2.98x faster |
| logistic regression | sklearn | 0.937716 | 0.937729 | — | 171 | 858-1,135 | 3.8-6.6x faster |
| linear regression (income, RMSE) | sklearn | 27922.124 | 27922.154 | — | 110 | 185 | 1.68x faster |

| parsing both CSVs (40 MB) | zarbor 38 ms | pandas 178 ms | 4.68x |
|---|---|---|---|

**CatBoost**, same data and protocol, symmetric depth-6 trees, `lambda` 3, its `GreedyLogSum`
binning, target statistics on the six categoricals:

| setting | zarbor AUC | CatBoost AUC | gap / envelope | zarbor model ms | CatBoost ms | speed |
|---|---:|---:|---:|---:|---:|---:|
| plain, file order, no sampling (F) | 0.940756 | 0.940803 | 0.09 | 1,755 | 2,204 | 1.26x faster |
| ordered boosting (H) | 0.940786 | 0.940730 | 0.09 | 5,113 | 5,939 | 1.16x faster |
| CatBoost's defaults: MVS, noise, combinations, permutations (G) | 0.940505 | 0.940643 | 0.25 | 10,324 | 6,105 | **1.69x slower** |

**[MEASURED] 2026-10-08** at a48ebd7 (CatBoost rows only). Accuracy agrees in every row, and
zarbor's AUCs are the same to the last digit as before the speed work (3,013 / 10,519 / 59,328
ms at 352d45f). What changed, found with `--profile=1`'s `sym_*` phases:

- Histograms were rebuilt from every row at every level. Below the root zarbor now counts only
  the rows on the smaller side of the newest split and takes the other side as the parent minus
  them, as CatBoost does (`SetSmallestSideControl` / `FixUpStats`); under ordered boosting each
  prefix keeps its own body and tail histograms for this (capped by `--ordered_bank_limit`).
- Combination statistics (CatBoost's defaults) were built one per thread with two hash lookups
  per row: 48 of the 59 s. They are now built in parallel row chunks from integer counts.
- Prefix derivatives, fold updates and leaf assignment ran serially over every row.

With its defaults zarbor is still slower; the remainder is the combinations (about 45% of
fit) and their histograms, which are new at each level and so cannot be subtracted.

How to read it:

- Rows A-D are from 23999ba; the linear rows and the CatBoost rows below were re-measured at
  352d45f, after the linear solver's loops were rewritten (fit 258 -> 92 ms on the income data,
  predictions byte-identical). scikit-learn's logistic fit varied between runs that day (631
  and 900 ms against zarbor's steady ~125 ms), hence the range.
- The GOSS gap is the ranking key, not an error. With `--goss_rank=gradient_hessian`
  (LightGBM's key) it fell inside the envelope, at 0.20 (measured 2026-09-23, not re-run).
- The forest is a different algorithm from scikit-learn's: histogram-binned and leaf-capped,
  where scikit-learn uses exact splits. It is both better and faster here.
- Compared with the 2026-09-23 run: the leafwise fit went from 1.07x *slower* than LightGBM
  to 1.16x faster, and parsing from 1.31x to 4.68x pandas. The forest AUC moved from
  0.939025 to 0.938148 with the binning changes; it is still ahead.

## 4. Real, messy data

**[MEASURED] 2026-10-07.** Every dataset from the other projects (16, from 525 to 1M rows)
went through `profile`, train, save and predict, the binning extremes (2 and 65535 bins, the
per-bin floor at 1 and 300, every policy at 4096), every model, wide categoricals and
`tune`. The harness is `~/workspace/research/stress/stress.py`.

The run found and fixed five bugs:

- the CSV reader silently cut records longer than the header;
- blank lines became rows;
- the histogram memory ceiling refused large trees, which aborted whole tune searches;
- `tune` overrode pinned flags;
- `--help` printed no usage for `tune` and `cv`.

After the fixes, every step completes, and every `predict` row count matches its test file.
The only refusals left are input that should be refused: malformed headers, missing target
values, and no features left after drops. Each is refused with a reason.

The most expensive cases: 1M x 20 continuous columns at 65535 bins takes 274 s and 2.8 GB
(it was refused before), and a 1024-leaf forest at 4096 bins takes 45 s. Setting
`--min_data_in_bin=300` gives the same AUC in 12.6 s.

## 5. How the code is kept honest

- **Unit tests** (about 150, `zig build test`): every binning policy against its reference's
  pinned output, every parity fix, the CSV edge cases, determinism at 1, 3 and 16 threads,
  model round trips, histogram eviction giving identical trees.
- **Golden tests** (`zig build golden`): 58 end-to-end CLI runs on Titanic compared byte for
  byte with `test/golden/expected/`. Any change to an output must be re-baselined on purpose.
- **ThreadSanitizer** (`zig build test-tsan`) whenever threaded code changes; the pre-commit
  hook runs it.
- **Mutation checks**: each new test is checked by breaking the code it guards, watching the
  test fail, and restoring the code. A mutation that only causes a compile error proves
  nothing and is redone.
- **Debug vs release.** Correctness is judged in Debug/ReleaseSafe, where assertions run.
  Speed is measured only in ReleaseFast, and every run prints its build mode.

## 6. Open questions

- CatBoost's Bernoulli bootstrap gap (section 2).
- Late divergence from CatBoost under ordered boosting at tree 157. An independent f64
  reference implementation diverges earlier, at tree 103, so the cause may be in neither.
- On EEG data, hold-out error at 4096 bins varies more than 5x between binning policies;
  unexplained.
- zarbor is 1.7x slower than CatBoost under CatBoost's defaults (section 3): the combinations.
- `bench/parity/parity.py` covers the exact cases; the statistical (sampling) comparisons
  still live outside the repo.
