# Tree-for-tree parity with XGBoost and LightGBM

**Status: [MEASURED] 2026-10-07**, zarbor a9ee25b plus the fixes listed below, against
XGBoost 3.4.1 (`tree_method=hist`) and LightGBM 4.7.0, both from PyPI, sources read at the
matching tags.

`vs-lightgbm.md` compares *scores* and asks whether a gap is inside a noise envelope. That can
say two implementations agree on average; it cannot say they compute the same thing. This
document asks the stronger question: given the same data and settings, does zarbor grow the
*same trees*?

## Method

Score-level agreement hides compensating differences, so the comparison is on predictions,
row for row, under conditions where a difference can only come from the arithmetic:

- **Binning cannot differ.** 100,000 rows, 13 integer columns with values 0–5 (the rating
  columns of Kaggle Playground S6E10, binary target). Every library can give every value its own
  bin, so every partition is reachable on every side.
- **Nothing random.** No row or column sampling.
- **Same start.** XGBoost's `base_score` is set to the label mean (zarbor's and LightGBM's
  default start).
- **Same constraints.** `min_child_samples=0` against XGBoost, which has no row-count floor.
- **Tolerance is the print rounding.** `zarbor predict` writes 6 decimals, so rows count as
  equal within 6e-7.

When predictions diverge, the harness finds the first divergent round by bisection, then walks
both trees of that round node by node (zarbor's `.zm` is read directly; XGBoost and LightGBM
dump JSON) and recomputes every candidate split at the first differing node in exact f64 from
the previous round's margins. That says which side picked the best valid split, and why the
other did not.

The harness lives outside the repo, beside the reference sources, because its data is
competition data that cannot be redistributed: `~/workspace/research/gbdt-refs/parity/`
(`parity.py` row-for-row cases, `zmread.py` model reader, `claims.py` one minimal experiment
per suspected difference). Porting it to synthetic data so it can live in `bench/` is open.

## Result

| comparison | trees | max abs difference | rows equal |
|---|---|---|---|
| depthwise vs XGBoost, depth 1, 1 tree | 1 | 3.4e-7 | 100% |
| depthwise vs XGBoost, depth 6, L1 + L2 + gamma + min_child_weight | 1 | 5.1e-7 | 100% |
| depthwise vs XGBoost, depth 6, lr 0.1 | 500 | 7.7e-7 | 99.8% (rest within 7.7e-7) |
| lossguide vs LightGBM, 31 leaves, lr 0.1, no row-count floor | 50 | 6.3e-7 | 99.96% |

With `--min_data_in_bin=1` (and, before the binning fix below, `--bin_policy=greedy`), zarbor's
depthwise booster reproduces XGBoost through 500 rounds to the print precision. Gradients and hessians, histogram sums, split gain
with L1/L2, `min_child_weight`, leaf weights, shrinkage and growth order all agree; an error in
any of them compounds and shows within a few rounds. The leaf-wise booster reproduces LightGBM
the same way, except where LightGBM's own shortcuts apply (below).

## What was wrong in zarbor

| defect | evidence | status |
|---|---|---|
| **The default `quantile` binning merges rare values**, even in a column with six distinct values: a value rarer than 1/255 of the rows never becomes a cut (`bin_edges.zig`, quantile branch). XGBoost and LightGBM give each distinct value its own bin when they fit. | First divergence in round 5 of the depthwise run traced to a split on `r11 < 1`, unreachable because 0 and 1 shared a bin. | **Fixed** (same day): every policy now gives one bin per value when the values fit, and `quantile` follows XGBoost's cut rule. See `binning.md`. |
| **No split could isolate missing from present.** Every scan kept at least one real bin beside the missing mass, and a feature with one real bin was skipped. XGBoost reaches this partition from its scan's end points. | A column that is a constant or NaN: XGBoost's root split gain 1157, zarbor a stump. | **Fixed.** `threshold = 0` with missing left is now a candidate for every feature with missing rows. Tests: `split_test.zig`. |
| **`min_split_gain` was half of XGBoost's `gamma`.** zarbor's gain carried a 0.5 factor that XGBoost's `loss_chg` and LightGBM's gain do not, so the same threshold rejected splits XGBoost accepts. | gamma = 0.6 × root gain: XGBoost split, zarbor did not. | **Fixed.** Gain is now the unhalved score sum on all three paths (numeric, one-vs-rest, sorted categorical). Test: `split_test.zig`. |

The missing-vs-present fix changed 13 of 55 golden outputs, all on Titanic where `body`,
`boat` and `age` are heavily missing. Disabling only that candidate restores all 55, so it is
the whole cause. The gain change moves no golden output (`min_split_gain` defaults to 0).

## Where the references cut corners and zarbor does not

Both are LightGBM speed shortcuts, found as divergences and confirmed in its source:

- **Row counts are estimated, not counted.** `min_data_in_leaf` is checked against
  `round(hessian × n / H)` (`feature_histogram.hpp`, `cnt_factor`). A child with 23 real rows
  was estimated at 15 and rejected under a floor of 20; zarbor counts rows.
- **A feature unsplittable at a node is skipped in every descendant.** If no cut of a feature
  has positive gain at a leaf, LightGBM marks it and never searches it again below
  (`serial_tree_learner.cpp`, `is_splittable`). Gain is not monotone down a branch: one column
  had best gain -0.18 at depth 5 and 17.8 at depth 11, where it was the best split and
  LightGBM could not see it. This is why 127-leaf trees diverge where 31-leaf trees do not.

Neither is a zarbor bug, and copying them would make zarbor less exact. Parity runs against
LightGBM must avoid them: `min_data_in_leaf=0` (with `min_child_samples=0`), and shallow
enough trees that no feature is pruned, or compare only up to the first such node.

## Remaining known differences

Measured or read in source; none moves a tree under the parity conditions above.

| item | zarbor | reference | effect |
|---|---|---|---|
| base score with `scale_pos_weight` | logit of the label mean, ignoring the weight | **LightGBM: the same.** XGBoost: one Newton step from 0 | First trees differ from XGBoost's when `scale_pos_weight != 1`. The references disagree; zarbor follows LightGBM. |
| `max_bin` | 256 = 255 real bins + the missing bin | XGBoost: 256 real; LightGBM: 255 including a NaN bin when present | One bin of budget. |
| `colsample_*` count | `floor(n × rate)`, at least 1 (was `round` until 2026-10-07) | XGBoost the same; LightGBM `round` over non-trivial features | 13 features at 0.5: zarbor and XGBoost 6, LightGBM 7. |
| `subsample` | exactly `round(n × rate)` rows | XGBoost: a Bernoulli draw per row | Same expectation, different variance. |
| missing direction when a node saw no missing rows | the larger child | XGBoost right; LightGBM left | Predictions on new NaNs only. |
| cut placement for unseen values | midpoint between training values (since the binning rewrite) | XGBoost the left value; LightGBM, CatBoost, scikit-learn the midpoint | Predictions on values between training values only. |
| gradient sigmoid | polynomial, max abs error 8.1e-7 | `expf` / `std::exp` | Tens of ulp in the tail; invisible at 500 trees (see the table above). |
| hessian floor | 1e-6 | XGBoost 1e-16, LightGBM none | Only where \|margin\| > ~13.8. |
| scores | f32 | XGBoost f32, LightGBM f64 | Invisible at 500 trees. |

`max_delta_step` was suspected (from reading) of entering XGBoost's split gain but not zarbor's;
the minimal experiment for it did not reproduce a difference, so it is not listed as one.

## How the audits went wrong

Two code-reading reviews were run beside the measurements. Both were useful and both made claims
the measurements contradicted: one called LightGBM's descendant pruning "effectively exact (a
child is never less constrained)", which the depth-5/depth-11 trace above disproves; one put the
sigmoid's error at 27 ulp, measured at 58. Every row in the tables above rests on an experiment,
not on a reading.

## Second round: linear models, regression, the remaining knobs, sampling

**[MEASURED] 2026-10-07**, same references. Harness: `parity/lincompare.py`, the `xgb_regression`,
`lgb_regression`, `xgb_max_delta_step` and `xgb_scale_pos_weight` cases of `parity.py`, and
`parity/sampling.py`.

### Linear models against scikit-learn 1.9.1

zarbor minimises `sum(loss) + lambda/2 ||w||^2 + alpha ||w||_1` (intercept free) on standardised
columns, so the scikit-learn equivalents are `C = 1/lambda` (logistic L2), `C = 1/alpha` (L1),
`Ridge(alpha=lambda)`, `Lasso(alpha=alpha/n)` and the matching elastic nets. On 30,000 rows of
0–5 columns plus a six-level categorical (one bin per value, so the inputs are identical):

| model | max prediction difference | objective |
|---|---|---|
| logistic L2, lambda 1 / 300 | 5.0e-5 / 1.1e-5 | |
| logistic L1, alpha 40 | 2.4e-4 | |
| logistic elastic net | 2.8e-5 | |
| least squares L2, lambda 1 / 3000 | 8.4e-5 / 9.6e-6 | equal to 1e-10 relative |
| least squares L1, alpha 2000 | 2.2e-4 | equal to 4e-9 relative, same 4 of 19 non-zero |
| least squares elastic net | 1.3e-5 | |

Same optimum. The prediction differences are f32 rounding along nearly flat directions (the
rating columns are correlated, and a full one-hot block plus the intercept is collinear); AUC and
RMSE agree to six decimals. zarbor stops when f32 can make no more progress, which its fit line
reports as converged.

### Exact cases

| case | result |
|---|---|
| regression vs XGBoost, 200 trees | identical (7.8e-7) |
| regression vs LightGBM, 100 trees | first difference at tree 45: LightGBM splits a leaf whose best gain is 2.78 instead of one whose best gain is 4.12, the second skipped by its feature-pruning shortcut (above). zarbor follows the gains. |
| `scale_pos_weight` 3 vs XGBoost, base score pinned, 100 trees | identical (7.1e-7) |
| `max_delta_step` 0.7 vs XGBoost, 100 trees | identical after the fix below (7.6e-7) |
| `max_delta_step` 0.05 | 99% of rows; heavy clipping makes many candidates score the same and XGBoost compares in f32 |

**Fixed:** with `max_delta_step`, zarbor scored a node by its free leaf weight and clipped only the
leaf; XGBoost and LightGBM score the clipped weight, `-(2 g w + (h + lambda) w^2 + 2 alpha |w|)`.
The first audit's experiment could not trigger it; one that clips many leaves diverged by 0.4.

### Sampling, statistically

Hold-out AUC over 10 seeds each, 200 trees, the parity data with 30% held out. A "no sampling" row
opens each block.

| XGBoost | xgboost | zarbor | diff / s.e. |
|---|---|---|---|
| no sampling | 0.950988 | 0.950988 | identical |
| `subsample` 0.7 (Bernoulli per row vs exactly 70%) | 0.951085 | 0.951102 | 0.20 |
| `colsample_bytree` 0.5 | 0.951090 | 0.951381 | **4.26** |
| `colsample_bytree` 0.54 | 0.951215 | 0.951381 | 1.88 |
| `colsample_bylevel` 0.5 / 0.54 | | | 0.10 / 0.45 |
| `colsample_bynode` 0.5 / 0.54 | | | 0.57 / 1.11 |

`colsample_bytree` 0.5 on 13 columns: zarbor kept `round(6.5) = 7`, XGBoost `floor = 6`. At 0.54
both keep 7 and the gap falls within two standard errors (zarbor's result is identical at 0.5 and
0.54, as it should be). **Changed since:** zarbor now counts `max(1, floor(n * rate))` like XGBoost
and scikit-learn, so a fraction is a ceiling; LightGBM rounds.

| LightGBM | lightgbm | zarbor | diff / s.e. |
|---|---|---|---|
| no sampling | 0.951189 | 0.951419 | differ (its two shortcuts, at 31 leaves and min_data_in_leaf 20) |
| bagging 0.7 every tree | 0.951431 | 0.951300 | -1.35 |
| `feature_fraction` 0.54 | 0.951477 | 0.951425 | -0.67 |
| `feature_fraction_bynode` 0.54 | 0.951697 | 0.951774 | 1.34 |
| GOSS 0.2 / 0.1, zarbor ranking `\|g\|` | 0.951488 | 0.951477 | -0.12 |
| GOSS, zarbor ranking `\|g h\|` | 0.951488 | 0.951643 | 1.81 |
| linear trees, lambda 1 | 0.950231 | 0.950803 | differ (deterministic) |

Every sampled row is within two standard errors once the column counts agree. Two differences of
design remain: LightGBM does not start GOSS until `1 / learning_rate` rounds in (goss.hpp:34;
zarbor samples from the first), and its linear leaves fit every numeric feature on the path where
zarbor's take at most `lin_leaf_max_terms` (8); with the trees already differing through
LightGBM's shortcuts, the linear-tree rows compare two algorithms, not an implementation.
