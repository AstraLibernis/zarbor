# CatBoost-style training

**Status: stages 1–2 of 6, [MEASURED] 2026-10-07** against catboost 1.2.10.

CatBoost is the same family as XGBoost and LightGBM: gradients and hessians, histogram split
search, Newton leaf values, shrinkage. What is genuinely different is how a tree is grown and how
categoricals and boosting order are handled. zarbor re-implements those pieces from CatBoost's
documented behaviour and a reading of its source (Apache-2.0; no code is copied). The full
implementation spec, with source references for every rule, is in
`~/workspace/research/gbdt-refs/catboost-spec.md`.

## Stage 1: symmetric trees (done)

`--grow_policy=symmetric` grows **oblivious trees**: one split per depth, shared by every node at
that depth, so a depth-6 tree is six yes/no questions and 64 leaves (`symmetric.zig`).

- Each level's split is chosen by a score summed over all current leaves:
  `--score_function=cosine` (CatBoost's default, and `auto` under symmetric), `l2`, or `gain`
  (the XGBoost/LightGBM Newton gain). Cosine and L2 use first derivatives only; the hessian enters
  only the leaf values.
- Leaves are Newton steps, `-G / (H + lambda)` times the learning rate. With
  `--leaf_estimation_iterations=N` the step is retaken N times at the moved score; as in CatBoost,
  `lambda` damps each step but carries no `lambda * value` term, so repeated steps head for the
  unregularised leaf optimum. CatBoost's backtracking is not implemented: CatBoost's own output
  with it changes with its thread count.
- If a split leaves every pair of leaves differing only in some bit with an empty side, that
  split is removed and the tree stops there (CatBoost's redundancy rule).
- No minimum leaf size or minimum gain, as in CatBoost: a symmetric tree always reaches
  `max_depth` unless the redundancy rule ends it.
- Missing values (bin 0) go left at every split, CatBoost's `nan_mode = Min`; the
  missing-vs-present cut is offered only for columns that have missing rows.
- The fitted tree is stored as an ordinary complete binary tree, so prediction, `.zm` files,
  `blend` and `cv` need nothing new.

Not yet supported under `symmetric`, and refused rather than ignored: column sampling, GOSS,
linear leaves, `alpha`, `max_delta_step`, `cat_split=optimal`; row sampling only through a
bootstrap type (stage 2). Categorical columns are
split on their ordinal ids until stage 3.

### Matching CatBoost

| zarbor | catboost 1.2.10 |
|---|---|
| `--grow_policy=symmetric --max_depth=6 --lambda=3` | `depth=6, l2_leaf_reg=3` (CatBoost's defaults) |
| `--base_score=0` | `boost_from_average=False` (CatBoost's default for Logloss) |
| `--min_data_in_bin=1` with `--bin_policy=logsum` | `border_count=254, feature_border_type='GreedyLogSum'` |
| (no sampling, no noise) | `boosting_type='Plain', bootstrap_type='No', random_strength=0` |
| `--leaf_estimation_iterations=N` | `leaf_estimation_method='Newton', leaf_estimation_iterations=N, leaf_estimation_backtracking='No'` |
| `--learning_rate=x --n_rounds=n` | explicit `learning_rate`, `iterations` (CatBoost's auto rules otherwise change both) |

Measured on the parity data (`parity.md`: 100,000 rows, 13 columns of values 0–5), predicted
probabilities row for row, tolerance the 6-decimal print rounding:

| case | max abs difference |
|---|---|
| 1 tree, depth 6, Cosine | 5.4e-7 |
| 50 trees, depth 6, Cosine | 6.4e-7 |
| 300 trees, depth 6, Cosine | 8.6e-7 |
| 50 trees, L2 score | 6.6e-7 |
| 50 trees, 10 Newton steps | 6.4e-7 |

The slight growth with tree count is zarbor's f32 score accumulation against CatBoost's f64; no
split differed. Harness: `~/workspace/research/gbdt-refs/parity/parity.py` (`cat_*` cases).

## Stage 2: bootstrap and split noise (done)

Under `symmetric`, `--bootstrap_type` draws per-tree row weights that enter **split scoring
only**; leaf values always use every row unweighted (checked against CatBoost: its leaves equal
full-data Newton steps exactly under all three types).

- `bernoulli`: keep each row with probability `--subsample`, weight 1.
- `bayesian`: weight `(-ln u)^bagging_temperature`; temperature 0 is no bootstrap.
- `mvs` (CatBoost's CPU default): per 8192-row block, keep a row with probability
  `min(1, sqrt(g^2 + lambda) / mu)` and weight it by the inverse, `mu` solved so the expected
  kept count is `subsample` of the block; `lambda` is `--mvs_reg` or CatBoost's rule (previous
  tree's mean |leaf| squared, the mean |g| squared on the first tree).

`--random_strength` adds normal noise to split scores, CatBoost's way: scale
`random_strength * sqrt(mean g^2)`, times a decay logistic in `ln n - iteration * learning_rate`,
so it fades once the model has grown. Within a feature the thresholds compete on noisy scores and
the winner keeps its noise-free score; the comparison across features draws fresh noise.

Every random stream is keyed by seed, tree, level and feature or block, so results do not depend
on the thread count (tested at 1, 3 and 16 threads).

CatBoost's random numbers cannot be reproduced, so parity is statistical: hold-out AUC over 10
seeds each, same split, 200 trees, depth 6 (harness `parity/cat_stat.py`):

| case | catboost | zarbor | difference / s.e. |
|---|---|---|---|
| no bootstrap, no noise | 0.951054 | 0.951054 | identical |
| MVS 0.8 | 0.950964 ± 0.000068 | 0.950962 ± 0.000108 | -0.05 |
| Bayesian, temperature 1 | 0.950751 ± 0.000115 | 0.950758 ± 0.000102 | 0.15 |
| noise, strength 1 | 0.951015 ± 0.000110 | 0.950967 ± 0.000134 | -0.88 |
| CatBoost defaults (MVS 0.8 + strength 1) | 0.950966 ± 0.000094 | 0.950941 ± 0.000133 | -0.48 |
| **Bernoulli 0.5** (20 seeds) | 0.950820 ± 0.000105 | 0.950959 ± 0.000101 | **+4.25** |

**Open:** Bernoulli costs CatBoost about twice the AUC it costs zarbor. Per tree the two agree
(leaves identical; over 30 seeds a single tree keeps the no-bootstrap splits 23 times in CatBoost,
19 in zarbor), and the λ scaling and sampling frequency were ruled out in CatBoost's source; the
difference accumulates over trees and its cause is not found. zarbor's Bernoulli is the better of
the two, not the worse.

## Still to build

| stage | what | how it can be checked |
|---|---|---|
| 3 | ordered target statistics for categoricals (CTRs), one-hot for two-level columns | row for row with `has_time=True` |
| 4 | ordered boosting | row for row with `has_time=True`; about 4x CatBoost's plain cost |
| 5 | categorical feature combinations | row for row with `has_time=True` |
| 6 | CatBoost's default permutations | statistically only |

CatBoost's CPU default is **plain** boosting at every data size; ordered boosting runs only when
asked for. Its defaults otherwise are MVS bootstrap at 0.8, `random_strength=1`, 10 Newton steps
with backtracking, and learning rate chosen by a formula on rows and iterations.
