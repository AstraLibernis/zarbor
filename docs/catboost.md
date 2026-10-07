# CatBoost-style training

**Status: stage 1 of 6, [MEASURED] 2026-10-07** against catboost 1.2.10.

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

Not yet supported under `symmetric`, and refused rather than ignored: row or column sampling,
GOSS, linear leaves, `alpha`, `max_delta_step`, `cat_split=optimal`. Categorical columns are
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

## Still to build

| stage | what | how it can be checked |
|---|---|---|
| 2 | bootstrap (Bayesian, Bernoulli, MVS) and `random_strength` split noise | statistically only: CatBoost's random draws cannot be reproduced |
| 3 | ordered target statistics for categoricals (CTRs), one-hot for two-level columns | row for row with `has_time=True` |
| 4 | ordered boosting | row for row with `has_time=True`; about 4x CatBoost's plain cost |
| 5 | categorical feature combinations | row for row with `has_time=True` |
| 6 | CatBoost's default permutations | statistically only |

CatBoost's CPU default is **plain** boosting at every data size; ordered boosting runs only when
asked for. Its defaults otherwise are MVS bootstrap at 0.8, `random_strength=1`, 10 Newton steps
with backtracking, and learning rate chosen by a formula on rows and iterations.
