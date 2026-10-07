# CatBoost-style training

**Status: stages 1–5 of 6, [MEASURED] 2026-10-07** against catboost 1.2.10.

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
bootstrap type (stage 2). Without `--cat_split=ctr`, categorical columns split on their ordinal
ids.

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

## Stage 3: categoricals as CatBoost has them (done)

`--cat_split=ctr` (symmetric trees, logistic loss) turns each categorical into split candidates
the CatBoost way:

- **One level** in training: dropped. **Up to `--one_hot_max_size` (2)**: one-hot, a split
  `level == v`.
- **Wider**: four *ordered target statistics* (CTRs), each a 16-bucket column: Borders with
  priors 0, 0.5 and 1, `(positives + prior) / (count + 1)`, and Counter,
  `count / (largest count + 1)`, bucketed `trunc(15 * value)` in f32. During training a row's
  Borders value counts **only the rows before it** in file order (CatBoost's `has_time`), so a
  row's own label never reaches its own feature. Counter uses every row, as CatBoost's
  `SkipTest` does.
- **Prediction** uses the counts over every training row. For a CTR on one column that is a fixed
  function of the level, so a fitted CTR split is stored as an ordinary categorical set split (the
  smaller side, the node marked inverted when that side is the bit-1 one): no new node kind and no
  model-format change. A level never seen in training gets bin 0, whose counts are the prior's
  unless training had missing values in that column.
- **Candidate order** is CatBoost's, for ties: numeric columns, then one-hot, then CTRs.
- **`--model_size_reg`** (0.5) shrinks a CTR's score by `(1 + uniq / max_uniq)^-0.5` until a CTR of
  the same type (Borders or Counter) on that column has been chosen. It counts as chosen the moment
  it is picked at a level, even if the redundancy rule then removes it, as in CatBoost.

Measured against catboost 1.2.10 with `has_time=True, one_hot_max_size=2, max_ctr_complexity=1`
on 20,000 training rows of S6E10 (three two-level categoricals, `Class` with three levels, `Age`
as a 75-level categorical, three numeric columns), predictions row for row on training and
held-out rows (harness `parity/cat_ctr.py`):

| case | train max diff | held-out max diff |
|---|---|---|
| 1 tree | 5.4e-7 | 5.4e-7 |
| 50 trees, `model_size_reg` 0 | 6.2e-7 | 5.8e-7 |
| 50 trees, `model_size_reg` 0.5 | 5.9e-7 | 5.8e-7 |
| 200 trees, `model_size_reg` 0.5 | 7.3e-7 | 7.3e-7 |

**A fix this found elsewhere.** `zarbor train` shuffled every row, even with `--valid-frac=0`, to
pick the validation rows. Row order is the time order a CTR reads, so CatBoost parity broke at tree
5. The shuffle now only chooses which rows validate; both sides go back to file order. Six golden
outputs changed, all from seeded row sampling (forest bootstrap, `subsample`, GOSS) picking by
position; runs without sampling are bit-identical. `cv` already kept file order.

## Stage 4: ordered boosting (done)

`--boosting_type=ordered` (symmetric trees). Rows are taken in file order and cut into prefixes,
CatBoost's dynamic folds: the first body is `min(100, n/50)` rows (1 when `n <= 500`), each tail
twice its body, each next body the previous tail. Each prefix keeps its own model, trained only
on its body rows. A split is scored by leaf estimates from each prefix's **body**, scored against
its **tail** (`num += v * tail_sum`, `den += v^2 * tail_weight`, summed over prefixes, then
`num / sqrt(den)`), so no row's gradient is judged by a model that saw that row. After each tree
every prefix model takes Newton steps fitted on its body and applied to all its rows. The model's
own leaves are still full-data Newton steps. Bootstrap weights and noise follow the tail rows, as
CatBoost's do. Each tree's histograms are built one prefix at a time, so memory stays at two
histogram banks.

Refused under ordered, not yet verified: `--score_function=gain` and more than one leaf step.

Measured against catboost 1.2.10 (`boosting_type='Ordered', has_time=True`), row for row:

| case | max abs difference |
|---|---|
| numeric parity data, 1 tree | 5.4e-7 |
| numeric parity data, 50 trees | 6.7e-7 |
| numeric parity data, 156 trees | identical (first difference at tree 157) |
| with categoricals (CTRs + one-hot), 50 trees, train and held-out | 5.8e-7 |

Late in training the two part ways: at tree 157 on the numeric data zarbor's root is a different
split (its top two candidates 0.9059 and 0.9056). The independent f64 Python reference the spec was
verified with (`catboost-parity/s4_ordered_boosting.py`, run on the same data for 160 trees) parts
from CatBoost **earlier**, at tree 103 (same three splits in a different order). So zarbor tracks
CatBoost longer than the reference does; the cause of the late divergence, in either, is not found.
CatBoost's ordered cost is about 4x its plain cost; zarbor's has not been timed.

## Stage 5: categorical combinations (done)

Under `--cat_split=ctr`, from the second level of each tree on, CatBoost's *tree CTRs*: the bases
are all numeric and one-hot splits above, taken together as one projection, and each CTR
projection already chosen in this tree. Each base gains each wide categorical not already in it,
while the projection's length stays within `--max_ctr_complexity` (4); duplicates are skipped, and
each new projection gives the same four CTR types. A row's key mixes its levels of the projection's
categoricals with its bits of the projection's splits, so statistics like "class x (age > 30)"
exist.

- **Length**, CatBoost's `GetFullProjectionLength`: the projection's categoricals, plus **one** if
  it has any split bits, however many. (The implementation spec said categoricals plus bits plus
  one-hots; that gave different trees from the first one with combinations, and CatBoost's
  projection.h settles it.)
- A chosen combination is a new node kind: the node holds a table index and a bucket threshold, the
  tree holds the table (projection, sorted keys, buckets from counts over all training rows, and
  the bucket an unseen key gets). Single-column CTRs keep the categorical set split.
- **Model format 5** adds the node kind and the per-tree combination tables; format 4 files still
  load. Saving goes through a tree copy that at first dropped the tables; the save/load round trip
  is now tested.
- Unlike CatBoost, `max_ctr_complexity` is not silently forced to 1 below 200 iterations.

Measured against catboost 1.2.10 (`has_time=True`, explicit `max_ctr_complexity`) on the stage-3
data, row for row on training and held-out rows (harness `parity/cat_combo.py`):

| case | max abs difference |
|---|---|
| complexity 2, 1 tree | 5.4e-7 |
| complexity 2, 3 and 4, 50 trees | 5.9e-7 |
| complexity 4, 200 trees (CatBoost used 28 combinations) | 6.6e-7 |

CatBoost visits its bases in hash order, zarbor in split order, so candidates with exactly equal
scores could break the other way; none did here.

## Still to build

| stage | what | how it can be checked |
|---|---|---|
| 6 | CatBoost's default permutations | statistically only |

CatBoost's CPU default is **plain** boosting at every data size; ordered boosting runs only when
asked for. Its defaults otherwise are MVS bootstrap at 0.8, `random_strength=1`, 10 Newton steps
with backtracking, and learning rate chosen by a formula on rows and iterations.
