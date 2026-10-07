# zarbor guide

How to use zarbor, and what it does with your data. For how we know it computes the right
thing, see [correctness.md](correctness.md). For raw timing numbers, see
[measurements.md](measurements.md). Older design notes and investigations are in
[archive/](archive/README.md).

Every setting is a flag named after its config field (`--max_depth=7`), generated from the
config structs, so flags and code cannot drift apart. Names follow XGBoost where an equivalent
exists and LightGBM or CatBoost where only they have the setting.

## 1. Commands

    zarbor <train.csv> --label=y [--save=m.zm]        train, report a hold-out score
    zarbor predict <data.csv> --model=m.zm [--out=p.csv] [--id-col=id] [--label=y]
    zarbor blend   <data.csv> --models=a.zm,b.zm [--weights=1,2] [--out=p.csv]
    zarbor cv      <train.csv> --label=y [--folds=5] [--repeats=N] [--group-col=g]
    zarbor tune    <train.csv> --label=y [--search=random|grid|bayes|bandit] [--trials=40]
    zarbor profile <data.csv>                          what is in the file, before any model
    zarbor info    --model=m.zm                        what is in a model

`train` holds out `--valid-frac` (0.2) of the rows, or the rows where `--split-col` is at
least 0.5, and prints the hold-out score. Build with `zig build -Doptimize=ReleaseFast`. Plain
`zig build` gives a Debug binary that is several times slower, and every run prints its
build mode.

## 2. Reading a file

**CSV rules.** Fields follow RFC 4180: quoted fields may hold commas, newlines and `""`, and
there is no column limit. A stray quote inside an unquoted field (`3" pipe`) makes zarbor
re-read the file leniently and keep the quote as text. Parsing runs on every thread.

**Shape is checked.** A record with more fields than the header is refused: train, cv, tune
and predict stop, and report how many records are long and which is first. This usually means
the first line is not the header, or a field has an unquoted comma. zarbor does not guess.
A record with fewer fields is read with the missing fields as missing, and a note says how
many. Blank lines are skipped. `profile` reports a bad shape and keeps going.

**Column kinds** are sniffed from the first 1000 rows: numeric if every value parses as a
number or is a missing marker, categorical otherwise. A non-number further down a numeric
column is counted as "unparsed" and reported, never silently dropped.

**`NA` depends on the column.** `NA`, `N/A`, `NaN`, `null`, `None`, `nil`, `?` and the empty
field mean *missing* in a numeric column. In a categorical column they are an ordinary level,
except the empty field, which is always missing. In House Prices, `LotFrontage=NA` is an
unrecorded value, while `PoolQC=NA` is the level "No Pool" in the data dictionary. Near misses
such as `NAmes` are never treated as missing.

**Categorical width.** A categorical with more than `--max_cat_levels` levels (255) is refused
and named, with advice. Usually that means an ID or free text (names, tickets); drop it or
replace it with a numeric summary. If you really want it, raise the limit (up to 65534).

**`zarbor profile`** shows each column's kind, missing count, distinct values, min, median and
max, and a count of values past the Tukey outer fence (`q1 - 3 IQR`, `q3 + 3 IQR`). That count
is reported, not acted on. The footer names unparsed values, columns that cannot inform a split
(all missing, or one value), and categoricals over the width limit. Every train and cv run
prints a one-line version.

**At prediction time, the model's schema wins.** `predict` and `blend` use the column kinds and
category dictionaries stored in the model, not ones re-sniffed from the new file. A slice of
valid data can look different (a holdout where every `PoolQC` is `NA`). A category not seen in
training is treated as missing.

## 3. Targets

Binary classification (`--objective=logistic`, the default) or regression
(`--objective=squared_error`). A text target is encoded by sorted class order, as in
scikit-learn's `LabelEncoder`, so `No < Yes` makes `Yes` the positive class; `--pos-label`
overrides. Refused with a reason: more than two classes, a single class, missing target
values, and numeric values outside [0, 1] under logistic. `--scale_pos_weight` weights the
positives.

## 4. Models

| `--algo` | what it is | key defaults |
|---|---|---|
| `gbdt` | gradient-boosted trees, three ways to grow them (below) | 500 rounds, lr 0.1, depth 6, `lambda` 1 |
| `random_forest` | bagged, unshrunk trees, averaged | 300 trees, 1024 leaves, bootstrap, sqrt(p) features per split |
| `linear` | logistic or linear regression, L1 + L2 | `lambda` 1, L-BFGS (OWL-QN for L1) on standardised columns |

**Tree growth** (`--grow_policy`) for `gbdt`:

- `depthwise` (default): XGBoost's `hist` method, level by level to `max_depth`.
- `lossguide`: LightGBM's method, always splitting the leaf with the best gain, up to
  `--max_leaves`. Use `--max_depth=0 --max_leaves=31` for LightGBM's defaults; add
  `--sampling=goss` for GOSS.
- `symmetric`: CatBoost's oblivious trees, one split per level shared by every node
  (section 7).

All three share one core: histogram binning, Newton leaves, L1 + L2 (`alpha`, `lambda`),
`min_child_weight`, `min_child_samples`, `min_split_gain`, `max_delta_step`, and row and
column sampling (`subsample`, `colsample_bytree/bylevel/bynode`). A forest tree is a boosting
tree fitted to the raw target with shrinkage off.

Column-sampling counts are `max(1, floor(n * rate))`, as in XGBoost and scikit-learn, so a
fraction is a ceiling. `subsample` takes exactly `round(n * rate)` rows.

**Linear models** fit `sum(loss) + lambda/2 ||w||^2 + alpha ||w||_1` on standardised columns
(keep `--lin_standardize=true`). scikit-learn equivalents: `LogisticRegression(C=1/lambda)`,
`Ridge(alpha=lambda)`, `Lasso(alpha=alpha/n)`; `--lambda=0` gives plain least squares.
`--lin_solver=adam` is a slow first-order fallback; trust `lbfgs` and read `STALLED` in the
fit line if it appears.

**Missing values** sit in bin 0. At each split the tree tries sending them left and right and
keeps the better choice. If a node had no missing rows, they go to the larger child. That
matters when a column is complete in training but has holes later.

**Determinism.** The same data and flags give bit-identical models at any `--n_threads`.
Every parallel sum is reduced in a fixed chunk order, and every random stream is keyed by
seed and purpose, not by thread.

## 5. Categoricals

`--cat_split` chooses how a categorical column is split:

| | how | when |
|---|---|---|
| `ordinal` (default) | a cut on the level's dictionary id (first appearance) | cheap; fine when categoricals carry little signal |
| `optimal` | LightGBM's: levels sorted by gradient/hessian, then the best cut of that order; one-vs-rest at `max_cat_to_onehot` (4) levels or fewer | categoricals that matter (House Prices: −1369 RMSE); 1.05–1.25x the time |
| `ctr` | CatBoost's ordered target statistics and combinations | `symmetric` trees, logistic loss only (section 7) |

`optimal` is regularised with `cat_smooth` (10), `max_cat_threshold` (32) and
`min_data_per_group` (100), as in LightGBM. It is not the default: in a pre-registered test
it helped one dataset of three (archive: `RESULTS.md`). `tune` searches it whenever the data
has a categorical column.

## 6. Binning

Every numeric column is cut into at most `--max_bin - 1` real bins (default 256 = 255 + the
missing bin) before training. The rules every policy shares:

- **One bin per value when they fit.** A column with no more distinct values than the budget
  gets one bin per value, whatever the policy.
- **At least `--min_data_in_bin` rows per bin** (3, LightGBM's). A rarer value or bin merges
  forward. This also caps the bin count at rows / `min_data_in_bin`, so with a high `max_bin`
  the number of bins grows with the data. `--min_data_in_bin=1` keeps every value, as XGBoost
  does.
- **Cuts fall between runs of equal values**, with edges at the midpoint between neighbouring
  training values, so an unseen value goes to the nearer side.

`--bin_policy` decides where the cuts go when a column has more values than bins:

| policy | from | rule |
|---|---|---|
| `quantile` (default) | XGBoost `hist` | equal-count bins by XGBoost's rule; matches its cuts exactly |
| `greedy` | LightGBM | `GreedyFindBin`, with zero in a bin of its own |
| `logsum` | CatBoost | `GreedyLogSum`: merges the pair of runs whose log-weight score suffers least |
| `uniform` | — | equal widths |

**Choosing `max_bin`.** Binning acts as regularisation, and the right amount depends on the
data's noise, not its size. In 5-fold CV on six datasets, the best `max_bin` ranged from 8
(Ames, 1.5k rows) to 8192 (adult), with no relation to row count. Above 256 bins the AUC gains
were around 0.0005; the regression gains were 1-2%. On the one dataset tested both ways
(housing), the gain from wider bins disappeared once the other settings were tuned. So:

- leave 256 unless you are tuning;
- `tune` searches `max_bin` 8..1024 and `min_data_in_bin` 1..300 by default;
- more than 256 distinct values in a column is fine. Wider columns are stored as 16-bit at a
  small cost, but training time grows with the total number of bins, and very steeply past a
  few thousand per column.

## 7. CatBoost-style training

`--grow_policy=symmetric` builds oblivious trees, with CatBoost's split scores
(`--score_function=cosine`, its default, or `l2` / `gain`) and its rule for removing
redundant splits. A symmetric tree always reaches `max_depth` unless that rule ends it, since
CatBoost has no minimum leaf size. The rest of CatBoost is opt-in, one flag each:

| flag | what |
|---|---|
| `--cat_split=ctr` | ordered target statistics (Borders x 3 priors, Counter), one-hot up to `--one_hot_max_size` (2), `--model_size_reg` (0.5) |
| `--max_ctr_complexity` (4) | categorical combinations, CatBoost's tree CTRs |
| `--bootstrap_type=bernoulli\|bayesian\|mvs` | row weights for split scoring (`--subsample`, `--bagging_temperature`, `--mvs_reg`) |
| `--random_strength` | CatBoost's decaying noise on split scores |
| `--boosting_type=ordered` | ordered boosting over CatBoost's dynamic prefixes |
| `--has_time` (false) | read rows in file order; otherwise CatBoost's permutations (`--permutation_count` 4, `--permutation_block`) |
| `--leaf_estimation_iterations` | repeated Newton leaf steps (no backtracking) |

The closest equivalent to `CatBoostClassifier()` with its defaults:

    --grow_policy=symmetric --cat_split=ctr --lambda=3 --base_score=0
    --bin_policy=logsum --bootstrap_type=mvs --subsample=0.8 --random_strength=1

plus an explicit `--learning_rate` and `--n_rounds`, since CatBoost picks the rate by a formula.
Combinations that have not been verified against CatBoost are refused, not ignored: under
`symmetric`, that means column sampling, GOSS, linear leaves, `alpha`, `max_delta_step` and
`cat_split=optimal`; under `ordered`, `score_function=gain` and more than one leaf step.

## 8. LightGBM extras

- **GOSS** (`--sampling=goss`, `--top_rate` 0.2, `--other_rate` 0.1): keeps the rows with the
  largest gradients plus a reweighted random sample of the rest. zarbor ranks by `|g|`, as the
  paper and LightGBM's docs say. LightGBM's code ranks by `|g*h|`; use
  `--goss_rank=gradient_hessian` to match it. Like LightGBM, the first `(int)(1/learning_rate)`
  rounds (10 at 0.1) train on every row before sampling starts; `--goss_warmup=false` samples
  from the first round, as the paper does.
- **Linear leaves** (`--linear_leaves=1`, `--lin_leaf_lambda` 1, `--lin_leaf_max_terms` 8):
  each leaf fits an affine function of the numeric features on its root-to-leaf path. This
  helped squared-error regression (California, −0.006 RMSE) and hurt logistic classification
  on every dataset tried, at 1.3-2.2x the fit time. Off by default.

## 9. Memory

Tree building keeps one histogram per open leaf, of size (total bins) x 32 bytes.
`--histogram_pool_size` (MiB, default 2048, LightGBM's name) caps that memory. A tree that
would need more evicts the histograms of queued leaves with the lowest gain, and later builds
their children directly instead of by subtraction. The result is the same tree, more slowly.
Training is refused only if three histograms do not fit. The cap covers histograms only, not
the data or the binned matrix.

## 10. Cross-validation

`zarbor cv` reads and bins once, then trains each fold on the binned matrix. It reports the
**pooled** out-of-fold score (every row predicted by the one model that did not see it), the
per-fold mean and sd, and with `--repeats=N` the spread over N fold seeds. Folds are
stratified for classification. Predictions are pooled on the probability scale, since each
fold has its own base score.

**`--group-col=NAME` keeps each group in one fold.** Use it whenever a unit appears in several
rows (a panel, repeated measures, the same ZIP in two months). Otherwise CV measures memory,
not generalisation: on a two-month panel of ZIP codes, row-wise CV reported RMSE 10.51 and
grouped CV 14.17. The group column is dropped as a feature. `--oof=FILE` writes the
out-of-fold predictions.

**Early stopping is nested.** With `--early_stopping_rounds=N`, each fold sets aside a tenth of
its own training rows (stratified, and grouped when `--group-col` is given) and stops on those.
It never stops on the rows it is scored on, which would let the score pick its own stopping
point. `cv` reports the mean rounds kept: use that as `--n_rounds` for the final `train`.

## 11. Hyperparameter search

`--search` picks the strategy:

| | assumes | |
|---|---|---|
| `random` (default) | few settings matter | each trial draws every setting afresh |
| `grid` | nothing | every combination; continuous axes cut into `--grid-steps` points |
| `bayes` | good settings cluster | TPE, after `--warmup` random trials |
| `bandit` | a cheap score ranks like an expensive one | successive halving over fold counts |

The space is declared with `--param`, any config field: `--param=max_depth=4,5,6`,
`--param=lambda=0.1..50:log`, `--param=n_rounds=200..800:int`. Without `--param`, each
algorithm has a default space:

- **gbdt:** `learning_rate` 0.02..0.2 (log), `max_depth` 3..8, `lambda` 0.1..300 (log),
  `min_child_weight` 0.01..300 (log), `subsample` 0.5..1, `colsample_bytree` 0.1..1,
  `max_bin` 8..1024 (log), `min_data_in_bin` 1..300 (log), and `cat_split` {ordinal,
  optimal} when the data has a categorical column. `n_rounds` is not searched: each fold
  stops early (patience 50, cap 5000, nested as in `cv`), every trial line shows the rounds
  it kept, and the winner's rounds are printed for `train`. Pinning `--n_rounds` or
  `--early_stopping_rounds` turns this off.
- **random_forest:** `n_rounds` {100, 200, 300}, `max_leaves` 128..2048, `min_child_samples`
  {1, 5, 20}, `colsample_bynode` 0.2..1, and the same two binning ranges.
- **linear:** `lambda` 0.0001..1000 (log), `alpha` {0, 0.1, 1, 10, 100}, `lin_epochs`
  {100, 300, 600}, `lin_standardize`.

Rules that keep a search honest:

- **A flag pins its setting.** `--n_rounds=100` removes `n_rounds` from the default space.
  Giving the same setting as both a flag and a `--param` is refused.
- **`edge` lines** flag a winning value in the outer 5% of its range, or at the first or last
  of three or more numeric choices. The best value may lie outside the range: widen it.
- **The winner is re-run** from scratch and must reproduce exactly.
- **`--confirm=N`** re-scores the top N on `--confirm-seeds` (2) fold seeds the search never
  used. The searched best is the maximum of many noisy trials and is optimistic; believe the
  confirmed score.
- A trial whose settings are invalid is skipped, not fatal. Trials that change binning
  re-bin the data; the count is reported.

`--out=FILE` writes every trial as CSV. A tune trial runs a full `cv`, so a search costs
roughly `trials x folds` model fits.

## 12. Models on disk, predicting, blending

A `.zm` file holds the model and the binning schema: column names, kinds, bin edges and
category dictionaries. That way a new file is binned exactly as in training. Save/load round
trips are bit-identical, and older format versions still load. `zarbor info` prints a model's
settings and per-column bins.

`predict --label=y` also scores against a labelled holdout, decoding the label with the
model's own class order. `blend` averages several models' predictions, with optional
weights. Blending zarbor with an independent library (such as LightGBM) is where zarbor has
earned its keep: on House Prices, a zarbor + LightGBM blend beat either alone in 20 of 20
fold seeds. A blend of two zarbor configurations gained nothing.

## 13. Flag reference

Defaults in brackets. `random_forest` and `linear` override some of them (section 4).

**Data:** `--label`, `--pos-label`, `--drop` (repeatable), `--valid-frac` [0.2],
`--split-col`, `--split-seed` [1], `--max-bytes` [2 GiB], `--max_cat_levels` [255],
`--n_threads` [0 = all cores].

**Binning:** `--max_bin` [256], `--min_data_in_bin` [3], `--bin_policy` [quantile].

**Boosting:** `--algo` [gbdt], `--objective` [logistic], `--n_rounds` [500],
`--learning_rate` [0.1], `--base_score` [label mean], `--scale_pos_weight` [1],
`--early_stopping_rounds` [0 = off], `--verbose_eval` [10], `--seed` [0].

**Trees:** `--grow_policy` [depthwise], `--max_depth` [6], `--max_leaves` [0 = none],
`--min_child_samples` [20], `--min_child_weight` [1], `--lambda` [1], `--alpha` [0],
`--min_split_gain` [0], `--max_delta_step` [0], `--subsample` [1], `--bootstrap` [false],
`--colsample_bytree/bylevel/bynode` [1], `--histogram_pool_size` [2048].

**Categoricals:** `--cat_split` [ordinal], `--cat_smooth` [10], `--max_cat_threshold` [32],
`--max_cat_to_onehot` [4], `--min_data_per_group` [100].

**LightGBM:** `--sampling` [uniform], `--goss_rank` [gradient], `--goss_warmup` [true],
`--top_rate` [0.2], `--other_rate` [0.1], `--linear_leaves` [false], `--lin_leaf_lambda` [1],
`--lin_leaf_max_terms` [8].

**CatBoost:** `--score_function` [auto = cosine under symmetric],
`--leaf_estimation_iterations` [1], `--bootstrap_type` [none], `--bagging_temperature` [1],
`--mvs_reg` [CatBoost's rule], `--random_strength` [0], `--one_hot_max_size` [2],
`--model_size_reg` [0.5], `--max_ctr_complexity` [4], `--has_time` [false],
`--permutation_count` [4], `--permutation_block` [0 = CatBoost's rule], `--boosting_type`
[plain].

**Linear:** `--lin_solver` [lbfgs], `--lin_epochs` [300], `--lin_lr` [0.05], `--lin_tol`
[1e-7], `--lin_standardize` [true].
