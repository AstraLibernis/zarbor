# Protocol: zarbor against the reference implementations

Pre-registered 2026-09-23, before any number in `docs/archive/arena.md` was produced.

One question, asked of every model zarbor ships: **against the implementation
that defines its family, how close is the score and how long does it take.**

`docs/archive/vs-lightgbm.md` already answered a narrower version of this — is the
GBDT *correct* — on five small datasets. This is the wider one: all four
models, one large dataset, accuracy *and* wall time.

## The dataset, and why this one

`kaggle-ev-purchases/data/train.csv` — Kaggle Playground S6E9.

    668,665 rows x 13 features, binary target, ROC-AUC
    6 numeric, 6 string categoricals (2-4 levels each), 1 integer-coded ordinal
    zero nulls, zero duplicate rows, zero duplicate ids
    17.46% positive

Chosen over `kaggle-housing` on four counts, and the choice is not a
preference:

1. **No missingness and no duplicates.** Nothing has to be imputed, so no
   imputation choice can differ between implementations and be mistaken for an
   algorithmic difference.
2. **No panel structure.** Housing is ZIP x Month, so an honest split has to
   hold out whole groups; the two sides would have to agree on group folding
   before they could disagree on anything else. Here rows are independent.
3. **Large enough to time.** 535k training rows puts every fit in the
   0.2-30 s range, where a wall-clock difference is signal. Housing's 13,660
   rows would put several fits under 100 ms, where process startup dominates.
4. **A known ceiling.** `kaggle-ev-purchases/README.md` establishes ~0.9420
   AUC as the ceiling on these columns, by a residual test showing no
   learnable structure remains. A benchmark where every correct implementation
   is *supposed* to converge to the same number is one where a wrong
   implementation is visible.

Housing is the dirty one on purpose, and its own `LOG.md` records that CV and
the leaderboard disagree there at the scale this work operates at. Measuring
implementation parity on a frame where the measurement itself is contested
would confound the two.

The label is `Will_Buy_EV` (`Yes`=1). `id` is dropped as an identifier.

## Two kinds of comparison, not one

This distinction governs how every row below should be read.

**Parity comparisons — same algorithm, two implementations.** zarbor's `gbdt`
is XGBoost's algorithm; `gbdt --grow_policy=lossguide` is LightGBM's. A gap
here is a *defect* in one of them, and is judged against a noise floor.

**Family comparisons — same idea, deliberately different construction.**
zarbor's `random_forest` splits on binned features under a 1024-leaf cap;
sklearn's splits on exact values with no cap. zarbor's `linear` fits the
binned, one-hot design matrix; sklearn's fits raw standardised columns. These
are not the same estimator and are not expected to agree to any tolerance.
The question is whether zarbor lands in the same accuracy neighbourhood, and
what it costs. A gap here is a *design difference* and has to be attributed to
one before it means anything.

Calling both "a comparison" without the distinction is how a design choice
gets reported as a bug, or a bug excused as a design choice.

## Model map

| # | kind | zarbor | reference |
|---|---|---|---|
| A | parity | `--algo=gbdt` (depthwise) | XGBoost 3.4.1 `tree_method=hist`, `grow_policy=depthwise` |
| B | parity | `--grow_policy=lossguide --cat_split=optimal --bin_policy=greedy` | LightGBM 4.7.0 |
| C | parity | B `+ --sampling=goss` | LightGBM `boosting_type=goss` |
| D | family | `--algo=random_forest` | sklearn 1.9.1 `RandomForestClassifier` |
| E | family | `--algo=linear` (L-BFGS / OWL-QN) | sklearn `LogisticRegression` |

`cv` and `tune` are not models and are out of scope here.

C# was considered and rejected: for A, B, D and E the canonical implementation
*is* the C/C++ core behind the Python package. ML.NET would add a
reimplementation to compare against rather than the original.

## Honest split

One 80/20 stratified split, seed 20260923, written out as two files --
`ev_train.csv` and `ev_valid_only.csv` -- that both sides read. Neither side
chooses its own split.

**Two files, not one file with a `__split` column.** The column version was
used first and was wrong in a way worth recording: `--split-col` makes zarbor
read the whole CSV and quantise it *before* splitting, so the bin edges are
derived partly from validation rows. XGBoost and LightGBM bin inside `fit()`
and therefore see training rows only. That is a real asymmetry -- small, but
in zarbor's favour and invisible in the output. Separate files remove it.
Measured cost of the fix: A moved 0.941323 -> 0.941395.

## Accuracy

Every model writes its validation-set probabilities to disk. All of them are
then scored by **one referee** — `sklearn.metrics.roc_auc_score` over the
written file — so a metric bug on either side cannot hide inside that side's
own reported number. zarbor's self-reported AUC is captured too and compared
to the referee's; a disagreement there is itself a finding.

**Noise floor, not a chosen tolerance.** For the three parity rows, the
envelope is the reference's own spread over `max_bin` in {63, 127, 255} — how
far the score moves when a defensible binning choice changes, which is the
largest *known* legitimate difference between two histogram implementations.
A gap inside the envelope is implementation variation. This is the yardstick
`docs/archive/vs-lightgbm.md` arrived at after the first attempt (spread over
`random_state`) turned out to measure nothing, because both libraries are
deterministic without sampling.

No envelope is claimed for D and E: the two constructions differ by design,
so there is no "same computation" whose variation could bound a gap.

## Speed, and the accounting that makes it fair

Timing is measured per phase, because a single number cannot say *where* an
implementation leads or lags, and because the phase boundaries do not fall in
the same place on both sides. Four phases:

| phase | zarbor | reference |
|---|---|---|
| **read** | `csv.zig`, reported as `read` | `pd.read_csv` |
| **prepare** | `quantise`, reported as `bin` | nil for xgb/lgb; one-hot + scale for sklearn |
| **fit** | reported as `train` | `.fit()` |
| **predict** | reported as `predict` | `.predict_proba()` |

Two rules follow, and the first version of this benchmark broke both.

**1. Binning is paid by whoever does it, whenever they do it.** XGBoost and
LightGBM bin *inside* `fit()`. zarbor bins before it and reports the two
separately. Comparing zarbor's `fit` against their `fit` therefore charges
them for work zarbor is not charged for. The headline comparison is
**prepare + fit + predict**, which contains the binning on both sides wherever
it happens. Quoting zarbor's bare `fit` against a reference `fit` overstated
its advantage by roughly 10 percentage points and reversed the sign on one
row.

**2. Categorical dictionary building is part of parsing, on both sides.**
zarbor's parser builds level dictionaries during the parse. pandas is
therefore read with `dtype=category` rather than a plain read followed by
`.astype("category")` -- otherwise that work lands in `prepare` on one side
and `read` on the other. It is also faster that way (136 ms against 180 ms),
so this is not a handicap applied to make a point.

**Neither side is charged for work the other is not doing.** zarbor trains
with `--valid-frac=0 --verbose_eval=0`, so it performs no per-round
validation prediction and no per-round metric; the reference side is fitted
with no `eval_set`. The earlier version left zarbor's per-round validation
running and then subtracted an estimate of it, which is a correction rather
than a measurement.

**Parsing is reported separately and belongs to no model.** Every model pays
it before it sees a bin. It is measured once, zarbor's `csv.zig` against
`pandas.read_csv`, and is never folded into a model's time. This is also why
parsing now lives in its own module.

Conditions: `--n_threads=16` and `n_jobs=16` / `num_threads=16`, matching the
16 hardware threads. **n=5 repeats, medians reported.** A single timing is not
a measurement; this repo has already recorded that a seeded, temperature-0 run
is not reproducible here, and wall clock is worse.

## Matched hyperparameters

Defaults differ between all four libraries, so defaults are not used. Every
parameter that exists on both sides is set explicitly and identically:

    n_rounds/n_estimators 200   learning_rate 0.1   max_depth 6
    lambda/reg_lambda 1.0       min_child_weight 1.0
    min_child_samples 20        max_bin 256         subsample 1.0

For B and C, capacity is set by `max_leaves`/`num_leaves` = 31 with depth
uncapped, which is what leafwise growth is for.

Known irreducible mismatch, recorded rather than smoothed over: LightGBM's
`max_bin` excludes its missing bin where zarbor's and XGBoost's include it, so
LightGBM is given 255 where the others get 256. That is inside the envelope
the envelope is built from.

**Erratum (2026-10-07, from the audit in `parity.md`).** Two statements above
are wrong. XGBoost's `max_bin` does not include a missing bin: 256 gives it 256
real bins, where zarbor's 256 gives 255 real plus missing. And
`min_child_samples` does not exist in XGBoost, so row A ran zarbor with a
constraint XGBoost did not have. Results recorded under this protocol stand as
measured; the comparison was not as like-for-like as stated.

For D, sklearn is run twice — once with `max_leaf_nodes=1024` to mirror
zarbor's cap, and once at its own uncapped default — because the cap is the
single largest construction difference and pinning it is the only way to tell
whether a gap comes from the cap or from binning.

For E, sklearn is likewise run twice: on raw standardised columns with one-hot
categoricals, and on zarbor's own binned design matrix. If the two sides agree
on the second and differ on the first, the difference is the design matrix and
not the solver.

## What would falsify a "pass"

- Any parity row landing outside the max_bin envelope.
- zarbor's self-reported AUC disagreeing with the referee's on the same
  predictions.
- A speed claim whose n=5 range overlaps the comparison's range — reported as
  "no difference measured", not as a win.
