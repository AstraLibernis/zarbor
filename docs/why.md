# What zarbor is for, and when to reach for LightGBM instead

Written 2026-09-23, after a run at Kaggle's House Prices in which LightGBM beat
zarbor on the leaderboard metric and the obvious question got asked: if the
reference implementation keeps winning, what are four hand-built models for?

Every number below names the file or command that produced it. Re-run them
rather than trusting this page; it is a snapshot, and snapshots rot.

## First, the premise is wrong: LightGBM does not continually win

Source: `docs/arena.md`, harness `bench/arena/arena.py`, raw
`bench/arena/results.json`. Kaggle Playground S6E9, 668,665 x 13, binary,
ROC-AUC, one shared stratified split, n=5 medians, 16 threads.

`parity` rows are the same algorithm in two implementations, judged against the
reference's own spread over `max_bin` in {63,127,255} -- a gap inside that
envelope is implementation variation, not a difference in quality.

| model | zarbor | reference | gap / envelope | verdict |
|---|---:|---:|---:|---|
| gbdt depthwise vs XGBoost | 0.941395 | 0.941441 | 0.05 | agrees |
| gbdt leafwise vs LightGBM | 0.941371 | 0.941427 | 0.05 | agrees |
| gbdt + GOSS vs LightGBM | 0.940098 | 0.940247 | 0.20 | agrees |
| random forest vs sklearn | **0.939025** | 0.934829 | — | zarbor ahead |
| logistic vs sklearn | 0.937713 | 0.937729 | — | agrees |

Second source, different dataset, matched hyperparameters:
`bench/vs_lightgbm.py --only=ames` puts LightGBM at 23936.09 RMSE and zarbor at
23273.58, a gap of 662.51 against a 1063.84 binning envelope -- 0.62 envelopes,
zarbor ahead.

Third source, the competition itself
(`kaggle-house-prices/work/ablation.txt`), 20 fold seeds:

| | RMSE on log(SalePrice) |
|---|---:|
| zarbor tuned + `cat_split=optimal` | 0.12617 |
| LightGBM tuned | 0.12581 |
| difference | 0.00036, against a fold sd of 0.002 |

LightGBM led on this competition **because of one zarbor default**, not because
of the learner. `cat_split` defaults to `ordinal` here and LightGBM partitions
categorical levels optimally by gradient over hessian; on a table that is 43
categorical columns of 79, that one flag was worth -0.00248 (20/20 seeds,
p < 1e-4) and closed the entire gap.

## Four things that earn their place

### 1. Blend diversity, and it is measured

Source: `kaggle-house-prices/work/blend.txt`, harness `scripts/blend_eval.py`.
Every subset of four candidates, 20 fold seeds, all members scored on identical
folds (LightGBM is handed zarbor's own fold assignment via `PredefinedSplit`),
equal weights and no fitting.

| | RMSE | vs best single | wins | p |
|---|---:|---:|---:|---:|
| LightGBM alone (best single) | 0.12581 | | | |
| **zarbor + LightGBM** | **0.12375** | -0.00206 | 20/20 | <1e-4 |
| zarbor depthwise + zarbor leafwise | 0.12586 | +0.00005 | 8/20 | **0.86** |

Read the last row against the middle one. Two zarbor configurations blended
together are worth **nothing**. A zarbor and a LightGBM blended together are
worth a real -0.00206. What helped was the *independent implementation* --
different bin edges, different tie-breaks, different base score -- decorrelating
errors in a way a second configuration of the same library cannot.

That is the strongest single argument on this page, because it is the one where
zarbor does something no LightGBM setting can do.

### 2. The random forest is better and faster than sklearn's

Source: `docs/arena.md`, rows D.

| | AUC | model time |
|---|---:|---:|
| zarbor | **0.939025** | **5,346 ms** |
| sklearn, 1024-leaf cap | 0.934829 | 6,459 ms (1.21x) |
| sklearn, uncapped default | 0.933515 | 9,543 ms (1.79x) |

State the caveat with the claim: this is a `family` comparison, not parity.
zarbor's forest is histogram-binned with a leaf cap; sklearn's does exact
splits. It is a different algorithm winning, not the same one done better. It
is still the model you would rather run.

### 3. The linear models are ~6x faster at the same accuracy

Source: `docs/arena.md`, row E; `docs/linear-solvers.md`.

| | AUC | model time |
|---|---:|---:|
| zarbor logistic (L-BFGS) | 0.937713 | **203 ms** |
| sklearn LogisticRegression | 0.937729 | 1,204 ms (5.93x) |

Same answer to four decimals, six times faster. sklearn's is genuinely slow on
a wide binned design.

### 4. One binary, no dependencies, CV and search in-process

`zgbdt cv` reads and bins once and loops folds over the binned matrix;
`--repeats=N` extends that across fold seeds. `zgbdt tune` runs the search in
the same process. No Python harness shelling out per fold, no environment to
reproduce. The parser alone is 1.31x `pandas.read_csv` (135 ms against 177 ms
on 40 MB, `docs/arena.md`).

### A fifth, harder to price: reimplementing is how you learn what the reference does

`docs/goss.md`: LightGBM ranks GOSS candidate rows by `|g*h|`. The GOSS paper
says `|g|`, and so does LightGBM's own documentation. The code
(`src/boosting/goss.hpp`, line 127) says otherwise. That was found by building
GOSS, disagreeing with LightGBM by 1.42 envelopes, and going to read why.

You do not find that by calling a library.

## The honest other side

**The GBDT models are at parity, not ahead.** Their value is as the shared core
the other three are built on -- a forest tree is a boosting tree fed
`g = -y, h = 1` with shrinkage off, and the linear model runs on the same
binned matrix. One core, four models. Judged alone against XGBoost and
LightGBM they are a tie, and leafwise `fit` is 0.93x, i.e. slower.

**Maintenance is a real cost and it is not zero.** Defects found in a single
day on one competition file:

| | |
|---|---|
| `NA` not recognised as missing | the competition's `train.csv` would not load at all |
| column kinds re-sniffed at predict time | valid holdouts failed with `FeatureKindMismatch` |
| missing direction chosen by loop order | degraded predictions on any column complete in training |
| `cat_split` absent from `defaultSpace()` | 80 tuning trials could not find the one flag that mattered |

Earlier sessions add: `Fit.stalled()` returning `false` unconditionally for the
adam solver, so it returned coin-flip models silently; a predict loop
triplicated across three files, costing 279 ms where 109 ms was available.

A LightGBM user pays none of that.

## So: which do I use

**If the goal is the best tabular score for the least effort, use LightGBM.**
It is mature, it is tuned, it is free, and on House Prices it needed no
defaults audited.

**Use zarbor when** you want a second, genuinely independent model in a blend
(measured: -0.00206, 20/20); when the model is a random forest or a regularised
linear fit, where it is faster and at least as accurate; when a single static
binary with no Python matters; or when the point is to understand the algorithm
rather than to call it.

**Use both** when the score matters most. That is what the submission did:
0.12375 cross-validated, 0.12469 on the leaderboard, which is the top quartile
of entries that are not looking the answers up.

## How to re-derive everything here

    # parity and speed across all five models
    python bench/arena/arena.py

    # matched-hyperparameter agreement on five datasets
    python bench/vs_lightgbm.py

    # the competition work
    cd ../kaggle-house-prices
    python scripts/map_data.py        # what is in the data
    python scripts/split_shift.py     # is CV trustworthy here
    python scripts/ablate.py          # which default cost what
    python scripts/blend_eval.py      # does blending help, and with whom
