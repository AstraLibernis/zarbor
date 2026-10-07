# Binning

**Status: [MEASURED] 2026-10-07** against xgboost 3.4.1, lightgbm 4.7.0 and catboost 1.2.10.

Every numeric column is cut into at most `max_bin - 1` real bins (bin 0 is missing) before
training; trees only ever see bin indices. Where the cuts go decides which splits exist at all,
so it is a correctness question before it is a tuning one: the 2026-10-07 parity audit
(`parity.md`) traced zarbor's first divergence from XGBoost to a split the binning had made
impossible.

## The rules every policy shares

**One bin per value when they fit.** A column with no more distinct values than the budget
gets one bin per value, whatever `bin_policy` says (`bin_edges.zig`, `perValue`). This refines
every other binning of the column, so it can only add splits, and it costs no histogram width
beyond the budget. XGBoost, LightGBM, CatBoost and scikit-learn all do this. A value with
fewer than `min_data_in_bin` rows (default 3, LightGBM's) merges into the next;
`--min_data_in_bin=1` keeps every value, as XGBoost does.

**At least `min_data_in_bin` rows per bin, every policy.** A column with more values than the
budget gets no more bins than its rows can fill (rows / `min_data_in_bin`), and a bin still
short of the floor merges into the next. `greedy` does this as LightGBM does, per sign while
it places cuts. `quantile` and `logsum` merge after placing them (`floorBounds`), so their
bins can come out a little coarser than the floor. `uniform` gets only the cap on the bin
count. With a high `max_bin`, this floor is what sets the bin count, so the bins grow with
the data. XGBoost parity needs `--min_data_in_bin=1`.

**Cuts between runs of equal values, never inside one.** Each policy chooses *boundaries*
between consecutive distinct values; a cut inside a run would split equal values, which no
tree could use.

**Edges at midpoints.** A boundary between training values `a < b` becomes the edge
`(a + b) / 2`, an inclusive upper bound. Training rows land exactly as with the edge on `a`; a
value never seen in training falls to the nearer side, as in LightGBM, CatBoost and
scikit-learn (XGBoost sends it to the lower). Where no `f32` lies strictly between `a` and `b`,
or either is infinite, the edge stays on `a`. Old models keep their stored edges, so their
predictions do not change.

## The policies

Each applies only when the column has more distinct values than the budget.

| `bin_policy` | follows | rule | bin-level parity |
|---|---|---|---|
| `quantile` (default) | XGBoost `hist` | equal-count targets; the run nearest each target rank (by mid-rank) starts a bin; a target landing on an already-started run moves to the next run instead of being dropped | **identical partition** whenever XGBoost's summary is exact (at most `8 * max_bin` distinct values) |
| `greedy` | LightGBM | `GreedyFindBin` on each sign separately, zero in a bin of its own, budget split by the signs' share of nonzero rows | **identical partition and bin count** on every column tested |
| `logsum` | CatBoost `GreedyLogSum` (its default) | split the bin whose middle cut most raises the sum of log bin sizes, until the budget is used | **identical partition** on every column tested; equal-score ties are broken leftmost, CatBoost's by heap order |
| `uniform` | — | equal widths between min and max | cheapest; outliers waste bins |

### Measured parity

150,000 rows of Kaggle S6E10, `max_bin` 256 (255 real bins), comparing which rows share a bin
(edge values differ by convention). LightGBM's bins were read from a random-label model that
splits on nearly every bin and its counts checked with `Dataset.feature_num_bin`; CatBoost's
from `Pool.quantize` plus `save_quantization_borders`. Harness:
`~/workspace/research/gbdt-refs/parity/binparity.py`.

| column | distinct | `quantile` vs XGBoost | `greedy` vs LightGBM | `logsum` vs CatBoost |
|---|---|---|---|---|
| Flight Distance | 3,167 | 77% of rows in matching bins (sketch approximate) | identical | identical |
| Age | 74 | identical | identical | identical |
| Departure Delay | 108 | identical | identical (81 bins each) | identical |
| Arrival Delay | 106 | identical | identical (79 bins each) | identical |

XGBoost's `quantile` gap on Flight Distance is its sketch, not the rule: the summary keeps
`8 * max_bin` entries, and above that its cuts are approximate. On 2,000 rows of the same
column (794 distinct values) the partitions are identical. zarbor counts exactly and does not
copy the approximation.

With the default policy and `--min_data_in_bin=1`, zarbor's depthwise booster now matches
XGBoost's predictions row for row through 500 rounds on the parity data (`parity.md`), where
the old `quantile` diverged at round 5.

## What changed on 2026-10-07

- `quantile` merged rare values even in a six-value column, and dropped every cut that
  repeated an atom (a 90%-zero column kept a handful of bins for the other 10%). Both fixed:
  the per-value floor, and XGBoost's advance rule.
- `greedy` gained LightGBM's zero bin; it had ported `GreedyFindBin` but not
  `FindBinWithZeroAsOneBin`.
- `logsum` is new.
- Edges moved from the lower training value to the midpoint.

28 of 55 golden outputs changed (Titanic's `age` and `fare` now bin differently); re-baselined.

## Choosing `max_bin` (measured 2026-10-07)

**[MEASURED]** at ce61145, ReleaseFast, 16 threads, Ryzen 7 9800X3D. Defaults otherwise
(`quantile`, depthwise, zarbor's default rounds and hold-out split). Time is the hyperfine mean
of 3 runs after 1 warm-up and covers the whole command (CSV load, binning, training, saving).
Peak RSS is from getrusage. "Bins used" is the total over numeric columns from `zarbor info`.
"Wide" counts the columns stored as u16. The hold-out metric is one run on one split, so treat
differences under about 0.001 AUC or 0.003 RMSE as noise (California moves non-monotonically
by that much). The harness is `~/workspace/research/wide-bins/timing.py`, outside the repo.

| data | max_bin | bins used | wide | train s | peak MB | hold-out |
|---|---:|---:|---:|---:|---:|---|
| California 20k | 64 | 501 | 0 | 0.14 | 16 | rmse 0.4548 |
| | 256 | 1845 | 0 | 0.25 | 18 | rmse 0.4467 |
| | 1024 | 6672 | 7 | 0.61 | 30 | rmse 0.4488 |
| | 4096 | 20715 | 7 | 1.74 | 65 | rmse 0.4505 |
| | 16384 | 48047 | 7 | 3.49 | 134 | rmse 0.4482 |
| Adult 49k | 64 | 337 | 0 | 0.22 | 23 | auc 0.91913 |
| | 256 | 631 | 0 | 0.20 | 23 | auc 0.92591 |
| | 1024 | 1399 | 1 | 0.27 | 23 | auc 0.92604 |
| | 4096 | 4471 | 1 | 0.50 | 30 | auc 0.92627 |
| | 16384 | 16759 | 1 | 1.50 | 58 | auc 0.92633 |
| Airline 100k | 64 | 346 | 0 | 0.34 | 46 | auc 0.95808 |
| | 256 | 564 | 0 | 0.36 | 48 | auc 0.95820 |
| | 1024 | 1332 | 1 | 0.42 | 50 | auc 0.95833 |
| | 4096 | 2890 | 1 | 0.59 | 49 | auc 0.95912 |
| | 16384 | 2890 | 1 | 0.57 | 48 | auc 0.95912 |
| Airline 700k | 64 | 346 | 0 | 1.58 | 180 | auc 0.95765 |
| | 256 | 676 | 0 | 1.52 | 180 | auc 0.95800 |
| | 1024 | 1444 | 1 | 1.60 | 183 | auc 0.95859 |
| | 4096 | 3670 | 1 | 1.88 | 183 | auc 0.95869 |
| | 16384 | 3670 | 1 | 1.96 | 184 | auc 0.95869 |
| Synthetic 1M x 20 continuous | 64 | 1280 | 0 | 2.69 | 327 | auc 0.84106 |
| | 256 | 5120 | 0 | 3.28 | 336 | auc 0.84155 |
| | 1024 | 20480 | 20 | 6.27 | 317 | auc 0.84163 |
| | 4096 | 81920 | 20 | 15.00 | 354 | auc 0.84157 |
| | 16384 | 327680 | 20 | 72.32 | 965 | auc 0.84140 |

What it shows:

- Cost follows **bins used**, not `max_bin`. Once every distinct value has its own bin
  (Airline at 4096), raising `max_bin` changes nothing and costs nothing.
- Wide (u16) storage has no visible penalty of its own: Airline 100k goes from 256 to 1024
  bins, one column wide, for +17% time. The cost comes from histogram size. Wall time is
  roughly flat up to about 5k bins in total and then grows faster than linearly: on the
  synthetic set, 82k to 328k bins is 4x the bins and 4.8x the time, with memory going from
  354 to 965 MB.
- The accuracy gain depends on the data. It is real but small where columns have a few
  thousand distinct values and there are many rows (Airline: +0.0009 AUC at 100k, +0.0007 at
  700k, both peaking at one bin per value). There is none on small data (California) or on
  continuous data, where the synthetic set peaks at 1024 and declines after it.

Guidance: keep 256 as the default. The single-split accuracy above is too noisy to choose
`max_bin` by data size. In 5-fold CV the best value does not follow row count: Ames, with
1.5k rows, is best at 8; housing, with 6.8k rows, at 1024; California, with 20k rows, at 1024
and overfitting past 2048. Nor does one `min_data_in_bin` suit them all: with `max_bin`
16384 and `greedy`, the best floor is 1 (adult, housing), 3 (bank), 10 (airline), 30 (Ames)
or 100 (California). The bin count acts as regularisation, so it is a setting to tune.
`tune`'s default space searches `max_bin` 8..16384 and `min_data_in_bin` 1..300 (both
log), and `tune` prints an `edge` line when the winner lands in the outer 5% of any range.

### CV sweep (5-fold), 2026-10-07

**[MEASURED]** at 055a295 (before the floor applied to all policies), ReleaseFast, default
gbdt, `quantile`, `min_data_in_bin` 3. Each cell is the out-of-fold score from one CV run
with one fold seed: AUC for adult, bank and airline (higher is better); RMSE for California,
Ames and housing (lower is better). Bold marks the best in each row. Harness:
`~/workspace/research/tune-ranges/ramp.py` plus direct `zarbor cv` runs above 512.

| data (rows) | 8 | 16 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|---|---|---|
| adult (49k) | | .90530 | .92228 | .92787 | .92755 | | .92767 | .92770 | .92794 | **.92806** |
| bank (45k) | | .92870 | .93162 | .93285 | .93306 | .93290 | .93321 | .93329 | .93327 | **.93338** |
| airline (100k) | | .95456 | .95495 | .95521 | .95559 | .95550 | .95583 | **.95593** | .95589 | .95589 |
| California (20k) | | .5035 | .4573 | .4502 | .4489 | .4489 | **.4485** | .4503 | .4509 | .4535 |
| Ames (1.5k) | **30041** | 30719 | 31400 | 30923 | 30943 | | | | | |
| housing (6.8k) | | 14.513 | 14.253 | 14.137 | 14.026 | 13.920 | **13.889** | 13.905 | | |

Ames was also run at 4 (32659) and 32 (30815). The scores do not climb to a single peak:
airline and bank dip at 512 and rise again, so a search that stops at the first decline
finds a false optimum.

`min_data_in_bin`, with `max_bin` 16384 and `greedy` (the only policy that used it then):

| data | 1 | 3 | 10 | 30 | 100 | 300 |
|---|---|---|---|---|---|---|
| adult (AUC) | **.92831** | .92815 | .92761 | .92720 | .92520 | .91988 |
| bank (AUC) | .93336 | **.93338** | .93330 | .93300 | .93264 | .93195 |
| California (RMSE) | .45286 | .45055 | .44963 | .44895 | **.44874** | .45792 |
| Ames (RMSE) | 31711 | 31569 | 31092 | **30404** | 31628 | 30938 |
| housing (RMSE) | **13.915** | 13.945 | 13.926 | 13.995 | 14.152 | 14.408 |
| airline (AUC) | .95590 | .95589 | **.95604** | .95579 | .95573 | .95530 |

With the best floor for each dataset, `greedy` with an unbounded `max_bin` beats the old
default (`quantile` at 256) on all six. The AUC gains are about 0.0003 to 0.0008, close to
the noise of a single CV run. The regression gains are clearer: Ames −1.7%, housing −0.8%.

## Not offered

- CatBoost's exact optimisers (`MaxLogSum`, `MinEntropy`): the same objective as `logsum`
  solved by dynamic programming. Slower, and no published evidence that exact beats greedy.
- Supervised binning (MDLP, ChiMerge): edges would depend on labels, and `cv` bins the whole
  frame once, so validation labels would shape training bins.
- Sampling rows before binning: the defect class fixed here is rare values going missing, and
  sampling reintroduces it. Measure binning time before considering it.

Which policy should be the default is open: it is to be decided by repeated paired CV across
several datasets, not by these parity results.
