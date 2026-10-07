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

## Not offered

- CatBoost's exact optimisers (`MaxLogSum`, `MinEntropy`): the same objective as `logsum`
  solved by dynamic programming. Slower, and no published evidence that exact beats greedy.
- Supervised binning (MDLP, ChiMerge): edges would depend on labels, and `cv` bins the whole
  frame once, so validation labels would shape training bins.
- Sampling rows before binning: the defect class fixed here is rare values going missing, and
  sampling reintroduces it. Measure binning time before considering it.

Which policy should be the default is open: it is to be decided by repeated paired CV across
several datasets, not by these parity results.
