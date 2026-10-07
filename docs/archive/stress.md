# Stress run: every dataset at hand

**[MEASURED]** 2026-10-07 at 9d22cd2, ReleaseFast, 16 threads, Ryzen 7 9800X3D, nothing else
running. This checks whether zarbor can *process* the data, not how accurate it is. The
harness is `~/workspace/research/stress/stress.py`, outside the repo. Raw logs are
`stress-before-fixes.log` (at 5a6ed9e) and `stress.log` (after the fixes below).

## What ran

Each dataset went through 15 steps:

- `profile`;
- train with `--save`, then `predict` on the competition test file where there is one;
- the binning extremes: `max_bin` 2, and 65535 with `min_data_in_bin` 1 and 300;
- `logsum`, `greedy` and `uniform` at 4096;
- lossguide, symmetric, forest and linear, all at 4096;
- a "wide cats" step that keeps the high-cardinality text columns, with `--max_cat_levels=65000`;
- a 12-trial random `tune`, 3 folds, `--n_rounds=100` pinned.

| data | rows | what makes it hard |
|---|---:|---|
| Titanic (Kaggle) | 891 | text IDs (`Name`, `Ticket`), 77%-missing `Cabin` |
| Spaceship Titanic | 8.7k | `Cabin` 6.5k levels, `Name` 8.5k, booleans as text |
| House Prices raw | 1.5k | 43 categoricals, `NA` as a level |
| Housing affordability | 13.7k | ZIPs with leading zeros, dates, quoted commas, `City` 3.1k levels |
| Kaggriculture leaderboard | 10k | only text and dates besides the target |
| matplotlib `Stocks.csv` | 525 | a `#` comment line above the header, blank target values |
| sklearn `breast_cancer.csv` | 569 | header line is counts, not names |
| CTR categoricals (parity) | 30k | all-categorical |
| discrete regression (parity) | 100k | 13 six-value columns |
| airline original | 130k | spaces and slashes in column names |
| EV purchases | 669k | mixed types, a 286k-row test file |
| airline S6E10 | 700k | a 300k-row test file |
| EEG motor imagery, 8 sessions | 860k | near-unique floats around 2.8e5, a 99.9%-missing marker column |
| synthetic continuous | 1M x 20 | every value distinct |

## Bugs it found, all fixed

1. **Records longer than the header were cut to fit, silently** (954ae78). `Stocks.csv`
   loaded as 1 column of 11, and `breast_cancer.csv` as 4 of 31. Now train, cv, tune and
   predict refuse the file and name the first bad record; `profile` reports it and carries
   on. Short records are still read, with NaN for the missing fields, and are now noted.
2. **A blank line was a row with every field missing** (954ae78), which in `predict` is an
   extra output row. Blank lines are now skipped, as pandas does.
3. **The histogram ceiling refused trees** (9d22cd2). A forest's 1024 leaves over 20 columns
   at 4096 bins, or a depth-6 tree at 65535 bins, failed with
   `TreeTooLargeForHistogramBudget`. Inside `tune`, that error aborts the whole search.
   The ceiling is now `histogram_pool_size` (MiB, default 2048). Past it, queued nodes give
   up their histograms and their children are built directly, which gives the same tree.
4. **`tune` overrode pinned flags** (9d22cd2). `--n_rounds=100` still ran trials at 200-800,
   because the default space included `n_rounds`. Pinned axes now leave the space, and a
   `--param` axis that is also pinned is refused.
5. **`tune --help` and `cv --help` printed only `error: FlagNeedsValue`**. They now print
   their usage text, as `train` does.

## After the fixes

All 16 datasets go through every step, except where the input is refused on purpose:

- the two malformed files, `RaggedRows`;
- `Stocks.csv` with AAPL as the target, `MissingLabelValue` (AAPL is blank before 1997);
- the leaderboard with its text columns dropped, `NoFeatures`. It trains in the wide-cats
  step, which keeps them.

Each refusal states the reason and the fix. Every `predict` row count matches its test file.

The costs at the extremes, from the logs:

| run | before the fixes | after |
|---|---|---|
| synthetic 1M, `max_bin` 65535, floor 1 | refused | 274 s, 2.8 GB peak |
| synthetic 1M, forest at 4096 | refused | 45 s, 2.2 GB |
| synthetic 1M, `max_bin` 65535, floor 300 | 12.6 s | 12.6 s |
| EEG 860k, `max_bin` 65535, floor 1 | 72 s, 1.2 GB | same |
| EEG 860k, forest at 4096 | 20.5 s, 985 MB | same |
| synthetic 1M, `tune` (12 trials) | 343 s (pin ignored) | 90 s |

The peak can pass `histogram_pool_size`, because that setting bounds histograms only, not
the data or the binned matrix. The 274 s run is where eviction costs most: 1.3M histogram
bins, rebuilt instead of subtracted. A per-value floor of 300 gives the same AUC in 12.6 s.

## Open, not fixed

- On EEG, hold-out RMSE varies more than 5x between policies at 4096 bins (`logsum` 22.0,
  `greedy` 63.9, `uniform` 104). Accuracy was not this run's question. It is a lead for the
  binning-policy decision, not a known defect.
- The forest's 1024-leaf default reaches the histogram ceiling at 20 continuous columns and
  4096 bins. It now trains, but slowly. Its `tune` space now stops `max_bin` at 1024
  (docs/archive/binning.md).
