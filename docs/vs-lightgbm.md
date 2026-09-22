# Are we building the tool, or fitting the data?

`bench/vs_lightgbm.py` exists to answer one question: does zarbor compute what
an independent implementation computes, on data it was never shaped around.
That comes apart from "does zarbor score well" exactly when a library has been
tuned to one frame, which is the failure worth guarding against.

Run it with `python bench/vs_lightgbm.py`. Every number below is from it.

## How the comparison is made honest

**One split, shared.** A `__split` column goes into a copy of the CSV and is
handed to `zgbdt --split-col`; the Python side masks on the same column.

**A noise floor, not a chosen tolerance.** The first version used LightGBM's
spread across `random_state`, which is exactly zero: with no row or feature
sampling LightGBM is deterministic, so that yardstick measured nothing. (The
same mistake as reporting sd 0.0000 from an unseeded `GroupKFold` earlier in
this project — a degenerate control reads as a very precise one.) The yardstick
used instead is LightGBM's own spread over `max_bin` in {63, 127, 255}: how
much the score moves when a defensible binning choice changes, which is the
largest *known* difference between the two. A gap inside that envelope is
implementation variation rather than a defect.

**Categoricals isolated.** Every dataset is run twice, with and without its
string columns, on both sides. A gap present in both is in the base learner; a
gap that opens only when categoricals are present is in the categorical path.

**A negative control.** zarbor is also run with ordinal categorical splits
against LightGBM's optimal ones. If that does not widen the gap, the harness
cannot detect a disagreement and its passes mean nothing.

## Result, before the parity work

| dataset | metric | LightGBM | zarbor | gap | binning envelope | gap/envelope | |
|---|---|---:|---:|---:|---:|---:|---|
| california | rmse | 0.453516 | 0.454806 | 0.00129 | 0.00703 | 0.18 | agrees |
| adult | auc | 0.928398 | 0.922598 | 0.00580 | 0.00016 | **37.4** | **differs** |
| adult, numeric only | auc | 0.874439 | 0.861136 | 0.01330 | 0.00045 | **29.7** | **differs** |
| bank | auc | 0.937388 | 0.937901 | 0.00051 | 0.00081 | 0.64 | agrees |
| bank, numeric only | auc | 0.889693 | 0.889626 | 0.00007 | 0.00093 | 0.07 | agrees |
| ames | rmse | 23936.1 | 23162.9 | 773.2 | 1063.8 | 0.73 | agrees |
| ames, numeric only | rmse | 25996.9 | 26396.7 | 399.8 | 1467.1 | 0.27 | agrees |
| housing | rmse | 13.1652 | 13.7448 | 0.5796 | 0.0310 | **18.7** | **differs** |
| housing, numeric only | rmse | 14.5347 | 14.6172 | 0.0825 | 0.3641 | 0.23 | agrees |

The negative control fires on every dataset that has categoricals: ordinal
splits always land further from LightGBM than optimal ones do. The harness can
see a difference, so its agreements are worth something.

## Two failures, and they are not the same failure

### 1. The categorical search leaves a third of the gain on the table

On `housing`, which is the frame this work was motivated by:

    LightGBM   13.1652 with categoricals, 14.5347 without   -> they are worth -1.3695
    zarbor     13.7448 with categoricals, 14.6172 without   -> they are worth -0.8723

The two agree to 0.23 envelopes with the categoricals dropped and disagree by
18.7 with them present. **zarbor captures 64% of what LightGBM extracts from
the same columns.** So the search in `categorical-splits.md` is directionally
right and quantitatively short.

Knob responses, one at a time, point at the mechanism rather than the constants:

| setting | RMSE | vs zarbor default |
|---|---:|---:|
| default (`cat_min_group` = `min_child_samples` = 20) | 13.7448 | — |
| `cat_min_group=100` (LightGBM's `min_data_per_group` default) | 14.3177 | +0.5729 |
| `cat_min_group=1` (no floor) | 14.5229 | +0.7781 |
| `cat_l2=0` | 13.6794 | −0.0655 |
| `cat_smooth=1` | 13.5593 | −0.1856 |

That the response to the group floor is **non-monotonic** — worse at 1, worse
at 100, best at 20 — is the tell. A regularisation constant does not usually
behave like that. It suggests the *mechanism* is wrong rather than its setting,
and there are two concrete suspects:

- **Levels below the floor are forced into the right child.** That is written
  down in `categorical-splits.md` as though it were free ("they are not thereby
  lost"), and it is not free: it is a hard constraint on the partition that
  LightGBM does not impose. At a floor of 1 nothing is excluded and rare levels
  are fitted; at 100 most levels are pinned right. Both are bad, which is
  exactly the shape observed.
- **`cat_l2` is added to the parent score as well as the children.** That was
  a deliberate choice for self-consistency. `cat_l2=0` improving suggests it
  biases the comparison against categorical splits rather than regularising
  them.

Neither is confirmed. Both are testable, and neither should be "fixed" by
tuning the constant on `housing`, which is the trap this document exists for.

### 2. `adult` disagrees with no categoricals in play at all

0.874 against 0.861 AUC on six numeric columns, 30 envelopes wide. Whatever
this is, it is not the categorical work: it is the base learner or the numeric
binning, and it is the only dataset of five that shows it.

The obvious suspect is tie handling in quantile binning. `adult` carries
`capital-gain` and `capital-loss`, both ~92% zeros, where the quantile edges
collapse onto a single value and how an implementation breaks that determines
how much of the column survives. Named as a suspect, not diagnosed.

## What this says about the question that was asked

zarbor agrees with LightGBM on three of five datasets and on four of five once
categoricals are removed. It is not fitted to `housing` — if it were, `housing`
is the one place it would agree. It disagrees precisely where the newest and
least-tested code is, which is the honest place for a young implementation to
be wrong.

But `--algo=lightgbm` should not ship claiming parity until (1) is closed. A
model named after another implementation that recovers two thirds of that
implementation's gain is a misleading name.


---

# Result after fixing both

`--cat_split=optimal --bin_policy=greedy`, same harness, same split:

| dataset | metric | LightGBM | zarbor | gap | envelope | gap/env | |
|---|---|---:|---:|---:|---:|---:|---|
| california | rmse | 0.453516 | 0.454602 | 0.00109 | 0.00703 | 0.15 | agrees |
| adult | auc | 0.928398 | 0.928496 | 0.00010 | 0.00016 | 0.63 | agrees |
| adult, numeric only | auc | 0.874439 | 0.874262 | 0.00018 | 0.00045 | 0.39 | agrees |
| bank | auc | 0.937388 | 0.937907 | 0.00052 | 0.00081 | 0.64 | agrees |
| bank, numeric only | auc | 0.889693 | 0.889812 | 0.00012 | 0.00093 | 0.13 | agrees |
| ames | rmse | 23936.1 | 23273.5 | 662.6 | 1063.8 | 0.62 | agrees |
| ames, numeric only | rmse | 25996.9 | 26170.3 | 173.3 | 1467.1 | 0.12 | agrees |
| housing | rmse | 13.1652 | 13.1644 | 0.00089 | 0.03104 | 0.03 | agrees |
| housing, numeric only | rmse | 14.5347 | 14.5890 | 0.05429 | 0.36409 | 0.15 | agrees |

Every row inside the envelope, and the ordinal control still widens the gap on
all three categorical sets, so the harness has not simply gone blind.

housing went from 18.7 envelopes to 0.03. adult, which disagreed with no
categoricals in play at all, went from 29.7 to 0.39.

## The second failure was binning, and the bisection found it in one step

Splitting `adult` column by column:

| column | LightGBM | zarbor quantile | zarbor greedy |
|---|---:|---:|---:|
| education-num | 0.71288 | 0.71288 | 0.71288 |
| age | 0.70339 | 0.70379 | 0.70337 |
| hours-per-week | 0.68601 | 0.68722 | 0.68577 |
| fnlwgt | 0.51917 | 0.52994 | 0.53420 |
| capital-loss | 0.56313 | 0.55014 | 0.56234 |
| capital-gain | 0.62242 | 0.60882 | 0.62223 |
| **all six** | **0.87444** | **0.86114** | **0.87426** |

`education-num` matching to the digit was the clue: sixteen distinct values, so
every value gets its own bin under any rule and there is nothing to disagree
about. The damage was in `capital-gain` and `capital-loss`, both ~92% zeros.

A quantile rule puts its cuts at fixed rank positions, so on such a column
almost every cut lands inside the mode, dedups against its neighbour, and the
budget collapses onto the few bins covering the 8% that carries the signal.
LightGBM's `GreedyFindBin` gives any value holding at least a bin's worth of
rows a bin to itself and recomputes the remaining budget over what is left.

`fnlwgt` is the one column greedy binning does *not* improve (0.0108 → 0.0150
against LightGBM). It is a census sampling weight with an AUC of 0.52 — noise —
and zarbor sits above LightGBM there, which is fitting a little more of that
noise. Recorded rather than explained away; it does not move the six-column
result.

## What stays off by default

Both fixes are opt-in. `--cat_split=optimal` and `--bin_policy=greedy` change
results, so making either the default would break the bit-identity that every
earlier protocol in this repo checks against. They are what `--algo=lightgbm`
will select.
