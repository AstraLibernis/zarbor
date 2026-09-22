# Measurement protocol — widening the bin type

**Pre-registered before `src/` is touched**, same discipline as
`PROTOCOL.md`. The change is motivated by a measurement I took myself
(`wide-categoricals.md`), which is exactly the situation the pre-registration
exists for.

## The change

`data.max_bins` rises from 256 to 65,536 and `Dataset.bins` / `bins_rm` become
`[]u16`. `Node.threshold` and `Split.threshold` follow. The numeric `max_bin`
default stays at 256 — this lifts a ceiling, it does not change how numeric
columns are cut.

The categorical split representation changes with it. A 256-bit mask was
affordable; at 65,536 levels it would be 8 KiB per split node. It becomes a
sorted list of the ≤ `max_cat_threshold` level ids on the left child, which is
smaller than the mask at every cardinality and is what the cap was already for.

Model format goes to 4. Versions 2 and 3 still load.

Not a comptime-generic bin type. That was the plan until half C of
`bench/widecat.zig` measured a `u16` index at 0.98–1.01x of `u8` in the
accumulation kernel: the loop is bound by the scattered histogram update, not
by the index read. One code path is therefore both simpler and free.

## Success criterion, fixed in advance

Same grid, seeds and fixed hyperparameters as `PROTOCOL.md`, plus **one new
dataset** that the current code cannot read at all:

    housing-wide = the housing frame with City, Metro and Zip3 kept
                   as categoricals rather than dropped

The change succeeds if **both** hold:

1. **Accuracy.** On `housing-wide` with `--cat_split=optimal`, RMSE improves
   over the same frame with only `State` by more than 2 sd across 5 seeds. The
   LightGBM reading of the same comparison is −1.00 RMSE at 17 sd; anything
   under −0.5 means zarbor's implementation is leaving most of it behind and
   should be treated as a partial failure, not a win.
2. **Cost.** On the four datasets that fit in `u8` today (`california`,
   `adult`, `bank`, `ames`), wall clock rises by **no more than 5%** against
   the current binary, and accuracy is **bit-identical**.

Criterion 2 is the one that can fail quietly. Widening the bin type touches
every workload, including every workload that gains nothing from it, and a
silent 20% tax on all of them would not show up anywhere in criterion 1.

## Controls

- **W1.** On the four `u8`-fitting datasets, the new binary reproduces the
  current one to the last digit. Widening an index must not move a model.
- **W2.** A v2 and a v3 `.zm` both load under the new binary and score
  identically to what they scored before.
- **W3.** Bit-identical across `--n_threads` 1/4/16, with the wide column and
  `--cat_split=optimal` both in play.
- **W4.** Save → load → predict is exact for a model containing a categorical
  split over more than 256 levels.
- **W5.** A categorical with more than `max_cat_threshold` levels on the left
  is rejected at fit time rather than silently truncated on load.

## What would falsify the change

If `housing-wide` does not beat `State`-only by at least 0.5 RMSE, the
implementation is wrong rather than the idea — LightGBM gets 1.00 from the same
columns under matched hyperparameters, and that gap is the check on whether
zarbor's categorical search is really doing what `categorical-splits.md`
claims.

## Reference

LightGBM 4.7.0 under matched hyperparameters is the oracle for criterion 1, and
the number to beat or explain. It is not a substitute for the implementation;
it is how the implementation is known to be right.
