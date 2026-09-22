# Results — optimal categorical splits, linear leaves

Measured against `PROTOCOL.md`, which was committed before `src/` was touched.
Raw runs: `bench/baseline_old.tsv`, `bench/f1_on.tsv`, `bench/f2_on.tsv`,
`bench/f2_on_scaled.tsv`. Reproduce with `bench/run.py` and compare with
`bench/summarise.py`.

Read `|d|/sd` as the pre-registered test: the effect counts only above 2.

## Verdict

**Neither feature meets the success criterion. Both ship off by default.**

| | criterion | what happened |
|---|---|---|
| **F1** optimal categorical splits | helps on ≥2 of adult/bank/ames, no significant regression anywhere | helps on **ames only**; adult regressed at n=3 and did not at n=8 |
| **F2** linear leaves | helps on ≥2 regression sets, no significant regression anywhere | helps on **california only**; adult and bank both regress, at 2–3x the fit time |

Both are real capabilities that do real work on the right data. Neither is a
default, and the numbers below are why.

## Controls

| | check | result |
|---|---|---|
| C1 | NEW-OFF reproduces OLD to the last digit, all 5 datasets × 3 seeds | **pass** |
| C2 | F1 on `california` (no categorical column) is bit-identical to off | **pass** |
| C3 | save → load → predict stable, with each feature on | **pass** |
| C4 | bit-identical across `--n_threads` 1/4/16, with each feature on | **pass** |
| C5 | a v2 model written by OLD loads and scores identically under NEW | **pass** |

C1 was re-run after every commit, including after the F2 work, so no result
below is contaminated by a change made for the other feature.

## F1 — optimal categorical splits

Pre-registered, n = 3 seeds:

| dataset | base | with F1 | delta | sd | \|d\|/sd | time | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| california (RMSE) | 0.454597 | 0.454597 | ±0 | 0 | — | 1.05x | **identical** (C2) |
| adult (AUC) | 0.923116 | 0.922811 | −0.000305 | 0.000121 | 2.53 | 1.25x | hurts |
| bank (AUC) | 0.935146 | 0.935332 | +0.000186 | 0.000167 | 1.11 | 1.16x | no effect |
| ames (RMSE) | 30359.3 | 28990.7 | **−1368.7** | 421.1 | 3.25 | 1.09x | **helps** |
| housing (RMSE) | 14.4987 | 13.9657 | **−0.5330** | 0.0486 | 10.96 | 1.06x | helps¹ |

¹ the motivating case, excluded from the criterion by design.

Secondary, n = 8 seeds (5 added **after** seeing the above, so it is not the
pre-registered test):

| dataset | delta | sd | \|d\|/sd | verdict |
|---|---:|---:|---:|---|
| adult | −0.000183 | 0.000282 | 0.65 | no effect |
| bank | +0.000399 | 0.000479 | 0.83 | no effect |
| ames | −1124.8 | 405.4 | 2.77 | helps |

The adult regression does not survive more seeds: at n = 3 the sd was
underestimated by a factor of two. The honest reading is that F1 is **narrow,
not harmful** — but the pre-registered verdict was computed at n = 3 and is
recorded as such.

Where it helps is exactly where the theory says it should. `ames` is 43
categorical columns out of 79; `housing`'s only surviving categorical is
`State`, and it still moves 0.53 RMSE. `bank`'s widest categorical has 12
levels, where an ordinal cut on 11 prefixes is already close to exhaustive, and
nothing moves. Cost is 1.05–1.25x.

**The ceiling matters more than the gain.** The bin type is `u8`, so
`data.zig` refuses any categorical with 256+ levels outright. On the `housing`
frame that means `City` (3,113), `Zip3` (753) and `Metro` (650) must be dropped
before zgbdt will read the file at all. F1 does nothing for them. High
cardinality remains a target-encoding problem, and lifting the bin width is a
separate change nobody has costed.

## F2 — linear leaves

As specified in `linear-leaves.md`, n = 3:

| dataset | base | with F2 | delta | sd | \|d\|/sd | time | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| california (RMSE) | 0.454597 | 0.448164 | **−0.006434** | 0.002194 | 2.93 | 1.68x | **helps** |
| adult (AUC) | 0.923116 | 0.922374 | −0.000743 | 0.000266 | 2.79 | 2.19x | hurts |
| bank (AUC) | 0.935146 | 0.933432 | −0.001714 | 0.000166 | 10.33 | 2.13x | hurts |
| ames (RMSE) | 30359.3 | 30598.7 | +239.4 | 823.4 | 0.29 | 1.44x | no effect |
| housing (RMSE) | 14.4987 | 14.2303 | −0.2684 | 0.0339 | 7.91 | 1.69x | helps¹ |

### A defect the measurement found

`lin_leaf_lambda` was added flat to the diagonal of the slope system. That
diagonal is `sum_h * var_j`, so a flat addition shrinks a column measured in
dollars by a different amount than one measured in people — it is not a ridge,
it is an arbitrary per-column shrinkage. `california`'s columns happen to share
a scale; `adult` carries `fnlwgt` in the hundreds of thousands next to `age` in
the tens, which is where the damage was.

Standardising each axis before the ridge and undoing it after (commit
`db7dde7`), n = 3 — **obtained after seeing the table above, and therefore not
a pre-registered result**:

| dataset | delta | sd | \|d\|/sd | time | verdict |
|---|---:|---:|---:|---:|---|
| california (RMSE) | −0.006039 | 0.000728 | 8.30 | 1.74x | helps |
| adult (AUC) | −0.000197 | 0.000021 | 9.58 | 2.21x | hurts |
| bank (AUC) | −0.000328 | 0.000134 | 2.45 | 2.15x | hurts |
| ames (RMSE) | −1014.3 | 863.5 | 1.17 | 1.34x | no effect |
| housing (RMSE) | −0.194150 | 0.027309 | 7.11 | 1.73x | helps¹ |

The fix cuts the classification damage by 4–5x (adult −0.00074 → −0.00020,
bank −0.0017 → −0.00033) and leaves california's gain intact. It does not
change the verdict: both classification sets still regress.

### What the pattern says

Linear leaves help squared-error regression and hurt logistic classification,
consistently, on every dataset here. That is not a tuning artefact:

* For regression the leaf value *is* the prediction, and a local slope is
  directly what a staircase was approximating.
* For logistic the leaf value is a log-odds, and an affine function of a raw
  feature is unbounded. A row at the edge of a leaf gets an extrapolated
  log-odds with nothing holding it in. The Newton weight `h = p(1−p)` also
  collapses for confident rows, so the slope is fitted almost entirely from the
  uncertain ones and then applied to all of them.

And it is never free: 1.3–2.2x fit time, because a linear leaf has no single
constant to add to a whole leaf span, so the booster's span fast path is
skipped and every row traverses the tree.

## Two bugs the protocol caught

Both would have shipped silently and neither was found by the accuracy numbers:

1. `dupeTrees` copied a tree's nodes but not its categorical mask store, so
   **every saved model containing a categorical split failed to load**. Caught
   by C3, which exists only because the protocol demanded a round-trip check.
2. The node reader built a struct literal out of side-effecting `Reader` calls
   whose fields are not in wire order. Caught by the same control.

Mutation testing then found a third gap, in the tests rather than the code:
deleting the un-standardisation after the Cholesky solve left every test
passing, because `synth` gives every column the same scale and a uniform
mis-scaling is invisible. The test that catches it asserts what the theory
requires — rescaling a column by 1000 changes no bin, so the model must not
move — and all three mutants now fail.

## If either is picked up again

* **F1** is worth turning on whenever categorical columns carry a real share of
  the signal, and is close to free. The thing to build next is not a better
  search but a wider bin, because the 255-level ceiling is what keeps it away
  from the columns that need it most.
* **F2** should probably be gated to `squared_error` rather than left as a
  global flag, on the evidence here. Before that, the extrapolation guard is
  the obvious missing piece: clipping each leaf's affine output to the range
  its own training rows spanned costs nothing and is exactly the failure the
  logistic results point at. Untested.
