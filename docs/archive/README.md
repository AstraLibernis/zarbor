# Archive

History: design notes, pre-registered experiments, investigations and full measurement
write-ups. Their conclusions are in the code and in the current documents
([guide.md](../guide.md), [correctness.md](../correctness.md)). Kept because the reasoning and
the dead ends are worth having, and because code comments cite them.

**Numbers here describe the code at the date shown, not the code today.** Where a current
document disagrees, the current document wins.

Archived 2026-10-07.

| file | last changed | what it is | current summary in |
|---|---|---|---|
| [parity.md](parity.md) | 2026-10-07 | tree-for-tree parity with XGBoost and LightGBM: method, bisection traces, LightGBM's two shortcuts, linear models vs scikit-learn, sampling statistics | correctness §1–2 |
| [catboost.md](catboost.md) | 2026-10-07 | the six stages of CatBoost-style training, each with its parity measurements | guide §7, correctness §1–2 |
| [binning.md](binning.md) | 2026-10-07 | binning policies, bin-level parity, `max_bin` timing by data size, the CV sweeps, the housing tune A/B | guide §6 |
| [stress.md](stress.md) | 2026-10-07 | the stress run over 16 datasets: steps, bugs found, costs at the extremes | correctness §4 |
| [arena.md](arena.md) | 2026-09-23 | **superseded numbers**: the first benchmark run, per-phase speed, linear leaves vs LightGBM | correctness §3 (re-run 2026-10-07) |
| [PROTOCOL-arena.md](PROTOCOL-arena.md) | 2026-10-07 | the benchmark's pre-registered protocol: dataset, honest split, envelope, accounting | correctness §3 |
| [why.md](why.md) | 2026-09-29 | what zarbor is for, with House Prices evidence (blend gain, maintenance cost) | README "When to use it" |
| [vs-lightgbm.md](vs-lightgbm.md) | 2026-10-07 | score-level comparison on five datasets, before and after the categorical and binning fixes | correctness |
| [parsing.md](parsing.md) | 2026-09-29 | the `NA` rule, `profile`, schema at predict time, missing direction; the measurements behind each | guide §2, §4 |
| [linear-solvers.md](linear-solvers.md) | 2026-09-29 | L-BFGS vs Adam, convergence reporting, agreement with scikit-learn | guide §4, correctness §1 |
| [goss.md](goss.md) | 2026-09-23 | why the GOSS row differed from LightGBM (`\|g*h\|` ranking) and the other divergences in goss.hpp | guide §8, correctness §1 |
| [PROTOCOL.md](PROTOCOL.md) | 2026-09-29 | pre-registered test of optimal categorical splits and linear leaves | guide §5, §8 |
| [RESULTS.md](RESULTS.md) | 2026-10-06 | its results: neither met its criterion, both ship off by default | guide §5, §8 |
| [categorical-splits.md](categorical-splits.md) | 2026-10-07 | design and derivation of optimal categorical splits | guide §5 |
| [linear-leaves.md](linear-leaves.md) | 2026-09-22 | design of linear leaves | guide §8 |
| [PROTOCOL-widecat.md](PROTOCOL-widecat.md) | 2026-09-22 | pre-registered test of widening the bin type to 16 bits | — |
| [wide-categoricals.md](wide-categoricals.md) | 2026-09-22 | its cost prototype and outcome, including three wrong performance guesses | — |

Harnesses referred to here that live outside the repo (competition data cannot be
redistributed): `~/workspace/research/gbdt-refs/parity/` (parity, binning, CatBoost),
`~/workspace/research/tune-ranges/` (range sweeps), `~/workspace/research/wide-bins/`
(`max_bin` timing), `~/workspace/research/stress/` (stress run).
