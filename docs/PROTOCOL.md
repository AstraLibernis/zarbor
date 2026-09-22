# Measurement protocol — optimal categorical splits, linear leaves

**Pre-registered 2026-09-22, before any change to `src/`.** Written first
precisely because both features were proposed by the same person measuring
them. Committed before implementation so the criteria cannot move afterwards.

Findings live in `RESULTS.md`. Derivations live in `categorical-splits.md` and
`linear-leaves.md`. None of the three belongs in `src/`.

## What is being tested

**F1 — optimal categorical splits.** A categorical feature is currently split
by an ordinal cut on its dictionary id, and those ids are assigned in order of
first appearance. The claim is that replacing this with the gradient-sorted
partition improves accuracy on tables with categorical columns.

**F2 — linear leaves.** A leaf currently emits one constant. The claim is that
fitting a small ridge over the features on the root-to-leaf path improves
accuracy on tables with smooth numeric structure.

Each claim may turn out false. The protocol is designed so that a false claim
is visible rather than absorbed.

## Datasets — fixed now, not chosen after seeing results

| id | task | metric | rows | features | string cols | max cardinality | role |
|---|---|---|---:|---:|---:|---:|---|
| `adult` | binary | AUC | 48,842 | 14 | 9 | 41 | F1 primary |
| `bank` | binary | AUC | 45,211 | 16 | 9 | 12 | F1 primary |
| `ames` | regression | RMSE | 1,460 | 79 | 43 | 25 | F1 primary, small n |
| `california` | regression | RMSE | 20,640 | 8 | 0 | — | **F1 negative control**, F2 primary |
| `housing` | regression | RMSE | 6,830 | 16 | 5 | 51 usable | motivating case, reported apart |

`california` carries no categorical column at all. F1 must leave it *bit
identical*. That is the control that catches an implementation which improves
things by accident — by perturbing binning, tie-breaking or feature sampling —
rather than by splitting categoricals better.

`housing` is the frame the hypothesis came from. Its result is reported in a
separate section and is not counted toward the success criterion, because a
hypothesis cannot be confirmed by the observation that produced it.

## Fixed configuration

Declared now, identical in both arms, not tuned for either:

    gbdt, n_rounds=500, learning_rate=0.05, max_depth=6,
    min_child_samples=20, lambda=1.0, max_bin=256
    objective: logistic for adult/bank, l2 for california/ames/housing

Evaluation is `zgbdt cv --folds=5`, pooled out-of-fold score, over
`--seed` in {0, 1, 2}. `housing` additionally uses `--group-col=Zip`.

Neither feature gets a hyperparameter search. A tuned new arm against an
untuned old arm measures the tuning, not the feature.

## Arms

    OLD   git 6167158, the binary as it stands
    NEW-OFF   the new binary with both features disabled
    NEW-ON    the new binary with exactly one feature enabled

Both features default to **off**. A single flag is the only difference between
NEW-OFF and NEW-ON, so no build, data, split or seed difference can be mistaken
for an effect.

## Controls that must pass before any result is reported

- **C1 — no silent drift.** NEW-OFF reproduces OLD to the last digit on all
  five datasets. If it does not, the refactor changed behaviour and every
  comparison downstream is void.
- **C2 — no effect where there is nothing to act on.** F1 on `california`
  (zero categorical columns) must be bit-identical to OFF.
- **C3 — round trip.** Save, load, predict reproduces in-process predictions
  exactly for models containing the new node kinds.
- **C4 — thread invariance survives.** The existing bit-identical-across-
  `--n_threads` property still holds with each feature on.
- **C5 — the old format still loads.** A `.zm` written by OLD loads and scores
  identically under NEW.

## Success criterion, fixed in advance

A feature is reported as **helping** only if, on the three F1-primary datasets
(F1) or on the regression datasets (F2):

    mean improvement across seeds  >  2 x (sd across seeds)

on **at least two independent datasets**, with no dataset showing a
significant regression by the same test.

Anything weaker is reported as **no measured effect**. Anything negative is
reported as **harmful**, with the numbers, and the feature is left off by
default or reverted. Wall-clock cost is reported alongside accuracy in every
case; a gain that costs 3x the fit time is a different proposition from a free
one and the reader gets to decide.

## What would falsify each claim

- F1 is false if gradient-sorted partitioning does not beat ordinal cuts on
  `adult`, `bank` and `ames` — the three tables that are mostly categorical.
- F2 is false if a linear leaf does not beat a constant leaf on `california`,
  which is eight smooth numeric columns and the most favourable case there is.

Recording this now so that a null result reads as a null result.
