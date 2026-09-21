# zmodels

Gradient-boosted trees, random forests and regularised linear models for
tabular CSV data, in Zig with no dependencies. Plus cross-validation and
hyperparameter search as first-class subcommands, so the whole loop runs in
one process instead of a Python harness shelling out per fold.

    zig build -Doptimize=ReleaseFast

    ./zig-out/bin/zgbdt train.csv --label=y --save=m.zm
    ./zig-out/bin/zgbdt predict test.csv --model=m.zm --out=p.csv
    ./zig-out/bin/zgbdt cv   train.csv --label=y --folds=5
    ./zig-out/bin/zgbdt tune train.csv --label=y --search=bayes --trials=60

## Models

One core — histogram binning, categorical dictionaries, NaN-as-missing,
L1+L2, row and feature sampling, a thread pool — because XGBoost and LightGBM
*are* one algorithm with a different node-selection rule, and a forest tree is
a boosting tree fed `g = -y, h = 1` with shrinkage off.

| `--algo` | what it is |
|---|---|
| `gbdt` (default) | XGBoost-style depthwise boosting |
| `gbdt --grow_policy=lossguide --sampling=goss` | LightGBM-style |
| `random_forest` | bagged unshrunk trees, `sqrt(p)` features per split |
| `linear` | logistic/linear regression, L-BFGS with OWL-QN for L1 |

Every `Config` field is exposed as `--field=value` by comptime reflection, so
the flag surface and the struct cannot drift apart.

## Cross-validation

`zgbdt cv` reads and bins once, then loops folds over the binned matrix.
It reports the **pooled** out-of-fold score — every row predicted by the one
model that did not train on it — alongside the per-fold mean and sd, because
those answer different questions and get conflated. Folds are stratified for a
classification objective.

Predictions are pooled on the natural scale, not raw log-odds: each fold is a
different model with its own base score, and a metric over the pooled vector
compares rows *across* folds, so the scale has to mean the same thing in all
of them.

## Hyperparameter search

`--search` picks between four families that differ in what they assume:

| | assumption | cost |
|---|---|---|
| `grid` | none | product of the axes; re-tests an important knob once per combination of the unimportant ones |
| `random` | few parameters matter | every trial gives each knob a fresh value |
| `bayes` | good configs cluster | TPE: Parzen density over good vs bad, proposing where the ratio peaks |
| `bandit` | a cheap score ranks like an expensive one | successive halving over fold count |

The space is declared, not hard-coded:

    --param=max_depth=4,5,6,7        # explicit choices
    --param=lambda=0.1..50:log       # log-uniform range
    --param=n_rounds=200..800:int    # integer range

Values are rendered back through the same reflection that defines the flag
surface, so `tune` can vary any `Config` field — including ones added later —
without knowing anything about it.

**`--confirm=N` re-scores the leaders on fold seeds they never saw.** The
headline number is the maximum of many trials on one split and is optimistic
by construction; the confirmed one is what to believe. The tuner also re-runs
its own winner and warns if the reported configuration does not reproduce.

## Properties worth knowing

- **Deterministic across `--n_threads`.** Results are bit-identical, which
  means anything parallel groups floating-point additions by a *fixed* chunk
  count rather than one per thread. A test asserts it.
- **Models serialise.** A `.zm` carries the model *and* the binning schema,
  because a categorical bin *is* its dictionary id — without the level strings
  a second file would bin the same category differently. Round trips are
  bit-identical.
- **Targets encode by sorted class order**, matching scikit-learn's
  `LabelEncoder`. Encoding by order of first appearance silently fits the
  inverse model on a positive-class-first file. `--pos-label=NAME` overrides.
- **`zig build` defaults to Debug, which is ~8x slower.** Every run prints its
  build mode so a benchmark cannot quietly measure the wrong binary.

## Not implemented, deliberately

LightGBM's EFB and optimal categorical splitting, and CatBoost's ordered
boosting. Categoricals currently split on dictionary-id order, which is
arbitrary; that is the most valuable of the three to add next.

## Benchmarks

`bench/compare.py` pits each model against its reference (xgboost, lightgbm,
scikit-learn) on one shared split, both charged for binning and neither for
validation. It is the only Python here, and it exists because comparing
against a reference implementation means running it.

## License

MIT. See `LICENSE`.
