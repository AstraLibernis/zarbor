# zarbor

Gradient-boosted trees (XGBoost-, LightGBM- and CatBoost-style), random forests and
regularised linear models for tabular CSV data. Written in Zig, with no dependencies.
Cross-validation and hyperparameter search are built in, so the whole loop runs in one
process.

    zig build -Doptimize=ReleaseFast

    ./zig-out/bin/zarbor train.csv --label=y --save=m.zm        # train, hold-out score
    ./zig-out/bin/zarbor predict test.csv --model=m.zm --out=p.csv
    ./zig-out/bin/zarbor cv   train.csv --label=y --folds=5
    ./zig-out/bin/zarbor tune train.csv --label=y --trials=60 --confirm=3
    ./zig-out/bin/zarbor profile train.csv                     # what is in the file

## What it does

| | |
|---|---|
| **XGBoost-style** (default) | depthwise trees; reproduces XGBoost's trees through 500 rounds |
| **LightGBM-style** | `--grow_policy=lossguide --max_depth=0 --max_leaves=31`, GOSS, linear leaves, optimal categorical splits |
| **CatBoost-style** | `--grow_policy=symmetric`: oblivious trees, ordered target statistics, combinations, MVS/Bayesian bootstrap, ordered boosting, permutations |
| **Random forest** | `--algo=random_forest`; better and 2-3x faster than scikit-learn's on the benchmark |
| **Linear** | `--algo=linear`: logistic or linear regression with L1 + L2; scikit-learn's optimum, 3x faster |
| **Binning** | four policies (XGBoost's, LightGBM's, CatBoost's, uniform), any number of bins |
| **cv** | pooled out-of-fold score, repeated fold seeds, grouped folds |
| **tune** | random, grid, Bayesian (TPE) and bandit search; warns when a winner sits at the edge of its range; re-scores leaders on unseen folds |

Binary classification and regression. Models save to one file holding the model and its
binning schema, so new data is binned exactly as in training. Results are bit-identical at
any thread count.

## Is it right? Is it fast?

Given the same data and settings, zarbor grows the same trees as XGBoost, LightGBM and
CatBoost, to the 6-decimal print precision. On a 535k-row benchmark (2026-10-07) its model
work is faster than every reference:

| | reference | AUC gap | speed |
|---|---|---:|---:|
| gbdt depthwise | XGBoost | 0.000016 | 1.39x faster |
| gbdt leafwise | LightGBM | 0.000008 | 1.16x faster |
| random forest | scikit-learn | zarbor ahead by 0.0033 | 2.0x faster |
| logistic regression | scikit-learn | 0.000013 | 3.2x faster |
| CSV parsing | pandas | | 4.7x faster |

Evidence, method and the known differences are in [docs/correctness.md](docs/correctness.md).

## When to use it

For the best score with the least effort, LightGBM or CatBoost are mature and well tuned.
zarbor is worth reaching for when:

- **you are blending.** An independent implementation decorrelates errors. On House Prices, a
  zarbor + LightGBM blend beat either alone in 20 of 20 fold seeds (−0.00206 RMSE); two zarbor
  configurations blended gained nothing.
- **the model is a random forest or a linear fit**, where zarbor is faster and at least as
  accurate.
- **you want one static binary**, no Python environment, with CV and search in-process.
- **you want to know what the reference actually computes.** Building zarbor found, among
  other things, that LightGBM ranks GOSS rows by `|g*h|` (its docs say `|g|`) and skips
  features in descendants after one bad node.

## Documentation

- [docs/guide.md](docs/guide.md): every command and setting, and what zarbor does with your
  data: CSV rules, missing values, categoricals, binning, memory, cv, tune.
- [docs/correctness.md](docs/correctness.md): parity with each reference, speed, the stress
  run, how the code is tested, open questions.
- [docs/measurements.md](docs/measurements.md): the measured numbers behind code-level
  decisions.
- [docs/archive/](docs/archive/README.md): design notes, pre-registered experiments and
  investigations, kept as history.

## Build and test

Requires Zig 0.16.0.

    zig build -Doptimize=ReleaseFast   # the binary; plain `zig build` is Debug, several times slower
    zig build test                     # unit tests
    zig build golden                   # CLI end to end on Titanic vs test/golden/expected
    zig build test-tsan                # ThreadSanitizer
    zig build check                    # compile every artifact, test and benchmark

## Layout

- `src/cli/`: the `zarbor` command, one file per subcommand, using the library only through
  `src/root.zig`.
- Data: `csv.zig` parses (on vendored [zsift](https://github.com/AstraLibernis/zsift)),
  `csv_profile.zig` describes, `data.zig` and `bin_edges.zig` bin, and `schema.zig` re-bins new
  files.
- Trees: `tree.zig`, `builder.zig`, `hist.zig`, `split.zig`, `partition.zig`, `symmetric.zig`
  (CatBoost-style), `leaf_linear.zig`.
- Models: `booster.zig` (+ `goss.zig`), `forest.zig`, `linear.zig` (+ `lin_solve.zig`);
  `fitted.zig` puts them behind one interface, and `model.zig` is the file format.
- Settings and evaluation: `config.zig`, `objective.zig`, `metric.zig`, `cv.zig`, `tune.zig`.
- Runtime: `pool.zig` (thread pool), `prof.zig` (`--profile=1`).
- Tests: `src/test/`; end to end, `test/golden/`. Benchmarks: `bench/` (`arena/arena.py`
  for the table above).

## License

LGPL-3.0-or-later · Copyright (C) 2026 AstraLibernis

zarbor is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. See `COPYING.LESSER`, which builds on the GNU General Public License in `COPYING`.

In short: any program, open or closed, may use this library, but the library itself and every change to it stay free.

Versions up to and including commit `155f317` were released under the MIT License; copies obtained under those terms keep them.

Contributions are welcome under the [Developer Certificate of Origin](https://developercertificate.org/): sign off each commit with `git commit -s`. You keep the copyright on your contribution.
