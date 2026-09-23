# zarbor against the reference implementation of each model it ships

Protocol pre-registered in `docs/PROTOCOL-arena.md`. Harness:
`bench/arena/arena.py`. Raw output: `bench/arena/results.json`.

Dataset: Kaggle Playground S6E9 (EV purchases), 668,665 x 13, binary, ROC-AUC.
534,932 train / 133,733 valid, one shared stratified split written as two
files. Zero nulls, zero duplicate rows.

Hardware: Ryzen 7 9800X3D, 16 threads, everything pinned to 16.
XGBoost 3.4.1, LightGBM 4.7.0, scikit-learn 1.9.1. n=5, medians.

## The parser, which belongs to no model

Every model pays this before it sees a bin, so it is measured once and on its
own. Both CSVs, 40 MB, 668,665 rows:

| | |
|---|---:|
| zarbor `src/csv.zig` | **135 ms** |
| `pandas.read_csv` | 177 ms |
| | 1.31x |

pandas is read with `dtype=category`, not a plain read followed by
`.astype("category")`. zarbor's parser builds its level dictionaries during
the parse, so pandas has to be allowed to do the same or the work lands in a
different phase on each side. It is also the faster way to call it (136 ms
against 180 ms on the train file alone), so this is not a handicap imposed to
make the comparison come out level.

## Accuracy

`parity` rows are the same algorithm in two implementations, judged against
the reference's own max_bin envelope. `family` rows are deliberately different
constructions and get no envelope.

| | kind | reference | zarbor | reference | gap | envelope | gap/env | |
|---|---|---|---:|---:|---:|---:|---:|---|
| A gbdt depthwise | parity | xgboost | 0.941395 | 0.941441 | 0.000046 | 0.000918 | 0.05 | agrees |
| B gbdt leafwise | parity | lightgbm | 0.941371 | 0.941427 | 0.000056 | 0.001103 | 0.05 | agrees |
| C gbdt + GOSS | parity | lightgbm-goss | 0.941321 | 0.940247 | 0.001074 | 0.000757 | **1.42** | **differs** |
| D random forest | family | sklearn, 1024-leaf cap | 0.939025 | 0.934829 | 0.004197 | — | — | zarbor ahead |
| D random forest | family | sklearn, uncapped default | 0.939025 | 0.933515 | 0.005510 | — | — | zarbor ahead |
| E logistic | family | sklearn lbfgs | 0.937713 | 0.937729 | 0.000016 | — | — | agrees |

**The metric path is clean.** On all six rows zarbor's own score for the
holdout equals the referee's score of the predictions it wrote, where the
referee reloaded the saved `.zm`, re-predicted, and used
`sklearn.metrics.roc_auc_score`. Agreement is to within the 6 decimals zarbor
prints, which is the resolution of the check rather than the size of any
disagreement.

## Speed, per phase

Binning is paid by whoever does it, whenever they do it: XGBoost and LightGBM
bin inside `fit()`, zarbor bins before it. So the comparison is
**prepare + fit + predict**, which contains the binning on both sides. All
figures in ms.

| | read | prepare | fit | predict | **model** | **total** |
|---|---:|---:|---:|---:|---:|---:|
| **A** zarbor | 135 | 44 | 518 | 39 | **601** | **736** |
| xgboost | 177 | 0 | 613 | 38 | 650 | 827 |
| | | | | | *1.08x* | *1.12x* |
| **B** zarbor | 135 | 47 | 492 | 31 | **570** | **705** |
| lightgbm | 177 | 0 | 483 | 49 | 532 | 709 |
| | | | | | *0.93x* | *1.01x* |
| **C** zarbor | 137 | 45 | 582 | 33 | **660** | **797** |
| lightgbm-goss | 177 | 0 | 633 | 52 | 685 | 861 |
| | | | | | *1.04x* | *1.08x* |
| **D** zarbor | 135 | 45 | 5022 | 279 | **5346** | **5481** |
| sklearn capped | 177 | 178 | 6193 | 87 | 6459 | 6635 |
| | | | | | *1.21x* | *1.21x* |
| sklearn uncapped | 177 | 178 | 8955 | 410 | 9543 | 9720 |
| | | | | | *1.79x* | *1.77x* |
| **E** zarbor | 135 | 44 | 158 | 1 | **203** | **338** |
| sklearn logreg | 177 | 224 | 975 | 6 | 1204 | 1381 |
| | | | | | *5.93x* | *4.08x* |

### What the phases say

**B is the row that changed sign.** An earlier version of this document
compared zarbor's bare `fit` against LightGBM's `fit` and reported zarbor
1.06x faster. LightGBM's fit contains its binning and zarbor's does not.
Charged symmetrically, **LightGBM is 1.07x faster than zarbor on model work**
(532 against 570 ms), and the two are level end-to-end only because zarbor's
parser is quicker. Leafwise growth against leafwise growth, LightGBM's
implementation is the better one here.

**zarbor's advantage is in the linear model and the forest, not in boosting.**
Against XGBoost it is 1.08x; against LightGBM it is behind. Against sklearn's
logistic regression it is 5.93x on model work, and against sklearn's forest
1.21x capped, 1.79x uncapped.

**Prediction is where the forest lags.** 279 ms against sklearn's 87 ms with
the leaf cap matched — the one phase where zarbor is beaten more than
threefold. 300 trees of up to 1024 leaves is a large model to walk, and
zarbor's traversal is not doing it as well as sklearn's. Against the uncapped
sklearn forest zarbor wins (279 against 410), but that model is much bigger,
so it is not the same comparison. This is the clearest single lead for
optimisation work in the repo.

**Binning is cheap and steady.** 44-47 ms across every model, against pandas'
178-224 ms for the one-hot and scaling sklearn needs. The prepare column is
where the shared data path pays for itself.

**Nothing is charged for work the other side is not doing.** zarbor trains
with `--valid-frac=0 --verbose_eval=0`, so it runs no per-round validation
prediction and no per-round metric; the reference is fitted with no
`eval_set`. The previous version of this benchmark left zarbor's per-round
validation running and subtracted an estimate afterwards, which is a
correction rather than a measurement.

## C: the one parity row that differs

GOSS costs zarbor 0.000050 AUC at the default rates and costs LightGBM
0.001180. Both implementations do real work — neither is ignoring the flag.

zarbor's is verified at the extremes; LightGBM's is verified applied, and
identically through both the legacy `boosting_type='goss'` and the current
`data_sample_strategy='goss'`. The disagreement closes monotonically as
sampling relaxes:

| top/other | rows kept | zarbor | lightgbm | gap |
|---|---:|---:|---:|---:|
| none | 100% | 0.941371 | 0.941427 | 0.000056 |
| 0.4/0.2 | 60% | 0.941372 | 0.941424 | 0.000052 |
| 0.2/0.1 | 30% | 0.941321 | 0.940247 | 0.001074 |
| 0.1/0.05 | 15% | 0.940180 | 0.937465 | 0.002715 |
| 0.01/0.01 | 2% | 0.930422 | 0.927330 | 0.003092 |

At 60% the gap is 0.000052 — indistinguishable from the no-sampling gap of
0.000056. The two agree on the boosting and disagree only on **how much
accuracy is lost per row discarded**, in proportion to how many are discarded.

Scoring higher is not a pass. The protocol calls an out-of-envelope parity row
a failure regardless of direction, and this one stays a failure until the
mechanism is identified. **Not diagnosed** — that means reading LightGBM's
`goss.hpp`. Suspects, named as suspects: whether the amplification factor
`(1-top_rate)/other_rate` is applied to the hessian as well as the gradient on
both sides, and whether the gradient threshold is an exact top-k (zarbor's is,
pinned by the `gossCut` test in `booster.zig`) or an approximation.

Separately, and not a defect in either: at 30% sampling GOSS is *slower* than
no sampling in both (zarbor 582 against 492 ms in fit, LightGBM 633 against
483). Selecting and amplifying 535k gradients per tree costs more than the
histogram build it saves at this size.

## D: where the forest gap comes from, and where it does not

zarbor's forest beats sklearn's by 0.0042 with the leaf cap matched and 0.0055
with sklearn uncapped. The cap is therefore **not** the source: removing it
makes sklearn worse, so unlimited depth is overfitting rather than being held
back.

That leaves binning as the leading candidate — zarbor splits on 256-bin
quantised features, which bounds where a split can be placed and regularises
in a way sklearn's exact splitter does not. Consistent with the measurement,
not established by it.

## E: the result that says binning is free

zarbor fits the binned, one-hot design matrix built by the same code path the
trees use. sklearn fits raw standardised columns. Different design matrices,
different code, agreeing to **0.000016 AUC** — with zarbor 5.93x faster on
model work.

`linear.zig`'s header argues that binning before a linear fit is "a small loss
of resolution and a real gain in robustness". On this dataset the loss of
resolution is 1.6e-5 AUC. That is the strongest evidence here for the shared
data path: the binning built for the trees costs the linear model essentially
nothing and saves it the 224 ms sklearn spends on one-hot and scaling.

## Summary

- Two parity rows inside their noise floor, one family row agreeing to 1.6e-5,
  one family row ahead of its reference, one parity row (GOSS) outside its
  envelope by 1.42x and characterised but not diagnosed.
- On model work zarbor leads XGBoost by 1.08x, **trails LightGBM by 1.07x**,
  leads sklearn's forest by 1.21x and its logistic regression by 5.93x.
- The parser is 1.31x faster than pandas and is the reason the LightGBM
  comparison is level end-to-end despite the fit being behind.
- Forest prediction, at 279 ms against sklearn's 87, is the largest single
  deficit measured and the clearest place to optimise next.
