# Plan: what zarbor adds next, and what it never will

**Status: [PLAN] 2026-10-08.** Nothing here is built yet. Each milestone ends with its own
verification, docs and a pushed commit; a milestone is done when its "Done when" list holds,
not when the code compiles.

## Purpose

Fast, verified, hackable tree and linear models for tabular data, as one static binary: for
the owner's own modelling, and as an independent second model in blends. Like the CSV parser,
it is excellent at a defined job and does not try to be everything. Every feature must either
change what a model can learn or help the user improve one; features that only connect zarbor
to other systems are out of scope (below).

## Order

| # | Milestone | Why it matters | Size |
|---|---|---|---|
| 1 | Multiclass | zarbor cannot attempt a 3+-class problem at all today (`MulticlassNotSupported`) | large |
| 2 | Feature importance and SHAP | shows where feature work pays off; features beat tuning by far in past competitions | medium |
| 3 | Sample weights, class weights, starting scores, continued training | rows of unequal importance, imbalanced classes, offsets and stacking; before 4 so every new loss gets them from the start | medium |
| 4 | More losses + more metrics, and choosing the early-stopping metric | training and stopping on what the task is scored by | medium |
| 5 | Monotonic constraints | known-direction effects: better generalisation on small data, required in regulated work | small |
| 6 | Two cheap regularisers: `path_smooth`, `extra_trees` | LightGBM users' tools against overfitting on small or noisy data | small |

Items 3 (class weights onward), 4 (metrics) and 6 were added after a full inventory of the
four reference libraries on 2026-10-08: weights and offsets were in every library's top five
missing items for results.

Size is the number of subsystems touched, not a time estimate. Durations are not given until
a milestone's first stage has been measured.

Shared rules for every milestone, from how zarbor is already kept honest
(docs/correctness.md section 5):

- **Exact parity first.** Where a reference computes the same thing, extend
  `bench/parity/parity.py` (synthetic data, every value its own bin) until predictions agree
  row for row within 1e-6. Where randomness differs, compare hold-out results over seeds.
- **Settings at their extremes.** Every new knob is tested at its minimum and maximum with the
  mechanism asserted, and every new test is mutation-checked (break the code, watch it fail).
- **Threads.** Anything parallel is tested at 1, 3 and 16 threads, bit for bit.
- **Old models keep working.** A model-format bump keeps every older version loadable.
- **Docs.** guide.md gains the settings; correctness.md gains the parity rows; the arena gains a
  row where a reference exists.
- **Task setting or tuning knob.** Every new setting is classified in guide.md as one or the
  other. A task setting (the loss, weights, constraints, offsets) is chosen from the problem and
  never searched. A tuning knob gets a default that is safe without tuning and, if it moves
  results, a range in `tune`'s default search space.

---

## 1. Multiclass

**Goal.** `--objective=softmax` for every algorithm: one probability per class, classes taken
from the label (strings or integers), as now for binary.

**Design.**
- *Boosting (depthwise, lossguide):* K trees per round, one per class, each fitted to that
  class's softmax gradient `p_k - y_k` and hessian `p_k (1 - p_k)` (times 2 in XGBoost's form;
  check which each reference uses). This is what XGBoost `multi:softprob` and LightGBM
  `multiclass` do. Histograms and splits are unchanged: the builder already grows a tree from
  any gradient pair.
- *Symmetric (CatBoost):* CatBoost's `MultiClass` grows ONE tree per round with a vector of K
  values per leaf, scoring splits over all classes together. That needs vector leaves and a
  multi-dimensional cosine score: a separate stage (1c), after 1a/1b are proven.
- *Linear:* multinomial logistic regression (softmax over K weight vectors), the model
  scikit-learn's `LogisticRegression` fits by default for K > 2. L-BFGS already handles the
  binary case; the parameter vector grows K-fold.
- *Random forest:* K-output regression trees (`g = -onehot(y)`, leaves average the one-hot
  vector) give class frequencies per leaf, which is what scikit-learn's forest averages.
  Like the symmetric case this wants vector leaves, so it comes with 1c.
- *Labels:* `label_encoder.zig` stops refusing K > 2 and stores all classes (sorted, as now).
- *Model format v6:* each tree records its class; a vector-leaf tree records K values per leaf.
- *Metrics:* multi-class log loss and accuracy; early stopping on log loss.
- *Output:* `predict` writes K probability columns named by class (and optionally the argmax).

**Stages.** 1a boosting + labels + format + metrics; 1b linear; 1c vector leaves (symmetric
trees, then the forest).

**Progress.** 1a done 2026-10-08: depthwise and lossguide boosting, labels (text and integer),
model format 6, multiclass log loss and accuracy, `predict`/`cv`/`tune`, stratified folds;
exact parity with XGBoost and LightGBM. 1b done 2026-10-08: multinomial logistic regression
(both solvers, L1 and L2), objective equal to scikit-learn's at the optimum. Refused for now
with softmax: GOSS, linear leaves (their multiclass rules are unchecked), `scale_pos_weight`,
symmetric trees and the forest.

**Done when.**
- XGBoost `multi:softprob` and LightGBM `multiclass`: exact parity on a synthetic 4-class set.
- scikit-learn multinomial logistic: objective equal at the optimum, as for binary.
- CatBoost `MultiClass` (1c): exact parity with sampling off; scikit-learn forest: hold-out
  log loss in the same family as today's binary forest comparison.
- A binary file trained before v6 still loads and predicts byte-identically.
- Arena: a multiclass dataset joins the benchmark (one with no licensing problem).

## 2. Feature importance and SHAP

**Goal.** `zarbor explain` reports, per feature, how much the model uses it, and per row, how
much each feature moved that row's prediction.

**Design.**
- *Gain and split count:* summed over every split node. Needs each node's gain stored at
  train time (the builder computes it and throws it away).
- *SHAP:* TreeSHAP (Lundberg et al., 2018), exact and polynomial-time for tree ensembles. It
  needs each node's *cover* (training hessian or row count reaching it), which the model file
  does not store today. Same format bump as gain (v7, or shared with v6 if 1a has not shipped).
  Linear models: SHAP is `coef * (x - mean)` exactly, on the binned design.
- *Symmetric trees:* expanded to the same node form, so the same TreeSHAP applies.
- *Output:* a ranked importance table on stdout; `--shap=FILE` writes one column per feature
  plus the bias, which sum to the raw prediction.

**Done when.**
- Every row's SHAP values sum to its raw prediction (to f32 rounding): a property test.
- XGBoost `pred_contribs=True` on the parity trees: equal within 1e-6, because zarbor grows
  the same trees. LightGBM `pred_contrib=True` likewise. CatBoost `ShapValues` on symmetric
  trees likewise.
- Gain importance equals XGBoost's `total_gain` on the parity trees.

## 3. Sample weights, class weights, starting scores, continued training

**Goal.** `--weight-col=NAME`: a column giving each row's weight (dropped as a feature, like
`--split-col`). `--class_weight=balanced` (or explicit per-class weights). `--init-col=NAME`:
each row's starting raw score (XGBoost `base_margin`, LightGBM `init_score`, CatBoost
`baseline`). `--init-model=M.zm`: keep boosting from a saved model.

**Design.** A weight multiplies the row's gradient and hessian, which is how XGBoost, LightGBM
and CatBoost apply it, so splits, leaf values and `min_child_weight` follow automatically.
Separately: the base score becomes a weighted mean; metrics (AUC, log loss, RMSE) become
weighted; the linear loss is weighted; the forest's bootstrap draws by weight as
scikit-learn's `sample_weight` does; GOSS and MVS sample on weighted gradients (check each
reference's exact rule). Open question to settle by reading CatBoost's source, not guessing:
whether its target statistics (CTRs) count rows by weight.

*Class weights* are sample weights computed from the label (balanced: `n / (K * n_k)`, as
scikit-learn defines it), so they cost nothing once weights exist. *Starting scores* replace
the base score row by row: the trees fit what is left, which is how an exposure offset
(`log(exposure)` under Poisson) or stacking on another model's output works. *Continued
training* loads a model, recomputes its raw scores on the training rows, and adds trees; the
binning schema must come from the loaded model so old and new trees agree.

**Done when.** Exact parity with weights for XGBoost, LightGBM (including a weighted GOSS
case) and CatBoost; `base_margin` / `init_score` parity with XGBoost and LightGBM; a model
trained 100 rounds then continued 100 equals a 200-round model where the references agree
that it should (no sampling); scikit-learn's weighted logistic and forest in family; all weights 1 gives
byte-identical models to no weight column; weight 0 is identical to dropping the row
(a property test).

## 4. More losses, and the early-stopping metric

**Goal.** Losses for the common regression cases, and early stopping on the metric the task
is scored by.

**Design.**
- `absolute_error` (MAE): gradient is the sign of the residual; leaves are re-fitted to the
  median residual of their rows (XGBoost's `reg:absoluteerror` line search). Without the
  re-fit, results are poor; it is part of the loss, not an option.
- `huber` (`--huber_delta`): XGBoost's `reg:pseudohubererror`.
- `quantile` (`--alpha_quantile`): pinball loss with per-leaf quantile re-fit, XGBoost's
  `reg:quantileerror`; predicts a chosen percentile, so two runs give a prediction interval.
- `poisson`: log link for counts (XGBoost `count:poisson`, including its default
  `max_delta_step` of 0.7, which matters).
- `--eval_metric=NAME` for early stopping and reporting: auc, logloss, rmse, mae, quantile
  loss, poisson deviance, multiclass log loss, accuracy, PR-AUC (average precision), error
  rate at a threshold, R². Adding a metric is one function in
  `metric.zig`: the "custom metric" story is that zarbor is meant to be edited.
- Linear: absolute error and quantile have no smooth optimum, so the linear model takes only
  the smooth losses (huber, poisson); the others are refused with a clear error.

**Done when.** Exact parity with XGBoost for each loss on synthetic data (and LightGBM's
equivalents where they compute the same thing); each metric checked against scikit-learn's;
early stopping on each metric stops where XGBoost's does.

## 5. Monotonic constraints

**Goal.** `--monotone=age:+1,price:-1`: predictions never decrease (or increase) as that
feature grows, all else equal.

**Design.** XGBoost's method, which LightGBM's default ("basic") matches: a split on a
constrained feature is rejected if its children's values are out of order, and each child
inherits bounds that keep the whole subtree consistent. Depthwise and lossguide trees only;
numeric features only (a constraint on a categorical is refused). Symmetric trees: CatBoost
has its own method; out of scope until someone needs it.

**Done when.** Exact parity with XGBoost `monotone_constraints` on synthetic data; a property
test sweeps each constrained feature across its range for many rows and finds no violation;
the constraint at both signs and on several features at once.

## 6. Two cheap regularisers

**Goal.** LightGBM's `path_smooth` and `extra_trees`.

**Design.**
- `path_smooth`: a leaf's value is shrunk toward its parent's, more strongly when the leaf has
  few rows (LightGBM's formula, `serial_tree_learner.cpp`/`feature_histogram.hpp`). It also
  changes split gains, so it is in the split scorer, not only the leaf output.
- `extra_trees`: each feature tries one random threshold per node instead of the best one,
  from a seeded stream that does not depend on threads.
- Both are tuning knobs: off by default, and in `tune`'s default space.

**Done when.** Exact parity with LightGBM for `path_smooth`; `extra_trees` in family with
LightGBM's (hold-out over seeds), plus a test that the chosen threshold is random but
reproducible at 1, 3 and 16 threads.

---

## Out of scope: use a big library for these

| Feature | Why not | Use instead |
|---|---|---|
| GPU training | zarbor already matches or beats the references on CPU at the sizes it targets | XGBoost/CatBoost GPU |
| Distributed or out-of-memory training | data that does not fit one machine is not zarbor's job | XGBoost/LightGBM distributed, Spark |
| Sparse input, LightGBM's feature bundling | speed and memory only, for very wide mostly-zero data | LightGBM |
| Python/R/scikit-learn bindings | integration, not results; the CLI's CSV in, CSV out covers the owner's use | the reference libraries |
| ONNX/PMML/other export | deployment into other systems | the reference libraries |
| Ranking objectives | a different problem shape (queries, groups); add only if a task needs it | LightGBM `lambdarank`, XGBoost `rank:*` |
| DART | small, occasional gain at a large speed cost | XGBoost/LightGBM `dart` |
| Tweedie and other specialist losses | added only when a task needs one; each is one objective in the same machinery | XGBoost |
