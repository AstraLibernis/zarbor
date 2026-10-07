# F2 — linear leaves

Written before the implementation. Derivation and design only; no measured
claim appears here. Numbers live in `RESULTS.md`, protocol in `PROTOCOL.md`.

## What is wrong today

A leaf emits one number. So a boosted tree is a step function, and a smooth
relationship — price rising with income, steadily, over the whole range — is
approximated by a staircase. Every step costs a split, and splits are the
scarce resource: a depth-6 tree has 64 of them to spend across all features.

It is also the general form of the defect that motivated F1's sibling idea. A
tree cannot represent `x_i - x_j`, or any other combination of two columns,
because a split reads one column at a time. A leaf that fits a line in several
columns at once can represent exactly that, inside the region the splits carved
out, without anyone precomputing the difference.

## The fit

At a leaf holding rows `R`, we currently choose the constant `w` minimising the
second-order approximation

    Σ_{i∈R} [ g_i w + ½ h_i w² ] + ½ λ w²      ⇒   w* = -G / (H + λ)

Let `P` be the numeric features tested on the root-to-leaf path, `x_i` the row's
values on them, and fit an affine function instead:

    f(x) = β₀ + Σ_{j∈P} β_j x_j

Substituting and differentiating gives the normal equations

    (XᵀHX + λI) β = -Xᵀg,        H = diag(h_i),  X = [1, x]

which is a `(|P|+1)`-square symmetric positive-definite system. With
`max_depth = 6` that is at most 7×7, solved by Cholesky at a cost that
disappears next to the histogram pass that produced the leaf.

Setting `β_j = 0` for `j ≠ 0` recovers the constant leaf exactly, so the fit
can only reduce the training objective. Whether it reduces the *test* objective
is the question the measurement answers.

## Three decisions that are not forced by the mathematics

**Which value stands for a bin.** Rows are stored as bin indices, not values.
`linear.zig` faces the same problem and prefers `ds.means` — the mean of the
training values in each bin — falling back to the midpoint of the bin's edges.
Means are unavailable here: at prediction time `ds.means` holds the *scoring*
file's within-bin means, which are not the training ones, so a model would
score differently depending on what else was in the file with it. Midpoints
come from `Schema.edges`, which the model carries, so they are identical at fit
and at score time. Midpoints it is, and the bias that costs `linear.zig` 0.0003
AUC is the price.

**Conditioning.** A path can hold a household count near 1 and an income near
10⁵. Left raw, `XᵀHX` has a condition number in the billions and the Cholesky
is meaningless in f32. Each feature is therefore centred on its own within-leaf
mean before the solve, which makes the system's off-diagonals covariances and
the intercept the leaf's weighted mean of `-g/h`. The centre is stored with the
coefficient rather than folded into the intercept, because folding it back
subtracts two large nearly-equal numbers.

**What happens when the solve fails.** A leaf with fewer rows than path
features, or a singular system after centring (a feature constant within the
leaf, which is common near the bottom of a tree), falls back to the constant
weight. This is a fallback, not an error: the constant leaf is the `β_j = 0`
solution and is always admissible.

## Cost, stated in advance

Two costs, both real:

* **Fitting.** `O(|R| · |P|²)` to accumulate the system per leaf. Summed over a
  level that is `O(n · |P|²)`, against `O(n · |F|)` for the histogram pass. With
  `|P| ≤ 6` and `|F|` in the dozens this should be minor, but it is not free.
* **Applying.** The booster updates raw scores by walking leaf spans and adding
  one constant per span. A linear leaf has no single constant, so that fast
  path cannot be used and every row must traverse the tree instead. That is the
  same path already taken whenever row sampling is on, so the code exists — but
  it is slower, and the measurement reports wall clock alongside accuracy for
  exactly this reason.

## Representation

`Node` gains `lin_ofs: u32` and `n_lin: u8`; `Tree` gains a flat `lin` array of
`{ feature: u32, coef: f32, center: f32 }`. A leaf with `n_lin = 0` is the
constant leaf and costs nothing, which is every leaf under the default.

Format version 3 already exists for F1 and absorbs this too.

## What would make this fail

The honest failure mode is not numerical. It is that boosting already has a
mechanism for adding resolution — more trees — and a linear leaf spends its
extra capacity in a direction the ensemble may already cover. If 500 shallow
trees can trace a smooth curve well enough, fitting a line in each leaf buys
nothing and costs the apply-path speedup. `california` is the test: eight
numeric columns, a smooth target, and nothing categorical to muddy it. If
linear leaves do not help there, they do not help.
