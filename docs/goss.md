# GOSS: the one parity gap, and what was behind it

`docs/arena.md` recorded one row outside its noise floor. GOSS cost zarbor
0.000050 AUC and cost LightGBM 0.001180 at the same rates — a 24x difference
in damage from the same named feature. Characterised there, not diagnosed.
This is the diagnosis.

## It is GOSS alone, not an interaction

The arena's GOSS row is leafwise growth *plus* optimal categorical splits
*plus* greedy binning *plus* GOSS, so the gap could have belonged to any
combination. It does not:

| zarbor config | no GOSS | with GOSS | GOSS costs |
|---|---:|---:|---:|
| ordinal cats, quantile bins (defaults) | 0.941329 | 0.941246 | −0.000083 |
| optimal cats, greedy bins (LightGBM-matched) | 0.941371 | 0.941321 | −0.000050 |

The cost of GOSS is the same either side of that switch. LightGBM's is
−0.001180. The disagreement is in the sampler, standalone.

## The cause: LightGBM does not rank by the gradient

From `src/boosting/goss.hpp` at v4.7.0:

```cpp
tmp_gradients[i] += std::fabs(gradients[idx] * hessians[idx]);
```

**LightGBM ranks rows by `|g * h|`.** zarbor ranks by `|g|`, which is what
Ke et al. (NeurIPS 2017) specify and what LightGBM's own `top_rate`
documentation describes ("the retain ratio of large *gradient* data").

For logistic loss `h = p(1-p)`, so the hessian factor pulls confidently-wrong
rows *out* of the kept set:

| row | p | y | \|g\| | \|g·h\| |
|---|---|---|---:|---:|
| confidently wrong | 0.9 | 0 | **0.90** | 0.081 |
| uncertain | 0.5 | 1 | 0.50 | **0.125** |

`|g|` keeps the large residual; `|g·h|` keeps the high-curvature row. The two
orderings are inverted. That inversion is the whole gap.

## Evidence

**Swap the key and zarbor becomes LightGBM.** Changing nothing else:

| | AUC | gap to LightGBM | envelope | |
|---|---:|---:|---:|---|
| zarbor, `\|g\|` | 0.941321 | 0.001074 | 0.000757 | **1.42x — differs** |
| zarbor, `\|g·h\|` | 0.940098 | 0.000149 | 0.000757 | **0.20x — agrees** |
| LightGBM | 0.940247 | — | — | |

One line moves the row from outside its noise floor to comfortably inside it.

**A falsifiable prediction, and it holds.** For squared error `h` is the
constant 1, so `|g·h|` *is* `|g|` and the two implementations must stop
disagreeing. On the same 534,932 rows with a continuous target:

| | no GOSS | with GOSS | GOSS costs |
|---|---:|---:|---:|
| zarbor | 27912.92 | 27924.39 | +11.47 |
| LightGBM | 27911.98 | 27928.32 | +16.34 |

Damage ratio 1.42x on regression against 23.6x on classification. The gap
under GOSS is 3.94 RMSE on a scale of 27,924 — 1.4e-4 relative. The
disagreement is specific to objectives where the hessian varies, which is
exactly what the diagnosis predicts and is not something a tuning difference
would produce.

## What was done about it

`--goss_rank` selects the key, defaulting to `gradient`:

    --goss_rank=gradient           |g|      the paper's, and the better score here
    --goss_rank=gradient_hessian   |g*h|    LightGBM's, for parity

The default is not chosen for flattery. `|g|` is what the algorithm's authors
specified and what LightGBM's parameter documentation still describes; it also
measures better on this dataset by 0.0012 AUC. But a flag named after
LightGBM's feature should be able to reproduce LightGBM's feature, so the
other key is one word away — the same shape as `--cat_split=optimal` and
`--bin_policy=greedy`, which exist for the same reason.

This is one dataset. `|g|` scoring better here is not a general claim, and
nothing in the argument above says the hessian factor is wrong — only that it
is undocumented and that it is the difference.

## Other divergences in goss.hpp, and why they are not the answer

Found while reading, and recorded so the next person does not re-derive them:

- **Truncation, not rounding.** `top_k = static_cast<data_size_t>(cnt * top_rate)`
  where zarbor rounds.
- **No sampling for the first `1/learning_rate` iterations.** `if (iter < 1.0f /
  learning_rate) return;` — at `lr = 0.1`, LightGBM trains ten full-data rounds
  before GOSS engages. zarbor samples from round 0.
- **Selection is per block, not global.** `Helper` runs inside
  `bagging_runner_.Run`, so `top_k` is chosen within each block independently
  — a stratification zarbor does not do.
- **Ties all go to the top set.** `grad >= threshold`, which can make the kept
  set larger than `top_k`. zarbor's cut takes exactly `k`, breaking bit-ties by
  row order.
- **Sequential probability sampling** (`prob = rest_need / rest_all`) rather
  than a partial Fisher-Yates. Statistically equivalent.

After matching only the ranking key, the residual is 0.000149 — inside the
0.000757 envelope. So all of the above together are immaterial on this data.
*In combination*: none was isolated individually, and this does not show that
each is separately harmless.
