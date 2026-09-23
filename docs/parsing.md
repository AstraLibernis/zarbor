# Reading a file, and saying what is in it

`src/csv.zig` is the parser. It is deliberately not part of any model: every
algorithm in this library pays it before it sees a bin, so it has its own
module, its own tests and its own timings. Two things live here that used to
live in a throwaway script per dataset.

## `NA` means two different things, and the column says which

The rule, in one line: **a missing marker only means "missing" in a column
whose other values are numbers.**

The recognised markers are `NA`, `N/A`, `#N/A`, `NaN`, `null`, `none`, `nil`
and `?`, case-insensitively, plus the empty field. They are the ones R,
pandas, Excel and SQL exports actually emit.

The Ames housing data is the clean example of why the rule is per column
rather than global:

| column | values | what `NA` is |
|---|---|---|
| `LotFrontage` | `65`, `80`, `NA`, … | a frontage nobody recorded |
| `PoolQC` | `Ex`, `Gd`, `TA`, `NA` | the documented level **"No Pool"** |

Both readings are defensible in isolation and both are wrong applied
everywhere:

- **Global "`NA` is missing"** — what `pandas.read_csv` does by default —
  converts 14 of that dataset's columns from a documented category to
  "unknown". The file's own data dictionary defines `NA` as `No Pool`,
  `No alley access`, `No Basement`, `No Fence`. That information is thrown
  away before a model can see it.
- **Global "`NA` is a level"** — what this parser did until 2026-09-23 — makes
  `MasVnrArea` a 328-level categorical and the file will not load at all:
  `error: CategoricalTooWide`. The Kaggle House Prices training file could not
  be read by this library.

The per-column rule gets both right and needs no configuration. It costs one
extra comparison in the sniff pass, which only looks at the first 1000 rows.

### What it is worth, measured

Less than expected, and the reason is worth knowing. Across 20 fold seeds on
House Prices, 5-fold CV, paired on the fold assignment so both readings see the
same splits:

| reading | RMSE on `log(SalePrice)` |
|---|---|
| `NA` kept as a level where the column is categorical | 0.13511 |
| `NA` blanked everywhere (the pandas convention) | 0.13488 |

Paired difference **-0.00023 +/- 0.00020**, p = 0.27. No effect.

The first explanation offered for that was wrong, and the way it was wrong is
worth keeping. It said: a histogram tree reserves bin 0 for missing and learns
a default direction for it at every split, so a hole already *is* a level to a
tree -- and therefore a model without that mechanism, like the linear one,
should show a difference. Run across all three algorithms, it does not:

| algo | NA kept | NA blanked | diff | p |
|---|---|---|---|---|
| `gbdt` | 0.13511 | 0.13488 | -0.00023 | 0.27 |
| `linear` | 0.15213 | 0.15209 | -0.00004 | **0.86** |
| `random_forest` | 0.14022 | 0.14009 | -0.00012 | 0.28 |

`linear` is the *flattest* of the three. The premise was wrong: it does not
lack the mechanism, because **nothing here trains on anything but the binned
matrix.** `buildDesign` emits one indicator per categorical bin from `1..nb`,
skipping bin 0 -- so a missing categorical is the all-zero row, which is
exactly the reference level that dummy coding drops anyway. An `NA` level gets
its own indicator; an `NA` hole is the baseline. With an intercept those carry
the same information.

So the real statement is broader than the tree one: in this library a missing
entry is already a distinguishable state in every representation any model
sees. Calling that state "the level NA" or "the hole" renames it and adds
nothing.

The change is therefore a **loading fix, not an accuracy fix**. What it bought
is that the file opens.

## `zgbdt profile <data.csv>`

    rows    1460
    cols    80

    column                  kind    missing  miss% distinct          min       median          max  far out
    LotFrontage             num         259  17.7%      110      21.0000      69.0000     313.0000       12
    LotArea                 num           0   0.0%     1073    1300.0000    9478.5000  215245.0000       34
    Neighborhood            cat           0   0.0%       25            -            -            -        -

Every train and `cv` run prints the one-line version of this next to its read
timing, because it is what catches a file read wrong -- a column silently
all-NaN, a categorical that was meant to be numeric -- before a bad score gets
blamed on the model.

Three things the footer reports rather than the table:

- **Unparsed values.** A field that was neither a number nor a recognised
  marker, in a column the sniff called numeric. The sniff reads the first 1000
  rows, so a nonzero count usually means the column changes character further
  down. These used to become `NaN` silently.
- **Columns that cannot inform a split** — every value missing, or one
  distinct value.
- **Categorical columns too wide for a `u8` bin**, named before training
  fails rather than after.

### `far out` is a count, not a verdict

It counts values past the Tukey **outer** fence, `q1 - 3*IQR` and
`q3 + 3*IQR`. The outer fence rather than the usual `1.5*IQR` because a skewed
column -- sale price, lot area, income -- legitimately puts a tenth of its rows
outside the inner one, and a flag that fires on a tenth of the data is not a
flag.

It is reported and not acted on. On house prices the tail is the data.

## Where the parser and the model meet

Two places they were deciding the same thing separately, found by testing the
competition's own situation rather than by reading the code.

### The schema outranks the file

`predict` re-sniffed column kinds from the file it was given, even though the
model had stored them at training time. A column's kind is inferred from
evidence, and a slice of a file can carry no usable evidence about a column
while being perfectly valid data.

The case that found it: a 298-row holdout of House Prices in which every
`PoolQC` happened to be `NA`. A column of nothing but missing markers is
numeric-compatible, so it sniffed numeric where training had it categorical,
and the prediction died with `FeatureKindMismatch`. Nothing was wrong with the
data. Note that this became possible only once markers were recognised at all
-- before, `NA` was an ordinary level and the column stayed categorical by
accident.

`readCsvHinted` takes the model's `names`/`kinds` and pins any column it
names; the rest are sniffed as usual, because a prediction file may carry
extras the model never saw and dropping those is the caller's job.

### A default direction learned from nothing

Bin 0 holds missing, and `bestSplit` scans each feature twice -- once sending
missing left, once right -- so the direction is learned per split. Where a
feature has **no** missing rows at a node, both scans score identically, so
the second was recomputing the first and the winner fell out of `consider`
keeping the first strictly-better candidate. `missing_left = true` won every
time, by loop order.

That flag is not inert: it is what routes a missing value at *prediction*
time, and a column can be complete in training and have holes later. On House
Prices fifteen columns are missing in test and never in train, so fifteen
default directions were set by loop order.

With no evidence the defensible choice is the larger child -- the side holding
more of the node's distribution, so the smaller bet. Measured by training on
80% of House Prices, blanking cells in the holdout only, and in only those
columns complete in the training half (60 seeds, paired on both the split and
the model):

| share of cells blanked | old | new | diff | new wins | p |
|---|---|---|---|---|---|
| 0 (the holdout's own, ~0.00002) | 0.13579 | 0.13580 | +0.00001 | 18/60 | 0.33 |
| **0.00025 (the competition's own rate)** | 0.13596 | **0.13580** | **-0.00016** | 33/60 | **0.005** |
| 0.01 | 0.14201 | 0.13746 | -0.00455 | 59/60 | <0.001 |
| 0.05 | 0.16722 | 0.14381 | -0.02341 | 60/60 | <0.001 |
| 0.20 | 0.29577 | 0.17470 | -0.12107 | 60/60 | <0.001 |

Read the second row against the first: at the competition's rate the new build
scores **0.13580, the same as with no injected missingness at all**, while the
old one has already drifted to 0.13596. The arbitrary direction starts costing
immediately; the principled one does not. By 20% blanked the gap is 0.296
against 0.175, which is the difference between a broken model and a working
one.

The win count is lower than the p-value suggests because at 0.00025 most seeds
blank nothing at all and tie; 33 seeds improve, and essentially none regress.

The gains are unchanged, so the chosen threshold is bit-identical; only the
flag nothing could inform is decided differently. `california.csv` has no
missing values at all and scores 0.449542 on both builds, which is the exact
control.

It is also **faster**, because the redundant second scan is skipped: 12.3% on
california, 10.1% on House Prices, 5.5% on adult, at identical or
near-identical scores.
