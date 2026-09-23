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
