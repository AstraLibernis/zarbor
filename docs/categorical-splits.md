# F1 — optimal categorical splits

Written before the implementation. Derivation and design only; no measured
claim appears here. Numbers live in `RESULTS.md`, protocol in `PROTOCOL.md`.

## What is wrong today

`data.zig` assigns a categorical level its dictionary id **in order of first
appearance in the file**. `split.zig:bestSplit` then treats that id like any
other bin index and searches cuts of the form `bin <= threshold`.

So the partitions reachable for a 41-level column are the 40 prefixes of an
ordering decided by which row happened to come first. The set of partitions a
categorical column *should* offer is all 2^40 of them. The tree gets 40
arbitrary ones.

A depth-6 tree can recover some of this by splitting the same column
repeatedly — each cut peels off one contiguous run of ids — but it spends a
level of depth for every peel, and depth is the scarce resource.

## The result that makes the search cheap

For one node, write each category `k`'s accumulated gradient and hessian as
`(G_k, H_k)`. Splitting the categories into sets `L` and `R` gives, under the
usual second-order approximation with L2 penalty `λ`,

    gain(L) = ½ [ (Σ_{k∈L} G_k)² / (Σ_{k∈L} H_k + λ)
                + (Σ_{k∈R} G_k)² / (Σ_{k∈R} H_k + λ)
                - G² / (H + λ) ]

**Claim (Fisher 1958).** Order the categories by the ratio `G_k / H_k`. The
`L` maximising `gain` is a prefix of that order.

*Sketch.* Only the first two terms vary with `L`, and `G_R = G − G_L`,
`H_R = H − H_L`, so the objective is a function of the pair `(G_L, H_L)`
alone. Suppose an optimal `L` contains `b` but not `a`, with
`G_a/H_a > G_b/H_b`. Swapping them moves `(G_L, H_L)` by
`(G_a − G_b, H_a − H_b)`. Writing the objective's directional derivative
along that move, its sign is that of
`(G_L/(H_L+λ) − G_R/(H_R+λ)) · (G_a/H_a − G_b/H_b)` to first order; the
second factor is positive by assumption, so one of the two swap directions
strictly increases the objective unless the first factor is zero, in which
case the split is already at an interior optimum where the two children share
a leaf value and the gain is zero. Hence no optimal non-trivial `L` inverts
the ratio order. ∎

This turns an exhaustive search over `2^K` subsets into a sort plus a linear
prefix scan: `O(K log K)`.

The same argument is why the numeric path is correct as it stands. A numeric
column is already sorted by value, and for a monotone relationship that
ordering coincides with the gradient ordering. A categorical column has no
such luck.

## Where the cheapness is a lie, and what to do about it

The prefix scan finds the partition that maximises gain *on the training rows
in this node*. For a category with three rows, `G_k/H_k` is an estimate from
three observations, and the sort will happily place it at one extreme. The
optimum is then real and useless: it isolates noise.

Four guards, all of them standard, all fixed before measuring:

| guard | value | what it stops |
|---|---|---|
| `cat_smooth` | 10.0 | the sort key is `G_k / (H_k + cat_smooth)`, so a category with little mass is pulled toward the middle of the order instead of the ends |
| `cat_l2` | 10.0 | extra L2 in the gain for categorical splits only, since a `K`-way choice overfits harder than a binary threshold at equal sample size |
| `cat_min_group` | `min_child_samples` | categories with fewer rows than the tree's own child-size floor are excluded from the scan and join the right child |
| `max_cat_threshold` | 32 | cap on how many categories may land on the left, so a split cannot enumerate a large set one rare level at a time |

`cat_smooth`, `cat_l2` and `max_cat_threshold` are LightGBM's defaults.
`cat_min_group` is coupled to `min_child_samples` rather than given LightGBM's
independent 100, because a second, larger, unrelated sample floor in the same
tree is a magic number, and because at `min_child_samples = 20` a flat 100
would disable the feature outright on the smaller tables.

Note that these guards are the feature's real risk. If F1 measures as harmful,
the first thing to check is whether it is the search or the regularisation
that failed, and the four constants above are where to look.

## Representation

A numeric split is one `u8` threshold. A categorical split is a subset, so it
needs a bitmask: `max_bin = 256` means four `u64` words, 32 bytes.

**Superseded by `fafa53b`, which replaced the mask with an id list.** The
paragraph below described the design as first written; it is kept because the
reasoning that follows it still applies, but it is not what the code does.

> `Node` grows a `kind` discriminant and a `u32` offset into a per-tree
> `cat_masks` array. Nodes go from 20 to 24 bytes. The mask has a bit per bin
> including bin 0, so *missing joins the left child exactly when its bit is
> set* and needs no separate flag — `missing_left` stays for numeric splits
> only.

As built: there is no `cat_masks` and no bitmask. `Node` carries `cat_ofs`
and `n_cat` into a per-tree `cat_ids` list, and a split tests membership by
binary search over the sorted ids. `Node` is **32 bytes**, not 24.

The missing sentence above is wrong in both halves, and in a way worth being
explicit about because it inverts the behaviour. Bin 0 never enters the
candidate set — the participation loop starts at `b = 1` — and both routing
paths (`tree.zig`'s `predictBinned` and the training-side partition) test
`bin == 0` *before* consulting `cat_ids`, so the id list is never asked about
missing at all. **A categorical split always sends missing right**, with
`missing_left` hard-set to `false`.

That is deliberate and matches LightGBM's `default_left = false`; the
authority is the doc comment on `bestCatSplit` in `split.zig`, which says so
directly. Only this file was left behind.

The model format goes to version 3. Version 2 files load unchanged: they
simply contain no node with `kind = 1`, and the reader gives them an empty
mask array.

## What this does not reach

The bin type is `u8`, and `data.zig` refuses a categorical column with 256 or
more levels outright (`error.CategoricalTooWide`). So F1 improves columns with
2–255 levels and does nothing whatever for the high-cardinality case — ZIP3
with 753 levels, a city name with 3,113 — which remains a target-encoding
problem. Raising that ceiling is a separate change to the bin width and is not
attempted here.
