# Should the bin type be widened? — cost prototype

`RESULTS.md` found that optimal categorical splits help, but only on columns
zarbor can represent, and the `u8` bin refuses anything with 256 or more
levels. This measures both sides of lifting that ceiling **before** writing it:
what the wider column would buy, and what it would cost.

Neither number is an estimate. Reproduce the cost with
`zig build bench-widecat -Doptimize=ReleaseFast`.

## What it would buy

Measured with LightGBM, which already supports high-cardinality categorical
splits, on the housing frame. Grouped 5-fold by ZIP, five genuinely different
fold draws (`GroupKFold` is unseeded, so repeating it with different `seed`
values measures nothing — the first attempt at this reported sd 0.0000 for
exactly that reason).

| model sees | RMSE | vs State-only | \|d\|/sd |
|---|---:|---:|---:|
| State (51) — zarbor's current ceiling | 13.785 ±0.092 | — | — |
| + Zip3 (753) | 13.124 | −0.661 | 8.8 |
| + Zip3 + Metro (651) | **12.787** | **−0.998** | 17.1 |
| + City (3,114) as well | 12.810 | −0.976 | 14.1 |

Two things follow. The gain is large — larger than any other single change
measured in this project. And **City buys nothing over Zip3 + Metro**, so the
ceiling that matters is about 1,000 levels, not 65,000.

The cheap alternative was checked first and is not viable: keeping the 254 most
common levels and pooling the rest puts 58% of housing rows into one bin for
City and 28% for Zip3, and that pooled tail carries a target sd of 27.7 against
28.4 overall. It discards the information rather than compressing it.

## What it would cost

Split in two, because only one half can be measured with today's code.

**A. Everything except the scatter.** `Bank` already takes a per-feature bin
count as a `u16`, so a feature can be *declared* 753 bins wide today. Rows only
land in bins 0..255, but the per-node clear, the reduce and the split search
all walk the full declared width — exactly what they would do after the change.

    rows=6,830 (the housing frame), 200 trees, depth 6
    levels    ordinal  vs 51     optimal  vs 51   slot KiB
        51    1626 ms  1.00x     1612 ms  1.00x       25.6
       255    1963 ms  1.21x     1755 ms  1.09x       32.0
       753    3616 ms  2.22x     3917 ms  2.43x       47.6
      1024    3994 ms  2.46x     4028 ms  2.50x       56.0
      3114    4293 ms  2.64x     4007 ms  2.49x      121.4

    rows=500,000, 20 trees, depth 6
        51    1988 ms  1.00x     1912 ms  1.00x       25.6
       753    2197 ms  1.10x     2294 ms  1.20x       47.6
      3114    1809 ms  0.91x     2021 ms  1.06x      121.4

**B. The scatter alone**, over bins that really do span the full width — the
one term half A cannot see, since its rows crowd into the first 256 cells.

    levels    ns/row   vs 51   hist KiB
        51     0.375   1.00x        1.6
       753     0.369   0.98x       23.5
      3114     0.389   1.04x       97.3
     16384     0.520   1.39x      512.0

## Reading

**The scatter is not the problem.** Accumulation is flat to 3,114 levels; a
97 KiB private histogram still lives in L2. It only degrades at 16k levels,
far past anything needed here.

**The fixed per-node cost is the problem, and only on small data.** At 500k
rows a 753-level column is 1.1–1.2x. At 6,830 rows it is **2.2–2.4x**, because
the clear/reduce/search is paid per node whatever the node's size, and half a
depth-6 tree's nodes are its last level. This is the same effect the packed
per-feature offsets in `hist.zig` were introduced to fight, reappearing at a
larger scale.

**It is affordable anyway.** A housing CV run goes from 1.2 s to roughly 2.7 s,
for 1.0 RMSE. The ratio looks bad; the absolute number does not.

**And there is obvious headroom.** At the bottom of a depth-6 tree over 6,830
rows a node holds ~100 rows, so a 753-bin column has at most 100 non-empty
bins and the other 650 are cleared, reduced and scanned for nothing. Tracking
the occupied range, or scanning only bins with a non-zero count, would remove
most of this. Not attempted — recorded so the 2.4x is understood as a first
implementation rather than a floor.

## Recommendation

Proceed, with the bin type made a comptime parameter and both `u8` and `u16`
instantiated. Files that fit in `u8` keep today's path bit-for-bit, so the
2.4x is paid only by the tables that asked for it.

The categorical split must stop being a 256-bit mask at the same time: at
`u16` that would be 8 KiB per split node. It becomes a sorted list of the
≤ `max_cat_threshold` level ids on the left, which is smaller than the mask at
any cardinality.

The protocol for the change should add a wall-clock budget to the accuracy
criterion, since this is the first feature here whose cost is a real part of
the decision rather than a footnote.

---

# Outcome of the change

`PROTOCOL-widecat.md` criterion 2 asked for no more than 5% wall clock against
the binary as it stood at `6167158`. Measured min-of-5 to keep a noisy box out
of it:

| dataset | 6167158 | after F1+F2 | after widening | F1+F2 cost | widening cost |
|---|---:|---:|---:|---:|---:|
| california | 1658 ms | 1721 ms | 1768 ms | 1.04x | **1.03x** |
| adult | 1014 ms | 1118 ms | 1137 ms | 1.10x | **1.02x** |
| bank | 1197 ms | 1316 ms | 1328 ms | 1.10x | **1.01x** |
| ames | 832 ms | 824 ms | 854 ms | 0.99x | **1.04x** |

Read against the criterion as written, it fails: cumulative cost is 1.03–1.12x.
Read against what the criterion was *for* — does widening the bin index tax
every workload — it passes. The widening's own contribution is 1.01–1.04x. The
10% on adult and bank arrived with F1 and F2 and was simply never measured,
because those features were judged on accuracy with wall clock as a footnote.

## Three wrong guesses, recorded because they were expensive

The first version of half C said a `u16` index was free, so the plan dropped
the comptime-generic bin type. Then the change landed at 1.35x on adult, and:

1. **Struct growth.** Shrinking `Split` from 224 to 160 bytes and `Work`'s path
   array from 16 to 8 recovered 2–5%. Not it.
2. **Scattered rows.** Half C was rewritten to walk a sparse subset rather than
   every row, on the theory that the root flattered the wide index. Still
   0.96–1.01x. Not it either.
3. **The column-major read.** Profiling put it in `partition`, 51 → 110 ms, and
   that reads column-major — so the column store was split into a byte mirror
   with per-feature wide overrides. Partition stayed at 112 ms.

What it actually was: `goesLeft` took `Split` **by value**, and F1 had grown
that struct to 160 bytes by giving a categorical split an inline id array. The
partition's row loop was copying it per row. Lifting the four scalars the
decision needs into `SplitTest` took partition to 61 ms.

So the bin width was never the cost. The lesson worth keeping is narrower than
"profile first": all three guesses were about *memory the change obviously
touched*, and the answer was a struct passed by value in a loop that the change
did not touch at all. The profiler said `partition` on the first reading and it
took two more experiments to believe which part of `partition`.

The byte mirror and the per-feature wide overrides were kept regardless. They
are not load-bearing for the measured result, but a genuinely wide column does
pay the element size there, and the split costs nothing when no column is wide.
