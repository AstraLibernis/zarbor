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
