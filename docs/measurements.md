# Measurements moved out of code comments

**Status: [MEASURED], unrefined.** On 2026-10-06 every benchmark and measurement number was
taken out of zarbor's code comments (the comments keep the decision and its reason). They are
collected here verbatim, with the commit that introduced each one, to be refined later.

What "refined" should mean for each entry: re-run it on the current code, record the machine,
build mode (ReleaseFast unless stated), dataset, thread count and commit, and either keep it with
that context or drop it. Until then treat every number below as a historical note from its
commit, not as a property of the current code. Several were last rewritten by c43a41f (comment
compression, 2026-09-29); the "Introduced" commit is where the number first appeared.

**Known error (2026-10-06):** `bench/binshape.zig` listed 203 bins for `Daily_Commute_km`;
the EV dataset bins it into 202 (checked with `zarbor info` on a model trained at f477167),
as `bench/layoutbench.zig` always had. So the "559 bins" and "437 of 559" totals quoted below
should read 558 and 436; binshape's `WIDTHS` is corrected.

## bench/barrier.zig

### file doc (cost of one parallel region)
- Original comment: "That looked like an obvious explanation for a scaling gap, so it was measured before anything was restructured: ~0.51 us at 8 threads, against roughly 57,000 submissions in a 200-tree fit, is about 29 ms. Not the answer."
- Introduced: 96a63db (2026-09-21) — `git blame`

## bench/binshape.zig

### file doc (real widths, count field)
- Original comment: "Real tables are not like that: the benchmark dataset has 559 bins across 13 features, and 437 of those belong to just two columns. The whole histogram is 16 KB and L1-resident, where the uniform-256 model makes it 80 KB and L2-resident — a different regime, and the regime the code actually runs in. [...] The question it exists to answer: the count field. Dropping it is worth 1.5x in histbench, but `min_child_samples` needs it."
- Introduced: 2f888ab (2026-09-21) — `git blame`

### `Bin16`
- Original comment: "The count dropped. Two RMWs, and the index is a shift rather than a multiply. This is the 1.5x, and the semantics we cannot afford to lose."
- Introduced: 2f888ab (2026-09-21) — `git blame`

## bench/expbench.zig

### file doc (sigmoid cost)
- Original comment: "Gradients were 15% of a boosting fit and almost all of it was `@exp`. Three candidates: Zig's `@exp` (compiler_rt), glibc's `expf`, and an inlined vector `2^x`. glibc is *not* faster -- scalar transcendental evaluation is the cost, not anyone's implementation -- and the vector form is 11.7x, accurate to under 1e-6 absolute."
- Introduced: 96a63db (2026-09-21) — `git blame`

## bench/histbench.zig

### file doc
- Original comment: "hist_build is 45-63% of training, so this isolates it from tree building entirely"
- Introduced: 793b34a (2026-09-21) — `git blame`

### prefetch variant
- Original comment: "This is the technique xgboost's binary is full of (~4,900 prefetch instructions, and no AVX-512 at all)."
- Introduced: 793b34a (2026-09-21) — `git blame`

### `rowOuterSplitCounts`
- Original comment: "The counts are what force the bin up to 24 bytes today. Split out, the gradient array is 13x272x16 = 56 KB and the count array 14 KB — and the count array is small enough to stay in L1 while the gradients stream."
- Introduced: 793b34a (2026-09-21) — `git blame`. Note: with `strideFor(Bin16)` on a 128-byte line the stride is 264, not 272, so the figures were already off.

## bench/layoutbench.zig

### file doc (uniform stride waste)
- Original comment: "The real dataset's features hold 47, 234, 202, 6, 17, 22, 7, 4, 4, 5, 3, 3 and 4 bins. A uniform stride sized by the widest one allocates 13x240=3120 slots for 558 real bins, so every zeroing, reduction and subtraction does 5.6x the necessary work, and the histogram spills a cache level it need not."
- Introduced: 793b34a (2026-09-21) — `git blame`

## src/bin_edges.zig

### `greedyBins` (quantile vs LightGBM on a 92%-zero column)
- Original comment: "LightGBM's `GreedyFindBin` (v4.7.0, src/io/bin.cpp). Quantile cuts on a 92%-zero column (adult `capital-gain`) land in the mass, dedup, and leave a few bins for the 8% with the signal: 0.6088 AUC vs LightGBM's 0.6224 on that column alone. Here a value with a bin's worth of rows gets its own bin and the budget is recomputed over the rest, repeatedly."
- Introduced: 290a1ac (2026-09-22) — `git log -S'0.6088'` (moved to bin_edges.zig in 18be012, 2026-09-29)

## src/booster.zig

### GossRank
- Original comment: "Magnitude GOSS ranks rows by to keep in full. The one place the GOSS paper and LightGBM's code differ; worth 0.0012 AUC on a binary problem (docs/archive/goss.md)."
- Introduced: 59ce1af (2026-09-23) — from `git log -S`; current wording from c43a41f (2026-09-29)

### sigmoid8
- Original comment: "`sigmoid` for eight rows. Scalar `@exp` is a libm call at 7.8 ns/element (glibc `expf` no faster). Inline `2^(-x*log2e)`, integer part into the exponent field, fraction by degree-5 minimax polynomial, vectorises to 0.67 ns: 11.7x, and gradients were 15% of a fit. Error < 1e-6 absolute (a few f32 ulp), pinned by `\"vectorised sigmoid matches the scalar one\"`. Still an approximation, so it is confined to gradients; predictions use the scalar `sigmoid`."
- Introduced: 96a63db (2026-09-21) — from `git log -S`; current wording from c43a41f (2026-09-29)
- **Re-measured 2026-10-07 [MEASURED]:** the "(a few f32 ulp)" claim was wrong. Against an f64
  sigmoid, over x in [-30, 30] at step 1e-4 (ReleaseSafe, the function copied verbatim from
  a9ee25b): max absolute error 8.11e-7, max relative error 4.92e-6, max 58.3 f32 ulp, the worst at
  x = -25.99 where the value is tiny. Harmless in practice: with it, zarbor's predictions match
  XGBoost's row for row through 500 trees (docs/archive/parity.md). Comment corrected.

### valid_min_chunk (duplicate in src/forest.zig, now a pointer to booster's)
- Original comment: "Validation rows per pool chunk, at least. 4096 cut 133k rows into 33 chunks for 16 workers, so the barrier waited on a worker with three while the average had two; 1024 gives 131. Rows are independent, so the chunking changes nothing but the wait."
- Introduced: 6d1d44f (2026-10-05) — from `git blame` (identical text in both files)

### train, validation scoring guard
- Original comment: "Score only when consumed: early stopping (every round), a log line, or the final round (`best_score`). Unguarded, on 668k rows it was 52% of training: a single-thread 133k-row sort per round, discarded 199 times in 200."
- Introduced: 2312f7e (2026-09-21) — from `git log -S "52%"`; current wording from c43a41f (2026-09-29)

## src/builder.zig

### `Builder.g` (gradients not permuted)
- Original comment: "that paid off when the histogram loop was feature-outer, but row-outer reads each once per row, and permuting tripled partition traffic (12 bytes a row vs 4): gather 4 ms vs moving 100+ ms per 200-tree fit."
- Introduced: c43a41f (2026-09-29) — `git blame` / `git log -S"100+ ms"` (the "tripled" / "4 ms" figures date from 2f888ab, 2026-09-21)

### `Builder.queue_head`
- Original comment: "`orderedRemove(0)` shifted the whole frontier of 256-byte `Work`s on every pop: 1.6 ms of a 17.7 ms 1,024-leaf forest tree."
- Introduced: b48a8f4 (2026-10-05) — `git blame`

### `totalOf` (parallel root sum)
- Original comment: "Only the root needs it (others inherit from the parent's split), but the root is every row: on 668k rows x 200 trees the serial version was 46 ms, 7% of a fit and one of three things holding 8-thread scaling to 4.8x vs xgboost's 5.3x."
- Introduced: 96a63db (2026-09-21) — `git log -S"46 ms"`; current wording from c43a41f (2026-09-29) per `git blame`

### `emitSample` (ascending sampled rows; doc was on `clearCounts`)
- Original comment: "In draw order every row was a cache miss: one histogram over the root's 535k bootstrap rows took 6.14 ms against 2.83 ms for the same rows sorted."
- Introduced: 412ead4 (2026-10-05) — `git blame`

### `expandBatch`
- Original comment: "A forest's lower levels are thousands of nodes of a few hundred rows, each too small to spread over the pool, so one at a time 16 threads ran a tree 2.9x faster than one (52 -> 17.7 ms)."
- Introduced: 3d3003f (2026-10-05) — `git blame`

## src/cli/cv.zig

### freeing the CSV frame before the folds
- Original comment: "Past binning the frame is read for one column, the groups, once per repeat: keep a copy of that and free the rest (~40 MB on 668k x 15) before the folds run."
- Introduced: 14acac7 (2026-10-05) — `git log -S'40 MB on'`

## src/cli/score.zig

### holdout target decoding (inverted-AUC example)
- Original comment: "Decode the holdout's target with the *model's* class order. This file built its own dictionary by first appearance, so reading its raw ids would silently invert the target whenever the two files disagree on which class appears first -- and an inverted AUC of 0.06 still looks like a number, not a bug."
- Introduced: 20c606c (2026-09-21) — `git log -S'AUC of 0.06'` (moved to src/cli/ in 46a3114, 2026-09-29)

## src/cli/train.zig

### build-mode line (Debug slowdown)
- Original comment: "`zig build` defaults to Debug, and a Debug binary is ~8x slower here. Printing the mode means a benchmark can never quietly measure the wrong build -- which is exactly what happened once."
- Introduced: 2f888ab (2026-09-21) — `git log -S'8x slower'` (moved to src/cli/ in 46a3114, 2026-09-29)

### freeing the CSV frame and full matrix before the fit
- Original comment: "The split owns copies of everything it needs, and a save needs only the schema, so the CSV frame and the full matrix are dead from here: freed now, not at exit, they were ~66 MB of the fit's peak on 668k rows."
- Introduced: 14acac7 (2026-10-05) — `git log -S'66 MB'`

### `reportLinearFit` (stall example)
- Original comment: "Silence here used to mean \"fitted\". It did not: with `--lin_standardize=false` on a column reaching 188,000, the coefficient steps fall under `lin_tol` after four iterations while the gradient is still enormous, and the result ranks by that one column and nothing else."
- Introduced: c3ebed5 (2026-09-21) — `git log -S'188,000'` (moved to src/cli/ in 46a3114, 2026-09-29)

## src/csv.zig

### module doc (CSV load speed vs pandas)
- Original comment: "CSV parsing: bytes on disk to a column-major `Frame`. Every model pays this before any bin, so it has its own module, tests and timings; `data.zig` re-exports `Frame`, `ColumnKind`, `readCsv`. 117 ms vs `pandas.read_csv` 179 ms on a 32 MB / 534,932-row file. See docs/archive/arena.md."
- Introduced: 99848ab (2026-09-23) — numbers first appear in `git log -S'117 ms'`; current wording c43a41f (2026-09-29)

### `bytes_per_worker`
- Original comment: "CSV bytes per worker. zsift's rule (`parallel.forEachField`, serial below 2 MiB) suits a near-free sink; this one converts every field. Swept 32–256 KiB vs the all-core loader on 7 real files (0.7–45 MB): only 64 KiB won on every file every round (medians 1.34–1.73×, worst round 1.08×)."
- Introduced: e48666f (2026-09-29) — `git log -S'1.34'`; current wording c43a41f (2026-09-29)

### `allocScratch` (Debug/ReleaseSafe fill cost)
- Original comment: "Unescape scratch is allocated raw, skipping `alloc`'s fill: Debug and ReleaseSafe write 0xAA over every allocation and again on free, and this is up to 16 MiB per worker, 272 MB on 16 threads (ReleaseSafe peaked at 384 MB vs 135 MB in ReleaseFast). Only a field containing `""` writes it."
- Introduced: 793861a (2026-10-05) — `git blame`. Note: "272 MB on 16 threads" was arithmetically wrong (16 x 16 MiB = 256 MiB).

## src/csv_profile.zig

### numeric column sort (radix vs block sort)
- Original comment: "Sorted by radix on (key << 32 | bits), not std.mem.sort on f64: the stable block sort was 171 ms of a 221 ms load on 668k rows x 8 numeric columns. The order is the same element for element: ..."
- Introduced: 312b0a5 (2026-10-05) — `git log -S'171 ms'`

## src/data.zig

### module doc (bins vs f32 footprint)
- Original comment: "CSV ingest and quantisation. Every column becomes bin indices up front so the design stays in cache (668k rows x 13 features: 17 MB as bins, 35 MB as f32). Split finding uses bins; raw values only report thresholds."
- Introduced: 21b5189 (2026-09-21) — `git log -S'35 MB'`; current wording c43a41f (2026-09-29)

### `BinIdx` (u16 vs u8 kernel cost)
- Original comment: "`u16` so categoricals can exceed 255 levels (ZIP3, city, metro). 0.98-1.01x of `u8` in the accumulation kernel (bound by the scattered histogram update), so it costs memory only. See docs/archive/wide-categoricals.md."
- Introduced: fafa53b (2026-09-22) — `git log -S'0.98-1.01x'`

### `Dataset.bins` (partition slowdown attribution)
- Original comment: "The 51 -> 110 ms partition slowdown once blamed on `u16` bins was `goesLeft` taking a 160-byte `Split` by value; the bin width \"was never the cost\" (docs/archive/wide-categoricals.md, \"Three wrong guesses\")."
- Introduced: ee0b3bd (2026-09-22) — `git log -S'110 ms'`; current wording 5b90a97 (2026-10-05, `git blame`)

### `Dataset.bins_rm` (row-major mirror gain and cost)
- Original comment: "Row-major copy: `bins_rm[r * n_features + f]`. Histogram building gains 1.74x from a contiguous row; partition would touch 13x the cache lines here. Cost: 2 bytes per feature per row (13 features: 17.4 MB on 668k rows; the 93 MB peak was measured at 1 byte) and one transpose at load."
- Introduced: 2f888ab (2026-09-21) — `git log -S'1.74x'` / `-S'93 MB'`; 17.4 MB figure 57f5ce6 (2026-09-29)

## src/forest.zig

### PredictCtx.run, trees-outer loop order
- Original comment: "Trees outer, rows inner: a 300 x 1024-leaf forest is ~19.7 MB of nodes, so row-major scattered each row across 300 uncached arrays; this keeps one tree (~65 KB) and the chunk's bins in L2. Bit-exact with row-major: each `out[r]` sums the same trees in the same order, rounding to f32 per step."
- Introduced: 38ea68f (2026-09-23) — from `git log -S`; current wording from c43a41f (2026-09-29)

### train, validation scoring guard
- Original comment: "Score only when consumed: no early stopping, so only the last and logged rounds use it, and the metric is a single-thread sort (133k rows: 1.18 s of a 2.35 s fit when scored every round; the booster has the same fix)."
- Introduced: 96a63db (2026-09-21) — from `git log -S "1.18 s"`; current wording from c43a41f (2026-09-29)

## src/goss.zig

### `gossCut`
- Original comment: "Two 16-bit counting passes over the bit patterns. Quickselect was 768 ms of a 1.57 s GOSS fit (49%): Hoare's scans branch on a coin flip, so nearly every element mispredicted; counting branches predictably and streams."
- Introduced: 2f888ab (2026-09-21) — `git log -S"768 ms"`; current wording from c43a41f (2026-09-29) per `git blame`

### `gossSelect` (parallel mark/emit)
- Original comment: "Marking and the final bitset walk run on the pool: they were 65% of a serial 1.5 ms per round, 35% of a GOSS fit."
- Introduced: fffd7dd (2026-10-05) — `git blame`

### `gossSelect` (bitset walk vs sort)
- Original comment: "Ascending row order via the bitset: the histogram kernel wants a forward walk, and sorting 160k ids would cost more than selection."
- Introduced: 2f888ab (2026-09-21) — `git log -S"160k"`; current wording from c43a41f (2026-09-29) per `git blame`

## src/hist.zig

### `Bin` (4-lane bin shape)
- Original comment: "Vs the 24-byte shape on real bin widths: 1.35x alone; composes with row-major layout (1.74x alone, 2.34x together) at every node size from 2k to 500k rows."
- Introduced: 2f888ab (2026-09-21) — `git log -S"2.34x"`; current wording from c43a41f (2026-09-29) per `git blame`

### `Bank` (packed per-feature offsets)
- Original comment: "widths are uneven (234 and 202 bins vs 3-4 for categoricals), so a stride wasted 5.6x (13x240 = 3120 slots for 558 bins). Clear+reduce is fixed per node, dominating the many small bottom nodes: packing is 1.08x at 500k rows, 3.15x at 8k, 4.09x at 2k; a depth-6 tree has half its internal nodes in the last level."
- Introduced: 793b34a (2026-09-21) — `git log -S"4.09x"`; current wording from c43a41f (2026-09-29) per `git blame`

### `BuildCtx` (row-outer accumulation)
- Original comment: "Row-outer accumulation over row-major bins: 1.74x over feature-outer column-major on real data. [...] An earlier benchmark saw 1.07x: 256 bins per feature makes collisions rare and spills to L2; real tables are lopsided (two columns held 437 of 559 bins, eleven had 3-7)."
- Introduced: 2f888ab (2026-09-21) — `git log -S"1.07x"`; current wording from c43a41f (2026-09-29) per `git blame`

### `BuildCtx.grads` (gradients indexed by row id)
- Original comment: "Indexed by original row id (`grads[rows[i]]`), not permuted: keeping it in step tripled what the partition moved, while this gather costs 4 ms across a 200-tree fit."
- Introduced: 2f888ab (2026-09-21) — `git log -S"tripled"`; current wording from c43a41f (2026-09-29) per `git blame`

### `ReduceCtx` (slot size)
- Original comment: "All features, not just selected ones: a flat branch-free pass over ~16 KB beats the index arithmetic of skipping."
- Introduced: 793b34a (2026-09-21) — `git log -S"16 KB"`; current wording from c43a41f (2026-09-29) per `git blame`

### `parallel_threshold`
- Original comment: "Below this many rows a node is built on one thread. Tuned against per-node clear+reduce cost (packing cut it ~4.5x): 200-tree fit at 8192 (old value) 1516 ms, at 512 1213 ms; flat 256-768, rising below 128 where chunks get too small for the barrier."
- Introduced: 793b34a (2026-09-21) — `git log -S"1516"`; lines last touched by c43a41f (2026-09-29) and a43e846 (2026-10-05) per `git blame`

### `max_parts`
- Original comment: "Swept on ev-purchases at 16 threads: 16 parts matched or beat the per-worker build (gbdt 1.209 -> 1.194 s, lossguide+goss 1.581 -> 1.571, forest 3.592 -> 3.521); 32 and 64 were slower, the extra partials costing more to clear and reduce than balance gained."
- Introduced: a43e846 (2026-10-05) — `git blame`

## src/lin_solve.zig

### Fit.stalled, 1e-3 gradient-fall threshold
- Original comment: "The gradient vanishes at an optimum under any scaling, so its fall is the scale-free arrival test. Measured: healthy runs end 4-5 orders down (1.4e-5), runs stalled by bad conditioning 3 (3e-3); 1e-3 sits between."
- Introduced: c3ebed5 (2026-09-21) — from `git log -S "1.4e-5"`; current wording from c43a41f (2026-09-29)

## src/linear.zig

### buildDesign, numeric bin representative
- Original comment: "Binning's per-bin value means. Without `quantise`, fall back to edge midpoints: biased under skew, 0.0003 AUC worse on the EV set."
- Introduced: 1afb3f7 (2026-09-21) — from `git log -S "0.0003"`; current wording from c43a41f (2026-09-29)

## src/metric.zig

### `auc` (radix vs comparison sort)
- Original comment: "Sorted by LSD radix (`radix.zig`) on `f32Key(score) << 32 | label`, not a comparison sort through an index: 9x faster on 133k rows (10.3 -> 1.2 ms), and it runs every `verbose_eval` round, every round under early stopping."
- Introduced: f044794 (2026-10-05) — `git log -S'10.3 -> 1.2'`

## src/objective.zig

### `Objective.squared_error` (adam epochs to converge)
- Original comment: "Squared error. Hessian is constant 1, so `adam` converges in a few hundred epochs where logistic needs tens of thousands."
- Introduced: 3237ace (2026-09-23) — `git log -S'tens of thousands'` (moved to objective.zig in 0fede05, 2026-09-29)

## src/partition.zig

### `parallel_partition_min`
- Original comment: "Below this many rows the barriers cost more than the scan saves. Tuned: on a 200-tree fit 32768 (the original guess) spends 360 ms in partition, 2048 spends 271 ms; the curve rises monotonically above that. Output identical at every setting."
- Introduced: 793b34a (2026-09-21) — `git log -S"360 ms"`; current wording from c43a41f (2026-09-29) per `git blame`

### `goesLeft` (SplitTest instead of Split)
- Original comment: "Takes the four scalars the decision needs, not the whole 160-byte `Split` (inline categorical id array): read per row per pass, it put partition at 2.2x. Bin element width, blamed twice, was worth nothing."
- Introduced: ee0b3bd (2026-09-22) — `git log -S"2.2x"`; current wording from c43a41f (2026-09-29) per `git blame`

### `partition` (parallel passes)
- Original comment: "Count / prefix-sum / scatter in three parallel passes: done serially, this per-level O(rows) work capped 16 threads at 1.66x over 1 and made time scale with depth, not node count."
- Introduced: 63f873f (2026-09-21) — `git log -S"1.66x"`; current wording from c43a41f (2026-09-29) per `git blame`

### `PartCtx` (narrow column element)
- Original comment: "Whole matrix as `u16`: 51 -> 110 ms on adult, row-major accumulate unmoved."
- Introduced: c43a41f (2026-09-29) — `git blame` / `git log -S"51 ->"`

## src/radix.zig

### module doc (speedup over the replaced comparison sorts)
- Original comment: "LSD radix sort on an f32 key carried in the high half of a u64, for the two places that sort hundreds of thousands of floats: the AUC (every logged round) and the column profile (every numeric column at load). Both comparison sorts it replaced were 7-9x slower."
- Introduced: 312b0a5 (2026-10-05) — `git log -S'7-9x'`

## src/schema.zig

### `levelBins` (per-row level scan cost)
- Original comment: "The bin of each of a new file's level ids: the saved level's position + 1, or 0 (missing) for a level the model never saw. Translating per row scanned the saved levels with a string compare each time, O(rows x levels): 136 ms to bin 668k rows of one 255-level column, 10 ms to parse them. A first match wins, as it did in that scan, should a saved level ever repeat."
- Introduced: 8864d21 (2026-10-05) — `git log -S'136 ms'`

## src/split.zig

### `SplitTest`
- Original comment: "`Split` is 160 bytes (inline categorical id array); reading through it cost partition 2.2x wall clock."
- Introduced: ee0b3bd (2026-09-22) — `git log -S"2.2x"`; current wording from c43a41f (2026-09-29) per `git blame`

### `bestCatSplit` (LightGBM parity)
- Original comment: "docs/archive/vs-lightgbm.md records the cost of six earlier differences (zarbor got 64% of LightGBM's gain on the same columns)."
- Introduced: ae0abf2 (2026-09-22) — `git log -S"64%"`; current wording from c43a41f (2026-09-29) per `git blame`

### `cat_scratch_small` / `bestCatSplitWide`
- Original comment: "Levels the sorted categorical search orders on the stack. `max_cat_levels` defaults to 255, so this covers it; wider columns go through `bestCatSplitWide`. bestSplit used to declare a `max_bins` (65,535 x 16 B = 1 MB) scratch on every call, cat search or not: Debug and ReleaseSafe fill an `undefined` array with 0xAA, and that fill was most of a 12x ReleaseSafe slowdown on a forest; the 1 MB frame also cost a stack probe per call in every mode. [...] The 1 MB scratch, in its own frame so only a categorical search over a wide column pays for it."
- Introduced: 7f72c71 (2026-10-05) — `git blame`

### `bestCatSplit` (insertion sort bound)
- Original comment: "Stable, allocation-free; one element per level present, so at most 255 at the default `max_cat_levels`, where it is far cheaper than the histogram. Quadratic above that."
- Introduced: 57f5ce6 (2026-09-29) — `git blame` (`git log -S"at most 255"` first hit: a1bde0e, 2026-09-22)

### `bestSplit` (missing direction with no evidence)
- Original comment: "a column complete in training often has holes later (Kaggle House Prices: fifteen columns missing only in the test half)."
- Introduced: a62a619 (2026-09-23) — `git log -S"fifteen columns"`; current wording from c43a41f (2026-09-29) per `git blame`

## src/test/booster_test.zig

### "vectorised sigmoid matches the scalar one"
- Original comment: "The gradient loop uses an inlined 2^x rather than libm's expf, which is 11.7x faster and approximate. This pins how approximate: anything worse than a few f32 ulp would start moving split decisions around."
- Introduced: 96a63db (2026-09-21) — from `git log -S "11.7x"`; moved to src/test/ in 3ee0d96 (2026-09-29)

## src/test/model_test.zig

### histogram stride sized by the widest feature
- Original comment: "Real tables are uneven — 234 bins for a continuous column next to 3 for a boolean — so that wasted 5.6x the slots on the dataset this was measured against, and the per-node clear and reduce paid for all of it."
- Introduced: 793b34a (2026-09-21)

### cat_l2 on the parent as well as the children
- Original comment: "adding cat_l2 to the parent score as well reads as the self-consistent choice, and cost 36% of the gain the categorical columns carry until LightGBM's source settled it."
- Introduced: 77b2ede (2026-09-22)

### quantile binning on a mostly-one-value column
- Original comment: "a quantile rule spends its budget inside that mass and leaves the tail a handful of bins. On adult's capital-gain that was 0.0136 AUC against LightGBM, and no test could see it."
- Introduced: 77b2ede (2026-09-22)

### bin midpoint as the linear model's bin value
- Original comment: "On the EV set that bias cost 0.0003 AUC against scikit-learn -- the entire remaining gap once the solver was fixed."
- Introduced: 1afb3f7 (2026-09-21)

## src/test/sweep_test.zig

### GOSS ranking-key test
- Original comment: "LightGBM's goss.hpp ranks by |g * h| where the paper -- and LightGBM's own `top_rate` docs -- say |g|. Worth 0.0012 AUC on the EV set, which is the whole of the one parity gap in docs/archive/arena.md. See docs/archive/goss.md."
- Introduced: 59ce1af (2026-09-23) — `git blame`
