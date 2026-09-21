//! Regression-tree construction over binned features.
//!
//! Rows belonging to a node are kept contiguous in `rows`. Gradients are
//! *not* permuted alongside them: they stay indexed by original row id and
//! the gradient gather happens once per tree, not once per node per feature,
//! and every histogram pass then reads gradients sequentially.
//!
//! Sibling histograms use the subtraction identity — a node's histogram is its
//! parent's minus its sibling's — so only the cheaper child of each pair is
//! ever accumulated.

const std = @import("std");
const Pool = @import("pool.zig").Pool;
const data = @import("data.zig");
const Dataset = data.Dataset;
const hist = @import("hist.zig");
const config = @import("config.zig");
const prof = @import("prof.zig");

pub const Node = extern struct {
    feature: u32 = 0,
    left: u32 = 0,
    right: u32 = 0,
    weight: f32 = 0,
    threshold: u8 = 0,
    missing_left: bool = true,
    is_leaf: bool = true,
    _pad: u8 = 0,
};

pub const Tree = struct {
    nodes: []Node,

    pub fn deinit(t: *Tree, gpa: std.mem.Allocator) void {
        gpa.free(t.nodes);
        t.* = undefined;
    }

    /// Raw score for one row of a dataset binned with the same edges.
    ///
    /// Reads the row-major matrix: a walk visits a different feature at every
    /// level, so column-major touched one cache line per level of depth,
    /// while a whole row is thirteen contiguous bytes.
    pub fn predictBinned(t: *const Tree, ds: *const Dataset, row: usize) f32 {
        const rb = ds.bins_rm[row * ds.n_features ..][0..ds.n_features];
        var i: u32 = 0;
        while (!t.nodes[i].is_leaf) {
            const n = t.nodes[i];
            const b = rb[n.feature];
            const go_left = if (b == 0) n.missing_left else b <= n.threshold;
            i = if (go_left) n.left else n.right;
        }
        return t.nodes[i].weight;
    }
};

/// A leaf's contiguous row span, so the booster can update raw scores by
/// walking ranges instead of re-traversing the tree for every row.
pub const LeafSpan = struct {
    start: usize,
    end: usize,
    weight: f32,
};

const Work = struct {
    node: u32,
    start: usize,
    end: usize,
    depth: u32,
    total: hist.Bin,
    slot: u32,
    split: hist.Split,
};

/// Hard ceiling on histogram slot memory. Beyond this the caller is asked to
/// reduce tree size rather than have the allocator decide for them.
const slot_memory_budget: usize = 2 << 30;

pub const Builder = struct {
    gpa: std.mem.Allocator,
    pool: *Pool,
    ds: *const Dataset,
    cfg: config.Config,
    bank: hist.Bank,

    /// `n_slots` histograms, each `bank.slotLen()` bins.
    slots: []hist.Bin,
    free_slots: std.ArrayList(u32),

    rows: []u32,
    /// Gradients indexed by original row id, borrowed for the current tree.
    ///
    /// These used to be permuted to match `rows`, so that a node's gradients
    /// were contiguous. That paid for itself when the histogram loop was
    /// feature-outer and read each gradient once *per feature*; row-outer
    /// reads it once per row, and the permutation then costs far more than it
    /// saves -- it tripled what the partition had to move (12 bytes a row
    /// against 4). Measured: the row-id gather costs 4 ms across a 200-tree
    /// fit, the moving cost over 100.
    g: []const hist.GradPair,
    /// Destination for the parallel partition, which cannot be done in place.
    rows_out: []u32,
    /// Per-chunk partial sums for `totalOf`.
    total_partial: []hist.Bin,
    /// Per-chunk left-hand counts, plus one for the total.
    part_counts: []usize,
    part_left: []usize,
    part_right: []usize,
    /// Rows active for the current tree: `rows[0..n_active]`.
    n_active: usize,

    nodes: std.ArrayList(Node),
    queue: std.ArrayList(Work),
    leaves: std.ArrayList(LeafSpan),

    all_features: []u32,
    tree_features: []u32,
    level_features: []u32,
    node_features: []u32,
    n_tree_features: usize,
    n_level_features: usize,
    level_depth: i64,

    rng: std.Random.DefaultPrng,

    pub fn init(
        gpa: std.mem.Allocator,
        pool: *Pool,
        ds: *const Dataset,
        cfg: config.Config,
    ) !Builder {
        var bank = try hist.Bank.init(gpa, pool.workerCount(), ds.n_features, ds.n_bins);
        errdefer bank.deinit();

        const slot_len = bank.slotLen();
        const want_slots: usize = @as(usize, cfg.leafBudget()) + 2;
        const affordable = slot_memory_budget / (slot_len * @sizeOf(hist.Bin));
        if (want_slots > affordable) return error.TreeTooLargeForHistogramBudget;

        const slots = try gpa.alloc(hist.Bin, want_slots * slot_len);
        errdefer gpa.free(slots);

        var free_slots: std.ArrayList(u32) = .empty;
        errdefer free_slots.deinit(gpa);
        try free_slots.ensureTotalCapacity(gpa, want_slots);
        var s: u32 = 0;
        while (s < want_slots) : (s += 1) free_slots.appendAssumeCapacity(s);

        const rows = try gpa.alloc(u32, ds.n_rows);
        errdefer gpa.free(rows);
        const rows_out = try gpa.alloc(u32, ds.n_rows);
        errdefer gpa.free(rows_out);
        const total_partial = try gpa.alloc(hist.Bin, total_chunks);
        errdefer gpa.free(total_partial);

        const n_chunks = pool.workerCount() * 4 + 2;
        const part_counts = try gpa.alloc(usize, n_chunks);
        errdefer gpa.free(part_counts);
        const part_left = try gpa.alloc(usize, n_chunks);
        errdefer gpa.free(part_left);
        const part_right = try gpa.alloc(usize, n_chunks);
        errdefer gpa.free(part_right);

        const all_features = try gpa.alloc(u32, ds.n_features);
        errdefer gpa.free(all_features);
        for (all_features, 0..) |*f, i| f.* = @intCast(i);
        const tree_features = try gpa.alloc(u32, ds.n_features);
        errdefer gpa.free(tree_features);
        const level_features = try gpa.alloc(u32, ds.n_features);
        errdefer gpa.free(level_features);
        const node_features = try gpa.alloc(u32, ds.n_features);

        return .{
            .gpa = gpa,
            .pool = pool,
            .ds = ds,
            .cfg = cfg,
            .bank = bank,
            .slots = slots,
            .free_slots = free_slots,
            .rows = rows,
            .g = &.{},
            .rows_out = rows_out,
            .total_partial = total_partial,
            .part_counts = part_counts,
            .part_left = part_left,
            .part_right = part_right,
            .n_active = 0,
            .nodes = .empty,
            .queue = .empty,
            .leaves = .empty,
            .all_features = all_features,
            .tree_features = tree_features,
            .level_features = level_features,
            .node_features = node_features,
            .n_tree_features = 0,
            .n_level_features = 0,
            .level_depth = -1,
            .rng = .init(cfg.seed),
        };
    }

    pub fn deinit(b: *Builder) void {
        const gpa = b.gpa;
        b.bank.deinit();
        gpa.free(b.slots);
        b.free_slots.deinit(gpa);
        gpa.free(b.rows);
        gpa.free(b.rows_out);
        gpa.free(b.total_partial);
        gpa.free(b.part_counts);
        gpa.free(b.part_left);
        gpa.free(b.part_right);
        gpa.free(b.all_features);
        gpa.free(b.tree_features);
        gpa.free(b.level_features);
        gpa.free(b.node_features);
        b.nodes.deinit(gpa);
        b.queue.deinit(gpa);
        b.leaves.deinit(gpa);
        b.* = undefined;
    }

    inline fn slot(b: *Builder, id: u32) []hist.Bin {
        const len = b.bank.slotLen();
        return b.slots[@as(usize, id) * len ..][0..len];
    }

    fn takeSlot(b: *Builder) u32 {
        return b.free_slots.pop().?;
    }

    fn giveSlot(b: *Builder, id: u32) void {
        b.free_slots.appendAssumeCapacity(id);
    }

    /// Partial Fisher-Yates: sample `k` of `src[0..n]` into `dst`.
    fn sample(b: *Builder, src: []u32, n: usize, dst: []u32, rate: f32) usize {
        if (rate >= 1.0) {
            @memcpy(dst[0..n], src[0..n]);
            return n;
        }
        var k: usize = @intFromFloat(@round(@as(f32, @floatFromInt(n)) * rate));
        k = std.math.clamp(k, 1, n);
        const r = b.rng.random();
        var i: usize = 0;
        while (i < k) : (i += 1) {
            const j = i + r.uintLessThan(usize, n - i);
            std.mem.swap(u32, &src[i], &src[j]);
            dst[i] = src[i];
        }
        return k;
    }

    /// Features whose histograms are materialised for every node of this tree.
    ///
    /// This is deliberately the per-*tree* sample and never the per-level or
    /// per-node one. A node's histogram is derived from its parent's by
    /// subtraction, which is only valid where parent and child cover the same
    /// features; sampling per node would subtract against bins the parent
    /// never accumulated. Level/node sampling therefore restricts which
    /// features are *considered* for a split, not which are built — which is
    /// also what XGBoost and LightGBM do.
    fn treeFeatures(b: *const Builder) []const u32 {
        return b.tree_features[0..b.n_tree_features];
    }

    fn featuresFor(b: *Builder, depth: u32) []u32 {
        if (b.level_depth != @as(i64, depth)) {
            b.n_level_features = b.sample(
                b.tree_features,
                b.n_tree_features,
                b.level_features,
                b.cfg.colsample_bylevel,
            );
            b.level_depth = @intCast(depth);
        }
        const k = b.sample(
            b.level_features,
            b.n_level_features,
            b.node_features,
            b.cfg.colsample_bynode,
        );
        return b.node_features[0..k];
    }

    fn splitParams(b: *const Builder) hist.SplitParams {
        return .{
            .lambda = b.cfg.lambda,
            .alpha = b.cfg.alpha,
            .min_split_gain = b.cfg.min_split_gain,
            .min_child_weight = b.cfg.min_child_weight,
            .min_child_samples = b.cfg.min_child_samples,
            .max_delta_step = b.cfg.max_delta_step,
        };
    }

    /// Chunks the root sum is split into. Fixed, not derived from the thread
    /// count, so the result does not depend on `--n_threads`.
    const total_chunks: usize = 64;
    /// Below this many rows the barrier costs more than the sum saves.
    const total_parallel_min: usize = 1 << 15;

    /// Below this many rows the barriers cost more than the scan saves.
    ///
    /// Tuned rather than guessed: sweeping it, 32768 (the original guess)
    /// spends 360 ms in partition on a 200-tree fit where 2048 spends 271 ms,
    /// and the curve rises monotonically above that. Output is identical at
    /// every setting.
    const parallel_partition_min: usize = 2048;

    inline fn goesLeft(bin: u8, sp: hist.Split) bool {
        return if (bin == 0) sp.missing_left else bin <= sp.threshold;
    }

    /// Reorder `rows[start..end]` so the left child's rows come first.
    /// Only row ids move; gradients are looked up by row id.
    ///
    /// This is per-level O(rows) work and used to be the serial half of tree
    /// building: with it single-threaded, 16 threads bought only 1.66x over 1,
    /// and time scaled with tree *depth* rather than node count. It is now a
    /// count / prefix-sum / scatter, which is three parallel passes instead of
    /// one serial one.
    fn partition(b: *Builder, start: usize, end: usize, sp: hist.Split) usize {
        const n = end - start;
        if (n < parallel_partition_min or b.pool.workerCount() == 1)
            return b.partitionSerial(start, end, sp);

        const col = b.ds.column(sp.feature);
        const n_chunks = b.pool.workerCount() * 4;
        const csize = (n + n_chunks - 1) / n_chunks;
        const used = (n + csize - 1) / csize;
        std.debug.assert(used <= b.part_counts.len);

        var ctx = PartCtx{
            .rows = b.rows,
            .rows_out = b.rows_out,
            .col = col,
            .sp = sp,
            .start = start,
            .chunk = csize,
            .counts = b.part_counts[0..used],
            .left = b.part_left[0..used],
            .right = b.part_right[0..used],
        };

        // 1. How many of each chunk's rows go left.
        const t_c = prof.start();
        b.pool.parallelFor(n, &ctx, PartCtx.count, csize);
        prof.stop(.part_count, t_c);

        // 2. Prefix sums. `used` is a few dozen at most, so serial is right.
        var total_left: usize = 0;
        for (ctx.counts) |c| total_left += c;
        var l: usize = start;
        var r: usize = start + total_left;
        for (ctx.counts, 0..) |c, i| {
            ctx.left[i] = l;
            ctx.right[i] = r;
            l += c;
            r += (@min((i + 1) * csize, n) - i * csize) - c;
        }

        // 3. Scatter to the scratch buffers, then copy the touched range back.
        const t_s = prof.start();
        b.pool.parallelFor(n, &ctx, PartCtx.scatter, csize);
        prof.stop(.part_scatter, t_s);
        const t_b = prof.start();
        b.pool.parallelFor(n, &ctx, PartCtx.copyBack, 8192);
        prof.stop(.part_copy, t_b);

        return start + total_left;
    }

    /// The one-thread path, and stable like the parallel one.
    ///
    /// It used to be an in-place Hoare swap, which is fewer passes but leaves
    /// a node's rows in arbitrary order. That matters now: the histogram
    /// kernel reads the row-major bin matrix, and an ascending row order is
    /// what lets the hardware prefetcher follow it. Two branchless passes and
    /// a memcpy beat one pass of mispredicted swaps anyway.
    fn partitionSerial(b: *Builder, start: usize, end: usize, sp: hist.Split) usize {
        const col = b.ds.column(sp.feature);
        const rows = b.rows[start..end];
        var n_left: usize = 0;
        for (rows) |row| n_left += @intFromBool(goesLeft(col[row], sp));

        var li = start;
        var ri = start + n_left;
        for (rows) |row| {
            const left = goesLeft(col[row], sp);
            const dst = if (left) li else ri;
            b.rows_out[dst] = row;
            li += @intFromBool(left);
            ri += @intFromBool(!left);
        }
        @memcpy(rows, b.rows_out[start..end]);
        return start + n_left;
    }

    fn makeLeaf(b: *Builder, w: Work) !void {
        const p = b.splitParams();
        // Shrinkage is applied here, so a finished tree's output is already
        // its contribution to the ensemble and no caller has to remember eta.
        const raw = hist.leafWeight(w.total.g, w.total.h, p);
        const weight: f32 = @floatCast(raw * b.cfg.learning_rate);
        b.nodes.items[w.node] = .{ .is_leaf = true, .weight = weight };
        try b.leaves.append(b.gpa, .{ .start = w.start, .end = w.end, .weight = weight });
        b.giveSlot(w.slot);
    }

    /// Gradient and hessian sums over a node's rows.
    ///
    /// Only the root needs this — every other node inherits its total from
    /// its parent's split — but the root is *every* row, so on 668k rows over
    /// 200 trees the serial version was 46 ms, 7% of a fit and one of the
    /// three things holding 8-thread scaling to 4.8x against xgboost's 5.3x.
    ///
    /// Summed in a fixed number of chunks, reduced in chunk order, regardless
    /// of how many threads are running. A plain `parallelFor` over rows would
    /// group the additions differently at one thread than at eight and make
    /// the model depend on `--n_threads`, which is a property worth more than
    /// the milliseconds.
    fn totalOf(b: *Builder, rows: []const u32) hist.Bin {
        const n = rows.len;
        // Deliberately *not* also conditioned on the worker count: that would
        // make one thread group the additions differently from eight, which
        // is the thing this is arranged to avoid. `parallelFor` already runs
        // the chunks inline when there is one worker.
        if (n < total_parallel_min) {
            var g: f64 = 0;
            var h: f64 = 0;
            for (rows) |r| {
                const p = b.g[r];
                g += p.g;
                h += p.h;
            }
            return .{ .g = g, .h = h, .n = @floatFromInt(n) };
        }

        var ctx = TotalCtx{
            .g = b.g,
            .rows = rows,
            .size = (n + total_chunks - 1) / total_chunks,
            .partial = b.total_partial,
        };
        b.pool.parallelFor(total_chunks, &ctx, TotalCtx.run, 1);

        var g: f64 = 0;
        var h: f64 = 0;
        for (b.total_partial) |t| {
            g += t.g;
            h += t.h;
        }
        return .{ .g = g, .h = h, .n = @floatFromInt(n) };
    }

    /// Fill `rows[0..n_active]` with the rows this tree will see.
    ///
    /// An explicit `subset` wins outright. Otherwise `bootstrap` draws
    /// `subsample * n` rows *with* replacement (bagging, which is what makes a
    /// forest a forest), and without it we shuffle and take a prefix.
    fn selectRows(b: *Builder, subset: ?[]const u32) void {
        const n_all = b.ds.n_rows;

        if (subset) |sel| {
            @memcpy(b.rows[0..sel.len], sel);
            b.n_active = sel.len;
            return;
        }

        var k: usize = n_all;
        if (b.cfg.subsample < 1.0) {
            k = @intFromFloat(@round(@as(f32, @floatFromInt(n_all)) * b.cfg.subsample));
            k = std.math.clamp(k, 1, n_all);
        }

        const r = b.rng.random();
        if (b.cfg.bootstrap) {
            // With replacement: duplicates are the point. A row drawn twice
            // simply carries twice the weight in this tree's histograms.
            for (b.rows[0..k]) |*slot_row| slot_row.* = r.uintLessThan(u32, @intCast(n_all));
            b.n_active = k;
            return;
        }

        for (b.rows, 0..) |*row, i| row.* = @intCast(i);
        if (k < n_all) {
            var i: usize = 0;
            while (i < k) : (i += 1) {
                const j = i + r.uintLessThan(usize, n_all - i);
                std.mem.swap(u32, &b.rows[i], &b.rows[j]);
            }
        }
        b.n_active = k;
    }

    /// Grow one tree against `gradients` (indexed by original row id).
    /// Returns the tree; `leafSpans()` describes where its leaves' rows landed.
    pub fn grow(b: *Builder, gradients: []const hist.GradPair) !Tree {
        return b.growRows(gradients, null);
    }

    /// As `grow`, but over an explicit row set. GOSS uses this to hand the
    /// builder the rows it chose; passing null falls back to the config's own
    /// `subsample`/`bootstrap` policy.
    pub fn growRows(
        b: *Builder,
        gradients: []const hist.GradPair,
        subset: ?[]const u32,
    ) !Tree {
        b.nodes.clearRetainingCapacity();
        b.queue.clearRetainingCapacity();
        b.leaves.clearRetainingCapacity();
        b.level_depth = -1;

        const t_sel = prof.start();
        b.selectRows(subset);
        prof.stop(.select_rows, t_sel);

        b.g = gradients;

        b.n_tree_features = b.sample(
            b.all_features,
            b.ds.n_features,
            b.tree_features,
            b.cfg.colsample_bytree,
        );

        const p = b.splitParams();
        const leaf_cap: usize = if (b.cfg.max_leaves != 0) b.cfg.max_leaves else std.math.maxInt(usize);

        // --- root ---
        try b.nodes.append(b.gpa, .{});
        const root_slot = b.takeSlot();
        const root_total = b.totalOf(b.rows[0..b.n_active]);
        const tree_feats = b.treeFeatures();
        const root_search = b.featuresFor(0);
        const t_rh = prof.start();
        hist.build(
            b.pool,
            &b.bank,
            b.ds,
            b.rows[0..b.n_active],
            b.g,
            tree_feats,
            b.slot(root_slot),
        );
        prof.stop(.hist_build, t_rh);
        const t_rs = prof.start();
        const root_split = hist.bestSplit(&b.bank, b.slot(root_slot), b.ds, root_search, root_total, p);
        prof.stop(.best_split, t_rs);
        try b.queue.append(b.gpa, .{
            .node = 0,
            .start = 0,
            .end = b.n_active,
            .depth = 0,
            .total = root_total,
            .slot = root_slot,
            .split = root_split,
        });

        // --- expansion ---
        while (b.queue.items.len != 0) {
            const idx = b.pickNext();
            const w = b.queue.orderedRemove(idx);

            const leaf_count = b.leaves.items.len + b.queue.items.len + 1;
            const depth_capped = b.cfg.max_depth != 0 and w.depth >= b.cfg.max_depth;
            if (!w.split.valid() or depth_capped or leaf_count >= leaf_cap) {
                try b.makeLeaf(w);
                continue;
            }

            const t_part = prof.start();
            const mid = b.partition(w.start, w.end, w.split);
            prof.stop(.partition, t_part);
            // A split the histogram endorsed but the partition cannot realise
            // (every row on one side) would loop forever; treat it as a leaf.
            if (mid == w.start or mid == w.end) {
                try b.makeLeaf(w);
                continue;
            }

            const li: u32 = @intCast(b.nodes.items.len);
            try b.nodes.append(b.gpa, .{});
            const ri: u32 = @intCast(b.nodes.items.len);
            try b.nodes.append(b.gpa, .{});
            b.nodes.items[w.node] = .{
                .feature = w.split.feature,
                .threshold = w.split.threshold,
                .missing_left = w.split.missing_left,
                .is_leaf = false,
                .left = li,
                .right = ri,
            };

            // Will these children only ever become leaves?
            //
            // `makeLeaf` reads a node's total, its row span and its slot --
            // never its histogram or its split. So for a child that is
            // certain to be a leaf, building the histogram and searching it
            // is pure waste, and it is not a small amount: half of a
            // depthwise tree's nodes are its last level, so half of all
            // histogram builds and split searches were thrown away.
            //
            // Two cases are decidable here. The depth cap is static. The leaf
            // cap is too, because `leaf_count` at pop time never decreases --
            // a pop either makes a leaf (queue -1, leaves +1) or splits
            // (queue -1 +2, leaves +0) -- so once the budget is reached every
            // remaining pop is a leaf. `b.queue` is post-removal here, and
            // the next pop will see `leaves + queue + 2`.
            const at_depth_cap = b.cfg.max_depth != 0 and w.depth + 1 >= b.cfg.max_depth;
            const at_leaf_cap = b.leaves.items.len + b.queue.items.len + 2 >= leaf_cap;
            const children_are_leaves = at_depth_cap or at_leaf_cap;

            const slot_l = b.takeSlot();
            const slot_r = b.takeSlot();

            // Accumulate the smaller side; derive the larger by subtraction.
            const n_left = mid - w.start;
            const n_right = w.end - mid;
            if (children_are_leaves) {
                // Nothing to build. The slots are still taken and released
                // through the normal path so the free list behaves the same.
            } else if (n_left <= n_right) {
                const t_hb = prof.start();
                hist.build(b.pool, &b.bank, b.ds, b.rows[w.start..mid], b.g, tree_feats, b.slot(slot_l));
                prof.stop(.hist_build, t_hb);
                const t_hs = prof.start();
                hist.subtract(b.pool, &b.bank, b.slot(slot_r), b.slot(w.slot), b.slot(slot_l));
                prof.stop(.hist_subtract, t_hs);
            } else {
                const t_hb = prof.start();
                hist.build(b.pool, &b.bank, b.ds, b.rows[mid..w.end], b.g, tree_feats, b.slot(slot_r));
                prof.stop(.hist_build, t_hb);
                const t_hs = prof.start();
                hist.subtract(b.pool, &b.bank, b.slot(slot_l), b.slot(w.slot), b.slot(slot_r));
                prof.stop(.hist_subtract, t_hs);
            }
            b.giveSlot(w.slot);

            // Each child draws its own candidate features, as XGBoost does.
            // The draws happen even when the search is skipped: they consume
            // the builder's RNG, and skipping them would shift every later
            // sample and change the trees under `colsample_bylevel/bynode`.
            const t_bs = prof.start();
            var left_split: hist.Split = .{};
            var right_split: hist.Split = .{};
            // Draw, use, draw, use. `featuresFor` hands back a slice of one
            // shared buffer, so hoisting both draws above both searches makes
            // them alias and the left child gets searched with the right
            // child's sample -- silently, and only when a colsample is below
            // 1.0. The draws happen even when the search is skipped: they
            // consume the builder's RNG, and dropping them would shift every
            // later sample.
            const left_search = b.featuresFor(w.depth + 1);
            if (!children_are_leaves)
                left_split = hist.bestSplit(&b.bank, b.slot(slot_l), b.ds, left_search, w.split.left, p);
            const right_search = b.featuresFor(w.depth + 1);
            if (!children_are_leaves)
                right_split = hist.bestSplit(&b.bank, b.slot(slot_r), b.ds, right_search, w.split.right, p);
            prof.stop(.best_split, t_bs);

            try b.queue.append(b.gpa, .{
                .node = li,
                .start = w.start,
                .end = mid,
                .depth = w.depth + 1,
                .total = w.split.left,
                .slot = slot_l,
                .split = left_split,
            });
            try b.queue.append(b.gpa, .{
                .node = ri,
                .start = mid,
                .end = w.end,
                .depth = w.depth + 1,
                .total = w.split.right,
                .slot = slot_r,
                .split = right_split,
            });
        }

        return .{ .nodes = try b.gpa.dupe(Node, b.nodes.items) };
    }

    /// Depthwise takes the oldest pending node (breadth-first); lossguide takes
    /// the one whose split buys the most.
    fn pickNext(b: *Builder) usize {
        if (b.cfg.grow_policy == .depthwise) return 0;
        var best: usize = 0;
        var best_gain = -std.math.inf(f64);
        for (b.queue.items, 0..) |w, i| {
            if (w.split.gain > best_gain) {
                best_gain = w.split.gain;
                best = i;
            }
        }
        return best;
    }

    pub fn leafSpans(b: *const Builder) []const LeafSpan {
        return b.leaves.items;
    }

    pub fn activeRows(b: *const Builder) []const u32 {
        return b.rows[0..b.n_active];
    }
};


/// Shared state for the three passes of a parallel partition.
///
/// `parallelFor` hands out fixed-size chunks from a cursor, so `begin / chunk`
/// recovers which chunk a worker got — which is what lets the counting pass
/// and the scatter pass agree on where each chunk's output belongs without any
/// coordination between them.
/// One chunk of the root's gradient sum.
const TotalCtx = struct {
    g: []const hist.GradPair,
    rows: []const u32,
    size: usize,
    partial: []hist.Bin,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *TotalCtx = @ptrCast(@alignCast(ctx));
        // `begin`/`end` index chunks, not rows, so a chunk always covers the
        // same rows however the pool hands them out.
        var c = begin;
        while (c < end) : (c += 1) {
            const lo = @min(c * self.size, self.rows.len);
            const hi = @min(lo + self.size, self.rows.len);
            var g: f64 = 0;
            var h: f64 = 0;
            for (self.rows[lo..hi]) |r| {
                const p = self.g[r];
                g += p.g;
                h += p.h;
            }
            self.partial[c] = .{ .g = g, .h = h, .n = @floatFromInt(hi - lo) };
        }
    }
};

const PartCtx = struct {
    rows: []u32,
    rows_out: []u32,
    col: []const u8,
    sp: hist.Split,
    start: usize,
    chunk: usize,
    counts: []usize,
    left: []usize,
    right: []usize,

    fn count(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *PartCtx = @ptrCast(@alignCast(ctx));
        var n: usize = 0;
        for (self.rows[self.start + begin .. self.start + end]) |row| {
            if (Builder.goesLeft(self.col[row], self.sp)) n += 1;
        }
        self.counts[begin / self.chunk] = n;
    }

    fn scatter(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *PartCtx = @ptrCast(@alignCast(ctx));
        const c = begin / self.chunk;
        var li = self.left[c];
        var ri = self.right[c];
        var i = self.start + begin;
        while (i < self.start + end) : (i += 1) {
            const row = self.rows[i];
            // Branchless: which side a row takes is close to a coin flip near
            // a good split, so a branch here mispredicts about half the time.
            // Selecting the cursor instead costs a cmov and nothing else.
            const left = Builder.goesLeft(self.col[row], self.sp);
            const dst = if (left) li else ri;
            self.rows_out[dst] = row;
            li += @intFromBool(left);
            ri += @intFromBool(!left);
        }
    }

    fn copyBack(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *PartCtx = @ptrCast(@alignCast(ctx));
        const a = self.start + begin;
        const b_ = self.start + end;
        @memcpy(self.rows[a..b_], self.rows_out[a..b_]);
    }
};

