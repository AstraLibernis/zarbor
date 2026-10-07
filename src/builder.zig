// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Growing one tree: sampling, histograms, split search, partitioning and leaves.

const std = @import("std");
const Pool = @import("pool.zig").Pool;
const data = @import("data.zig");
const Dataset = data.Dataset;
const hist = @import("hist.zig");
const prof = @import("prof.zig");
const split = @import("split.zig");
const tree = @import("tree.zig");
const Params = tree.Params;
const Node = tree.Node;
const LinTerm = tree.LinTerm;
const LeafSpan = tree.LeafSpan;
const Tree = tree.Tree;

/// Hard cap on root-to-leaf features a linear leaf may use; independent of `max_depth` so `Work` is
/// fixed-size.
pub const max_path: usize = 8;

pub const Work = struct {
    node: u32,
    start: usize,
    end: usize,
    depth: u32,
    total: hist.Bin,
    slot: u32,
    split: split.Split,
    /// Distinct numeric features tested between root and this node. Empty unless `linear_leaves`:
    /// otherwise pure cost for something nothing reads.
    path: [max_path]u32 = undefined,
    n_path: u8 = 0,
};

/// `expandBatch` takes runs of pending nodes up to this many rows. Two invariants keep a batch's
/// trees bit-identical to one-at-a-time expansion: at most `2 * hist.parallel_threshold`, so the
/// smaller child is at most `hist.parallel_threshold` and `hist.build` would build it serially
/// too; and below partition.zig's `parallel_partition_min`, so `partition` would run serially too
/// (`partitionSmall` asserts this, in Debug only).
const batch_max_rows: usize = 2 * hist.parallel_threshold;
const batch_max_nodes: usize = 256;
/// Fewer nodes than this are not worth two barriers.
const batch_min_nodes: usize = 4;

/// One split node of a batch: what `expandBatch`'s parallel step needs, fixed in its serial step.
const BatchJob = struct {
    parent_slot: u32,
    slot_l: u32,
    slot_r: u32,
    start: usize,
    mid: usize,
    end: usize,
    /// False when both children can only be leaves: no histogram, no search.
    search: bool,
    total_l: hist.Bin,
    total_r: hist.Bin,
    feats_l: []const u32,
    feats_r: []const u32,
    /// The left child's index in `queue.items`; the right child is next.
    queue_left: usize,
};

const BatchPartCtx = struct {
    b: *Builder,
    work: []const Work,
    mids: []usize,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *BatchPartCtx = @ptrCast(@alignCast(ctx));
        for (self.work[begin..end], self.mids[begin..end]) |w, *mid| {
            mid.* = if (self.b.willSplit(w)) partitionSmall(self.b, w.start, w.end, w.split) else w.start;
        }
    }
};

const BatchBuildCtx = struct {
    b: *Builder,
    jobs: []const BatchJob,
    p: split.SplitParams,
    tree_feats: []const u32,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *BatchBuildCtx = @ptrCast(@alignCast(ctx));
        const b = self.b;
        for (self.jobs[begin..end]) |j| {
            if (!j.search) continue;
            const sl = b.slot(j.slot_l);
            const sr = b.slot(j.slot_r);
            // Accumulate the smaller side; derive the larger by subtraction.
            if (j.mid - j.start <= j.end - j.mid) {
                hist.buildInto(&b.bank, b.ds, b.rows[j.start..j.mid], b.g, self.tree_feats, sl);
                hist.subtractInto(&b.bank, sr, b.slot(j.parent_slot), sl);
            } else {
                hist.buildInto(&b.bank, b.ds, b.rows[j.mid..j.end], b.g, self.tree_feats, sr);
                hist.subtractInto(&b.bank, sl, b.slot(j.parent_slot), sr);
            }
            const q = b.queue.items[j.queue_left..][0..2];
            q[0].split = split.bestSplit(&b.bank, sl, b.ds, j.feats_l, j.total_l, self.p);
            q[1].split = split.bestSplit(&b.bank, sr, b.ds, j.feats_r, j.total_r, self.p);
        }
    }
};

const partitionSmall = @import("partition.zig").partitionSmall;

/// `Builder.clearCounts` / `emitSample` over row ids, in chunks of `chunk`; `total` and `emit` walk
/// their range chunk by chunk, since one worker can get the whole range in one call.
const SampleCtx = struct {
    b: *Builder,
    chunk: usize,

    fn clear(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *SampleCtx = @ptrCast(@alignCast(ctx));
        @memset(self.b.row_counts[begin..end], 0);
    }

    fn total(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *SampleCtx = @ptrCast(@alignCast(ctx));
        var lo = begin;
        while (lo < end) {
            const hi = @min(lo + self.chunk, end);
            var t: usize = 0;
            for (self.b.row_counts[lo..hi]) |c| t += c;
            self.b.sample_at[lo / self.chunk] = t;
            lo = hi;
        }
    }

    fn emit(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *SampleCtx = @ptrCast(@alignCast(ctx));
        const rows = self.b.rows;
        var lo = begin;
        while (lo < end) {
            const hi = @min(lo + self.chunk, end);
            var at = self.b.sample_at[lo / self.chunk];
            for (self.b.row_counts[lo..hi], lo..) |c, row| {
                for (0..c) |_| {
                    rows[at] = @intCast(row);
                    at += 1;
                }
            }
            lo = hi;
        }
    }
};

/// Hard ceiling on histogram slot memory; beyond it the caller must shrink the tree, not the
/// allocator decide.
const slot_memory_budget: usize = 2 << 30;

pub const Builder = struct {
    gpa: std.mem.Allocator,
    pool: *Pool,
    ds: *const Dataset,
    cfg: Params,
    bank: hist.Bank,

    /// `cfg.leafBudget() + 2` histograms (`want_slots` in `init`), each `bank.slotLen()` bins.
    slots: []hist.Bin,
    free_slots: std.ArrayList(u32),

    rows: []u32,
    /// Gradients indexed by original row id, borrowed for the current tree. Not permuted to match
    /// `rows`: that paid off when the histogram loop was feature-outer, but row-outer reads each
    /// once per row, and permuting makes the partition move a `hist.GradPair` with every row id;
    /// the gather costs far less than that traffic (see docs/measurements.md).
    g: []const hist.GradPair,
    /// Destination for the parallel partition, which cannot be done in place.
    rows_out: []u32,
    /// Per-chunk partial sums for `totalOf`.
    total_partial: []hist.Bin,
    /// Per-chunk left-hand counts for the parallel `partition`; `part_left` / `part_right` hold
    /// each chunk's first left and right output position. All three are `workerCount() * 4 + 2`
    /// long; `partition` uses at most `workerCount() * 4` chunks.
    part_counts: []usize,
    part_left: []usize,
    part_right: []usize,
    /// Rows active for the current tree: `rows[0..n_active]`.
    n_active: usize,
    /// Per-row draw counts for putting a sampled tree's rows in ascending order (`emitSample`).
    row_counts: []u32,
    /// Per-chunk row totals, then output offsets, for `emitSample`'s parallel emit (`SampleCtx`).
    sample_at: []usize,

    nodes: std.ArrayList(Node),
    cat_ids: std.ArrayList(data.BinIdx),
    lin: std.ArrayList(LinTerm),
    queue: std.ArrayList(Work),
    /// Depthwise pops `queue.items[queue_head]` and advances; the items before it are spent.
    /// `orderedRemove(0)` shifted the whole frontier of `Work`s (`@sizeOf(Work)` each) on every
    /// pop, a measurable share of a large forest tree (see docs/measurements.md). Lossguide still removes in place (its queue is short, and
    /// the remaining order breaks gain ties), so its head stays 0.
    queue_head: usize = 0,
    leaves: std.ArrayList(LeafSpan),

    all_features: []u32,
    tree_features: []u32,
    level_features: []u32,
    node_features: []u32,
    n_tree_features: usize,
    n_level_features: usize,
    level_depth: i64,

    rng: std.Random.DefaultPrng,

    /// Scratch for `expandBatch`: the batch's `Work`, each node's partition point, the split nodes'
    /// jobs, and two feature samples per job (`featuresFor` returns one shared buffer).
    batch_work: []Work,
    batch_mid: []usize,
    batch_jobs: []BatchJob,
    batch_feats: []u32,

    pub fn init(
        gpa: std.mem.Allocator,
        pool: *Pool,
        ds: *const Dataset,
        cfg: Params,
    ) !Builder {
        var bank = try hist.Bank.init(gpa, ds.n_features, ds.n_bins);
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
        errdefer gpa.free(node_features);
        const batch_work = try gpa.alloc(Work, batch_max_nodes);
        errdefer gpa.free(batch_work);
        const batch_mid = try gpa.alloc(usize, batch_max_nodes);
        errdefer gpa.free(batch_mid);
        const batch_jobs = try gpa.alloc(BatchJob, batch_max_nodes);
        errdefer gpa.free(batch_jobs);
        const batch_feats = try gpa.alloc(u32, 2 * batch_max_nodes * @as(usize, ds.n_features));
        errdefer gpa.free(batch_feats);
        const row_counts = try gpa.alloc(u32, ds.n_rows);
        errdefer gpa.free(row_counts);
        const sample_at = try gpa.alloc(usize, pool.workerCount() * 4);

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
            .row_counts = row_counts,
            .sample_at = sample_at,
            .nodes = .empty,
            .cat_ids = .empty,
            .lin = .empty,
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
            .batch_work = batch_work,
            .batch_mid = batch_mid,
            .batch_jobs = batch_jobs,
            .batch_feats = batch_feats,
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
        gpa.free(b.batch_work);
        gpa.free(b.batch_mid);
        gpa.free(b.batch_jobs);
        gpa.free(b.batch_feats);
        gpa.free(b.row_counts);
        gpa.free(b.sample_at);
        b.nodes.deinit(gpa);
        b.cat_ids.deinit(gpa);
        b.lin.deinit(gpa);
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

    /// Partial Fisher-Yates: sample `k` of `src[0..n]` into `dst`. `k` is `floor(n * rate)`, at
    /// least 1, as XGBoost (whose `colsample_*` names these are) and scikit-learn's `max_features`
    /// count it, so a fraction is a ceiling on the columns a tree sees. LightGBM rounds instead; on
    /// 13 columns at 0.5 that is 7 against 6 (docs/parity.md).
    fn sample(b: *Builder, src: []u32, n: usize, dst: []u32, rate: f32) usize {
        if (rate >= 1.0) {
            @memcpy(dst[0..n], src[0..n]);
            return n;
        }
        var k: usize = @intFromFloat(@floor(@as(f32, @floatFromInt(n)) * rate));
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

    /// Features whose histograms are materialised for every node of this tree: always the per-tree
    /// sample, never per-level/node. Subtraction needs parent and child to cover the same features,
    /// so level/node sampling restricts which features are considered, not built (as XGBoost and
    /// LightGBM do).
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

    fn splitParams(b: *const Builder) split.SplitParams {
        return .{
            .lambda = b.cfg.lambda,
            .alpha = b.cfg.alpha,
            .min_split_gain = b.cfg.min_split_gain,
            .min_child_weight = b.cfg.min_child_weight,
            .min_child_samples = b.cfg.min_child_samples,
            .max_delta_step = b.cfg.max_delta_step,
            .cat_optimal = b.cfg.cat_split == .optimal,
            .cat_smooth = b.cfg.cat_smooth,
            .cat_l2 = b.cfg.cat_l2,
            .max_cat_threshold = b.cfg.max_cat_threshold,
            .max_cat_to_onehot = b.cfg.max_cat_to_onehot,
            .min_data_per_group = b.cfg.min_data_per_group,
        };
    }

    /// Chunks the root sum is split into; fixed so the result does not depend on `--n_threads`.
    const total_chunks: usize = 64;
    /// Below this many rows the barrier costs more than the sum saves.
    const total_parallel_min: usize = 1 << 15;

    const partition = @import("partition.zig").partition;

    fn makeLeaf(b: *Builder, w: Work) !void {
        const p = b.splitParams();
        // Shrinkage applied here: a finished tree's output is already its ensemble contribution (no
        // eta downstream).
        const raw = split.leafWeight(w.total.g, w.total.h, p);
        const weight: f32 = @floatCast(raw * b.cfg.learning_rate);
        var n_lin: u8 = 0;
        var lin_ofs: u32 = 0;
        if (b.cfg.linear_leaves and w.n_path != 0) {
            lin_ofs = @intCast(b.lin.items.len);
            n_lin = try b.fitLinearLeaf(w);
        }
        b.nodes.items[w.node] = .{
            .is_leaf = true,
            .weight = weight,
            .n_lin = n_lin,
            .lin_ofs = lin_ofs,
        };
        try b.leaves.append(b.gpa, .{ .start = w.start, .end = w.end, .weight = weight });
        b.giveSlot(w.slot);
    }

    const fitLinearLeaf = @import("leaf_linear.zig").fitLinearLeaf;

    /// Gradient and hessian sums over a node's rows. Only the root needs it (others inherit from
    /// the parent's split), but the root is every row, and done serially it was one of the things
    /// holding back thread scaling against xgboost (see docs/measurements.md). Fixed chunk count,
    /// reduced in chunk order at any thread count: a plain `parallelFor` over rows would group
    /// additions per thread count, making the model depend on `--n_threads`, which matters more
    /// than the speed.
    fn totalOf(b: *Builder, rows: []const u32) hist.Bin {
        const n = rows.len;
        // Deliberately not conditioned on worker count: that would group additions differently at
        // one thread than at eight. `parallelFor` already runs the chunks inline with one worker.
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

    /// Fill `rows[0..n_active]` with this tree's rows. An explicit `subset` wins. Else `bootstrap`
    /// draws `subsample * n` rows with replacement (bagging: what makes a forest a forest); without
    /// it, shuffle and take a prefix.
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
            // With replacement: duplicates are the point; a row drawn twice carries twice the
            // weight.
            // Counted as drawn; `emitSample` writes them out in ascending order.
            b.clearCounts();
            for (0..k) |_| b.row_counts[r.uintLessThan(u32, @intCast(n_all))] += 1;
            b.emitSample(k);
            return;
        }

        for (b.rows, 0..) |*row, i| row.* = @intCast(i);
        if (k < n_all) {
            var i: usize = 0;
            while (i < k) : (i += 1) {
                const j = i + r.uintLessThan(usize, n_all - i);
                std.mem.swap(u32, &b.rows[i], &b.rows[j]);
            }
            b.clearCounts();
            for (b.rows[0..k]) |row| b.row_counts[row] += 1;
            b.emitSample(k);
            return;
        }
        b.n_active = k;
    }

    /// Zero `row_counts` on the pool, ready for a sampled tree's draws to be counted into it.
    fn clearCounts(b: *Builder) void {
        var ctx = SampleCtx{ .b = b, .chunk = b.sampleChunk() };
        b.pool.parallelFor(b.row_counts.len, &ctx, SampleCtx.clear, ctx.chunk);
    }

    /// Write the `k` rows counted in `row_counts` into `rows[0..k]` in ascending order, duplicates
    /// adjacent, and set `n_active`. The histogram kernel walks rows in order through the row-major
    /// matrix and the gradients, and the stable partition keeps the root's order all the way down;
    /// in draw order every row was a cache miss (see docs/measurements.md). Same rows, same
    /// multiplicities, same RNG draws (so every later sample is unchanged); only the order of
    /// additions inside a histogram moves. On the pool: per-chunk totals, a serial prefix, and each
    /// chunk fills its own output range, so the order is the same at any thread count.
    fn emitSample(b: *Builder, k: usize) void {
        var ctx = SampleCtx{ .b = b, .chunk = b.sampleChunk() };
        const n = b.row_counts.len;
        b.pool.parallelFor(n, &ctx, SampleCtx.total, ctx.chunk);
        var at: usize = 0;
        for (b.sample_at[0 .. (n + ctx.chunk - 1) / ctx.chunk]) |*t| {
            const c = t.*;
            t.* = at;
            at += c;
        }
        std.debug.assert(at == k);
        b.pool.parallelFor(n, &ctx, SampleCtx.emit, ctx.chunk);
        b.n_active = k;
    }

    /// No more chunks than `sample_at` holds, so the size asked for is the one the pool uses.
    fn sampleChunk(b: *const Builder) usize {
        const n = b.row_counts.len;
        return @max((n + b.sample_at.len - 1) / b.sample_at.len, 16384);
    }

    /// Grow one tree against `gradients` (indexed by original row id); `leafSpans()` gives leaf row
    /// spans.
    pub fn grow(b: *Builder, gradients: []const hist.GradPair) !Tree {
        return b.growRows(gradients, null);
    }

    /// As `grow`, over an explicit row set (GOSS hands in its chosen rows); null falls back to the
    /// config's `subsample`/`bootstrap` policy.
    pub fn growRows(
        b: *Builder,
        gradients: []const hist.GradPair,
        subset: ?[]const u32,
    ) !Tree {
        b.nodes.clearRetainingCapacity();
        b.cat_ids.clearRetainingCapacity();
        b.lin.clearRetainingCapacity();
        b.queue.clearRetainingCapacity();
        b.queue_head = 0;
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
        const root_split = split.bestSplit(&b.bank, b.slot(root_slot), b.ds, root_search, root_total, p);
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
        while (b.pending() != 0) {
            const k = b.batchSize(leaf_cap);
            if (k >= batch_min_nodes) {
                try b.expandBatch(k, p, leaf_cap, tree_feats);
                continue;
            }

            const w = b.popNext();

            const leaf_count = b.leaves.items.len + b.pending() + 1;
            if (!b.willSplit(w) or leaf_count >= leaf_cap) {
                try b.makeLeaf(w);
                continue;
            }

            const t_part = prof.start();
            const mid = b.partition(w.start, w.end, w.split);
            prof.stop(.partition, t_part);
            // A split the histogram endorsed but the partition cannot realise (all rows one side)
            // would loop forever; treat it as a leaf.
            if (mid == w.start or mid == w.end) {
                try b.makeLeaf(w);
                continue;
            }

            const kids = try b.splitNode(w);
            const li = kids[0];
            const ri = kids[1];
            const children_are_leaves = b.childrenAreLeaves(w, leaf_cap);

            const slot_l = b.takeSlot();
            const slot_r = b.takeSlot();

            // Accumulate the smaller side; derive the larger by subtraction.
            const n_left = mid - w.start;
            const n_right = w.end - mid;
            if (children_are_leaves) {
                // Nothing to build; slots still go through the normal path so the free list behaves
                // the same.
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

            // Each child draws its own candidate features, as XGBoost does. Draws happen even when
            // the search is skipped: they consume the RNG, and dropping them would shift every
            // later sample and change the trees under `colsample_bylevel/bynode`.
            const t_bs = prof.start();
            var left_split: split.Split = .{};
            var right_split: split.Split = .{};
            // Draw, use, draw, use: `featuresFor` returns a slice of one shared buffer, so hoisting
            // both draws aliases them and the left child silently gets the right's sample
            // (colsample < 1).
            const left_search = b.featuresFor(w.depth + 1);
            if (!children_are_leaves)
                left_split = split.bestSplit(&b.bank, b.slot(slot_l), b.ds, left_search, w.split.left, p);
            const right_search = b.featuresFor(w.depth + 1);
            if (!children_are_leaves)
                right_split = split.bestSplit(&b.bank, b.slot(slot_r), b.ds, right_search, w.split.right, p);
            prof.stop(.best_split, t_bs);

            try b.pushChildren(w, li, ri, mid, slot_l, slot_r, left_split, right_split);
        }

        const nodes = try b.gpa.dupe(Node, b.nodes.items);
        errdefer b.gpa.free(nodes);
        const ids: []data.BinIdx = if (b.cat_ids.items.len == 0)
            &.{}
        else
            try b.gpa.dupe(data.BinIdx, b.cat_ids.items);
        errdefer if (ids.len != 0) b.gpa.free(ids);
        const lin: []LinTerm = if (b.lin.items.len == 0)
            &.{}
        else
            try b.gpa.dupe(LinTerm, b.lin.items);
        return .{ .nodes = nodes, .cat_ids = ids, .lin = lin };
    }

    /// Whether a popped node splits, leaf budget aside (the caller checks that against its count).
    fn willSplit(b: *const Builder, w: Work) bool {
        const depth_capped = b.cfg.max_depth != 0 and w.depth >= b.cfg.max_depth;
        return w.split.valid() and !depth_capped;
    }

    /// Turn `w`'s node into a split: append its two children (ids returned) and its category ids.
    fn splitNode(b: *Builder, w: Work) ![2]u32 {
        const li: u32 = @intCast(b.nodes.items.len);
        try b.nodes.append(b.gpa, .{});
        const ri: u32 = @intCast(b.nodes.items.len);
        try b.nodes.append(b.gpa, .{});
        var cat_ofs: u32 = 0;
        if (w.split.is_cat) {
            cat_ofs = @intCast(b.cat_ids.items.len);
            try b.cat_ids.appendSlice(b.gpa, w.split.cat_ids[0..w.split.n_cat]);
        }
        b.nodes.items[w.node] = .{
            .feature = w.split.feature,
            .threshold = w.split.threshold,
            .missing_left = w.split.missing_left,
            .is_cat = w.split.is_cat,
            .n_cat = w.split.n_cat,
            .cat_ofs = cat_ofs,
            .is_leaf = false,
            .left = li,
            .right = ri,
        };
        return .{ li, ri };
    }

    /// Will these children only ever be leaves? `makeLeaf` reads a node's total, row span and slot,
    /// never its histogram or split, so building and searching those is waste: half a depthwise
    /// tree's nodes are its last level. Both caps are decidable here: depth is static, and
    /// `leaf_count` at pop time never decreases (a pop makes a leaf: queue -1, leaves +1; or
    /// splits: queue -1 +2), so once the budget is hit every remaining pop is a leaf. Called after
    /// the pop, before the children are queued: the next pop sees `leaves + queue + 2`.
    fn childrenAreLeaves(b: *const Builder, w: Work, leaf_cap: usize) bool {
        const at_depth_cap = b.cfg.max_depth != 0 and w.depth + 1 >= b.cfg.max_depth;
        const at_leaf_cap = b.leaves.items.len + b.pending() + 2 >= leaf_cap;
        return at_depth_cap or at_leaf_cap;
    }

    /// Queue `w`'s children. They inherit the path plus the feature just tested, unless categorical
    /// (a dictionary id is no number to fit a slope in) or already present.
    fn pushChildren(
        b: *Builder,
        w: Work,
        li: u32,
        ri: u32,
        mid: usize,
        slot_l: u32,
        slot_r: u32,
        left_split: split.Split,
        right_split: split.Split,
    ) !void {
        var path = w.path;
        var n_path = w.n_path;
        if (b.cfg.linear_leaves and
            b.ds.kinds[w.split.feature] == .numeric and
            n_path < max_path)
        {
            var seen = false;
            for (path[0..n_path]) |f| seen = seen or f == w.split.feature;
            if (!seen) {
                path[n_path] = w.split.feature;
                n_path += 1;
            }
        }
        try b.queue.append(b.gpa, .{
            .node = li,
            .start = w.start,
            .end = mid,
            .depth = w.depth + 1,
            .total = w.split.left,
            .slot = slot_l,
            .split = left_split,
            .path = path,
            .n_path = n_path,
        });
        try b.queue.append(b.gpa, .{
            .node = ri,
            .start = mid,
            .end = w.end,
            .depth = w.depth + 1,
            .total = w.split.right,
            .slot = slot_r,
            .split = right_split,
            .path = path,
            .n_path = n_path,
        });
    }

    /// How many pending nodes, from the head, `expandBatch` may take: a depthwise run of nodes of
    /// at most `batch_max_rows`, few enough that the leaf budget cannot bind inside the batch and
    /// that two fresh slots per node are free. 0 = expand one at a time.
    fn batchSize(b: *const Builder, leaf_cap: usize) usize {
        if (b.cfg.grow_policy != .depthwise or b.pool.workerCount() == 1) return 0;
        // The k-th pop of a batch sees `leaf_count` = leaves + pending at batch start + splits so
        // far <= that + k - 1, so `k <= leaf_cap - (leaves + pending)` keeps every pop under it.
        const used = b.leaves.items.len + b.pending();
        if (used >= leaf_cap) return 0;
        const limit = @min(@min(leaf_cap - used, b.free_slots.items.len / 2), batch_max_nodes);
        var k: usize = 0;
        for (b.queue.items[b.queue_head..]) |w| {
            if (k == limit or w.end - w.start > batch_max_rows) break;
            k += 1;
        }
        return k;
    }

    /// Expand the next `k` pending nodes together. A forest's lower levels are thousands of nodes of
    /// a few hundred rows, each too small to spread over the pool, so one at a time scaled poorly
    /// with threads (see docs/measurements.md). Here the per-node work (partition, histogram,
    /// subtraction, split search) runs one node per task, while everything order-sensitive -- leaf
    /// order, node and category ids, slot takes, RNG draws, queue order -- happens on this thread
    /// in pop order, exactly as the one-at-a-time loop does it. Every node here is small enough
    /// (`batch_max_rows`, below partition.zig's `parallel_partition_min` and at most
    /// `2 * hist.parallel_threshold`) that the one-at-a-time loop would partition it and build its
    /// smaller child serially too, with the same kernels, so the trees are identical. Only
    /// histogram slot ids differ: parents' slots go back after the batch, not per node, and slot
    /// ids reach nothing in the model.
    fn expandBatch(b: *Builder, k: usize, p: split.SplitParams, leaf_cap: usize, tree_feats: []const u32) !void {
        const work = b.batch_work[0..k];
        @memcpy(work, b.queue.items[b.queue_head..][0..k]);
        const mids = b.batch_mid[0..k];

        // 1. Partition every node that will split; their row ranges are disjoint.
        const t_part = prof.start();
        var pctx = BatchPartCtx{ .b = b, .work = work, .mids = mids };
        b.pool.parallelFor(k, &pctx, BatchPartCtx.run, 1);
        prof.stop(.partition, t_part);

        // 2. Bookkeeping, in pop order.
        const nf: usize = b.ds.n_features;
        var n_jobs: usize = 0;
        for (work, mids) |queued, mid| {
            const w = b.popNext();
            std.debug.assert(w.node == queued.node);
            std.debug.assert(b.leaves.items.len + b.pending() + 1 < leaf_cap);
            if (!b.willSplit(w) or mid == w.start or mid == w.end) {
                try b.makeLeaf(w);
                continue;
            }
            const kids = try b.splitNode(w);
            const children_are_leaves = b.childrenAreLeaves(w, leaf_cap);
            const slot_l = b.takeSlot();
            const slot_r = b.takeSlot();
            // Drawn even when unused, as the one-at-a-time loop does; copied out because the next
            // draw reuses the buffer.
            const feats = b.batch_feats[2 * n_jobs * nf ..][0 .. 2 * nf];
            const left_search = b.featuresFor(w.depth + 1);
            @memcpy(feats[0..left_search.len], left_search);
            const n_left_search = left_search.len;
            const right_search = b.featuresFor(w.depth + 1);
            @memcpy(feats[nf..][0..right_search.len], right_search);
            const q = b.queue.items.len;
            try b.pushChildren(w, kids[0], kids[1], mid, slot_l, slot_r, .{}, .{});
            b.batch_jobs[n_jobs] = .{
                .parent_slot = w.slot,
                .slot_l = slot_l,
                .slot_r = slot_r,
                .start = w.start,
                .mid = mid,
                .end = w.end,
                .search = !children_are_leaves,
                .total_l = w.split.left,
                .total_r = w.split.right,
                .feats_l = feats[0..n_left_search],
                .feats_r = feats[nf..][0..right_search.len],
                .queue_left = q,
            };
            n_jobs += 1;
        }

        // 3. Histograms and split searches, one task per split node.
        const t_build = prof.start();
        var cctx = BatchBuildCtx{ .b = b, .jobs = b.batch_jobs[0..n_jobs], .p = p, .tree_feats = tree_feats };
        b.pool.parallelFor(n_jobs, &cctx, BatchBuildCtx.run, 1);
        prof.stop(.hist_build, t_build);

        // 4. Every parent histogram has been read; hand the slots back.
        for (b.batch_jobs[0..n_jobs]) |j| b.giveSlot(j.parent_slot);
    }

    fn pending(b: *const Builder) usize {
        return b.queue.items.len - b.queue_head;
    }

    /// Depthwise takes the oldest pending node (breadth-first); lossguide takes the one whose split
    /// buys the most.
    fn popNext(b: *Builder) Work {
        if (b.cfg.grow_policy == .depthwise) {
            b.queue_head += 1;
            return b.queue.items[b.queue_head - 1];
        }
        var best: usize = 0;
        var best_gain = -std.math.inf(f64);
        for (b.queue.items, 0..) |w, i| {
            if (w.split.gain > best_gain) {
                best_gain = w.split.gain;
                best = i;
            }
        }
        return b.queue.orderedRemove(best);
    }

    pub fn leafSpans(b: *const Builder) []const LeafSpan {
        return b.leaves.items;
    }

    pub fn activeRows(b: *const Builder) []const u32 {
        return b.rows[0..b.n_active];
    }
};

/// One chunk of the root's gradient sum.
const TotalCtx = struct {
    g: []const hist.GradPair,
    rows: []const u32,
    size: usize,
    partial: []hist.Bin,

    fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
        _ = worker;
        const self: *TotalCtx = @ptrCast(@alignCast(ctx));
        // `begin`/`end` index chunks, not rows, so a chunk covers the same rows however handed out.
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
