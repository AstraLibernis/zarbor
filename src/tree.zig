//! Regression-tree construction over binned features.
//!
//! Rows belonging to a node are kept contiguous in `rows`, and the node's
//! gradients are permuted alongside them in `grads`. That pairing is the point:
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
    pub fn predictBinned(t: *const Tree, ds: *const Dataset, row: usize) f32 {
        var i: u32 = 0;
        while (!t.nodes[i].is_leaf) {
            const n = t.nodes[i];
            const b = ds.bins[@as(usize, n.feature) * ds.n_rows + row];
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
    grads: []hist.GradPair,
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
        var bank = try hist.Bank.init(gpa, pool.workerCount(), ds.n_features, ds.maxBins());
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
        const grads = try gpa.alloc(hist.GradPair, ds.n_rows);
        errdefer gpa.free(grads);

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
            .grads = grads,
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
        gpa.free(b.grads);
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

    /// Reorder `rows[start..end]` so the left child's rows come first.
    /// Gradients move with them, keeping `grads[i]` the gradient of `rows[i]`.
    fn partition(b: *Builder, start: usize, end: usize, sp: hist.Split) usize {
        const col = b.ds.column(sp.feature);
        var i = start;
        var j = end;
        while (i < j) {
            const bin = col[b.rows[i]];
            const go_left = if (bin == 0) sp.missing_left else bin <= sp.threshold;
            if (go_left) {
                i += 1;
            } else {
                j -= 1;
                std.mem.swap(u32, &b.rows[i], &b.rows[j]);
                std.mem.swap(hist.GradPair, &b.grads[i], &b.grads[j]);
            }
        }
        return i;
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

    fn totalOf(grads: []const hist.GradPair) hist.Bin {
        var g: f64 = 0;
        var h: f64 = 0;
        for (grads) |p| {
            g += p.g;
            h += p.h;
        }
        return .{ .g = g, .h = h, .n = @intCast(grads.len) };
    }

    /// Grow one tree against `gradients` (indexed by original row id).
    /// Returns the tree; `leafSpans()` describes where its leaves' rows landed.
    pub fn grow(b: *Builder, gradients: []const hist.GradPair) !Tree {
        b.nodes.clearRetainingCapacity();
        b.queue.clearRetainingCapacity();
        b.leaves.clearRetainingCapacity();
        b.level_depth = -1;

        // --- row sampling: shuffle, then take a prefix ---
        for (b.rows, 0..) |*r, i| r.* = @intCast(i);
        const n_all = b.ds.n_rows;
        if (b.cfg.subsample < 1.0) {
            const r = b.rng.random();
            var k: usize = @intFromFloat(@round(@as(f32, @floatFromInt(n_all)) * b.cfg.subsample));
            k = std.math.clamp(k, 1, n_all);
            var i: usize = 0;
            while (i < k) : (i += 1) {
                const j = i + r.uintLessThan(usize, n_all - i);
                std.mem.swap(u32, &b.rows[i], &b.rows[j]);
            }
            b.n_active = k;
        } else {
            b.n_active = n_all;
        }

        // One gather for the whole tree.
        for (0..b.n_active) |i| b.grads[i] = gradients[b.rows[i]];

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
        const root_total = totalOf(b.grads[0..b.n_active]);
        const root_feats = b.featuresFor(0);
        hist.build(
            b.pool,
            &b.bank,
            b.ds,
            b.rows[0..b.n_active],
            b.grads[0..b.n_active],
            root_feats,
            b.slot(root_slot),
        );
        const root_split = hist.bestSplit(&b.bank, b.slot(root_slot), b.ds, root_feats, root_total, p);
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

            const mid = b.partition(w.start, w.end, w.split);
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

            const child_feats = b.featuresFor(w.depth + 1);
            const slot_l = b.takeSlot();
            const slot_r = b.takeSlot();

            // Accumulate the smaller side; derive the larger by subtraction.
            const n_left = mid - w.start;
            const n_right = w.end - mid;
            if (n_left <= n_right) {
                hist.build(b.pool, &b.bank, b.ds, b.rows[w.start..mid], b.grads[w.start..mid], child_feats, b.slot(slot_l));
                hist.subtract(&b.bank, b.slot(slot_r), b.slot(w.slot), b.slot(slot_l), child_feats);
            } else {
                hist.build(b.pool, &b.bank, b.ds, b.rows[mid..w.end], b.grads[mid..w.end], child_feats, b.slot(slot_r));
                hist.subtract(&b.bank, b.slot(slot_l), b.slot(w.slot), b.slot(slot_r), child_feats);
            }
            b.giveSlot(w.slot);

            try b.queue.append(b.gpa, .{
                .node = li,
                .start = w.start,
                .end = mid,
                .depth = w.depth + 1,
                .total = w.split.left,
                .slot = slot_l,
                .split = hist.bestSplit(&b.bank, b.slot(slot_l), b.ds, child_feats, w.split.left, p),
            });
            try b.queue.append(b.gpa, .{
                .node = ri,
                .start = mid,
                .end = w.end,
                .depth = w.depth + 1,
                .total = w.split.right,
                .slot = slot_r,
                .split = hist.bestSplit(&b.bank, b.slot(slot_r), b.ds, child_feats, w.split.right, p),
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
