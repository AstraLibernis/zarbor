// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! A persistent parallel-for pool. Zig 0.16 dropped `std.Thread.Pool`/`Mutex`
//! for `std.Io`, built for blocking IO. A round issues hundreds of barriers
//! over tens-of-microsecond items, so wake latency rules: workers spin then
//! yield on an epoch counter, no syscall on the fast path. A shared cursor, not
//! a static partition, so a descheduled thread cannot stall the barrier.

const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;

/// Zig 0.16 has no portable futex, so raw Linux syscall; elsewhere the yield
/// loop, correct but wasteful.
const can_park = builtin.os.tag == .linux;

const wait_op: linux.FUTEX_OP = .{ .cmd = .WAIT, .private = true };
const wake_op: linux.FUTEX_OP = .{ .cmd = .WAKE, .private = true };

/// Sleeps until `v` != `expect`; returns at once if already so (closes the publish race).
fn park(v: *std.atomic.Value(u32), expect: u32) void {
    _ = linux.futex_4arg(&v.raw, wait_op, expect, null);
}

fn unparkAll(v: *std.atomic.Value(u32)) void {
    _ = linux.futex_3arg(&v.raw, wake_op, std.math.maxInt(i32));
}

pub const cache_line = std.atomic.cache_line;

/// Half-open range plus worker id, for lock-free per-thread scratch.
pub const TaskFn = *const fn (ctx: *anyopaque, worker: usize, begin: usize, end: usize) void;

pub const Pool = struct {
    gpa: std.mem.Allocator,
    /// Spawned helpers; the submitter is worker 0, so concurrency is `threads.len + 1`.
    threads: []std.Thread,

    // --- job description; published by the release store to `epoch` --------
    task: TaskFn = undefined,
    ctx: *anyopaque = undefined,
    total: usize = 0,
    chunk: usize = 1,

    // Hot-path writes by every worker: separate cache lines avoid contention.
    cursor: std.atomic.Value(usize) align(cache_line) = .init(0),
    done: std.atomic.Value(usize) align(cache_line) = .init(0),
    epoch: std.atomic.Value(u32) align(cache_line) = .init(0),
    quit: std.atomic.Value(bool) align(cache_line) = .init(false),
    /// Workers asleep on `epoch`; lets the publisher skip the wake syscall when
    /// back-to-back `parallelFor` calls keep everyone spinning.
    parked: std.atomic.Value(u32) align(cache_line) = .init(0),

    /// `n_threads` 0 = one per logical core. Heap-allocated: workers hold a
    /// `*Pool` for life, so its address must be stable.
    pub fn init(gpa: std.mem.Allocator, n_threads: u32) !*Pool {
        const cores = std.Thread.getCpuCount() catch 1;
        const want = if (n_threads == 0) cores else n_threads;
        const helpers = if (want <= 1) 0 else want - 1;

        const p = try gpa.create(Pool);
        errdefer gpa.destroy(p);

        p.* = .{ .gpa = gpa, .threads = try gpa.alloc(std.Thread, helpers) };
        errdefer gpa.free(p.threads);

        // The pool must be fully initialised before the first worker reads it.
        var spawned: usize = 0;
        errdefer {
            p.quit.store(true, .release);
            _ = p.epoch.fetchAdd(1, .release);
            if (can_park) unparkAll(&p.epoch);
            for (p.threads[0..spawned]) |t| t.join();
        }
        while (spawned < helpers) : (spawned += 1) {
            p.threads[spawned] = try std.Thread.spawn(.{}, workerMain, .{ p, spawned + 1 });
        }
        return p;
    }

    pub fn deinit(p: *Pool) void {
        const gpa = p.gpa;
        const threads = p.threads;
        p.quit.store(true, .release);
        // Bump the epoch so workers parked on it re-check `quit`.
        _ = p.epoch.fetchAdd(1, .release);
        if (can_park) unparkAll(&p.epoch);
        for (threads) |t| t.join();
        gpa.free(threads);
        gpa.destroy(p);
    }

    pub fn workerCount(p: *const Pool) usize {
        return p.threads.len + 1;
    }

    /// Run `task` over `[0, total)`. Under `min_chunk` the barrier costs more: run inline.
    pub fn parallelFor(
        p: *Pool,
        total: usize,
        ctx: *anyopaque,
        task: TaskFn,
        min_chunk: usize,
    ) void {
        if (total == 0) return;
        const workers = p.workerCount();
        if (workers == 1 or total <= min_chunk) {
            task(ctx, 0, 0, total);
            return;
        }

        // Several chunks per worker so a slow core cannot hold the barrier;
        // few enough that the cursor is not the bottleneck.
        var chunk = (total + workers * 4 - 1) / (workers * 4);
        if (chunk < min_chunk) chunk = min_chunk;

        p.task = task;
        p.ctx = ctx;
        p.total = total;
        p.chunk = chunk;
        p.cursor.store(0, .monotonic);
        p.done.store(0, .monotonic);

        // Release: visible to any worker acquiring the new epoch.
        _ = p.epoch.fetchAdd(1, .release);
        if (can_park and p.parked.load(.acquire) != 0) unparkAll(&p.epoch);

        drain(p, 0);

        var spins: u32 = 0;
        while (p.done.load(.acquire) < p.threads.len) {
            spins +%= 1;
            if (spins < spin_budget) std.atomic.spinLoopHint() else std.Thread.yield() catch {}; // zsnag:ok — a failed yield just means spin again
        }
    }
};

/// Spin before yielding; covers a histogram chunk, past which the core goes back.
pub const spin_budget: u32 = 8192;

fn drain(p: *Pool, worker: usize) void {
    while (true) {
        const begin = p.cursor.fetchAdd(p.chunk, .monotonic);
        if (begin >= p.total) break;
        const end = @min(begin + p.chunk, p.total);
        p.task(p.ctx, worker, begin, end);
    }
}

fn workerMain(p: *Pool, id: usize) void {
    var seen: u32 = 0;
    while (true) {
        var spins: u32 = 0;
        while (true) {
            if (p.quit.load(.acquire)) return;
            const e = p.epoch.load(.acquire);
            if (e != seen) {
                seen = e;
                break;
            }
            spins +%= 1;
            if (spins < spin_budget) {
                std.atomic.spinLoopHint();
                continue;
            }
            if (!can_park) {
                std.Thread.yield() catch {}; // zsnag:ok — a failed yield just means spin again
                continue;
            }

            // Past the spin budget (usually serial tree building): park.
            // Announce *before* re-reading the epoch: a later bump sees
            // `parked` and wakes us; an earlier one fails the futex compare.
            _ = p.parked.fetchAdd(1, .acq_rel);
            if (p.epoch.load(.acquire) == e and !p.quit.load(.acquire)) park(&p.epoch, e);
            _ = p.parked.fetchSub(1, .acq_rel);
            spins = 0;
        }
        if (p.quit.load(.acquire)) return;
        drain(p, id);
        _ = p.done.fetchAdd(1, .release);
    }
}
