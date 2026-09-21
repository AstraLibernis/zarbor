//! A persistent parallel-for pool.
//!
//! Zig 0.16 removed `std.Thread.Pool`, `std.Thread.Mutex` and the futex layer;
//! the replacement lives behind the new `std.Io` interface, which is built for
//! blocking IO rather than for compute fan-out. A boosting round issues a few
//! hundred barriers over work items that take tens of microseconds, so the
//! cost that matters is wake-up latency, not scheduling generality. This pool
//! therefore parks workers on a spin-then-yield loop over an epoch counter and
//! never makes a syscall on the fast path.
//!
//! Work is handed out as a shared cursor rather than a static partition, so a
//! thread descheduled by the host cannot stall the barrier behind it.

const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;

/// Parking needs a futex. Zig 0.16 has no portable one — `std.Thread.Mutex`,
/// `Condition` and the futex layer are all gone — but the raw Linux syscall is
/// right there, and this only ever runs on Linux. Anywhere else falls back to
/// the yield loop, which is correct, merely wasteful.
const can_park = builtin.os.tag == .linux;

const wait_op: linux.FUTEX_OP = .{ .cmd = .WAIT, .private = true };
const wake_op: linux.FUTEX_OP = .{ .cmd = .WAKE, .private = true };

/// Sleeps until `v` differs from `expect`. Returns immediately if it already
/// does, which is what closes the race against a concurrent publisher.
fn park(v: *std.atomic.Value(u32), expect: u32) void {
    _ = linux.futex_4arg(&v.raw, wait_op, expect, null);
}

fn unparkAll(v: *std.atomic.Value(u32)) void {
    _ = linux.futex_3arg(&v.raw, wake_op, std.math.maxInt(i32));
}

pub const cache_line = std.atomic.cache_line;

/// Called with a half-open row range and the id of the worker running it.
/// The worker id lets a task index per-thread scratch without locking.
pub const TaskFn = *const fn (ctx: *anyopaque, worker: usize, begin: usize, end: usize) void;

pub const Pool = struct {
    gpa: std.mem.Allocator,
    /// Spawned helpers. The submitting thread also participates, as worker 0,
    /// so total concurrency is `threads.len + 1`.
    threads: []std.Thread,

    // --- job description; published by the release store to `epoch` --------
    task: TaskFn = undefined,
    ctx: *anyopaque = undefined,
    total: usize = 0,
    chunk: usize = 1,

    // Each of these is written by every worker on the hot path. Keeping them
    // on separate cache lines is the difference between a shared counter and
    // a contended one.
    cursor: std.atomic.Value(usize) align(cache_line) = .init(0),
    done: std.atomic.Value(usize) align(cache_line) = .init(0),
    epoch: std.atomic.Value(u32) align(cache_line) = .init(0),
    quit: std.atomic.Value(bool) align(cache_line) = .init(false),
    /// Workers currently asleep on `epoch`. Lets the publisher skip the wake
    /// syscall entirely in the common case, where a burst of back-to-back
    /// `parallelFor` calls keeps every worker spinning and nobody parks.
    parked: std.atomic.Value(u32) align(cache_line) = .init(0),

    /// `n_threads` of 0 means one worker per logical core.
    ///
    /// Heap-allocated because every worker holds a `*Pool` for its whole life:
    /// the pool's address must outlive this call, which a by-value return
    /// cannot promise.
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

    /// Run `task` over `[0, total)`, returning once every element is done.
    ///
    /// `min_chunk` is the smallest range worth handing to another core; below
    /// it the barrier costs more than the work, so the caller's thread just
    /// runs the whole range inline.
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

        // Several chunks per worker so a slow core cannot hold the barrier,
        // but few enough that the atomic cursor is not itself the bottleneck.
        var chunk = (total + workers * 4 - 1) / (workers * 4);
        if (chunk < min_chunk) chunk = min_chunk;

        p.task = task;
        p.ctx = ctx;
        p.total = total;
        p.chunk = chunk;
        p.cursor.store(0, .monotonic);
        p.done.store(0, .monotonic);

        // Release: everything above is visible to any worker that sees the
        // new epoch with an acquire load.
        _ = p.epoch.fetchAdd(1, .release);
        if (can_park and p.parked.load(.acquire) != 0) unparkAll(&p.epoch);

        drain(p, 0);

        var spins: u32 = 0;
        while (p.done.load(.acquire) < p.threads.len) {
            spins +%= 1;
            if (spins < spin_budget) std.atomic.spinLoopHint() else std.Thread.yield() catch {};
        }
    }
};

/// How long to spin before yielding. Tuned to comfortably cover a histogram
/// chunk; past that the thread is waiting on something real and the
/// scheduler should have the core back.
const spin_budget: u32 = 8192;

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
                std.Thread.yield() catch {};
                continue;
            }

            // Past the spin budget this thread is waiting on something real —
            // usually a serial stretch of tree building — so give the core
            // back instead of burning it.
            //
            // Announce the park *before* re-reading the epoch. A publisher
            // that bumps the epoch after our re-check will see `parked` and
            // wake us; one that bumps before it makes the futex compare fail,
            // so the sleep returns immediately. Either order is safe, which is
            // the whole point of doing it in this sequence.
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

// ---------------------------------------------------------------------------

const testing = std.testing;

test "parallelFor covers every index exactly once" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 0);
    defer pool.deinit();

    const n = 100_000;
    const marks = try gpa.alloc(u8, n);
    defer gpa.free(marks);
    @memset(marks, 0);

    const Ctx = struct {
        marks: []u8,
        fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
            _ = worker;
            const self: *@This() = @ptrCast(@alignCast(ctx));
            for (self.marks[begin..end]) |*m| m.* += 1;
        }
    };
    var ctx = Ctx{ .marks = marks };
    pool.parallelFor(n, &ctx, Ctx.run, 1);

    for (marks) |m| try testing.expectEqual(@as(u8, 1), m);
}

test "workers park and wake correctly across an idle gap" {
    // The fast tests never leave a gap long enough for a worker to exceed the
    // spin budget, so they exercise the spin path only. This one stalls the
    // main thread past that budget, forcing every helper onto the futex, and
    // then checks a subsequent round still completes — which is where a lost
    // wakeup would hang forever.
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 4);
    defer pool.deinit();

    const n = 10_000;
    const vals = try gpa.alloc(u32, n);
    defer gpa.free(vals);
    @memset(vals, 0);

    const Ctx = struct {
        vals: []u32,
        fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
            _ = worker;
            const self: *@This() = @ptrCast(@alignCast(ctx));
            for (self.vals[begin..end]) |*v| v.* += 1;
        }
    };
    var ctx = Ctx{ .vals = vals };

    for (0..3) |_| {
        pool.parallelFor(n, &ctx, Ctx.run, 1);
        // Stall well past `spin_budget` so the helpers actually sleep.
        var sink: u64 = 0;
        for (0..spin_budget * 64) |i| sink +%= i;
        std.mem.doNotOptimizeAway(sink);
    }

    for (vals) |v| try testing.expectEqual(@as(u32, 3), v);
}

test "parallelFor is reusable across rounds" {
    const gpa = testing.allocator;
    const pool = try Pool.init(gpa, 4);
    defer pool.deinit();

    const n = 10_000;
    const vals = try gpa.alloc(u32, n);
    defer gpa.free(vals);
    @memset(vals, 0);

    const Ctx = struct {
        vals: []u32,
        fn run(ctx: *anyopaque, worker: usize, begin: usize, end: usize) void {
            _ = worker;
            const self: *@This() = @ptrCast(@alignCast(ctx));
            for (self.vals[begin..end]) |*v| v.* += 1;
        }
    };
    var ctx = Ctx{ .vals = vals };
    for (0..50) |_| pool.parallelFor(n, &ctx, Ctx.run, 1);

    for (vals) |v| try testing.expectEqual(@as(u32, 50), v);
}
