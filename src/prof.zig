//! Phase timers for tree building.
//!
//! Exists because `perf` cannot run in this sandbox (no CAP_PERFMON, and
//! `perf_event_paranoid` is host-owned), and because two successive guesses at
//! the bottleneck — the serial partition, then histogram cache residency —
//! were each only half right or plain wrong. Guessing is more expensive than
//! measuring.
//!
//! Timers are read by the submitting thread around whole parallel regions, so
//! they measure wall time per phase, not CPU time, and never appear inside a
//! worker's inner loop.

const std = @import("std");
const linux = std.os.linux;

pub const Phase = enum {
    select_rows,
    gather,
    hist_build,
    hist_subtract,
    best_split,
    partition,
    grad,
    apply,
    valid_predict,
    valid_metric,
    other,
};

pub inline fn now() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// Off unless `--profile` is passed, so the timestamp calls stay out of the
/// way of an ordinary run.
pub var enabled: bool = false;

var totals: [@typeInfo(Phase).@"enum".fields.len]u64 = @splat(0);

pub inline fn start() u64 {
    return if (enabled) now() else 0;
}

pub inline fn stop(p: Phase, t0: u64) void {
    if (!enabled) return;
    totals[@intFromEnum(p)] += now() - t0;
}

pub fn reset() void {
    totals = @splat(0);
}

pub fn report(w: *std.Io.Writer) !void {
    var sum: u64 = 0;
    for (totals) |t| sum += t;
    if (sum == 0) return;
    try w.print("\nphase breakdown (wall, main thread)\n", .{});
    inline for (@typeInfo(Phase).@"enum".fields) |f| {
        const t = totals[f.value];
        if (t != 0) {
            try w.print("  {s:<14} {d:>7} ms  {d:>5.1}%\n", .{
                f.name,
                t / 1_000_000,
                100.0 * @as(f64, @floatFromInt(t)) / @as(f64, @floatFromInt(sum)),
            });
        }
    }
    try w.print("  {s:<14} {d:>7} ms\n", .{ "measured", sum / 1_000_000 });
    try w.flush();
}
