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
    /// GOSS row selection, which happens in the boosting loop rather than in
    /// the tree builder.
    goss_select,
    select_rows,
    hist_build,
    /// Sub-phases of `hist_build`: clearing the private slots, accumulating,
    /// and reducing them. Inside `hist_build`, not additional to it.
    hist_clear,
    hist_accum,
    hist_reduce,
    hist_subtract,
    best_split,
    partition,
    /// Sub-phases of the parallel partition, so the three passes can be sized
    /// against each other. They are inside `partition`, not additional to it.
    part_count,
    part_scatter,
    part_copy,
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

/// `part_*` time is *inside* `partition`, so it must not be added to the
/// total or counted as its own percentage.
fn nested(p: Phase) bool {
    return switch (p) {
        .part_count, .part_scatter, .part_copy => true,
        .hist_clear, .hist_accum, .hist_reduce => true,
        else => false,
    };
}

pub fn report(w: *std.Io.Writer) !void {
    var sum: u64 = 0;
    inline for (@typeInfo(Phase).@"enum".fields) |f| {
        if (comptime !nested(@field(Phase, f.name))) sum += totals[f.value];
    }
    if (sum == 0) return;
    try w.print("\nphase breakdown (wall, main thread)\n", .{});
    inline for (@typeInfo(Phase).@"enum".fields) |f| {
        const t = totals[f.value];
        if (t != 0) {
            const is_nested = comptime nested(@field(Phase, f.name));
            try w.print("  {s}{s:<14} {d:>7} ms  {d:>5.1}%{s}\n", .{
                if (is_nested) "  " else "",
                f.name,
                t / 1_000_000,
                100.0 * @as(f64, @floatFromInt(t)) / @as(f64, @floatFromInt(sum)),
                if (is_nested) "  (nested)" else "",
            });
        }
    }
    try w.print("  {s:<14} {d:>7} ms\n", .{ "measured", sum / 1_000_000 });
    try w.flush();
}
