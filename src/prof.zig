// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Phase timers for tree building. `perf` cannot run here (no CAP_PERFMON,
//! host-owned `perf_event_paranoid`), and two bottleneck guesses (serial
//! partition, histogram cache residency) were wrong or half right. Read by the
//! submitter around whole parallel regions: wall time per phase, never inside
//! a worker's inner loop.

const std = @import("std");
const linux = std.os.linux;

pub const Phase = enum {
    /// GOSS row selection, in the boosting loop, not the tree builder.
    goss_select,
    select_rows,
    hist_build,
    /// `hist_build` sub-phases (clear, accumulate, reduce); inside it, not added.
    hist_clear,
    hist_accum,
    hist_reduce,
    hist_subtract,
    best_split,
    partition,
    /// The partition's three passes, sized against each other; inside it, not added.
    part_count,
    part_scatter,
    part_copy,
    grad,
    apply,
    valid_predict,
    valid_metric,
    /// Symmetric (CatBoost-style) builder phases, in `symmetric.zig`.
    sym_init,
    sym_setup,
    sym_combo,
    sym_hist,
    sym_score,
    sym_split,
    sym_leaves,
    sym_folds,
    other,
};

pub inline fn now() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// Off unless `--profile`, keeping timestamp calls out of ordinary runs.
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

/// `part_*` is *inside* `partition`: not added to the total or given its own percentage.
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
