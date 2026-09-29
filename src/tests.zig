// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Test root. Tests live in src/test/, not beside the code they test.

const std = @import("std");

test {
    // The tests.
    _ = @import("test/booster_test.zig");
    _ = @import("test/csv_test.zig");
    _ = @import("test/cv_test.zig");
    _ = @import("test/linear_test.zig");
    _ = @import("test/metric_test.zig");
    _ = @import("test/model_test.zig");
    _ = @import("test/pool_test.zig");
    _ = @import("test/sweep_test.zig");
    _ = @import("test/tune_test.zig");

    // Compile coverage: every module, so none escapes the test build.
    // Add a new module here as well as in root.zig. The CLI (src/cli/) is
    // compiled by `zig build` and `zig build check`.
    std.testing.refAllDecls(@import("booster.zig"));
    std.testing.refAllDecls(@import("config.zig"));
    std.testing.refAllDecls(@import("csv.zig"));
    std.testing.refAllDecls(@import("cv.zig"));
    std.testing.refAllDecls(@import("data.zig"));
    std.testing.refAllDecls(@import("forest.zig"));
    std.testing.refAllDecls(@import("fitted.zig"));
    std.testing.refAllDecls(@import("hist.zig"));
    std.testing.refAllDecls(@import("split.zig"));
    std.testing.refAllDecls(@import("linear.zig"));
    std.testing.refAllDecls(@import("metric.zig"));
    std.testing.refAllDecls(@import("model.zig"));
    std.testing.refAllDecls(@import("objective.zig"));
    std.testing.refAllDecls(@import("pool.zig"));
    std.testing.refAllDecls(@import("prof.zig"));
    std.testing.refAllDecls(@import("root.zig"));
    std.testing.refAllDecls(@import("tree.zig"));
    std.testing.refAllDecls(@import("tune.zig"));
}
