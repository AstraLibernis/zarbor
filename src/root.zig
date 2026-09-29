// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! zarbor — gradient-boosted decision trees for binned tabular data.

pub const config = @import("config.zig");
pub const csv = @import("csv.zig");
pub const data = @import("data.zig");
pub const hist = @import("hist.zig");
pub const split = @import("split.zig");
pub const metric = @import("metric.zig");
pub const pool = @import("pool.zig");
pub const prof = @import("prof.zig");
pub const tree = @import("tree.zig");
pub const booster = @import("booster.zig");
pub const forest = @import("forest.zig");
pub const linear = @import("linear.zig");
pub const objective = @import("objective.zig");
pub const model = @import("model.zig");
pub const fitted = @import("fitted.zig");
pub const cv = @import("cv.zig");
pub const tune = @import("tune.zig");
