//! zmodels — gradient-boosted decision trees for binned tabular data.

pub const config = @import("config.zig");
pub const csv = @import("csv.zig");
pub const data = @import("data.zig");
pub const hist = @import("hist.zig");
pub const metric = @import("metric.zig");
pub const pool = @import("pool.zig");
pub const prof = @import("prof.zig");
pub const tree = @import("tree.zig");
pub const booster = @import("booster.zig");
pub const forest = @import("forest.zig");
pub const linear = @import("linear.zig");
pub const model = @import("model.zig");

test {
    _ = @import("cv.zig");
    _ = @import("tune.zig");
    _ = @import("model_test.zig");
    _ = @import("sweep_test.zig");
    @import("std").testing.refAllDecls(@This());
}
