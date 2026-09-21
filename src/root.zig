//! zmodels — gradient-boosted decision trees for binned tabular data.

pub const config = @import("config.zig");
pub const data = @import("data.zig");
pub const hist = @import("hist.zig");
pub const metric = @import("metric.zig");
pub const pool = @import("pool.zig");
pub const tree = @import("tree.zig");
pub const booster = @import("booster.zig");
pub const forest = @import("forest.zig");
pub const linear = @import("linear.zig");

test {
    _ = @import("model_test.zig");
    _ = @import("sweep_test.zig");
    @import("std").testing.refAllDecls(@This());
}
