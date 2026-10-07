// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Hyperparameter surface for every model in zarbor.
//! Names follow XGBoost where an equivalent exists, so published tunings transfer
//! without a translation table; LightGBM-only knobs keep LightGBM's names. Where the
//! same name still means something slightly different (`max_bin` counts the missing
//! bin, `subsample` draws exactly k rows, `colsample_*` rounds), docs/archive/parity.md says so.

const std = @import("std");
const data = @import("data.zig");
const booster = @import("booster.zig");
const forest = @import("forest.zig");
const linear = @import("linear.zig");
const Objective = @import("objective.zig").Objective;

/// Which model to fit; all share `data.zig` binning, missing-value and categorical handling.
pub const Algo = enum {
    /// Gradient-boosted trees; `grow_policy` selects XGBoost- or LightGBM-style growth.
    gbdt,
    /// Bagged unshrunk trees, averaged. No boosting.
    random_forest,
    /// Regularised linear or logistic regression on the binned design matrix.
    linear,
};

fn parseInto(comptime T: type, val: []const u8) !T {
    return switch (@typeInfo(T)) {
        .int => try std.fmt.parseInt(T, val, 10),
        .float => try std.fmt.parseFloat(T, val),
        .bool => std.mem.eql(u8, val, "true") or std.mem.eql(u8, val, "1"),
        .@"enum" => std.meta.stringToEnum(T, val) orelse error.UnknownEnumValue,
        .optional => |o| try parseInto(o.child, val),
        else => @compileError("config field type not parseable: " ++ @typeName(T)),
    };
}

/// Set a field by name from its string form in every group that has it (`lambda`
/// reaches both ensembles' trees and the linear model). False when `key` names no
/// field, so a caller can fall through to its own flags. train, cv and tune parse
/// flags through this, so their flags cannot drift from the structs, and a tuner can
/// pass any field through; predict and blend parse their few flags by hand.
pub fn applyFlag(cfg: *Config, key: []const u8, val: []const u8) !bool {
    return applyIn(cfg, key, val);
}

fn applyIn(ptr: anytype, key: []const u8, val: []const u8) !bool {
    var found = false;
    inline for (@typeInfo(@TypeOf(ptr.*)).@"struct".fields) |f| {
        if (@typeInfo(f.type) == .@"struct") {
            if (try applyIn(&@field(ptr, f.name), key, val)) found = true;
        } else if (std.mem.eql(u8, f.name, key)) {
            @field(ptr, f.name) = try parseInto(f.type, val);
            found = true;
        }
    }
    return found;
}

/// Whether `key` names a Config field at all, without setting it.
pub fn hasField(key: []const u8) bool {
    return hasIn(Config, key);
}

fn hasIn(comptime T: type, key: []const u8) bool {
    inline for (@typeInfo(T).@"struct".fields) |f| {
        if (@typeInfo(f.type) == .@"struct") {
            if (hasIn(f.type, key)) return true;
        } else if (std.mem.eql(u8, f.name, key)) return true;
    }
    return false;
}

/// Every setting, grouped by who reads it. Each model's group carries that
/// model's own defaults; a flag sets its name in every group that has it.
pub const Config = struct {
    algo: Algo = .gbdt,
    /// Worker threads. 0 means "one per logical core".
    n_threads: u32 = 0,
    bin: data.BinParams = .{},
    gbdt: booster.Params = .{},
    random_forest: forest.Params = .{},
    linear: linear.Params = .{},

    /// A Config from flat `name = value` pairs, set as flags would be:
    /// `Config.from(.{ .algo = .linear, .lambda = 2 })`.
    pub fn from(flat: anytype) Config {
        var c: Config = .{};
        inline for (@typeInfo(@TypeOf(flat)).@"struct".fields) |f| c.set(f.name, @field(flat, f.name));
        return c;
    }

    /// Set `name` in every group that has it, as `applyFlag` does.
    pub fn set(c: *Config, comptime name: []const u8, value: anytype) void {
        comptime if (!hasIn(Config, name)) @compileError("no config field named " ++ name);
        setIn(c, name, value);
    }

    fn setIn(ptr: anytype, comptime name: []const u8, value: anytype) void {
        inline for (@typeInfo(@TypeOf(ptr.*)).@"struct".fields) |f| {
            if (comptime @typeInfo(f.type) == .@"struct") {
                if (comptime hasIn(f.type, name)) setIn(&@field(ptr, f.name), name, value);
            } else if (comptime std.mem.eql(u8, f.name, name)) {
                @field(ptr, f.name) = value;
            }
        }
    }

    pub fn objective(c: Config) Objective {
        return switch (c.algo) {
            .gbdt => c.gbdt.objective,
            .random_forest => c.random_forest.objective,
            .linear => c.linear.objective,
        };
    }

    /// Standard name of the fitted model, so the run says what it is. Needed for
    /// `linear`, where `--objective` silently switches between two textbook models.
    pub fn modelName(c: Config) []const u8 {
        const l = c.linear;
        return switch (c.algo) {
            .gbdt => switch (c.gbdt.tree.grow_policy) {
                .depthwise => "gradient-boosted trees (depthwise, XGBoost-style)",
                .lossguide => "gradient-boosted trees (leafwise, LightGBM-style)",
                .symmetric => "gradient-boosted trees (symmetric, CatBoost-style)",
            },
            .random_forest => "random forest (bagged unshrunk trees)",
            .linear => switch (l.objective) {
                .logistic => if (l.alpha > 0 and l.lambda > 0)
                    "logistic regression (elastic net)"
                else if (l.alpha > 0)
                    "logistic regression (L1 / lasso)"
                else if (l.lambda > 0)
                    "logistic regression (L2 / ridge)"
                else
                    "logistic regression (unpenalised)",
                .squared_error => if (l.alpha > 0 and l.lambda > 0)
                    "linear regression (elastic net)"
                else if (l.alpha > 0)
                    "linear regression (L1 / lasso)"
                else if (l.lambda > 0)
                    "linear regression (L2 / ridge)"
                else
                    "linear regression (ordinary least squares)",
            },
        };
    }

    /// Breiman's per-split feature default (sqrt(p) classification, p/3 regression);
    /// needs the feature count, so set after binning. An explicit flag wins.
    pub fn applyForestFeatureDefault(
        c: *Config,
        n_features: usize,
        explicit: []const []const u8,
    ) void {
        if (c.algo != .random_forest) return;
        for (explicit) |e| if (std.mem.eql(u8, e, "colsample_bynode")) return;
        if (n_features == 0) return;
        const p_f: f32 = @floatFromInt(n_features);
        const want: f32 = switch (c.random_forest.objective) {
            .logistic => @sqrt(p_f),
            .squared_error => p_f / 3.0,
        };
        c.random_forest.tree.colsample_bynode = std.math.clamp(want / p_f, 1.0 / p_f, 1.0);
    }

    /// Binning plus the chosen model's own checks.
    pub fn validate(c: Config) !void {
        try c.bin.validate();
        switch (c.algo) {
            .gbdt => try c.gbdt.validate(),
            .random_forest => try c.random_forest.validate(),
            .linear => try c.linear.validate(),
        }
    }
};
