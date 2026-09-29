// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{ .name = "zgbdt", .root_module = mod });
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    run.step.dependOn(b.getInstallStep());
    if (b.args) |args| run.addArgs(args);
    b.step("run", "Train/predict with the zarbor GBDT").dependOn(&run.step);

    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.step("test", "Run unit tests").dependOn(&b.addRunArtifact(tests).step);

    // Cost prototype for widening the bin type; see docs/wide-categoricals.md.
    const lib = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    const wc = b.addExecutable(.{ .name = "widecat", .root_module = b.createModule(.{
        .root_source_file = b.path("bench/widecat.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "zarbor", .module = lib }},
    }) });
    const sz = b.addExecutable(.{ .name = "sz", .root_module = b.createModule(.{
        .root_source_file = b.path("bench/sz.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "zarbor", .module = lib }},
    }) });
    b.step("bench-sz", "Print hot struct sizes").dependOn(&b.addRunArtifact(sz).step);

    b.step("bench-widecat", "Measure what a wide categorical column costs")
        .dependOn(&b.addRunArtifact(wc).step);

    // Benchmark glue in Zig (bench/tools.zig; replaces the non-baseline Python/shell).
    const tools_paths = b.addOptions();
    tools_paths.addOption([]const u8, "root", b.pathFromRoot("."));
    const tools_mod = b.createModule(.{
        .root_source_file = b.path("bench/tools.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "zarbor", .module = lib }},
    });
    tools_mod.addOptions("tools_paths", tools_paths);
    const tools = b.addExecutable(.{ .name = "tools", .root_module = tools_mod });
    b.installArtifact(tools);
    const run_tools = b.addRunArtifact(tools);
    if (b.args) |args| run_tools.addArgs(args);
    b.step("tools", "Benchmark glue: `zig build tools -- summarise|figure|grid|controls|solver-stress ...`").dependOn(&run_tools.step);

    // Standalone microbenchmarks, each answering one recorded question (see its header).
    const pool_mod = b.createModule(.{ .root_source_file = b.path("src/pool.zig"), .target = target, .optimize = optimize });
    const micro = [_]struct { step: []const u8, file: []const u8, desc: []const u8, libc: bool = false, pool: bool = false }{
        .{ .step = "bench-barrier", .file = "bench/barrier.zig", .desc = "What one parallel region costs", .pool = true },
        .{ .step = "bench-binshape", .file = "bench/binshape.zig", .desc = "Histogram bin shape at real feature widths" },
        .{ .step = "bench-exp", .file = "bench/expbench.zig", .desc = "sigmoid: @exp vs glibc expf vs vector 2^x", .libc = true },
        .{ .step = "bench-hist", .file = "bench/histbench.zig", .desc = "Histogram accumulation kernel variants" },
        .{ .step = "bench-layout", .file = "bench/layoutbench.zig", .desc = "Uniform stride vs packed per-feature offsets" },
    };
    for (micro) |m| {
        const micro_mod = b.createModule(.{ .root_source_file = b.path(m.file), .target = target, .optimize = optimize, .link_libc = m.libc });
        if (m.pool) micro_mod.addImport("pool", pool_mod);
        const exe_m = b.addExecutable(.{ .name = m.step, .root_module = micro_mod });
        b.step(m.step, m.desc).dependOn(&b.addRunArtifact(exe_m).step);
    }
}
