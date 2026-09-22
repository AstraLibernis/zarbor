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
    b.step("run", "Train/predict with the zmodels GBDT").dependOn(&run.step);

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
    b.step("bench-widecat", "Measure what a wide categorical column costs")
        .dependOn(&b.addRunArtifact(wc).step);
}
