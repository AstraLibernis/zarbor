// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! zarbor — the CLI: fit gradient-boosted trees, a random forest or a linear
//! model on a CSV (`train`, the default command), and `predict`, `blend`,
//! `info`, `profile`, `cv`, `tune`.
//!
//! For train, cv and tune every field of `Config` is exposed as `--field=value`
//! by comptime reflection (`config.applyFlag`), so those flags and the struct
//! cannot drift apart.

const std = @import("std");
const train = @import("train.zig");
const score = @import("score.zig");
const info = @import("info.zig");
const profile = @import("profile.zig");
const cv = @import("cv.zig");
const tune = @import("tune.zig");
const explain = @import("explain.zig");

pub const usage =
    \\usage: zarbor <train.csv> --label=<column> [options]
    \\
    \\  --algo=NAME         gbdt | random_forest | linear   (default gbdt)
    \\  --label=NAME        target column (required)
    \\  --pos-label=NAME    class of a string target to encode as 1
    \\                      (default: classes sorted, so "No"<"Yes" -> Yes=1)
    \\  --drop=NAME         exclude a column; repeatable
    \\  --valid-frac=F      fraction held out for validation (default 0.2)
    \\  --split-seed=N      seed for the validation split (default 1)
    \\  --max-bytes=N       CSV size cap in bytes (default 1<<31)
    \\  --split-col=NAME    column assigning rows to train(<0.5)/valid(>=0.5);
    \\                      overrides --valid-frac, and is dropped as a feature
    \\  --save=FILE         write the trained model to FILE
    \\
    \\other commands:
    \\  zarbor predict <data.csv> --model=M.zm [--out=P.csv] [--id-col=id]
    \\  zarbor blend   <data.csv> --models=A.zm,B.zm [--weights=1,2] [--out=P.csv]
    \\  zarbor info    --model=M.zm
    \\  zarbor profile <data.csv>   what is in the file, before any model
    \\  zarbor cv      <train.csv> --label=<column> [--folds=5]
    \\  zarbor tune    <train.csv> --label=<column> [--search=random]
    \\  zarbor explain [data.csv] --model=M.zm   importance and SHAP values
    \\
    \\predict and blend bin the new data with the schema stored in the model,
    \\so categorical levels map to the same bins they did in training. Pass
    \\--label=NAME as well to score against a labelled holdout; it is decoded
    \\with the model's own class order, not the holdout file's.
    \\
    \\Any Config field is also a flag, e.g.:
    \\  --n_rounds=800 --learning_rate=0.05 --max_depth=7 --lambda=2.0
    \\  --grow_policy=lossguide --max_leaves=64 --subsample=0.8
    \\  --colsample_bytree=0.8 --early_stopping_rounds=50 --n_threads=16
    \\
    \\XGBoost-style is the default; LightGBM-style is:
    \\  --grow_policy=lossguide --max_leaves=64 --sampling=goss
    \\
    \\Each algo sets its own defaults for anything you do not pass:
    \\  --algo=random_forest   bagged, unshrunk, 1024-leaf trees, sqrt(p)/split
    \\  --algo=linear          L-BFGS + L1/L2 on the binned design matrix
    \\
;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var out_buf: [16 * 1024]u8 = undefined;
    var fw = std.Io.File.stdout().writerStreaming(io, &out_buf);
    const out = &fw.interface;

    // Subcommand dispatch. A bare CSV path still means "train", so every
    // existing invocation keeps working.
    var probe = std.process.Args.Iterator.init(init.minimal.args);
    _ = probe.skip();
    if (probe.next()) |first| {
        if (std.mem.eql(u8, first, "predict")) return score.run(init, gpa, out, .predict);
        if (std.mem.eql(u8, first, "blend")) return score.run(init, gpa, out, .blend);
        if (std.mem.eql(u8, first, "info")) return info.run(init, gpa, out);
        if (std.mem.eql(u8, first, "profile")) return profile.run(init, gpa, out);
        if (std.mem.eql(u8, first, "cv")) return cv.run(init, gpa, out);
        if (std.mem.eql(u8, first, "tune")) return tune.run(init, gpa, out);
        if (std.mem.eql(u8, first, "explain")) return explain.run(init, gpa, out);
    }
    return train.run(init, gpa, out);
}
