// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! zsift — public API surface for parsing delimited/tabular text (CSV by default;
//! the delimiter and quote are configurable). This file only re-exports; the
//! implementations live in focused modules under `core/`:
//!
//!   * `core/convert.zig` — `[]const u8 → T` field converters (the bottom layer)
//!   * `core/types.zig`  — `Options`, `Field`, `Error` (shared by every parser)
//!   * `core/scalar.zig` — `Parser`: lenient byte-at-a-time, in-memory
//!   * `core/simd.zig`   — `SimdParser` (pull) and `forEachField` (push), strict;
//!                         also the `classify` chunk-classification primitives
//!   * `core/stream.zig` — `streamReader` and the auto-selecting `parseReader`
//!   * `core/header.zig`/`record.zig`/`reader.zig` — opt-in typed layers over the
//!                         borrowed fields (named columns, a `Record` view, typed rows)
//!
//! Layering: convert → types → simd(classify) → {scalar, stream} →
//!           {header, record, reader} → this facade.

const types = @import("core/types.zig");

pub const Options = types.Options;
pub const Error = types.Error;
pub const Field = types.Field;

/// Opt-in `[]const u8 → T` field converters. Also reachable as `Field` methods
/// (`field.as(T)`, `field.asInt(T)`, …). See `core/convert.zig`.
pub const convert = @import("core/convert.zig");
pub const ConvertError = convert.ConvertError;

/// Zero-alloc name→column view over a header record. See `core/header.zig`.
pub const Header = @import("core/header.zig").Header;

/// Ergonomic borrowed view over one record's fields (`.at`/`.get`/`.as`/`.count`).
/// See `core/record.zig`.
pub const Record = @import("core/record.zig").Record;

/// Typed row reader: `reader(T).init(input, scratch, .{})` deserializes each record
/// into a struct `T` (positional by default; `.withHeader()` matches field names).
/// `readerWide(T, max_cols)` tolerates headered rows wider than `T`. See `core/reader.zig`.
const reader_mod = @import("core/reader.zig");
pub const reader = reader_mod.Reader;
pub const readerWide = reader_mod.ReaderWide;
pub const ReaderError = reader_mod.ReaderError;

/// Lenient scalar parser over an in-memory slice. See `core/scalar.zig`.
pub const Parser = @import("core/scalar.zig").Parser;

/// The vectorized fast-path parsers (well-formed RFC 4180 input). See `core/simd.zig`.
pub const simd = @import("core/simd.zig");
pub const SimdParser = simd.SimdParser;

/// SIMD chunk-classification primitives (the vector layer), re-exported from
/// `core/simd.zig`. Exposed so the benchmark's method-matrix experiment can build
/// alternative parse strategies.
pub const classify = simd.classify;

/// Streaming over a `std.Io.Reader` plus the auto-selecting facade. See `core/stream.zig`.
/// Exact record boundaries for splitting one input across workers (v0.4).
pub const parallel = @import("core/parallel.zig");

pub const stream = @import("core/stream.zig");
pub const streamReader = stream.streamReader;
pub const parseReader = stream.parseReader;
pub const Strategy = stream.Strategy;
pub const AutoOptions = stream.AutoOptions;
