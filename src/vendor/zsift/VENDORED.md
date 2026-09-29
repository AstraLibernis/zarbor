# Vendored: zsift

The CSV parser zarbor loads data with. Copied, not fetched, so zarbor has no
dependencies of any kind.

- Source: https://github.com/AstraLibernis/zsift
- Commit: `a0e581c27bb935edf1931dde9a7ca47e901c7636` (after zsift v0.4 milestone M5), copied 2026-09-29
- Licence: LGPL-3.0-or-later, same as zarbor (SPDX headers kept in every file)
- Copied: `src/csv.zig` (its `test { ... }` block removed: zsift's tests are not
  vendored) and every file in `src/core/`. Nothing else is changed.

## Refreshing

1. In a zsift checkout at the wanted commit, run `zig build test` and
   `zig build verify` (both must pass).
2. Replace `src/vendor/zsift/core/` with that checkout's `src/core/`, and
   `src/vendor/zsift/csv.zig` with its `src/csv.zig` minus the `test` block.
3. Update the commit above, then run zarbor's `zig build test` and
   `zig build loadcmp` (see `bench/loadcmp.zig`).
