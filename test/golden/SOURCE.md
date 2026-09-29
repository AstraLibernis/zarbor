# titanic3.csv

Vanderbilt University Department of Biostatistics, `titanic3` (1309 passengers,
14 columns), downloaded 2026-09-29 from https://hbiostat.org/data/repo/titanic3.csv
and committed unchanged. Chosen because it is public and small, and has real
missing values, quoted fields with embedded commas, and categorical columns
both narrow and too wide for a bin.

`expected/` holds one record per case in `golden.zig`. Re-baseline only on
purpose (`zig build golden -- --update`) and say why in the commit.
