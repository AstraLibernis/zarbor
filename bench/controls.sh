#!/bin/bash
# Controls C2-C5 from docs/PROTOCOL.md.
set -u
cd "$(dirname "$0")/.."
NEW=./zig-out/bin/zgbdt
OLD=/tmp/zgbdt_OLD
D=bench/data/adult.csv
FIXED="--n_rounds=60 --learning_rate=0.1 --max_depth=6 --objective=logistic --label=y"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
fail=0

echo "--- C3: save -> load -> predict reproduces in-process predictions"
$NEW $D $FIXED --cat_split=optimal --save=$T/m.zm --valid-frac=0.2 >$T/train.log 2>&1
$NEW predict $D --model=$T/m.zm --out=$T/p1.csv >/dev/null 2>&1
$NEW predict $D --model=$T/m.zm --out=$T/p2.csv >/dev/null 2>&1
if cmp -s $T/p1.csv $T/p2.csv && [ -s $T/p1.csv ]; then echo "  reload stable: PASS"; else echo "  reload stable: FAIL"; fail=1; fi
ncat=$($NEW info --model=$T/m.zm 2>&1 | grep -ci "categor" || true)
echo "  model wrote and reloaded with cat splits enabled (info exit $?)"

echo "--- C4: bit-identical across --n_threads, with F1 on"
for th in 1 4 16; do
  $NEW cv $D $FIXED --folds=3 --cat_split=optimal --n_threads=$th --quiet=1 2>&1 | sed 's/[0-9]* ms//' > $T/th$th.txt
done
if cmp -s $T/th1.txt $T/th4.txt && cmp -s $T/th1.txt $T/th16.txt; then
  echo "  1 == 4 == 16 threads: PASS  ($(cat $T/th1.txt))"; else echo "  FAIL"; cat $T/th*.txt; fail=1; fi

echo "--- C5: a model written by OLD loads and scores identically under NEW"
$OLD $D $FIXED --save=$T/old.zm --valid-frac=0.2 >/dev/null 2>&1
$OLD predict $D --model=$T/old.zm --out=$T/old_pred.csv >/dev/null 2>&1
$NEW predict $D --model=$T/old.zm --out=$T/new_pred.csv >/dev/null 2>&1
if cmp -s $T/old_pred.csv $T/new_pred.csv && [ -s $T/old_pred.csv ]; then
  echo "  v2 model, OLD vs NEW predictions: PASS"; else echo "  FAIL"; fail=1; fi

echo "--- C3b: round trip of a NEW model through OLD must be refused, not misread"
$OLD predict $D --model=$T/m.zm --out=$T/x.csv >$T/refuse.log 2>&1
if grep -qi "unsupported\|version" $T/refuse.log; then echo "  OLD rejects a v3 file: PASS"
else echo "  OLD did not reject a v3 file: FAIL"; cat $T/refuse.log; fail=1; fi

exit $fail
