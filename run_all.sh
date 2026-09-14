#!/usr/bin/env bash
# Run every stage in order. Pass a target name to restrict, or "all".
#
#   ./run_all.sh all                    full prediction (needs fpocket)
#   ./run_all.sh all --replay            reuse the published cavity, no fpocket
#
set -euo pipefail
TARGET="${1:-all}"
MODE="${2:-}"
PY="${PYTHON:-python}"
WORK="${WORK:-work}"
REF="${REF:-reference}"

$PY stages/00_structures.py  "$TARGET" --work "$WORK"
$PY stages/01_sites.py       "$TARGET" --work "$WORK"
$PY stages/02_network.py     "$TARGET" --work "$WORK"
$PY stages/03_propagation.py "$TARGET" --work "$WORK"
if [ "$MODE" = "--replay" ]; then
  $PY stages/04_pockets.py   "$TARGET" --work "$WORK" --replay-root "$REF"
else
  $PY stages/04_pockets.py   "$TARGET" --work "$WORK"
fi
$PY stages/05_evaluation.py  "$TARGET" --work "$WORK"
$PY stages/06_baselines.py   "$TARGET" --work "$WORK"
$PY stages/07_robustness.py  "$TARGET" --work "$WORK"
$PY stages/08_report.py --work "$WORK" --reference "$REF"
