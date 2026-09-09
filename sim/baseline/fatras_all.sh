#!/usr/bin/env bash
# The Fatras arm over the same merged events the Geant4 arm uses. One at a time,
# because the Geant4 batch is running alongside and memory is the constraint on
# this box.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${BASELINE0_ROOT:-$HOME/baseline0}"
RUNNER="$HERE/run_fatras.sh"
mkdir -p "$ROOT/logs"

for i in "$@"; do
  [ -f "$ROOT/out/runs/$i/merged_events.hepmc3" ] || { echo "run $i: no merged events, skipping"; continue; }
  t0=$SECONDS
  bash "$RUNNER" "runs/$i/merged_events.hepmc3" "fatras/$i" 4 \
    > "$ROOT/logs/fatras_$i.log" 2>&1
  rc=$?
  echo "fatras run $i rc=$rc $((SECONDS-t0))s"
  [ $rc -eq 0 ] || tail -12 "$ROOT/logs/fatras_$i.log"
done
echo "=== fatras done"
for i in "$@"; do
  printf "  fatras/%s: " "$i"
  [ -f "$ROOT/out/fatras/$i/performance_finding_ambi.root" ] && echo DONE || echo MISSING
done
