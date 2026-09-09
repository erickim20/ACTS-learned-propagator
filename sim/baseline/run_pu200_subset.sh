#!/usr/bin/env bash
# Run a named set of pileup 200 runs. Same body as run_pu200.sh, but takes an
# explicit list so a failed or contaminated run can be redone on its own.
#
#   run_pu200_subset.sh [concurrency] [run ids...]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${BASELINE0_ROOT:-$HOME/baseline0}"
CONC="${1:-2}"; shift || true
RUNS=("$@")
[ ${#RUNS[@]} -gt 0 ] || RUNS=(0 6 7)
mkdir -p "$ROOT/logs"

one_run() {
  local i=$1
  local t0=$SECONDS
  for stage in pythia:pythia_gen.py:pythia_config.yaml \
               ddsim:ddsim_run.py:simulation_config.yaml \
               digi:digi_and_reco.py:digitization_config.yaml; do
    local name=${stage%%:*}; local rest=${stage#*:}
    local script=${rest%%:*}; local cfg=${rest#*:}
    "$HERE/run_stage.sh" "simulation/$script" "pu200/$cfg" "$i" \
      > "$ROOT/logs/pu200_${i}_${name}.log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
      echo "run $i FAILED at $name (rc=$rc) after $((SECONDS-t0))s"
      tail -20 "$ROOT/logs/pu200_${i}_${name}.log"
      return $rc
    fi
  done
  echo "run $i ok $((SECONDS-t0))s"
}

echo "runs: ${RUNS[*]}, $CONC at a time"
for i in "${RUNS[@]}"; do
  one_run "$i" &
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 20; done
done
wait
echo "=== subset done"
