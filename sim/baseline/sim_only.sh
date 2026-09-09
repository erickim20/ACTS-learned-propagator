#!/usr/bin/env bash
# ddsim then digi for runs that already have merged_events.hepmc3.
#
# Concurrency is 2 and not more. One ddsim at pileup 200 grows to about 3 GB and
# WSL has 15, so five at once exhausted memory, the OOM killer took a run, and
# the machine stopped responding. Two leaves plenty of headroom.
#
#   sim_only.sh <run id> [run id ...]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${BASELINE0_ROOT:-$HOME/baseline0}"
CONC=2
mkdir -p "$ROOT/logs"

one_run() {
  local i=$1
  local t0=$SECONDS
  for stage in ddsim:ddsim_run.py:simulation_config.yaml \
               digi:digi_and_reco.py:digitization_config.yaml; do
    local name=${stage%%:*}; local rest=${stage#*:}
    local script=${rest%%:*}; local cfg=${rest#*:}
    "$HERE/run_stage.sh" "simulation/$script" "pu200/$cfg" "$i" \
      > "$ROOT/logs/pu200_${i}_${name}.log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
      echo "run $i FAILED at $name (rc=$rc) after $((SECONDS-t0))s"
      tail -12 "$ROOT/logs/pu200_${i}_${name}.log"
      return $rc
    fi
  done
  echo "run $i ok $((SECONDS-t0))s"
}

echo "runs: $*  concurrency $CONC"
for i in "$@"; do
  one_run "$i" &
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do
    sleep 30
    free -m | awk 'NR==2 {printf "    mem used %sM avail %sM\n", $3, $7}'
  done
done
wait
echo "=== batch done"
for i in "$@"; do
  printf "  run %s: " "$i"
  [ -f "$ROOT/out/runs/$i/performance_finding_ambi.root" ] && echo DONE || echo MISSING
done
