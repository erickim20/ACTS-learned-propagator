#!/usr/bin/env bash
# The item 0 sample: ttbar at pileup 200 through Geant4 and the stock CKF.
#
# Each run is 4 events and carries its own slice of the MadGraph hard scatter,
# so runs are independent and the sample is not N copies of the same events.
#
# Concurrency is set by memory, not by cores. One ddsim at pileup 200 grows to
# about 3 GB and WSL has 15, so the ceiling is 2 with room to spare and 3 at a
# push. Five at once exhausted memory: the OOM killer took a run, and WSL itself
# stopped responding and had to be restarted with `wsl --shutdown`. The box has
# 16 cores and none of that helps here.
#
#   run_pu200.sh [n_runs] [concurrency]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${BASELINE0_ROOT:-$HOME/baseline0}"
NRUNS="${1:-8}"
CONC="${2:-2}"
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

echo "starting $NRUNS runs, $CONC at a time, 4 events each"
pids=()
for i in $(seq 0 $((NRUNS-1))); do
  one_run "$i" &
  pids+=($!)
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do
    sleep 20
    free -g | awk 'NR==2 {printf "    mem used %sG avail %sG\n", $3, $7}'
  done
done
wait

echo "=== done"
for i in $(seq 0 $((NRUNS-1))); do
  printf "run %s: " "$i"
  ls "$ROOT/out/runs/$i"/performance_finding_ambi.root >/dev/null 2>&1 \
    && echo "has performance" || echo "MISSING"
done
