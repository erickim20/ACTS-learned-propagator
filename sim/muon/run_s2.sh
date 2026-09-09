#!/usr/bin/env bash
# S2: muons from the origin inside pileup 200, in pT bins.
#
#   run_s2.sh <bin> <run> [events] [muons per event]
#     e.g. run_s2.sh 1 eval 500 20      # 10,000 signal muons in bin 1
#          run_s2.sh all eval 500 20
#
# Three stages:
#
#   gun_stage.py     ->  events.hepmc3          this repository's
#   pythia_gen.py    ->  merged_events.hepmc3   production, unchanged
#   fatras_reco.py   ->  performance_finding_*.root, tracksummary_*.root
#
# Two departures from S1, both measured.
#
# Fatras replaces Geant4. Geant4 at pileup 200 costs 512 s an event against
# Fatras's 7.5, and the teacher this experiment trains against is a Fatras
# sample, so running the evaluation through Fatras makes the two ends share the
# simulation instead of straddling two of them. The cuts downstream are
# identical: sim/fatras_reco.py restates digi_and_reco.py's selections value for
# value, and section 6's table is the check.
#
# The gun fires several muons from the one vertex. The cost of an event is set
# by its 200 pileup events, so N signal muons cost what one costs.
#
# Output lands in $BASELINE0_ROOT/out/runs/mu_s2_b<bin>_<run>.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ROOT="${BASELINE0_ROOT:-$HOME/baseline0}"

PT_MIN=([1]=1 [2]=2 [3]=4 [4]=8 [5]=20 [0]=0.5)
PT_MAX=([1]=2 [2]=4 [3]=8 [4]=20 [5]=50 [0]=1)

seed_for() {
  case "$1" in
    eval)  echo 100 ;;
    train) echo 200 ;;
    *)     echo 300 ;;
  esac
}

[[ $# -ge 2 ]] || {
  echo "usage: $0 <bin 0..5|all> <run name> [events] [muons per event]" >&2
  exit 2; }
BIN=$1; RUN=$2; EVENTS="${3:-500}"; MUONS="${4:-20}"

bins=()
if [[ "$BIN" == "all" ]]; then bins=(1 2 3 4 5); else bins=("$BIN"); fi

for b in "${bins[@]}"; do
  lo=${PT_MIN[$b]:-}; hi=${PT_MAX[$b]:-}
  [[ -n "$lo" ]] || { echo "no such bin: $b" >&2; exit 2; }

  SUBDIR="mu_s2_b${b}_${RUN}"
  export SEED=$(( $(seed_for "$RUN") + b ))
  export CONFIG_DIR="$HERE"
  mkdir -p "$ROOT/logs"

  echo "=== S2 bin $b  pT $lo to $hi GeV  run $RUN  seed $SEED" \
       " $EVENTS events x $MUONS muons"
  t0=$SECONDS

  ts=$SECONDS
  "$HERE/run_gun.sh" "$SUBDIR" --events "$EVENTS" \
    --energy-min "$lo" --energy-max "$hi" --muons-per-event "$MUONS" \
    > "$ROOT/logs/${SUBDIR}_gun.log" 2>&1
  rc=$?
  echo "    gun rc=$rc $((SECONDS-ts))s"
  if [ $rc -ne 0 ]; then tail -25 "$ROOT/logs/${SUBDIR}_gun.log"; exit $rc; fi

  ts=$SECONDS
  "$REPO/sim/baseline/run_stage.sh" simulation/pythia_gen.py \
    pileup_config.yaml "$SUBDIR" --events "$EVENTS" \
    > "$ROOT/logs/${SUBDIR}_pileup.log" 2>&1
  rc=$?
  echo "    pileup rc=$rc $((SECONDS-ts))s"
  if [ $rc -ne 0 ]; then tail -25 "$ROOT/logs/${SUBDIR}_pileup.log"; exit $rc; fi

  ts=$SECONDS
  "$REPO/sim/baseline/run_fatras.sh" \
    "runs/$SUBDIR/merged_events.hepmc3" "runs/$SUBDIR" "$EVENTS" \
    > "$ROOT/logs/${SUBDIR}_fatras.log" 2>&1
  rc=$?
  echo "    fatras rc=$rc $((SECONDS-ts))s"
  if [ $rc -ne 0 ]; then tail -25 "$ROOT/logs/${SUBDIR}_fatras.log"; exit $rc; fi

  echo "=== bin $b done in $((SECONDS-t0))s -> runs/$SUBDIR"
done
