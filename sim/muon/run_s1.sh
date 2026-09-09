#!/usr/bin/env bash
# S1: one muon per event from the origin, no pileup, in pT bins.
#
#   run_s1.sh <bin> <run> [events]
#     e.g. run_s1.sh 1 eval 500        # 1 to 2 GeV, the reconstruction run
#          run_s1.sh 1 train 500       # the same window, a different seed
#          run_s1.sh all eval 500      # every bin in turn
#
# Three stages. The last two are the production ones, unmodified. The first is
# this repository's, because the production gun does not run against the pinned
# image; sim/muon/gun_stage.py says which call fails and why.
#
#   gun_stage.py        ->  events.hepmc3
#   ddsim_run.py        ->  edm4hep.root
#   digi_and_reco.py    ->  performance_finding_*.root, tracksummary_*.root
#
# Output lands in $BASELINE0_ROOT/out/runs/mu_s1_b<bin>_<run>, so every reader
# already in this repository (sim/baseline/aggregate.sh, sim/run_slice_perf.sh,
# sim/run_reco_ckf.sh) finds it without being told anything new.
#
# The bin edges below are a choice and nothing measured sets them. Roughly one
# octave per bin to
# 20 GeV, then a wider top bin because a defect that grows with momentum shows
# there and statistics are cheap.
#
# The run name picks the seed as well as the directory, because section 7 wants
# the training run and the reconstruction run to be different events and not
# merely different tracks in one set.
#
# Two environment variables exist for regenerating the same sample through a
# different detector, which is what the field map needed:
#
#   SUBDIR_SUFFIX   appended to the run directory, seed untouched. The seed has
#                   to stay 100+bin or the events are not the same events, and
#                   the run name is what picks it, so the directory cannot
#                   carry the distinction on its own.
#   REUSE_EVENTS=1  copy events.hepmc3 from the unsuffixed run and skip the gun
#                   stage. The gun builds no detector and reads no field
#                   (`gun_stage.py` imports acts and HepMC3Writer and nothing
#                   else), so the same file is correct for any geometry.
#                   Copying rather than regenerating means the two samples have
#                   byte-identical input events instead of merely the same
#                   seed.
#
# ODD_TREE and ODD_TREE_INSTALL pass through to run_stage.sh and pick the
# detector. Unset is the ODD as shipped.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ROOT="${BASELINE0_ROOT:-$HOME/baseline0}"
IMG=ghcr.io/opendatadetector/sw@sha256:20e4df7b0d6befd70ce0113d852350b25cbd9aac659d1bc530036a2c85a68f89

# bin -> pT window in GeV
PT_MIN=([1]=1 [2]=2 [3]=4 [4]=8 [5]=20 [0]=0.5)
PT_MAX=([1]=2 [2]=4 [3]=8 [4]=20 [5]=50 [0]=1)

# Bin 0 is the diagnostic arm of section 6, not part of the ladder: it sits
# between the seeding minimum and the truth selection, so it can produce fake
# tracks and cannot produce efficiency. It is never summed with the others.

# run name -> seed offset
seed_for() {
  case "$1" in
    eval)  echo 100 ;;
    train) echo 200 ;;
    *)     echo 300 ;;
  esac
}

[[ $# -ge 2 ]] || {
  echo "usage: $0 <bin 0..5|all> <run name> [events]" >&2; exit 2; }
BIN=$1; RUN=$2; EVENTS="${3:-500}"

bins=()
if [[ "$BIN" == "all" ]]; then bins=(1 2 3 4 5); else bins=("$BIN"); fi

for b in "${bins[@]}"; do
  lo=${PT_MIN[$b]:-}; hi=${PT_MAX[$b]:-}
  [[ -n "$lo" ]] || { echo "no such bin: $b" >&2; exit 2; }

  BASE="mu_s1_b${b}_${RUN}"
  SUBDIR="${BASE}${SUBDIR_SUFFIX:-}"
  export SEED=$(( $(seed_for "$RUN") + b ))
  export CONFIG_DIR="$HERE"
  mkdir -p "$ROOT/logs"

  echo "=== S1 bin $b  pT $lo to $hi GeV  run $RUN  seed $SEED  events $EVENTS"
  echo "    -> runs/$SUBDIR  detector ${ODD_TREE:-image ODD}"
  t0=$SECONDS

  ts=$SECONDS
  if [ "${REUSE_EVENTS:-}" = "1" ]; then
    if [ ! -f "$ROOT/out/runs/$BASE/events.hepmc3" ]; then
      echo "    REUSE_EVENTS=1 but no runs/$BASE/events.hepmc3" >&2; exit 2
    fi
    # The copy runs in the image, not here. Every stage writes to
    # $ROOT/out/runs from inside a container, so the whole tree is owned by
    # root and a host-side mkdir fails with EACCES.
    docker run --rm -v "$ROOT/out:/output" --entrypoint /bin/bash \
      "$IMG" -c "mkdir -p /output/runs/$SUBDIR && \
                 cp -p /output/runs/$BASE/events.hepmc3* /output/runs/$SUBDIR/"
    rc=$?
    echo "    gun reused from runs/$BASE rc=$rc $((SECONDS-ts))s"
    if [ $rc -ne 0 ]; then exit $rc; fi
  else
    "$HERE/run_gun.sh" "$SUBDIR" --events "$EVENTS" \
      --energy-min "$lo" --energy-max "$hi" \
      > "$ROOT/logs/${SUBDIR}_gun.log" 2>&1
    rc=$?
    echo "    gun rc=$rc $((SECONDS-ts))s"
    if [ $rc -ne 0 ]; then tail -25 "$ROOT/logs/${SUBDIR}_gun.log"; exit $rc; fi
  fi

  for stage in ddsim:ddsim_run.py:simulation_config.yaml \
               digi:digi_and_reco.py:digitization_config.yaml; do
    name=${stage%%:*}; rest=${stage#*:}
    script=${rest%%:*}; cfg=${rest#*:}

    ts=$SECONDS
    "$REPO/sim/baseline/run_stage.sh" "simulation/$script" "$cfg" "$SUBDIR" \
      --events "$EVENTS" > "$ROOT/logs/${SUBDIR}_${name}.log" 2>&1
    rc=$?
    echo "    $name rc=$rc $((SECONDS-ts))s"
    if [ $rc -ne 0 ]; then
      tail -25 "$ROOT/logs/${SUBDIR}_${name}.log"
      exit $rc
    fi
  done
  echo "=== bin $b done in $((SECONDS-t0))s -> runs/$SUBDIR"
done
