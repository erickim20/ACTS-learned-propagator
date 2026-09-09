#!/usr/bin/env bash
# Paired timing over blocks, written down so the next measurement is the same
# one.
#
#   sim/run_timing_blocks.sh ~/partB/timing.csv 10 stock new-off-walkoff new-off
#
# One block is (repeat, bin) and holds every configuration back to back, so a
# difference is taken inside a block and anything that drifts over the two
# hours lands on every configuration of that block. The analysis is
# `python -m prop.paired_timing <csv> --baseline <first configuration>`.
#
# The configurations are tokens rather than stepper names, because two of them
# differ from a third by an environment variable and not by an arm:
#
#   stock             ACTS's own factory, which is the prebuilt SympyStepper
#                     CKF, so a difference against it is never one change
#   learned-off       Propagator<LearnedStepper, Navigator> with the transport
#                     off, so the inner EigenStepper runs behind ACTS's own
#                     navigator. The rung that separates the integrator swap
#                     and this translation unit from anything this repository
#                     navigates with
#   new-off           the moved seam, transport off
#   new-off-walkoff   ... with NAV_WALK_OFF=1, so every navigator member
#                     forwards. Measured bit-identical to
#                     Propagator<LearnedStepper, Navigator>
#   new-off-walkoff-nopublish
#                     ... and NO_PUBLISH_TRACK=1, so `publishTrack` is
#                     off too. Only a comparison with
#                     the walk off: the factory refuses the other
#                     combination, because NewNavigator reads what
#                     publishTrack writes.
#   new-helix         the moved seam with the learned transport, network off
#   new-helix-plan    ... and PLANNED_ONLY=1 with the cell sigma table
#   new-learned       ... network on
#   new-learned-plan  ... network on, PLANNED_ONLY=1, cell sigma table
#
# CELL_SIGMA is set by the token and not inherited from the environment. An
# empty table is a different arm and it is silent, so the two learned tokens
# name the table themselves and every other token clears it.
#
# Written to the csv as it goes, one row per run, so a job that dies leaves
# every block it finished. `slot` is the position in the block, which is what
# says whether a configuration was systematically first or last.
#
# ALGO_LOG_LEVEL=INFO on every run. The finalize statistics are the only place
# the candidate count is written, and the arms whose transport changes what the
# filter builds cannot be compared per event without it. It is set on every arm
# of every block rather than on some, so whatever the extra logging costs is
# inside every paired difference and cancels.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
ROOT="${BASELINE0_ROOT:-$HOME/baseline0}"
BINS="${BINS:-1 2 3 4 5}"
QTABLE="${QTABLE:-/repo/cpp/q_table_B.bin}"
PREFIX="${PREFIX:-tb}"

[[ $# -ge 3 ]] || {
  echo "usage: $0 <csv> <repeats> <configuration...>" >&2; exit 2; }
CSV=$1; REPEATS=$2; shift 2
ARMS=("$@")

mkdir -p "$(dirname "$CSV")"
LOGS="$(dirname "$CSV")/logs"
mkdir -p "$LOGS"
[[ -s "$CSV" ]] || echo "bin,repeat,slot,arm,stage_s,trackfinding_s,seeds,dedup_seeds,found,selected,stopped" > "$CSV"

export CONFIG="${CONFIG:-sim/muon/digitization_config_eval.yaml}"
export ODD_TREE="${ODD_TREE:-/cache/odd-map}"
export ODD_TREE_INSTALL="${ODD_TREE_INSTALL:-/cache/odd-map-install}"
export ALGO_LOG_LEVEL="${ALGO_LOG_LEVEL:-INFO}"

for r in $(seq 1 "$REPEATS"); do
  for b in $BINS; do
    slot=0
    for arm in "${ARMS[@]}"; do
      unset NAV_WALK_OFF PLANNED_ONLY NO_PUBLISH_TRACK
      qt=""
      export CELL_SIGMA=""
      case "$arm" in
        stock)            stepper=stock ;;
        learned-off)      stepper=learned-off ;;
        new-off)          stepper=new-off ;;
        new-off-walkoff)  stepper=new-off; export NAV_WALK_OFF=1 ;;
        new-off-walkoff-nopublish)
                          stepper=new-off; export NAV_WALK_OFF=1
                          export NO_PUBLISH_TRACK=1 ;;
        new-helix)        stepper=new-helix; qt=$QTABLE ;;
        new-helix-plan)   stepper=new-helix; qt=$QTABLE; export PLANNED_ONLY=1
                          export CELL_SIGMA="${CELL_SIGMA_BIN:-/repo/cpp/cell_sigma.bin}" ;;
        new-learned)      stepper=new-learned; qt=$QTABLE ;;
        new-learned-plan) stepper=new-learned; qt=$QTABLE; export PLANNED_ONLY=1
                          export CELL_SIGMA="${CELL_SIGMA_BIN:-/repo/cpp/cell_sigma.bin}" ;;
        *) echo "unknown configuration $arm" >&2; exit 2 ;;
      esac

      sub="${PREFIX}_${arm}_b${b}"
      log="$LOGS/${sub}_r${r}.log"
      SEED=$((100 + b)) ALGO_TIMING="/output/runs/$sub" \
        "$HERE/run_reco_ckf.sh" "mu_s1_b${b}_eval_map" "$sub" "$stepper" "$qt" \
        > "$log" 2>&1
      rc=$?
      if [[ $rc -ne 0 ]]; then
        echo "FAILED r=$r b=$b $arm rc=$rc, see $log" >&2
        exit 3
      fi

      tf=$(awk -F, '/^Algorithm:TrackFindingAlgorithm,/{print $2; exit}' \
           "$ROOT/out/runs/$sub/timing.csv")
      st=$(sed -n 's/.*Completed stage: ACTS Reconstruction in \([0-9.]*\) seconds.*/\1/p' \
           "$log" | head -1)
      # TrackFindingAlgorithm::finalize, which INFO is what makes visible.
      # `selected tracks` is the candidate count the filter paid for and is
      # the only denominator the arms that change the filter's work share.
      stat() { sed -n "s/.*- $1: \([0-9][0-9]*\).*/\1/p" "$log" | head -1; }
      sd=$(stat "total seeds"); dd=$(stat "deduplicated seeds")
      fd=$(stat "found tracks"); sl=$(stat "selected tracks")
      sb=$(stat "stopped branches")
      echo "$b,$r,$slot,$arm,${st:-nan},${tf:-nan},${sd:-nan},${dd:-nan},${fd:-nan},${sl:-nan},${sb:-nan}" >> "$CSV"
      echo "r=$r b=$b slot=$slot $arm stage=${st:-nan} tf=${tf:-nan} sel=${sl:-nan}"
      slot=$((slot + 1))
    done
  done
done
echo "done, $CSV"
