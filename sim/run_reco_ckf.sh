#!/usr/bin/env bash
# Re-run item 0's reconstruction from an existing edm4hep.root, with the CKF's
# stepper chosen on the command line.
#
#   sim/run_reco_ckf.sh <source run> <output subdir> <arm> [qtable]
#
#   arm: stock | learned-off | learned-on | helix | new-off | new-helix |
#        new-learned
#     e.g. sim/run_reco_ckf.sh 0 ckf_accept_stock stock
#          sim/run_reco_ckf.sh 0 ckf_run0_learned learned-on cpp/q_table.bin
#
# FEATURE_DUMP writes the fourteen network inputs as this run builds them, for
# src/prop/feature_check.py. The path is a container path, so it belongs under
# /output. Empty is off, which is what a run that is being measured wants.
#
#   FEATURE_DUMP=/output/runs/featcheck/features.bin sim/run_reco_ckf.sh ...
#
# RECO_EXTRA is appended to what digi_and_reco.py is given, for the arguments
# this driver has no opinion about. `-n 200` is the one that gets used: the
# feature dump wants a couple of hundred events and the performance tables want
# all ten thousand, and they are otherwise the same run.
#
#   RECO_EXTRA="-n 200" FEATURE_DUMP=... sim/run_reco_ckf.sh ...
#
# NAV_WALK_OFF=1 keeps NewNavigator in the propagator and switches its walk
# off, so every navigator member forwards. Applies only to `new-off`.
#
# PLANNED_ONLY=1 fires the learned transport only on the destination the walk
# named for that transport. Applies to `new-helix` and `new-learned`.
#
# FIELD_GATE=0.1 keeps the helix and skips the network wherever the map at the
# source of the transport is within 0.1 T of 2 T. Applies to `new-learned`.
#
# CORE_BZ=source runs the helix core of the learned transport at the map value
# at each jump's source instead of the fixed 2 T of LearnedTransport.hpp's
# kBHelix. The three runtime sites, behind one switch, so one
# binary carries both cores. It applies to `helix` and `new-helix`; reco_ckf.py
# refuses it on an arm with no learned transport and on an arm with the network
# on, because the fit was made at 2 T.
#
# PERF=1 records a symbol profile of the run with the host's perf, and writes
# perf.data and the three reports beside the run's own output. The arrangement
# is `sim/profile/run_profile.sh`'s and the reasons are in
# `sim/profile/README.md`: the image ships no perf so the host's is bind
# mounted into `odd-sw:perf`, and the container's default seccomp profile
# blocks `perf_event_open` and cannot be changed on a running container.
# `sim/profile/run_profile.sh` profiles item 0's ttbar chain; this profiles one
# of this repository's arms on whatever sample it is given, which is what
# comparing two arms in `sim/profile/bucket_perf.py`'s buckets needs.
#
#   PERF=1 CONFIG=sim/muon/digitization_config_noio.yaml sim/run_reco_ckf.sh ...
#
# NO_RESOLVE_MATERIAL=1 sets Navigator::Config::resolveMaterial false in the
# navigator this repository's factory builds. It has no effect on `stock`,
# which builds no navigator from this tree, and reco_ckf.py refuses that
# combination rather than running an arm that is not the one it is labelled.
#
# Same mounts and cache as sim/profile/run_profile.sh, so the geometry, the
# config and the seed are item 0's own; only the output subdir is new. The
# acceptance criterion is that the `stock` arm reproduces the source run's
# performance_finding numbers exactly, which a plain re-run is known to do.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
ROOT="${BASELINE0_ROOT:-$HOME/baseline0}"
CML="${COLLIDERML_REPO:-/data/ColliderML-Production}"
IMG=ghcr.io/opendatadetector/sw@sha256:20e4df7b0d6befd70ce0113d852350b25cbd9aac659d1bc530036a2c85a68f89
SEED="${SEED:-42}"
CONFIG="${CONFIG:-sim/baseline/pu200/digitization_config.yaml}"

[[ $# -ge 3 ]] || {
  echo "usage: $0 <source run> <output subdir> <arm> [qtable]" >&2
  echo "  arm: stock|learned-off|learned-on|helix|new-off|new-helix|new-learned" >&2
  exit 2
}
SRC=$1; SUBDIR=$2; STEPPER=$3; QTABLE="${4:-}"

[[ -f "$ROOT/out/runs/$SRC/edm4hep.root" ]] || {
  echo "no $ROOT/out/runs/$SRC/edm4hep.root" >&2; exit 2; }

opts=()
if [[ "${PERF:-0}" == "1" ]]; then
  # The same image and the same host perf `sim/profile/run_profile.sh` uses,
  # built by the same recipe if it is not there, so the two profiles are
  # comparable.
  PERF_IMG=odd-sw:perf
  if ! docker image inspect "$PERF_IMG" >/dev/null 2>&1; then
    echo "=== building $PERF_IMG from the pinned image"
    docker rm -f oddperf >/dev/null 2>&1 || true
    docker run --name oddperf --entrypoint /bin/bash "$IMG" -lc \
      'apt-get update -qq >/dev/null 2>&1
       apt-get install -y -qq libdw1 libunwind8 libslang2 libnuma1 libtraceevent1 >/dev/null 2>&1'
    docker commit oddperf "$PERF_IMG" >/dev/null
    docker rm -f oddperf >/dev/null
  fi
  TOOLS="$(dirname "$(find /usr/lib/linux-tools-* -maxdepth 1 -name perf -type f 2>/dev/null | head -1)")"
  [[ -n "$TOOLS" ]] || { echo "no perf under /usr/lib/linux-tools-*" >&2; exit 2; }
  echo "=== perf from $TOOLS"
  IMG="$PERF_IMG"
  opts+=(--privileged --security-opt seccomp=unconfined -v "$TOOLS:/hostperf:ro")
fi

echo "=== reco run $SRC -> $SUBDIR  stepper=$STEPPER qtable=${QTABLE:-none} perf=${PERF:-0}"

exec docker run --rm "${opts[@]}" \
  -v "$CML:/workspace:ro" \
  -v "$ROOT/out:/output" \
  -v "$ROOT/cache:/cache" \
  -v "$REPO:/repo:ro" \
  -e COLLIDERML_CACHE=/cache \
  -e ODD_PATH=/cache/odd-v4 \
  -e ODD_TREE="${ODD_TREE:-}" \
  -e ODD_TREE_INSTALL="${ODD_TREE_INSTALL:-}" \
  -e SKIP_G4_DOWNLOAD=1 \
  -e CONFIG="$CONFIG" \
  -e SUBDIR="$SUBDIR" \
  -e SEED="$SEED" \
  -e EXTRA="--input-file /output/runs/$SRC/edm4hep.root ${RECO_EXTRA:-}" \
  -e STEPPER="$STEPPER" \
  -e QTABLE="$QTABLE" \
  -e CELL_SIGMA="${CELL_SIGMA:-}" \
  -e SIGMA_HEAD="${SIGMA_HEAD:-}" \
  -e NO_RESOLVE_MATERIAL="${NO_RESOLVE_MATERIAL:-}" \
  -e NAV_WALK_OFF="${NAV_WALK_OFF:-}" \
  -e NO_PUBLISH_TRACK="${NO_PUBLISH_TRACK:-}" \
  -e PLANNED_ONLY="${PLANNED_ONLY:-}" \
  -e FIELD_GATE="${FIELD_GATE:-}" \
  -e CORE_BZ="${CORE_BZ:-}" \
  -e NET_CORE_FIT="${NET_CORE_FIT:-}" \
  -e NO_SEAM_MATERIAL="${NO_SEAM_MATERIAL:-}" \
  -e NO_SEAM_SLAB="${NO_SEAM_SLAB:-}" \
  -e ALGO_LOG_LEVEL="${ALGO_LOG_LEVEL:-}" \
  -e TRANSPORT_CENSUS="${TRANSPORT_CENSUS:-}" \
  -e FEATURE_DUMP="${FEATURE_DUMP:-}" \
  -e FEATURE_DUMP_MAX="${FEATURE_DUMP_MAX:-}" \
  -e ALGO_TIMING="${ALGO_TIMING:-}" \
  -e PERF="${PERF:-0}" \
  -e PERF_FREQ="${PERF_FREQ:-4999}" \
  --entrypoint /bin/bash "$IMG" /repo/sim/incontainer_reco_ckf.sh
