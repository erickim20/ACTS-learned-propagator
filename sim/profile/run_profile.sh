#!/usr/bin/env bash
# Where the time goes inside reconstruction, at two levels.
#
#   run_profile.sh <source run> <output subdir> [--perf]
#     e.g. run_profile.sh 0 prof0            per-algorithm timing.tsv only
#          run_profile.sh 0 perf0 --perf     timing.tsv and a symbol profile
#
# The source run is one of item 0's directories under $BASELINE0_ROOT/out/runs,
# which already holds the Geant4 `edm4hep.root`. Reconstruction is re-run from
# it into a fresh output directory, so item 0's own outputs are left alone. The
# configuration is `sim/baseline/pu200/digitization_config.yaml`, the same file
# item 0 was measured with, so this profiles that chain and not another one.
#
# Two things about perf, and both are properties of this box rather than of the
# measurement. The image ships no perf and has no `linux-perf` apt candidate, so
# the host's `linux-tools` build is bind-mounted; it needs libdw, libunwind,
# libslang, libnuma and libtraceevent, which is what `odd-sw:perf` is. And the
# container's default seccomp profile blocks `perf_event_open`, which cannot be
# lifted on a running container, so the container is created with it off.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ROOT="${BASELINE0_ROOT:-$HOME/baseline0}"
CML="${COLLIDERML_REPO:-/data/ColliderML-Production}"
IMG=ghcr.io/opendatadetector/sw@sha256:20e4df7b0d6befd70ce0113d852350b25cbd9aac659d1bc530036a2c85a68f89
PERF_IMG=odd-sw:perf
SEED="${SEED:-42}"
# Relative to the repository root, so that both item 0's own configuration and
# the writers-off one below can be named without moving either of them.
CONFIG="${CONFIG:-sim/baseline/pu200/digitization_config.yaml}"

[[ $# -ge 2 ]] || { echo "usage: $0 <source run> <output subdir> [--perf]" >&2; exit 2; }
SRC=$1; SUBDIR=$2; shift 2
PERF=0
[[ "${1:-}" == "--perf" ]] && PERF=1

[[ -f "$ROOT/out/runs/$SRC/edm4hep.root" ]] || {
  echo "no $ROOT/out/runs/$SRC/edm4hep.root" >&2; exit 2; }

opts=()
if [[ $PERF == 1 ]]; then
  # The host's perf, and the libraries it needs, committed on top of the pinned
  # image. Built here rather than by hand so the profile is reproducible.
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

echo "=== reconstructing run $SRC into $SUBDIR  config=$CONFIG perf=$PERF"
# The output directory is made inside the container. item 0's runs were written
# by a container too, so `$ROOT/out/runs` is owned by root and the host user
# cannot create a sibling.

exec docker run --rm "${opts[@]}" \
  -v "$CML:/workspace:ro" \
  -v "$ROOT/out:/output" \
  -v "$ROOT/cache:/cache" \
  -v "$REPO:/repo:ro" \
  -v "$HERE:/work:ro" \
  -e COLLIDERML_CACHE=/cache \
  -e ODD_PATH=/cache/odd-v4 \
  -e SKIP_G4_DOWNLOAD=1 \
  -e CONFIG="$CONFIG" \
  -e SUBDIR="$SUBDIR" \
  -e SEED="$SEED" \
  -e EXTRA="--input-file /output/runs/$SRC/edm4hep.root" \
  -e TIMING_DIR="/output/runs/$SUBDIR" \
  -e PERF="$PERF" \
  -e PERF_FREQ="${PERF_FREQ:-499}" \
  --entrypoint /bin/bash "$IMG" /work/incontainer_profile.sh
