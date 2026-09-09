#!/usr/bin/env bash
# Run one of the analysis scripts here inside the image, where ROOT lives.
#
#   aggregate.sh aggregate.py ambi runs/1 runs/2 ...
#   aggregate.sh baseline_numbers.py runs/1
#
# Paths are relative to BASELINE0_ROOT/out, which is mounted at /output.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${BASELINE0_ROOT:-$HOME/baseline0}"
CML="${COLLIDERML_REPO:-/data/ColliderML-Production}"
IMG=ghcr.io/opendatadetector/sw@sha256:20e4df7b0d6befd70ce0113d852350b25cbd9aac659d1bc530036a2c85a68f89

[[ $# -ge 1 ]] || { echo "usage: $0 <script.py> [args...]" >&2; exit 2; }
PY=$1; shift

ARGS=""
for a in "$@"; do
  case "$a" in
    runs/*|fatras/*|validation/*) ARGS="$ARGS /output/$a" ;;
    *) ARGS="$ARGS $a" ;;
  esac
done

exec docker run --rm \
  -v "$ROOT/out:/output:ro" \
  -v "$HERE:/scr:ro" \
  -v "$CML:/workspace:ro" \
  -v "$ROOT/cache:/cache" \
  -e COLLIDERML_CACHE=/cache \
  -e ODD_PATH=/cache/odd-v4 \
  -e SKIP_G4_DOWNLOAD=1 \
  --entrypoint /bin/bash "$IMG" -c "
    set -o pipefail
    source /workspace/scripts/cli/setup_container_env.sh > /dev/null 2>&1
    exec python3 /scr/$PY $ARGS
  "
