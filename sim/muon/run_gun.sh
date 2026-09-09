#!/usr/bin/env bash
# Host half of sim/muon/gun_stage.py. Same mounts as sim/run_reco_ckf.sh, so the
# gun writes into the same run directory the rest of the chain reads.
#
#   run_gun.sh <output subdir> [args for gun_stage.py]
#     e.g. run_gun.sh mu_s1_b1_eval --events 500 --energy-min 1 --energy-max 2
#
# The stage lives in this repository rather than in the production one because
# the production gun does not run against the pinned image; gun_stage.py's
# docstring says which call fails and why.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ROOT="${BASELINE0_ROOT:-$HOME/baseline0}"
CML="${COLLIDERML_REPO:-/data/ColliderML-Production}"
IMG=ghcr.io/opendatadetector/sw@sha256:20e4df7b0d6befd70ce0113d852350b25cbd9aac659d1bc530036a2c85a68f89
SEED="${SEED:-42}"

[[ $# -ge 1 ]] || { echo "usage: $0 <output subdir> [args...]" >&2; exit 2; }
SUBDIR=$1; shift
EXTRA="$*"

mkdir -p "$ROOT/out" "$ROOT/cache"

exec docker run --rm \
  -v "$CML:/workspace:ro" \
  -v "$ROOT/out:/output" \
  -v "$ROOT/cache:/cache" \
  -v "$REPO:/repo:ro" \
  -e COLLIDERML_CACHE=/cache \
  -e ODD_PATH=/cache/odd-v4 \
  -e SKIP_G4_DOWNLOAD=1 \
  -e SUBDIR="$SUBDIR" \
  -e SEED="$SEED" \
  -e EXTRA="$EXTRA" \
  --entrypoint /bin/bash "$IMG" /repo/sim/muon/incontainer_gun.sh
