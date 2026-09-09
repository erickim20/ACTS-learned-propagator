#!/usr/bin/env bash
# Run field_gate.py's three paths in the pinned image and print the verdict.
#
#   sim/odd_map/run_gate.sh              # the map-field tree
#   ODD_TREE= sim/odd_map/run_gate.sh    # the ODD as shipped, for contrast
#
# ODD_TREE and ODD_TREE_INSTALL are the same variables every other runner in
# this repository takes, and the same in-container override applies: the
# image's /root/.bashrc exports ODD_PATH=/opt/odd at :182 and
# setup_container_env.sh then assigns with ${ODD_PATH:-...}, so a docker -e is
# discarded and the export has to happen after the source.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ROOT="${BASELINE0_ROOT:-$HOME/baseline0}"
CML="${COLLIDERML_REPO:-/data/ColliderML-Production}"
IMG=ghcr.io/opendatadetector/sw@sha256:20e4df7b0d6befd70ce0113d852350b25cbd9aac659d1bc530036a2c85a68f89

exec docker run --rm \
  -v "$CML:/workspace:ro" \
  -v "$ROOT/cache:/cache" \
  -v "$REPO:/repo:ro" \
  -v "$HERE:/work:ro" \
  -e COLLIDERML_CACHE=/cache \
  -e SKIP_G4_DOWNLOAD=1 \
  -e ODD_TREE="${ODD_TREE-/cache/odd-map}" \
  -e ODD_TREE_INSTALL="${ODD_TREE_INSTALL-/cache/odd-map-install}" \
  --entrypoint /bin/bash "$IMG" /work/incontainer_gate.sh
