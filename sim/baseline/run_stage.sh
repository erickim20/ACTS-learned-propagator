#!/usr/bin/env bash
# Run one ColliderML simulation stage in the pinned ODD image.
#
# This is ColliderML's own scripts/cli/run_pipeline_docker.sh reduced to a
# single stage, with two deliberate differences:
#
#   - The image is pinned by digest, not the :0.2.2 tag that script
#     defaults to. The baseline has to come out of the same ACTS build the
#     learned stepper will later be linked against (44.99.99-colliderml-arrow),
#     or it is not a baseline for this project.
#   - The ODD tree is chosen here rather than taken from the image. ODD_TREE
#     and ODD_TREE_INSTALL name a tree and its factory library; empty leaves
#     the image's own ODD, which is the detector the release describes.
#     `sim/odd_tree.sh` applies them inside the container and says why they
#     cannot be applied from out here.
#
# The `-e ODD_PATH=/cache/odd-v4` below has never had any effect and is kept
# only because removing it would change nothing: `setup_container_env.sh`
# sources /root/.bashrc, which exports ODD_PATH=/opt/odd, and then assigns with
# ${ODD_PATH:-...}. The two trees are byte-identical in xml/ and data/, so
# nothing measured before this was affected.
#
#   run_stage.sh <stage script> <config under CONFIG_DIR> <run id> [args...]
#     e.g. run_stage.sh simulation/pythia_gen.py pu200/pythia_config.yaml 1
#
# BASELINE0_ROOT holds the cache and the output. The configs and the in-container
# script come from the repository, so the repository is the record. CONFIG_DIR
# defaults to this directory and is what sim/muon overrides: the ladder there is
# a different sample through the same stages, not a different chain.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CONFIG_DIR="$(cd "${CONFIG_DIR:-$HERE}" && pwd)"
ROOT="${BASELINE0_ROOT:-$HOME/baseline0}"
CML="${COLLIDERML_REPO:-/data/ColliderML-Production}"
IMG=ghcr.io/opendatadetector/sw@sha256:20e4df7b0d6befd70ce0113d852350b25cbd9aac659d1bc530036a2c85a68f89
SEED="${SEED:-42}"

[[ $# -ge 3 ]] || { echo "usage: $0 <stage script> <config> <run id> [args...]" >&2; exit 2; }
STAGE=$1; CONFIG=$2; SUBDIR=$3; shift 3
EXTRA="$*"

echo "=== stage $STAGE  config $CONFIG  run $SUBDIR"
mkdir -p "$ROOT/out" "$ROOT/cache"

exec docker run --rm \
  -v "$CML:/workspace:ro" \
  -v "$ROOT/out:/output" \
  -v "$ROOT/cache:/cache" \
  -v "$CONFIG_DIR:/configs:ro" \
  -v "$HERE:/work:ro" \
  -v "$REPO:/repo:ro" \
  -e COLLIDERML_CACHE=/cache \
  -e ODD_PATH=/cache/odd-v4 \
  -e ODD_TREE="${ODD_TREE:-}" \
  -e ODD_TREE_INSTALL="${ODD_TREE_INSTALL:-}" \
  -e SKIP_G4_DOWNLOAD=1 \
  -e STAGE="$STAGE" \
  -e CONFIG="$CONFIG" \
  -e SUBDIR="$SUBDIR" \
  -e SEED="$SEED" \
  -e EXTRA="$EXTRA" \
  --entrypoint /bin/bash "$IMG" /work/incontainer.sh
