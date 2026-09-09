#!/usr/bin/env bash
# Build the learned_ckf python extension INSIDE the pinned ODD image.
#
#   cpp/fetch_ckf_src.sh        # once, needs network on the host
#   cpp/build_learned_ckf.sh
#
# The .so lands in cpp/ on the mount, so it survives the container and is
# importable by anything that runs through sim/run_in_odd.sh (the repo is /data
# there; add /data/cpp to sys.path).
set -euo pipefail
cd "$(dirname "$0")"

IMG=ghcr.io/opendatadetector/sw@sha256:20e4df7b0d6befd70ce0113d852350b25cbd9aac659d1bc530036a2c85a68f89
REPO="$(cd .. && pwd)"

docker run --rm -v "$REPO:/repo" -w /repo/cpp --entrypoint /bin/bash "$IMG" \
  /repo/cpp/incontainer_build_ckf.sh
