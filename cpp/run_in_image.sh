#!/usr/bin/env bash
# Build and run one of the ACTS-linked probes INSIDE the ODD image.
#
# `build.sh` handles everything that needs only headers. These need to LINK
# against libActsCore, and geom_probe additionally needs DD4hep and ROOT, so
# they are compiled where those libraries live rather than pulled out of the
# image.
#
#   ./run_in_image.sh geom_probe          # the adapter in a real Propagator
#   ./run_in_image.sh cov_probe           # one jump's covariance against RKN
#   ./run_in_image.sh nav_probe           # can the next module be named early
#   ./run_in_image.sh --map nav_truth     # ... and is it the module reached
#
# `--map` runs against the tree whose compact carries the field map instead of
# the image's own ODD. Three things have to be true together for that and none
# of them announces itself when it is not:
#
#   the tree is mounted at /cache, because the compact names the map by the
#   absolute path /cache/odd-map/data/odd-bfield-xyz.bin;
#
#   the factory library built FROM that tree comes first on LD_LIBRARY_PATH,
#   because `ODDFieldMapXyz` is its plugin and the image's own
#   libOpenDataDetector.so does not carry it. sim/odd_tree.sh: a new .so with an
#   old compact is worse than a failure, it is silent and the detector is a
#   uniform 2 T;
#
#   the compact path is handed to the probe, which is why --map appends it.
#
# A probe run this way should print the field it actually got. That is the only
# check that survives all three going wrong quietly.
#
# The image's ENTRYPOINT is not a shell, hence --entrypoint /bin/bash.
# The digest and not the tag: the tag is not pinned to a single build.
set -euo pipefail
cd "$(dirname "$0")"

MAP=0
if [[ "${1:-}" == "--map" ]]; then
  MAP=1
  shift
fi

[[ $# -ge 1 ]] || { echo "usage: $0 [--map] <probe-name> [args...]" >&2; exit 2; }

IMG=ghcr.io/opendatadetector/sw@sha256:20e4df7b0d6befd70ce0113d852350b25cbd9aac659d1bc530036a2c85a68f89
REPO="$(cd .. && pwd)"

MOUNTS=(-v "$REPO:/repo")
ENVS=()
ARGS=("$@")

if [[ $MAP -eq 1 ]]; then
  # Same host location every other script in this repository uses.
  CACHE="${BASELINE0_ROOT:-$HOME/baseline0}/cache"
  [[ -d "$CACHE/odd-map/xml" ]] || {
    echo "no map tree at $CACHE/odd-map; sim/odd_map/build_odd_map.sh builds it" >&2
    exit 2
  }
  [[ -f "$CACHE/odd-map-install/lib/libOpenDataDetector.so" ]] || {
    echo "no factory library at $CACHE/odd-map-install/lib" >&2
    exit 2
  }
  MOUNTS+=(-v "$CACHE:/cache")
  ENVS+=(-e ODD_TREE_INSTALL=/cache/odd-map-install)
  ARGS+=(/cache/odd-map/xml/OpenDataDetector.xml)
fi

docker run --rm "${MOUNTS[@]}" "${ENVS[@]}" -w /repo/cpp \
  --entrypoint /bin/bash "$IMG" /repo/cpp/incontainer_build.sh "${ARGS[@]}"
