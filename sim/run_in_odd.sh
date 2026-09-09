#!/usr/bin/env bash
# Run a script inside the ColliderML ODD image, with ACTS actually importable.
#
# Three things are not obvious and each cost a while to find:
#
#  1. The image's ENTRYPOINT does not give you a shell, so `docker run IMG bash`
#     fails with "cannot execute binary file". Use --entrypoint /bin/bash.
#  2. libOpenDataDetector.so is in /opt/odd-install/lib, not in
#     /opt/odd/factory where odd.py's own error message sends you, and
#     ODD_PATH has to be set on top of that.
#  3. `python3` on PATH is 3.12 and CANNOT import acts. ACTS is built against
#     the spack python 3.13, and neither PYTHONPATH nor LD_LIBRARY_PATH is set
#     up for it by default. That is what the exports below are for.
#  4. The image is pinned by digest below, and must be. `:latest` is a single
#     arm64 manifest whose ACTS is a different build and which has no /opt/odd
#     at all, so a tag lookup can pick an image that cannot run the ODD.
#
# The image is linux/amd64 and the digest fixes that, so no --platform is
# needed anywhere. This repo assumes an x86_64 host and no emulation. On an
# arm64 Mac docker emulates it with a warning, and instantiating the CKF
# against LearnedStepper kills the emulator, which is why the work moved to
# one machine.
#
#   sim/run_in_odd.sh sim/gen_teacher.py --events 200 --tracks 100 --pdg pion
#   sim/run_in_odd.sh --shell                     # poke around by hand
#
# Run it from the repository root. The root is mounted at /data and is the
# working directory, so paths are the same inside and out, and output written
# under /data lands in the repo.
#  5. ODD_TREE and ODD_TREE_INSTALL pick a different ODD, the same two
#     variables every other runner here takes. The cache is mounted at /cache
#     so that a tree living there is reachable; without ODD_TREE nothing about
#     the geometry changes and the image's own ODD loads.
set -euo pipefail
cd "$(dirname "$0")/.."
CACHE="${BASELINE0_ROOT:-$HOME/baseline0}/cache"

IMG=ghcr.io/opendatadetector/sw@sha256:20e4df7b0d6befd70ce0113d852350b25cbd9aac659d1bc530036a2c85a68f89
docker image inspect "$IMG" >/dev/null 2>&1 || {
  echo "ODD image not present. Pull it by digest, not by tag:" >&2
  echo "  docker pull $IMG" >&2
  exit 1
}

SETUP='
SPACK=/spack/opt/spack/linux-x86_64
ACTSDIR=$(ls -d $SPACK/acts-*/ | head -1)
PY=$(ls -d $SPACK/python-3.13*/bin/python3 | head -1)
set +u
source $(ls -d $SPACK/dd4hep-*)/bin/thisdd4hep.sh
set -u
export PYTHONPATH="${ACTSDIR}python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${ACTSDIR}lib:/opt/odd-install/lib:${LD_LIBRARY_PATH:-}"
export ODD_PATH=/opt/odd
export ODD_DIR=/opt/odd
if [ -n "${ODD_TREE:-}" ]; then
  export ODD_PATH="$ODD_TREE"
  export ODD_DIR="$ODD_TREE"
  [ -n "${ODD_TREE_INSTALL:-}" ] && \
    export LD_LIBRARY_PATH="$ODD_TREE_INSTALL/lib:$LD_LIBRARY_PATH"
  echo "=== ODD tree: $ODD_PATH  factory: ${ODD_TREE_INSTALL:-image default}"
fi
'

if [[ "${1:-}" == "--shell" ]]; then
  exec docker run --rm -it -v "$PWD:/data" -v "$CACHE:/cache" -w /data \
    -e ODD_TREE="${ODD_TREE:-}" -e ODD_TREE_INSTALL="${ODD_TREE_INSTALL:-}" \
    --entrypoint /bin/bash "$IMG" -lc "$SETUP"' exec bash'
fi

[[ $# -ge 1 ]] || { echo "usage: $0 <script.py> [args...]  |  $0 --shell" >&2; exit 2; }
SCRIPT=$1; shift
printf -v ARGS '%q ' "$@"

exec docker run --rm -v "$PWD:/data" -v "$CACHE:/cache" -w /data \
  -e ODD_TREE="${ODD_TREE:-}" -e ODD_TREE_INSTALL="${ODD_TREE_INSTALL:-}" \
  --entrypoint /bin/bash "$IMG" -lc "$SETUP"' exec "$PY" '"/data/$SCRIPT $ARGS"
