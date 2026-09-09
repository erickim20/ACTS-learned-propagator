#!/usr/bin/env bash
# Run a script from this repository inside the pinned image with ROOT importable.
#
#   sim/run_root.sh sim/signal_eff.py runs/mu_s2_b1_eval
#
# sim/run_slice_perf.sh is the same environment wired to one script; this is the
# general form, for the offline readers. It needs
# ROOT and not ACTS, so the setup is smaller than sim/run_in_odd.sh's.
#
# ROOT's python module sits in $ROOTSYS/lib/root, not $ROOTSYS/lib, and
# thisroot.sh in this spack build refuses to source unless you cd into the
# prefix first, so both paths are set by hand. `python3` on PATH is 3.12 and has
# no ROOT; the spack python 3.13 is the one the bindings are built for.
#
# $BASELINE0_ROOT/out is mounted read-only at /out. Anything the script writes
# has to go under /repo, which is also read-only, so readers here print rather
# than write.
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT_DIR="${BASELINE0_ROOT:-$HOME/baseline0}"
IMG=ghcr.io/opendatadetector/sw@sha256:20e4df7b0d6befd70ce0113d852350b25cbd9aac659d1bc530036a2c85a68f89

[[ $# -ge 1 ]] || { echo "usage: $0 <script under the repo> [args...]" >&2; exit 2; }
SCRIPT=$1; shift

SETUP='
SPACK=/spack/opt/spack/linux-x86_64
ROOTSYS=$(ls -d $SPACK/root-*/ | head -1); ROOTSYS=${ROOTSYS%/}
PY=$(ls -d $SPACK/python-3.13*/bin/python3 | head -1)
export ROOTSYS
export PYTHONPATH="$ROOTSYS/lib/root:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$ROOTSYS/lib/root:${LD_LIBRARY_PATH:-}"
export CPPYY_API_PATH=none
'

ARGS=""
if [[ $# -gt 0 ]]; then printf -v ARGS '%q ' "$@"; fi

exec docker run --rm \
  -v "$PWD:/repo:ro" \
  -v "$ROOT_DIR/out:/out:ro" \
  --entrypoint /bin/bash "$IMG" \
  -lc "$SETUP"' exec "$PY" '"/repo/$SCRIPT $ARGS"
