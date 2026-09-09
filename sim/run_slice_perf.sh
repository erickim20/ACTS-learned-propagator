#!/usr/bin/env bash
# Host half of sim/slice_perf.py.
#
#   sim/run_slice_perf.sh                                 # item 11's four arms
#   sim/run_slice_perf.sh --stage ckf                     # the _ckf files, not _ambi
#   sim/run_slice_perf.sh 'muon=ckf_muon_learnedonQ'      # any other subdir
#
# NOTES 6.11 and 6.13 quote the `_ambi` files, which is the default here.
#
# Reads only, and writes nothing: $BASELINE0_ROOT/out is mounted read-only.
#
# It needs ROOT, not ACTS, so the environment is smaller than sim/run_in_odd.sh's.
# ROOT's python module sits in $ROOTSYS/lib/root, not in $ROOTSYS/lib, and
# `thisroot.sh` in this spack build refuses to source unless you cd into the
# prefix first, so both paths are set by hand below. `python3` on PATH is 3.12
# and has no ROOT; the spack python 3.13 is the one the bindings are built for.
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT_DIR="${BASELINE0_ROOT:-$HOME/baseline0}"
IMG=ghcr.io/opendatadetector/sw@sha256:20e4df7b0d6befd70ce0113d852350b25cbd9aac659d1bc530036a2c85a68f89

[[ -d "$ROOT_DIR/out/runs" ]] || { echo "no $ROOT_DIR/out/runs" >&2; exit 2; }

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
  -lc "$SETUP"' exec "$PY" /repo/sim/slice_perf.py '"$ARGS"
