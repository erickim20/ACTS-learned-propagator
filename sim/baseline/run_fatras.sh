#!/usr/bin/env bash
# Run sim/fatras_reco.py in the pinned image against a merged_events.hepmc3
# already produced by the ColliderML chain.
#
#   run_fatras.sh <hepmc3 path under out/> <out subdir under out/> <events>
#     e.g. run_fatras.sh validation/run0/merged_events.hepmc3 fatras_smoke 2
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${BASELINE0_ROOT:-$HOME/baseline0}"
REPO="$(cd "$HERE/../.." && pwd)"
CML="${COLLIDERML_REPO:-/data/ColliderML-Production}"
IMG=ghcr.io/opendatadetector/sw@sha256:20e4df7b0d6befd70ce0113d852350b25cbd9aac659d1bc530036a2c85a68f89

IN=$1; OUTSUB=$2; NEV=$3
mkdir -p "$ROOT/out/$OUTSUB"

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
'

CMD='exec "$PY" /repo/sim/fatras_reco.py'
CMD="$CMD --input /output/$IN"
CMD="$CMD --output /output/$OUTSUB"
CMD="$CMD --events $NEV"
CMD="$CMD --digi-config /workspace/scripts/simulation/odd-full-geo-digi-config.json"
CMD="$CMD --seed ${SEED:-42}"

exec docker run --rm \
  -v "$REPO:/repo:ro" \
  -v "$ROOT/out:/output" \
  -v "$CML:/workspace:ro" \
  --entrypoint /bin/bash "$IMG" -lc "$SETUP $CMD"
