#!/usr/bin/env bash
# Prepare /data/baseline0 for item 0: the stock ACTS CKF baseline on a
# ColliderML ttbar sample produced locally.
#
# Two things are seeded deliberately rather than left to
# scripts/cli/setup_container_env.sh:
#
#  1. The ODD geometry. That script clones ODD v4.0.4 from CERN GitLab. Every
#     geometry number in NOTES (the 18,824 planes of 6.2, the volume ids in
#     full_chain_odd.py) was measured on the ODD inside the pinned image, so the
#     baseline has to use the same one or it is not the same detector. Copying
#     /opt/odd and /opt/odd-install into the cache satisfies both of that
#     script's guards, so it neither clones nor rebuilds.
#
#  2. The Geant4 physics datasets. The image ships an empty data directory and
#     the setup script downloads ~2 GB behind a 600 s timeout. Doing it once
#     here, uncapped, keeps that out of the run.
set -euo pipefail

# /data is root-owned and this runs as eric. $HOME is the same ext4 filesystem.
ROOT="${BASELINE0_ROOT:-$HOME/baseline0}"
IMG=ghcr.io/opendatadetector/sw@sha256:20e4df7b0d6befd70ce0113d852350b25cbd9aac659d1bc530036a2c85a68f89

mkdir -p "$ROOT"/{cache,out,configs}

if [ ! -f "$ROOT/cache/odd-v4/xml/OpenDataDetector.xml" ]; then
  echo "seeding ODD geometry from the image..."
  docker run --rm -v "$ROOT/cache:/cache" --entrypoint /bin/bash "$IMG" -c '
    mkdir -p /cache/odd-v4 /cache/odd-v4-install
    cp -a /opt/odd/. /cache/odd-v4/
    cp -a /opt/odd-install/. /cache/odd-v4-install/
  '
  echo "  ODD seeded."
else
  echo "ODD already seeded."
fi

if [ -z "$(ls -A "$ROOT/cache/g4data" 2>/dev/null)" ]; then
  echo "downloading Geant4 datasets (~2 GB, once)..."
  mkdir -p "$ROOT/cache/g4data"
  docker run --rm -v "$ROOT/cache:/cache" --entrypoint /bin/bash "$IMG" -c '
    set -e
    SPACK=/spack/opt/spack/linux-x86_64
    source $(ls $SPACK/geant4-*/bin/geant4.sh | head -1)
    G4=$(ls -d $SPACK/geant4-*/share/Geant4/data | head -1)
    download_geant4_datasets.sh
    for d in "$G4"/*/; do
      [ -d "$d" ] || continue
      mv "$d" /cache/g4data/$(basename "$d")
    done
    ls /cache/g4data
  '
  echo "  Geant4 datasets cached."
else
  echo "Geant4 datasets already cached."
fi

echo "--- $ROOT:"
du -sh "$ROOT"/cache/* 2>/dev/null || true
