#!/usr/bin/env bash
# Build the map-field ODD: convert the map, write the tree, compile the factory.
#
#   sim/odd_map/build_odd_map.sh
#   sim/odd_map/build_odd_map.sh --factory-only
#
# Three stages, each skippable once done, because the first is a 515 MB csv
# parse and the second copies a 1.1 GB tree.
#
# `--factory-only` refreshes the plugin source in an existing tree and rebuilds
# the library, leaving the compact and the map file alone. It is for a change
# to `cpp/ODDFieldMapXyz.cpp` that does not change the field, and it is the
# difference between one minute and eleven.
#
#   1. odd-bfield.csv -> oddb.npz -> odd-bfield-xyz.bin   (host venv)
#   2. odd-v4 -> odd-map, compact patched, plugin added   (host venv)
#   3. cmake + make + install                             (pinned image)
#
# The result is not the ODD. `sim/odd_map/make_tree.py` prints the whole diff
# against the release so that what "not the ODD" means is one field element,
# one factory source and one data file, and can be read rather than trusted.
#
# Paths are on the host; the tree is mounted into the image at /cache/odd-map,
# which is the path the compact records.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ROOT="${BASELINE0_ROOT:-$HOME/baseline0}"
CACHE="$ROOT/cache"
SRC="$CACHE/odd-v4"
DST="${ODD_MAP_TREE:-$CACHE/odd-map}"
INSTALL="${ODD_MAP_INSTALL:-$CACHE/odd-map-install}"
WORK="${ODD_MAP_WORK:-$HOME/exp/odd_map}"
IMG=ghcr.io/opendatadetector/sw@sha256:20e4df7b0d6befd70ce0113d852350b25cbd9aac659d1bc530036a2c85a68f89

# Where the compact will look for the map. A path inside the image.
MAP_IN_IMAGE=/cache/odd-map/data/odd-bfield-xyz.bin

FACTORY_ONLY=""
[[ "${1:-}" == "--factory-only" ]] && FACTORY_ONLY=1

PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || { echo "no $PY" >&2; exit 2; }
[[ -d "$SRC" ]] || { echo "no unmodified ODD at $SRC" >&2; exit 2; }

mkdir -p "$WORK"
cd "$REPO"

if [[ -n "$FACTORY_ONLY" ]]; then
  [[ -f "$DST/xml/OpenDataDetector.xml" ]] || {
    echo "--factory-only but no tree at $DST; run without the flag first" >&2
    exit 2; }
  echo "=== --factory-only: refreshing the plugin in $DST, compact untouched"
  cp -p "$REPO/cpp/ODDFieldMapXyz.cpp" "$DST/factory/field/ODDFieldMapXyz.cpp"
  grep -n "<field" "$DST/xml/OpenDataDetector.xml"
  exec docker run --rm \
    -v "$CACHE:/cache" -v "$HERE:/work:ro" \
    -e ODD_MAP_TREE=/cache/"$(basename "$DST")" \
    -e ODD_MAP_INSTALL=/cache/"$(basename "$INSTALL")" \
    --entrypoint /bin/bash "$IMG" /work/incontainer_build.sh
fi

# ---- 1. the map, csv -> npz -> 3-component binary -------------------------
# The csv is the one the ODD ships. `data/odd-bfield.csv` in the image and in
# the odd-v4 clone are the same file (md5 ceb0f77c1caaa1efa7c3a81086e18c49) and
# the repository's own odd-bfield.txt is that file with the header stripped.
NPZ="$WORK/oddb.npz"
BIN="$WORK/odd-bfield-xyz.bin"
if [[ ! -f "$NPZ" ]]; then
  echo "=== 1a. $SRC/data/odd-bfield.csv -> $NPZ"
  "$PY" -m prop.build_fieldmap "$SRC/data/odd-bfield.csv" --out "$NPZ"
else
  echo "=== 1a. $NPZ present, skipped"
fi
if [[ ! -f "$BIN" ]]; then
  echo "=== 1b. $NPZ -> $BIN"
  "$PY" -m prop.export_fieldbin --npz "$NPZ" --components xyz --out "$BIN"
else
  echo "=== 1b. $BIN present, skipped"
fi

# ---- 2. the tree ----------------------------------------------------------
echo "=== 2. $SRC -> $DST"
"$PY" "$HERE/make_tree.py" --src "$SRC" --dst "$DST" --map "$BIN" \
  --map-container-path "$MAP_IN_IMAGE" --plugin "$REPO/cpp/ODDFieldMapXyz.cpp"

# ---- 3. the factory library ----------------------------------------------
echo "=== 3. build libOpenDataDetector.so with the field plugin"
mkdir -p "$INSTALL"
exec docker run --rm \
  -v "$CACHE:/cache" \
  -v "$HERE:/work:ro" \
  -e ODD_MAP_TREE=/cache/"$(basename "$DST")" \
  -e ODD_MAP_INSTALL=/cache/"$(basename "$INSTALL")" \
  --entrypoint /bin/bash "$IMG" /work/incontainer_build.sh
