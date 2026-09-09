#!/usr/bin/env bash
# The in-container half of run_in_image.sh. Kept in a file rather than a -c
# string: this call goes PowerShell -> wsl -> bash -> docker -> bash, and every
# layer of that chain eats one round of quoting, which loses variables and
# truncates strings at the first space.
#
#   incontainer_build.sh <probe-name> [args...]
set -o pipefail
SPACK=/spack/opt/spack/linux-x86_64

[[ $# -ge 1 ]] || { echo "usage: $0 <probe-name> [args...]" >&2; exit 2; }
PROBE=$1
shift
[[ -f "$PROBE.cpp" ]] || { echo "no such source: $PROBE.cpp" >&2; exit 2; }

# thisdd4hep.sh is not optional and its absence is not loud. Without it the
# binary links, loads, and dies inside fromCompact with
#
#   Failed to locate plugin to interprete files of type "lccdd"
#
# followed by a ROOT stack trace, because DD4hep's plugin service finds its
# factories through DD4HEP_LIBRARY_PATH rather than through the linker. The
# error names the XML, so it reads as a problem with the detector description.
set +u
source "$(ls -d "$SPACK"/dd4hep-*)/bin/thisdd4hep.sh"
set -eu

# libOpenDataDetector.so is in /opt/odd-install/lib, not where the loader error
# says to look.
export LD_LIBRARY_PATH="/opt/odd-install/lib:${LD_LIBRARY_PATH:-}"

# A tree other than the image's own, set by `run_in_image.sh --map`. It goes in
# FRONT, because the factory that carries the compact's field plugin has to be
# the one found. sim/odd_tree.sh does the same for the simulation side, and its
# warning applies here: a new compact with the old library fails loudly, an old
# compact with the new library does not fail at all and the detector is a
# uniform 2 T.
if [[ -n "${ODD_TREE_INSTALL:-}" ]]; then
  export LD_LIBRARY_PATH="$ODD_TREE_INSTALL/lib:$LD_LIBRARY_PATH"
  echo "ODD tree factory  $ODD_TREE_INSTALL/lib"
fi
ACTS_DIR=$(ls -d "$SPACK"/acts-*/ 2>/dev/null | head -1)
DD4HEP_DIR=$(ls -d "$SPACK"/dd4hep-*/ 2>/dev/null | head -1)
ROOT_DIR=$(ls -d "$SPACK"/root-*/ 2>/dev/null | head -1)
EIGEN_DIR=$(ls -d "$SPACK"/eigen-*/ 2>/dev/null | head -1)
BOOST_DIR=$(ls -d "$SPACK"/boost-*/ 2>/dev/null | head -1)

for v in ACTS_DIR DD4HEP_DIR ROOT_DIR EIGEN_DIR BOOST_DIR; do
  if [[ -z "${!v}" ]]; then
    echo "missing $v under $SPACK" >&2
    exit 2
  fi
done
echo "ACTS   $ACTS_DIR"
echo "DD4hep $DD4HEP_DIR"
echo "ROOT   $ROOT_DIR"

OUT=${TMPDIR:-/tmp}/$PROBE

# NO -march=native, and do not add it. libActsCore.so is a spack build
# targeting generic x86_64; building against it at a different baseline moves
# Eigen's alignment across the library boundary and the result is
# std::bad_variant_access out of the propagator on the first jump, which reads
# exactly like an ACTS bug. cpp/README.md 3.
# -DWITH_ACTS unconditionally: this script only ever runs where libActsCore is,
# and it links against it below. bench_kernel.cpp is the only probe that reads
# the macro, and without it that file silently drops the two arms that make its
# ratio a measurement rather than a comparison against local code alone.
g++ -std=c++20 -O2 -DNDEBUG -DWITH_ACTS -I. \
  -I"$ACTS_DIR/include" -I"$EIGEN_DIR/include/eigen3" \
  -I"$BOOST_DIR/include" -I"$DD4HEP_DIR/include" \
  -I"$ROOT_DIR/include" -I"$ROOT_DIR/include/root" \
  "$PROBE.cpp" \
  -L"$ACTS_DIR/lib" -L"$ACTS_DIR/lib64" -lActsCore -lActsPluginDD4hep \
  -lActsPluginRoot \
  -L"$DD4HEP_DIR/lib" -lDDCore \
  -L"$ROOT_DIR/lib" -L"$ROOT_DIR/lib/root" -lCore -lGeom \
  -Wl,-rpath,"$ACTS_DIR/lib" -Wl,-rpath,"$ACTS_DIR/lib64" \
  -Wl,-rpath,"$DD4HEP_DIR/lib" -Wl,-rpath,"$ROOT_DIR/lib/root" \
  -o "$OUT"

echo "built $OUT"
exec "$OUT" "$@"
