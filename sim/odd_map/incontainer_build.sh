#!/usr/bin/env bash
# In-container half of build_odd_map.sh: compile the ODD factory library with
# the field plugin in it.
#
# Driven by environment: ODD_MAP_TREE ODD_MAP_INSTALL, both container paths.
#
# No `set -e` and no `set -u` while sourcing the image's own environment; see
# sim/baseline/incontainer.sh for why that costs a silent exit otherwise.
#
# /root/.bashrc is what sets CMAKE_PREFIX_PATH for the spack packages, and
# DD4hepConfig.cmake needs it: it does find_dependency(Boost), and without the
# prefix path cmake looks in /usr and fails with "Could NOT find Boost". The
# .bashrc guards itself behind an interactive-shell check, so PS1 is set first,
# which is the same bypass setup_container_env.sh:110-114 uses.
set -o pipefail

SPACK=/spack/opt/spack/linux-x86_64
set +u
export PS1="${PS1:-odd-map}"
source /root/.bashrc 2>/dev/null || true
source "$(ls -d $SPACK/dd4hep-*)/bin/thisdd4hep.sh"
set -u
set -e

: "${ODD_MAP_TREE:?}"
: "${ODD_MAP_INSTALL:?}"

echo "=== DD4hep $(ls -d $SPACK/dd4hep-* | head -1)"
echo "=== tree    $ODD_MAP_TREE"
echo "=== install $ODD_MAP_INSTALL"

BUILD=/tmp/odd-map-build
rm -rf "$BUILD"
mkdir -p "$BUILD"
cd "$BUILD"

# Never -march=native here. The image's libraries are a generic x86-64 spack
# build and wider instructions across that boundary fail at run time in ways
# that look like somebody else's bug.
cmake "$ODD_MAP_TREE" -DCMAKE_INSTALL_PREFIX="$ODD_MAP_INSTALL" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j"$(nproc)"
make install

echo "=== installed:"
ls -la "$ODD_MAP_INSTALL/lib/"
echo "=== factories the library declares:"
cat "$ODD_MAP_INSTALL/lib/libOpenDataDetector.components"
echo "=== the field plugin has to be one of them:"
grep -q "ODDFieldMapXyz" "$ODD_MAP_INSTALL/lib/libOpenDataDetector.components" \
  && echo "ODDFieldMapXyz registered" \
  || { echo "ODDFieldMapXyz MISSING from the rootmap" >&2; exit 3; }
