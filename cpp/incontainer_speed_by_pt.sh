#!/usr/bin/env bash
# In-container half of run_speed_by_pt.sh: build bench_kernel ONCE, then run it
# once per pT bin into one CSV.
#
# Build once and not once per bin, because the curve requires one
# binary for the whole curve. Five separate compiles of the same source with the
# same flags would almost certainly agree, but "almost certainly" is not a
# property a latency curve should rest on.
#
# The compile line is incontainer_build.sh's, including the missing
# -march=native. libActsCore.so is a spack build targeting generic x86_64;
# building against it at a different baseline moves Eigen's alignment across the
# library boundary and the result is std::bad_variant_access out of the
# propagator on the first jump, which reads exactly like an ACTS bug.
# cpp/README.md 3. It is also the fair comparison: all three arms are in one
# translation unit, so all three get identical codegen.
#
# Driven by environment: JUMPS FIELD CSV REPEATS BINS.
set -o pipefail
SPACK=/spack/opt/spack/linux-x86_64

set +u
source "$(ls -d "$SPACK"/dd4hep-*)/bin/thisdd4hep.sh"
set -eu

export LD_LIBRARY_PATH="/opt/odd-install/lib:${LD_LIBRARY_PATH:-}"
ACTS_DIR=$(ls -d "$SPACK"/acts-*/ | head -1)
DD4HEP_DIR=$(ls -d "$SPACK"/dd4hep-*/ | head -1)
ROOT_DIR=$(ls -d "$SPACK"/root-*/ | head -1)
EIGEN_DIR=$(ls -d "$SPACK"/eigen-*/ | head -1)
BOOST_DIR=$(ls -d "$SPACK"/boost-*/ | head -1)

OUT=${TMPDIR:-/tmp}/bench_kernel
echo "=== building $OUT at the generic x86-64 baseline"
g++ -std=c++20 -O2 -DNDEBUG -DWITH_ACTS -I. \
  -I"$ACTS_DIR/include" -I"$EIGEN_DIR/include/eigen3" \
  -I"$BOOST_DIR/include" -I"$DD4HEP_DIR/include" \
  -I"$ROOT_DIR/include" -I"$ROOT_DIR/include/root" \
  bench_kernel.cpp \
  -L"$ACTS_DIR/lib" -L"$ACTS_DIR/lib64" -lActsCore -lActsPluginDD4hep \
  -L"$DD4HEP_DIR/lib" -lDDCore \
  -L"$ROOT_DIR/lib" -L"$ROOT_DIR/lib/root" -lCore -lGeom \
  -Wl,-rpath,"$ACTS_DIR/lib" -Wl,-rpath,"$ACTS_DIR/lib64" \
  -Wl,-rpath,"$DD4HEP_DIR/lib" -Wl,-rpath,"$ROOT_DIR/lib/root" \
  -o "$OUT"
echo "built, md5 $(md5sum "$OUT" | cut -d' ' -f1)"
echo

# A fresh CSV per invocation: bench_kernel appends, so a stale file would grow
# a second copy of every bin and the plotting would average two runs silently.
rm -f "$CSV"

# Each bin names its own jump file, because the eight bins of HANDOFF item 1
# span three generator runs: 0.5 to 1, 1 to 50, and 50 to 100 GeV. One binary
# still produces every cell, which is what section 14 requires; only the
# transports handed to it change.
for bin in $BINS; do
  lo=${bin%%:*}
  rest=${bin#*:}
  hi=${rest%%:*}
  src=${rest#*:}
  echo "############ pT $lo to $hi GeV   from $(basename "$src")"
  "$OUT" "$src" "$FIELD" --pt "$lo" "$hi" --repeats "$REPEATS" --csv "$CSV" \
    2>&1 | grep -E "^(build:|[0-9]+ jumps|--- (section 14|per step)|pT |  (acts_|kernel_|helix_|ratio|attempted|counters|a missed)|appended)" \
    || true
  echo
done

echo "=== $CSV"
cat "$CSV"
