#!/usr/bin/env bash
# In-container half of build_learned_ckf.sh. Compiles cpp/learned_ckf.cpp into
# the python extension the spack Python 3.13 can import, next to the source.
#
# Everything comes from the image's own spack prefixes, because the module has
# to share an ABI with the prebuilt acts bindings: same g++ 13.3, same
# pybind11 3.0.0, same libstdc++, same generic x86_64 baseline. NO
# -march=native (Eigen alignment across the libActsCore boundary
# turns it into std::bad_variant_access on the first jump).
set -euo pipefail
SPACK=/spack/opt/spack/linux-x86_64
cd /repo/cpp

[[ -s acts_examples_src/TrackFindingAlgorithm.hpp ]] || {
  echo "acts_examples_src/ is empty; run cpp/fetch_ckf_src.sh on the host" >&2
  exit 2
}

ACTS_DIR=$(ls -d "$SPACK"/acts-*/ | head -1)
EIGEN_DIR=$(ls -d "$SPACK"/eigen-*/ | head -1)
BOOST_DIR=$(ls -d "$SPACK"/boost-*/ | head -1)
TBB_DIR=$(ls -d "$SPACK"/intel-tbb-*/ | head -1)
PY_DIR=$(ls -d "$SPACK"/python-3.13*/ | head -1)
PB_DIR=$(ls -d "$SPACK"/py-pybind11-*/ | head -1)
PYINC=$(ls -d "$PY_DIR"/include/python3.13*/ | head -1)
EXT=$("$PY_DIR/bin/python3" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# The md5 of the weights this build is about to compile in. Computed here
# rather than carried in a comment, because a comment is what the `source:`
# line already is, and a comment is not evidence.
# `NoiseTable::load` compares it against the one the Q table records, so a
# table measured on another model is refused rather than armed.
WMD5=$(md5sum gtheta_weights.hpp | cut -c1-32)

echo "ACTS     $ACTS_DIR"
echo "pybind11 $PB_DIR"
echo "python   $PYINC"
echo "ext      learned_ckf$EXT"
echo "weights  $WMD5  (gtheta_weights.hpp)"

time g++ -std=c++20 -O2 -DNDEBUG -shared -fPIC \
  -DGTHETA_WEIGHTS_MD5="\"$WMD5\"" \
  -I. \
  -I"$ACTS_DIR/include" \
  -I"$EIGEN_DIR/include/eigen3" \
  -I"$BOOST_DIR/include" \
  -I"$TBB_DIR/include" \
  -I"$PYINC" \
  -I"$PB_DIR/include" \
  learned_ckf.cpp \
  -L"$ACTS_DIR/lib" -lActsCore \
  -Wl,-rpath,"$ACTS_DIR/lib" \
  -o "learned_ckf$EXT"

echo "built cpp/learned_ckf$EXT"
