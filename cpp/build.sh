#!/usr/bin/env bash
# Build and run the kernel check and the latency benchmark.
#
# Eigen is taken from wherever it can be found rather than installed: the ODD
# image already carries 3.4.0, which is the exact version ACTS was built
# against in that image, so extracting it is both free and more faithful than
# a system package that might be a different release.
#
#   ./build.sh                  # find Eigen, build, run both
#   EIGEN=/path/to/eigen3 ./build.sh
set -euo pipefail
cd "$(dirname "$0")"

find_eigen() {
  if [[ -n "${EIGEN:-}" ]]; then echo "$EIGEN"; return; fi
  for p in \
      "$(brew --prefix eigen 2>/dev/null)/include/eigen3" \
      /opt/homebrew/include/eigen3 /usr/local/include/eigen3 \
      /usr/include/eigen3; do
    [[ -f "$p/Eigen/Dense" ]] && { echo "$p"; return; }
  done
  # last resort: pull it out of the ODD image
  local img cid dst
  # by digest, never by tag
  img=ghcr.io/opendatadetector/sw@sha256:20e4df7b0d6befd70ce0113d852350b25cbd9aac659d1bc530036a2c85a68f89
  docker image inspect "$img" >/dev/null 2>&1 || return 1
  # HOME, not /tmp: the WSL VM shuts down when idle and comes back with an
  # empty /tmp, so a cache there does not survive between two commands.
  dst="$HOME/.cache/odd-eigen"
  if [[ ! -f "$dst/Eigen/Dense" ]]; then
    mkdir -p "$dst.raw"
    cid=$(docker create "$img" /bin/true)
    # `--wildcards` with the pattern positional: GNU tar's syntax. `--include=`
    # is BSD's and GNU rejects it, silently here because of the `|| true`.
    docker export "$cid" 2>/dev/null \
      | (cd "$dst.raw" && tar -xf - --wildcards "*/eigen-3.4.0-*/include/eigen3/*" 2>/dev/null || true)
    docker rm "$cid" >/dev/null
    local found
    found=$(find "$dst.raw" -type d -name eigen3 | head -1)
    [[ -z "$found" ]] && return 1
    rm -rf "$dst"; mv "$found" "$dst"; rm -rf "$dst.raw"
  fi
  echo "$dst"
}

EIG=$(find_eigen) || { echo "no Eigen found; set EIGEN=" >&2; exit 1; }
echo "Eigen: $EIG"

FLAGS=(-std=c++20 -O2 -march=native -DNDEBUG -I"$EIG")
c++ "${FLAGS[@]}" test_kernel.cpp    -o test_kernel
c++ "${FLAGS[@]}" test_jacobian.cpp  -o test_jacobian

# The benchmark's third arm -- ACTS's own EigenStepper, so the ratio is not
# quoted against a baseline measured on another machine. Needs the ACTS headers
# and libActsCore, so it is a link, not just a compile; without them the two
# in-house arms still build and the binary says so in its own output.
if [[ -n "${ACTS_INC:-}" && -n "${BOOST_INC:-}" && -n "${ACTS_LIB:-}" ]]; then
  echo "ACTS: $ACTS_LIB  (bench gets its third arm)"
  # NOTE the missing -march=native, and do not put it back.
  #
  # libActsCore.so is a spack build with target=x86_64 -- generic, no AVX-512
  # anywhere in it. `-march=native` on this box is znver4, which moves Eigen's
  # alignment and vectorisation, and Eigen types cross the boundary into that
  # library in every ACTS call. The result is not a compile error and not a
  # wrong number: it is `std::bad_variant_access` out of the propagator on the
  # very first jump, which reads like an ACTS bug and is not one.
  #
  # Matching the library is also the fair comparison. All three arms are in one
  # translation unit, so this way all three get identical codegen instead of
  # handing the kernel AVX-512 and ACTS's core SSE2.
  BENCH_FLAGS=(-std=c++20 -O2 -DNDEBUG -I"$EIG" -DWITH_ACTS
               -I"$ACTS_INC" -I"$BOOST_INC"
               -L"$ACTS_LIB" -lActsCore -Wl,-rpath,"$ACTS_LIB")
else
  BENCH_FLAGS=("${FLAGS[@]}")
  echo "ACTS: not set (ACTS_INC/BOOST_INC/ACTS_LIB); bench builds without the"
  echo "      EigenStepper arm and will say so."
fi
c++ "${BENCH_FLAGS[@]}" bench_kernel.cpp -o bench_kernel
echo

[[ -f reference.bin ]] || {
  echo "run, from the repository root: python -m prop.export_kernel" >&2; exit 1; }
./test_kernel reference.bin || true
echo

if [[ -f reference_jac.bin ]]; then
  ./test_jacobian reference_jac.bin || true
  echo
else
  echo "cpp/reference_jac.bin missing; run, from the repository root:"
  echo "  python -m prop.export_jacobian --pairs teacher_phys.parquet"
  echo
fi

./bench_kernel reference.bin

# --- the ACTS-dependent check, if the headers are around.
#
# LearnedStepper is only worth anything if it is still substitutable for
# EigenStepper, and that is a compile-time question with a compile-time answer.
# Extract the trees with acts_headers.sh; skipped silently otherwise, because
# the kernel above stands on its own.
if [[ -n "${ACTS_INC:-}" && -n "${BOOST_INC:-}" ]]; then
  echo
  cat > /tmp/acts_substitution.cpp <<'EOF'
#include "LearnedStepper.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/Propagator.hpp"
using P = Acts::Propagator<collider_ml::LearnedStepper, Acts::Navigator>;
static_assert(Acts::StepperConcept<collider_ml::LearnedStepper>);
int main() { return sizeof(P) > 0 ? 0 : 1; }
EOF
  if c++ -std=c++20 -fsyntax-only -I. -I"$ACTS_INC" -I"$EIG" -I"$BOOST_INC" \
        /tmp/acts_substitution.cpp; then
    echo "ACTS: StepperConcept<LearnedStepper> holds, and"
    echo "      Propagator<LearnedStepper, Navigator> instantiates."
  else
    echo "ACTS: SUBSTITUTION BROKEN -- see above" >&2
    exit 1
  fi
fi
