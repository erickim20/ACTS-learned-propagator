#!/usr/bin/env bash
# What the image offers for putting a stepper into the CKF.
#
# The route is not a matter of taste: it depends on what is actually
# installed. Three possibilities, in
# decreasing order of how standard they are.
#
#   1. ActsExamples ships its own extension point. TrackFindingAlgorithm holds
#      a type-erased TrackFinderFunction, built by makeTrackFinderFunction(),
#      and that factory is the single place EigenStepper is named. If the
#      headers and the library are installed, a local translation unit can
#      define the same factory over LearnedStepper and link against everything
#      else unchanged. Nothing is patched and the reco chain stays ColliderML's.
#
#   2. ACTS is buildable from source here. Then the stepper goes where steppers
#      go, in Acts/Propagator, and the whole chain is stock. This is what
#      upstreaming would look like, and it is the only route that ends in a
#      contribution rather than a local artefact.
#
#   3. Neither. Then the honest answer is that this image is a runtime and not
#      a development environment, and the work moves to an ACTS dev image at
#      the matching version.
#
# This script only reports. It builds nothing.
#
#   cpp/probe_ckf_seam.sh
set -euo pipefail

IMG=ghcr.io/opendatadetector/sw@sha256:20e4df7b0d6befd70ce0113d852350b25cbd9aac659d1bc530036a2c85a68f89

docker run --rm --entrypoint /bin/bash "$IMG" -lc '
say() { printf "\n=== %s ===\n" "$1"; }

say "ACTS version, as the headers state it"
v=$(find / -name ActsVersion.hpp -path "*/Acts/*" 2>/dev/null | head -1)
echo "${v:-not found}"
[ -n "$v" ] && grep -E "VERSION_(MAJOR|MINOR|PATCH)|COMMIT_HASH" "$v" | head -8

say "spack, and whether a source tree survived the build"
command -v spack >/dev/null && spack find --paths acts 2>/dev/null | head -5 || echo "no spack on PATH"
find / -maxdepth 8 -type d -name "acts-44*" 2>/dev/null | head -5

say "1. ActsExamples headers"
find / -name "TrackFindingAlgorithm.hpp" 2>/dev/null | head -3
find / -type d -name ActsExamples 2>/dev/null | head -5

say "1. the seam itself, if the header is there"
h=$(find / -name "TrackFindingAlgorithm.hpp" 2>/dev/null | head -1)
if [ -n "$h" ]; then
  grep -n -A6 "TrackFinderFunction\|makeTrackFinderFunction" "$h" | head -40
else
  echo "no TrackFindingAlgorithm.hpp: the type-erased seam is not installed,"
  echo "so route 1 is out and only a source build can reach the stepper."
fi

say "1. ActsExamples libraries"
find / -name "libActsExamples*" 2>/dev/null | head -10 || true

say "1/2. CMake package config, i.e. can a downstream project find_package it"
find / -name "ActsConfig.cmake" -o -name "ActsExamplesConfig.cmake" 2>/dev/null | head -5

say "2. toolchain, if a source build is on the table"
cmake --version 2>/dev/null | head -1 || echo "no cmake"
c++ --version 2>/dev/null | head -1 || echo "no c++"
nproc

say "2. the dependencies a source build would need to find"
for p in eigen boost dd4hep root geant4 nlohmann-json; do
  printf "%-16s " "$p"
  find / -maxdepth 8 -type d -name "${p}-*" 2>/dev/null | head -1 || true
  echo
done

say "what addCKFTracks actually constructs"
python3 - <<PY 2>/dev/null || echo "acts python bindings not importable as python3"
import inspect, acts.examples.reconstruction as r
src = inspect.getsource(r.addCKFTracks)
print(inspect.getsourcefile(r))
for line in src.splitlines():
    if "TrackFinding" in line or "Function" in line:
        print("   ", line.strip())
PY

say "verdict inputs"
echo "route 1 needs: TrackFindingAlgorithm.hpp AND libActsExamples AND a CMake config"
echo "route 2 needs: a source tree or a matching upstream tag, plus the deps above"
'
