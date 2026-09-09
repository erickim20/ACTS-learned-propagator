#!/usr/bin/env bash
# Second pass on the CKF seam. `probe_ckf_seam.sh` left four things open.
#
#   1. It printed the ACTS prefix but no version or commit hash, because its
#      grep pattern matched nothing in ActsVersion.hpp. The commit hash is the
#      one thing that says which upstream tree 44.99.99-colliderml-arrow came
#      from, and route 2 cannot start without it.
#   2. It cut the ActsExamples library list at ten entries, so whether
#      libActsExamplesTrackFinding exists is still unknown.
#   3. It concluded route 1 is out from a missing header. A missing header does
#      not mean a missing seam: the algorithm may be compiled into a library and
#      exposed through the Python bindings, which is a different seam in the
#      same place.
#   4. It asked `python3` what addCKFTracks constructs, and `python3` on PATH
#      cannot import acts. reconstruction.py is
#      plain Python sitting in the image, so read the file instead of importing
#      the package.
#
# This script only reports. It builds nothing.
#
#   cpp/probe_ckf_seam2.sh
set -euo pipefail

IMG=ghcr.io/opendatadetector/sw@sha256:20e4df7b0d6befd70ce0113d852350b25cbd9aac659d1bc530036a2c85a68f89

docker run --rm --entrypoint /bin/bash "$IMG" -lc '
set -u
say() { printf "\n=== %s ===\n" "$1"; }

V=$(find / -name ActsVersion.hpp -path "*/Acts/*" 2>/dev/null | head -1)
PREFIX=$(dirname "$(dirname "$(dirname "$V")")")
echo "prefix: $PREFIX"

say "1. the version and the commit hash, whatever they are called"
grep -nEi "version|commit|hash" "$V" | head -40

say "1. spack provenance: which source, which ref, which patches"
ls -a "$PREFIX/.spack" 2>/dev/null || echo "no .spack manifest in the prefix"
for f in "$PREFIX/.spack/spec.json" "$PREFIX/.spack/spec.yaml"; do
  [ -f "$f" ] && echo "--- $f" && python3 - "$f" <<PY
import json, sys
d = json.load(open(sys.argv[1]))
nodes = d.get("spec", {}).get("nodes", d.get("spec", []))
if isinstance(nodes, dict): nodes = [nodes]
for n in nodes:
    if not isinstance(n, dict): continue
    if "acts" not in str(n.get("name", "")): continue
    for k in ("name","version","hash","package_hash"):
        if k in n: print("  %-14s %s" % (k, n[k]))
    for k in ("parameters","patches","external","dev_path","annotations","compiler"):
        if k in n: print("  %-14s %s" % (k, json.dumps(n[k])[:600]))
PY
done

say "1. the spack recipe, which is where a git ref would be written"
r=$(find / -name package.py -path "*acts*" 2>/dev/null | head -3)
echo "${r:-no acts package.py in the image}"
for p in $r; do
  echo "--- $p"
  grep -nE "git|url|version\(|commit|branch|tag" "$p" | head -25
done
command -v spack || ls -d /spack/bin 2>/dev/null || echo "no spack driver"

say "2. every ActsExamples library, not the first ten"
ls -1 "$PREFIX/lib" | grep -i actsexamples || echo "none"
echo "--- count:"; ls -1 "$PREFIX/lib" | grep -ci actsexamples || true

say "2. every ActsExamples header directory that got installed"
ls -1 "$PREFIX/include/ActsExamples" 2>/dev/null || echo "no ActsExamples include tree"

say "3. the seam by name, anywhere in the installed headers"
grep -rl "TrackFinderFunction\|makeTrackFinderFunction" "$PREFIX/include" 2>/dev/null | head -10 \
  || echo "not named in any installed header"

say "3. the seam by symbol, in the libraries"
if command -v nm >/dev/null; then
  for so in "$PREFIX"/lib/libActsExamples*.so "$PREFIX"/lib/*ActsPythonBindings*.so; do
    [ -e "$so" ] || continue
    n=$(nm -DC --defined-only "$so" 2>/dev/null | grep -ci "TrackFinderFunction\|TrackFindingAlgorithm" || true)
    [ "$n" != "0" ] && echo "$(basename "$so"): $n matching defined symbols"
  done
else
  echo "no nm"
fi

say "3. makeTrackFinderFunction as a defined symbol, demangled"
for so in "$PREFIX"/lib/*.so; do
  nm -DC --defined-only "$so" 2>/dev/null | grep -i "makeTrackFinderFunction" | head -3 \
    | sed "s|^|$(basename "$so"): |"
done | head -10

say "4. the python bindings module and the reconstruction source"
find / -name "*ActsPythonBindings*" 2>/dev/null | head -5
R=$(find / -name reconstruction.py -path "*acts/examples*" 2>/dev/null | head -1)
echo "reconstruction.py: ${R:-not found}"

say "4. every add* entry point reconstruction.py defines"
[ -n "$R" ] && grep -n "^def " "$R" | head -30

say "4. where the track finder is named in reconstruction.py"
[ -n "$R" ] && grep -n "TrackFinding\|findTracks\|makeTrackFinderFunction\|CkfConfig" "$R" | head -30

say "4. how the bindings expose it, if at all"
BSO=$(find / -name "*ActsPythonBindings*.so" 2>/dev/null | head -1)
echo "bindings: ${BSO:-none}"
if [ -n "$BSO" ]; then
  echo "--- defined symbols naming the algorithm:"
  nm -DC "$BSO" 2>/dev/null | grep -c "TrackFindingAlgorithm" || true
  echo "--- what pybind registers, by string:"
  strings "$BSO" | grep -i "makeTrackFinderFunction\|TrackFindingAlgorithm\|findTracks" | sort -u | head -10
fi
for f in $(find / -name "*.pyi" -path "*acts*" 2>/dev/null | head -3); do
  echo "--- $f"
  grep -n "TrackFinding\|TrackFinderFunction" "$f" | head -10
done

say "4b. the interpreter that can actually import acts"
find / -maxdepth 7 -path "*python-3*/bin/python3" -type f 2>/dev/null | head -3
head -3 "$(find / -name "*.pth" -path "*acts*" 2>/dev/null | head -1)" 2>/dev/null || true

say "6. what spack patched on top of the commit, i.e. why -dirty"
PKG=$(find / -name package.py -path "*colliderml*acts*" 2>/dev/null | head -1)
echo "recipe: ${PKG:-none}"
[ -n "$PKG" ] && ls -1 "$(dirname "$PKG")"
[ -n "$PKG" ] && grep -nE "^\s*patch\(|def patch|filter_file" "$PKG" | head -20
echo "--- patches recorded in the spec:"
python3 - "$PREFIX/.spack/spec.json" <<PY
import json, sys
d = json.load(open(sys.argv[1]))
nodes = d.get("spec", {}).get("nodes", [])
for n in nodes:
    if n.get("name") == "acts":
        print("  patches:", n.get("patches"))
        print("  param patches:", n.get("parameters", {}).get("patches"))
PY

say "5. route 2 sanity: are the core headers whole"
for h in Propagator/EigenStepper.hpp Propagator/Propagator.hpp \
         TrackFinding/CombinatorialKalmanFilter.hpp TrackFitting/KalmanFitter.hpp; do
  printf "%-52s " "$h"
  [ -f "$PREFIX/include/Acts/$h" ] && echo present || echo MISSING
done
'
