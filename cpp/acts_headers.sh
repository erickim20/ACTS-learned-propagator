#!/usr/bin/env bash
# Pull the ACTS, Eigen and Boost headers out of the ODD image so the ACTS-side
# compile checks can run without installing anything.
#
# `docker export | tar -x` gets them out without ever running the image, and
# they are the exact versions ACTS 44.99.99 was built against in there, which a
# system package would not be.
#
#   eval "$(./acts_headers.sh)"     # sets ACTS_INC, BOOST_INC, EIGEN_INC
#   ./build.sh                      # now runs the substitution check too
set -euo pipefail

# Under HOME, not /tmp. The WSL VM shuts down when idle and comes back with an
# empty /tmp, so a cache there is gone between two commands in the same session
# and the extraction (a full `docker export` of a multi-GB image) is paid again
# each time. HOME is on the ext4 disk and survives.
DEST="${ODD_HEADERS:-$HOME/.cache/odd-headers}"
# By digest, never by tag. `:latest` is a different ACTS build, so a tag lookup
# extracts headers that do not match the library everything is linked against,
# and nothing says so until a template fails to substitute.
IMG=ghcr.io/opendatadetector/sw@sha256:20e4df7b0d6befd70ce0113d852350b25cbd9aac659d1bc530036a2c85a68f89
docker image inspect "$IMG" >/dev/null 2>&1 || IMG=""

pull() {  # pull <name> <tar-glob> <strip>
  local name="$1"
  local glob="$2"
  local strip="$3"
  local out="$DEST/$name"
  # Non-empty, not merely present. An extraction that fails leaves the mkdir
  # behind, and a `-d` test then reports that directory as done forever. That
  # is how three empty trees survived a `set -e` script: the failure was inside
  # a subshell ending in `|| true`, with its message sent to /dev/null.
  if [[ -d "$out" ]] && [[ -n "$(find "$out" -type f -print -quit 2>/dev/null)" ]]; then
    echo "$out"
    return
  fi
  [[ -z "$IMG" ]] && { echo "" ; return; }
  rm -rf "$out"
  mkdir -p "$out"
  local cid
  cid=$(docker create "$IMG" /bin/true)
  # `--wildcards` and the pattern as a POSITIONAL member, which is GNU tar's
  # syntax. `--include=` is BSD tar's; GNU rejects it outright, so on Linux
  # this extracted nothing at all while looking like it had worked.
  #
  # NOTE the strip count. Extracting without it flattens every module into one
  # directory, and the python-bindings Helpers.hpp then shadows the core one --
  # which shows up much later as a bogus "pybind11.h not found" from a header
  # that has nothing to do with python.
  docker export "$cid" 2>/dev/null \
    | (cd "$out" && tar -xf - --strip-components="$strip" --wildcards "$glob" \
        2>/dev/null || true)
  docker rm "$cid" >/dev/null
  if [[ -z "$(find "$out" -type f -print -quit 2>/dev/null)" ]]; then
    echo "# extracted nothing for $name with glob $glob strip $strip" >&2
    rm -rf "$out"
    echo ""
    return
  fi
  echo "$out"
}

ACTS=$(pull acts  '*/acts-44.99.99-*/include/Acts/*' 6)
BOOST=$(pull boost '*/boost-1.88.0-*/include/boost/*' 6)
EIGEN=$(pull eigen '*/eigen-3.4.0-*/include/eigen3/*' 7)

{
  [[ -n "$ACTS"  ]] && echo "export ACTS_INC='$ACTS'"
  [[ -n "$BOOST" ]] && echo "export BOOST_INC='$BOOST'"
  [[ -n "$EIGEN" ]] && echo "export EIGEN='$EIGEN'"
} || true

if [[ -z "$ACTS" ]]; then
  echo "# no ODD image found (ghcr.io/opendatadetector/sw); ACTS checks will skip" >&2
fi
