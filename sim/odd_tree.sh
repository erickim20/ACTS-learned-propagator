# Point the container at a different ODD tree. Sourced, never executed.
#
# Every in-container script in this repository sources this immediately after
# `setup_container_env.sh`, and that ordering is the whole point.
#
# ODD_PATH cannot be set from outside. The image's `/root/.bashrc` exports
# `ODD_PATH=/opt/odd` at :182; `setup_container_env.sh` sources that .bashrc in
# its step 1 and then assigns `export ODD_PATH="${ODD_PATH:-$_odd_src}"`, which
# keeps the non-empty value it has just been handed. So `docker run -e
# ODD_PATH=...` is discarded, and `sim/baseline/run_stage.sh` and
# `sim/run_reco_ckf.sh` both passed one for months without effect. The only
# assignment that wins is one made after the source, which is this file.
#
# The same applies to the factory library. `getOpenDataDetector` finds
# `libOpenDataDetector.so` on LD_LIBRARY_PATH, and a tree whose compact names a
# field plugin that the first library on that path does not carry fails to
# build the detector. A new .so with an old compact is worse: it is silent, and
# the detector is a uniform 2 T.
#
#   ODD_TREE          container path of the tree whose xml/ is to be loaded
#   ODD_TREE_INSTALL  container path of the matching install prefix, whose
#                     lib/ holds the factory library built from that tree
#
# Unset or empty leaves the image's own ODD alone, which is the detector the
# ODD release describes. `/cache/odd-map` is the one whose compact carries the
# field map, and it is not the ODD.
if [ -n "${ODD_TREE:-}" ]; then
  export ODD_PATH="$ODD_TREE"
  if [ -n "${ODD_TREE_INSTALL:-}" ]; then
    export LD_LIBRARY_PATH="$ODD_TREE_INSTALL/lib:${LD_LIBRARY_PATH:-}"
  fi
  echo "=== ODD tree: $ODD_PATH  factory: ${ODD_TREE_INSTALL:-image default}"
  echo "=== field element in that compact:"
  grep -n "<field" "$ODD_PATH/xml/OpenDataDetector.xml" || true
else
  echo "=== ODD tree: image default ($ODD_PATH), analytic solenoid"
fi
