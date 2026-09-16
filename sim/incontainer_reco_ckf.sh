#!/usr/bin/env bash
# In-container half of run_reco_ckf.sh. Same env dance as
# sim/baseline/incontainer.sh, and the same reasons for the set-option order.
# Driven by environment: CONFIG SUBDIR SEED EXTRA STEPPER QTABLE CELL_SIGMA
# SIGMA_HEAD
# NO_RESOLVE_MATERIAL, NAV_WALK_OFF, PLANNED_ONLY, FIELD_GATE, CORE_BZ,
# NET_CORE_FIT PERF
# PERF_FREQ.
set -o pipefail
source /workspace/scripts/cli/setup_container_env.sh

# After the source, never before. sim/odd_tree.sh says why.
set -u
# shellcheck source=/dev/null
source /repo/sim/odd_tree.sh

set -eu
cd /workspace/scripts/simulation

args=(--config "/repo/$CONFIG" --output /output/runs
      --output-subdir "$SUBDIR" --seed "$SEED")
if [ -n "${EXTRA:-}" ]; then
  # shellcheck disable=SC2206
  args+=($EXTRA)
fi

sw=(--stepper "$STEPPER")
[ -n "${QTABLE:-}" ] && sw+=(--qtable "$QTABLE")
[ -n "${CELL_SIGMA:-}" ] && sw+=(--cell-sigma "$CELL_SIGMA")
[ -n "${SIGMA_HEAD:-}" ] && sw+=(--sigma-head)
[ -n "${NO_RESOLVE_MATERIAL:-}" ] && sw+=(--no-resolve-material)
[ -n "${NAV_WALK_OFF:-}" ] && sw+=(--nav-walk-off)
[ -n "${NO_PUBLISH_TRACK:-}" ] && sw+=(--no-publish-track)
[ -n "${PLANNED_ONLY:-}" ] && sw+=(--planned-only)
[ -n "${FIELD_GATE:-}" ] && sw+=(--field-gate "$FIELD_GATE")
[ -n "${SIGMA_CONST:-}" ] && sw+=(--sigma-const "$SIGMA_CONST")
[ -n "${SIGMA_HELIX:-}" ] && sw+=(--sigma-helix "$SIGMA_HELIX")
[ -n "${CORE_BZ:-}" ] && sw+=(--core-bz "$CORE_BZ")
[ -n "${NET_CORE_FIT:-}" ] && sw+=(--net-core-fit)
[ -n "${NO_SEAM_MATERIAL:-}" ] && sw+=(--no-seam-material)
[ -n "${NO_SEAM_SLAB:-}" ] && sw+=(--no-seam-slab)
[ -n "${ALGO_LOG_LEVEL:-}" ] && sw+=(--algo-log-level "$ALGO_LOG_LEVEL")

echo "=== reco_ckf ${sw[*]} -> runs/$SUBDIR perf=${PERF:-0}"

if [ "${PERF:-0}" != "1" ]; then
  exec python3 /repo/sim/reco_ckf.py "${sw[@]}" "${args[@]}"
fi

# The perf half. Same event, same reports and the same reasons as
# `sim/profile/incontainer_profile.sh`: WSL2 exposes no PMU so `cycles` does
# not exist, and this build has no frame pointers so the call graphs are
# addresses and the reports are taken flat.
OUT=/output/runs/$SUBDIR
mkdir -p "$OUT"
PERF_BIN=/hostperf/perf
DATA="$OUT/perf.data"

"$PERF_BIN" record -e cpu-clock -F "${PERF_FREQ:-4999}" -g -o "$DATA" \
  -- python3 /repo/sim/reco_ckf.py "${sw[@]}" "${args[@]}" || echo "perf record rc=$?"

"$PERF_BIN" report -i "$DATA" --stdio --no-children -g none --sort symbol \
  --percent-limit 0.01 > "$OUT/perf_flat.txt" 2>/dev/null || true
"$PERF_BIN" report -i "$DATA" --stdio --no-children -g none --sort dso \
  --percent-limit 0.01 > "$OUT/perf_dso.txt" 2>/dev/null || true
"$PERF_BIN" report -i "$DATA" --stdio --no-children -g none --sort dso,symbol \
  --percent-limit 0.005 > "$OUT/perf_dso_sym.txt" 2>/dev/null || true

ls -la "$OUT"
