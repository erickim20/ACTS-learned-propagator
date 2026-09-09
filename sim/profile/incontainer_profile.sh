#!/usr/bin/env bash
# The part that runs inside the image. Same shape as
# `sim/baseline/incontainer.sh`, with two differences: the Sequencer is told to
# write `timing.tsv`, and the reconstruction can be run under perf.
#
# Driven by environment: CONFIG SUBDIR SEED EXTRA TIMING_DIR PERF PERF_FREQ.

# No `set -e` and no `set -u` while sourcing, for the reason
# `sim/baseline/incontainer.sh` records: that script probes for optional pieces
# and several probes exit non-zero by design.
set -o pipefail
source /workspace/scripts/cli/setup_container_env.sh

set -eu
export RECO_SCRIPT=/workspace/scripts/simulation/digi_and_reco.py
export TIMING_DIR
cd /workspace/scripts/simulation
mkdir -p "$TIMING_DIR"

args=(--config "/repo/$CONFIG" --output /output/runs
      --output-subdir "$SUBDIR" --seed "$SEED")
if [ -n "${EXTRA:-}" ]; then
  # shellcheck disable=SC2206
  args+=($EXTRA)
fi

echo "=== reconstruction, timing to $TIMING_DIR/timing.tsv perf=${PERF:-0}"

if [ "${PERF:-0}" != "1" ]; then
  exec python3 /work/seq_timing.py "${args[@]}"
fi

PERF_BIN=/hostperf/perf
DATA="$TIMING_DIR/perf.data"

# cpu-clock rather than the default cycles: WSL2 exposes no PMU, so the
# hardware event is not there at all and perf fails rather than falling back.
"$PERF_BIN" record -e cpu-clock -F "${PERF_FREQ:-499}" -g -o "$DATA" \
  -- python3 /work/seq_timing.py "${args[@]}" || echo "perf record rc=$?"

# The reports are taken here rather than on the host: the DSO paths in
# `perf.data` are the container's, so symbols only resolve inside it.
#
# `-g none` on the reports, although the record has call graphs in it. This
# build has no frame pointers, so the fp chains are addresses rather than
# frames and every one of them is noise. What survives is the flat self time,
# which is what the buckets are read off.
"$PERF_BIN" report -i "$DATA" --stdio --no-children -g none --sort symbol \
  --percent-limit 0.01 > "$TIMING_DIR/perf_flat.txt" 2>/dev/null || true
"$PERF_BIN" report -i "$DATA" --stdio --no-children -g none --sort dso \
  --percent-limit 0.01 > "$TIMING_DIR/perf_dso.txt" 2>/dev/null || true
# The one the buckets are read off. A symbol on its own does not always say
# which algorithm ran it, and the library often does.
"$PERF_BIN" report -i "$DATA" --stdio --no-children -g none --sort dso,symbol \
  --percent-limit 0.005 > "$TIMING_DIR/perf_dso_sym.txt" 2>/dev/null || true
"$PERF_BIN" script -i "$DATA" -F comm,ip,sym,dso \
  > "$TIMING_DIR/perf_script.txt" 2>/dev/null || true

ls -la "$TIMING_DIR"
