#!/usr/bin/env bash
# In-container half of run_gate.sh.
set -o pipefail
source /workspace/scripts/cli/setup_container_env.sh

set -u
# shellcheck source=/dev/null
source /repo/sim/odd_tree.sh
set -e

W=/tmp/gate
mkdir -p "$W"

python3 /work/field_gate.py --via dd4hep --out "$W/g_dd4hep.csv"
python3 /work/field_gate.py --via acts   --out "$W/g_acts.csv"
python3 /work/field_gate.py --via map    --out "$W/g_map.csv" \
        --mapfile /repo/odd-bfield.txt

python3 /work/field_gate.py --report "$W/g_dd4hep.csv" "$W/g_acts.csv" \
        "$W/g_map.csv"
