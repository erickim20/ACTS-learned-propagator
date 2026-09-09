#!/usr/bin/env bash
# In-container half of run_gun.sh. Same env dance and the same set-option order
# as sim/baseline/incontainer.sh, for the same reason: setup_container_env.sh
# probes for optional pieces and sources /root/.bashrc, and under `set -u` that
# exits the whole shell with the message swallowed.
#
# Driven by environment: SUBDIR SEED EXTRA.
set -o pipefail
source /workspace/scripts/cli/setup_container_env.sh

set -eu
echo "=== gun_stage -> runs/$SUBDIR"
exec python3 /repo/sim/muon/gun_stage.py \
    --config /repo/sim/muon/gun_config.yaml \
    --output /output/runs \
    --output-subdir "$SUBDIR" \
    --seed "$SEED" \
    ${EXTRA:-}
