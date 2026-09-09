#!/usr/bin/env bash
# The part that runs inside the ODD image. Kept in its own file so that no
# quoting has to survive PowerShell -> wsl -> bash -> docker -> bash.
#
# Driven by environment: STAGE CONFIG SUBDIR SEED EXTRA.

# No `set -e` and no `set -u` while sourcing. That script probes for optional
# pieces and several probes exit non-zero by design, and it sources
# /root/.bashrc, which touches unset variables. Under `set -u` that is not a
# non-zero return the `|| true` can absorb, it is an immediate exit of the whole
# shell, with the message swallowed by the script's own 2>/dev/null. The symptom
# is a container that produces no output at all and returns 1.
set -o pipefail
source /workspace/scripts/cli/setup_container_env.sh

# After the source, never before: setup_container_env.sh sources
# /root/.bashrc, which exports ODD_PATH=/opt/odd, and then keeps it with
# ${ODD_PATH:-...}. sim/odd_tree.sh says the rest.
set -u
# shellcheck source=/dev/null
source /repo/sim/odd_tree.sh

set -eu
cd "/workspace/scripts/$(dirname "$STAGE")"

echo "=== running $(basename "$STAGE") config=$CONFIG subdir=$SUBDIR seed=$SEED extra=${EXTRA:-}"
exec python3 "$(basename "$STAGE")" \
    --config "/configs/$CONFIG" \
    --output /output/runs \
    --output-subdir "$SUBDIR" \
    --seed "$SEED" \
    ${EXTRA:-}
