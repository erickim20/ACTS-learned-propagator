#!/usr/bin/env bash
# The latency curve, one cell per (pT bin, arm).
#
#   cpp/run_speed_by_pt.sh [csv] [repeats]
#     e.g. cpp/run_speed_by_pt.sh speed_by_pt.csv 5
#
# The jump files come from `python -m prop.export_jumps`. Everything runs inside
# the pinned ODD image because the ACTS arm links against the libActsCore that
# lives there, and it runs in ONE container so that one binary produces the
# whole curve.
#
# The output CSV is generated and is not committed.
set -euo pipefail
cd "$(dirname "$0")/.."

IMG=ghcr.io/opendatadetector/sw@sha256:20e4df7b0d6befd70ce0113d852350b25cbd9aac659d1bc530036a2c85a68f89
CSV="${1:-speed_by_pt.csv}"
REPEATS="${2:-5}"

# Section 6's five ladder bins plus HANDOFF item 1's three.
#
# The floor of section 6 is a statement about the accuracy measurement: nothing
# below the truth selection's 0.999 GeV can move an efficiency number. Latency
# is not bounded by that. A track finder transports along every candidate it
# explores and starts making candidates at the seeding minimum of 0.5 GeV, so
# the range that matters here is where transports happen and not where
# particles are scored. 50 to 100 GeV is a check that the ratio does not invert
# at the top, not a working range.
#
# 0.1 to 0.5 GeV is below the seeding minimum and is here because the census
# puts 7.8 % of the stock arm's transports and 15.3 % of the learned arm's
# there, and a bin with no cell is priced at nothing by anyone who multiplies
# the curve by the census. The soft generator run was fired at 0.5 to 1 GeV, so
# this cell holds only the 179 transports that fell below its own window and is
# thin by the standard of every other cell. It is reported with that count
# rather than dropped, and the two constant-work arms land within a few percent
# of their neighbours there, which is the only reason it is usable at all.
#
# lo:hi:jumpfile. Three generator runs cover the nine bins and each bin names
# the one it is drawn from.
SOFT=/repo/cpp/muon_jumps_soft.bin      # 0.5 to 1 GeV
MID=/repo/cpp/muon_jumps.bin            # 1 to 50
HARD=/repo/cpp/muon_jumps_hard.bin      # 50 to 100
BINS="0.1:0.5:$SOFT 0.5:0.7:$SOFT 0.7:1:$SOFT 1:2:$MID 2:4:$MID 4:8:$MID 8:20:$MID 20:50:$MID 50:100:$HARD"

for f in cpp/muon_jumps_soft.bin cpp/muon_jumps.bin cpp/muon_jumps_hard.bin; do
  [[ -f "$f" ]] || {
    echo "no $f; run: python -m prop.export_jumps --pairs <teacher>.parquet" >&2
    exit 2; }
done
[[ -f cpp/field.bin ]] || {
  echo "no cpp/field.bin; the RKN and ACTS arms would run in a constant 2 T" >&2
  exit 2; }

exec docker run --rm -v "$PWD:/repo" -w /repo/cpp \
  -e FIELD=/repo/cpp/field.bin \
  -e CSV="/repo/$CSV" \
  -e REPEATS="$REPEATS" \
  -e BINS="$BINS" \
  --entrypoint /bin/bash "$IMG" /repo/cpp/incontainer_speed_by_pt.sh
