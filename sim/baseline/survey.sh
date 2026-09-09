#!/usr/bin/env bash
# What each run directory actually has, so the next batch only redoes what is
# missing. A run is done when digi_and_reco has written its performance file.
ROOT="${BASELINE0_ROOT:-$HOME/baseline0}"
printf "%-6s %-10s %-12s %-12s %s\n" run hepmc merged edm4hep perf
for d in "$ROOT"/out/runs/*/; do
  n=$(basename "$d")
  case "$n" in all) continue;; esac
  hs="-"; [ -f "$d/events.hepmc" ] && hs="yes"
  mg="-"; [ -f "$d/merged_events.hepmc3" ] && mg=$(du -h "$d/merged_events.hepmc3" | cut -f1)
  e4="-"; [ -f "$d/edm4hep.root" ] && e4=$(du -h "$d/edm4hep.root" | cut -f1)
  pf="-"; [ -f "$d/performance_finding_ambi.root" ] && pf="DONE"
  printf "%-6s %-10s %-12s %-12s %s\n" "$n" "$hs" "$mg" "$e4" "$pf"
done
echo
echo "docker: $(docker ps -q | wc -l) running"
free -m | head -2
df -h "$ROOT" | tail -1
