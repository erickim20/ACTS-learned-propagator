"""The four numbers item 0 asks for, out of one run directory.

    tracking efficiency, fake rate, hole rate, time profile

Efficiency and fake rate are the scalars the ACTS TrackFinderPerformanceWriter
stores at the top of performance_finding_*.root. Both denominators are reported
because they answer different questions: *_particles is per selected truth
particle, *_tracks is per reconstructed track.

Holes come from the per-track tracksummary tree rather than the eta profile, so
the number is a plain mean over tracks with its own spread.

Usage: baseline_numbers.py <run_dir> [<run_dir> ...]
"""
import sys
from pathlib import Path

import ROOT

ROOT.gROOT.SetBatch(True)

SCALARS = [
    ("eff_particles", "efficiency, per truth particle"),
    ("eff_tracks", "efficiency, per track"),
    ("fakeratio_particles", "fake rate, per truth particle"),
    ("fakeratio_tracks", "fake rate, per track"),
    ("duplicateratio_particles", "duplicate rate, per truth particle"),
    ("duplicateratio_tracks", "duplicate rate, per track"),
]


def scalar(f, name):
    v = f.Get(name)
    if not v:
        return None
    return float(v[0])


def report(run_dir, stage):
    perf = run_dir / f"performance_finding_{stage}.root"
    summ = run_dir / f"tracksummary_{stage}.root"
    if not perf.exists():
        return False

    print(f"--- {run_dir.name}  stage={stage}")
    f = ROOT.TFile.Open(str(perf))
    for name, label in SCALARS:
        val = scalar(f, name)
        if val is not None:
            print(f"    {label:38s} {100 * val:8.3f} %")
    f.Close()

    if summ.exists():
        g = ROOT.TFile.Open(str(summ))
        t = g.Get("tracksummary")
        # One entry per event, with vector branches over the tracks in it.
        n_tracks = 0
        tot_holes = 0
        tot_meas = 0
        tot_outl = 0
        for _ in t:
            for h in t.nHoles:
                tot_holes += h
                n_tracks += 1
            for m in t.nMeasurements:
                tot_meas += m
            for o in t.nOutliers:
                tot_outl += o
        if n_tracks:
            print(f"    {'tracks':38s} {n_tracks:8d}")
            print(f"    {'holes per track':38s} {tot_holes / n_tracks:8.4f}")
            print(f"    {'measurements per track':38s} {tot_meas / n_tracks:8.3f}")
            print(f"    {'outliers per track':38s} {tot_outl / n_tracks:8.4f}")
        g.Close()
    return True


for arg in sys.argv[1:]:
    run_dir = Path(arg)
    print(f"=== {run_dir}")
    found = False
    for stage in ("ckf", "ambi"):
        found |= report(run_dir, stage)
    if not found:
        print("    no performance_finding_*.root here")

    timing = run_dir / "timing_summary.txt"
    if timing.exists():
        print("--- timing_summary.txt")
        for line in timing.read_text().splitlines():
            print(f"    {line}")
