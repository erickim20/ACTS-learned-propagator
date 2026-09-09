"""Combine several run directories into one baseline number.

The per-run scalars are ratios, so averaging them across runs is wrong whenever
the runs hold different numbers of particles or tracks. Each is combined here
with the weight its own denominator implies.

Efficiency is summed from the trackeff TEfficiency, whose passed and total
histograms add: that reproduces the per-run eff_particles scalars exactly, which
is the check that the summing is right.

Fake and duplicate come from the writer's own scalars weighted by tracks,
not from the corresponding TEfficiency. The two disagree at the ckf stage: the
TEfficiency denominator there is smaller than the number of tracks in the
summary tree, so it is not the per-track ratio the scalar reports and not the
one the old ch2 numbers quoted. They agree after ambiguity resolution.

Usage: aggregate.py <stage> <run_dir> [<run_dir> ...]
"""
import sys
from pathlib import Path

import ROOT

ROOT.gROOT.SetBatch(True)


def integrate(teff):
    passed = teff.GetPassedHistogram()
    total = teff.GetTotalHistogram()
    return (
        passed.Integral(0, passed.GetNbinsX() + 1),
        total.Integral(0, total.GetNbinsX() + 1),
    )


def scalar(f, name):
    v = f.Get(name)
    return float(v[0]) if v else None


def main():
    stage = sys.argv[1]
    dirs = [Path(d) for d in sys.argv[2:]]

    eff_passed = eff_total = 0.0
    fake_w = dup_w = 0.0
    n_tracks = 0
    tot_holes = tot_meas = tot_outl = 0
    used = []

    print(f"    {'run':>8s} {'eff_part':>9s} {'fake_trk':>9s} {'dup_trk':>9s} {'tracks':>8s}")
    for d in dirs:
        perf = d / f"performance_finding_{stage}.root"
        summ = d / f"tracksummary_{stage}.root"
        if not perf.exists() or not summ.exists():
            print(f"    {d.name:>8s}  incomplete, skipped")
            continue
        used.append(d.name)

        # Tracks first: they are the weight for the fake and duplicate scalars.
        g = ROOT.TFile.Open(str(summ))
        t = g.Get("tracksummary")
        run_tracks = 0
        for _ in t:
            for h in t.nHoles:
                tot_holes += h
                run_tracks += 1
            for m in t.nMeasurements:
                tot_meas += m
            for o in t.nOutliers:
                tot_outl += o
        g.Close()
        n_tracks += run_tracks

        f = ROOT.TFile.Open(str(perf))
        te = f.Get("trackeff_vs_eta")
        p = tt = 0.0
        if te:
            p, tt = integrate(te)
            eff_passed += p
            eff_total += tt
        fake = scalar(f, "fakeratio_tracks") or 0.0
        dup = scalar(f, "duplicateratio_tracks") or 0.0
        f.Close()
        fake_w += fake * run_tracks
        dup_w += dup * run_tracks

        eff = 100 * p / tt if tt else float("nan")
        print(
            f"    {d.name:>8s} {eff:8.2f}% {100 * fake:8.2f}% "
            f"{100 * dup:8.2f}% {run_tracks:8d}"
        )

    print()
    print(f"=== combined, stage={stage}, runs {' '.join(used)}")
    if eff_total:
        print(
            f"    {'efficiency, per truth particle':34s} "
            f"{100 * eff_passed / eff_total:8.3f} %   "
            f"({eff_passed:.0f} of {eff_total:.0f})"
        )
    if n_tracks:
        print(f"    {'fake rate, per track':34s} {100 * fake_w / n_tracks:8.3f} %")
        print(f"    {'duplicate rate, per track':34s} {100 * dup_w / n_tracks:8.3f} %")
        print(f"    {'tracks':34s} {n_tracks:8d}")
        print(f"    {'holes per track':34s} {tot_holes / n_tracks:8.4f}")
        print(f"    {'measurements per track':34s} {tot_meas / n_tracks:8.3f}")
        print(f"    {'outliers per track':34s} {tot_outl / n_tracks:8.4f}")


main()
