#!/usr/bin/env python3
"""Slice the ACTS track-finding performance files by pT and |eta|.

No new runs and no retraining: the performance ROOT files of the four arms are
already on disk and the performance writer bins
efficiency and fake ratio against pT and eta on its own. This reads those
histograms back and folds them into readable bins.

What the numbers are, taken from the writer at the image's own ACTS commit
(7cba36b, `Examples/Framework/src/Validation/{EffPlotTool,FakePlotTool}.cpp`):

  efficiency  per truth particle. Denominator is the selected truth particles,
              numerator is those with a matched track. Binned in TRUTH pT and
              TRUTH eta. `trackeff_vs_pT` and `trackeff_vs_LogPt` carry no pT
              cut of their own; `trackeff_vs_eta` cuts at the tool's minTruthPt.
  fake ratio  per track. Denominator is all tracks that carry matching
              information, numerator is those classified Fake. Binned in the
              FITTED track pT and eta, not in truth. This is the
              `fakeratio_tracks` scalar of NOTES 6.11, sliced.

The two are therefore not two views of one population, and the tables say so.

Runs inside the ODD image because it needs ROOT; `sim/run_slice_perf.sh` is the
host half.
"""

from __future__ import annotations

import argparse
import sys

import ROOT

ROOT.gROOT.SetBatch(True)

# The four arms of NOTES 6.11, in the order that table lists them.
DEFAULT_ARMS = [
    ("stock", "ckf_accept_stock"),
    ("learned off", "ckf_run0_learnedoff"),
    ("learned on", "ckf_run0_learnedon"),
    ("learned on + Q", "ckf_run0_learnedonQ"),
]

# Fake ratio lives on a 2.5 GeV axis and efficiency on a log axis, so the two
# cannot share edges. Both sets below fall on existing bin boundaries, which
# `fold` checks rather than assumes.
FAKE_PT_EDGES = [0.0, 2.5, 5.0, 10.0, 20.0, 50.0, 100.0]
ABSETA_EDGES = [0.0, 0.6, 1.2, 1.8, 2.4, 3.0]
FAKE_ABSETA_EDGES = [0.0, 0.6, 1.2, 1.8, 2.4, 3.0, 4.0]
# Production radius is not what item 18 asked for. It is in the same file for
# free and it is the variable the vertex-spread hypothesis lives in, so the
# same fold is applied to it. First bin is the beam pipe.
PRODR_EDGES = [0.0, 2.0, 10.0, 50.0, 200.0]
# And |z0|, which is where the pileup vertex spread actually is. rho is cut at
# 24 mm by the truth selection; z is cut at a metre.
ABSZ0_EDGES = [0.0, 8.0, 24.0, 56.0, 120.0, 200.0]

# Fitted pT of the tracks themselves, read from the track summary rather than
# the performance file, because the writer's pT axis stops at 100 GeV and the
# learned arms put tracks past it.
SPECTRUM_EDGES = [0.0, 1.0, 2.5, 5.0, 10.0, 20.0, 50.0, 100.0, float("inf")]

CL = 0.683  # one sigma, Clopper-Pearson, the interval TEfficiency defaults to


def open_file(path: str):
    """`TFile.Open`, but a missing file is one line rather than a traceback.

    PyROOT pythonizes `TFile.Open` to raise OSError, so the usual null check
    never fires.
    """
    try:
        return ROOT.TFile.Open(path)
    except OSError:
        raise SystemExit(f"cannot open {path}") from None


def interval(total: int, passed: int) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    lo = ROOT.TEfficiency.ClopperPearson(total, passed, CL, False)
    hi = ROOT.TEfficiency.ClopperPearson(total, passed, CL, True)
    return (100.0 * lo, 100.0 * hi)


def native_bins(eff) -> list[tuple[float, float, int, int]]:
    """(lo, hi, total, passed) for every bin of a 1D TEfficiency, as filled."""
    tot = eff.GetTotalHistogram()
    pas = eff.GetPassedHistogram()
    ax = tot.GetXaxis()
    out = []
    for i in range(1, ax.GetNbins() + 1):
        out.append(
            (
                ax.GetBinLowEdge(i),
                ax.GetBinUpEdge(i),
                int(round(tot.GetBinContent(i))),
                int(round(pas.GetBinContent(i))),
            )
        )
    return out


def fold(eff, edges: list[float], absolute: bool) -> list[tuple[int, int]]:
    """Sum a 1D TEfficiency into `edges`, optionally folding +x onto -x.

    The last entry returned is everything outside the axis, under- and
    overflow together. On these files it is always zero, because the Acts axis
    drops an out-of-range fill rather than storing it, and it is summed anyway
    so that a file where it is not zero is visible instead of silently short.
    What the axis dropped can only be recovered by difference: on run 0 the
    learned-on-Q arm reads 5,730 in `fakeRatio_vs_pT` against 5,953 tracks, and
    the 223 missing are the tracks above 100 GeV that the track summary shows.

    Every edge has to coincide with a bin boundary of the underlying axis, or
    the sum would split a bin and the counts would stop being counts. That is
    checked here rather than assumed, because the two axes this is used on
    (2.5 GeV in pT, 0.2 in eta) have different widths.
    """
    tot = eff.GetTotalHistogram()
    pas = eff.GetPassedHistogram()
    ax = tot.GetXaxis()
    boundaries = [ax.GetBinLowEdge(i) for i in range(1, ax.GetNbins() + 2)]

    def on_boundary(x: float) -> bool:
        return any(abs(x - b) < 1e-9 for b in boundaries)

    for e in edges:
        if not on_boundary(e) and not (absolute and on_boundary(-e)):
            raise SystemExit(
                f"edge {e} is not a bin boundary of {eff.GetName()}; "
                f"axis runs {boundaries[0]} to {boundaries[-1]} "
                f"in {ax.GetNbins()} bins"
            )

    sums = [[0, 0] for _ in range(len(edges))]
    for i in range(0, ax.GetNbins() + 2):
        t = int(round(tot.GetBinContent(i)))
        p = int(round(pas.GetBinContent(i)))
        if i == 0 or i == ax.GetNbins() + 1:
            sums[-1][0] += t
            sums[-1][1] += p
            continue
        c = ax.GetBinCenter(i)
        if absolute:
            c = abs(c)
        for k in range(len(edges) - 1):
            if edges[k] <= c < edges[k + 1]:
                sums[k][0] += t
                sums[k][1] += p
                break
        else:
            sums[-1][0] += t
            sums[-1][1] += p
    return [(t, p) for t, p in sums]


def track_stats(path: str) -> tuple[list[float], float, float]:
    """Fitted pT of every track in GeV, and mean measurements and holes.

    The performance file cannot answer the first: its pT axis ends at 100 GeV
    and the Acts axis discards a fill past the end rather than putting it in an
    overflow bin, so tracks above 100 GeV are absent from it entirely.
    """
    f = open_file(path)
    tree = f.Get("tracksummary")
    if not tree:
        raise SystemExit(f"no tracksummary tree in {path}")
    pts = []
    meas = holes = n = 0
    for entry in tree:
        for theta, qop, has in zip(
            entry.eTHETA_fit, entry.eQOP_fit, entry.hasFittedParams
        ):
            if not has or qop == 0.0:
                continue
            pts.append(abs(ROOT.TMath.Sin(theta) / qop))
        for m, h in zip(entry.nMeasurements, entry.nHoles):
            meas += m
            holes += h
            n += 1
    f.Close()
    return pts, meas / n, holes / n


def cell(total: int, passed: int) -> str:
    if total == 0:
        return "n/a"
    return f"{100.0 * passed / total:.2f} ({passed})"


def fake_cell(total: int, passed: int) -> str:
    if total == 0:
        return "n/a"
    return f"{100.0 * passed / total:.1f} ({passed}/{total})"


def table(header: list[str], rows: list[list[str]]) -> str:
    align = ["---"] + ["---:"] * (len(header) - 1)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def pt_label(lo: float, hi: float) -> str:
    def f(x: float) -> str:
        return f"{x:.0f}" if abs(x - round(x)) < 5e-3 else f"{x:.2f}"

    return f"{f(lo)} to {f(hi)}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "arms",
        nargs="*",
        metavar="LABEL=SUBDIR",
        help="arms to slice; default is item 11's four",
    )
    ap.add_argument("--runs", default="/out/runs", help="directory holding the subdirs")
    ap.add_argument(
        "--stage",
        default="ambi",
        choices=["ambi", "ckf"],
        help="after ambiguity resolution (default) or at the CKF stage",
    )
    args = ap.parse_args()

    arms = DEFAULT_ARMS
    if args.arms:
        arms = []
        for a in args.arms:
            label, _, subdir = a.partition("=")
            if not subdir:
                raise SystemExit(f"expected LABEL=SUBDIR, got {a!r}")
            arms.append((label, subdir))

    files = {}
    for label, subdir in arms:
        path = f"{args.runs}/{subdir}/performance_finding_{args.stage}.root"
        files[label] = open_file(path)

    labels = [label for label, _ in arms]

    print(f"# performance_finding_{args.stage}.root, sliced")
    print()
    for label, subdir in arms:
        f = files[label]
        eff = f.Get("eff_particles")[0]
        fake = f.Get("fakeratio_tracks")[0]
        h = f.Get("fakeRatio_vs_eta").GetTotalHistogram()
        ntr = int(round(h.Integral(0, h.GetXaxis().GetNbins() + 1)))
        print(
            f"    {label:16s} {subdir:22s} "
            f"eff_particles {100 * eff:.3f}%  "
            f"fakeratio_tracks {100 * fake:.3f}%  tracks {ntr}"
        )
    print()

    # ---- efficiency vs truth pT, on the writer's own log axis ---------------
    per_arm = {label: native_bins(files[label].Get("trackeff_vs_LogPt")) for label in labels}
    ref = per_arm[labels[0]]
    for label in labels[1:]:
        if [b[2] for b in per_arm[label]] != [b[2] for b in ref]:
            raise SystemExit(
                f"denominator of {label} differs from {labels[0]}; "
                "the arms are not the same particle sample"
            )

    rows = []
    for i, (lo, hi, total, _) in enumerate(ref):
        if total == 0:
            continue
        row = [pt_label(lo, hi), str(total)]
        for label in labels:
            row.append(cell(total, per_arm[label][i][3]))
        rows.append(row)
    total_all = sum(b[2] for b in ref)
    row = ["all", str(total_all)]
    for label in labels:
        row.append(cell(total_all, sum(b[3] for b in per_arm[label])))
    rows.append(row)

    print("## Efficiency vs truth pT")
    print()
    print(table(["pT [GeV]", "particles"] + labels, rows))
    print()
    print("Percent of selected truth particles with a matched track, matched count in")
    print("brackets. `trackeff_vs_LogPt`, so the binning is the writer's own and the")
    print("denominator carries no pT cut beyond the sample's own selection.")
    print()

    # ---- efficiency vs |eta| ------------------------------------------------
    per_arm = {
        label: fold(files[label].Get("trackeff_vs_eta"), ABSETA_EDGES, True)
        for label in labels
    }
    ref = per_arm[labels[0]]
    rows = []
    for i in range(len(ABSETA_EDGES) - 1):
        total = ref[i][0]
        row = [f"{ABSETA_EDGES[i]:.1f} to {ABSETA_EDGES[i + 1]:.1f}", str(total)]
        for label in labels:
            row.append(cell(total, per_arm[label][i][1]))
        rows.append(row)
    if ref[-1][0] > 0:
        row = [f"over {ABSETA_EDGES[-1]:.1f}", str(ref[-1][0])]
        for label in labels:
            row.append(cell(ref[-1][0], per_arm[label][-1][1]))
        rows.append(row)
    total_all = sum(t for t, _ in ref)
    row = ["all", str(total_all)]
    for label in labels:
        row.append(cell(total_all, sum(p for _, p in per_arm[label])))
    rows.append(row)

    print("## Efficiency vs truth |eta|")
    print()
    print(table(["|eta|", "particles"] + labels, rows))
    print()
    print("Same quantity, folded onto |eta| from `trackeff_vs_eta`, whose axis stops at")
    print("3 and which carries the tool's own pT cut, so the totals are below the pT")
    print("table's.")
    print()

    # ---- fake ratio vs fitted pT -------------------------------------------
    per_arm = {
        label: fold(files[label].Get("fakeRatio_vs_pT"), FAKE_PT_EDGES, False)
        for label in labels
    }
    rows = []
    for i in range(len(FAKE_PT_EDGES) - 1):
        row = [pt_label(FAKE_PT_EDGES[i], FAKE_PT_EDGES[i + 1])]
        for label in labels:
            row.append(fake_cell(*per_arm[label][i]))
        rows.append(row)
    if any(per_arm[label][-1][0] for label in labels):
        row = [f"over {FAKE_PT_EDGES[-1]:.0f}"]
        for label in labels:
            row.append(fake_cell(*per_arm[label][-1]))
        rows.append(row)
    row = ["all"]
    for label in labels:
        row.append(
            fake_cell(
                sum(t for t, _ in per_arm[label]), sum(p for _, p in per_arm[label])
            )
        )
    rows.append(row)

    print("## Fake ratio vs fitted pT")
    print()
    print(table(["pT [GeV]"] + labels, rows))
    print()
    print("Percent of tracks classified fake, fakes over tracks in brackets. Per track")
    print("and binned in the fitted track pT, so the denominator is the arm's own track")
    print("count and differs between arms. The axis ends at 100 GeV and the Acts axis")
    print("drops a fill past the end instead of holding it in an overflow bin, so the")
    print("totals here are short of the arms' track counts by whatever sits above 100")
    print("GeV. The fitted pT table at the end of this output is where those tracks are.")
    print()

    # ---- fake ratio vs fitted |eta| ----------------------------------------
    per_arm = {
        label: fold(files[label].Get("fakeRatio_vs_eta"), FAKE_ABSETA_EDGES, True)
        for label in labels
    }
    rows = []
    for i in range(len(FAKE_ABSETA_EDGES) - 1):
        row = [f"{FAKE_ABSETA_EDGES[i]:.1f} to {FAKE_ABSETA_EDGES[i + 1]:.1f}"]
        for label in labels:
            row.append(fake_cell(*per_arm[label][i]))
        rows.append(row)
    if any(per_arm[label][-1][0] for label in labels):
        row = [f"over {FAKE_ABSETA_EDGES[-1]:.1f}"]
        for label in labels:
            row.append(fake_cell(*per_arm[label][-1]))
        rows.append(row)
    row = ["all"]
    for label in labels:
        row.append(
            fake_cell(
                sum(t for t, _ in per_arm[label]), sum(p for _, p in per_arm[label])
            )
        )
    rows.append(row)

    print("## Fake ratio vs fitted |eta|")
    print()
    print(table(["|eta|"] + labels, rows))
    print()
    print("Same quantity folded onto |eta|.")
    print()

    # ---- efficiency vs production radius -----------------------------------
    per_arm = {
        label: fold(files[label].Get("trackeff_vs_prodR"), PRODR_EDGES, False)
        for label in labels
    }
    ref = per_arm[labels[0]]
    rows = []
    for i in range(len(PRODR_EDGES) - 1):
        total = ref[i][0]
        row = [f"{PRODR_EDGES[i]:.0f} to {PRODR_EDGES[i + 1]:.0f}", str(total)]
        for label in labels:
            row.append(cell(total, per_arm[label][i][1]))
        rows.append(row)
    total_all = sum(t for t, _ in ref)
    row = ["all", str(total_all)]
    for label in labels:
        row.append(cell(total_all, sum(p for _, p in per_arm[label])))
    rows.append(row)

    print("## Efficiency vs production radius")
    print()
    print(table(["prodR [mm]"] + ["particles"] + labels, rows))
    print()
    print("Not asked for by item 18, in the same file for free. The gun the teacher was")
    print("trained on fires from the origin, so this is one of the two variables the")
    print("vertex-spread hypothesis lives in. First bin is inside the beam pipe, and the")
    print("truth selection cuts the last one off at 24 mm.")
    print()

    # ---- efficiency vs |z0| -------------------------------------------------
    per_arm = {
        label: fold(files[label].Get("trackeff_vs_z0"), ABSZ0_EDGES, True)
        for label in labels
    }
    ref = per_arm[labels[0]]
    rows = []
    for i in range(len(ABSZ0_EDGES) - 1):
        total = ref[i][0]
        row = [f"{ABSZ0_EDGES[i]:.0f} to {ABSZ0_EDGES[i + 1]:.0f}", str(total)]
        for label in labels:
            row.append(cell(total, per_arm[label][i][1]))
        rows.append(row)
    total_all = sum(t for t, _ in ref)
    row = ["all", str(total_all)]
    for label in labels:
        row.append(cell(total_all, sum(p for _, p in per_arm[label])))
    rows.append(row)

    print("## Efficiency vs |z0|")
    print()
    print(table(["|z0| [mm]"] + ["particles"] + labels, rows))
    print()
    print("The other one. z0 is the truth particle's longitudinal impact parameter at the")
    print("beamline, so for a pileup particle it is where its vertex sits. The gun fires")
    print("from z = 0.")
    print()

    # ---- is the surviving efficiency where the particles are ---------------
    # With efficiency below 1% the per-bin percentages are a handful of counts,
    # so the shape question is asked as a ratio: matched particles in a bin
    # against the number a flat efficiency would put there.
    # With efficiency below 1% the per-bin percentages are a handful of counts,
    # so the shape question is asked as a ratio: matched particles in a bin
    # against the number a flat efficiency would put there, and a bin is only
    # called different when its 95% interval excludes the arm's own average.
    print("## Shape of what survives")
    print()

    pt_bins = [
        (pt_label(lo, hi), i)
        for i, (lo, hi, t, _) in enumerate(native_bins(files[labels[0]].Get("trackeff_vs_LogPt")))
        if t > 0
    ]
    eta_bins = [
        (f"{ABSETA_EDGES[i]:.1f} to {ABSETA_EDGES[i + 1]:.1f}", i)
        for i in range(len(ABSETA_EDGES) - 1)
    ]

    def pt_counts(label):
        b = native_bins(files[label].Get("trackeff_vs_LogPt"))
        return [(b[i][2], b[i][3]) for _, i in pt_bins]

    def eta_counts(label):
        b = fold(files[label].Get("trackeff_vs_eta"), ABSETA_EDGES, True)
        return [b[i] for _, i in eta_bins]

    prodr_bins = [
        (f"{PRODR_EDGES[i]:.0f} to {PRODR_EDGES[i + 1]:.0f}", i)
        for i in range(len(PRODR_EDGES) - 1)
    ]

    def prodr_counts(label):
        b = fold(files[label].Get("trackeff_vs_prodR"), PRODR_EDGES, False)
        return [b[i] for _, i in prodr_bins]

    z0_bins = [
        (f"{ABSZ0_EDGES[i]:.0f} to {ABSZ0_EDGES[i + 1]:.0f}", i)
        for i in range(len(ABSZ0_EDGES) - 1)
    ]

    def z0_counts(label):
        b = fold(files[label].Get("trackeff_vs_z0"), ABSZ0_EDGES, True)
        return [b[i] for _, i in z0_bins]

    for var, unit, bins, counts in [
        ("truth pT", "pT [GeV]", pt_bins, pt_counts),
        ("truth |eta|", "|eta|", eta_bins, eta_counts),
        ("production radius", "prodR [mm]", prodr_bins, prodr_counts),
        ("|z0|", "|z0| [mm]", z0_bins, z0_counts),
    ]:
        print(f"### Against {var}")
        print()
        for label in labels[1:]:
            c = counts(label)
            matched = sum(p for _, p in c)
            grand = sum(t for t, _ in c)
            if matched == 0:
                print(f"{label}: no matched particles, no shape to report.")
                print()
                continue
            avg = matched / grand
            rows = []
            outliers = []
            for (name, _), (t, p) in zip(bins, c):
                exp = matched * t / grand
                lo68, hi68 = interval(t, p)
                lo95 = 100.0 * ROOT.TEfficiency.ClopperPearson(t, p, 0.95, False)
                hi95 = 100.0 * ROOT.TEfficiency.ClopperPearson(t, p, 0.95, True)
                flag = ""
                if 100.0 * avg < lo95 or 100.0 * avg > hi95:
                    flag = "yes"
                    outliers.append(name)
                rows.append(
                    [
                        name,
                        str(t),
                        str(p),
                        f"{exp:.1f}",
                        f"{100.0 * p / t:.2f}" if t else "n/a",
                        f"{lo68:.2f} to {hi68:.2f}",
                        flag,
                    ]
                )
            print(f"{label}, {matched} matched particles, {100.0 * avg:.3f}% overall.")
            print()
            print(
                table(
                    [unit, "particles", "matched", "flat", "eff [%]",
                     "68% interval", "differs"],
                    rows,
                )
            )
            print()
            if outliers:
                print("Bins whose 95% interval excludes the arm's own average: "
                      + ", ".join(outliers) + ".")
            else:
                print("No bin's 95% interval excludes the arm's own average.")
            print()

    print("`flat` is the matched count a bin would hold if the arm's efficiency were the")
    print("same everywhere. Intervals are Clopper-Pearson.")
    print()

    # ---- fitted momentum of the tracks -------------------------------------
    print("## Fitted pT of the tracks")
    print()
    rows = []
    stats = []
    for label, subdir in arms:
        pts, meas, holes = track_stats(
            f"{args.runs}/{subdir}/tracksummary_{args.stage}.root"
        )
        counts = [
            sum(1 for p in pts if lo <= p < hi)
            for lo, hi in zip(SPECTRUM_EDGES[:-1], SPECTRUM_EDGES[1:])
        ]
        rows.append([label, str(len(pts))] + [str(c) for c in counts])
        pts.sort()

        def q(frac: float) -> float:
            return pts[min(len(pts) - 1, int(frac * len(pts)))]

        stats.append(
            [
                label,
                f"{q(0.5):.2f}",
                f"{q(0.9):.2f}",
                f"{q(0.99):.1f}",
                f"{pts[-1]:.0f}",
                f"{meas:.3f}",
                f"{holes:.3f}",
            ]
        )

    header = ["arm", "tracks"] + [
        (f"over {lo:.0f}" if hi == float("inf") else pt_label(lo, hi))
        for lo, hi in zip(SPECTRUM_EDGES[:-1], SPECTRUM_EDGES[1:])
    ]
    print(table(header, rows))
    print()
    print(
        table(
            ["arm", "median", "p90", "p99", "max", "measurements", "holes"],
            stats,
        )
    )
    print()
    print("Track counts by fitted pT in GeV, from `eTHETA_fit` and `eQOP_fit` in the")
    print("track summary, over tracks with fitted parameters. Quantiles in GeV;")
    print("measurements and holes are means per track.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
