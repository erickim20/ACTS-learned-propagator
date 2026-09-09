"""Every number the four-arm comparison is drawn from, as one JSON dump.

    sim/run_root.sh sim/arm_grid.py stock=runs/seam_b%d_stock \\
        new-off=runs/f_off_b%d new-helix=runs/f_helix_b%d \\
        new-learned=runs/f_learn_b%d > arms.json

Each argument is LABEL=PATTERN with one %d, filled with the bin number 1 to 5,
the same convention as `sim/seam_table.py` and `sim/eta_table.py`.

Those two print tables and this prints the numbers behind them, because a plot
that re-derives its own numbers from a second reader is a second measurement.

Two things per arm and bin: the writer's own scalars and the means over the
tracksummary tree, which is what `sim/seam_table.py` reports; and
`trackeff_vs_eta`'s passed and total histograms folded to \\|eta\\| and binned,
which is what `sim/eta_table.py` pools. Keeping the pT bin and the \\|eta\\|
band both unpooled is the whole point: the loss the moved seam's transport
carries is a function of the two together and neither table alone shows it.

`$BASELINE0_ROOT/out` is mounted read only at /out, so this prints and does not
write. Redirect it.
"""
import json
import sys

import ROOT

BINS = [1, 2, 3, 4, 5]
PT_EDGES = [1.0, 2.0, 4.0, 8.0, 20.0, 50.0]
# The same split used elsewhere, so the two can be read against each other.
ETA_EDGES = [0.0, 0.6, 1.2, 1.8, 2.4, 3.0, 4.0]


def open_file(path):
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        raise SystemExit(f"cannot open {path}")
    return f


def scalars(subdir, stage="ambi"):
    f = open_file(f"/out/{subdir}/performance_finding_{stage}.root")
    eff = 100.0 * f.Get("eff_particles")[0]
    fake = 100.0 * f.Get("fakeratio_tracks")[0]
    h = f.Get("fakeRatio_vs_eta").GetTotalHistogram()
    ntr = int(round(h.Integral(0, h.GetXaxis().GetNbins() + 1)))
    f.Close()
    return eff, fake, ntr


def folded(subdir, stage="ambi"):
    """(passed, total) per \\|eta\\| band for one run."""
    passed = [0.0] * (len(ETA_EDGES) - 1)
    total = [0.0] * (len(ETA_EDGES) - 1)
    f = open_file(f"/out/{subdir}/performance_finding_{stage}.root")
    eff = f.Get("trackeff_vs_eta")
    if not eff:
        raise SystemExit(f"no trackeff_vs_eta in {subdir}")
    tot, pas = eff.GetTotalHistogram(), eff.GetPassedHistogram()
    for i in range(1, tot.GetNbinsX() + 1):
        centre = abs(tot.GetXaxis().GetBinCenter(i))
        for k in range(len(ETA_EDGES) - 1):
            if ETA_EDGES[k] <= centre < ETA_EDGES[k + 1]:
                total[k] += tot.GetBinContent(i)
                passed[k] += pas.GetBinContent(i)
                break
    f.Close()
    return passed, total


def means(subdir):
    """Mean measurements, holes and states per track."""
    f = open_file(f"/out/{subdir}/tracksummary_ambi.root")
    tree = f.Get("tracksummary")
    if not tree:
        raise SystemExit(f"no tracksummary tree in {subdir}")
    meas = holes = states = 0.0
    n = 0
    for entry in tree:
        for m, h, s in zip(entry.nMeasurements, entry.nHoles, entry.nStates):
            meas += m
            holes += h
            states += s
            n += 1
    f.Close()
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return meas / n, holes / n, states / n, n


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    out = {"pt_edges": PT_EDGES, "eta_edges": ETA_EDGES, "bins": BINS,
           "arms": {}}
    for a in args:
        if "=" not in a or "%d" not in a:
            raise SystemExit(f"expected LABEL=PATTERN with one %d, got {a!r}")
        label, pattern = a.split("=", 1)
        rows = []
        for b in BINS:
            sub = pattern % b
            eff, fake, ntr = scalars(sub)
            meas, holes, states, ntrk = means(sub)
            passed, total = folded(sub)
            rows.append({"bin": b, "run": sub, "efficiency": eff,
                         "fake": fake, "tracks": ntr, "measurements": meas,
                         "holes": holes, "states": states,
                         "summary_tracks": ntrk,
                         "eta_passed": passed, "eta_total": total})
            print(f"read {sub}", file=sys.stderr)
        out["arms"][label] = rows
    json.dump(out, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
