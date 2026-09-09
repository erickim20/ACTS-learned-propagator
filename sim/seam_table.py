"""The three-arm table, per pT bin.

    sim/run_root.sh sim/seam_table.py stock=seam_b%d_stock \\
        maton=seam_b%d_maton matoff=seam_b%d_matoff

Each argument is LABEL=PATTERN with one %d, filled with the bin number 1 to 5.

Five numbers per arm per bin. Efficiency and fake ratio come from the writer's
own scalars, which is what slice_perf.py reads:

  eff_particles     matched truth particles over selected truth particles
  fakeratio_tracks  tracks classified fake over tracks carrying matching info

Measurements, holes and states are means over tracks, from the tracksummary
tree. `nStates` is the one slice_perf.py does not carry, and it is the whole
point here: the material-only surfaces add a track state each,
so the state count is where removing them has to show if it shows anywhere.
"""
import sys

import ROOT

BINS = [1, 2, 3, 4, 5]
PT_LABEL = {1: "1 to 2", 2: "2 to 4", 3: "4 to 8", 4: "8 to 20",
            5: "20 to 50"}


def open_file(path):
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        raise SystemExit(f"cannot open {path}")
    return f


def perf(subdir, stage="ambi"):
    f = open_file(f"/out/{subdir}/performance_finding_{stage}.root")
    eff = f.Get("eff_particles")[0]
    fake = f.Get("fakeratio_tracks")[0]
    h = f.Get("fakeRatio_vs_eta").GetTotalHistogram()
    ntr = int(round(h.Integral(0, h.GetXaxis().GetNbins() + 1)))
    f.Close()
    return 100.0 * eff, 100.0 * fake, ntr


def summary(subdir, stage="ambi"):
    """Mean measurements, holes and states per track, and the track count."""
    f = open_file(f"/out/{subdir}/tracksummary_{stage}.root")
    tree = f.Get("tracksummary")
    if not tree:
        raise SystemExit(f"no tracksummary tree in {subdir}")
    meas = holes = states = 0
    n = 0
    for entry in tree:
        for m, h, s in zip(entry.nMeasurements, entry.nHoles, entry.nStates):
            meas += m
            holes += h
            states += s
            n += 1
    f.Close()
    if n == 0:
        raise SystemExit(f"no tracks in {subdir}")
    return meas / n, holes / n, states / n, n


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    arms = []
    for a in args:
        label, _, pattern = a.partition("=")
        if not pattern:
            raise SystemExit(f"expected LABEL=PATTERN, got {a!r}")
        arms.append((label, pattern))

    rows = []
    for b in BINS:
        for label, pattern in arms:
            sub = pattern % b
            eff, fake, ntr = perf(sub)
            meas, holes, states, ntrk = summary(sub)
            rows.append((b, label, eff, fake, ntr, meas, holes, states, ntrk))
            print(f"read {sub}", flush=True)

    print()
    print("| pT [GeV] | arm | efficiency | fake ratio | tracks | "
          "measurements | holes | states |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    last = None
    for b, label, eff, fake, ntr, meas, holes, states, ntrk in rows:
        head = PT_LABEL[b] if b != last else ""
        last = b
        print(f"| {head} | {label} | {eff:.3f} % | {fake:.3f} % | {ntr:,} | "
              f"{meas:.3f} | {holes:.4f} | {states:.3f} |")

    # The plumbing row, stated as a difference rather than left to the reader.
    print()
    print("Differences against the first arm, per bin.")
    print()
    print("| pT [GeV] | arm | d efficiency | d fake | d measurements | "
          "d holes | d states |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    base = {}
    for b, label, eff, fake, ntr, meas, holes, states, ntrk in rows:
        if label == arms[0][0]:
            base[b] = (eff, fake, meas, holes, states)
    last = None
    for b, label, eff, fake, ntr, meas, holes, states, ntrk in rows:
        if label == arms[0][0]:
            continue
        e0, f0, m0, h0, s0 = base[b]
        head = PT_LABEL[b] if b != last else ""
        last = b
        print(f"| {head} | {label} | {eff - e0:+.3f} | {fake - f0:+.3f} | "
              f"{meas - m0:+.3f} | {holes - h0:+.4f} | {states - s0:+.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
