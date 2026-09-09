"""Where an arm loses its tracks, when the per-bin table says only that it did.

    sim/run_root.sh sim/gate_diag.py runs/seam_b5_stock runs/gate2b_b5

`sim/seam_table.py` reports efficiency after ambiguity resolution and five means
over the surviving tracks. That is enough to see a loss and not enough to say
where it happened. This splits it three ways:

  the CKF stage against the ambiguity stage, which separates a track that was
  never found from one that was found and then dropped;

  the track counts and the mean quantities at both stages;

  the distribution of measurements, holes, outliers and chi2 per track, which is
  what says whether the surviving tracks are the same tracks.
"""
import sys

import ROOT

SCALARS = ["eff_particles", "fakeratio_tracks", "duplicationRate_tracks"]


def open_file(path):
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        raise SystemExit(f"cannot open {path}")
    return f


def perf(subdir, stage):
    f = open_file(f"/out/{subdir}/performance_finding_{stage}.root")
    out = {}
    for s in SCALARS:
        o = f.Get(s)
        out[s] = 100.0 * o[0] if o else float("nan")
    h = f.Get("fakeRatio_vs_eta").GetTotalHistogram()
    out["tracks"] = int(round(h.Integral(0, h.GetXaxis().GetNbins() + 1)))
    f.Close()
    return out


def quantiles(v, ps=(0.1, 0.5, 0.9, 1.0)):
    if not v:
        return [float("nan")] * len(ps)
    v = sorted(v)
    return [v[int(p * (len(v) - 1))] for p in ps]


def summary(subdir):
    f = open_file(f"/out/{subdir}/tracksummary_ambi.root")
    tree = f.Get("tracksummary")
    cols = {k: [] for k in ("nMeasurements", "nHoles", "nOutliers",
                            "nStates", "nSharedHits", "chi2Sum", "NDF")}
    for entry in tree:
        for k in cols:
            try:
                cols[k] += list(getattr(entry, k))
            except AttributeError:
                pass
    f.Close()
    return cols


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a, b = sys.argv[1], sys.argv[2]

    print(f"A {a}\nB {b}\n")
    print(f"{'stage':>6}  {'arm':<4} {'eff [%]':>9} {'fake [%]':>9} "
          f"{'dup [%]':>9} {'tracks':>9}")
    for stage in ("ckf", "ambi"):
        for label, sub in (("A", a), ("B", b)):
            p = perf(sub, stage)
            print(f"{stage:>6}  {label:<4} {p['eff_particles']:9.3f} "
                  f"{p['fakeratio_tracks']:9.3f} "
                  f"{p['duplicationRate_tracks']:9.3f} {p['tracks']:9,}")

    ca, cb = summary(a), summary(b)
    print()
    print(f"{'quantity':<14} {'arm':<4} {'n':>8} {'mean':>10} {'p10':>10} "
          f"{'p50':>10} {'p90':>10} {'max':>10}")
    for k in ("nMeasurements", "nHoles", "nOutliers", "nStates", "nSharedHits",
              "chi2Sum", "NDF"):
        for label, c in (("A", ca), ("B", cb)):
            v = c[k]
            if not v:
                continue
            q = quantiles(v)
            print(f"{k:<14} {label:<4} {len(v):8,} {sum(v)/len(v):10.4f} "
                  f"{q[0]:10.4f} {q[1]:10.4f} {q[2]:10.4f} {q[3]:10.4f}")

    # Where the loss sits. A loss spread evenly over the detector and a loss
    # confined to one region are different bugs.
    for var in ("eta", "pT"):
        print()
        print(f"efficiency vs {var}, CKF stage")
        fa = open_file(f"/out/{a}/performance_finding_ckf.root")
        fb = open_file(f"/out/{b}/performance_finding_ckf.root")
        ha, hb = fa.Get(f"trackeff_vs_{var}"), fb.Get(f"trackeff_vs_{var}")
        if ha and hb:
            print(f"{'bin':>12} {'A tot':>8} {'A eff':>8} {'B tot':>8} "
                  f"{'B eff':>8} {'d':>8}")
            ta = ha.GetTotalHistogram()
            tb = hb.GetTotalHistogram()
            pa = ha.GetPassedHistogram()
            pb = hb.GetPassedHistogram()
            for i in range(1, ta.GetNbinsX() + 1):
                na, nb = ta.GetBinContent(i), tb.GetBinContent(i)
                if na < 20 and nb < 20:
                    continue
                ea = 100.0 * pa.GetBinContent(i) / na if na else float("nan")
                eb = 100.0 * pb.GetBinContent(i) / nb if nb else float("nan")
                lo = ta.GetXaxis().GetBinLowEdge(i)
                print(f"{lo:12.2f} {na:8.0f} {ea:8.2f} {nb:8.0f} {eb:8.2f} "
                      f"{eb - ea:+8.2f}")
        else:
            print("  no trackeff_vs_" + var)
        fa.Close()
        fb.Close()

    # How many tracks are short. A track that lost measurements one at a time
    # and a track that was never extended look the same in a mean.
    print()
    print("measurements per track, as a histogram")
    lo = min(min(ca["nMeasurements"]), min(cb["nMeasurements"]))
    hi = max(max(ca["nMeasurements"]), max(cb["nMeasurements"]))
    print(f"{'n':>4} {'A':>8} {'B':>8}")
    for n in range(int(lo), int(hi) + 1):
        na = sum(1 for x in ca["nMeasurements"] if x == n)
        nb = sum(1 for x in cb["nMeasurements"] if x == n)
        if na or nb:
            print(f"{n:>4} {na:8,} {nb:8,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
