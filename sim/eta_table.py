"""Efficiency against \\|eta\\|, several arms at once, pooled over the pT bins.

    sim/run_root.sh sim/eta_table.py stock=runs/seam_b%d_stock \\
        newhelix=runs/ac_helix_b%d

Each argument is LABEL=PATTERN with one %d, filled with the bin number 1 to 5,
the same convention as `sim/seam_table.py`.

The moved seam's transport should lose most where the map is furthest from
2 T, which is the endcap: inside the barrel the map is within 5 % of 2 T at every module, and in
the endcap it runs 0.532 to 1.970 T with 18.6 % within 5 %. The per-pT table
cannot see that and this is what can.

`trackeff_vs_eta` is a `TEfficiency` in signed eta, so the passed and total
histograms are folded to \\|eta\\| and summed over the five pT bins before the
ratio is taken. Summing the ratios instead would weight a bin with 40 truth
particles the same as one with 4,000.

The ambiguity stage, to match `sim/seam_table.py`. The CKF stage is printed
underneath because a track that was never found and one that was found and then
dropped are different failures and the pooled number cannot tell them apart.
"""
import sys

import ROOT

BINS = [1, 2, 3, 4, 5]
# The same split used elsewhere, so the two tables can be read against each
# other: |n_z| > 0.5 is a disc, and 1.2 is where the ODD's barrel ends.
EDGES = [0.0, 0.6, 1.2, 1.8, 2.4, 3.0, 4.0]


def open_file(path):
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        raise SystemExit(f"cannot open {path}")
    return f


def folded(pattern, stage):
    """(passed, total) per |eta| band, summed over the five pT bins."""
    passed = [0.0] * (len(EDGES) - 1)
    total = [0.0] * (len(EDGES) - 1)
    for b in BINS:
        f = open_file(f"/out/{pattern % b}/performance_finding_{stage}.root")
        eff = f.Get("trackeff_vs_eta")
        if not eff:
            raise SystemExit(f"no trackeff_vs_eta in {pattern % b}")
        tot = eff.GetTotalHistogram()
        pas = eff.GetPassedHistogram()
        for i in range(1, tot.GetNbinsX() + 1):
            centre = abs(tot.GetXaxis().GetBinCenter(i))
            for k in range(len(EDGES) - 1):
                if EDGES[k] <= centre < EDGES[k + 1]:
                    total[k] += tot.GetBinContent(i)
                    passed[k] += pas.GetBinContent(i)
                    break
        f.Close()
    return passed, total


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    arms = []
    for a in args:
        if "=" not in a or "%d" not in a:
            raise SystemExit(f"expected LABEL=PATTERN with one %d, got {a!r}")
        label, pattern = a.split("=", 1)
        arms.append((label, pattern))

    for stage in ("ambi", "ckf"):
        print(f"\nefficiency [%] against |eta|, {stage} stage, "
              f"five pT bins pooled\n")
        head = f"{'|eta|':<12} {'truth':>9}"
        for label, _ in arms:
            head += f" {label:>12}"
        if len(arms) > 1:
            head += f" {'d':>9}"
        print(head)
        cols = [folded(p, stage) for _, p in arms]
        for k in range(len(EDGES) - 1):
            n = cols[0][1][k]
            if n < 20:
                continue
            row = f"{EDGES[k]:.1f} to {EDGES[k + 1]:.1f}".ljust(12)
            row += f" {n:9,.0f}"
            effs = []
            for passed, total in cols:
                e = 100.0 * passed[k] / total[k] if total[k] else float("nan")
                effs.append(e)
                row += f" {e:12.3f}"
            if len(effs) > 1:
                row += f" {effs[-1] - effs[0]:+9.3f}"
            print(row)
        row = "all".ljust(12)
        row += f" {sum(cols[0][1]):9,.0f}"
        effs = []
        for passed, total in cols:
            e = 100.0 * sum(passed) / sum(total) if sum(total) else float("nan")
            effs.append(e)
            row += f" {e:12.3f}"
        if len(effs) > 1:
            row += f" {effs[-1] - effs[0]:+9.3f}"
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
