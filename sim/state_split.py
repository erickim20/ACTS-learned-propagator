"""Split nStates into what the writer can account for and what is left.

    sim/run_root.sh sim/state_split.py seam_b1_maton seam_b1_matoff ...
    sim/run_root.sh sim/state_split.py --stage ckf pa2_stock_b1 pa_cell_b1

`nStates` counts every track state on the trajectory. `nMeasurements`,
`nOutliers` and `nHoles` name three of the kinds. What is left over is the
material-only states, and it is the only column that says how
much of the material channel an arm actually removed.

The totals are printed beside the per-track means because they are a
denominator. Track finding is timed per candidate, and whether that is the
right normaliser is answered by dividing the same seconds by the states and by
the measurements those candidates carry, which needs the totals rather than
the means.

`--stage ckf` reads `tracksummary_ckf` rather than `tracksummary_ambi`. The
ambiguity solver drops tracks the finder already paid for, so the ckf stage is
the one whose track count is the finder's own `selected` count.
"""
import sys

import ROOT

STAGES = ("ambi", "ckf")


def main():
    args = sys.argv[1:]
    stage = "ambi"
    if len(args) >= 2 and args[0] == "--stage":
        stage = args[1]
        args = args[2:]
        if stage not in STAGES:
            raise SystemExit(f"--stage is one of {STAGES}, got {stage!r}")
    if not args:
        raise SystemExit(__doc__)
    print(f"`tracksummary_{stage}`, per track and in total.")
    print()
    print("| run | tracks | states | measurements | outliers | holes | other "
          "| total states | total measurements |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for sub in args:
        f = ROOT.TFile.Open(f"/out/runs/{sub}/tracksummary_{stage}.root")
        if not f or f.IsZombie():
            raise SystemExit(f"cannot open {sub}")
        tree = f.Get("tracksummary")
        st = me = ou = ho = 0
        n = 0
        for entry in tree:
            for s, m, o, h in zip(entry.nStates, entry.nMeasurements,
                                  entry.nOutliers, entry.nHoles):
                st += s
                me += m
                ou += o
                ho += h
                n += 1
        f.Close()
        print(f"| {sub} | {n:,} | {st / n:.3f} | {me / n:.3f} | {ou / n:.4f} | "
              f"{ho / n:.4f} | {(st - me - ou - ho) / n:.3f} | {st:,} | "
              f"{me:,} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
