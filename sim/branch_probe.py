"""Print the branches of a run's tracksummary tree.

    sim/run_root.sh sim/branch_probe.py runs/ckf_map_b1_stock

Written because the track state count has to come out of a
branch that exists rather than out of one that is remembered.
"""
import sys

import ROOT


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: branch_probe.py <run subdir under /out>")
    sub = sys.argv[1]
    for stage in ("ambi", "ckf"):
        path = f"/out/{sub}/tracksummary_{stage}.root"
        f = ROOT.TFile.Open(path)
        if not f or f.IsZombie():
            print(f"{path}: cannot open")
            continue
        tree = f.Get("tracksummary")
        if not tree:
            print(f"{path}: no tracksummary tree")
            f.Close()
            continue
        print(f"== {path}  entries {tree.GetEntries()}")
        for b in tree.GetListOfBranches():
            print(f"   {b.GetName()}")
        f.Close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
