"""Are two reconstruction runs bit-identical, entry by entry and branch by
branch.

    sim/run_root.sh sim/tracks_identical.py runs/a runs/b

The gate asks for bit-identical and not for agreement to three
decimal places, and the per-bin table in `sim/seam_table.py` cannot answer that:
it prints five means, and two runs whose tracks differ can share a mean. This
reads `tracksummary_ambi` in full, compares every branch of every entry with
`==` on the stored value, and stops at the first difference with the entry, the
branch and the two values.

`==` on a double is the right comparison here and not a tolerance. The question
is whether the same arithmetic ran, so any difference at all is the answer;
a tolerance would hide exactly the case this gate exists to catch. NaN is
compared through its bit pattern for the same reason, because `nan != nan` would
report every hole as a difference.

The scalars from `performance_finding_{ckf,ambi}.root` are checked too. They are
what the efficiency tables quote, so a run that matched the
summary tree and not those would be identical in the thing nobody reads.
"""
import math
import struct
import sys

import ROOT

TREE = "tracksummary"
SUMMARIES = ["tracksummary_ambi"]
PERF = ["performance_finding_ckf", "performance_finding_ambi"]
SCALARS = ["eff_particles", "fakeratio_tracks", "duplicationRate_tracks"]


def open_file(path):
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        raise SystemExit(f"cannot open {path}")
    return f


def same(a, b):
    """Exact equality, with NaN equal to the identically-patterned NaN.

    A hole's chi2 comes back as NaN in this writer and `nan == nan` is false,
    so a plain comparison would call every run different from itself. The bit
    pattern is what "the same arithmetic ran" means.
    """
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return struct.pack("<d", a) == struct.pack("<d", b)
    return a == b


def compare_tree(pa, pb, name):
    fa, fb = open_file(f"/out/{pa}/{name}.root"), open_file(f"/out/{pb}/{name}.root")
    ta, tb = fa.Get(TREE), fb.Get(TREE)
    if not ta or not tb:
        raise SystemExit(f"no {TREE} tree in {name}")
    na, nb = ta.GetEntries(), tb.GetEntries()
    if na != nb:
        return [f"{name}: {na} entries against {nb}"]

    branches = [b.GetName() for b in ta.GetListOfBranches()]
    hb = [b.GetName() for b in tb.GetListOfBranches()]
    if branches != hb:
        return [f"{name}: branch list differs"]

    diffs = []
    nvals = 0
    for i in range(na):
        ta.GetEntry(i)
        tb.GetEntry(i)
        for br in branches:
            va, vb = getattr(ta, br), getattr(tb, br)
            try:
                la, lb = list(va), list(vb)
            except TypeError:
                nvals += 1
                if not same(va, vb):
                    diffs.append(f"{name}: entry {i} branch {br}: {va!r} != {vb!r}")
                continue
            if len(la) != len(lb):
                diffs.append(
                    f"{name}: entry {i} branch {br}: {len(la)} values "
                    f"against {len(lb)}")
                continue
            nvals += len(la)
            for k, (x, y) in enumerate(zip(la, lb)):
                if not same(x, y):
                    diffs.append(
                        f"{name}: entry {i} branch {br}[{k}]: {x!r} != {y!r}")
            if len(diffs) > 10:
                break
        if len(diffs) > 10:
            break
    fa.Close()
    fb.Close()
    print(f"{name}: {na:,} entries, {len(branches)} branches, "
          f"{nvals:,} values compared", flush=True)
    return diffs


def compare_perf(pa, pb, name):
    fa, fb = open_file(f"/out/{pa}/{name}.root"), open_file(f"/out/{pb}/{name}.root")
    diffs = []
    for s in SCALARS:
        oa, ob = fa.Get(s), fb.Get(s)
        if not oa or not ob:
            continue
        if not same(oa[0], ob[0]):
            diffs.append(f"{name}: {s}: {oa[0]!r} != {ob[0]!r}")
    fa.Close()
    fb.Close()
    return diffs


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a, b = sys.argv[1], sys.argv[2]
    print(f"A {a}\nB {b}\n", flush=True)
    diffs = []
    for name in SUMMARIES:
        diffs += compare_tree(a, b, name)
    for name in PERF:
        diffs += compare_perf(a, b, name)
    print()
    if not diffs:
        print("IDENTICAL")
        return 0
    print(f"DIFFERENT, first {min(len(diffs), 12)}:")
    for d in diffs[:12]:
        print("  " + d)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
