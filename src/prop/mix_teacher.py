"""Combine per-species teacher sets into one that matches a physics event.

The first teacher was 12,000 muons, pT log-uniform 0.5-20 GeV, |eta| < 2.7, and
it does not transfer: on a real event the correction it produced is WORSE than
no correction. Measured against the trackable
population of higgs_portal at pileup 10 -- charged, at least six tracker hits,
13,474 particles over ten events -- that teacher is wrong on three axes:

                        first teacher      physics, trackable
    pT median           3.06 GeV           0.276 GeV
    frac pT < 0.5       0.4%               72.1%
    |eta|               capped at 2.7      p99 = 3.72
    species             100% muon          54.2% pi, 30.0% e, 7.6% p,
                                           4.7% K, 3.4% mu

None of the three can be repaired by reweighting the old set: 0.4% of 110,530
jumps is 442, |eta| > 2.7 is not in it at all, and a pion is not a muon with a
different mass -- it interacts hadronically, and an electron bremsstrahlungs.

So each species is generated separately with `gen_teacher.py --pdg`, over the
range the physics sample actually occupies, and this script draws from them in
the measured proportions.

Sampling is track-wise, not jump-wise. Jumps from one track share surfaces and
states, so a jump-level draw leaks between the parts and inflates a held-out
score -- that is what took s_theta from 33% held out to 59% in sample.

    python mix_teacher.py --out teacher_phys.parquet
    python mix_teacher.py --tracks 20000 --species pion electron
    python mix_teacher.py --pt-floor 1.0 --out teacher_pt1.parquet
"""
import argparse
import pathlib

import numpy as np
import polars as pl

# charged, >= 6 tracker hits, higgs_portal pileup 10, 10 events (13,474)
FRACTION = {
    "pion": 0.542,
    "electron": 0.300,
    "proton": 0.076,
    "kaon": 0.047,
    "muon": 0.034,
}

# The same population's pT spectrum. Matching this as well as the species is
# not optional: generating log-uniform gives COVERAGE, and coverage is what
# reweighting cannot buy, but the jump-level distribution that comes out is
# still nothing like the target. Soft tracks curl up and cross few surfaces, so
# per-jump they are heavily under-represented even when per-track they are not
# -- the raw pion set comes out at a 1.76 GeV median against the target's
# 0.276. Sampling tracks per (species, pT bin) fixes the second half.
PT_EDGES = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, np.inf]
PT_TARGET = [0.2485, 0.1529, 0.1300, 0.1898, 0.1789, 0.0750, 0.0220, 0.0030]


def floor_target(floor):
    """PT_TARGET with the bins below `floor` removed and the rest renormalised.

    A floor is not a filter on top of the target: 90.5% of the target's weight
    is below 1 GeV, so applying the floor and leaving the target alone asks for
    tracks that cannot exist and delivers a tenth of the requested sample. The
    floor has to land on a bin edge, because a bin the floor cuts in half still
    carries its whole share and would be filled from the surviving upper part.
    """
    if floor <= 0:
        return PT_EDGES, PT_TARGET
    if floor not in PT_EDGES:
        raise SystemExit(f"--pt-floor {floor} is not one of PT_EDGES "
                         f"{PT_EDGES}; add the edge or move the floor")
    k = PT_EDGES.index(floor)
    kept = PT_TARGET[k:]
    total = sum(kept)
    return PT_EDGES[k:], [f / total for f in kept]


def load(paths, species, offset):
    """One species, possibly from several runs, with a collision-free track id.

    The id stays an INTEGER. Downstream `train_gtheta.py` sorts by track and
    takes `np.diff` to find the boundaries, which a string id breaks with a
    TypeError several hundred lines after the mistake was made.
    """
    parts = []
    for i, p in enumerate(paths):
        t = pl.read_parquet(p)
        # each run numbers its tracks from zero, so separate the runs too
        parts.append(t.with_columns(
            (pl.col("track").cast(pl.Int64) + offset + i * 10_000_000)
            .alias("track")))
    out = pl.concat(parts, how="vertical_relaxed")
    return out.with_columns(pl.lit(species).alias("species"))


def draw(t, n_tracks, rng, match_pt=True, edges=None, target=None):
    """n_tracks whole tracks, drawn so the pT spectrum matches PT_TARGET.

    Without matching, the draw is uniform over tracks and the jump-level pT
    distribution is whatever the geometry hands back. With it, each pT bin gets
    its target share -- capped by what was generated, and the shortfall is
    reported rather than quietly redistributed, because a bin that silently
    borrows from its neighbour looks exactly like a bin that was filled.
    """
    edges = PT_EDGES if edges is None else edges
    target = PT_TARGET if target is None else target
    per = (t.group_by("track").agg(pl.col("pt").first().alias("pt")))
    ids = per["track"].to_numpy()
    pts = per["pt"].to_numpy()
    if not match_pt or n_tracks >= len(ids):
        if n_tracks >= len(ids):
            return t, len(ids), []
        keep = rng.choice(ids, size=n_tracks, replace=False)
        return t.filter(pl.col("track").is_in(keep.tolist())), n_tracks, []

    b = np.digitize(pts, edges) - 1
    keep, short = [], []
    for i, frac in enumerate(target):
        avail = ids[b == i]
        want = int(round(n_tracks * frac))
        take = min(want, len(avail))
        if take < want:
            short.append((edges[i], edges[i + 1], want, take))
        if take:
            keep.append(rng.choice(avail, size=take, replace=False))
    keep = np.concatenate(keep) if keep else np.array([], dtype=ids.dtype)
    return (t.filter(pl.col("track").is_in(keep.tolist())), len(keep), short)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".",
                    help="where teacher_phys_<species>.parquet live")
    ap.add_argument("--prefix", default="teacher_phys_")
    ap.add_argument("--soft-prefix", default="teacher_soft_",
                    help="a second, low-pT-only run per species, merged in")
    ap.add_argument("--species", nargs="*", default=sorted(FRACTION))
    ap.add_argument("--tracks", type=int, default=12000,
                    help="total track budget, split by the measured fractions")
    ap.add_argument("--out", default="teacher_phys.parquet")
    ap.add_argument("--no-match-pt", action="store_true",
                    help="draw tracks uniformly instead of matching PT_TARGET")
    ap.add_argument("--pt-floor", type=float, default=0.0,
                    help="drop every track below this pT and renormalise "
                         "PT_TARGET over the bins that survive. Must land on a "
                         "PT_EDGES boundary. The output scales of "
                         "train_gtheta.py are the standard deviation of the "
                         "target over whatever this produces, so the floor is "
                         "the one argument that changes them.")
    ap.add_argument("--seed", type=int, default=20260810)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    edges, target = floor_target(a.pt_floor)
    if a.pt_floor > 0:
        print(f"  pT floor {a.pt_floor} GeV: {len(PT_TARGET) - len(target)} "
              f"of {len(PT_TARGET)} bins dropped, carrying "
              f"{1 - sum(PT_TARGET[len(PT_TARGET) - len(target):]):.1%} of the "
              f"physics target; the rest renormalised to "
              + " ".join(f"{f:.3f}" for f in target))
    total = sum(FRACTION[s] for s in a.species)
    parts, report = [], []
    for k, s in enumerate(a.species):
        # the log-uniform run gives coverage; the soft run gives the low-pT
        # bins the statistics they need, because a soft track curls up after
        # very few surfaces and contributes almost no jumps
        found = [p for p in (pathlib.Path(a.dir) / f"{pre}{s}.parquet"
                             for pre in (a.prefix, a.soft_prefix))
                 if p.exists()]
        if not found:
            print(f"  skip {s}: no {a.prefix}{s}.parquet")
            continue
        want = int(round(a.tracks * FRACTION[s] / total))
        one = load(found, s, k * 100_000_000)
        if a.pt_floor > 0:
            one = one.filter(pl.col("pt") >= a.pt_floor)
        t, got, short = draw(one, want, rng,
                             match_pt=not a.no_match_pt,
                             edges=edges, target=target)
        parts.append(t)
        report.append((s, want, got, t.height, short, len(found)))

    if not parts:
        raise SystemExit("nothing to mix")
    out = pl.concat(parts, how="vertical_relaxed")
    out.write_parquet(a.out)

    print(f"\n{a.out}: {out.height:,} jumps, "
          f"{out['track'].n_unique():,} tracks\n")
    print(f"  {'species':10s} {'wanted':>8s} {'drawn':>8s} {'jumps':>9s} "
          f"{'share':>7s}")
    for s, want, got, n, short, nrun in report:
        print(f"  {s:10s} {want:8,d} {got:8,d} {n:9,d} "
              f"{n / out.height:6.1%}  {nrun} run(s)")
        for lo, hi, w, g in short:
            print(f"    short in pT {lo:.2f}-{hi:.2f}: wanted {w}, had {g}")

    pt = out["pt"].to_numpy()
    ae = out["abs_eta"].to_numpy()
    print(f"\n  pT   median {np.median(pt):.3f} GeV   "
          f"frac<0.5 {np.mean(pt < 0.5):.3f}   frac<1 {np.mean(pt < 1):.3f}")
    print(f"  |eta| median {np.median(ae):.2f}   p90 {np.percentile(ae, 90):.2f}"
          f"   p99 {np.percentile(ae, 99):.2f}")
    print("\n  target (physics, trackable): pT median 0.276, frac<0.5 0.721,"
          " |eta| p90 3.08, p99 3.72")


if __name__ == "__main__":
    main()
