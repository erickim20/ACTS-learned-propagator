"""How far the stepper flies per destination and what field it flies through.

Reads the `*_dist.csv` files `cpp/TransportCensus.hpp` writes and turns the
histograms into the table HANDOFF task 1 asks for: path per destination and the
map's Bz at the source of each destination-aimed step, split by pT and by
\\|eta\\|, one block per arm.

    python -m prop.census_paths --runs ~/baseline0/out/runs \\
        switch-off=cen_b1_off helix=cen_b1_helix model=cen_b1_modelB

Percentiles come from the histogram, so they carry its bin width: 5.9 % on a
path and 0.01 T on a field. The mean is taken from the exact sum the census
carries beside the histogram and is not a histogram artefact.
"""
import argparse
import pathlib

import numpy as np
import polars as pl

B_HELIX = 2.0            # LearnedTransport.hpp:72, what the transport assumes


def percentile(lo, hi, n, q, log):
    """q-th percentile of a histogram, interpolated inside the bin it lands in.

    Linear inside the bin, in log space for the path histogram, because a bin
    there spans 5.9 % and putting the answer at an edge would be a bigger error
    than the bin.
    """
    if n.sum() == 0:
        return float("nan")
    c = np.cumsum(n)
    t = q * c[-1]
    k = int(np.searchsorted(c, t))
    k = min(k, len(n) - 1)
    below = c[k - 1] if k > 0 else 0
    f = (t - below) / n[k] if n[k] else 0.0
    if log:
        return 10 ** (np.log10(lo[k]) + f * (np.log10(hi[k]) - np.log10(lo[k])))
    return lo[k] + f * (hi[k] - lo[k])


def block(d, metric, split):
    """One arm's histogram for one metric and one split, in bin order.

    SORTED BY `lo`, and it has to be. One arm is several runs concatenated, so
    a slice's bins arrive interleaved by file, and a cumulative sum over them
    in file order is not a distribution.
    """
    t = d.filter((pl.col("metric") == metric) & (pl.col("split") == split))
    out = {}
    for slice_ in t["slice"].unique(maintain_order=True):
        s = t.filter(pl.col("slice") == slice_).sort("lo")
        out[slice_] = (s["lo"].to_numpy(), s["hi"].to_numpy(),
                       s["count"].to_numpy())
    return out


def scalars(d):
    s = d.filter(pl.col("metric") == "scalar")
    return {r["slice"]: r["count"] for r in s.iter_rows(named=True)}


def path_table(arms, split, title):
    print(f"### Path per destination, by {title}")
    print()
    print("| arm | slice | destinations | median | p10 | p90 | max | mean |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label, d in arms.items():
        b = block(d, "path", split)
        for slice_, (lo, hi, n) in b.items():
            tot = n.sum()
            if tot == 0:
                continue
            print(f"| {label} | {slice_} | {tot:,.0f} | "
                  f"{percentile(lo, hi, n, 0.5, True):.1f} | "
                  f"{percentile(lo, hi, n, 0.1, True):.1f} | "
                  f"{percentile(lo, hi, n, 0.9, True):.1f} | "
                  f"{hi[n > 0][-1]:.0f} | "
                  f"{np.sum(np.sqrt(lo * hi) * n) / tot:.1f} |")
    print()
    print("Millimetres of 3D path, summed over the steps aimed at one "
          "destination. `max` is the upper edge of the highest occupied bin.")
    print()


def field_table(arms, split, title):
    print(f"### The map's \\|Bz\\| at the source of each destination-aimed step, "
          f"by {title}")
    print()
    print("| arm | slice | steps | median [T] | p10 | p90 | min | "
          "median / 2.0 | p10 / 2.0 | within 5 % of 2 T |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label, d in arms.items():
        b = block(d, "field", split)
        for slice_, (lo, hi, n) in b.items():
            tot = n.sum()
            if tot == 0:
                continue
            med = percentile(lo, hi, n, 0.5, False)
            p10 = percentile(lo, hi, n, 0.1, False)
            mid = 0.5 * (lo + hi)
            near = n[np.abs(mid - B_HELIX) / B_HELIX < 0.05].sum()
            print(f"| {label} | {slice_} | {tot:,.0f} | {med:.3f} | {p10:.3f} | "
                  f"{percentile(lo, hi, n, 0.9, False):.3f} | "
                  f"{lo[n > 0][0]:.2f} | {med / B_HELIX:.3f} | "
                  f"{p10 / B_HELIX:.3f} | {100 * near / tot:.1f} % |")
    print()
    print(f"Tesla. The last three columns are against the {B_HELIX:g} T the "
          f"helix core assumes at `LearnedTransport.hpp:72`.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(pathlib.Path.home()
                                          / "baseline0/out/runs"))
    ap.add_argument("--name", default="transport",
                    help="census file stem inside each run directory")
    ap.add_argument("arms", nargs="+", metavar="LABEL=SUBDIR")
    a = ap.parse_args()

    arms = {}
    for spec in a.arms:
        label, _, subs = spec.partition("=")
        parts = []
        for sub in subs.split(","):
            p = pathlib.Path(a.runs) / sub / f"{a.name}_dist.csv"
            if not p.exists():
                raise SystemExit(f"no {p}")
            # `count` carries histogram counts and the scalar sums in the same
            # column, so it is read as a float. Both are exact in a double.
            parts.append(pl.read_csv(p, schema_overrides={"count": pl.Float64}))
        # Several runs of one arm add: the bin edges are fixed by the header,
        # so summing counts on (metric, split, slice, lo, hi) is the same
        # histogram the runs would have filled together. The scalars add too,
        # which is what they mean.
        d = pl.concat(parts, how="vertical")
        arms[label] = (d.group_by(["metric", "split", "slice", "lo", "hi"],
                                  maintain_order=True)
                        .agg(pl.col("count").sum()))

    print("### Coverage and the two totals")
    print()
    print("| arm | transports | with a path record | path at destinations "
          "[m] | path elsewhere [m] | elsewhere / destination [mm] |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for label, d in arms.items():
        s = scalars(d)
        n_t, n_p = s["n_transport"], s["n_path"]
        dest, other = s["path_sum_mm"], s["path_other_sum_mm"]
        print(f"| {label} | {n_t:,.0f} | {100 * n_p / n_t:.3f} % | "
              f"{dest / 1000:,.0f} | {other / 1000:,.0f} | {other / n_t:.1f} |")
    print()
    print("A destination still pending when its propagation ends is never "
          "retired and carries no path record, which is the shortfall in "
          "column three.")
    print()

    for split, title in (("pt", "pT of the track at the step"),
                         ("eta", "\\|eta\\| of the track at the step")):
        path_table(arms, split, title)
        field_table(arms, split, title)


if __name__ == "__main__":
    main()
