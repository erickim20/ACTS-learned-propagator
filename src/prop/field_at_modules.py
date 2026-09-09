"""The ODD field map evaluated where the detector's sensitive modules are.

`LearnedTransport.hpp:72` is `kBHelix = 2.0` and both call sites, `:131` and
`:289`, use it, so the helix core runs at a fixed 2 T everywhere. `bzMap` is
the network's twelfth input and never enters the helix.

The map is not 2 T everywhere. Averaged over the tracker volume that is a large
error, but a grid node is the wrong weight: nodes are spread evenly and hits are
not. A hit happens on a sensitive module, and the modules sit where the detector
puts them. This script re-weights the same map by module centre, which is the
second-best weight; the best one is a per-transport `bzMap` column out of a real
run.

    python -m prop.field_at_modules
    python -m prop.field_at_modules --modules cpp/modules.csv --field oddb.npz

Prints the volume-weighted table first, on the same cut, so the two weights are
read against each other and so the map being used is identifiable.
"""
import argparse

import numpy as np
import polars as pl

# The tracker volume. Everything outside it is solenoid, calorimeter or air,
# and no transport this project prices happens there.
R_MAX = 1080.0
Z_MAX = 3030.0

# What the helix assumes.
B_REF = 2.0


class Field:
    """Trilinear interpolation on the ODD grid, all three components.

    `helix_variants.FieldMap` returns Bz alone. The transverse component
    matters here because a helix about z is exactly the approximation that a
    transverse field breaks, so it is reported rather than assumed small.
    """

    def __init__(self, path):
        d = np.load(path)
        self.x, self.y, self.z = d["x"], d["y"], d["z"]
        n = (len(self.x), len(self.y), len(self.z))
        self.B = d["B"].reshape(*n, 3)
        self.o = np.array([self.x[0], self.y[0], self.z[0]])
        self.h = np.array([self.x[1] - self.x[0], self.y[1] - self.y[0],
                           self.z[1] - self.z[0]])
        self.n = np.array(n)

    def at(self, p):
        """p is (N,3) in mm; returns (N,3) in tesla."""
        f = (p - self.o) / self.h
        i0 = np.clip(np.floor(f).astype(int), 0, self.n - 2)
        t = np.clip(f - i0, 0.0, 1.0)
        out = np.zeros((len(p), 3))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = ((t[:, 0] if dx else 1 - t[:, 0])
                         * (t[:, 1] if dy else 1 - t[:, 1])
                         * (t[:, 2] if dz else 1 - t[:, 2]))
                    out += w[:, None] * self.B[i0[:, 0] + dx, i0[:, 1] + dy,
                                               i0[:, 2] + dz]
        return out


def stats(label, bz, bt, n_all=None):
    """One row of the table, plus the three tolerance fractions."""
    n = len(bz)
    if n == 0:
        print(f"| {label} | 0 | | | | | | | | |")
        return
    a = np.abs(bz)
    rel = np.abs(a - B_REF) / B_REF
    share = "" if n_all is None else f" ({100 * n / n_all:.1f} %)"
    print(f"| {label} | {n:,}{share} | {np.median(a):.3f} | "
          f"{np.percentile(a, 10):.3f} | {np.percentile(a, 90):.3f} | "
          f"{a.min():.3f} | {a.max():.3f} | "
          f"{100 * np.mean(rel < 0.01):.1f} % | "
          f"{100 * np.mean(rel < 0.05):.1f} % | "
          f"{100 * np.mean(rel < 0.10):.1f} % | "
          f"{np.abs(bt).max():.3f} |")


HEAD = ("| where | n | median | p10 | p90 | min | max | within 1 % | "
        "within 5 % | within 10 % | max \\|Bt\\| |")
RULE = ("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", default="cpp/modules.csv",
                    help="geom_dump output, one row per surface")
    ap.add_argument("--field", default="oddb.npz")
    a = ap.parse_args()

    fm = Field(a.field)
    print(f"field {a.field}: {fm.n[0]}x{fm.n[1]}x{fm.n[2]} nodes, "
          f"x {fm.x[0]:g} to {fm.x[-1]:g}, z {fm.z[0]:g} to {fm.z[-1]:g} mm, "
          f"spacing {fm.h[0]:g} {fm.h[1]:g} {fm.h[2]:g} mm")
    print()

    # ---- weight one: the grid, which is what a volume average sees ---------
    gx, gy, gz = np.meshgrid(fm.x, fm.y, fm.z, indexing="ij")
    inside = (np.hypot(gx, gy) < R_MAX) & (np.abs(gz) < Z_MAX)
    gB = fm.B[inside]
    print(f"## Weighted by grid node, r < {R_MAX:g} mm, |z| < {Z_MAX:g} mm")
    print()
    print(HEAD)
    print(RULE)
    stats("tracker volume", gB[:, 2], np.hypot(gB[:, 0], gB[:, 1]))
    print()
    print("One row per grid node inside the tracker volume. This is the weight "
          "that says the field is not 2 T.")
    print()

    # ---- weight two: the modules, which is where a hit can happen ----------
    # Every row is already a sensitive plane: geom_dump.cpp:112 skips anything
    # `isSensitive()` is false for. `sensitive` is the geometry identifier's
    # own index of the module within its layer, counting from 1, and filtering
    # on it keeps one module per layer.
    m = pl.read_csv(a.modules).unique(subset="geometry_id")
    c = np.stack([m["cx"].to_numpy(), m["cy"].to_numpy(),
                  m["cz"].to_numpy()], axis=-1)
    nz = np.abs(m["nz"].to_numpy())
    B = fm.at(c)
    bz, bt = B[:, 2], np.hypot(B[:, 0], B[:, 1])

    # A disc's normal is along z and a barrel stave's is radial, so the normal
    # separates the two without a cut on |z| that would put the outer end of a
    # long barrel layer in with the discs.
    disc = nz > 0.5
    r = np.hypot(c[:, 0], c[:, 1])

    print(f"## Weighted by sensitive module centre")
    print()
    print(HEAD)
    print(RULE)
    n_all = len(bz)
    stats("all modules", bz, bt)
    stats("barrel", bz[~disc], bt[~disc], n_all)
    stats("endcap", bz[disc], bt[disc], n_all)
    print()
    print(f"One row per sensitive module, {n_all:,} of them, at the module "
          f"centre. Barrel and endcap are split by the module's own normal, "
          f"\\|nz\\| > 0.5 being a disc.")
    print()
    print(f"  barrel  r {r[~disc].min():.0f} to {r[~disc].max():.0f} mm, "
          f"|z| up to {np.abs(c[~disc, 2]).max():.0f} mm")
    print(f"  endcap  r {r[disc].min():.0f} to {r[disc].max():.0f} mm, "
          f"|z| {np.abs(c[disc, 2]).min():.0f} to "
          f"{np.abs(c[disc, 2]).max():.0f} mm")
    print()

    # ---- the same, by |eta| of the module centre --------------------------
    # The performance writer's own |eta| slicing, so the rows line up with
    # efficiency. sim/slice_perf.py holds the edges.
    theta = np.arctan2(r, c[:, 2])
    eta = np.abs(-np.log(np.tan(np.clip(theta, 1e-9, np.pi - 1e-9) / 2)))
    edges = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    print("## The same, by \\|eta\\| of the module centre")
    print()
    print(HEAD)
    print(RULE)
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (eta >= lo) & (eta < hi)
        stats(f"{lo:g} to {hi:g}", bz[k], bt[k], n_all)
    print()
    print("Module centres binned on their own \\|eta\\|. A module is not a hit: "
          "an inner barrel layer is crossed by every track and a forward disc "
          "is not.")


if __name__ == "__main__":
    main()
