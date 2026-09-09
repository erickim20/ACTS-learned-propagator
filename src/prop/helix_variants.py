"""Does the learning target survive a better choice of B?

The first teacher sample showed the per-jump field-map residual is small in
the bulk (median 3.6 um) with a long tail (9.4% above 200 um, max 14 mm), and
that the tail is entirely endcap, low-pT, long-|dz| — exactly where a helix
assuming a global 2.0 T is being used in a field nearer 1.5 T. If the tail is
just the wrong B, then sampling the field costs one lookup (measured at 3.4%
of the track-following core, i.e. cheap) and there may be little left for
g_theta to learn.

Three physics cores, same jumps, same targets:

  fixed  : B = 2.0 T everywhere            — the baseline core
  source : Bz sampled at the source hit    — one lookup, usable at run time
  avg    : Bz averaged over source and destination — needs the destination,
           so it is an optimistic bound on what a midpoint sample could give
           rather than a deployable model

Usage:
    python helix_variants.py teacher_map_nomat.parquet --field oddb.npz
"""
import argparse

import numpy as np
import polars as pl

MM = 1e-3


class FieldMap:
    """Trilinear interpolation on the ODD grid (x,y,z regular, mm/tesla)."""

    def __init__(self, path):
        d = np.load(path)
        self.x, self.y, self.z = d["x"], d["y"], d["z"]
        nx, ny, nz = len(self.x), len(self.y), len(self.z)
        # csv is ordered x outer, y middle, z inner
        self.B = d["B"].reshape(nx, ny, nz, 3)
        self.o = np.array([self.x[0], self.y[0], self.z[0]])
        self.h = np.array([self.x[1] - self.x[0], self.y[1] - self.y[0],
                           self.z[1] - self.z[0]])
        self.n = np.array([nx, ny, nz])

    def bz(self, x, y, z):
        p = np.stack([x, y, z], axis=-1)
        f = (p - self.o) / self.h
        i0 = np.clip(np.floor(f).astype(int), 0, self.n - 2)
        t = np.clip(f - i0, 0.0, 1.0)
        out = np.zeros(len(x))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = ((t[:, 0] if dx else 1 - t[:, 0])
                         * (t[:, 1] if dy else 1 - t[:, 1])
                         * (t[:, 2] if dz else 1 - t[:, 2]))
                    out += w * self.B[i0[:, 0] + dx, i0[:, 1] + dy,
                                      i0[:, 2] + dz, 2]
        return out


class ConstField:
    """A uniform Bz, with FieldMap's interface.

    The reconstruction runs a uniform 2 T: `digi_and_reco.py:167` takes
    `detector.field`, which is a `DD4hepFieldAdapter` and returns 2.0000 T with
    zero transverse component everywhere inside the solenoid.
    `gen_teacher.py --field const` generates the teacher in that same field, and
    a fit on that teacher has to build the field feature the same way, or the
    input claims a map the transports never flew through.

    Under a uniform field the feature is a constant, so it carries no
    information and `train_gtheta` normalises it to zero. That is the truth
    about a uniform field rather than a defect.
    """

    def __init__(self, bz=2.0):
        self.value = float(bz)

    def bz(self, x, y, z):
        return np.full(len(x), self.value)


def field_source(spec):
    """`--field` as either a map npz or a uniform field.

        const        the 2 T the reconstruction runs
        const:1.5    any other uniform value
        <path>.npz   the ODD map on its grid, trilinear

    One place, because `train_gtheta` and `export_kernel` have to agree about
    it: the header's `mu` and `sd` come from the first and the reference vectors
    the port is checked against come from the second.
    """
    if isinstance(spec, str) and spec.startswith("const"):
        _, _, v = spec.partition(":")
        return ConstField(float(v) if v else 2.0)
    return FieldMap(spec)


def helix_pos(x0, y0, z0, px, py, pz, q, bz, s):
    pt = np.hypot(px, py)
    phi0 = np.arctan2(py, px)
    kappa = -0.3 * q * bz / pt * MM
    phi = phi0 + kappa * s
    return (x0 + (np.sin(phi) - np.sin(phi0)) / kappa,
            y0 - (np.cos(phi) - np.cos(phi0)) / kappa,
            z0 + (pz / pt) * s)


def predict(d, q, bz, endcap, iters=60):
    x0, y0, z0 = d["x"], d["y"], d["z"]
    px, py, pz = d["px"], d["py"], d["pz"]
    pt = np.hypot(px, py)
    s_plane = np.where(np.abs(pz) > 1e-9, (d["z1"] - z0) * pt / pz, 0.0)
    lo, hi = np.zeros_like(pt), np.full_like(pt, 1e4)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        x, y, _ = helix_pos(x0, y0, z0, px, py, pz, q, bz, mid)
        inside = np.hypot(x, y) < d["r1"]
        lo = np.where(inside, mid, lo)
        hi = np.where(inside, hi, mid)
    s = np.where(endcap, s_plane, 0.5 * (lo + hi))
    hx, hy, hz = helix_pos(x0, y0, z0, px, py, pz, q, bz, s)
    dphi = (np.arctan2(hy, hx) - np.arctan2(d["y1"], d["x1"]) + np.pi) \
        % (2 * np.pi) - np.pi
    return d["r1"] * dphi * 1000.0        # um


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("--field", default="/tmp/oddb.npz")
    args = ap.parse_args()

    t = pl.read_parquet(args.pairs)
    d = {c: t[c].to_numpy().astype(float) for c in
         ("x", "y", "z", "px", "py", "pz", "x1", "y1", "z1", "r1",
          "qop", "pt", "abs_eta")}
    q = np.sign(d["qop"])
    endcap = t["endcap"].to_numpy()

    fm = FieldMap(args.field)
    bz_src = fm.bz(d["x"], d["y"], d["z"])
    bz_dst = fm.bz(d["x1"], d["y1"], d["z1"])
    print(f"sampled Bz: source {bz_src.min():.3f}–{bz_src.max():.3f} T, "
          f"mean {bz_src.mean():.3f}")

    res = {
        "fixed 2.0 T": predict(d, q, np.full_like(q, 2.0), endcap),
        "Bz at source": predict(d, q, bz_src, endcap),
        "Bz src+dst avg": predict(d, q, 0.5 * (bz_src + bz_dst), endcap),
    }

    print(f"\n=== |r*phi| residual per jump, {len(q):,} jumps ===")
    print(f"{'core':<16}{'median':>9}{'p90':>9}{'p99':>10}{'max':>10}"
          f"{'>200um':>9}")
    for k, v in res.items():
        a = np.abs(v)
        print(f"{k:<16}{np.median(a):9.1f}{np.percentile(a,90):9.1f}"
              f"{np.percentile(a,99):10.1f}{a.max():10.0f}"
              f"{(a>200).mean()*100:8.1f}%")

    print("\n=== by |eta| (median / fraction above 200 um) ===")
    bins = [(0, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 3)]
    print(f"{'|eta|':<10}" + "".join(f"{k:>22}" for k in res))
    for lo, hi in bins:
        m = (d["abs_eta"] >= lo) & (d["abs_eta"] < hi)
        if m.sum() == 0:
            continue
        row = f"{lo}-{hi:<6}"
        for v in res.values():
            a = np.abs(v[m])
            row += f"{np.median(a):14.1f} /{(a>200).mean()*100:5.1f}%"
        print(row + f"   (n={m.sum()})")

    tail = np.abs(res["fixed 2.0 T"]) > 1000
    print(f"\n=== the {tail.sum()} jumps that were >1 mm with fixed B ===")
    for k, v in res.items():
        a = np.abs(v[tail])
        print(f"  {k:<16} median {np.median(a):8.1f} um   max {a.max():8.0f}")


if __name__ == "__main__":
    main()
