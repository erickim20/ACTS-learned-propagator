#!/usr/bin/env python3
"""Draw what the transport actually does, in three dimensions.

One figure per case. Each shows a track's true hits, the helix arc fired from
every hit towards the next surface, and where that arc lands instead. The point
of the figure is the gap between the arc's end and the next true hit, which is
the quantity a correction removes and the filter's chi2 gate judges.

That gap is small. On a real track it is tens to hundreds of microns against a
detector a metre across, so at true scale it is invisible and a figure drawn at
true scale says nothing. Every figure here therefore states a magnification on
the axis label and applies it only to the miss, never to the geometry. A figure
without that label is not readable and should not be shown.

The helix is sampled from the state at the source hit, closed form, in a
constant 2 T field along z:

    dt/ds = (q/p) (t x B)

with |p| conserved, so q/p is constant along the path and the equation is
linear with constant coefficients. The transverse direction turns at a constant
rate and the polar angle does not change, which integrates to

    x(s) = x0 + sin(theta) [sin(phi0) - sin(phi0 - k s)] / k
    y(s) = y0 + sin(theta) [cos(phi0 - k s) - cos(phi0)] / k
    z(s) = z0 + cos(theta) s          k = 0.3 q B / p

Checked against the `helix_*` columns of a teacher table: with B = 2.0 T and
that sign, the sampled endpoint reproduces the table's own helix point to
0.0e+00 mm median in the transverse plane. The path length is not the table's
`helix_s` column, which is not the three-dimensional path length in mm; it is
derived from the endpoint instead, by dz / cos(theta) where the track has polar
reach and from the transverse chord where it does not.

Usage:
    python -m prop.plot_paths --table fixture/teacher_small.parquet --out fig/
    python -m prop.plot_paths --table t.parquet --select worst --n 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

B_TESLA = 2.0
KAPPA = 0.3               # GeV / (T m), this project's constant


# ------------------------------------------------------------------ geometry

def helix_curvature(qop):
    """Turn rate per mm of path. Positive q turns one way, negative the other."""
    p = 1.0 / np.abs(qop)
    q = np.sign(qop)
    return KAPPA * q * B_TESLA / p / 1000.0


def path_length(x, y, z, px, py, pz, qop, ex, ey, ez):
    """Path length in mm from the source state to a point known to lie on the
    helix.

    Two routes, because each degenerates where the other is fine. A track with
    polar reach gives the path straight from dz. A track lying in the
    transverse plane has dz of zero and gives it from the chord instead, via
    the turn angle of a circle of known radius.
    """
    p = 1.0 / np.abs(qop)
    pt = np.hypot(px, py)
    sinth, costh = pt / p, pz / p

    with np.errstate(divide="ignore", invalid="ignore"):
        s_z = (ez - z) / costh

    radius = pt / (KAPPA * B_TESLA) * 1000.0           # mm
    chord = np.hypot(ex - x, ey - y)
    ratio = np.clip(chord / (2.0 * radius), -1.0, 1.0)
    s_t = 2.0 * radius * np.arcsin(ratio) / np.maximum(sinth, 1e-12)

    return np.where(np.abs(costh) > 0.05, s_z, s_t)


def sample_helix(x, y, z, px, py, pz, qop, s, n=40):
    """n points along the helix from the source state out to path length s.

    Returns three arrays shaped (len(x), n).
    """
    p = 1.0 / np.abs(qop)
    pt = np.hypot(px, py)
    sinth, costh = pt / p, pz / p
    phi0 = np.arctan2(py, px)
    k = helix_curvature(qop)

    u = np.linspace(0.0, 1.0, n)[None, :] * s[:, None]
    phi = phi0[:, None] - k[:, None] * u

    hx = x[:, None] + sinth[:, None] * (np.sin(phi0)[:, None] - np.sin(phi)) / k[:, None]
    hy = y[:, None] + sinth[:, None] * (np.cos(phi) - np.cos(phi0)[:, None]) / k[:, None]
    hz = z[:, None] + costh[:, None] * u
    return hx, hy, hz


# --------------------------------------------------------------------- cases

def miss(t):
    """Three-dimensional distance from the helix landing point to the true
    next hit, in microns."""
    d = np.sqrt(
        (t["helix_x"].to_numpy() - t["x1"].to_numpy()) ** 2
        + (t["helix_y"].to_numpy() - t["y1"].to_numpy()) ** 2
        + (t["helix_z"].to_numpy() - t["z1"].to_numpy()) ** 2
    )
    return d * 1000.0


def pick(t, how, n, seed=0):
    """Track ids for the cases to draw.

    Every criterion is a property of the track rather than of one transport, so
    a track is never selected for a single outlying jump and then drawn as if
    it were typical.
    """
    per = t.group_by("track").agg(
        pl.col("pt").median().alias("pt"),
        pl.col("abs_eta").median().alias("abs_eta"),
        pl.col("endcap").mean().alias("endcap_frac"),
        pl.col("_miss").median().alias("miss_med"),
        pl.len().alias("n_jump"),
    ).filter(pl.col("n_jump") >= 5)

    if how == "worst":
        per = per.sort("miss_med", descending=True)
    elif how == "best":
        per = per.sort("miss_med")
    elif how == "lowpt":
        per = per.sort("pt")
    elif how == "highpt":
        per = per.sort("pt", descending=True)
    elif how == "barrel":
        per = per.filter(pl.col("endcap_frac") < 0.2).sort("miss_med", descending=True)
    elif how == "endcap":
        per = per.filter(pl.col("endcap_frac") > 0.8).sort("miss_med", descending=True)
    elif how == "spread":
        # one track from each quartile of the miss distribution, so the figure
        # carries the range rather than one end of it
        per = per.sort("miss_med")
        idx = np.linspace(0, len(per) - 1, n).astype(int)
        return per[idx]
    else:
        per = per.sample(min(n, len(per)), seed=seed)
    return per.head(n)


# ---------------------------------------------------------------------- plot

def chain(tr):
    """Put a track's transports in the order the particle met them.

    Sorting by radius is wrong and quietly so: a forward track leaves the
    detector through the discs, its radius is not monotonic along the path, and
    the drawn polyline doubles back on itself. The transports already carry the
    order, because transport i ends on the surface transport i+1 starts from,
    so the chain is exact rather than inferred. `surf0` and `surf1` are the
    surface identifiers of those two ends.
    """
    if not {"surf0", "surf1"} <= set(tr.columns):
        return tr.sort("r0")

    s0 = tr["surf0"].to_list()
    s1 = tr["surf1"].to_list()
    nxt = {a: i for i, a in enumerate(s0)}
    heads = [i for i, a in enumerate(s0) if a not in set(s1)]
    if len(heads) != 1:
        return tr.sort("r0")            # branched or looped, leave it alone

    order, seen, i = [], set(), heads[0]
    while i is not None and i not in seen:
        order.append(i)
        seen.add(i)
        i = nxt.get(s1[i])
    if len(order) != len(s0):
        return tr.sort("r0")
    return tr[order]


def auto_magnify(tr, extent):
    """A magnification that makes the median miss about 5% of the panel.

    One factor for the whole figure cannot work: the miss spans four orders of
    magnitude across tracks, so a factor that makes a 7 um miss visible turns a
    4 mm miss into a line longer than the detector. Each panel therefore gets
    its own factor and states it.
    """
    med_mm = float(np.median(tr["_miss"].to_numpy())) / 1000.0
    if med_mm <= 0:
        return 1.0
    raw = 0.05 * extent / med_mm
    if raw <= 1.0:
        return 1.0
    return float(10.0 ** np.floor(np.log10(raw)))


def draw_track(ax, tr, magnify):
    """One track: true hits, the helix arc out of each, the magnified miss."""
    cols = ("x", "y", "z", "px", "py", "pz", "qop", "x1", "y1", "z1",
            "helix_x", "helix_y", "helix_z")
    x, y, z, px, py, pz, qop, x1, y1, z1, hx1, hy1, hz1 = (
        tr[c].to_numpy() for c in cols)

    s = path_length(x, y, z, px, py, pz, qop, hx1, hy1, hz1)
    ax_, ay_, az_ = sample_helix(x, y, z, px, py, pz, qop, s)

    for i in range(len(x)):
        ax.plot(ax_[i], ay_[i], az_[i], color="tab:blue", lw=1.0,
                label="helix" if i == 0 else None)

    # the true hits, in the order the particle met them
    tx = np.append(x, x1[-1])
    ty = np.append(y, y1[-1])
    tz = np.append(z, z1[-1])
    ax.plot(tx, ty, tz, color="black", lw=0.8, ls=":", label="true hits")
    ax.scatter(tx, ty, tz, color="black", s=9, depthshade=False)

    # The miss, magnified, drawn from the true hit towards where the helix went.
    #
    # The factor is set by the track's median, so a single tail transport drawn
    # at that factor runs off the panel and takes the axes with it. The drawn
    # length is therefore clipped, and a clipped one is drawn dashed so that it
    # cannot be read as a true length. The tail is not hidden by this: the
    # companion figure carries every miss at true size.
    extent = max(float(a.max() - a.min()) for a in (x, y, z)) or 1.0
    cap = 0.15 * extent

    dx, dy, dz = (hx1 - x1) * magnify, (hy1 - y1) * magnify, (hz1 - z1) * magnify
    length = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    scale = np.where(length > cap, cap / np.maximum(length, 1e-12), 1.0)

    for i in range(len(x1)):
        ax.plot([x1[i], x1[i] + dx[i] * scale[i]],
                [y1[i], y1[i] + dy[i] * scale[i]],
                [z1[i], z1[i] + dz[i] * scale[i]],
                color="tab:red", lw=1.4, ls="--" if scale[i] < 1.0 else "-",
                label=f"miss x{magnify:g}" if i == 0 else None)


def figure_paths(t, cases, magnify, out, title):
    n = len(cases)
    ncol = min(2, n)
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(7.0 * ncol, 5.6 * nrow))

    for j, row in enumerate(cases.iter_rows(named=True)):
        tr = chain(t.filter(pl.col("track") == row["track"]))
        span = [tr[c].to_numpy() for c in ("x", "y", "z")]
        extent = max(float(a.max() - a.min()) for a in span) or 1.0
        mag = magnify if magnify > 0 else auto_magnify(tr, extent)

        ax = fig.add_subplot(nrow, ncol, j + 1, projection="3d")
        draw_track(ax, tr, mag)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_zlabel("z [mm]")
        ax.set_title(
            f"pT {row['pt']:.2f} GeV, |eta| {row['abs_eta']:.2f}, "
            f"{row['n_jump']} transports\n"
            f"median miss {row['miss_med']:.0f} um, drawn x{mag:g}",
            fontsize=9)
        if j == 0:
            ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(f"{title}. Geometry is true scale; the miss is magnified by "
                 f"the factor each panel states.", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_projections(t, cases, out, title):
    """r-z and x-y, which read faster than a 3D view for anyone checking
    whether the track went where a track should go."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))

    for row in cases.iter_rows(named=True):
        tr = chain(t.filter(pl.col("track") == row["track"]))
        x, y, z = (tr[c].to_numpy() for c in ("x", "y", "z"))
        x1, y1, z1 = (tr[c].to_numpy() for c in ("x1", "y1", "z1"))
        tx, ty, tz = np.append(x, x1[-1]), np.append(y, y1[-1]), np.append(z, z1[-1])
        lab = f"pT {row['pt']:.2f}"
        axes[0].plot(tz, np.hypot(tx, ty), marker="o", ms=2.5, lw=0.8, label=lab)
        axes[1].plot(tx, ty, marker="o", ms=2.5, lw=0.8, label=lab)

    axes[0].set_xlabel("z [mm]")
    axes[0].set_ylabel("r [mm]")
    axes[0].set_title("r-z")
    axes[1].set_xlabel("x [mm]")
    axes[1].set_ylabel("y [mm]")
    axes[1].set_title("x-y")
    axes[1].set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_miss_growth(t, cases, out):
    """Miss against transport index, one line per track.

    This is the honest companion to the 3D figure: it carries the same
    quantity at true size, on a log axis, with the sensor resolutions marked so
    that the reader can see whether a miss is large or small compared with what
    the detector can measure.
    """
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for row in cases.iter_rows(named=True):
        tr = chain(t.filter(pl.col("track") == row["track"]))
        ax.plot(np.arange(len(tr)), tr["_miss"].to_numpy(), marker="o", ms=3,
                lw=1.0, label=f"pT {row['pt']:.2f} GeV, |eta| {row['abs_eta']:.2f}")

    for um, name in ((15.0, "pixel 15 um"), (43.0, "short strip 43 um"),
                     (72.0, "long strip 72 um")):
        ax.axhline(um, color="gray", lw=0.7, ls="--")
        ax.text(0.02, um * 1.06, name, fontsize=7, color="gray")

    ax.set_yscale("log")
    ax.set_xlabel("transport index along the track")
    ax.set_ylabel("helix miss at the destination [um]")
    ax.set_title("Miss at true size, against the sensor resolutions", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, help="teacher parquet")
    ap.add_argument("--out", default="fig", help="output directory")
    ap.add_argument("--select", default="spread",
                    choices=["spread", "worst", "best", "lowpt", "highpt",
                             "barrel", "endcap", "random"])
    ap.add_argument("--n", type=int, default=4, help="tracks to draw")
    ap.add_argument("--magnify", type=float, default=200.0,
                    help="miss magnification in the 3D figure")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t = pl.read_parquet(args.table)
    need = {"x", "y", "z", "px", "py", "pz", "qop", "x1", "y1", "z1",
            "helix_x", "helix_y", "helix_z", "track", "pt", "abs_eta", "r0"}
    missing = need - set(t.columns)
    if missing:
        raise SystemExit(f"table is missing {sorted(missing)}")

    t = t.with_columns(pl.Series("_miss", miss(t)))
    cases = pick(t, args.select, args.n, args.seed)
    if len(cases) == 0:
        raise SystemExit(f"no track matched --select {args.select}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.select

    print(figure_paths(t, cases, args.magnify, out / f"paths_3d_{tag}.png",
                       f"Transport in three dimensions, {tag}"))
    print(figure_projections(t, cases, out / f"paths_proj_{tag}.png",
                             f"Same tracks in projection, {tag}"))
    print(figure_miss_growth(t, cases, out / f"miss_growth_{tag}.png"))

    m = t["_miss"].to_numpy()
    print(f"\n{len(t):,} transports over {t['track'].n_unique():,} tracks")
    print(f"miss [um]: median {np.median(m):.1f}  p99 {np.percentile(m, 99):.1f}"
          f"  max {m.max():.1f}")


if __name__ == "__main__":
    main()
