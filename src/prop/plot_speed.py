#!/usr/bin/env python3
"""Speed against pT.

The measurement this reads does not exist yet as a curve. What exists is one
average over a mixed sample, and an average is a statement about that sample
rather than about the detector: feed the same two steppers a different pT
spectrum and the ratio moves, by an amount nobody has measured. A curve does not
have that problem. Anyone can integrate it against their own spectrum.

There is a reason to expect the curve to have shape. The learned kernel costs
the same arithmetic on every transport: one field read, one forward pass, one
plane intersection, one Jacobian. An adaptive Runge-Kutta stepper does not. Its
step size is bounded by how fast the trajectory turns, a soft track turns fast,
so a soft track costs more steps and each step reads the field at three distinct
positions. The ratio should therefore be largest at low pT and shrink as tracks
straighten, and it may cross one somewhere. Where it crosses, if it crosses, is
a result and belongs on the slide with the ratio.

INPUT CONTRACT. One CSV, one row per (pT bin, arm):

    pt_lo, pt_hi      bin edges in GeV, the bin the transports were drawn from
    arm               free string, e.g. acts_eigen, acts_eigen_cov, kernel,
                      kernel_matched, helix, helix_matched
    n_transport       transports timed in this cell
    ns_median         nanoseconds per transport
    ns_lo, ns_hi      spread across repeats, same units. Blank if not repeated
    rk_steps          mean integration steps per transport. Blank for arms that
                      do not step
    field_lookups     mean magnetic field reads per transport. Blank if not
                      counted

Blank cells are permitted and are dropped from the panel they would feed rather
than plotted as zero. A zero step count and an uncounted step count are not the
same statement and the figure must not conflate them.

Usage:
    python -m prop.plot_speed --csv speed_by_pt.csv --baseline acts_eigen_cov
    python -m prop.plot_speed --demo /tmp/demo.csv        # synthetic, to smoke test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

COLUMNS = ["pt_lo", "pt_hi", "arm", "n_transport", "ns_median",
           "ns_lo", "ns_hi", "rk_steps", "field_lookups"]


def load(path):
    d = pl.read_csv(path, null_values=["", "NA", "nan"])
    missing = {"pt_lo", "pt_hi", "arm", "ns_median"} - set(d.columns)
    if missing:
        raise SystemExit(f"csv is missing required columns {sorted(missing)}")
    for c in COLUMNS:
        if c not in d.columns:
            d = d.with_columns(pl.lit(None).alias(c))
    return d.with_columns(
        ((pl.col("pt_lo") * pl.col("pt_hi")) ** 0.5).alias("pt_mid"))


def _bin_labels(d):
    e = d.select("pt_lo", "pt_hi").unique().sort("pt_lo")
    return [f"{lo:g}-{hi:g}" for lo, hi in e.iter_rows()]


def _bin_ticks(ax, d):
    """Tick at each bin, labelled with the bin rather than with a power of ten.

    A log axis defaults to labels like 3x10^0 and 4x10^0, which collide at this
    figure width and name a coordinate the measurement does not have. The
    measurement has five bins, so the axis gets five ticks.
    """
    e = d.select("pt_lo", "pt_hi").unique().sort("pt_lo")
    pos = [(lo * hi) ** 0.5 for lo, hi in e.iter_rows()]
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{lo:g}-{hi:g}" for lo, hi in e.iter_rows()], fontsize=8)
    ax.set_xticks([], minor=True)


def panel_latency(ax, d):
    for arm in d["arm"].unique(maintain_order=True):
        a = d.filter(pl.col("arm") == arm).sort("pt_mid")
        x, y = a["pt_mid"].to_numpy(), a["ns_median"].to_numpy()
        ax.plot(x, y, marker="o", ms=4, lw=1.2, label=arm)
        lo, hi = a["ns_lo"].to_numpy(), a["ns_hi"].to_numpy()
        ok = ~(np.isnan(lo.astype(float)) | np.isnan(hi.astype(float)))
        if ok.any():
            ax.fill_between(x[ok], lo.astype(float)[ok], hi.astype(float)[ok],
                            alpha=0.15, lw=0)
    ax.set_xscale("log")
    _bin_ticks(ax, d)
    ax.set_xlabel("pT [GeV], bin centre")
    ax.set_ylabel("ns per surface-to-surface transport")
    ax.set_title("Latency", fontsize=10)
    ax.legend(fontsize=8)


def panel_ratio(ax, d, baseline):
    base = d.filter(pl.col("arm") == baseline).sort("pt_mid")
    if len(base) == 0:
        ax.set_visible(False)
        return None
    bx, by = base["pt_mid"].to_numpy(), base["ns_median"].to_numpy()

    crossing = None
    for arm in d["arm"].unique(maintain_order=True):
        if arm == baseline:
            continue
        a = d.filter(pl.col("arm") == arm).sort("pt_mid")
        x, y = a["pt_mid"].to_numpy(), a["ns_median"].to_numpy()
        if len(x) != len(bx) or not np.allclose(x, bx):
            continue
        r = by / y
        ax.plot(x, r, marker="o", ms=4, lw=1.2, label=f"{baseline} / {arm}")
        if np.any(r < 1.0) and np.any(r > 1.0):
            i = int(np.argmax(r < 1.0))
            crossing = float(np.interp(1.0, [r[i], r[i - 1]], [x[i], x[i - 1]]))

    ax.axhline(1.0, color="black", lw=0.9)
    ax.text(ax.get_xlim()[0], 1.02, "parity", fontsize=7, va="bottom")
    if crossing is not None:
        ax.axvline(crossing, color="tab:red", lw=0.9, ls="--")
        ax.text(crossing, ax.get_ylim()[1], f" crossover {crossing:.1f} GeV",
                fontsize=8, color="tab:red", va="top")
    ax.set_xscale("log")
    _bin_ticks(ax, d)
    ax.set_xlabel("pT [GeV], bin centre")
    ax.set_ylabel("speedup, baseline over arm")
    ax.set_title("Ratio. Above 1 the arm is faster than the baseline", fontsize=10)
    ax.legend(fontsize=8)
    return crossing


def panel_work(ax, d, column, ylabel, title):
    any_data = False
    for arm in d["arm"].unique(maintain_order=True):
        a = d.filter(pl.col("arm") == arm).sort("pt_mid")
        y = a[column].to_numpy().astype(float)
        if np.all(np.isnan(y)):
            continue
        any_data = True
        ax.plot(a["pt_mid"].to_numpy(), y, marker="o", ms=4, lw=1.2, label=arm)
    if not any_data:
        ax.text(0.5, 0.5, f"{column} not recorded", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_axis_off()
        return
    ax.set_xscale("log")
    _bin_ticks(ax, d)
    ax.set_xlabel("pT [GeV], bin centre")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)


def figure(d, baseline, out, note=""):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    panel_latency(axes[0][0], d)
    crossing = panel_ratio(axes[0][1], d, baseline)
    panel_work(axes[1][0], d, "rk_steps", "integration steps per transport",
               "Why. Steps rise as the track turns harder")
    panel_work(axes[1][1], d, "field_lookups", "field reads per transport",
               # Not "three per step". RKN4 evaluates at four points but three
               # distinct positions, so three is what the method suggests and
               # not what an implementation does: counted on the pinned ACTS
               # build this comes out at 3.3 to 4.2 reads per attempted step.
               # The panel carries the count and lets it speak.
               "Why. Field reads per transport, counted")

    total = int(np.nansum(d["n_transport"].to_numpy().astype(float)))
    sub = (f"Bins: {', '.join(_bin_labels(d))} GeV. {total:,} transports timed. "
           f"Baseline {baseline}.")
    if note:
        sub += f"  {note}"
    fig.suptitle("Transport latency against pT\n" + sub, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, crossing


# ------------------------------------------------------------------ synthetic

def write_demo(path):
    """A file that satisfies the contract, so the plotting can be smoke tested
    before the real measurement exists.

    The numbers are invented. They are not a prediction and no figure drawn from
    them may leave this machine, which is why every such figure is stamped.
    """
    edges = [(1, 2), (2, 4), (4, 8), (8, 20), (20, 50)]
    rows = []
    for lo, hi in edges:
        mid = (lo * hi) ** 0.5
        steps = 2.0 + 6.0 / mid
        acts = 300.0 + 340.0 * steps
        rows.append(dict(pt_lo=lo, pt_hi=hi, arm="acts_eigen_cov",
                         n_transport=4000, ns_median=round(acts, 1),
                         ns_lo=round(acts * 0.98, 1), ns_hi=round(acts * 1.02, 1),
                         rk_steps=round(steps, 2),
                         field_lookups=round(3 * steps, 2)))
        rows.append(dict(pt_lo=lo, pt_hi=hi, arm="kernel_matched",
                         n_transport=4000, ns_median=2400.0,
                         ns_lo=2350.0, ns_hi=2450.0,
                         rk_steps=None, field_lookups=1.0))
    pl.DataFrame(rows).write_csv(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="measurement file, see the contract above")
    ap.add_argument("--demo", metavar="PATH",
                    help="write a synthetic file there and plot it")
    ap.add_argument("--baseline", default="acts_eigen_cov")
    ap.add_argument("--out", default="fig/speed_by_pt.png")
    # The file also carries HANDOFF item 2c's per-step and per-jump arms, whose
    # unit is one stepper call rather than one transport. Drawing them on these
    # axes would put two different quantities on one y, so the default is the
    # three whole-transport arms and anything else has to be asked for.
    ap.add_argument("--arms", default="acts_eigen_cov,kernel_matched,"
                                      "helix_matched",
                    help="comma-separated arms to draw, or `all`")
    args = ap.parse_args()

    note = ""
    if args.demo:
        args.csv = write_demo(args.demo)
        note = "SYNTHETIC DEMO DATA, NOT A MEASUREMENT"
    if not args.csv:
        raise SystemExit("give --csv or --demo")

    d = load(args.csv)
    if args.arms != "all":
        want = [x.strip() for x in args.arms.split(",") if x.strip()]
        have = set(d["arm"].unique().to_list())
        missing = [x for x in want if x not in have]
        if missing:
            raise SystemExit(f"{args.csv} has no rows for {missing}")
        d = d.filter(pl.col("arm").is_in(want))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    path, crossing = figure(d, args.baseline, out, note)
    print(path)
    if crossing is not None:
        print(f"ratio crosses 1 at about {crossing:.1f} GeV")
    else:
        print("ratio does not cross 1 inside the measured range")


if __name__ == "__main__":
    main()
