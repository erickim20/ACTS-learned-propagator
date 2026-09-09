#!/usr/bin/env python3
"""What each change between ACTS's CKF and this one costs the track-finding
stage, one segment per change.

Where the learned transport sits inside one leg is a path-length statement;
this draws what the same change costs in seconds, which is a different quantity
and wants its own figure.

The forwarding members are not the cost. Not one of the eighteen is
emitted as a symbol, every `collider_ml::` symbol together is 0.19 % of the run,
and the 0.24 s is `SympyStepper` replaced by `EigenStepper` plus the propagator
and CKF templates re-instantiated in this translation unit. That is one segment
in this figure and it is labelled by what it is.

THE LADDER. Six configurations, each one change from the one above:

    stock             ACTS prebuilt, SympyStepper
    learned-off       this tree's translation unit, EigenStepper, ACTS's navigator
    new-off-walkoff   NewNavigator, every member forwarding
    new-off           the walk
    new-helix-plan    the helix replaces Runge-Kutta over the whole transport
    new-learned-plan  the network

There are two panels because the first four rungs reconstruct the same tracks:
`new-off` reproduces `stock` to 0.010 efficiency points with the same track
counts, so a paired difference between them is the
seam and nothing else and the unit is seconds. The last two do not. They hand
the filter a different number of candidates, so a per-event second divides two
different amounts of work by the same denominator. They go in their own panel,
in nanoseconds per candidate the filter built, which is a DIFFERENT QUANTITY
and is not a speedup. Drawing all six as one stack would present it as one.

The clock. This box steps `CLOCK_REALTIME` backwards, so some rows are reported
seconds short. The step measured 2.3 s in one job and 0.7 s in the next, so an
absolute floor cannot catch it in both, and the rule here is
relative to each (bin, arm) median. The whole (bin, repeat) block goes, because
a block missing one arm is not a paired difference. The count that survives is
printed and goes on the figure.

    python -m prop.plot_wrapper ~/j3/A/timing.csv --out fig/wrapper_split.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import polars as pl

# (arm, what the step from the arm above it adds). The order is the ladder.
LADDER = [
    ("stock", "stock, the SympyStepper CKF"),
    ("learned-off", "EigenStepper, and the CKF re-instantiated here"),
    ("new-off-walkoff", "NewNavigator, every member forwarding"),
    ("new-off", "the walk"),
]
PER_CAND = [
    ("new-helix-plan", "the helix over the whole transport"),
    ("new-learned-plan", "the network"),
]
SHADE = ["0.80", "0.55", "0.35", "0.15"]


def drop_blocks(d, frac):
    """Blocks holding a row far below its own (bin, arm) median, dropped whole.

    Returns the surviving frame and (dropped, total) blocks.
    """
    med = (d.group_by(["bin", "arm"])
            .agg(pl.col("trackfinding_s").median().alias("med")))
    j = d.join(med, on=["bin", "arm"], how="inner")
    bad = (j.filter(pl.col("trackfinding_s") < frac * pl.col("med"))
            .select(["bin", "repeat"]).unique())
    total = d.select(["bin", "repeat"]).unique().height
    if bad.height == 0:
        return d, 0, total
    return d.join(bad, on=["bin", "repeat"], how="anti"), bad.height, total


def paired(d, arm, baseline):
    """One difference per surviving block, arm minus baseline, pooled."""
    b = (d.filter(pl.col("arm") == baseline)
          .select(["bin", "repeat", "trackfinding_s"])
          .rename({"trackfinding_s": "base"}))
    j = (d.filter(pl.col("arm") == arm)
          .join(b, on=["bin", "repeat"], how="inner"))
    return (j["trackfinding_s"] - j["base"]).to_numpy()


def ns_per_candidate(d, arm):
    """Median nanoseconds of track finding per candidate the filter built."""
    s = d.filter((pl.col("arm") == arm) & (pl.col("selected") > 0))
    if s.height == 0:
        return float("nan")
    return float(np.median(1e9 * s["trackfinding_s"].to_numpy()
                           / s["selected"].to_numpy()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="the CSV sim/run_timing_blocks.sh wrote")
    ap.add_argument("--out", default="fig/wrapper_split.png")
    ap.add_argument("--frac", type=float, default=0.9,
                    help="drop the whole (bin, repeat) block when any row in "
                         "it is below this fraction of its own (bin, arm) "
                         "median. Nothing sits between 0.50 "
                         "and 0.75 of a median, so the threshold is not fitted")
    a = ap.parse_args()

    d = pl.read_csv(a.csv, null_values=["NA", "nan"])
    d = d.filter(pl.col("stage_s").is_not_null())
    have = set(d["arm"].unique().to_list())
    want = [arm for arm, _ in LADDER + PER_CAND]
    missing = [x for x in want if x not in have]
    if missing:
        raise SystemExit(f"{a.csv} has no rows for {missing}")

    d, dropped, total = drop_blocks(d, a.frac)
    kept = total - dropped

    base = float(np.median(
        d.filter(pl.col("arm") == LADDER[0][0])["trackfinding_s"].to_numpy()))
    segs = [(LADDER[0][1], base, SHADE[0])]
    for (prev, _), (arm, label) in zip(LADDER, LADDER[1:]):
        segs.append((label, float(np.median(paired(d, arm, prev))),
                     SHADE[len(segs) % len(SHADE)]))

    fig = plt.figure(figsize=(10.5, 6.0))
    ax = fig.add_axes((0.06, 0.755, 0.92, 0.115))
    lx = fig.add_axes((0.06, 0.500, 0.92, 0.185))
    bx = fig.add_axes((0.30, 0.105, 0.62, 0.270))

    left = 0.0
    handles = []
    for i, (label, w, colour) in enumerate(segs):
        ax.barh(0, w, left=left, height=0.42, color=colour, edgecolor="black",
                linewidth=0.8)
        left += w
        handles.append(Patch(facecolor=colour, edgecolor="black",
                             label=f"{label}   "
                                   + (f"{w:.2f} s" if i == 0
                                      else f"{w:+.2f} s")))
    # Drawn to scale, so the additions are slivers and the numbers go in the
    # legend rather than inside segments too narrow to hold them. A segment
    # that measured zero has no width at all, which would read as an omission,
    # so it is named on the axis where it would have been.
    for i, (label, w, _) in enumerate(segs):
        if i and abs(w) < 0.02:
            xpos = sum(x[1] for x in segs[:i])
            ax.annotate(f"{w:+.2f} s", xy=(xpos, 0.21),
                        xytext=(xpos - 0.16 * left, 0.62), fontsize=9,
                        ha="right", va="bottom",
                        arrowprops=dict(arrowstyle="-", lw=0.8, color="black",
                                        shrinkA=2, shrinkB=1))
    ax.set_xlim(0, left * 1.02)
    ax.set_ylim(-0.30, 0.95)
    ax.set_yticks([])
    ax.set_xlabel("seconds of Algorithm:TrackFindingAlgorithm, 10,000 events")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title(
        "Each change from ACTS's CKF to this one, in seconds\n"
        "the four rungs that reconstruct the same tracks; paired differences "
        f"inside (bin, repeat) blocks,\nmedians over the survivors, "
        f"{kept} of {total} blocks after the clock", fontsize=10, pad=10)

    lx.set_axis_off()
    lx.legend(handles=handles, loc="upper center", ncol=1, frameon=False,
              fontsize=9.5, handlelength=1.8, labelspacing=0.7)

    # The third panel. A different quantity, its own axis, its own words.
    rows = [(LADDER[0][0], "stock")] + [(arm, lbl) for arm, lbl in PER_CAND]
    names = [lbl for _, lbl in rows]
    vals = [ns_per_candidate(d, arm) for arm, _ in rows]
    y = np.arange(len(rows))[::-1]
    bx.barh(y, vals, height=0.5, color="0.65", edgecolor="black", linewidth=0.8)
    for yy, v in zip(y, vals):
        bx.text(v, yy, f"  {v:,.0f}", va="center", ha="left", fontsize=9)
    bx.set_yticks(y)
    bx.set_yticklabels(names, fontsize=9)
    bx.set_xlim(0, max(vals) * 1.25)
    bx.set_xlabel("nanoseconds per candidate the filter built")
    for side in ("top", "right"):
        bx.spines[side].set_visible(False)
    bx.set_title("The two rungs that do not reconstruct the same tracks\n"
                 "a per-candidate time is not a speedup: it is here so arms "
                 "building different\namounts of filter work can be compared "
                 "at all", fontsize=10, pad=10)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)

    print(f"blocks {kept} of {total} kept, {dropped} dropped whole")
    for i, (label, w, _) in enumerate(segs):
        print(f"{label:<46}{w:+.2f} s" if i else f"{label:<46}{w:.2f} s")
    print(f"{'total, rungs 1 to 4':<46}{left:.2f} s")
    print()
    for (arm, label), v in zip(rows, vals):
        print(f"{label:<46}{v:>10,.0f} ns/candidate   ({arm})")
    print(out)


if __name__ == "__main__":
    main()
