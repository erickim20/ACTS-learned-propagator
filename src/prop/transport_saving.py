#!/usr/bin/env python3
"""Turn the latency curve into seconds. HANDOFF items 2, 2b and 2c.

The curve says what one surface-to-surface transport costs, per pT bin. The
census says how many transports happen, at what momentum, whether they arrive
and how many stepper calls they take. Neither is a saving on its own:

    total = sum over bins of (ns saved per call) x (calls in bin)

The second factor matters more than it sounds. The gap between the arms is
about 1300 ns per transport in the softest bin and about 350 in the hardest, so
the same curve gives answers a factor of nearly four apart depending only on
where the transports are.

Two things the census counts that a track-level number would not. A track uses
about thirteen transports, so tracks and transports are different quantities.
And a combinatorial track finder branches and abandons most of what it explores,
and an abandoned branch has already paid for its transports, so the momentum
distribution of transport calls is not the distribution of surviving tracks.

Two predictions, and they do not agree.

`--unit transport` multiplies the per-transport arms by the transport count.
That is item 2's arithmetic and it is what the benchmark's own rows invite.

`--unit step` multiplies a per-step arm by the STEP count and a per-jump arm by
the transport count. That is item 2c's, and it exists because the census showed
the two units are not interchangeable: inside a CKF the stock stepper spends
one to one and a half `step()` calls per sensitive destination, while the
benchmark's whole-transport arm spends two to four. The benchmark flies the
entire gap between two hits; the navigator hands the destination over only for
the last part of it.

    python -m prop.transport_saving --speed speed_by_pt.csv \\
        --census transport_pt.csv --arm helix_matched \\
        --observed-baseline 72.5 --observed-arm 74.8

`--observed-*` are wall-clock reconstruction seconds for the two arms on the
same events. Given them, the tool prints the predicted saving against the
observed one. **The gap is the result and is not to be tuned away.** If the
prediction does not match, the learned path is doing work the standalone
benchmark does not contain, and naming that work is the next thing to do.
"""
from __future__ import annotations

import argparse
import sys

import polars as pl

CENSUS_COUNTS = ["n_transport", "n_reached", "n_step", "n_step_reached"]


def load_speed(path):
    d = pl.read_csv(path, null_values=["", "NA", "nan"])
    need = {"pt_lo", "pt_hi", "arm", "ns_median"}
    if need - set(d.columns):
        raise SystemExit(f"{path} is missing {sorted(need - set(d.columns))}")
    return d


def load_census(path):
    """The census file, without its total row and its two open-ended rows.

    Underflow and overflow are kept separately: they were paid for and belong in
    the total, but neither has a measured latency because neither is a bin of
    the curve, so folding them into the prediction would invent one.

    A census written before HANDOFF item 2c carries only `n_transport`. The
    missing columns are filled with null rather than zero, so a prediction that
    needs them fails loudly instead of quietly reporting no steps.
    """
    d = pl.read_csv(path, null_values=[""])
    for c in CENSUS_COUNTS:
        if c not in d.columns:
            d = d.with_columns(pl.lit(None, dtype=pl.Int64).alias(c))
    totals = {}
    tot = d.filter(pl.col("pt_lo") == "total")
    if tot.height:
        totals = {c: tot[c][0] for c in CENSUS_COUNTS}
    other = d.filter(pl.col("pt_lo") == "other")
    totals["n_step_other"] = int(other["n_step"][0]) if other.height else None
    # `!= "total"` alone drops the underflow row too: its pt_lo is empty, which
    # the reader turns into null, and null != "total" is null rather than true.
    d = d.filter(pl.col("pt_lo").is_null()
                 | ~pl.col("pt_lo").is_in(["total", "other"]))
    d = d.with_columns(
        pl.col("pt_lo").cast(pl.Float64, strict=False),
        pl.col("pt_hi").cast(pl.Float64, strict=False),
    )
    under = d.filter(pl.col("pt_lo").is_null())
    over = d.filter(pl.col("pt_hi").is_null())
    binned = d.filter(pl.col("pt_lo").is_not_null()
                      & pl.col("pt_hi").is_not_null())

    def outside(frame):
        if not frame.height:
            return {c: 0 for c in CENSUS_COUNTS}
        d = {c: (frame[c][0] or 0) for c in CENSUS_COUNTS}
        d["edge"] = frame["pt_hi"][0] if frame["pt_lo"][0] is None \
            else frame["pt_lo"][0]
        return d

    return binned, outside(under), outside(over), totals


def price_edges(binned, under, over, speed):
    """Give the underflow and overflow rows the edges of a cell that covers them.

    The census bins on the curve's own edges and keeps whatever falls outside
    them, because a transport below 0.5 GeV was still paid for. If the curve
    later grows a cell that covers one of those rows, the row is no longer
    outside anything and leaving it unpriced would charge those transports
    nothing at all. That is not a small correction: the learned arm puts 15 % of
    its transports below the seeding minimum.

    A row is adopted only if the curve has exactly one cell ending at the
    underflow's edge, or starting at the overflow's. Anything else is left
    unpriced and reported as such.
    """
    rows = []
    lo = speed.filter(pl.col("pt_hi") == under.get("edge"))
    if under.get("n_transport") and lo.height:
        edge = lo["pt_lo"].min()
        rows.append({"pt_lo": float(edge), "pt_hi": float(under["edge"]),
                     **{c: int(under[c]) for c in CENSUS_COUNTS}})
        under = dict(under, priced=True)
    hi = speed.filter(pl.col("pt_lo") == over.get("edge"))
    if over.get("n_transport") and hi.height:
        edge = hi["pt_hi"].max()
        rows.append({"pt_lo": float(over["edge"]), "pt_hi": float(edge),
                     **{c: int(over[c]) for c in CENSUS_COUNTS}})
        over = dict(over, priced=True)
    if rows:
        binned = pl.concat(
            [binned.select("pt_lo", "pt_hi", *CENSUS_COUNTS),
             pl.DataFrame(rows).select("pt_lo", "pt_hi", *CENSUS_COUNTS)],
            how="vertical_relaxed").sort("pt_lo")
    return binned, under, over


def ns(frame, arm, path):
    r = frame.filter(pl.col("arm") == arm)
    if r.is_empty():
        raise SystemExit(f"{path} has no rows for {arm!r}")
    return r.select("pt_lo", "pt_hi",
                    pl.col("ns_median").alias(f"ns_{arm}"))


def arrival(name, binned, under, over, totals):
    n = totals.get("n_transport")
    r = totals.get("n_reached")
    s = totals.get("n_step")
    print(f"--- {name}")
    if n is None:
        print("    no total row")
        return
    print(f"    transport calls        {n:,}")
    if r is not None:
        miss = n - r
        print(f"    reached the surface    {r:,}  ({100.0 * r / n:.3f}%)")
        print(f"    did not                {miss:,}  ({100.0 * miss / n:.3f}%)")
    if s is not None:
        print(f"    stepper calls          {s:,}  ({s / n:.3f} per transport)")
    o = totals.get("n_step_other")
    if o is not None:
        # Paid by every arm, because the learned jump fires only on sensitive
        # destinations. If this differs across arms the per-step prediction is
        # missing a term.
        print(f"    steps toward a portal  {o:,}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--speed", default="speed_by_pt.csv")
    ap.add_argument("--census", required=True,
                    help="census of the BASELINE arm, from TRANSPORT_CENSUS")
    ap.add_argument("--census-arm", default=None,
                    help="census of the arm being predicted, if it differs. "
                         "The learned arm's transport count is not the stock "
                         "arm's and assuming it is was HANDOFF 2b's doubt")
    ap.add_argument("--unit", choices=["transport", "step"],
                    default="transport",
                    help="transport: item 2's arithmetic, whole-transport arms "
                         "times the transport count. step: item 2c's, a "
                         "per-step arm times the step count against a per-jump "
                         "arm times the transport count")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--observed-baseline", type=float, default=None,
                    help="wall-clock reconstruction seconds for the baseline")
    ap.add_argument("--observed-arm", type=float, default=None)
    a = ap.parse_args()

    if a.unit == "transport":
        baseline = a.baseline or "acts_eigen_cov"
        arm = a.arm or "helix_matched"
    else:
        baseline = a.baseline or "acts_step"
        arm = a.arm or "helix_jump"

    speed = load_speed(a.speed)
    edges = speed.filter(pl.col("arm") == baseline).select("pt_lo", "pt_hi")
    binned, under, over, totals = load_census(a.census)
    binned, under, over = price_edges(binned, under, over, edges)
    if a.census_arm:
        armBinned, armUnder, armOver, armTotals = load_census(a.census_arm)
        armBinned, armUnder, armOver = price_edges(armBinned, armUnder,
                                                   armOver, edges)
    else:
        armBinned, armUnder, armOver, armTotals = binned, under, over, totals

    print(f"latency  {a.speed}")
    print(f"census   {a.census}")
    if a.census_arm:
        print(f"         {a.census_arm}   (the arm's own)")
    print(f"unit     one {a.unit}")
    print(f"arm      {arm}   against baseline {baseline}")
    print()

    arrival(f"baseline census, {a.census}", binned, under, over, totals)
    if a.census_arm:
        arrival(f"arm census, {a.census_arm}", armBinned, armUnder, armOver,
                armTotals)

    # What each side is multiplied by. In `transport` units both sides are paid
    # once per transport. In `step` units the baseline pays per stepper call and
    # the learned arm pays once per destination, which is what the census
    # measured: its step column reads 1.000 on the helix arm exactly because
    # the jump lands on the surface in one call.
    baseCount = "n_step" if a.unit == "step" else "n_transport"
    armCount = "n_transport"

    j = (ns(speed, baseline, a.speed).rename({f"ns_{baseline}": "ns_base"})
         .join(ns(speed, arm, a.speed).rename({f"ns_{arm}": "ns_arm"}),
               on=["pt_lo", "pt_hi"], how="inner")
         .join(binned.select("pt_lo", "pt_hi",
                             pl.col(baseCount).alias("n_base")),
               on=["pt_lo", "pt_hi"], how="left")
         .join(armBinned.select("pt_lo", "pt_hi",
                                pl.col(armCount).alias("n_arm"),
                                pl.col("n_step").alias("n_arm_step")),
               on=["pt_lo", "pt_hi"], how="left")
         .sort("pt_lo"))

    if j["n_base"].null_count() == j.height:
        raise SystemExit(f"{a.census} carries no {baseCount} column; it "
                         f"predates HANDOFF item 2c. Re-run the census.")

    j = j.with_columns(
        pl.col("n_base").fill_null(0), pl.col("n_arm").fill_null(0),
        pl.col("n_arm_step").fill_null(0),
    ).with_columns(
        ((pl.col("ns_base") * pl.col("n_base")
          - pl.col("ns_arm") * pl.col("n_arm")) * 1e-9).alias("s_saved"))

    head = (f"{'pT bin':>12s} {'ns base':>9s} {'base calls':>12s} "
            f"{'ns arm':>9s} {'arm calls':>12s} {'s saved':>10s}")
    print(head)
    print("-" * len(head))
    for r in j.iter_rows(named=True):
        print(f"{r['pt_lo']:5g}-{r['pt_hi']:<6g} {r['ns_base']:9.0f} "
              f"{r['n_base']:12,d} {r['ns_arm']:9.0f} {r['n_arm']:12,d} "
              f"{r['s_saved']:10.3f}")

    predicted = float(j["s_saved"].sum())
    print("-" * len(head))
    print(f"{'':>12s} {'':>9s} {int(j['n_base'].sum()):12,d} {'':>9s} "
          f"{int(j['n_arm'].sum()):12,d} {predicted:10.3f}")
    print()

    if a.unit == "step":
        # Two terms the per-transport arithmetic never had, and both are real
        # stepper calls that a reconstruction pays.
        #
        # FALLBACK. A learned jump whose solve fails does not replace the step;
        # `LearnedStepper::step` reads `p.ok` only after `transport()` has run
        # and then falls through to the inner EigenStepper, so the arm pays
        # both. The census sees this as more stepper calls than destinations.
        #
        # PORTALS. Three quarters of a CKF's stepper calls are aimed at a
        # portal or a layer approach rather than at a module. The learned jump
        # never fires on those, so they cancel between two arms that navigate
        # identically and do not cancel between two that do not.
        #
        # Both are priced at the step-count-weighted mean of `acts_step`,
        # because neither carries a pT bin of its own.
        w = j.select((pl.col("ns_base") * pl.col("n_base")).sum()
                     / pl.col("n_base").sum()).item()
        extra = j.with_columns(
            (pl.col("n_arm_step") - pl.col("n_arm")).alias("fallback"))
        fb = int(extra["fallback"].sum())
        fbS = float((extra["fallback"] * extra["ns_base"]).sum()) * 1e-9
        oBase = totals.get("n_step_other") or 0
        oArm = armTotals.get("n_step_other") or 0
        oS = (oBase - oArm) * w * 1e-9
        print(f"mean cost of one step, weighted   {w:8.1f} ns")
        print(f"arm's fallback steps              {fb:12,d}  "
              f"{-fbS:+10.3f} s")
        print(f"steps toward a portal, baseline   {oBase:12,d}")
        print(f"steps toward a portal, arm        {oArm:12,d}  "
              f"{oS:+10.3f} s")
        predicted += -fbS + oS
        print()

    counted = int(j["n_arm"].sum())
    print(f"transports inside the curve's bins   {counted:,}")
    for name, side in (("below the lowest bin edge", armUnder),
                       ("above the highest bin edge", armOver)):
        mark = "priced" if side.get("priced") else "NOT priced"
        print(f"{name:<36s} {side['n_transport']:,} ({mark})")
    total = armTotals.get("n_transport")
    if total:
        outside = total - counted
        print(f"total transport calls                {total:,}")
        print(f"outside the curve, so unpriced        {outside:,} "
              f"({100.0 * outside / total:.1f}%)")
    print()
    print(f"PREDICTED saving, {arm} against {baseline}, "
          f"per {a.unit}: {predicted:+.2f} s")

    if a.observed_baseline is not None and a.observed_arm is not None:
        observed = a.observed_baseline - a.observed_arm
        print(f"OBSERVED  {a.observed_baseline:.1f} s against "
              f"{a.observed_arm:.1f} s, so {observed:+.2f} s")
        print(f"GAP       {observed - predicted:+.2f} s "
              f"(observed minus predicted)")
        print()
        print("The gap is the result. A prediction that does not match means")
        print("the arm does work the standalone benchmark does not contain.")
        print("Name that work; do not tune anything to close it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
