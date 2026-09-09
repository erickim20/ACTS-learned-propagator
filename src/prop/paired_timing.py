"""Paired arm-to-arm timing, with the resolution stated.

HANDOFF task 2. The previous measurement took each arm as its own block of runs
and compared medians, and the repeat-to-repeat spread was the size of the
arm-to-arm difference, so it settled a sign and nothing else. Three things
change here and this reads all of them:

  Blocks. One block is (bin, repeat) and holds every arm back to back, so a
  difference is taken inside a block and anything that drifts over the run
  lands on every arm of that block. The statistic is the distribution of the
  paired differences, not the difference of two medians.

  Ten repeats, so a difference has a spread of its own.

  The per-algorithm line. Digitisation, seeding, the writers and the geometry
  build are identical in every arm and are most of a run's wall clock, so their
  variance is injected into every comparison while carrying no signal.
  `Algorithm:TrackFindingAlgorithm` is the part the stepper is in.

    python -m prop.paired_timing ~/exp/t2/t2.csv --baseline stock

Reports, per arm and per bin, the median of the ten paired differences against
the baseline and the full range, and says which differences are larger than
their own spread.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import polars as pl

COLS = [("stage_s", "whole reconstruction stage"),
        ("trackfinding_s", "TrackFindingAlgorithm")]


def drop_relative(d, frac):
    """Blocks holding a row far below its own (bin, arm) median, dropped whole.

    The clock step is a fixed number of seconds, and the arms it lands on run
    at anything from 2 to 4 s, so an absolute floor that catches it in the
    fastest arm passes it in the slowest. The median is taken per (bin, arm)
    because that is the only grouping in which every row is the same
    measurement of the same thing.

    The block goes and not the row, for `--floor`'s reason: a block with one
    arm missing is not a paired difference.

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


def paired(d, col, baseline):
    """(bin, arm) -> array of one difference per repeat, arm minus baseline."""
    base = (d.filter(pl.col("arm") == baseline)
             .select(["bin", "repeat", col])
             .rename({col: "base"}))
    j = d.join(base, on=["bin", "repeat"], how="inner")
    j = j.with_columns((pl.col(col) - pl.col("base")).alias("diff"))
    out = {}
    for arm in d["arm"].unique(maintain_order=True):
        for b in sorted(d["bin"].unique()):
            s = j.filter((pl.col("arm") == arm) & (pl.col("bin") == b))
            out[(arm, b)] = s["diff"].to_numpy()
    return out


def table(d, col, label, baseline):
    arms = [a for a in d["arm"].unique(maintain_order=True) if a != baseline]
    bins = sorted(d["bin"].unique())
    p = paired(d, col, baseline)

    print(f"### {label}, seconds")
    print()
    print("| bin | " + " | ".join(f"{a}" for a in [baseline] + arms) + " |")
    print("| --- | " + " | ".join("---:" for _ in [baseline] + arms) + " |")
    for b in bins:
        row = [f"{b}"]
        for a in [baseline] + arms:
            v = d.filter((pl.col("arm") == a) & (pl.col("bin") == b))[col]
            row.append(f"{np.median(v):.2f} ({v.min():.2f} to {v.max():.2f})")
        print("| " + " | ".join(row) + " |")
    print()
    print(f"Median of ten repeats, full range in brackets.")
    print()

    print(f"### {label}, paired difference against {baseline}")
    print()
    print("| bin | arm | median | min | max | spread | resolved |")
    print("| --- | --- | ---: | ---: | ---: | ---: | --- |")
    for b in bins:
        for a in arms:
            v = p[(a, b)]
            if len(v) == 0:
                continue
            spread = v.max() - v.min()
            med = float(np.median(v))
            # A difference is called resolved when every one of the ten paired
            # differences has the same sign, which is what the pairing is for:
            # a spread that does not cross zero is a difference the box's drift
            # cannot explain.
            resolved = "yes" if (v > 0).all() or (v < 0).all() else "no"
            print(f"| {b} | {a} | {med:+.2f} | {v.min():+.2f} | {v.max():+.2f} "
                  f"| {spread:.2f} | {resolved} |")
    print()
    print("One difference per repeat, taken inside the block that holds both "
          "arms. `resolved` is yes when all ten differences have the same "
          "sign.")
    print()

    print(f"### {label}, pooled over the five bins")
    print()
    print("| arm | n | median | p10 | p90 | min | max | resolved |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for a in arms:
        v = np.concatenate([p[(a, b)] for b in bins])
        resolved = "yes" if (v > 0).all() or (v < 0).all() else "no"
        print(f"| {a} | {len(v)} | {np.median(v):+.2f} | "
              f"{np.percentile(v, 10):+.2f} | {np.percentile(v, 90):+.2f} | "
              f"{v.min():+.2f} | {v.max():+.2f} | {resolved} |")
    print()

    # The resolution of the measurement, which is the sentence the handoff
    # wants a number in: the largest paired difference the baseline shows
    # against itself is zero by construction, so the floor is taken from the
    # spread of the differences that are measured.
    worst = max(float(np.ptp(np.concatenate([p[(a, b)] for b in bins])))
                for a in arms)
    print(f"Widest spread on any pooled paired difference: {worst:.2f} s. A "
          f"difference smaller than that is not separated by this "
          f"measurement.")
    print()


def per_candidate(d, baseline):
    """Nanoseconds of track finding per candidate the filter built.

    A DIFFERENT QUANTITY FROM EVERYTHING ABOVE, and it is here because for two
    of the arms nothing above is legitimate. The arms whose transport is the
    learned one do not reconstruct the same tracks as `stock`: they build a
    different number of candidates, so a per-event time divides two different
    amounts of work by the same denominator. `finalize`'s selected count is the
    work the filter actually did, and it is the only denominator the arms
    share.

    It is not a speedup and it does not rank an arm against `stock`. An arm
    that builds a third more candidates and takes a third longer reads the
    same here, which is the point: it says the per-candidate cost did not
    change, not that the arm is free.
    """
    if "selected" not in d.columns:
        return
    d = d.filter(pl.col("selected").is_not_null() & (pl.col("selected") > 0))
    if d.height == 0:
        return
    d = d.with_columns(
        (1e9 * pl.col("trackfinding_s") / pl.col("selected")).alias("ns"))
    arms = d["arm"].unique(maintain_order=True).to_list()
    bins = sorted(d["bin"].unique())

    print("### Nanoseconds per candidate the filter built")
    print()
    print("| bin | " + " | ".join(arms) + " |")
    print("| --- | " + " | ".join("---:" for _ in arms) + " |")
    for b in bins:
        row = [f"{b}"]
        for a in arms:
            v = d.filter((pl.col("arm") == a) & (pl.col("bin") == b))["ns"]
            row.append(f"{np.median(v):.0f}" if len(v) else "-")
        print("| " + " | ".join(row) + " |")
    print()

    print("### Candidates the filter built, and against " + baseline)
    print()
    print("| bin | " + " | ".join(arms) + " |")
    print("| --- | " + " | ".join("---:" for _ in arms) + " |")
    for b in bins:
        row = [f"{b}"]
        base = d.filter((pl.col("arm") == baseline) &
                        (pl.col("bin") == b))["selected"]
        base = float(np.median(base)) if len(base) else float("nan")
        for a in arms:
            v = d.filter((pl.col("arm") == a) & (pl.col("bin") == b))["selected"]
            if not len(v):
                row.append("-")
                continue
            m = float(np.median(v))
            row.append(f"{m:,.0f} ({100 * (m - base) / base:+.1f} %)")
        print("| " + " | ".join(row) + " |")
    print()
    print("Median over the surviving repeats. The per cent is against "
          f"`{baseline}`'s count in the same bin. A per-candidate time is not "
          "a speedup: it exists so that arms building different amounts of "
          "filter work can be compared at all.")
    print()


LOG_NAME = re.compile(r"^[^_]+_(?P<arm>.+)_b(?P<bin>\d+)_r(?P<repeat>\d+)\.log$")
JUMP_FIRED = re.compile(r"jump_fired=(\d+)")


def read_jump_fired(logs):
    """{(arm, bin, repeat): jump_fired} from each run's `[navcensus]` block.

    `cpp/NavCensus.hpp` prints the block unconditionally and `jump_fired` is
    the count of destinations the kernel answered and wrote a state back for,
    which is the count of learned transports the run paid for. It is zero on
    an arm whose transport is off, and that is a statement about the arm and
    not a missing number.
    """
    rows = []
    for path in sorted(Path(logs).glob("*.log")):
        m = LOG_NAME.match(path.name)
        if not m:
            continue
        fired = None
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("[navcensus]"):
                f = JUMP_FIRED.search(line)
                if f:
                    fired = int(f.group(1))
        if fired is None:
            continue
        rows.append({"arm": m["arm"], "bin": int(m["bin"]),
                     "repeat": int(m["repeat"]), "jump_fired": fired})
    if not rows:
        return None
    return pl.DataFrame(rows)


def ceiling(d, logs, ns_transport):
    """How much of the per-candidate time a free transport could remove.

    The benchmark in `cpp/bench_kernel.cpp` prices one surface-to-surface
    transport and says in its own output that it excludes the navigator and
    material. This is the arithmetic that turns that number into a share of
    what the filter pays per candidate:

        transports per candidate = jump_fired / the finalize selected count
        predicted transport time = transports per candidate x ns_transport
        the ceiling              = predicted over the measured per-candidate

    The ceiling is an upper bound in three separate ways. The benchmark is the
    kernel's own cost with nothing around it; a transport that were free would
    still leave the call, the plan and the write-back; and the arms that fire
    the transport build more candidates than the arms that do not, so the
    denominator is already larger than `stock`'s.
    """
    fired = read_jump_fired(logs)
    if fired is None:
        print(f"no `[navcensus]` blocks under {logs}")
        return
    d = d.filter(pl.col("selected").is_not_null() & (pl.col("selected") > 0))
    j = d.join(fired, on=["arm", "bin", "repeat"], how="inner")
    if j.height == 0:
        print(f"no run under {logs} matches a row of the timing csv")
        return
    j = j.with_columns([
        (pl.col("jump_fired") / pl.col("selected")).alias("per_cand"),
        (1e9 * pl.col("trackfinding_s") / pl.col("selected")).alias("ns"),
    ])

    arms = j["arm"].unique(maintain_order=True).to_list()
    bins = sorted(j["bin"].unique())

    print("### The ceiling: the share of the per-candidate time the transport "
          "could be")
    print()
    print(f"One transport is {ns_transport:.0f} ns.")
    print()
    print("| bin | arm | jump_fired | candidates | transports per candidate | "
          "predicted [ns] | measured [ns] | ceiling |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for b in bins:
        for a in arms:
            v = j.filter((pl.col("arm") == a) & (pl.col("bin") == b))
            if v.height == 0:
                continue
            f = float(np.median(v["jump_fired"].to_numpy()))
            c = float(np.median(v["selected"].to_numpy()))
            pc = float(np.median(v["per_cand"].to_numpy()))
            ns = float(np.median(v["ns"].to_numpy()))
            pred = pc * ns_transport
            print(f"| {b} | {a} | {f:,.0f} | {c:,.0f} | {pc:.2f} | "
                  f"{pred:,.0f} | {ns:,.0f} | {100 * pred / ns:.1f} % |")
    print()
    print("Median over the surviving repeats, per bin. `jump_fired` is "
          "`cpp/NavCensus.hpp`'s counter and `candidates` is "
          "`TrackFindingAlgorithm::finalize`'s selected count, both read off "
          "the same run. An arm whose transport is off fires none and its "
          "ceiling is zero by construction.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--baseline", default="stock")
    ap.add_argument("--floor", type=float, default=0.0,
                    help="drop the whole (bin, repeat) block when any arm in "
                         "it reports a track-finding time below this. The "
                         "Sequencer times with std::chrono::high_resolution_"
                         "clock, which is CLOCK_REALTIME in libstdc++, and "
                         "this box steps that clock backwards by about 2.3 s "
                         "at intervals, so an algorithm the step lands on is "
                         "reported 2.3 s short and sometimes negative. The "
                         "block goes rather than the row, because a block with "
                         "one arm missing is not a paired difference. AN "
                         "ABSOLUTE THRESHOLD, and the step is not the same "
                         "size between jobs: 2.3 s in one and 0.7 s in the "
                         "next, so --floor 1.0 caught the first and missed the "
                         "second. Prefer --rel-floor")
    ap.add_argument("--rel-floor", type=float, default=0.0,
                    help="drop the whole (bin, repeat) block when any row in "
                         "it is below this fraction of its own (bin, arm) "
                         "median. Relative rather than absolute, because the "
                         "backward clock step is a fixed number of seconds off "
                         "an arm whose own time is 2 s in one arm and 4 s in "
                         "another, so no single absolute threshold catches it "
                         "in every arm. 0.9 is the value used; nothing sits "
                         "between 0.50 and 0.75 of a "
                         "median, so the threshold is not fitted")
    ap.add_argument("--logs", default=None,
                    help="directory of run logs named <prefix>_<arm>_b<bin>_"
                         "r<repeat>.log, for the ceiling section. Each log's "
                         "`[navcensus]` block carries `jump_fired`, which is "
                         "the transport count the per-candidate time is "
                         "divided against")
    ap.add_argument("--ns-transport", type=float, default=2200.0,
                    help="nanoseconds for one surface-to-surface transport, "
                         "for the ceiling. The default is the kernel column "
                         "of `speed_by_pt_job.csv`, which is flat across pT "
                         "at about 2200 ns; the helix column is about 1380 "
                         "and ACTS's own is 2550 to 3520")
    a = ap.parse_args()

    d = pl.read_csv(a.csv, null_values=["NA"])
    bad = d.filter(pl.col("stage_s").is_null())
    if bad.height:
        print(f"{bad.height} runs failed and are dropped:")
        print(bad.select(["bin", "repeat", "arm"]))
        print()
        d = d.filter(pl.col("stage_s").is_not_null())

    if a.floor > 0.0:
        bad_blocks = (d.filter(pl.col("trackfinding_s") < a.floor)
                       .select(["bin", "repeat"]).unique())
        if bad_blocks.height:
            keep = d.join(bad_blocks, on=["bin", "repeat"], how="anti")
            print(f"{bad_blocks.height} of "
                  f"{d.select(['bin', 'repeat']).unique().height} blocks hold "
                  f"a track-finding time below {a.floor:.1f} s and are dropped "
                  f"whole. That is the clock step, not the arm.")
            print()
            print(bad_blocks.sort(["bin", "repeat"]))
            print()
            d = keep

    if a.rel_floor > 0.0:
        d, dropped, total = drop_relative(d, a.rel_floor)
        print(f"{dropped} of {total} blocks hold a track-finding time below "
              f"{a.rel_floor:g} of its own (bin, arm) median and are dropped "
              f"whole. That is the clock step, not the arm.")
        print()

    n = d.group_by("arm").len().sort("arm")
    print("### What was run")
    print()
    print("| arm | runs |")
    print("| --- | ---: |")
    for r in n.iter_rows(named=True):
        print(f"| {r['arm']} | {r['len']} |")
    print()

    # The shared stages, to show they carry no signal and are most of the time.
    print("### The shared stages, all arms together")
    print()
    print("| stage | median [s] | share of the reconstruction stage |")
    print("| --- | ---: | ---: |")
    stage = float(np.median(d["stage_s"].to_numpy()))
    for col, label in (("digi_s", "DigitizationAlgorithm"),
                       ("seeding_s", "GridTripletSeedingAlgorithm"),
                       ("trackfinding_s", "TrackFindingAlgorithm")):
        if col not in d.columns:
            continue
        v = float(np.median(d[col].to_numpy()))
        print(f"| {label} | {v:.2f} | {100 * v / stage:.1f} % |")
    print(f"| whole stage | {stage:.2f} | 100 % |")
    print()
    print("Median over every run of every arm. The per-algorithm lines sum to "
          "well under the stage: the difference is geometry building, material "
          "decoration and process start, which happen once and are identical "
          "in every arm.")
    print()

    for col, label in COLS:
        table(d, col, label, a.baseline)

    per_candidate(d, a.baseline)

    if a.logs:
        ceiling(d, a.logs, a.ns_transport)


if __name__ == "__main__":
    main()
