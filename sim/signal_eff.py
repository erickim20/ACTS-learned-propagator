#!/usr/bin/env python3
"""Tracking efficiency for the gun's muons alone.

The ACTS performance writer bins efficiency over every selected truth particle.
On S1 that is the gun's muons and nothing else. On S2 it is the gun's muons plus
whatever the 200 pileup events put inside the truth selection's own acceptance,
which is thousands of particles an event, so the ACTS number there answers a
different question from the one the experiment asks. This forms the signal-only
number and reports the all-particle one beside it.

The gun's muons are identified by their barcode, not by their kinematics. The
merge reads the hard scatter first (`pythia_gen.py:334`, `Fixed(signal, 1)`) and
the gun is the hard scatter, so every gun muon carries vertex_primary 1. Pileup
muons exist and carry other primaries; measured on bin 1, vertex_primary 1 held
exactly 20 muons an event and the next most populated primary held 2.

Inputs, both written by the reconstruction:

  particles.root          the truth selection's output, so the denominator.
                          fatras_reco.py writes `particles_digitized_selected`
                          for this reason; digi_and_reco.py's own particle file
                          is `particles_simulated` and is not a denominator.
  tracksummary_<stage>    one entry an event, vector branches over its tracks,
                          carrying the majority particle's barcode and the
                          match classification.

A particle counts as found when some track has it as the majority particle and
that track is classified Matched. Duplicate and Fake do not count, following
`TrackMatchClassification` in ActsExamples/EventData/TruthMatching.hpp:30-38.

    sim/run_root.sh sim/signal_eff.py runs/mu_s2_b1_eval runs/mu_s2_b2_eval
    sim/run_root.sh sim/signal_eff.py --stage ckf runs/mu_s1_b1_eval
"""

from __future__ import annotations

import argparse
import sys

import ROOT

ROOT.gROOT.SetBatch(True)

MATCHED = 1

# The gun is the hard scatter and the merge reads it first, so its vertex is
# primary 1. On S1 there is no merge and the gun is the only source at all.
SIGNAL_VERTEX_PRIMARY = 1
MUON_PDG = 13


def barcode(tree, i, prefix=""):
    return (
        tree.__getattr__(f"{prefix}vertex_primary")[i],
        tree.__getattr__(f"{prefix}vertex_secondary")[i],
        tree.__getattr__(f"{prefix}particle")[i],
        tree.__getattr__(f"{prefix}generation")[i],
        tree.__getattr__(f"{prefix}sub_particle")[i],
    )


def read_run(run_dir, stage):
    """Return the counts one run directory implies, per event summed."""
    fp = ROOT.TFile.Open(f"{run_dir}/particles.root")
    if not fp or fp.IsZombie():
        return None
    particles = fp.Get("particles")

    fs = ROOT.TFile.Open(f"{run_dir}/tracksummary_{stage}.root")
    if not fs or fs.IsZombie():
        return None
    summary = fs.Get("tracksummary")

    # Matched majority barcodes, per event. The summary carries one entry an
    # event, so the two trees are walked in step rather than joined.
    matched_by_event = {}
    holes_signal = []
    n_tracks = 0
    n_class = {0: 0, 1: 0, 2: 0, 3: 0}
    for ev in summary:
        found = set()
        for i in range(len(ev.trackClassification)):
            n_tracks += 1
            c = int(ev.trackClassification[i])
            n_class[c] = n_class.get(c, 0) + 1
            if c != MATCHED:
                continue
            found.add((
                ev.majorityParticleId_vertex_primary[i],
                ev.majorityParticleId_vertex_secondary[i],
                ev.majorityParticleId_particle[i],
                ev.majorityParticleId_generation[i],
                ev.majorityParticleId_sub_particle[i],
            ))
        matched_by_event[int(ev.event_nr)] = (found, ev)

    sig_total = sig_found = 0
    all_total = all_found = 0
    n_events = 0
    for ev in particles:
        n_events += 1
        entry = matched_by_event.get(int(ev.event_id))
        found = entry[0] if entry else set()
        summ = entry[1] if entry else None

        for i in range(len(ev.particle_type)):
            bc = barcode(ev, i)
            is_found = bc in found
            all_total += 1
            all_found += is_found
            if (ev.vertex_primary[i] == SIGNAL_VERTEX_PRIMARY
                    and abs(ev.particle_type[i]) == MUON_PDG):
                sig_total += 1
                sig_found += is_found

        if summ is not None:
            for i in range(len(summ.trackClassification)):
                if (int(summ.trackClassification[i]) == MATCHED
                        and summ.majorityParticleId_vertex_primary[i]
                        == SIGNAL_VERTEX_PRIMARY):
                    holes_signal.append(float(summ.nHoles[i]))

    fp.Close()
    fs.Close()
    return {
        "events": n_events,
        "signal_total": sig_total,
        "signal_found": sig_found,
        "all_total": all_total,
        "all_found": all_found,
        "tracks": n_tracks,
        "class": n_class,
        "holes_signal": holes_signal,
    }


def acts_scalar(run_dir, stage, name):
    f = ROOT.TFile.Open(f"{run_dir}/performance_finding_{stage}.root")
    if not f or f.IsZombie():
        return None
    v = f.Get(name)
    out = float(v[0]) if v else None
    f.Close()
    return out


def ratio(num, den):
    return 100.0 * num / den if den else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", nargs="+", help="run directories, e.g. runs/mu_s2_b1_eval")
    p.add_argument("--stage", default="ambi", choices=["ambi", "ckf"])
    p.add_argument("--root", default="/out", help="mount point of the output tree")
    args = p.parse_args()

    rows = []
    for r in args.runs:
        path = r if r.startswith("/") else f"{args.root}/{r}"
        got = read_run(path, args.stage)
        if got is None:
            print(f"!! cannot read {path}", file=sys.stderr)
            continue
        got["name"] = r.rstrip("/").split("/")[-1]
        got["acts_eff"] = acts_scalar(path, args.stage, "eff_particles")
        rows.append(got)

    if not rows:
        return 1

    print(f"stage {args.stage}")
    print()
    head = (f"{'run':22s} {'events':>7s} {'signal mu':>10s} {'eff signal':>11s} "
            f"{'eff all':>9s} {'ACTS eff':>9s} {'holes/trk':>10s}")
    print(head)
    print("-" * len(head))
    for g in rows:
        holes = (sum(g["holes_signal"]) / len(g["holes_signal"])
                 if g["holes_signal"] else float("nan"))
        print(f"{g['name']:22s} {g['events']:7d} {g['signal_total']:10d} "
              f"{ratio(g['signal_found'], g['signal_total']):10.3f}% "
              f"{ratio(g['all_found'], g['all_total']):8.3f}% "
              f"{100 * g['acts_eff'] if g['acts_eff'] is not None else float('nan'):8.3f}% "
              f"{holes:10.4f}")

    print()
    print("eff signal  gun muons with a Matched track, over gun muons in the "
          "truth selection")
    print("eff all     the same over every particle in the truth selection")
    print("ACTS eff    eff_particles from the performance file, the writer's own")
    print("holes/trk   mean nHoles over the Matched tracks whose majority "
          "particle is a gun muon")
    print()
    for g in rows:
        c = g["class"]
        print(f"{g['name']:22s} tracks {g['tracks']:7d}  "
              f"unknown {c.get(0, 0):6d}  matched {c.get(1, 0):6d}  "
              f"duplicate {c.get(2, 0):6d}  fake {c.get(3, 0):6d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
