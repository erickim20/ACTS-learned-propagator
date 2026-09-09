"""Round 2 item B: measure sensitive crossings per track directly.

Every per-jump number so far divided ns/track by 12.47 crossings, which was
back-derived from the numpy model's 4.7 steps/jump -- i.e. from the very model
round 1 set out to test. This counts crossings instead.

A "sensitive crossing" is a hit on a sensitive tracker surface, which is
exactly what Fatras records. The particle gun is configured identically to
bench_acts_propagation.py (same seed, pdg, momentum, eta, phi, vertex) so the
track population is the same one the timing ran over.

Two variants:
  interactions off -> matches the timing bench (energyLoss/MS both False)
  interactions on  -> what real CKF propagation sees
"""
import argparse
import collections
import csv
import glob
import math
from pathlib import Path

import acts
import acts.examples
from acts.examples.odd import getOpenDataDetector, getOpenDataDetectorDirectory
from acts.examples.simulation import (
    addParticleGun, addFatras, EtaConfig, MomentumConfig, ParticleConfig,
    PhiConfig,
)

u = acts.UnitConstants
LOG = acts.logging.WARNING

# Tracker volumes only.
TRACKER_VOLUMES = {16, 17, 18, 23, 24, 25, 28, 29, 30}


def volume_of(geometry_id):
    """ACTS packs volume in the high bits of geometry_id."""
    return (int(geometry_id) >> 56) & 0xFF


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--events", type=int, default=20)
    p.add_argument("--tracks", type=int, default=200)
    p.add_argument("--interactions", action="store_true")
    p.add_argument("--out", default="/data/crossings")
    args = p.parse_args()

    geoDir = getOpenDataDetectorDirectory()
    deco = acts.IMaterialDecorator.fromFile(
        geoDir / "data/odd-material-maps.root", level=LOG)
    detector = getOpenDataDetector(odd_dir=geoDir, materialDecorator=deco)
    trackingGeometry = detector.trackingGeometry()
    field = acts.ConstantBField(acts.Vector3(0, 0, 2.0 * u.T))

    out = Path(args.out) / ("on" if args.interactions else "off")
    out.mkdir(parents=True, exist_ok=True)

    s = acts.examples.Sequencer(events=args.events, numThreads=1, logLevel=LOG,
                                trackFpes=False)
    rnd = acts.examples.RandomNumbers(seed=42)
    # identical to bench_acts_propagation.py
    addParticleGun(
        s,
        ParticleConfig(num=args.tracks, pdg=acts.PdgParticle.eMuon,
                       randomizeCharge=True),
        MomentumConfig(0.5 * u.GeV, 20.0 * u.GeV, transverse=True),
        EtaConfig(-2.7, 2.7, uniform=True),
        PhiConfig(0.0, 2 * math.pi),
        vtxGen=acts.examples.GaussianVertexGenerator(
            stddev=acts.Vector4(0, 0, 0, 0), mean=acts.Vector4(0, 0, 0, 0)),
        multiplicity=1, rnd=rnd,
    )
    addFatras(s, trackingGeometry, field, rnd=rnd,
              enableInteractions=args.interactions,
              inputParticles="particles_generated",
              outputDirCsv=str(out))
    s.run()

    # the writer splits the barcode into components; the tuple is the identity
    ID_COLS = ("particle_id_pv", "particle_id_sv", "particle_id_part",
               "particle_id_gen", "particle_id_subpart")

    per_particle = collections.Counter()
    per_particle_tracker = collections.Counter()
    for path in glob.glob(str(out / "*hits.csv")):
        with open(path) as f:
            for row in csv.DictReader(f):
                pid = (path,) + tuple(row[c] for c in ID_COLS)
                per_particle[pid] += 1
                gid = row.get("geometry_id") or row.get("geoid") or 0
                if volume_of(gid) in TRACKER_VOLUMES:
                    per_particle_tracker[pid] += 1

    n = len(per_particle)
    if not n:
        raise SystemExit("no hits parsed -- check writer output in " + str(out))
    tot = sum(per_particle.values())
    tot_trk = sum(per_particle_tracker.values())
    counts = sorted(per_particle.values())
    vols = collections.Counter()
    for path in glob.glob(str(out / "*hits.csv")):
        with open(path) as f:
            for row in csv.DictReader(f):
                vols[volume_of(row.get("geometry_id") or 0)] += 1

    print(f"\n=== sensitive crossings per track "
          f"(interactions={'on' if args.interactions else 'off'}) ===")
    print(f"  particles with >=1 hit : {n}")
    print(f"  all sensitive hits     : {tot}  -> {tot/n:.2f} per track")
    print(f"  tracker volumes only   : {tot_trk} -> {tot_trk/n:.2f} per track")
    print(f"  median / p25 / p75     : {counts[n//2]} / "
          f"{counts[n//4]} / {counts[3*n//4]}")
    print(f"  min / max              : {counts[0]} / {counts[-1]}")
    print(f"  hits by volume         : {dict(sorted(vols.items()))}")
    print(f"\n  compare: 12.47 crossings/track was ASSUMED (58.6 steps / 4.7)")


if __name__ == "__main__":
    main()
