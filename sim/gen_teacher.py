"""Teacher dataset generator: surface-to-surface transport pairs under the ODD
field map. Runs INSIDE the ColliderML ODD image.

The plan had this blocked behind "recover output states from obj step dumps",
because this build exposes no ROOT propagation writer and
`Propagator.propagate` is not bound to Python. Neither is needed: Fatras
records a SimHit at every sensitive-surface crossing, and
`CsvSimHitWriter` writes position and momentum at that crossing
(`tx,ty,tz` / `tpx,tpy,tpz`), plus the surface (`geometry_id`) and the
material kick applied there (`deltapx..deltae`, zero when interactions are
off). Chaining a particle's hits in time therefore gives exactly the pairs

    (state on surface k, surface k, surface k+1) -> state on surface k+1

with the propagation done by the real ACTS propagator under whichever field
Fatras is given.

Two configurations matter and both are cheap to produce:

  --no-material : pure field transport. The residual against a helix is then
                  the field-map content alone -- what g_theta is meant to
                  absorb.
  --material    : energy loss + multiple scattering on.

The delta* columns do not isolate the material kick per surface: this build
writes
deltapx..deltae as identically zero, in the material run as much as in the
no-material one. The interactions are on (momentum along a track falls 0.10%
median with material, exactly 0 without); it is the writer that does not fill
them. To separate material from field, run `--field const --material`
instead: the helix is exact under a constant field, so the residual there is
material and nothing else.

Sampling is log-uniform in pT and stratified in |eta| because the target
lives at high |eta| (uniform-pT sampling left the earlier heatmap with n<=5
in exactly the cells that decide the fallback boundary).

Usage:
    python3 /data/gen_teacher.py --events 200 --tracks 100 --field map
    python3 /data/gen_teacher.py --events 200 --tracks 100 --field map --material
"""
import argparse
import math
import os
from pathlib import Path

import acts
import acts.examples
from acts.examples.odd import getOpenDataDetector, getOpenDataDetectorDirectory
from acts.examples.simulation import (
    addFatras, EtaConfig, MomentumConfig, ParticleConfig, PhiConfig,
    addParticleGun,
)

u = acts.UnitConstants
LOG = acts.logging.WARNING

# The species the trackable population of a physics event is actually made of,
# with the fractions measured on higgs_portal pileup 10, charged, >= 6 tracker
# hits (13,474 particles over 10 events).
PDG = {
    "pion": (acts.PdgParticle.ePionPlus, 0.542),
    "electron": (acts.PdgParticle.eElectron, 0.300),
    "proton": (acts.PdgParticle.eProton, 0.076),
    "kaon": (acts.PdgParticle.eKaonPlus, 0.047),
    "muon": (acts.PdgParticle.eMuon, 0.034),
}


def build_field(kind, mapfile):
    if kind == "const":
        return acts.ConstantBField(acts.Vector3(0, 0, 2.0 * u.T))
    if kind == "map":
        if not os.path.exists(mapfile):
            raise SystemExit(
                f"{mapfile} missing. Convert the ODD csv first:\n"
                "  tail -n +2 /opt/odd/data/odd-bfield.csv | tr ',' ' ' > "
                f"{mapfile}")
        return acts.MagneticFieldMapXyz(mapfile)
    return None          # 'solenoid' -> caller substitutes detector.field


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--events", type=int, default=200)
    p.add_argument("--tracks", type=int, default=100,
                   help="particles per event")
    p.add_argument("--field", default="map",
                   choices=["map", "const", "solenoid"])
    p.add_argument("--material", action="store_true",
                   help="enable energy loss + multiple scattering")
    p.add_argument("--pt-min", type=float, default=0.5)
    p.add_argument("--pt-max", type=float, default=20.0)
    p.add_argument("--eta-max", type=float, default=2.7)
    # The defaults above describe the first teacher, and that teacher does
    # not transfer. Measured
    # against the trackable population of a higgs_portal event at pileup 10,
    # it is wrong on three axes at once:
    #
    #   pT      median 3.06 GeV against 0.276; 72% of the target sits inside
    #           the teacher's bottom 0.4%
    #   |eta|   capped at 2.7, but the target's p99 is 3.72 -- the teacher
    #           does not merely undersample that region, it excludes it
    #   species pure muon, against 54% pion / 30% electron / 7.6% proton /
    #           4.7% kaon / 3.4% muon. Pions interact hadronically and
    #           electrons bremsstrahlung; neither is a muon with a different
    #           mass, and no amount of reweighting reaches them
    #
    # Reweighting cannot fix any of the three: you cannot reweight your way to
    # statistics you never generated, and 0.4% of 110,530 jumps is 442.
    p.add_argument("--pdg", default="muon",
                   choices=sorted(PDG.keys()),
                   help="species to fire. The physics sample is pion-dominated")
    p.add_argument("--tag", default=None,
                   help="override the output directory name")
    p.add_argument("--uniform-pt", action="store_true",
                   help="sample pT uniformly instead of log-uniformly; only "
                        "for reproducing the pre-2026-07-30 sets")
    p.add_argument("--mapfile", default="/data/odd-bfield.txt")
    p.add_argument("--out", default="/data/teacher")
    args = p.parse_args()

    # log-uniform sets get their own tag so they cannot be silently mixed
    # with the uniform-pT sets generated before 2026-07-30
    tag = args.tag or (
        f"{args.field}_{'mat' if args.material else 'nomat'}"
        + ("" if args.uniform_pt else "_logpt")
        + ("" if args.pdg == "muon" else f"_{args.pdg}"))
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)

    geoDir = getOpenDataDetectorDirectory()
    deco = acts.IMaterialDecorator.fromFile(
        geoDir / "data/odd-material-maps.root", level=LOG)
    detector = getOpenDataDetector(odd_dir=geoDir, materialDecorator=deco)
    trackingGeometry = detector.trackingGeometry()
    field = build_field(args.field, args.mapfile) or detector.field

    s = acts.examples.Sequencer(events=args.events, numThreads=1,
                                logLevel=LOG, trackFpes=False)
    rnd = acts.examples.RandomNumbers(seed=4242)

    # Log-uniform in pT. `MomentumConfig` takes `logUniform`, so the gun does
    # it directly rather than deferring the imbalance to analysis-stage
    # weighting. It matters a lot.
    # The residual scales as 1/pT, so uniform sampling over 0.5-20 GeV spends
    # half the sample above 10 GeV where there is almost nothing to learn and
    # leaves 2.6% below 1 GeV where the target lives. Log-uniform puts ~19%
    # in each octave. Weighting cannot fix this: you cannot reweight your way
    # to tail statistics you never generated.
    addParticleGun(
        s,
        ParticleConfig(num=args.tracks, pdg=PDG[args.pdg][0],
                       randomizeCharge=True),
        MomentumConfig(args.pt_min * u.GeV, args.pt_max * u.GeV,
                       transverse=True, logUniform=not args.uniform_pt),
        EtaConfig(-args.eta_max, args.eta_max, uniform=True),
        PhiConfig(0.0, 2 * math.pi),
        vtxGen=acts.examples.GaussianVertexGenerator(
            stddev=acts.Vector4(0, 0, 0, 0), mean=acts.Vector4(0, 0, 0, 0)),
        multiplicity=1,
        rnd=rnd,
    )
    addFatras(s, trackingGeometry, field, rnd=rnd,
              enableInteractions=args.material)

    # Both collections are needed: hits give the states, particles give the
    # charge and the generated kinematics for binning.
    s.addWriter(acts.examples.CsvSimHitWriter(
        level=LOG, inputSimHits="simhits", outputDir=str(out),
        outputStem="hits"))
    s.addWriter(acts.examples.CsvParticleWriter(
        level=LOG, inputParticles="particles_generated",
        outputDir=str(out), outputStem="particles"))
    s.run()

    n = len(list(out.glob("*hits.csv")))
    print(f"\nwrote {n} event files to {out}")
    print(f"  field={args.field}  material={args.material}")
    print(f"  {args.events} events x {args.tracks} {args.pdg}s, "
          f"pT {args.pt_min}-{args.pt_max} GeV, |eta| < {args.eta_max}")
    print("\nnext: copy out and run make_teacher_pairs.py on the host")


if __name__ == "__main__":
    main()
