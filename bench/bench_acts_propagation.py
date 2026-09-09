"""Real per-call cost of an ACTS propagation, to replace the numpy estimate.

Runs INSIDE the ColliderML ODD image. The host-half cost model
(profile_stepper.py) reported ns per jump from a numpy re-implementation, and
its factorisation was

    cost per jump = (steps per jump) x (ns per step)

with 6.2 / 4.7 steps per jump (map / const 2 T) and, implicitly, 830 / 284 ns
per step. Both factors are measurable here, in the same shape, so which one
the model got wrong is visible, and the ns/step from real C++ ACTS is the
number the learned path actually has to beat. numpy penalises the stepper's
sequential step control far more than it penalises three matmuls, so the
2.4x from the model is expected to shrink; this measures by how much.

Two passes, because the step writer is not free:

  timing  : PropagationAlgorithm alone, no writers, seconds/event from the
            Sequencer's own timing.tsv -> ns per track propagation
  counting: same configuration, few events, with ObjPropagationStepsWriter
            -> steps per track and sensitive crossings ("jumps") per track

ns/step = (ns per track) / (steps per track), and ns/jump likewise.

IMPORTANT: run natively. Under qemu the numbers are meaningless -- check with
    docker run --rm --entrypoint /bin/bash <img> -c 'uname -m'
and confirm no qemu handler in /proc/sys/fs/binfmt_misc.

Usage (inside the container, env sourced):
    python3 /data/bench_acts_propagation.py --field const --events 20 --tracks 200
    python3 /data/bench_acts_propagation.py --field map   --events 20 --tracks 200
"""
import argparse
import csv
import glob
import math
import os
from pathlib import Path

import acts
import acts.examples
from acts.examples.odd import getOpenDataDetector, getOpenDataDetectorDirectory
from acts.examples.simulation import (
    addParticleGun, EtaConfig, MomentumConfig, ParticleConfig, PhiConfig,
)

u = acts.UnitConstants
LOG = acts.logging.WARNING


def build_field(kind, mapfile):
    if kind == "const":
        return acts.ConstantBField(acts.Vector3(0, 0, 2.0 * u.T))
    if kind == "map":
        if not os.path.exists(mapfile):
            raise SystemExit(
                f"{mapfile} missing. Convert the ODD csv first:\n"
                "  tail -n +2 /opt/odd/data/odd-bfield.csv | tr ',' ' ' > "
                f"{mapfile}\n(only ODD v5.0.0 ships the map; v4 and v6 do not)")
        return acts.MagneticFieldMapXyz(mapfile)
    if kind == "solenoid":
        return None          # caller substitutes detector.field
    raise SystemExit(f"unknown field {kind}")


def run(args, field, trackingGeometry, outdir, write_steps):
    nav = acts.Navigator(trackingGeometry=trackingGeometry, level=LOG)
    # --stepper straight shares the navigator but does no field integration, so
    # the gap between the two brackets the stepper's share from below. Not
    # exact: straight tracks visit a different set of surfaces, so step counts
    # differ and the gap mixes stepper cost with a changed traversal.
    if getattr(args, "stepper", "eigen") == "straight":
        stepper = acts.StraightLineStepper()
    else:
        stepper = acts.EigenStepper(field)
    propagator = acts.examples.ConcretePropagator(acts.Propagator(stepper, nav))

    s = acts.examples.Sequencer(
        events=args.events, numThreads=1, logLevel=LOG, trackFpes=False,
        outputDir=str(outdir), outputTimingFile="timing.tsv",
    )
    rnd = acts.examples.RandomNumbers(seed=42)
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
    s.addAlgorithm(acts.examples.ParticleTrackParamExtractor(
        level=LOG, inputParticles="particles_generated",
        outputTrackParameters="params"))
    s.addAlgorithm(acts.examples.PropagationAlgorithm(
        level=LOG, propagatorImpl=propagator, sterileLogger=False,
        inputTrackParameters="params", outputSummaryCollection="summary",
        energyLoss=args.material, multipleScattering=args.material,
        covarianceTransport=args.covariance))
    if write_steps:
        s.addWriter(acts.examples.ObjPropagationStepsWriter(
            level=LOG, collection="summary", outputDir=str(outdir),
            outputPrecision=6))
    s.run()


def read_prop_time(path):
    """Sequencer writes .tsv with COMMA separators. Returns s/event."""
    with open(path) as f:
        for r in csv.DictReader(f):
            ident = r.get("identifier") or r.get("Identifier") or ""
            if "Propagation" in ident:
                return float(r.get("time_perevent_s") or r.get("Time/Event") or 0)
    return 0.0


def count_steps(outdir, n_tracks_total):
    """Steps and sensitive-surface crossings per track, from the obj dumps."""
    steps = 0
    for path in glob.glob(str(outdir / "*.obj")):
        with open(path) as f:
            steps += sum(1 for line in f if line.startswith("v "))
    return steps / max(n_tracks_total, 1)


def emulation_warning():
    """Timings under qemu are meaningless; the step COUNT is still valid."""
    handlers = glob.glob("/proc/sys/fs/binfmt_misc/*qemu*")
    machine = os.uname().machine
    if handlers or machine not in ("x86_64", "amd64"):
        print("\n" + "!" * 68)
        print("!! EMULATED HOST -- ns figures below are NOT usable.")
        print(f"!! machine={machine} qemu handlers={[Path(h).name for h in handlers]}")
        print("!! steps/track is machine-independent and still valid.")
        print("!" * 68)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--field", default="const",
                   choices=["const", "map", "solenoid"])
    p.add_argument("--events", type=int, default=20)
    p.add_argument("--tracks", type=int, default=200)
    p.add_argument("--count-events", type=int, default=3,
                   help="events for the step-counting pass (writer is slow)")
    p.add_argument("--stepper", default="eigen", choices=["eigen", "straight"],
                   help="straight = StraightLineStepper: same navigator, no "
                        "field integration (round 3 item D)")
    p.add_argument("--material", action="store_true",
                   help="enable energy loss + multiple scattering; the "
                        "default (off) is cheaper than real CKF propagation, "
                        "where material is ~3%% of the track-following core")
    p.add_argument("--covariance", action="store_true",
                   help="enable covariance transport (off by default)")
    p.add_argument("--mapfile", default="/data/odd-bfield.txt")
    p.add_argument("--out", default="/data/acts_prop_bench")
    args = p.parse_args()

    geoDir = getOpenDataDetectorDirectory()
    deco = acts.IMaterialDecorator.fromFile(
        geoDir / "data/odd-material-maps.root", level=LOG)
    detector = getOpenDataDetector(odd_dir=geoDir, materialDecorator=deco)
    trackingGeometry = detector.trackingGeometry()
    field = build_field(args.field, args.mapfile) or detector.field

    out = Path(args.out) / args.field
    (out / "time").mkdir(parents=True, exist_ok=True)
    (out / "count").mkdir(parents=True, exist_ok=True)

    run(args, field, trackingGeometry, out / "time", write_steps=False)
    per_event_s = read_prop_time(out / "time" / "timing.tsv")

    cnt = argparse.Namespace(**vars(args))
    cnt.events = args.count_events
    run(cnt, field, trackingGeometry, out / "count", write_steps=True)
    steps_per_track = count_steps(out / "count",
                                  args.count_events * args.tracks)

    ns_per_track = per_event_s / args.tracks * 1e9
    ns_per_step = ns_per_track / steps_per_track if steps_per_track else float("nan")

    emulation_warning()
    print(f"\n=== ACTS propagation, field={args.field} ===")
    print(f"  {args.events} events x {args.tracks} tracks, 1 thread")
    print(f"  propagation algorithm : {per_event_s*1e3:9.2f} ms/event")
    print(f"  per track (full traverse) : {ns_per_track:9.0f} ns")
    print(f"  steps per track           : {steps_per_track:9.1f}")
    print(f"  ns per RKN step           : {ns_per_step:9.1f}")
    print("\n  compare with the numpy model (profile_stepper.py):")
    print("    const 2 T : 4.7 steps/jump, 1335 ns/jump  -> ~284 ns/step")
    print("    field map : 6.2 steps/jump, 5147 ns/jump  -> ~830 ns/step")
    print("  and with helix + g_theta at 558 ns/jump (numpy, relu, untrained).")
    print("  A jump is a fraction of a full traverse: divide ns/track by the")
    print("  number of sensitive crossings to compare per-jump numbers.")


if __name__ == "__main__":
    main()
