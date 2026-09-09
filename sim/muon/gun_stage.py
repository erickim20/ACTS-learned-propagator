"""The muon gun stage, writing the file the merge already reads.

This exists because the production gun does not run against the pinned image.
`particlegun_gen.py:222` passes `perEvent=False` to `HepMC3Writer`, and
`HepMC3Writer::Config` in ACTS 44.99.99-colliderml-arrow has no such field: its
fields are compression, inputEvent, maxEventsPending, outputPath and
writeEventsInOrder. The constructor raises before a single event is generated.
The production repository is mounted read-only and is not this project's to
patch, so the stage is reproduced here against the API the image actually
exposes.

Everything else follows particlegun_gen.py exactly, so that the file this
writes is the file the rest of the production chain expects:

  one vertex per event, fixed multiplicity 1        (particlegun_gen.py:194)
  the exact origin, zero-width Gaussian vertex      (particlegun_gen.py:195-198)
  charge randomised, eta and phi sampled per muon   (particlegun_gen.py:199-209)
  written to <output>/<subdir>/events.hepmc3        (particlegun_gen.py:218)

One departure, and it is the reason S2 is affordable. `--muons-per-event` sets
`numParticles` on the generator, so one vertex can carry several muons with
independent eta, phi, pT and charge. Geant4 at pileup 200 costs 512 s per event
and is paid on the pileup, not on the signal, so N muons in one event cost the
same as one and multiply the signal statistics by N. S1 leaves it at 1, where an
event costs nothing and there is nothing to amortise.

`events.hepmc3` is the name both merge drivers list as a hard-scatter candidate
(`pythia_gen.py:271-276`, `acts_merge.py:111-115`), which is what lets S2 put
this sample inside pileup without any change to the production chain.

Charge randomisation means `--particle mu-` produces both signs, as it does
upstream.
"""
import argparse
import math
from pathlib import Path

import yaml

import acts
import acts.examples
from acts.examples.hepmc3 import HepMC3Writer

u = acts.UnitConstants

# The names particlegun_gen.py accepts, kept to the species this experiment
# uses. A PDG code as a string works as well.
NAMES = {
    "mu-": acts.PdgParticle.eMuon,
    "mu+": acts.PdgParticle.eAntiMuon,
    "muon": acts.PdgParticle.eMuon,
    "pi+": acts.PdgParticle.ePionPlus,
    "pi-": acts.PdgParticle.ePionMinus,
    "e-": acts.PdgParticle.eElectron,
    "e+": acts.PdgParticle.ePositron,
}


def to_pdg(name):
    if name in NAMES:
        return NAMES[name]
    if name.lower() in NAMES:
        return NAMES[name.lower()]
    return acts.PdgParticle(int(name))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--output", "-o", type=Path, required=True)
    p.add_argument("--output-subdir", default=None)
    p.add_argument("--events", "-n", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--particle", default=None)
    p.add_argument("--energy-min", type=float, default=None)
    p.add_argument("--energy-max", type=float, default=None)
    p.add_argument("--transverse", action="store_true", default=None)
    p.add_argument("--log-uniform", action="store_true", default=None)
    p.add_argument("--eta-min", type=float, default=None)
    p.add_argument("--eta-max", type=float, default=None)
    p.add_argument("--phi-min", type=float, default=None)
    p.add_argument("--phi-max", type=float, default=None)
    p.add_argument("--muons-per-event", type=int, default=None,
                   help="particles from the one vertex; 1 for S1")
    # Accepted and ignored, so that the same argument list works for every
    # stage the driver runs.
    p.add_argument("--performance-metrics", action="store_true", default=None)
    args = p.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        for key, value in cfg.items():
            key = key.replace("-", "_")
            if getattr(args, key, None) is None:
                setattr(args, key, value)
    return args


def main():
    args = parse_args()

    out_dir = args.output
    if args.output_subdir:
        out_dir = out_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "events.hepmc3"

    pdg = to_pdg(args.particle or "mu-")
    p_min = (args.energy_min if args.energy_min is not None else 1.0) * u.GeV
    p_max = (args.energy_max if args.energy_max is not None else 100.0) * u.GeV
    eta_min = args.eta_min if args.eta_min is not None else -2.5
    eta_max = args.eta_max if args.eta_max is not None else 2.5
    phi_min = args.phi_min if args.phi_min is not None else 0.0
    phi_max = args.phi_max if args.phi_max is not None else 2.0 * math.pi
    events = args.events if args.events is not None else 10
    seed = args.seed if args.seed is not None else 42
    per_event = args.muons_per_event if args.muons_per_event is not None else 1

    print(f"[gun] {events} events x {per_event} muons, "
          f"{args.particle or 'mu-'} ({pdg}), "
          f"{'pT' if args.transverse else 'p'} "
          f"{p_min / u.GeV} to {p_max / u.GeV} GeV, "
          f"{'log-uniform' if args.log_uniform else 'uniform'}, "
          f"eta {eta_min} to {eta_max}, seed {seed}", flush=True)
    print(f"[gun] writing {out_path}", flush=True)

    s = acts.examples.Sequencer(numThreads=1, events=events,
                                logLevel=acts.logging.INFO)
    rnd = acts.examples.RandomNumbers(seed=seed)

    evGen = acts.examples.EventGenerator(
        level=acts.logging.INFO,
        generators=[
            acts.examples.EventGenerator.Generator(
                multiplicity=acts.examples.FixedMultiplicityGenerator(n=1),
                vertex=acts.examples.GaussianVertexGenerator(
                    mean=acts.Vector4(0, 0, 0, 0),
                    stddev=acts.Vector4(0, 0, 0, 0),
                ),
                particles=acts.examples.ParametricParticleGenerator(
                    p=(p_min, p_max),
                    pLogUniform=bool(args.log_uniform),
                    pTransverse=bool(args.transverse),
                    eta=(eta_min, eta_max),
                    phi=(phi_min, phi_max),
                    etaUniform=True,
                    numParticles=per_event,
                    pdg=pdg,
                    randomizeCharge=True,
                ),
            )
        ],
        outputEvent="particle_gun_event",
        randomNumbers=rnd,
    )
    s.addReader(evGen)

    s.addWriter(
        HepMC3Writer(
            acts.logging.INFO,
            inputEvent=evGen.config.outputEvent,
            outputPath=out_path,
        )
    )

    s.run()
    print(f"[gun] done: {out_path}", flush=True)


if __name__ == "__main__":
    main()
