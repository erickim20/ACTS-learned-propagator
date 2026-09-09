"""The Fatras arm of the item 0 baseline. Runs INSIDE the ColliderML ODD image.

ColliderML's own chain is HepMC3 -> ddsim (Geant4) -> EDM4hep -> digi_and_reco.
This reads the same `merged_events.hepmc3` and replaces Geant4 with Fatras,
leaving everything downstream identical, so the difference between the two
arms is the simulation and nothing else.

Everything from the digitization onward is copied from
`ColliderML-Production/scripts/simulation/digi_and_reco.py` deliberately, cut
for cut. If that file changes, this one is wrong until it is changed with it.
The values are restated here rather than imported because that script's
reconstruction is not callable without its EDM4hep reader.

    sim/run_in_odd.sh sim/fatras_reco.py --input merged_events.hepmc3 \
        --output out --events 4 --digi-config odd-full-geo-digi-config.json
"""
import argparse
from pathlib import Path

import acts
import acts.examples
import acts.examples.hepmc3 as hepmc3
import acts.examples.root
from acts.examples import Sequencer
from acts.examples.odd import getOpenDataDetector, getOpenDataDetectorDirectory
from acts.examples.simulation import (
    ParticleSelectorConfig,
    addDigiParticleSelection,
    addDigitization,
    addFatras,
    addSimParticleSelection,
)
from acts.examples.reconstruction import (
    AmbiguityResolutionConfig,
    CkfConfig,
    SeedFinderConfigArg,
    SeedingAlgorithm,
    TrackSelectorConfig,
    addAmbiguityResolution,
    addCKFTracks,
    addSeeding,
    addTrackWriters,
)

u = acts.UnitConstants
LOG_LEVEL = acts.logging.FATAL


def main():
    p = argparse.ArgumentParser(description="Fatras + stock CKF on a HepMC3 sample")
    p.add_argument("--input", required=True, help="merged_events.hepmc3")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--events", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument(
        "--digi-config",
        default=None,
        help="digitization json. A bare name is looked up under the ODD's "
        "config/. Omitted means the ODD default, which is NOT what the "
        "ColliderML production configuration uses",
    )
    p.add_argument("--num-seeds-per-spm", type=int, default=40)
    p.add_argument(
        "--no-particles",
        dest="write_particles",
        action="store_false",
        help="skip particles.root, the selected truth particles. Written by "
        "default because the signal-only efficiency "
        "cannot be formed without it",
    )
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    geoDir = getOpenDataDetectorDirectory()
    oddMaterialMap = geoDir / "data/odd-material-maps.root"
    if args.digi_config:
        dc = Path(args.digi_config)
        oddDigiConfig = dc if dc.is_file() else (geoDir / f"config/{args.digi_config}")
    else:
        oddDigiConfig = geoDir / "config/odd-digi-smearing-config.json"
    oddSeedingSel = geoDir / "config/odd-seeding-config.json"

    detector = getOpenDataDetector(
        odd_dir=geoDir,
        materialDecorator=acts.IMaterialDecorator.fromFile(oddMaterialMap),
    )
    trackingGeometry = detector.trackingGeometry()
    field = detector.field
    rnd = acts.examples.RandomNumbers(seed=args.seed)

    # outputDir makes the Sequencer write timing.tsv, the per-algorithm profile.
    # digi_and_reco.py does not set it, so the Geant4 arm has stage wall times
    # but no breakdown; the algorithms and their configuration are the same on
    # both arms, so the shape of the profile is read off this one.
    s = Sequencer(
        numThreads=args.threads,
        events=args.events,
        logLevel=LOG_LEVEL,
        trackFpes=False,
        outputDir=str(args.output),
    )

    # Read the merged HepMC3 and turn it into SimParticles. This is the seam
    # where the Geant4 arm instead reads EDM4hep written by ddsim.
    s.addReader(
        hepmc3.HepMC3Reader(
            hepmc3.HepMC3Reader.Config(
                inputPath=str(args.input),
                outputEvent="hepmc3_event",
            ),
            LOG_LEVEL,
        )
    )
    s.addAlgorithm(
        hepmc3.HepMC3InputConverter(
            hepmc3.HepMC3InputConverter.Config(
                inputEvent="hepmc3_event",
                outputParticles="particles_generated",
                outputVertices="vertices_generated",
            ),
            LOG_LEVEL,
        )
    )

    addFatras(
        s,
        trackingGeometry,
        field,
        rnd=rnd,
        enableInteractions=True,
        inputParticles="particles_generated",
        outputParticles="particles_simulated",
        outputSimHits="simhits",
        logLevel=LOG_LEVEL,
    )
    s.addWhiteboardAlias("particles", "particles_simulated")

    # digi_and_reco.py:243
    addSimParticleSelection(
        s,
        ParticleSelectorConfig(
            rho=(0.0, 1080 * u.mm),
            absZ=(0.0, 3.03 * u.m),
            pt=(150 * u.MeV, None),
        ),
    )

    addDigitization(
        s,
        trackingGeometry,
        field,
        digiConfigFile=oddDigiConfig,
        outputDirRoot=args.output,
        outputDirCsv=None,
        rnd=rnd,
        logLevel=LOG_LEVEL,
    )

    # digi_and_reco.py:274. At least three pixel hits, volumes 16, 17, 18.
    def make_geoid(vol):
        geoid = acts.GeometryIdentifier()
        geoid.volume = vol
        return geoid

    measurementCounter = acts.examples.ParticleSelector.MeasurementCounter()
    measurementCounter.addCounter(
        [make_geoid(16), make_geoid(17), make_geoid(18)], 3, 2**31 - 1
    )

    addDigiParticleSelection(
        s,
        ParticleSelectorConfig(
            rho=(0.0, 24 * u.mm),
            absZ=(0.0, 1.0 * u.m),
            eta=(-3.0, 3.0),
            pt=(0.999 * u.GeV, None),
            measurements=(6, None),
            removeNeutral=True,
            removeSecondaries=False,
            nMeasurementsGroupMin=measurementCounter,
        ),
    )

    # digi_and_reco.py:557-569, under its output_particles_root flag, with one
    # change. That writer takes `particles`, which is the alias for
    # `particles_simulated` and holds everything the simulation produced: 27,000
    # a event at pileup 200. The efficiency denominator is the selection's
    # output, which `addDigiParticleSelection` publishes under
    # `particles_digitized_selected` and never under `particles`
    # (acts/examples/simulation.py:925-928). The efficiency needs the
    # identity of each particle in that denominator, because on a pileup sample
    # the denominator is not the gun's muons alone.
    if args.write_particles:
        s.addWriter(
            acts.examples.root.RootParticleWriter(
                config=acts.examples.root.RootParticleWriter.Config(
                    filePath=str(args.output / "particles.root"),
                    inputParticles="particles_digitized_selected",
                    referencePoint=acts.Vector3(0.0, 0.0, 0.0),
                    bField=field,
                    writeHelixParameters=True,
                ),
                level=LOG_LEVEL,
            )
        )

    addSeeding(
        s,
        trackingGeometry,
        field,
        seedingAlgorithm=SeedingAlgorithm.GridTriplet,
        particleHypothesis=acts.ParticleHypothesis.pion,
        seedFinderConfigArg=SeedFinderConfigArg(
            r=(33 * u.mm, 200 * u.mm),
            deltaR=(1 * u.mm, 300 * u.mm),
            collisionRegion=(-250 * u.mm, 250 * u.mm),
            z=(-2000 * u.mm, 2000 * u.mm),
            maxSeedsPerSpM=args.num_seeds_per_spm,
            sigmaScattering=5,
            radLengthPerSeed=0.1,
            minPt=0.5 * u.GeV,
            impactMax=3 * u.mm,
            zBinEdges=[-1600, -1000, -600, 0, 600, 1000, 1600],
        ),
        initialSigmas=[
            1 * u.mm,
            1 * u.mm,
            1 * u.degree,
            1 * u.degree,
            0.1 * u.e / u.GeV,
            1 * u.ns,
        ],
        initialSigmaQoverPt=0.1 * u.e / u.GeV,
        initialSigmaPtRel=0.1,
        initialVarInflation=[1e0] * 6,
        geoSelectionConfigFile=oddSeedingSel,
        outputDirRoot=None,
    )

    addCKFTracks(
        s,
        trackingGeometry,
        field,
        trackSelectorConfig=TrackSelectorConfig(
            pt=(0.7 * u.GeV, None),
            absEta=(None, 3.5),
            nMeasurementsMin=6,
            maxHolesAndOutliers=3,
        ),
        ckfConfig=CkfConfig(
            chi2CutOffMeasurement=15.0,
            chi2CutOffOutlier=25.0,
            numMeasurementsCutOff=1,
            seedDeduplication=True,
            stayOnSeed=True,
        ),
        twoWay=True,
        outputDirRoot=None,
        writeCovMat=False,
        writeTrackStates=False,
        writeTrackSummary=False,
        writePerformance=False,
    )
    addTrackWriters(
        s,
        name="ckf",
        tracks="tracks",
        outputDirCsv=None,
        outputDirRoot=args.output,
        writeSummary=True,
        writeStates=False,
        writeFitterPerformance=True,
        writeFinderPerformance=True,
        logLevel=LOG_LEVEL,
        writeCovMat=False,
    )

    addAmbiguityResolution(
        s,
        config=AmbiguityResolutionConfig(
            maximumSharedHits=3,
            maximumIterations=1000000,
            nMeasurementsMin=6,
        ),
        outputDirRoot=None,
        writeTrackSummary=False,
        writeTrackStates=False,
        writePerformance=False,
        writeCovMat=False,
    )
    addTrackWriters(
        s,
        name="ambi",
        tracks="tracks",
        outputDirCsv=None,
        outputDirRoot=args.output,
        writeSummary=True,
        writeStates=False,
        writeFitterPerformance=True,
        writeFinderPerformance=True,
        logLevel=LOG_LEVEL,
        writeCovMat=False,
    )

    s.run()


if __name__ == "__main__":
    main()
