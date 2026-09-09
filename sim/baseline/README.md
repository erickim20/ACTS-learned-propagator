# Stock baseline

Stock ACTS CKF on the ODD, no learned component anywhere, on a
ttbar sample at pileup 200 produced with the ColliderML production chain.

Everything here runs inside the pinned ODD image. The configuration is kept in
this repository rather than only in the working directory because the number is
worth nothing without the sample, the pileup and the cuts beside it.

## What produced the number

| | |
| --- | --- |
| image | `ghcr.io/opendatadetector/sw@sha256:20e4df7b0d6befd70ce0113d852350b25cbd9aac659d1bc530036a2c85a68f89` |
| ACTS | 44.99.99-colliderml-arrow |
| ColliderML | `/data/ColliderML-Production`, commit `351ae166`, 2026-06-02 |
| geometry | the image's own `/opt/odd`, copied into the cache so the setup script does not clone a different ODD |
| hard scatter | MadGraph 3.5.9, `generate p p > t t~ [QCD]`, 14 TeV, NLO, showered by Pythia 8.313 |
| pileup | 200, Poisson sampled |
| simulation | Geant4 11.3.2 through ddsim, and Fatras as a second arm on the same events |
| digitization | `odd-full-geo-digi-config.json` |

The process definition is `p p > t t~ [QCD]` alone. ColliderML's own production
configuration adds `+1j` and `+2j` with FxFx merging. That changes the jet
multiplicity tail and costs a great deal more to generate; it moves the track
population very little, which is what this number is about.

## Cuts

Read out of `digi_and_reco.py`. They are not the cuts in
`/opt/odd/ci/full_chain_odd.py`, and the two differ.

| | |
| --- | --- |
| truth selection | rho < 24 mm, abs(z) < 1 m, abs(eta) < 3.0, pT > 0.999 GeV, at least 6 measurements of which at least 3 in the pixels, charged, secondaries kept |
| seeding | GridTriplet, r 33 to 200 mm, deltaR 1 to 300 mm, collision region ±250 mm, maxSeedsPerSpM 40, minPt 0.5 GeV, impactMax 3 mm |
| CKF | chi2 < 15 measurement, < 25 outlier, numMeasurementsCutOff 1, seed deduplication, stay on seed, two way |
| track selection | pT > 0.7 GeV, abs(eta) < 3.5, at least 6 measurements, at most 3 holes and outliers |
| ambiguity | greedy, at most 3 shared hits, at least 6 measurements |

## Running it

From the repository root, on the WSL side. `BASELINE0_ROOT` defaults to
`$HOME/baseline0` and holds the cache, the configs and the output.

    sim/baseline/setup.sh                       # seed ODD and the Geant4 datasets, once
    sim/baseline/run_stage.sh simulation/madgraph_init.py pu200/madgraph_init_config.yaml 0
    sim/baseline/run_stage.sh simulation/madgraph_gen.py  pu200/madgraph_generation_config.yaml 0
    sim/baseline/run_pu200.sh 8 2               # 8 runs of 4 events, 2 at a time

`madgraph_generation_config.yaml` splits its output into one file per run, which
is what makes the runs independent. Without the split every run merges pileup
onto the same first events and the sample is N copies of one thing.

Then the Fatras arm on the same merged events, and the numbers:

    sim/baseline/run_fatras.sh runs/1/merged_events.hepmc3 fatras/1 4
    sim/baseline/aggregate.sh aggregate.py ambi runs/1 runs/2
    sim/baseline/aggregate.sh baseline_numbers.py runs/1

## What the box will take

One ddsim at pileup 200 grows to about 3 GB, and WSL has 15. Two at a time is
comfortable, three is the edge. Five exhausted memory: the OOM killer took a
run and WSL stopped responding, so the run had to be restarted after
`wsl --shutdown`. The 16 cores do not enter into it.

Per event, measured: Geant4 about 430 s, digitization and reconstruction about
20 s, and 292 MB of `edm4hep.root`. Fatras is about 9 s per event.

## Two traps

`pythia_gen.py` looks for `events_signal.hepmc3` before `events.hepmc` when it
picks up a hard scatter. A run directory that has been used before will silently
merge pileup onto the old signal. Run directories must be clean.

`digi_and_reco.py` falls back to the ODD's own `odd-digi-smearing-config.json`
when `digi_config` is not set. That is a different and much smaller
digitization, so leaving the line out produces a number that is not the
production one.
