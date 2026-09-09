# Profile

Where reconstruction time goes, at two levels: per algorithm from the
Sequencer's own timing, and per symbol from perf.

Everything here runs inside the pinned ODD image on item 0's own events, and
reconstruction is re-run from an existing `edm4hep.root` rather than simulated
again, so a profile costs about 80 s rather than half an hour.

## Running it

From the repository root, on the WSL side. The first argument is one of item 0's
run directories under `$BASELINE0_ROOT/out/runs`, the second is where to write.

    sim/profile/run_profile.sh 0 prof0                  # timing.tsv only
    sim/profile/run_profile.sh 0 perf0 --perf           # and a symbol profile

    CONFIG=sim/profile/digitization_noio.yaml \
      sim/profile/run_profile.sh 0 perfnoio --perf      # writers off

    python3 sim/profile/bucket_perf.py \
      $HOME/baseline0/out/runs/perfnoio [--audit]

`--audit` prints the largest symbols in each bucket, which is how the bucket
rules are checked rather than trusted.

## What each piece is

| | |
| --- | --- |
| `run_profile.sh` | the host side: mounts, the image, and perf's requirements |
| `incontainer_profile.sh` | the in-image side, and the perf invocation |
| `seq_timing.py` | patches the Sequencer so `digi_and_reco.py` writes `timing.tsv` |
| `digitization_noio.yaml` | item 0's configuration with every writer off |
| `bucket_perf.py` | buckets a perf report into stepper, navigator and the rest |

`digi_and_reco.py` is not copied. `seq_timing.py` replaces
`acts.examples.Sequencer` with a wrapper that fills in `outputDir` and
`outputTimingFile` and then runs the production script as it stands, so the
chain and its cuts stay ColliderML's. A copy would answer the same question
today and drift from it silently.

That the patch is transparent is checked rather than assumed. Reconstructing
run 0 through it reproduces run 0's own ten performance numbers exactly:

    sim/baseline/aggregate.sh baseline_numbers.py runs/0
    sim/baseline/aggregate.sh baseline_numbers.py runs/prof0

## What the box needs, and why

The container's default seccomp profile blocks `perf_event_open`. It is fixed at
container creation, so `docker exec --privileged` cannot lift it and the
container is created with `--privileged --security-opt seccomp=unconfined`.

The image ships no perf and has no `linux-perf` apt candidate, so the host's
`/usr/lib/linux-tools-*/perf` is bind-mounted at `/hostperf`. It needs `libdw1`,
`libunwind8`, `libslang2`, `libnuma1` and `libtraceevent1`, which is all
`odd-sw:perf` is: the pinned image with those five packages. `run_profile.sh`
builds it if it is not there.

The event is `cpu-clock`, not the default `cycles`. WSL2 exposes no PMU, so the
hardware event does not exist and perf fails rather than falling back. The host
perf is 6.8.12 against a 6.18 WSL2 kernel, and `/usr/bin/perf` is a wrapper that
refuses that mismatch, so the versioned binary is called directly.

Frame pointers are not there. `-g` is recorded but every chain is addresses
rather than frames, so the reports are taken with `-g none` and the buckets are
read off flat self time. Attributing a sample by its caller is not available.

`cpu-clock` samples running CPU time, so time spent waiting on IO does not
appear. The particle writer is 23.7% of the event in `timing.tsv` and 7% of the
samples, and both are correct.

## Two things seen while running it

One `timing.tsv` recorded `Algorithm:ParticleSelector` at -1.458 s per event.
The Sequencer's per-algorithm accounting can go negative; the other entries in
the same file agree with the run beside it, so it is that one counter.

Track finding reads 6.51 to 7.03 s per event across three runs of the same four
events on this box. Shares are what these runs are for.
