"""One reconstruction run per arm, as the event display's JSON.

The display draws a track from its perigee parameters, which is what
`tracksummary_<stage>.root` carries: `eLOC0_fit`, `eLOC1_fit`, `ePHI_fit`,
`eTHETA_fit`, `eQOP_fit`, one entry per event and one array element per track.
`sim/slice_perf.py:173` reads the same four for its pT table. The hit cloud is
`measurements.root`, read the way `src/prop/baseline_hits.py` reads it, and the
digitized position `rec_g*` is used because that is what the filter saw.

Every arm of one comparison is reconstructed from the same `edm4hep.root`, so
the measurements are identical across arms and only the tracks differ. The hits
are therefore read once, from `--hits` or from the first arm's directory.

    python -m prop.export_display \
        --runs ~/baseline0/out/runs \
        --arm stock=ckf_accept_stock \
        --arm new-off=ckf_run0_newoff \
        --arm helix=ckf_run0_helix \
        --arm new-learned=ckf_run0_newlearned \
        --events 0,1,2 --out ~/display/data

Writes `events.json` and one `event_<n>.json` per event into `--out`, which is
the directory `viewer.html` is served from.

Track counts at pileup 200 are large and the JSON is uncompressed: one event of
item 0's ttbar is tens of MB with every hit in it. `--hit-stride` thins the hit
cloud for the display only; it changes nothing about the tracks.

Truth trajectories are off by default. `--truth` groups the measurement rows by
ACTS barcode and emits one polyline per particle, which is a polyline through
the hits a particle left, not a propagated trajectory.
"""
import argparse
import json
import os

import numpy as np

try:
    import awkward as ak
    import uproot
except ImportError as exc:  # pragma: no cover - depends on the [root] extra
    raise SystemExit(
        "export_display needs the root extra: pip install -e .[root]") from exc

from prop.baseline_hits import BARCODE, pack_barcode

# What the display needs from a track, and the tracksummary branch it is in.
TRACK_BRANCHES = {
    "d0": "eLOC0_fit",
    "z0": "eLOC1_fit",
    "phi": "ePHI_fit",
    "theta": "eTHETA_fit",
    "qop": "eQOP_fit",
    "has_fit": "hasFittedParams",
}
# Drawn in the readout when the file carries them, skipped when it does not.
OPTIONAL_BRANCHES = {"n_meas": "nMeasurements", "n_holes": "nHoles",
                     "n_outliers": "nOutliers", "chi2": "chi2Sum"}

ARM_COLOURS = ["#22d3ee", "#f5a524", "#a855f7", "#34d399",
               "#f87171", "#60a5fa"]


def read_tracks(path, events):
    """Perigee parameters per event, from one arm's tracksummary file."""
    tree = uproot.open(path)["tracksummary"]
    have = set(tree.keys())
    missing = [b for b in TRACK_BRANCHES.values() if b not in have]
    if missing:
        raise SystemExit(
            f"{path}: no {', '.join(missing)} in the tracksummary tree. "
            f"It has: {', '.join(sorted(have))}")

    names = dict(TRACK_BRANCHES)
    names.update({k: v for k, v in OPTIONAL_BRANCHES.items() if v in have})
    if "event_nr" not in have:
        raise SystemExit(f"{path}: no event_nr branch to split events on")

    arrays = tree.arrays(list(names.values()) + ["event_nr"])
    event_nr = ak.to_numpy(arrays["event_nr"]).astype(np.int64)

    out = {}
    for i, ev in enumerate(event_nr):
        if events is not None and int(ev) not in events:
            continue
        cols = {k: ak.to_list(arrays[b][i]) for k, b in names.items()}
        n = len(cols["d0"])
        tracks = []
        for j in range(n):
            if not cols["has_fit"][j]:
                continue
            t = {k: float(cols[k][j]) for k in
                 ("d0", "z0", "phi", "theta", "qop")}
            for k in OPTIONAL_BRANCHES:
                if k in cols:
                    t[k] = float(cols[k][j])
            tracks.append(t)
        out.setdefault(int(ev), []).extend(tracks)
    return out


def read_hits(path, events, stride):
    """Digitized hit positions per event, one row per measurement."""
    tree = uproot.open(path)["measurements"]
    want = ["event_nr", "volume_id", "rec_gx", "rec_gy", "rec_gz"]
    have = set(tree.keys())
    missing = [b for b in want if b not in have]
    if missing:
        raise SystemExit(
            f"{path}: no {', '.join(missing)} in the measurements tree")
    flat = tree.arrays(want, library="np")

    out = {}
    for ev in np.unique(flat["event_nr"]):
        if events is not None and int(ev) not in events:
            continue
        sel = np.flatnonzero(flat["event_nr"] == ev)[::stride]
        out[int(ev)] = [
            {"x": float(flat["rec_gx"][k]), "y": float(flat["rec_gy"][k]),
             "z": float(flat["rec_gz"][k]),
             "volume_id": int(flat["volume_id"][k])}
            for k in sel
        ]
    return out


def read_truth(path, events, min_hits):
    """One polyline per particle, through the hits it left.

    A merged cluster carries more than one truth link, so this reads the
    per-link expansion `baseline_hits.read_run` documents and keeps every
    (measurement, particle) pair. The points are ordered by radius, which is
    the order a track crosses the barrel but not the order it crosses an
    endcap disk at fixed r.
    """
    tree = uproot.open(path)["measurements"]
    flat = tree.arrays(["event_nr", "rec_gx", "rec_gy", "rec_gz"],
                       library="np")
    jag = tree.arrays([f"particles_{n}" for n, _ in BARCODE])
    n_link = ak.to_numpy(ak.num(jag["particles_particle"]))
    rep = np.repeat(np.arange(len(n_link)), n_link)
    parts = {n: ak.to_numpy(ak.flatten(jag[f"particles_{n}"]))
             for n, _ in BARCODE}
    barcode = pack_barcode(parts)

    ev_all = flat["event_nr"][rep]
    x, y, z = (flat[f"rec_g{a}"][rep] for a in "xyz")

    out = {}
    for ev in np.unique(ev_all):
        if events is not None and int(ev) not in events:
            continue
        m = ev_all == ev
        lines = []
        for pid in np.unique(barcode[m]):
            k = np.flatnonzero(m & (barcode == pid))
            if len(k) < min_hits:
                continue
            k = k[np.argsort(np.hypot(x[k], y[k]))]
            lines.append({
                "particle_id": int(pid),
                "points": [{"x": float(x[i]), "y": float(y[i]),
                            "z": float(z[i])} for i in k],
            })
        out[int(ev)] = lines
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default=os.path.expanduser(
        "~/baseline0/out/runs"), help="directory holding the run subdirs")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=SUBDIR",
                    help="one arm, e.g. new-learned=ckf_run0_newlearned. "
                         "Repeat for each arm; order sets the colour")
    ap.add_argument("--stage", default="ckf",
                    help="the tracksummary stage suffix, tracksummary_<stage>"
                         ".root. The chain also writes an ambiguity-resolved "
                         "stage under a different name")
    ap.add_argument("--hits", default="",
                    help="measurements.root to draw. Defaults to the first "
                         "arm's, which is the same file every arm read")
    ap.add_argument("--events", default="0",
                    help="comma separated event_nr to export, or all")
    ap.add_argument("--hit-stride", type=int, default=1,
                    help="keep every Nth hit in the display only")
    ap.add_argument("--truth", action="store_true",
                    help="also emit a polyline per truth particle")
    ap.add_argument("--truth-min-hits", type=int, default=4)
    ap.add_argument("--out", default="display_data")
    args = ap.parse_args()

    if not args.arm:
        raise SystemExit("no --arm given")
    arms = []
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm wants NAME=SUBDIR, got {spec}")
        name, subdir = spec.split("=", 1)
        arms.append((name, subdir))

    events = None
    if args.events.strip() != "all":
        events = {int(e) for e in args.events.split(",") if e.strip()}

    per_arm = {}
    for name, subdir in arms:
        path = os.path.join(args.runs, subdir, f"tracksummary_{args.stage}.root")
        if not os.path.exists(path):
            raise SystemExit(f"no {path}")
        per_arm[name] = read_tracks(path, events)
        n = sum(len(v) for v in per_arm[name].values())
        print(f"  {name:<12} {n:>7,} fitted tracks  {os.path.basename(path)}")

    hits_path = args.hits or os.path.join(
        args.runs, arms[0][1], "measurements.root")
    if not os.path.exists(hits_path):
        raise SystemExit(f"no {hits_path}")
    hits = read_hits(hits_path, events, max(1, args.hit_stride))
    print(f"  hits         {sum(len(v) for v in hits.values()):>7,} "
          f"(stride {args.hit_stride})  {hits_path}")

    truth = read_truth(hits_path, events, args.truth_min_hits) if args.truth \
        else {}
    if args.truth:
        print(f"  truth        {sum(len(v) for v in truth.values()):>7,} "
              f"particles with >= {args.truth_min_hits} hits")

    ids = sorted(hits) if events is None else sorted(events & set(hits))
    if not ids:
        raise SystemExit("no event matched --events in the measurements file")

    os.makedirs(args.out, exist_ok=True)
    for ev in ids:
        doc = {
            "metadata": {"event_id": ev, "source": args.runs,
                         "stage": args.stage,
                         "hit_stride": args.hit_stride},
            "all_tracker_hits": hits.get(ev, []),
            "arms": {name: per_arm[name].get(ev, []) for name, _ in arms},
            "tracks": [{"particle_id": t["particle_id"], "pT": 0.0,
                        "pdg_id": 0, "hit_ids": [],
                        "reco_info": {"has_reco": False},
                        "points": t["points"]}
                       for t in truth.get(ev, [])],
        }
        path = os.path.join(args.out, f"event_{ev}.json")
        with open(path, "w") as f:
            json.dump(doc, f)
        size = os.path.getsize(path) / 1e6
        counts = " ".join(f"{n}:{len(per_arm[n].get(ev, []))}"
                          for n, _ in arms)
        print(f"  event {ev:<4} {size:6.1f} MB  {len(doc['all_tracker_hits'])}"
              f" hits  {counts}  -> {path}")

    index = {
        "events": ids,
        "track_sources": [n for n, _ in arms],
        "arms": [{"name": n, "subdir": s,
                  "colour": ARM_COLOURS[i % len(ARM_COLOURS)]}
                 for i, (n, s) in enumerate(arms)],
    }
    with open(os.path.join(args.out, "events.json"), "w") as f:
        json.dump(index, f)
    print(f"\n{len(ids)} events, {len(arms)} arms -> {args.out}/events.json")


if __name__ == "__main__":
    main()
