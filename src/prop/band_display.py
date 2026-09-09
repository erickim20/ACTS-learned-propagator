"""The event display, restricted to one |eta| band and to what an arm lost.

`sim/arm_grid.py` reports track finding efficiency as a number per pT bin and
|eta| band, and `prop.arm_grid_table` prints the grid. This picks the events
those numbers are made of and writes them as the display's JSON, so a band can
be looked at instead of read.

    python -m prop.band_display \\
        --runs ~/baseline0/out/runs \\
        --ref stock=t23_stock_b%d \\
        --arm g010=t24_g010_b%d \\
        --bins 1,2,3,4,5 --band 1.2-1.8 --pick lost \\
        --n 12 --out ~/display/g010_eta12

The sample is one generated muon per event, so the denominator of a band is
the muons generated into it and the table this prints is the grid's own
quantity rather than a proxy for it. Against `~/j24/arms_a24.json` the bands
0.6-1.2, 1.2-1.8 and 2.4-3.0 reproduce to the third decimal. The denominator
here is 8 muons per 10000 larger than `trackeff_vs_eta`'s at |eta| < 0.15 and
2 per 10000 larger in 1.8-2.4, all of them unmatched, so 0.0-0.6 reads about
0.56 points low and 1.8-2.4 about 0.08 low. What drops those muons from the
writer's histogram is not identified. It does not touch the band the deficit
is in.

A muon counts as found by an arm when a track of that arm carries it as the
majority particle over at least `--matching-ratio` of the track's
measurements, which is `TrackTruthMatcher`'s `recoMatched` test at its
default, on the stage already written. Ambiguity resolution is whichever
stage `--stage` names.

    --pick lost     the reference found the muon and the arm did not. This is
                    the band's deficit.
    --pick found    both found it.
    --pick miss     neither found it.
    --pick all      every muon in the band.

Each arm's track is drawn through the measurements it collected, in
`measurementIDs` order, so the picture shows where a track stops rather than
what its perigee fit extrapolates to. The truth polyline is the unsmeared
position `true_*` of every measurement the muon left, ordered by distance from
the origin, which is the crossing order for a track that does not turn back
inside the tracker. Both are drawn beside the digitized cloud `rec_g*`, which
is what the filter saw.

`display/make_geometry.py --root` writes the `geometry.json` the viewer wants
beside the events; a single muon illuminates too few modules to build the
envelope from these events themselves.
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
        "band_display needs the root extra: pip install -e .[root]") from exc

from prop.baseline_hits import BARCODE, pack_barcode
from prop.export_display import ARM_COLOURS

# `sim/arm_grid.py`'s own edges, so a band here is a column there.
ETA_EDGES = [0.0, 0.6, 1.2, 1.8, 2.4, 3.0, 4.0]

MAJORITY = [f"majorityParticleId_{n}" for n, _ in BARCODE]
PERIGEE = {"d0": "eLOC0_fit", "z0": "eLOC1_fit", "phi": "ePHI_fit",
           "theta": "eTHETA_fit", "qop": "eQOP_fit"}
SCALARS = ["nMeasurements", "nMajorityHits", "nHoles", "nOutliers", "chi2Sum",
           "hasFittedParams", "t_eta", "t_pT"] + list(PERIGEE.values()) \
    + MAJORITY


def flatten(arrays, keys):
    """The jagged per-event arrays as one row per track, and its event."""
    counts = ak.to_numpy(ak.num(arrays[keys[0]]))
    event = np.repeat(ak.to_numpy(arrays["event_nr"]).astype(np.int64), counts)
    index = np.concatenate([np.arange(c) for c in counts]).astype(np.int64) \
        if counts.size else np.zeros(0, dtype=np.int64)
    flat = {k: ak.to_numpy(ak.flatten(arrays[k])) for k in keys}
    return event, index, flat


def read_primaries(path):
    """The generated muons of one run, keyed by (event, barcode)."""
    tree = uproot.open(path)["particles"]
    names = [n for n, _ in BARCODE]
    arrays = tree.arrays(names + ["event_id", "eta", "pt", "particle_type",
                                  "number_of_hits"])
    counts = ak.to_numpy(ak.num(arrays["eta"]))
    event = np.repeat(ak.to_numpy(arrays["event_id"]).astype(np.int64), counts)
    flat = {k: ak.to_numpy(ak.flatten(arrays[k]))
            for k in names + ["eta", "pt", "particle_type", "number_of_hits"]}
    barcode = pack_barcode({n: flat[n] for n in names})

    keep = np.flatnonzero(flat["generation"] == 0)
    return [{"event": int(event[i]), "barcode": int(barcode[i]),
             "eta": float(flat["eta"][i]), "pt": float(flat["pt"][i]),
             "pdg": int(flat["particle_type"][i]),
             "n_hits": int(flat["number_of_hits"][i])} for i in keep]


def read_arm(path, ratio):
    """Every track of one run, and the (event, barcode) each one matched."""
    tree = uproot.open(path)["tracksummary"]
    have = set(tree.keys())
    missing = [b for b in SCALARS + ["event_nr"] if b not in have]
    if missing:
        raise SystemExit(f"{path}: no {', '.join(missing)} in tracksummary")

    arrays = tree.arrays(SCALARS + ["event_nr"])
    event, index, flat = flatten(arrays, SCALARS)
    entry_of_event = {int(e): i for i, e in
                      enumerate(ak.to_numpy(arrays["event_nr"]))}
    barcode = pack_barcode({n: flat[f"majorityParticleId_{n}"]
                            for n, _ in BARCODE})
    n_meas = flat["nMeasurements"].astype(np.float64)
    share = np.divide(flat["nMajorityHits"], n_meas,
                      out=np.zeros_like(n_meas), where=n_meas > 0)

    matched = {}
    for i in np.flatnonzero(share >= ratio):
        key = (int(event[i]), int(barcode[i]))
        best = matched.get(key)
        if best is None or flat["nMajorityHits"][i] > flat["nMajorityHits"][best]:
            matched[key] = int(i)
    return {"tree": tree, "event": event, "index": index, "flat": flat,
            "barcode": barcode, "share": share, "matched": matched,
            "entry_of_event": entry_of_event}


def measurement_ids(arm, event):
    """`measurementIDs` for one event, read only for the events written."""
    entry = arm["entry_of_event"].get(int(event))
    if entry is None:
        return None
    return arm["tree"]["measurementIDs"].array(
        entry_start=entry, entry_stop=entry + 1)[0]


def read_measurements(path):
    """One run's measurements, flat, plus the truth link expansion."""
    tree = uproot.open(path)["measurements"]
    flat = tree.arrays(["event_nr", "volume_id", "rec_gx", "rec_gy", "rec_gz",
                        "true_x", "true_y", "true_z"], library="np")
    jag = tree.arrays([f"particles_{n}" for n, _ in BARCODE])
    n_link = ak.to_numpy(ak.num(jag["particles_particle"]))
    link_row = np.repeat(np.arange(len(n_link)), n_link)
    link_barcode = pack_barcode({n: ak.to_numpy(ak.flatten(jag[f"particles_{n}"]))
                                 for n, _ in BARCODE})

    event = flat["event_nr"].astype(np.int64)
    first = {}
    for ev in np.unique(event):
        rows = np.flatnonzero(event == ev)
        first[int(ev)] = (rows[0], len(rows))
    return {"flat": flat, "event": event, "row_of_event": first,
            "link_row": link_row, "link_barcode": link_barcode}


def arm_track(arm, row, ids, meas):
    """One track as the display draws it: its own measurements, in order."""
    base, n_rows = meas["row_of_event"][int(arm["event"][row])]
    flat, points, kept = meas["flat"], [], []
    for mid in ids:
        k = base + int(mid)
        if not 0 <= int(mid) < n_rows:
            continue
        kept.append(int(mid))
        points.append({"x": float(flat["rec_gx"][k]),
                       "y": float(flat["rec_gy"][k]),
                       "z": float(flat["rec_gz"][k])})
    f = arm["flat"]
    track = {k: float(f[b][row]) for k, b in PERIGEE.items()}
    track.update({
        "n_meas": int(f["nMeasurements"][row]),
        "n_holes": int(f["nHoles"][row]),
        "n_outliers": int(f["nOutliers"][row]),
        "chi2": float(f["chi2Sum"][row]),
        "n_majority": int(f["nMajorityHits"][row]),
        "majority_share": float(arm["share"][row]),
        "has_fit": bool(f["hasFittedParams"][row]),
        "measurement_ids": kept,
        "points": points,
    })
    return track


def truth_points(meas, event, barcode):
    """The muon's own crossings, ordered by distance from the origin."""
    flat = meas["flat"]
    rows = meas["link_row"][meas["link_barcode"] == barcode]
    rows = np.unique(rows[meas["event"][rows] == event])
    if not len(rows):
        return [], []
    x, y, z = (flat[f"true_{a}"][rows] for a in "xyz")
    order = np.argsort(np.sqrt(x ** 2 + y ** 2 + z ** 2))
    rows = rows[order]
    base = meas["row_of_event"][int(event)][0]
    return ([{"x": float(flat["true_x"][k]), "y": float(flat["true_y"][k]),
              "z": float(flat["true_z"][k])} for k in rows],
            [int(k - base) for k in rows])


def event_hits(meas, event):
    flat = meas["flat"]
    base, n_rows = meas["row_of_event"][int(event)]
    return [{"x": float(flat["rec_gx"][k]), "y": float(flat["rec_gy"][k]),
             "z": float(flat["rec_gz"][k]),
             "volume_id": int(flat["volume_id"][k])}
            for k in range(base, base + n_rows)]


def band_of(eta, edges):
    k = int(np.searchsorted(edges, abs(eta), side="right") - 1)
    return k if 0 <= k < len(edges) - 1 else None


def parse_band(text, edges):
    lo, hi = (float(v) for v in text.replace(" ", "").split("-"))
    for k in range(len(edges) - 1):
        if abs(edges[k] - lo) < 1e-9 and abs(edges[k + 1] - hi) < 1e-9:
            return k
    have = ", ".join(f"{edges[k]:.1f}-{edges[k+1]:.1f}"
                     for k in range(len(edges) - 1))
    raise SystemExit(f"--band {text} is not one of the grid's bands: {have}")


def efficiency_table(rows, names, edges):
    """Found over generated per |eta| band, the grid's own quantity."""
    head = f"{'|eta|':>10}" + "".join(f"{n:>10}" for n in names) + \
        f"{'muons':>10}"
    out = [head]
    for k in range(len(edges) - 1):
        sel = [r for r in rows if r["band"] == k]
        if not sel:
            continue
        line = f"{edges[k]:.1f}-{edges[k+1]:.1f}".rjust(10)
        for n in names:
            line += f"{100.0 * sum(r['found'][n] for r in sel) / len(sel):>10.3f}"
        out.append(line + f"{len(sel):>10}")
    line = "all".rjust(10)
    for n in names:
        line += f"{100.0 * sum(r['found'][n] for r in rows) / len(rows):>10.3f}"
    out.append(line + f"{len(rows):>10}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.expanduser(
        "~/baseline0/out/runs"), help="directory holding the run subdirs")
    ap.add_argument("--ref", default="stock=t23_stock_b%d",
                    metavar="NAME=PATTERN",
                    help="the arm a muon is called found by, for --pick lost")
    ap.add_argument("--arm", action="append", default=[],
                    metavar="NAME=PATTERN",
                    help="one arm, PATTERN carrying one %%d for the pT bin. "
                         "Repeat; the order sets the colour")
    ap.add_argument("--bins", default="1,2,3,4,5",
                    help="the pT bins the pattern is filled with")
    ap.add_argument("--band", default="",
                    help="one |eta| band, e.g. 1.2-1.8. Empty keeps all")
    ap.add_argument("--pick", default="lost",
                    choices=["lost", "found", "miss", "all"])
    ap.add_argument("--n", type=int, default=12,
                    help="how many events to write")
    ap.add_argument("--stage", default="ambi",
                    help="tracksummary_<stage>.root. The grid reads ambi")
    ap.add_argument("--matching-ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260730,
                    help="which of the matching events are written")
    ap.add_argument("--out", default="display_band")
    args = ap.parse_args()

    if not args.arm:
        raise SystemExit("no --arm given")
    bins = [int(b) for b in args.bins.split(",") if b.strip()]
    arms = []
    for spec in [args.ref] + args.arm:
        if "=" not in spec:
            raise SystemExit(f"wants NAME=PATTERN, got {spec}")
        arms.append(tuple(spec.split("=", 1)))
    ref_name = arms[0][0]
    names = [n for n, _ in arms]
    want = None if not args.band else parse_band(args.band, ETA_EDGES)

    rows, state = [], {}
    for b in bins:
        per_arm = {}
        for name, pattern in arms:
            subdir = pattern % b if "%d" in pattern else pattern
            path = os.path.join(args.runs, subdir,
                                f"tracksummary_{args.stage}.root")
            if not os.path.exists(path):
                raise SystemExit(f"no {path}")
            per_arm[name] = read_arm(path, args.matching_ratio)
        ref_dir = (arms[0][1] % b) if "%d" in arms[0][1] else arms[0][1]
        primaries = read_primaries(
            os.path.join(args.runs, ref_dir, "particles.root"))
        state[b] = {"arms": per_arm, "dirs": {
            n: (p % b if "%d" in p else p) for n, p in arms}}

        for p in primaries:
            k = band_of(p["eta"], ETA_EDGES)
            if k is None:
                continue
            key = (p["event"], p["barcode"])
            rows.append({**p, "bin": b, "band": k, "found": {
                n: key in per_arm[n]["matched"] for n in names}})
        counts = []
        for name in names:
            matched = per_arm[name]["matched"]
            n_found = sum(1 for p in primaries
                          if (p["event"], p["barcode"]) in matched)
            counts.append(f"{name} {n_found}")
        print(f"  bin {b}: {len(primaries)} muons, " + ", ".join(counts))

    print()
    print(efficiency_table(rows, names, ETA_EDGES))
    print()
    print("Found over generated, one muon per event, matching ratio "
          f"{args.matching_ratio} on stage {args.stage}. The grid's own "
          "quantity. 0.0-0.6 and 1.8-2.4 carry a denominator this counts and "
          "trackeff_vs_eta does not; see the module docstring.")
    print()

    def wanted(r):
        if want is not None and r["band"] != want:
            return False
        found = [r["found"][n] for n in names[1:]]
        if args.pick == "lost":
            return r["found"][ref_name] and not any(found)
        if args.pick == "found":
            return r["found"][ref_name] and all(found)
        if args.pick == "miss":
            return not r["found"][ref_name] and not any(found)
        return True

    pool = [r for r in rows if wanted(r)]
    if not pool:
        raise SystemExit(f"--pick {args.pick} matched no muon in this band")
    rng = np.random.default_rng(args.seed)
    take = sorted(rng.choice(len(pool), size=min(args.n, len(pool)),
                             replace=False).tolist())
    chosen = [pool[i] for i in take]
    band_text = args.band or "all"
    print(f"--pick {args.pick} in |eta| {band_text}: {len(pool)} muons, "
          f"writing {len(chosen)}")

    os.makedirs(args.out, exist_ok=True)
    meas_cache, labels, ids = {}, {}, []
    checked = False
    for i, r in enumerate(chosen):
        b, ev = r["bin"], r["event"]
        if b not in meas_cache:
            meas_cache[b] = read_measurements(os.path.join(
                args.runs, state[b]["dirs"][ref_name], "measurements.root"))
        meas = meas_cache[b]
        points, hit_ids = truth_points(meas, ev, r["barcode"])

        arms_out = {}
        for name in names:
            arm = state[b]["arms"][name]
            sel = np.flatnonzero(arm["event"] == ev)
            ids_all = measurement_ids(arm, ev)
            out = []
            for row in sel:
                ids_of = ak.to_list(ids_all[int(arm["index"][row])])
                t = arm_track(arm, int(row), ids_of, meas)
                t["matched"] = ((ev, int(arm["barcode"][row]))
                                in arm["matched"]) and \
                    int(arm["barcode"][row]) == r["barcode"]
                out.append(t)
                if not checked and t["matched"]:
                    on = sum(1 for m in t["measurement_ids"] if m in hit_ids)
                    print(f"  measurementIDs check: {on} of "
                          f"{len(t['measurement_ids'])} of the matched track's "
                          f"ids are the muon's own hits, nMajorityHits says "
                          f"{t['n_majority']}")
                    checked = True
            arms_out[name] = out

        doc = {
            "metadata": {"bin": b, "event_nr": ev, "eta": r["eta"],
                         "pt": r["pt"], "band": band_text, "pick": args.pick,
                         "runs": args.runs, "stage": args.stage,
                         "dirs": state[b]["dirs"]},
            "all_tracker_hits": event_hits(meas, ev),
            "tracks": [{"particle_id": r["barcode"], "pdg_id": r["pdg"],
                        "pT": r["pt"], "eta": r["eta"], "hit_ids": hit_ids,
                        "reco_info": {"has_reco": False}, "points": points}],
            "arms": arms_out,
        }
        with open(os.path.join(args.out, f"event_{i}.json"), "w") as f:
            json.dump(doc, f)
        ids.append(i)
        found = " ".join(f"{n}{'+' if r['found'][n] else '-'}" for n in names)
        labels[str(i)] = (f"b{b} ev {ev} · |eta| {abs(r['eta']):.2f} · "
                          f"pT {r['pt']:.2f} GeV · {found}")
        print(f"  event {i:<3} {labels[str(i)]}  "
              f"{len(doc['all_tracker_hits'])} hits, "
              + " ".join(f"{n}:{len(arms_out[n])}" for n in names))

    index = {
        "events": ids,
        "labels": labels,
        "title": f"|eta| {band_text} · {args.pick}",
        "arms": [{"name": n, "subdir": p,
                  "colour": ARM_COLOURS[i % len(ARM_COLOURS)]}
                 for i, (n, p) in enumerate(arms)],
    }
    with open(os.path.join(args.out, "events.json"), "w") as f:
        json.dump(index, f)
    print(f"\n{len(ids)} events, {len(arms)} arms -> {args.out}/events.json")


if __name__ == "__main__":
    main()
