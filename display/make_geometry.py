"""Derive an approximate ODD layer geometry from a directory of event JSON.

The event JSONs carry a volume_id per tracker hit but no layer_id and no
surface description, so the layers here are reconstructed from the hits
themselves: barrel volumes are clustered in r, endcap volumes in z. The
result is an envelope of where hits were recorded, not the ODD surface
geometry. A layer no event illuminated is absent.

Runs against either input, because both write `all_tracker_hits` with a
volume_id: the ColliderML sample events, or a directory written by
`python -m prop.export_display`.

    python display/make_geometry.py --events <dir> --out <dir>/geometry.json

`--root` reads a `measurements.root` directly instead, which is what a single
muon export wants: one muon lights up a dozen modules and the clustering needs
a pileup event to find every layer.

    python display/make_geometry.py \
        --root ~/baseline0/out/runs/0/measurements.root \
        --out ~/display/geometry.json

Against one pileup 200 event the barrel radii come out at the ODD's own
33/68/114/170 mm pixel, 260/360/500/660 mm short strip, 820/1020 mm long
strip.
"""

import argparse
import json
import pathlib

GROUPS = [
    # name, colour, barrel volume, endcap volumes, r-cluster tol, z-cluster tol
    ("pixel", "#4a90d9", 17, (16, 18), 8.0, 20.0),
    ("short_strip", "#4caf50", 24, (23, 25), 25.0, 40.0),
    ("long_strip", "#ff9800", 29, (28, 30), 40.0, 60.0),
]

MIN_HITS = 6


def cluster(values, tol):
    """Split a sorted 1-D list wherever consecutive values differ by > tol."""
    out, cur = [], []
    for v in sorted(values):
        if cur and v - cur[-1][0] > tol:
            out.append(cur)
            cur = []
        cur.append((v,))
    if cur:
        out.append(cur)
    return [[x[0] for x in c] for c in out]


def cluster_pairs(pairs, tol, key=0):
    """cluster() over (a, b) pairs, splitting on the `key` component."""
    out, cur = [], []
    for p in sorted(pairs, key=lambda p: p[key]):
        if cur and p[key] - cur[-1][key] > tol:
            out.append(cur)
            cur = []
        cur.append(p)
    if cur:
        out.append(cur)
    return out


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--events", default="events",
                help="directory of event_*.json")
ap.add_argument("--root", default="",
                help="a measurements.root to read instead of --events. One "
                     "pileup event illuminates every module; a single muon "
                     "event does not, so a single muon export wants this "
                     "pointed at a ttbar run and --out into its own directory")
ap.add_argument("--root-events", type=int, default=1,
                help="how many events of --root to pool")
ap.add_argument("--out", default="",
                help="defaults to geometry.json beside --events")
args = ap.parse_args()

events_dir = pathlib.Path(args.events)
hits = []
if args.root:
    import uproot

    tree = uproot.open(args.root)["measurements"]
    flat = tree.arrays(["event_nr", "volume_id", "rec_gx", "rec_gy", "rec_gz"],
                       library="np")
    keep = flat["event_nr"] < args.root_events
    source = f"{keep.sum():,} hits from {args.root_events} event(s) of " \
             f"{args.root}"
    for v, x, y, z in zip(flat["volume_id"][keep], flat["rec_gx"][keep],
                          flat["rec_gy"][keep], flat["rec_gz"][keep]):
        hits.append((int(v), float((x ** 2 + y ** 2) ** 0.5), float(z)))
    paths = [args.root]
else:
    paths = sorted(events_dir.glob("event_*.json"))
    if not paths:
        raise SystemExit(f"no event_*.json under {events_dir}")
    source = f"derived from hit positions in {len(paths)} event files under " \
             f"{events_dir}"
    for path in paths:
        d = json.loads(path.read_text())
        for h in d["all_tracker_hits"]:
            r = (h["x"] ** 2 + h["y"] ** 2) ** 0.5
            hits.append((h["volume_id"], r, h["z"]))

layers = []
for name, colour, barrel_vol, endcap_vols, r_tol, z_tol in GROUPS:
    # Barrel: cluster in r, extent in z.
    bar = [(r, z) for v, r, z in hits if v == barrel_vol]
    for i, c in enumerate(cluster_pairs(bar, r_tol, key=0)):
        if len(c) < MIN_HITS:
            continue
        rs = [p[0] for p in c]
        zs = [abs(p[1]) for p in c]
        layers.append(
            {
                "group": name,
                "colour": colour,
                "kind": "cylinder",
                "label": f"{barrel_vol}·{i}",
                "r": sum(rs) / len(rs),
                "half_z": max(zs),
                "n_hits": len(c),
            }
        )

    # Endcaps: cluster in z, extent in r.
    for vol in endcap_vols:
        ec = [(z, r) for v, r, z in hits if v == vol]
        for i, c in enumerate(cluster_pairs(ec, z_tol, key=0)):
            if len(c) < MIN_HITS:
                continue
            zs = [p[0] for p in c]
            rs = [p[1] for p in c]
            layers.append(
                {
                    "group": name,
                    "colour": colour,
                    "kind": "disk",
                    "label": f"{vol}·{i}",
                    "z": sum(zs) / len(zs),
                    "r_min": min(rs),
                    "r_max": max(rs),
                    "n_hits": len(c),
                }
            )

out = {"source": source, "layers": layers}
dest = pathlib.Path(args.out) if args.out else events_dir / "geometry.json"
dest.write_text(json.dumps(out))
print(f"{len(hits):,} hits -> {dest}")
print(source)

for g, _, _, _, _, _ in GROUPS:
    cyl = [l for l in layers if l["group"] == g and l["kind"] == "cylinder"]
    dsk = [l for l in layers if l["group"] == g and l["kind"] == "disk"]
    print(f"{g:12s} {len(cyl)} barrel layers r = "
          + ", ".join(f"{l['r']:.0f}" for l in cyl))
    print(f"{'':12s} {len(dsk)} endcap disks z = "
          + ", ".join(f"{l['z']:.0f}" for l in dsk))
