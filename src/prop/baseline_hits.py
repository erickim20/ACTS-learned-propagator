"""The item-0 ttbar baseline's measurements, in the table `ckf_prototype` reads.

`ckf_prototype.load_event_hits` was written against the ColliderML production
parquet. That sample is not on this box; the baseline of item 0 is, as ACTS
ROOT output under `out/runs/<n>/measurements.root`. This converts one into the
other so the real-event comparison runs on ttbar at pileup 200, which is the
evaluation sample from item 0 onward.

What the columns mean:

  x, y, z            the DIGITIZED global position, `rec_g*`. This is what a
                     filter sees.
  true_x, y, z       the unsmeared position. Scoring only.
  particle_id        the ACTS barcode, packed into one int64.

**One row per (measurement, contributing particle), not per measurement.** A
merged cluster carries more than one truth link: 97.4% of measurements have
exactly one, the tail reaches 18. Emitting one row each is what makes
`true_tracks` see every hit a particle actually left. A merged cluster then
appears twice in the layer's hit list, at one position, so the gate can pick
either and scores against the seed's barcode either way, which is the answer
that was wanted.

Cycles: each `measurements.root` holds two TTree cycles. The higher one is the
final write and the lower is a ROOT autosave snapshot, so `f["measurements"]`
without a cycle is correct and is what this uses.

This is the only thing here that reads ROOT, so `uproot` is the `root` extra
rather than a dependency:

    pip install -e .[root]
    python -m prop.baseline_hits --runs sim/baseline/out/runs \
        --out ttbar_hits.parquet
"""
import argparse
import os
import re

import awkward as ak
import numpy as np
import polars as pl
import uproot

# ACTS's own barcode field widths, in bits, most significant first.
BARCODE = (("vertex_primary", 12), ("vertex_secondary", 12),
           ("particle", 16), ("generation", 8), ("sub_particle", 16))


def pack_barcode(parts):
    """The five barcode fields as one int64, ACTS's own bit layout.

    Only identity is ever asked of the result, but the layout is ACTS's so the
    value can be read back against `particles.root` without a translation.
    """
    out = np.zeros(len(parts["particle"]), dtype=np.int64)
    shift = 64
    for name, width in BARCODE:
        shift -= width
        v = parts[name].astype(np.int64)
        if v.size and int(v.max()) >= (1 << width):
            raise ValueError(f"{name} overflows {width} bits: max {v.max()}")
        out |= v << shift
    return out


def read_run(path, event_offset):
    """One `measurements.root` to flat arrays, one row per truth link."""
    t = uproot.open(path)["measurements"]
    flat = t.arrays(["event_nr", "volume_id", "layer_id", "surface_id",
                     "extra_id", "rec_gx", "rec_gy", "rec_gz",
                     "true_x", "true_y", "true_z"], library="np")
    jag = t.arrays([f"particles_{n}" for n, _ in BARCODE])

    n_link = ak.to_numpy(ak.num(jag["particles_particle"]))
    rep = np.repeat(np.arange(len(n_link)), n_link)

    parts = {n: ak.to_numpy(ak.flatten(jag[f"particles_{n}"]))
             for n, _ in BARCODE}
    return {
        "event_id": flat["event_nr"][rep].astype(np.int64) + event_offset,
        "volume_id": flat["volume_id"][rep].astype(np.int32),
        "layer_id": flat["layer_id"][rep].astype(np.int32),
        "surface_id": flat["surface_id"][rep].astype(np.int32),
        "extra_id": flat["extra_id"][rep].astype(np.int32),
        "x": flat["rec_gx"][rep].astype(np.float64),
        "y": flat["rec_gy"][rep].astype(np.float64),
        "z": flat["rec_gz"][rep].astype(np.float64),
        "true_x": flat["true_x"][rep].astype(np.float64),
        "true_y": flat["true_y"][rep].astype(np.float64),
        "true_z": flat["true_z"][rep].astype(np.float64),
        "particle_id": pack_barcode(parts),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="sim/baseline/out/runs",
                    help="the baseline's per-run output directories")
    ap.add_argument("--out", default="ttbar_hits.parquet")
    ap.add_argument("--events-per-run", type=int, default=1000,
                    help="stride for the global event id. Only has to exceed "
                         "the events in any one run directory")
    ap.add_argument("--dirs", default="0,1,2,3,4,6",
                    help="which run directories make up the sample. The "
                         "default is item 0's 24 events, six runs of four. "
                         "The tree also holds partial runs that the baseline "
                         "number was not measured on. Empty means all")
    args = ap.parse_args()

    dirs = sorted((d for d in os.listdir(args.runs)
                   if re.fullmatch(r"\d+", d)), key=int)
    if args.dirs.strip():
        want = args.dirs.split(",")
        dirs = [d for d in dirs if d in want]

    cols, n_ev, used, skipped = [], 0, [], []
    for d in dirs:
        p = os.path.join(args.runs, d, "measurements.root")
        if not os.path.exists(p):
            skipped.append(d)
            continue
        c = read_run(p, int(d) * args.events_per_run)
        ne = len(np.unique(c["event_id"]))
        n_ev += ne
        used.append((d, ne, len(c["event_id"])))
        cols.append(c)
        print(f"  run {d:>3}  {ne} events  {len(c['event_id']):>9,} rows")

    if not cols:
        raise SystemExit(f"no measurements.root under {args.runs}")

    df = pl.DataFrame({k: np.concatenate([c[k] for c in cols])
                       for k in cols[0]})
    df.write_parquet(args.out)

    print(f"\n{len(used)} run directories, {n_ev} events, {len(df):,} rows "
          f"-> {args.out}")
    if skipped:
        print(f"no measurements.root in {len(skipped)} directories: "
              f"{','.join(skipped)}")
    print(f"volumes {sorted(df['volume_id'].unique().to_list())}")
    print(f"distinct particles {df['particle_id'].n_unique():,}")


if __name__ == "__main__":
    main()
