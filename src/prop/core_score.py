"""Score the three physics cores on the metric every other arm is quoted on.

`helix_variants.py` asks whether the learning target survives a better choice
of B, and answers it on one transverse coordinate against a cylinder r1 or a
plane z1, with no gate. That is not the metric anything else in this repository
uses. This script asks the same question on the scoring path:

  * the residual is in-plane `loc0` and `loc1` against the destination
    MODULE's own frame, which is the frame the sensor resolutions are quoted
    in and the frame `res_loc0_um` is written in;
  * S = V + measured C, from `chi2_gate.measured_cov` and the digi config;
  * the gate is `chi2_gate.gate` at the CKF's own `chi2CutOffMeasurement`;
  * the fold is the seed's own held-out third, split by track, taken exactly
    as `train_gtheta.main` takes it.

Three cores, identical rows, identical split, identical gate:

  fixed   B = 2.0 T everywhere. `LearnedTransport.hpp`'s `kBHelix`, what ships.
  source  Bz(x, y, z) at the transport's source. One map lookup, and the
          stepper already takes it (`LearnedStepper.hpp:1241`), so this is
          deployable.
  avg     0.5 (Bz(source) + Bz(destination)). Needs the destination point,
          which is what the transport is solving for, so it is an upper bound
          on what a midpoint sample could give and not a model.

No fit. No model is loaded and no network output enters any number here. The
`fixed` core is recomputed from the source state rather than read out of the
teacher's `res_loc0_um` column, and the check that it reproduces that column
to floating point is the check that the other two cores are solved the same
way.

Usage:
    python -m prop.core_score teacher_muon_pt1to50.parquet \
        --field oddb.npz --cov sigma_C.parquet \
        --digi config/odd-digi-smearing-config.json --seed 20260730
"""
import argparse

import numpy as np
import polars as pl

from .chi2_gate import (ETA_EDGES, PT_EDGES, gate, measured_cov, resolutions)
from .helix_variants import field_source
from .make_teacher_pairs import plane_predict

CORES = ("fixed", "source", "avg")


def mad(r):
    """1.4826 x the median absolute deviation about the median, in mm."""
    return 1.4826 * np.median(np.abs(r - np.median(r)))


def rms(r):
    """RMS about zero, in mm. Not about the mean: the gate is centred on the
    module's own position and a bias is a real cost, not a nuisance offset."""
    return float(np.sqrt(np.mean(r ** 2)))


def plane_residual(t, d, cen, u0, u1, nrm, q, bz):
    """loc0 and loc1 of the helix miss on the module plane, in um.

    `make_teacher_pairs.plane_predict` is the solver, so the arithmetic is the
    one the teacher's own column was written with and the only thing that
    varies between cores is `bz`.
    """
    out = plane_predict(d["x"], d["y"], d["z"], d["px"], d["py"], d["pz"],
                        q, bz, cen, nrm)
    pred = np.column_stack(out[:3])
    b0 = (((pred - cen) * u0).sum(1) - t["hit_loc0"].to_numpy()) * 1000.0
    b1 = (((pred - cen) * u1).sum(1) - t["hit_loc1"].to_numpy()) * 1000.0
    return b0, b1, out[-1]


def block(name, r0, keep, keep0, n):
    """One arm's four numbers on one population."""
    return dict(name=name, n=n, mad=mad(r0), rms=rms(r0),
                lost=100.0 * (keep0.mean() - keep.mean()),
                kept=100.0 * keep.mean())


def table(rows, title, note):
    print(f"\n### {title}")
    print(f"{'core':<10}{'n':>9}{'MAD [mm]':>12}{'RMS [mm]':>12}"
          f"{'hits lost':>12}{'kept [%]':>11}")
    for r in rows:
        print(f"{r['name']:<10}{r['n']:>9,}{r['mad']:12.4f}{r['rms']:12.4f}"
              f"{r['lost']:12.3f}{r['kept']:11.3f}")
    print(f"  {note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("--field", default="oddb.npz")
    ap.add_argument("--cov", default="sigma_C.parquet")
    ap.add_argument("--digi", default="config/odd-digi-smearing-config.json")
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--field-gate", type=float, default=0.10)
    args = ap.parse_args()

    t = pl.read_parquet(args.pairs)
    cols = ("x", "y", "z", "px", "py", "pz", "qop", "pt", "abs_eta",
            "x1", "y1", "z1")
    d = {c: t[c].to_numpy().astype(float) for c in cols}
    q = np.sign(d["qop"])
    cen = np.column_stack([t[c].to_numpy() for c in ("mc_x", "mc_y", "mc_z")])
    nrm = np.column_stack([t[c].to_numpy() for c in ("mn_x", "mn_y", "mn_z")])
    u0 = np.column_stack([t[c].to_numpy() for c in ("mu0_x", "mu0_y", "mu0_z")])
    u1 = np.column_stack([t[c].to_numpy() for c in ("mu1_x", "mu1_y", "mu1_z")])

    # --- the gate's sigma, exactly as train_gtheta builds it
    res = resolutions(args.digi)
    vol = t["surf1"].to_numpy() >> 56
    v0 = np.array([res[v][0] for v in vol])
    v1r = [res[v][1] for v in vol]
    one_d = np.array([w is None for w in v1r])
    v1 = np.array([1.0 if w is None else w for w in v1r])
    c0, c1 = measured_cov(t, vol, args.cov)
    s0, s1 = np.hypot(v0, c0), np.hypot(v1, c1)

    # --- the field
    fm = field_source(args.field)
    bz_src = fm.bz(d["x"], d["y"], d["z"])
    bz_dst = fm.bz(d["x1"], d["y1"], d["z1"])
    print(f"\n{len(t):,} transports. Bz at the source {bz_src.min():.4f} to "
          f"{bz_src.max():.4f} T, median {np.median(bz_src):.4f}. "
          f"|Bz(dst) - Bz(src)| median {np.median(np.abs(bz_dst-bz_src)):.4f} "
          f"T, p99 {np.percentile(np.abs(bz_dst-bz_src), 99):.4f} T")

    bzs = {"fixed": np.full(len(t), 2.0),
           "source": bz_src,
           "avg": 0.5 * (bz_src + bz_dst)}

    B0, B1, OK = {}, {}, {}
    for k, bz in bzs.items():
        B0[k], B1[k], OK[k] = plane_residual(t, d, cen, u0, u1, nrm, q, bz)

    # --- the check that has to pass before the other two cores are read
    ref0 = t["res_loc0_um"].to_numpy()
    ref1 = t["res_loc1_um"].to_numpy()
    print(f"\nfixed core against the teacher's own res_loc0_um: max |diff| "
          f"{np.abs(B0['fixed'] - ref0).max():.3e} um, res_loc1_um "
          f"{np.abs(B1['fixed'] - ref1).max():.3e} um")

    good = OK["fixed"] & OK["source"] & OK["avg"]
    print(f"plane solve: {(~OK['fixed']).sum()} fixed / "
          f"{(~OK['source']).sum()} source / {(~OK['avg']).sum()} avg rows "
          f"with no solution; {(~good).sum()} rows dropped from all three "
          f"so the comparison stays paired")

    # --- the split, taken exactly as train_gtheta.main takes it
    rng = np.random.default_rng(args.seed)
    tracks = t["track"].to_numpy()
    uniq = rng.permutation(np.unique(tracks))
    n = len(uniq)
    lut = {tr: (0 if i < 0.70 * n else (1 if i < 0.85 * n else 2))
           for i, tr in enumerate(uniq)}
    fold = np.array([lut[x] for x in tracks])
    te = np.flatnonzero((fold == 2) & good)
    print(f"{len(t):,} jumps, {n:,} tracks -> held-out fold {len(te):,}")

    gmask = np.abs(bz_src - 2.0) > args.field_gate
    print(f"field gate |bz - 2 T| > {args.field_gate:g} T: {gmask.sum():,} of "
          f"{len(t):,} ({100*gmask.mean():.2f} %); on the held-out fold "
          f"{gmask[te].sum():,} of {len(te):,}")

    g = np.random.default_rng(args.seed)
    _, keep0 = gate(np.zeros(len(te)), np.zeros(len(te)), s0[te], s1[te],
                    one_d[te], g)
    keep = {}
    for k in CORES:
        _, keep[k] = gate(B0[k][te], B1[k][te], s0[te], s1[te], one_d[te], g)

    def rows_on(sel, gsel=None):
        """One table's worth: the three cores over a boolean mask of `te`."""
        out = []
        for k in CORES:
            out.append(block(k, B0[k][te][sel] / 1000.0, keep[k][sel],
                             keep0[sel], int(sel.sum())))
        return out

    allm = np.ones(len(te), dtype=bool)
    table(rows_on(allm), "Pooled, whole held-out fold",
          "1.4826 x MAD of the loc0 residual about its median and RMS about "
          "zero, both mm; hits lost in points of hit efficiency against a "
          "perfect transport under the CKF's own chi2 cut of 15.0.")

    ae = d["abs_eta"][te]
    pt = d["pt"][te]
    for lo, hi in zip(ETA_EDGES[:-1], ETA_EDGES[1:]):
        m = (ae >= lo) & (ae < hi)
        if m.sum() == 0:
            continue
        table(rows_on(m), f"|eta| {lo} to {hi}", "same fold, same gate.")

    for lo, hi in zip(PT_EDGES[:-1], PT_EDGES[1:]):
        m = (pt >= lo) & (pt < hi)
        if m.sum() == 0:
            continue
        nm = "inf" if hi > 1e8 else f"{hi:g}"
        table(rows_on(m), f"pT {lo:g} to {nm} GeV", "same fold, same gate.")

    print("\n### By (pT, |eta|) cell, chi2_gate's own edges")
    for stat, fn in (("MAD [mm]", lambda r, kp: mad(r)),
                     ("RMS [mm]", lambda r, kp: rms(r)),
                     ("hits lost", None)):
        print(f"\n{stat}, fixed / source / avg")
        head = "".join(f"{f'{ETA_EDGES[j]}-{ETA_EDGES[j+1]}':>26}"
                       for j in range(len(ETA_EDGES) - 1))
        print(f"{'pT [GeV]':<10}{head}")
        for i in range(len(PT_EDGES) - 1):
            hi = "inf" if PT_EDGES[i + 1] > 1e8 else f"{PT_EDGES[i+1]:g}"
            line = f"{PT_EDGES[i]:g}-{hi:<8}"
            for j in range(len(ETA_EDGES) - 1):
                m = ((pt >= PT_EDGES[i]) & (pt < PT_EDGES[i + 1])
                     & (ae >= ETA_EDGES[j]) & (ae < ETA_EDGES[j + 1]))
                if m.sum() == 0:
                    line += f"{'-':>26}"
                    continue
                vals = []
                for k in CORES:
                    if stat == "hits lost":
                        vals.append(100.0 * (keep0[m].mean()
                                             - keep[k][m].mean()))
                    else:
                        vals.append(fn(B0[k][te][m] / 1000.0, None))
                f = "{:.2f}" if stat == "hits lost" else "{:.4f}"
                line += "{:>26}".format(" / ".join(f.format(v) for v in vals))
            print(line)
    print("\ncell counts")
    print(f"{'pT [GeV]':<10}" + "".join(
        f"{f'{ETA_EDGES[j]}-{ETA_EDGES[j+1]}':>12}"
        for j in range(len(ETA_EDGES) - 1)))
    for i in range(len(PT_EDGES) - 1):
        hi = "inf" if PT_EDGES[i + 1] > 1e8 else f"{PT_EDGES[i+1]:g}"
        line = f"{PT_EDGES[i]:g}-{hi:<8}"
        for j in range(len(ETA_EDGES) - 1):
            m = ((pt >= PT_EDGES[i]) & (pt < PT_EDGES[i + 1])
                 & (ae >= ETA_EDGES[j]) & (ae < ETA_EDGES[j + 1]))
            line += f"{int(m.sum()):12,}"
        print(line)

    # --- A3: the runtime's own routing condition
    gm = gmask[te]
    table(rows_on(gm), "Gated: |bz - 2 T| > 0.10 T at the source",
          "the population the deployed configuration hands to the network.")
    table(rows_on(~gm), "Not gated: |bz - 2 T| <= 0.10 T at the source",
          "the population that runs the raw core with no correction at all.")

    for lo, hi in zip(ETA_EDGES[:-1], ETA_EDGES[1:]):
        m = (ae >= lo) & (ae < hi) & ~gm
        if m.sum() == 0:
            continue
        table(rows_on(m), f"Not gated, |eta| {lo} to {hi}",
              "raw core, no correction anywhere in these rows.")

    for lo, hi in zip(PT_EDGES[:-1], PT_EDGES[1:]):
        nm = "inf" if hi > 1e8 else f"{hi:g}"
        for gname, gsel in (("Gated", gm), ("Not gated", ~gm)):
            m = (pt >= lo) & (pt < hi) & gsel
            if m.sum() == 0:
                continue
            table(rows_on(m), f"{gname}, pT {lo:g} to {nm} GeV",
                  "same fold, same gate.")

    for lo, hi in zip(ETA_EDGES[:-1], ETA_EDGES[1:]):
        m = (ae >= lo) & (ae < hi) & gm
        if m.sum() == 0:
            continue
        table(rows_on(m), f"Gated, |eta| {lo} to {hi}",
              "the rows the network is switched on for in this band.")

    # how much B actually differs between the cores, split the same way
    for nm, m in (("gated", gm), ("not gated", ~gm)):
        db = np.abs(bz_src[te][m] - 2.0)
        da = np.abs(bzs["avg"][te][m] - bz_src[te][m])
        print(f"\n{nm}: |Bz(src) - 2 T| p50 {np.median(db):.4f} p90 "
              f"{np.percentile(db, 90):.4f} max {db.max():.4f} T   |  "
              f"|Bz(avg) - Bz(src)| p50 {np.median(da):.4f} p90 "
              f"{np.percentile(da, 90):.4f} max {da.max():.4f} T")

    # tail, pooled and gated
    for nm, m in (("whole fold", allm), ("gated", gm), ("not gated", ~gm)):
        print(f"\n|loc0 residual| quantiles [mm], {nm}")
        print(f"{'core':<10}{'p50':>10}{'p90':>10}{'p99':>10}{'max':>12}"
              f"{'>0.5 mm [%]':>14}")
        for k in CORES:
            a = np.abs(B0[k][te][m]) / 1000.0
            print(f"{k:<10}{np.median(a):10.4f}"
                  f"{np.percentile(a, 90):10.4f}{np.percentile(a, 99):10.4f}"
                  f"{a.max():12.4f}{100*(a > 0.5).mean():14.3f}")


if __name__ == "__main__":
    main()
