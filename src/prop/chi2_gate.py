"""Turn the helix bias into the criterion CKF actually applies.

Everything so far has been compared against "the ~200 um measurement scale",
which was a placeholder. The real gate is in the MeasurementSelector: a
candidate hit is kept only if

    chi2 = r^T S^-1 r  <  chi2CutOffMeasurement

with r the residual between the measurement and the predicted state on that
surface, and S = V + H C H^T the innovation covariance. Both numbers are
observable rather than assumed:

  chi2CutOffMeasurement = 15.0   (acts.examples.reconstruction.CkfConfig,
                                  read out of the container build)
  V                     = odd-digi-smearing-config.json, per volume:
                            16,17,18  pixel        loc0 15 um, loc1 15 um
                            23,24,25  short strip  loc0 43 um, loc1 1.2 mm
                            28,29,30  long strip   loc0 72 um  (1D)

The volume is the top byte of the geometry id (kVolumeMask = 0xff<<56,
checked against Acts/Geometry/GeometryIdentifier.hpp; the nine values present
in the teacher set are exactly the nine in the digi config).

C is now measured. On the Windows box the container DOES bind
`acts.examples.root.RootTrackStatesWriter`, so `measure_predicted_cov.py` runs
the production CKF with the track-state writer on and `read_predicted_cov.py`
reduces it to sqrt(C_00) per (pT, |eta|, module class). Pass that table with
--cov and the bracket below collapses to a number. Note that S = V + C then,
not V + sigma_MS + C: the predicted covariance already carries the process
noise the propagator added for material, so adding --ms on top would count
scattering twice.

Without --cov the script still brackets, which is what the Mac container was
limited to:

  oracle : S = V. The propagator is assumed perfect and the only spread is
           the measurement smearing. This is the smallest S possible, so it
           gives the *largest* failure fraction -- an upper bound.
  +MS    : S = V + sigma_MS^2, sigma_MS measured rather than assumed.

sigma_MS comes from a constant-field run with material on. Under a constant
field the helix is exact, so whatever residual survives is material and
nothing else -- the same isolation trick that separated field content from
everything else in the July notes, run the other way round. The delta
columns in the SimHit CSV would have given the per-surface kick directly, but
this build writes them as zeros (checked: identically 0 in both the material
and no-material runs, while the momentum loss along a track is 0.10% median
with material and exactly 0 without, so the interactions are on and it is the
writer that does not fill them).

The real CKF sits above "+MS", because C also carries the track-parameter
uncertainty surviving from earlier hits. Both brackets are therefore
pessimistic and the true failure fraction is below what is printed.

Usage:
    python chi2_gate.py teacher_map_nomat.parquet
    python chi2_gate.py teacher_map_nomat.parquet --ms teacher_const_mat.parquet
    python chi2_gate.py teacher_map_nomat.parquet --cov sigma_C.parquet
"""
import argparse
import math
import json

import numpy as np
import polars as pl

CHI2_CUT = 15.0             # CkfConfig.chi2CutOffMeasurement
N_MC = 20000                # noise draws per jump for the efficiency estimate


def resolutions(path):
    """volume -> (sigma_loc0 [um], sigma_loc1 [um] or None for 1D)."""
    out = {}
    for e in json.load(open(path))["entries"]:
        sm = {s["index"]: s["stddev"] for s in e["value"].get("smearing", [])}
        out[e["volume"]] = (sm[0] * 1000.0,
                            sm[1] * 1000.0 if 1 in sm else None)
    return out


def in_plane_residual(t):
    """Helix miss resolved in the destination surface, in [um].

    On a table that carries the module plane (`res_loc0_um`, from
    `make_teacher_pairs --modules`) this is that miss projected onto the
    MODULE's own local axes. That is the frame the sensor resolutions are
    quoted in, and it is not the cylinder frame: the ODD barrel modules sit
    150 mrad off the cylinder tangent, so the two differ by
    more than a relabelling.

    The cylinder branch below is what a table without the module columns gets.
    It is kept so the older tables still read, and it is worth knowing what it
    reports. Barrel modules are treated as cylinder-like, with an r*phi part
    and a z part; endcap modules as disc-like, with z fixed by construction and
    the second coordinate radial. Both targets are the true next hit's own r1
    or z1, so one of the two coordinates is a structural zero on every jump.
    """
    if "res_loc0_um" in t.columns:
        return t["res_loc0_um"].to_numpy(), t["res_loc1_um"].to_numpy()

    hx, hy, hz = (t[c].to_numpy() for c in ("helix_x", "helix_y", "helix_z"))
    x1, y1, z1 = (t[c].to_numpy() for c in ("x1", "y1", "z1"))
    r1 = t["r1"].to_numpy()
    endcap = t["endcap"].to_numpy()

    dphi = (np.arctan2(hy, hx) - np.arctan2(y1, x1) + np.pi) % (2*np.pi) - np.pi
    b_rphi = r1 * dphi * 1000.0
    b_other = np.where(endcap,
                       (np.hypot(hx, hy) - r1) * 1000.0,      # radial
                       (hz - z1) * 1000.0)                    # z
    return b_rphi, b_other


try:
    from scipy.stats import ncx2
except ImportError:                          # the Mac container has no scipy
    ncx2 = None


def gate(b0, b1, s0, s1, one_d, rng, force_mc=False):
    """Deterministic chi2 of the bias, and the pass fraction under noise.

    The unbiased residual is a draw from N(0, S). Adding a fixed bias b makes
    chi2 noncentral with lambda = sum (b_i/s_i)^2 and 1 or 2 degrees of
    freedom, so the pass fraction is exactly ncx2.cdf(cut, dof, lambda).

    Earlier rounds sampled it with N_MC = 20k draws per jump because the Mac
    container ships no scipy. This box has scipy 1.18, so the exact CDF is
    used when it is importable -- same quantity, no sampling error, and the
    367k-jump teacher set goes from tens of minutes to under a second.
    --mc forces the old path so the two can be compared.
    """
    t0 = b0 / s0
    t1 = np.where(one_d, 0.0, b1 / np.where(one_d, 1.0, s1))
    lam = t0 ** 2 + t1 ** 2

    if ncx2 is not None and not force_mc:
        dof = np.where(one_d, 1, 2)
        return lam, ncx2.cdf(CHI2_CUT, dof, lam)

    # chunked over jumps: the full (N_MC, njumps) draw is tens of GB at the
    # sample sizes here
    keep = np.empty(len(b0))
    step = max(1, 4_000_000 // N_MC)
    for i in range(0, len(b0), step):
        sl = slice(i, i + step)
        chi2 = (rng.standard_normal((N_MC, sl.stop - i
                                     if sl.stop <= len(b0)
                                     else len(b0) - i)) + t0[sl]) ** 2
        chi2 += np.where(one_d[sl], 0.0,
                         (rng.standard_normal(chi2.shape) + t1[sl]) ** 2)
        keep[sl] = (chi2 < CHI2_CUT).mean(axis=0)
    return lam, keep


def check_exact_against_mc(b0, b1, s0, s1, one_d, rng, n=20000):
    """Confirm the CDF path reproduces the sampled one before trusting it."""
    if ncx2 is None:
        return
    idx = rng.choice(len(b0), size=min(n, len(b0)), replace=False)
    a = gate(b0[idx], b1[idx], s0[idx], s1[idx], one_d[idx], rng)[1]
    m = gate(b0[idx], b1[idx], s0[idx], s1[idx], one_d[idx], rng,
             force_mc=True)[1]
    print(f"\nexact ncx2 vs the {N_MC:,}-draw MC on {len(idx):,} jumps: "
          f"mean kept {100*a.mean():.3f}% vs {100*m.mean():.3f}%, "
          f"max |diff| per jump {np.abs(a-m).max():.4f}")


def report(name, t, s0, s1, one_d, b0, b1, rng):
    lam, keep = gate(b0, b1, s0, s1, one_d, rng)

    # baseline: the same surfaces with a perfect propagator, so bias = 0.
    _, keep0 = gate(np.zeros_like(b0), np.zeros_like(b1), s0, s1, one_d, rng)

    print(f"\n### {name}")
    print(f"  bias chi2 (deterministic part, cut = {CHI2_CUT}):"
          f"  median {np.median(lam):8.2f}   p90 {np.percentile(lam,90):10.1f}"
          f"   p99 {np.percentile(lam,99):12.1f}")
    print(f"  fails on bias alone (chi2_bias > {CHI2_CUT}):"
          f" {100*(lam > CHI2_CUT).mean():5.1f}%")
    print(f"  hit kept, helix propagation : {100*keep.mean():5.2f}%")
    print(f"  hit kept, perfect propagation: {100*keep0.mean():5.2f}%"
          f"   -> loss {100*(keep0.mean()-keep.mean()):5.2f} points")

    print(f"  {'|eta|':<10}{'n':>7}{'med chi2':>11}{'fail-on-bias':>14}"
          f"{'kept':>9}")
    ae = t["abs_eta"].to_numpy()
    for lo, hi in [(0, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 3)]:
        m = (ae >= lo) & (ae < hi)
        if m.sum() == 0:
            continue
        print(f"  {lo}-{hi:<8}{m.sum():7d}{np.median(lam[m]):11.2f}"
              f"{100*(lam[m] > CHI2_CUT).mean():13.1f}%{100*keep[m].mean():8.1f}%")

    # per track: a rejected hit is a hole, and it is the hole count that
    # decides whether the track is kept. Which limit applies is a config
    # question -- CkfConfig ships maxPixelHoles/maxStripHoles as None in this
    # build and the ODD chain sets them elsewhere -- so print the
    # distribution rather than a pass/fail against a guessed threshold.
    tr = t["track"].to_numpy()
    order = np.argsort(tr, kind="stable")
    tr_s, miss_s = tr[order], (1.0 - keep)[order]
    edges = np.flatnonzero(np.diff(tr_s)) + 1
    holes = np.array([g.sum() for g in np.split(miss_s, edges)])
    nhit = np.array([len(g) for g in np.split(miss_s, edges)])
    trk_eta = np.array([g[0] for g in np.split(ae[order], edges)])
    print(f"  expected holes/track: mean {holes.mean():.2f} of "
          f"{nhit.mean():.1f} hits", end="")
    print(f"   |  >=1: {100*(holes >= 1).mean():.0f}%"
          f"   >=3: {100*(holes >= 3).mean():.0f}%"
          f"   ({len(holes):,} tracks)")
    fwd = trk_eta > 1.5
    if fwd.any():
        print(f"                        |eta|>1.5 only: mean "
              f"{holes[fwd].mean():.2f} of {nhit[fwd].mean():.1f}"
              f"   >=3: {100*(holes[fwd] >= 3).mean():.0f}%")
    return lam, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("--digi", default="odd-digi-smearing-config.json")
    ap.add_argument("--ms", help="material-on teacher parquet; its per-jump "
                                 "scattering displacement is added to S")
    ap.add_argument("--cov", help="sigma_C table from read_predicted_cov.py; "
                                  "replaces the k-scan with the measurement")
    args = ap.parse_args()

    t = pl.read_parquet(args.pairs)
    res = resolutions(args.digi)

    vol = (t["surf1"].to_numpy() >> 56)
    unknown = set(np.unique(vol).tolist()) - set(res)
    if unknown:
        raise SystemExit(f"destination volumes not in the digi config: {unknown}")

    s0 = np.array([res[v][0] for v in vol])
    s1_raw = [res[v][1] for v in vol]
    one_d = np.array([v is None for v in s1_raw])
    s1 = np.array([1.0 if v is None else v for v in s1_raw])

    b0, b1 = in_plane_residual(t)
    rng = np.random.default_rng(20260730)
    check_exact_against_mc(b0, b1, s0, s1, one_d, rng)

    print(f"{len(t):,} jumps.  destination volumes: "
          + ", ".join(f"{v}:{(vol==v).sum()}" for v in sorted(set(vol.tolist()))))
    print(f"pixel {int((vol<20).sum()):,}  short strip "
          f"{int(((vol>20)&(vol<27)).sum()):,}  long strip "
          f"{int((vol>27).sum()):,}")
    # the single number that replaces "the ~200 um measurement scale": with
    # the bias almost entirely in loc0, a hit fails on the bias alone once
    # |b| > sqrt(cut) * sigma_loc0.
    print("\nbias that fails the gate on its own, sqrt(15) * sigma_loc0:")
    for v in sorted(set(vol.tolist())):
        print(f"    volume {v}: sigma_loc0 {res[v][0]:5.1f} um  ->  "
              f"{math.sqrt(CHI2_CUT)*res[v][0]:6.0f} um   "
              f"({int((vol==v).sum()):,} jumps)")

    print(f"\nhelix bias [um]: |r*phi| median {np.median(np.abs(b0)):.1f}, "
          f"p90 {np.percentile(np.abs(b0),90):.1f}; "
          f"second coord median {np.median(np.abs(b1)):.1f}, "
          f"p90 {np.percentile(np.abs(b1),90):.1f}")

    report("oracle prediction (S = V only) -- upper bound on the loss",
           t, s0, s1, one_d, b0, b1, rng)

    if args.ms:
        add = scattering_sigma(t, pl.read_parquet(args.ms))
        s0m, s1m = np.hypot(s0, add), np.hypot(s1, add)
        report("S = V + scattering (the honest bracket)", t,
               s0m, s1m, one_d, b0, b1, rng)

        # C is the one thing that cannot be measured here, so ask how much
        # of it would
        # be needed to matter. Inflating sigma_loc0 by k stands in for a
        # predicted-position uncertainty of sqrt(k^2-1) * sigma on top.
        print("\nsensitivity to the covariance we cannot measure "
              "(inflate sigma_loc0 by k):")
        print(f"  {'k':>5}{'sigma_loc0 pixel':>20}{'hits lost [pts]':>18}"
              f"{'holes/track':>14}")
        tr = t["track"].to_numpy()
        order = np.argsort(tr, kind="stable")
        edges = np.flatnonzero(np.diff(tr[order])) + 1
        for k in (1, 2, 3, 5, 10, 20):
            _, keep = gate(b0, b1, s0m*k, s1m*k, one_d, rng)
            _, keep0 = gate(np.zeros_like(b0), np.zeros_like(b1),
                            s0m*k, s1m*k, one_d, rng)
            holes = np.array([g.sum() for g in
                              np.split((1.0-keep)[order], edges)])
            print(f"  {k:5d}{15.0*k:18.0f} um"
                  f"{100*(keep0.mean()-keep.mean()):18.2f}{holes.mean():14.2f}")

    if args.cov:
        # S = V + C, not V + sigma_MS + C. The CKF's predicted covariance
        # already contains the process noise the propagator added for
        # material, so folding --ms in as well would count scattering twice.
        c0, c1 = measured_cov(t, vol, args.cov)
        report("S = V + measured C -- the gate the CKF actually applies",
               t, np.hypot(s0, c0), np.hypot(s1, c1), one_d, b0, b1, rng)

        # Second question, from the pull widths: ACTS's pixel C over-covers
        # by 1/0.75, so the gate it applies is wider than its own prediction
        # error justifies. Rescaling to the measured spread says what the
        # helix would cost a filter whose covariance were exact -- which is
        # the relevant number for a learned g_theta, since the point of
        # learning per-hit uncertainty is to make C honest.
        c0c, c1c = measured_cov(t, vol, args.cov, calibrate=True)
        report("S = V + C rescaled to the measured pull width",
               t, np.hypot(s0, c0c), np.hypot(s1, c1c), one_d, b0, b1, rng)


PT_EDGES = np.array([0, 1, 2, 5, 10, 1e9])
ETA_EDGES = np.array([0, .5, 1, 1.5, 2, 3])

# volume -> module class, the same mapping read_predicted_cov.py writes with
CLASS_OF = {16: "pixel", 17: "pixel", 18: "pixel",
            23: "sstrip", 24: "sstrip", 25: "sstrip",
            28: "lstrip", 29: "lstrip", 30: "lstrip"}


def measured_cov(t, vol, path, calibrate=False):
    """sqrt(C_00), sqrt(C_11) per jump, from a CKF measurement.

    The table is keyed on (module class, pT bin, |eta| bin) with the same
    edges used here. Cells the CKF sample did not populate fall back to the
    class median, stored under pt_bin = eta_bin = -1; a jump whose class has
    no entry at all is a bug rather than a fallback, so it raises.
    """
    tab = pl.read_parquet(path)
    cell = {(r["cls"], r["pt_bin"], r["eta_bin"]): (r["sig_c0_um"],
                                                    r["sig_c1_um"])
            for r in tab.iter_rows(named=True)}
    # width of (predicted - truth)/sqrt(C_00) measured on the same states.
    # 1 means C is exactly the spread of its own prediction; the ODD pixels
    # come out at 0.75, i.e. C over-covers there and the gate ACTS applies is
    # correspondingly more permissive than a calibrated one would be.
    width = {r["cls"]: r["pull_width"] for r in tab.iter_rows(named=True)
             if r.get("pull_width") is not None}
    if calibrate and not width:
        raise SystemExit("--calibrate needs a pull_width column; regenerate "
                         "the table with read_predicted_cov.py")

    ip = np.clip(np.digitize(t["pt"].to_numpy(), PT_EDGES) - 1,
                 0, len(PT_EDGES) - 2)
    ie = np.clip(np.digitize(t["abs_eta"].to_numpy(), ETA_EDGES) - 1,
                 0, len(ETA_EDGES) - 2)

    c0 = np.zeros(len(t))
    c1 = np.zeros(len(t))
    nfall = 0
    for i in range(len(t)):
        cls = CLASS_OF[int(vol[i])]
        v = cell.get((cls, int(ip[i]), int(ie[i])))
        if v is None or v[0] is None:
            v = cell.get((cls, -1, -1))
            nfall += 1
            if v is None or v[0] is None:
                raise SystemExit(f"no measured C for class {cls}")
        scale = width.get(cls, 1.0) if calibrate else 1.0
        c0[i] = v[0] * scale
        c1[i] = (v[1] if v[1] is not None else 0.0) * scale

    print(f"\nmeasured C from {path}"
          + ("  (scaled to the measured pull width)" if calibrate else ""))
    print(f"  {'class':8s}{'jumps':>9s}{'median sqrt(C_00)':>20s}"
          f"{'sigma_V':>10s}{'k_eff':>8s}{'pull w':>9s}")
    for cls in ("pixel", "sstrip", "lstrip"):
        m = np.array([CLASS_OF[int(v)] == cls for v in vol])
        if not m.any():
            continue
        sv = {"pixel": 15.0, "sstrip": 43.0, "lstrip": 72.0}[cls]
        print(f"  {cls:8s}{int(m.sum()):9,d}{np.median(c0[m]):20.1f}"
              f"{sv:10.1f}{np.median(np.hypot(sv, c0[m]))/sv:8.2f}"
              f"{width.get(cls, float('nan')):9.2f}")
    if nfall:
        print(f"  ({nfall:,} jumps fell back to the class median because the "
              f"CKF sample did not fill their (pT, |eta|) cell)")
    return c0, c1


def _cell(t):
    a = np.clip(np.digitize(t["pt"].to_numpy(), PT_EDGES) - 1, 0, len(PT_EDGES)-2)
    b = np.clip(np.digitize(t["abs_eta"].to_numpy(), ETA_EDGES) - 1,
                0, len(ETA_EDGES)-2)
    return a * (len(ETA_EDGES) - 1) + b


def scattering_sigma(t, m):
    """Per-jump scattering scale, read off a constant-field material-on run.

    In that run the helix is exact, so |res_rphi| is the material effect over
    one step and nothing else. Its median maps to a Gaussian sigma through
    median|x| = 0.6745 sigma. Binned in (pT, |eta|) because the deflection
    goes as 1/p and a single number would be wrong by a factor of ~30 across
    the sample.
    """
    cm, ct = _cell(m), _cell(t)
    res = np.abs(m["res_rphi_um"].to_numpy())
    sig = np.zeros(len(PT_EDGES[:-1]) * len(ETA_EDGES[:-1]))
    for i in range(len(sig)):
        k = cm == i
        if k.sum() >= 20:
            sig[i] = np.median(res[k]) / 0.6745

    print(f"\nsigma_MS [um] from {len(m):,} constant-field jumps "
          f"(rows pT, cols |eta|):")
    print(f"  {'pT':<10}" + "".join(f"{ETA_EDGES[j]}-{ETA_EDGES[j+1]:<6}"
                                    for j in range(len(ETA_EDGES)-1)))
    for i in range(len(PT_EDGES)-1):
        hi = "inf" if PT_EDGES[i+1] > 1e8 else f"{PT_EDGES[i+1]:g}"
        row = f"  {PT_EDGES[i]:g}-{hi:<8}"
        for j in range(len(ETA_EDGES)-1):
            row += f"{sig[i*(len(ETA_EDGES)-1)+j]:10.1f}"
        print(row)
    missing = (sig[ct] == 0).sum()
    if missing:
        print(f"  ({missing} jumps fall in cells with <20 constant-field "
              f"entries; sigma_MS taken as 0 there, which is pessimistic)")
    return sig[ct]


if __name__ == "__main__":
    main()
