"""The process noise Q of the learned propagator, measured from its residuals.

`C_pred = F C_in F^T + Q`. This script measures the second term and nothing
else, which is possible right now for a reason that will stop being true the
moment anything is chained:

    every jump in the teacher set starts from a true state, so C_in = 0 and
    the spread of the surviving residual is the contribution of this one step.

Source is the test-fold dump `train_gtheta.py --dump` writes, not
`perjump_residuals.parquet`. That file is a helix-only study: barrel only,
B = 2 T, no |eta| column, and it predates g_theta.
The dumps carry the quantity this needs: `rem0`/`rem1`, the in-surface
position residual left AFTER g_theta, on the held-out fold.

Q is not taken from RKN's predicted covariance, and not from `sigma_C.parquet`
either. Both describe a different estimator's uncertainty. `sigma_C.parquet`
is still read, as the thing to compare against: it is what ACTS's own C is,
and the ratio Q/C says whether the learned transport is tighter or looser than
the one it would replace.

Five components, not one. The dumps run with `--v1` also carry the direction
and q/p residuals, so Q comes out as a full 5x5 in

    (loc0, loc1, phi, theta, q/p)

and the off-diagonal is the part that matters for chaining: a Kalman update
against a position measurement can only pull the direction back if C has
position-direction correlation. A diagonal Q says it cannot, and the chain
then has no mechanism to damp a direction error at all.

Two width estimators are reported side by side and the difference is the
point. 1.4826 * MAD is what a Gaussian filter wants; the RMS is what the tail
actually is. Where they disagree by a lot, the residual is not Gaussian and a
single sigma is a lie the gate will pay for.

Usage:
    python qtable.py                      # runs/mat.npz, the deployed case
    python qtable.py --dump runs/nomat.npz --out Q_nomat.parquet
"""
import argparse

import numpy as np
import polars as pl

from .chi2_gate import (CHI2_CUT, CLASS_OF, ETA_EDGES, PT_EDGES, gate,
                       resolutions)

CLASSES = ("pixel", "sstrip", "lstrip")
SIGMA_V = {"pixel": 15.0, "sstrip": 43.0, "lstrip": 72.0}   # digi loc0, um
NMIN = 40          # cells below this fall back to the class value
MAD_K = 1.4826     # MAD -> Gaussian sigma

# The band a branch's pull width has to be inside for the table to be
# deployable. `pull_by_branch` says what the pull is. On the deployed model's
# own dump the five rows the network transported run 0.954 to 1.335, so the
# band is a factor of four either side of one: the nearest of those rows sits
# 3.8 times inside the near edge and 3.0 times inside the far one. A factor of
# four on the width is sixteen on the variance, which is deliberately loose.
# The gate is here to catch a covariance wrong by orders of magnitude, which is
# what a table armed on a transport it was not measured on is, and not to
# referee a calibration. It is not a band chosen to pass: it rejects the
# deployed model's own one-branch table on two of its ten rows.
PULL_BAND = (0.25, 4.0)
PULL_NMIN = 200    # a width below this many jumps is not read


def robust_sigma(x):
    """1.4826 * MAD about the median. Insensitive to the tail the RMS chases."""
    if len(x) == 0:
        return np.nan
    return MAD_K * np.median(np.abs(x - np.median(x)))


def widths(x):
    """(sigma_MAD, sigma_RMS, bias, n) for one sample."""
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    return robust_sigma(x), float(np.std(x)), float(np.median(x)), len(x)


def load(path):
    d = np.load(path)
    have_v1 = "v1_rem" in d.files
    out = dict(
        rem0=d["rem0"], rem1=d["rem1"], one_d=d["one_d"], vol=d["vol"],
        pt=d["pt"], eta=d["abs_eta"], track=d["track"], endcap=d["endcap"],
        b0=d["b0"], b1=d["b1"], s0=d["s0"], s1=d["s1"],
    )
    out["cls"] = np.array([CLASS_OF[int(v)] for v in out["vol"]])
    if have_v1:
        # v1 columns are (dphi, dtheta, dq/p) in rad, rad and absolute q/p
        out["dphi"] = d["v1_rem"][:, 0]
        out["dtheta"] = d["v1_rem"][:, 1]
        out["dqop"] = d["v1_rem"][:, 2]
        out["arm"] = float(d["arm"])
    return out, have_v1


def cell_index(pt, eta):
    ip = np.clip(np.digitize(pt, PT_EDGES) - 1, 0, len(PT_EDGES) - 2)
    ie = np.clip(np.digitize(eta, ETA_EDGES) - 1, 0, len(ETA_EDGES) - 2)
    return ip, ie


def build_table(D, have_v1):
    """One row per (class, pT bin, |eta| bin), plus a (-1, -1) class fallback.

    The fallback rows are not decoration: `chi2_gate.measured_cov` looks for
    pt_bin = eta_bin = -1 when a cell is empty, so a table without them raises
    on the first sparse cell.
    """
    ip, ie = cell_index(D["pt"], D["eta"])
    rows = []
    for cls in CLASSES:
        m_cls = D["cls"] == cls
        if not m_cls.any():
            continue
        # 1D modules never measure the second coordinate, so its residual is
        # a structural zero rather than a small number
        one_d = bool(D["one_d"][m_cls].all())

        cells = [(-1, -1, m_cls)]
        for i in range(len(PT_EDGES) - 1):
            for j in range(len(ETA_EDGES) - 1):
                cells.append((i, j, m_cls & (ip == i) & (ie == j)))

        for i, j, m in cells:
            n = int(m.sum())
            if i >= 0 and n < NMIN:
                continue
            s0_mad, s0_rms, b0, _ = widths(D["rem0"][m])
            if one_d:
                s1_mad = s1_rms = b1 = None
            else:
                k = m & ~D["one_d"]
                s1_mad, s1_rms, b1, _ = widths(D["rem1"][k])
            r = dict(cls=cls, pt_bin=i, eta_bin=j, n=n,
                     sig_c0_um=float(s0_mad), sig_c1_um=s1_mad,
                     rms_c0_um=float(s0_rms), rms_c1_um=s1_rms,
                     bias_c0_um=float(b0), bias_c1_um=b1,
                     pull_width=1.0)
            if have_v1:
                r["sig_phi_mrad"] = float(robust_sigma(D["dphi"][m]) * 1e3)
                r["sig_theta_mrad"] = float(robust_sigma(D["dtheta"][m]) * 1e3)
                r["sig_qop_rel"] = float(robust_sigma(D["dqop"][m]))
            rows.append(r)
    return pl.DataFrame(rows)


def correlations(D):
    """The 5x5 correlation of the per-step residual, per module class.

    Pooled over cells rather than measured per cell: a correlation needs many
    more samples than a width does, and the question it answers -- can a
    position update reach the direction -- is not a per-cell question.
    """
    names = ["loc0", "loc1", "phi", "theta", "q/p"]
    out = {}
    for cls in CLASSES:
        m = D["cls"] == cls
        one_d = bool(D["one_d"][m].all())
        if not one_d:
            m = m & ~D["one_d"]
        cols = [D["rem0"][m], D["rem1"][m], D["dphi"][m], D["dtheta"][m],
                D["dqop"][m]]
        M = np.vstack(cols)
        # loc1 of a 1D module is a structural zero, so it has no correlation
        # to report rather than a zero one -- drop the row and put NaN back
        keep = M.std(1) > 0
        R = np.full((5, 5), np.nan)
        if keep.sum() > 1:
            sub = np.corrcoef(M[keep])
            idx = np.flatnonzero(keep)
            R[np.ix_(idx, idx)] = sub
        out[cls] = R
    return names, out


def compare_with_acts(D, tab, acts_path="sigma_C.parquet"):
    """Q (learned, per step, C_in = 0) against C (ACTS, accumulated). Not equal
    quantities, because ACTS's C carries everything surviving from earlier
    hits, so the ratio is a bound: anywhere Q > C the single step is already
    worse than the filter's whole accumulated uncertainty."""
    a = pl.read_parquet(acts_path)
    print(f"\n{'class':8}{'jumps':>9}{'sigma_V':>10}{'Q ours':>10}"
          f"{'C ACTS':>10}{'Q/C':>8}")
    for cls in CLASSES:
        q = tab.filter((pl.col("cls") == cls) & (pl.col("pt_bin") == -1))
        c = a.filter((pl.col("cls") == cls) & (pl.col("pt_bin") == -1))
        if c.height == 0:                     # no fallback row in that table
            c = a.filter(pl.col("cls") == cls).select(
                pl.col("sig_c0_um").median().alias("sig_c0_um"))
        if q.height == 0:
            continue
        qv, cv = q["sig_c0_um"][0], c["sig_c0_um"][0]
        print(f"{cls:8}{int((D['cls'] == cls).sum()):9,d}"
              f"{SIGMA_V[cls]:10.1f}{qv:10.1f}{cv:10.1f}{qv / cv:8.2f}")


def pull_and_gate(D, tab, digi="odd-digi-smearing-config.json"):
    """Scale Q by k and read off the pull width and what the gate does.

    The pull is (residual + measurement noise) / sqrt(V + k^2 Q). At k = 1 it
    is 1 by construction IF the residual is Gaussian and the MAD estimator is
    right, so the number that carries information is how far off 1 it lands --
    that gap is the tail, and it is the same tail the gate charges for.

    This is the one-step version of A3. The chained version is chain_sim.py;
    the point of doing it here is that it costs nothing and gives A3 a zero.
    """
    res = resolutions(digi)
    v0 = np.array([res[int(v)][0] for v in D["vol"]])
    v1r = [res[int(v)][1] for v in D["vol"]]
    v1 = np.array([1.0 if q is None else q for q in v1r])

    # per-jump Q from the table, cell first, class fallback second
    look = {(r["cls"], r["pt_bin"], r["eta_bin"]): (r["sig_c0_um"],
                                                    r["sig_c1_um"])
            for r in tab.iter_rows(named=True)}
    ip, ie = cell_index(D["pt"], D["eta"])
    q0 = np.empty(len(ip))
    q1 = np.empty(len(ip))
    for i in range(len(ip)):
        c = D["cls"][i]
        v = look.get((c, int(ip[i]), int(ie[i]))) or look[(c, -1, -1)]
        q0[i] = v[0]
        q1[i] = v[1] if v[1] is not None else 0.0

    rng = np.random.default_rng(20260806)
    tr = D["track"]
    order = np.argsort(tr, kind="stable")
    edges = np.flatnonzero(np.diff(tr[order])) + 1

    print(f"\npull and gate under S = V + (k Q)^2, one step from truth")
    print(f"  {'k':>5}{'pull w loc0':>13}{'chi2>15 on bias':>18}"
          f"{'hits lost [pts]':>18}{'holes/track':>14}")
    curve = []
    for k in (0.5, 1.0, 2.0, 5.0, 10.0):
        s0 = np.hypot(v0, k * q0)
        s1 = np.hypot(v1, k * q1)
        # the innovation the filter sees: the leftover bias plus the smearing
        inn0 = D["rem0"] + rng.normal(0, v0)
        pull = robust_sigma(inn0 / s0)
        lam, keep = gate(D["rem0"], D["rem1"], s0, s1, D["one_d"], rng)
        _, keep0 = gate(np.zeros(len(s0)), np.zeros(len(s0)), s0, s1,
                        D["one_d"], rng)
        holes = np.array([g.sum() for g in
                          np.split((1.0 - keep)[order], edges)]).mean()
        lost = 100 * (keep0.mean() - keep.mean())
        print(f"  {k:5.1f}{pull:13.3f}{100 * (lam > CHI2_CUT).mean():17.2f}%"
              f"{lost:18.2f}{holes:14.3f}")
        curve.append((k, pull, float((lam > CHI2_CUT).mean()), lost, holes))
    return np.array(curve)


def pull_by_branch(D, tab):
    """The pull of each transport this table will be armed on, per class.

    `armNoise` arms one table on both branches unless the file carries a second
    measurement, so a one-branch table is a covariance for two different
    transports: the network's, above the field gate, and the helix's own, at or
    below it. `rem0`/`rem1` are the residual the first leaves and `b0`/`b1` are
    the miss the second leaves, and the dump carries both on the same jumps.

    The pull is the miss over the sigma this table writes for that jump's cell,
    which is Q alone. It is 1 when the
    table describes the transport and it is 46 when the table was measured on
    the other one.
    """
    look = {(r["cls"], r["pt_bin"], r["eta_bin"]): (r["sig_c0_um"],
                                                    r["sig_c1_um"])
            for r in tab.iter_rows(named=True)}
    ip, ie = cell_index(D["pt"], D["eta"])
    q0 = np.empty(len(ip))
    q1 = np.empty(len(ip))
    for i in range(len(ip)):
        c = D["cls"][i]
        v = look.get((c, int(ip[i]), int(ie[i]))) or look[(c, -1, -1)]
        q0[i] = v[0]
        q1[i] = v[1] if v[1] is not None else 0.0

    rows = []
    for cls in CLASSES:
        m_cls = D["cls"] == cls
        for branch, (r0, r1) in (("network", (D["rem0"], D["rem1"])),
                                 ("helix alone", (D["b0"], D["b1"]))):
            for coord, miss, sig, extra in (("loc0", r0, q0, m_cls),
                                            ("loc1", r1, q1,
                                             m_cls & ~D["one_d"])):
                k = extra & (sig > 0)
                n = int(k.sum())
                if n < PULL_NMIN:
                    continue
                rows.append(dict(cls=cls, branch=branch, coord=coord, n=n,
                                 width=float(robust_sigma(miss[k] / sig[k]))))
    return rows


def pull_gate(rows, band):
    """Fail rather than print when a branch's pull width leaves the band.

    One measured mismatch is a pixel loc1 pull of 46 where the helix runs
    alone, against 0.39 where the network fires, on a table that
    loaded, passed every check the format had, and was armed for six weeks. The
    format's checks were binning, branch count and material, and all three were
    right. This is the check that was missing.
    """
    lo, hi = band
    print(f"\n=== pull per branch, miss over the sigma this table writes ===")
    print(f"  {'class':8}{'branch':14}{'coord':7}{'jumps':>9}{'pull':>10}"
          f"{'':4}")
    bad = []
    for r in rows:
        ok = lo <= r["width"] <= hi
        if not ok:
            bad.append(r)
        print(f"  {r['cls']:8}{r['branch']:14}{r['coord']:7}{r['n']:9,d}"
              f"{r['width']:10.3f}    {'' if ok else 'OUTSIDE THE BAND'}")
    print(f"  band {lo} to {hi}, on the {len(rows)} rows with at least "
          f"{PULL_NMIN} jumps")
    if bad:
        names = ", ".join(f"{r['cls']} {r['branch']} {r['coord']} "
                          f"{r['width']:.3f}" for r in bad)
        raise SystemExit(
            f"pull gate: {len(bad)} of {len(rows)} rows outside "
            f"[{lo}, {hi}] -- {names}.\n"
            "This table is armed on both transports. A pull far from 1 on one "
            "of them is a covariance measured on the other, which is the "
            "known defect. Measure the second branch and export "
            "it with export_qtable --helix-table, or do not deploy this table.")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="runs/mat.npz")
    ap.add_argument("--out", default="Q_table.parquet")
    ap.add_argument("--acts", default="sigma_C.parquet")
    ap.add_argument("--digi", default="odd-digi-smearing-config.json")
    ap.add_argument("--pull-band", nargs=2, type=float, default=PULL_BAND,
                    metavar=("LO", "HI"),
                    help="the band a branch's pull width has to be inside. "
                         f"Default {PULL_BAND[0]} to {PULL_BAND[1]}.")
    args = ap.parse_args()

    D, have_v1 = load(args.dump)
    print(f"{args.dump}: {len(D['rem0']):,} held-out jumps, "
          f"{len(np.unique(D['track'])):,} tracks, "
          f"v1 outputs {'present' if have_v1 else 'ABSENT'}")

    tab = build_table(D, have_v1)
    tab.write_parquet(args.out)

    print(f"\n=== per-step Q, sigma of the residual after helix + g_theta "
          f"[um] ===")
    print(f"  {'class':8}{'n':>8}{'MAD sigma':>12}{'RMS':>10}{'RMS/MAD':>10}"
          f"{'median':>10}{'p99':>10}")
    for cls in CLASSES:
        m = D["cls"] == cls
        s_mad, s_rms, bias, n = widths(D["rem0"][m])
        print(f"  {cls:8}{n:8,d}{s_mad:12.1f}{s_rms:10.1f}"
              f"{s_rms / s_mad:10.2f}{bias:10.1f}"
              f"{np.percentile(np.abs(D['rem0'][m]), 99):10.1f}")
    print("  RMS/MAD far above 1 means the tail, not the core, sets the "
          "variance --\n  a single sigma then over-covers the bulk and still "
          "under-covers the tail.")

    if have_v1:
        print(f"\n=== the other three components (median step {D['arm']:.0f} "
              f"mm as lever arm) ===")
        for nm, x, u in (("phi   [mrad]", D["dphi"], 1e3),
                         ("theta [mrad]", D["dtheta"], 1e3),
                         ("q/p   [rel] ", D["dqop"], 1.0)):
            s = robust_sigma(x) * u
            extra = (f"   -> {robust_sigma(x) * D['arm'] * 1e3:8.1f} um over "
                     f"the next step" if u == 1e3 else "")
            print(f"  {nm}  sigma {s:10.4f}{extra}")

        names, C = correlations(D)
        # Saved, not only printed. `export_qtable.py` needs it to build the
        # 5x5 the stepper adds, and without it Q goes into ACTS diagonal --
        # which drops the -0.86 position-direction term that is the chain's
        # only brake (§2.6). Printing a load-bearing number and not writing it
        # down is how it gets dropped.
        corr_out = args.out.replace(".parquet", "_corr.npz")
        np.savez(corr_out, **{cls: C[cls] for cls in CLASSES if cls in C})
        print(f"\n=== correlation of the per-step residual ===")
        print(f"  written to {corr_out}")
        print("  A position update can only correct the direction through "
              "these off-diagonals.")
        for cls in CLASSES:
            print(f"\n  {cls}")
            print("       " + "".join(f"{n:>9}" for n in names))
            for i, n in enumerate(names):
                print(f"  {n:>5}" + "".join(
                    f"{C[cls][i, j]:9.2f}" for j in range(5)))

    compare_with_acts(D, tab, args.acts)
    curve = pull_and_gate(D, tab, args.digi)
    np.savez(args.out.replace(".parquet", "_pull.npz"), curve=curve)
    print(f"\nwrote {args.out} ({tab.height} rows) and "
          f"{args.out.replace('.parquet', '_pull.npz')}")
    # Last, so the table and the curve are on disk to look at when it fails.
    # It raises: a table that does not cover a transport it will be armed on is
    # not a table with a caveat, it is a covariance for something else.
    pull_gate(pull_by_branch(D, tab), tuple(args.pull_band))


if __name__ == "__main__":
    main()
