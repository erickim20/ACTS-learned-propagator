"""A2/A3 -- chain the learned transport with a real gate and a real update.

Every number in the talk is one step from a true state. This runs the loop a
CKF actually runs: at each surface, propagate the state and its covariance,
gate on chi2 = r^T S^-1 r with S = H C H^T + V, and Kalman-update against the
accepted hit. The question it answers is whether the update damps the error
faster than the transport grows it, over the ~9 layers a real track crosses.

    e_pred = F e_upd + w            F from jacobian_helix (exact, checked)
    C_pred = F C_upd F^T + Q        Q from qtable (measured, per cell)
    r      = eps - H e_pred         eps ~ N(0, V), the digitisation
    chi2   = r^T S^-1 r  <  15      the MeasurementSelector's own cut
    K      = C_pred H^T S^-1
    e_upd  = e_pred + K r,   C_upd = (I - K H) C_pred

Three things are worth being explicit about, because they bound what this can
claim.

1. **g_theta is not re-run.** It cannot be: the model takes B_z from the ODD
   field map as an input and that map exists only on the WSL box, and no
   weights were saved -- the runs directory holds test-fold dumps, not
   parameters. So the per-step error w is drawn from the measured residual
   pool rather than produced by the network. The consequence is that
   **distribution shift is assumed away**: w is sampled as if the model still
   made the same errors when its input state is no longer the true one. That
   assumption is the single largest caveat here and it is not testable
   without the field map.

2. **The truth noise and the filter's belief are deliberately different.**
   w is resampled from the actual 5-vectors, which keeps the measured tail and
   the measured correlations exactly. The filter's Q is the Gaussian table.
   That mismatch is not an oversight -- it is the thing being measured, since
   a filter can only carry a covariance and A1 showed the residual is not
   Gaussian (RMS/MAD up to 10).

3. **The geometry comes from tracks the Q table was not fitted on.** Q was
   measured on the test fold; the chains run over the training fold, so the
   residual pool and the trajectories are disjoint.

Usage:
    python chain_sim.py                          # A2
    python chain_sim.py --sweep                  # A3, k = 0.5 .. 10
    python chain_sim.py --noise gauss            # what the tail is worth
"""
import argparse

import numpy as np
import polars as pl

from .chi2_gate import (CHI2_CUT, CLASS_OF, ETA_EDGES, PT_EDGES,
                       in_plane_residual, resolutions)
from .jacobian_helix import BARREL_VOLUMES, F_bound
from .qtable import CLASSES, correlations, robust_sigma

UM = 1e-3          # um -> mm, the unit F is written in


# ------------------------------------------------------------------ the folds

def folds(tracks, seed=20260730):
    """Reproduce train_gtheta's split exactly.

    The permutation is the first draw from the generator in that script, so
    the same seed reproduces the same assignment -- which is what makes
    'fitted on the test fold, simulated on the training fold' a real claim
    rather than a hope.
    """
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(tracks))
    n = len(uniq)
    lut = {tr: (0 if i < 0.70 * n else (1 if i < 0.85 * n else 2))
           for i, tr in enumerate(uniq)}
    return np.array([lut[x] for x in tracks])


def cells(pt, eta):
    ip = np.clip(np.digitize(pt, PT_EDGES) - 1, 0, len(PT_EDGES) - 2)
    ie = np.clip(np.digitize(eta, ETA_EDGES) - 1, 0, len(ETA_EDGES) - 2)
    return ip, ie


# ----------------------------------------------------------- the noise source

def noise_pool(dump):
    """The measured per-step 5-vector, indexed by (class, pT bin, |eta| bin).

    Resampling these keeps the tail and the correlation structure that a
    Gaussian draw from the table would throw away -- which A1 measured at
    RMS/MAD up to 10, i.e. not a detail.
    """
    d = np.load(dump)
    cls = np.array([CLASS_OF[int(v)] for v in d["vol"]])
    ip, ie = cells(d["pt"], d["abs_eta"])
    w = np.column_stack([d["rem0"] * UM, d["rem1"] * UM,
                         d["v1_rem"][:, 0], d["v1_rem"][:, 1],
                         d["v1_rem"][:, 2]])
    pool, fallback = {}, {}
    for c in CLASSES:
        m = cls == c
        fallback[c] = w[m]
        for i in range(len(PT_EDGES) - 1):
            for j in range(len(ETA_EDGES) - 1):
                k = m & (ip == i) & (ie == j)
                if k.sum() >= 25:
                    pool[(c, i, j)] = w[k]
    return pool, fallback


def loc1_for_1d(t):
    """Q's loc1 for 1D modules, which the dump cannot supply.

    `train_gtheta` masks the second coordinate out of the loss on long-strip
    surfaces -- they measure one coordinate, so there is nothing to supervise
    -- and zeroes it in the dump. But the filter still CARRIES a loc1
    uncertainty there, and it is the helix's own, uncorrected. That is
    measurable from the teacher table directly, so it is measured rather than
    guessed or borrowed from another class.
    """
    vol = t["surf1"].to_numpy() >> 56
    m = np.array([CLASS_OF[int(v)] == "lstrip" for v in vol])
    _, b1 = in_plane_residual(t)
    ip, ie = cells(t["pt"].to_numpy(), t["abs_eta"].to_numpy())
    out = {}
    for i in range(len(PT_EDGES) - 1):
        for j in range(len(ETA_EDGES) - 1):
            k = m & (ip == i) & (ie == j)
            if k.sum() >= 25:
                out[(i, j)] = robust_sigma(b1[k]) * UM
    out[(-1, -1)] = robust_sigma(b1[m]) * UM
    return out


def filter_Q(tab, corr, l1d):
    """The 5x5 the filter believes, per (class, cell). Q = D R D."""
    out = {}
    for r in tab.iter_rows(named=True):
        c = r["cls"]
        s1 = r["sig_c1_um"]
        if s1 is None:                      # 1D module: measured separately
            s1 = l1d.get((r["pt_bin"], r["eta_bin"]), l1d[(-1, -1)]) / UM
        D = np.diag([r["sig_c0_um"] * UM, s1 * UM,
                     r["sig_phi_mrad"] * 1e-3, r["sig_theta_mrad"] * 1e-3,
                     max(r["sig_qop_rel"], 1e-12)])
        R = np.nan_to_num(corr[c], nan=0.0)
        np.fill_diagonal(R, 1.0)
        out[(c, r["pt_bin"], r["eta_bin"])] = D @ R @ D
    return out


# -------------------------------------------------------------- the occupancy

def occupancy(path, n_layers_fallback=1.0):
    """Hits per mm^2 and per mm of arc, per (volume, layer), from a real
    ColliderML event sample. Branch multiplicity needs a hit density and
    there is no way to get one from a single-muon teacher set."""
    d = pl.read_parquet(path)
    lc = [c for c, ty in zip(d.columns, d.dtypes) if ty == pl.List]
    h = d.explode(lc).with_columns(
        (pl.col("x") ** 2 + pl.col("y") ** 2).sqrt().alias("r"))
    nev = d.height
    g = (h.group_by(["volume_id", "layer_id"])
          .agg(pl.len().alias("n"), pl.col("r").median().alias("r"),
               pl.col("r").min().alias("rmin"), pl.col("r").max().alias("rmax"),
               pl.col("z").min().alias("zmin"), pl.col("z").max().alias("zmax")))
    rho_a, rho_l = {}, {}
    for row in g.iter_rows(named=True):
        v, la = int(row["volume_id"]), int(row["layer_id"])
        per_ev = row["n"] / nev
        if v in BARREL_VOLUMES:
            area = 2 * np.pi * row["r"] * max(row["zmax"] - row["zmin"], 1.0)
        else:
            area = np.pi * max(row["rmax"] ** 2 - row["rmin"] ** 2, 1.0)
        rho_a[(v, la)] = per_ev / area                 # hits / mm^2
        rho_l[(v, la)] = per_ev / (2 * np.pi * row["r"])   # hits / mm of arc
    return rho_a, rho_l, nev


# --------------------------------------------------------------- the chain

def run(t, Fs, Qf, pool, fb, rho_a, rho_l, args, kQ=1.0, rng=None):
    rng = rng or np.random.default_rng(args.seed)
    res = resolutions(args.digi)

    tracks = t["track"].to_numpy()
    surf0 = t["surf0"].to_numpy()
    surf1 = t["surf1"].to_numpy()
    vol = surf1 >> 56
    lay = (surf1 >> 36) & 0xFFF
    cls = np.array([CLASS_OF[int(v)] for v in vol])
    ip, ie = cells(t["pt"].to_numpy(), t["abs_eta"].to_numpy())

    V0 = np.array([res[int(v)][0] for v in vol]) * UM
    v1r = [res[int(v)][1] for v in vol]
    one_d = np.array([q is None for q in v1r])
    V1 = np.array([1.0 if q is None else q for q in v1r]) * UM

    # segments: a chain is only a chain while the destination of one jump is
    # the source of the next. 23% of consecutive rows do not link, because
    # the pair builder drops overlaps and unsolvable jumps -- treating those
    # as chained would invent a step that never happened.
    link = np.zeros(len(t), dtype=bool)
    link[1:] = (tracks[1:] == tracks[:-1]) & (surf0[1:] == surf1[:-1])
    seg_start = np.flatnonzero(~link)
    seg_end = np.append(seg_start[1:], len(t))

    H2 = np.zeros((2, 5))
    H2[0, 0] = H2[1, 1] = 1.0
    H1 = H2[:1]

    n_seg = 0
    holes, nhit, seglen = [], [], []
    by_depth = [[] for _ in range(args.max_depth)]      # |loc0 residual| [um]
    pull_depth = [[] for _ in range(args.max_depth)]
    hole_depth = [[] for _ in range(args.max_depth)]
    branch_depth = [[] for _ in range(args.max_depth)]
    chi2_all = []

    for a, b in zip(seg_start, seg_end):
        if b - a < args.min_len:
            continue
        n_seg += 1
        e = np.zeros(5)
        C = np.diag(args.c0) ** 2
        nh = 0
        for d, i in enumerate(range(a, b)):
            key = (str(cls[i]), int(ip[i]), int(ie[i]))
            Q = Qf.get(key)
            if Q is None:
                Q = Qf[(str(cls[i]), -1, -1)]
            F = Fs[i]

            # truth: the transport error this step actually adds
            w = pool.get(key)
            if w is None:
                w = fb[str(cls[i])]
            if args.noise == "gauss":
                wk = rng.multivariate_normal(np.zeros(5), Q)
            else:
                wk = w[rng.integers(len(w))]

            e = F @ e + wk
            C = F @ C @ F.T + (kQ ** 2) * Q

            H = H1 if one_d[i] else H2
            V = np.diag([V0[i] ** 2]) if one_d[i] \
                else np.diag([V0[i] ** 2, V1[i] ** 2])
            eps = rng.normal(0, np.sqrt(np.diag(V)))
            r = eps - H @ e
            S = H @ C @ H.T + V
            Si = np.linalg.inv(S)
            chi2 = float(r @ Si @ r)
            chi2_all.append(chi2)

            if d < args.max_depth:
                by_depth[d].append(abs(e[0]) / UM)
                pull_depth[d].append(r[0] / np.sqrt(S[0, 0]))
                # candidates the gate lets through: the true hit plus whatever
                # of the event's own hits fall inside the chi2 contour
                # recorded as FAKES, not as candidates. The true hit is one
                # per layer whatever the gate does, so quoting 1 + fakes
                # buries the only part that responds to Q: at k = 0.5 -> 10
                # the candidate count moves 10.3 -> 13.5 and looks flat, while
                # the fakes inside it move 1.1 -> 4.3.
                if one_d[i]:
                    n_fake = (rho_l.get((int(vol[i]), int(lay[i])), 0.0)
                              * 2 * np.sqrt(CHI2_CUT * S[0, 0]))
                else:
                    n_fake = (rho_a.get((int(vol[i]), int(lay[i])), 0.0)
                              * np.pi * CHI2_CUT * np.sqrt(np.linalg.det(S)))
                branch_depth[d].append(n_fake)

            if chi2 < CHI2_CUT:
                K = C @ H.T @ Si
                e = e + K @ r
                C = C - K @ H @ C
                if d < args.max_depth:
                    hole_depth[d].append(0.0)
            else:
                nh += 1
                if d < args.max_depth:
                    hole_depth[d].append(1.0)
        holes.append(nh)
        nhit.append(b - a)
        seglen.append(b - a)

    return dict(
        n_seg=n_seg, holes=np.array(holes), nhit=np.array(nhit),
        by_depth=[np.array(x) for x in by_depth],
        pull_depth=[np.array(x) for x in pull_depth],
        hole_depth=[np.array(x) for x in hole_depth],
        branch_depth=[np.array(x) for x in branch_depth],
        chi2=np.array(chi2_all))


def report(R, label):
    h, n = R["holes"], R["nhit"]
    print(f"\n### {label}")
    print(f"  {R['n_seg']:,} chains, {n.mean():.2f} surfaces each "
          f"({n.sum():,} jumps)")
    print(f"  holes/track {h.mean():.3f}   >=1: {100*(h>=1).mean():.1f}%   "
          f">=3: {100*(h>=3).mean():.1f}%")
    print(f"  {'layer':>6}{'|loc0 err| med [um]':>22}{'p90':>10}"
          f"{'pull w':>9}{'hole rate':>11}{'branches':>10}{'n':>8}")
    for d, (x, p, hd, br) in enumerate(zip(R["by_depth"], R["pull_depth"],
                                           R["hole_depth"],
                                           R["branch_depth"])):
        if len(x) < 20:
            break
        print(f"  {d:6d}{np.median(x):22.1f}{np.percentile(x, 90):10.1f}"
              f"{robust_sigma(p):9.2f}{100*hd.mean():10.1f}%"
              f"{np.mean(br):10.2f}{len(x):8,d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="teacher_map_mat_logpt.parquet")
    ap.add_argument("--dump", default="runs/mat.npz")
    ap.add_argument("--qtable", default="Q_table.parquet")
    ap.add_argument("--digi", default="odd-digi-smearing-config.json")
    ap.add_argument("--hits",
                    default="sim_out/runs/ch2/tracker_hits/"
                            "tracker_hits_000000-001000.parquet")
    ap.add_argument("--bz", type=float, default=2.0)
    ap.add_argument("--tracks", type=int, default=3000)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--max-depth", type=int, default=12)
    ap.add_argument("--noise", default="empirical",
                    choices=["empirical", "gauss"])
    ap.add_argument("--c0", type=float, nargs=5,
                    default=[0.0, 0.0, 0.0, 0.0, 0.0],
                    help="seed covariance as five sigmas (mm, mm, rad, rad, "
                         "1/GeV). Zero means the chain starts from a perfect "
                         "state, which is optimistic and deliberate: it "
                         "isolates what the chain itself adds.")
    ap.add_argument("--sweep", action="store_true",
                    help="A3: scale the filter's Q and read off holes and "
                         "branches")
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--out", default="chain_sim.npz")
    args = ap.parse_args()

    t = pl.read_parquet(args.pairs)
    fold = folds(t["track"].to_numpy())
    t = t.filter(pl.Series(fold == 0))          # training fold: Q never saw it
    uniq = np.unique(t["track"].to_numpy())
    if args.tracks < len(uniq):
        rng0 = np.random.default_rng(args.seed)
        keep = set(rng0.choice(uniq, args.tracks, replace=False).tolist())
        t = t.filter(pl.Series([x in keep for x in t["track"].to_numpy()]))
    print(f"{len(t):,} jumps over {t['track'].n_unique():,} tracks "
          f"(training fold only)")

    u = np.column_stack([t[c].to_numpy() for c in
                         ("x", "y", "z", "px", "py", "pz")])
    q = np.sign(t["qop"].to_numpy())
    endcap = t["endcap"].to_numpy()
    src_ec = ~np.isin(t["surf0"].to_numpy() >> 56, BARREL_VOLUMES)
    Fs, _, ok, _ = F_bound(u, t["r1"].to_numpy(), t["z1"].to_numpy(),
                           endcap, q, args.bz, src_endcap=src_ec)
    print(f"  F built for every jump ({int((~ok).sum())} without a helix "
          f"solution)")

    tab = pl.read_parquet(args.qtable)
    d = np.load(args.dump)
    D = dict(rem0=d["rem0"], rem1=d["rem1"], one_d=d["one_d"],
             dphi=d["v1_rem"][:, 0], dtheta=d["v1_rem"][:, 1],
             dqop=d["v1_rem"][:, 2],
             cls=np.array([CLASS_OF[int(v)] for v in d["vol"]]))
    _, corr = correlations(D)
    Qf = filter_Q(tab, corr, loc1_for_1d(t))
    pool, fb = noise_pool(args.dump)
    rho_a, rho_l, nev = occupancy(args.hits)
    print(f"  occupancy from {nev} ColliderML events, "
          f"{len(rho_a)} (volume, layer) cells")

    if not args.sweep:
        R = run(t, Fs, Qf, pool, fb, rho_a, rho_l, args)
        report(R, f"chained, noise = {args.noise}")
        np.savez_compressed(
            args.out, holes=R["holes"], nhit=R["nhit"],
            depth_med=np.array([np.median(x) if len(x) else np.nan
                                for x in R["by_depth"]]),
            depth_p90=np.array([np.percentile(x, 90) if len(x) else np.nan
                                for x in R["by_depth"]]),
            depth_pull=np.array([robust_sigma(x) if len(x) else np.nan
                                 for x in R["pull_depth"]]),
            depth_hole=np.array([x.mean() if len(x) else np.nan
                                 for x in R["hole_depth"]]),
            depth_branch=np.array([x.mean() if len(x) else np.nan
                                   for x in R["branch_depth"]]),
            depth_n=np.array([len(x) for x in R["by_depth"]]))
        print(f"\nwrote {args.out}")
        return

    print("\n=== A3: scale the filter's Q ===")
    print("  Q too small -> gate too tight -> true hits rejected -> holes up")
    print("  Q too large -> gate too wide  -> wrong hits accepted -> fakes up")
    print(f"\n  {'k':>6}{'holes/track':>14}{'>=3 holes':>12}"
          f"{'fakes/track':>14}{'final |err| p90':>18}")
    rows = []
    for k in (0.5, 1.0, 2.0, 5.0, 10.0):
        R = run(t, Fs, Qf, pool, fb, rho_a, rho_l, args, kQ=k,
                rng=np.random.default_rng(args.seed))
        br = np.array([x.mean() for x in R["branch_depth"] if len(x) >= 20])
        last = [x for x in R["by_depth"] if len(x) >= 20][-1]
        rows.append((k, R["holes"].mean(), (R["holes"] >= 3).mean(),
                     br.sum(), np.percentile(last, 90)))
        print(f"  {k:6.1f}{rows[-1][1]:14.3f}{100*rows[-1][2]:11.1f}%"
              f"{rows[-1][3]:14.2f}{rows[-1][4]:18.1f}")
    np.savez_compressed("chain_sweep.npz", sweep=np.array(rows))
    print("\n  fakes are counted against a pileup-10 ttbar occupancy, so they "
          "are a FLOOR:\n  the quantity that decides the trade-off is the one "
          "most sensitive to running\n  this at the pileup the experiment "
          "actually has.")
    print("wrote chain_sweep.npz")


if __name__ == "__main__":
    main()
