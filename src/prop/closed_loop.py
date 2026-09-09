"""Closed-loop chaining: feed g_theta its own filtered state.

`chain_sim.py` assumed distribution shift away. It had to: the model takes B_z
from the ODD field map as an input, the map was believed to be Windows-only,
and no weights were ever saved -- so the per-step error was *resampled* from
the measured pool as if the model still made the same errors when its input is
no longer the true state. That assumption was the largest caveat on the
chained measurement and this file removes it.

The map turned out to ship inside the ODD image that is already on this
machine (`build_fieldmap.py`), and `train_gtheta.py --save` now writes the
parameters, so g_theta can be run on a state it has never seen -- which is the
entire point. The loop is what a CKF does:

    helix from the CURRENT (wrong) state onto the named surface
    + g_theta(features of the CURRENT state)          <- the shifted input
    -> predicted state, gate, Kalman update, repeat

**The verification comes first.** Re-implementing `features()` in numpy is the
one place a silent mismatch would invalidate everything downstream, and it
would look like a physics result rather than a bug. So `--verify` runs the same
loop with the state forced back to truth at every step, and checks that the
residual it produces reproduces the training dump's `rem0` jump for jump. If
that does not agree to roundoff, nothing else here is worth reading.

Two quantities come out that `chain_sim.py` could not produce:

  w_closed   the error this step adds, with the propagated input error
             removed: (prediction - truth) - F (input error). Comparing its
             width to the from-truth residual is the distribution shift.
  holes      the same currency as everything else, now without the assumption

Usage:
    python closed_loop.py --verify
    python closed_loop.py --tracks 1500
"""
import argparse

import numpy as np
import polars as pl

from .chi2_gate import (CHI2_CUT, CLASS_OF, ETA_EDGES, PT_EDGES, resolutions)
from .chain_sim import cells, filter_Q, folds, loc1_for_1d, occupancy
from .helix_variants import FieldMap
from .jacobian_helix import (BARREL_VOLUMES, F_bound, bound_in, bound_out, frame,
                            helix_state, solve_s)
from .make_teacher_pairs import plane_predict
from .qtable import correlations, robust_sigma
from .train_gtheta import ACTS, layer_id

UM = 1e-3


# ------------------------------------------------------------------ the model

class Model:
    """The saved network, plus the arithmetic that turns its six raw outputs
    back into a corrected state."""

    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.W = [d[f"W{i}"] for i in range(3)]
        self.b = [d[f"b{i}"] for i in range(3)]
        self.mu, self.sd = d["mu"], d["sd"]
        self.v1_scale = d["v1_scale"]
        self.act = ACTS[str(d["act"][0])]
        self.field = str(d["field"][0])
        self.cov = str(d["cov"][0])
        self.digi = str(d["digi"][0])
        self.nin = int(d["meta"][1])

    def forward(self, x):
        h1, _ = self.act(x @ self.W[0] + self.b[0])
        h2, _ = self.act(h1 @ self.W[1] + self.b[1])
        return h2 @ self.W[2] + self.b[2]

    def forward_hidden(self, x):
        """Outputs and the last hidden layer, which is what the sigma head
        reads. Returned together because the loop needs both and computing
        them twice would misrepresent the cost."""
        h1, _ = self.act(x @ self.W[0] + self.b[0])
        h2, _ = self.act(h1 @ self.W[1] + self.b[1])
        return h2 @ self.W[2] + self.b[2], h2


def features_np(u, qop, r1, z1, endcap, hpos, fm):
    """`train_gtheta.features`, cylinder branch, on arrays.

    Kept for the older tables and for `--verify`. The deployable path is
    `features_plane` below, and the two must not be mixed: they are twelve
    numbers each and the model does not say which twelve it was trained on.
    """
    x, y, z, px, py, pz = u.T
    r0 = np.hypot(x, y)
    hx, hy, hz = hpos.T
    hr = np.hypot(hx, hy)

    phi0 = np.arctan2(y, x)
    c, s = np.cos(phi0), np.sin(phi0)
    surf = np.where(endcap, z1, r1)
    dsurf = surf - np.where(endcap, z, r0)
    helix_free = np.where(endcap, hr, hz)
    dphi_h = (np.arctan2(hy, hx) - phi0 + np.pi) % (2 * np.pi) - np.pi

    return np.column_stack([
        r0, z, px * c + py * s, -px * s + py * c, pz, qop, surf, dsurf,
        helix_free, hr * dphi_h, fm.bz(x, y, z), endcap.astype(float)])


def features_plane(u, qop, cen, nrm, e0, e1, hpos, fm):
    """`train_gtheta.features`, plane branch, on arrays.

    Mirrored line for line against that function and against
    `LearnedTransport.hpp:features`, because three implementations of one
    feature vector is two chances to disagree. `export_kernel.py` checks the
    third against this one.

    Inputs 7 and 8 are the pair that is invariant under the module normal's
    sign: which face DD4hep calls the front is a per-module convention, 36.4%
    of real jumps have p . n < 0, and both `d_plane` and `cos_inc` flip with
    it. Their ratio and the absolute cosine do not.
    """
    x, y, z, px, py, pz = u.T
    r0 = np.hypot(x, y)
    phi0 = np.arctan2(y, x)
    c, s = np.cos(phi0), np.sin(phi0)

    p_abs = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    cos_inc = (np.column_stack([px, py, pz]) * nrm).sum(1) / p_abs
    d_plane = ((cen - u[:, :3]) * nrm).sum(1)
    path_to_plane = np.where(np.abs(cos_inc) > 1e-9,
                             d_plane / np.where(np.abs(cos_inc) > 1e-9,
                                                cos_inc, 1.0), 0.0)
    d = hpos - cen
    # 13 and 14: where e0 points inside the plane, resolved in the cylindrical
    # frame at the module centre. Without them the network is asked for
    # components in a basis it cannot see: 6.3% of the loss captured against
    # 37.3% with them.
    rc = np.hypot(cen[:, 0], cen[:, 1])
    rc = np.where(rc > 0, rc, 1.0)
    e0_phi = (-e0[:, 0] * cen[:, 1] + e0[:, 1] * cen[:, 0]) / rc
    return np.column_stack([
        r0, z, px * c + py * s, -px * s + py * c, pz, qop,
        path_to_plane, np.abs(cos_inc), (d * e0).sum(1), (d * e1).sum(1),
        fm.bz(x, y, z), np.abs(nrm[:, 2]), e0_phi, e0[:, 2]])


def sigma_lookup(cov_path, digi_path):
    """(class, pT bin, |eta| bin) -> the sigma g_theta's position output is
    written in, i.e. hypot(sigma_V, sigma_C). Same table `measured_cov` reads;
    reproduced here because the closed loop keys it on the ESTIMATED state
    rather than the true one, which is the honest thing to do and is also what
    a filter would have."""
    tab = pl.read_parquet(cov_path)
    res = resolutions(digi_path)
    cell = {(r["cls"], r["pt_bin"], r["eta_bin"]): (r["sig_c0_um"],
                                                    r["sig_c1_um"])
            for r in tab.iter_rows(named=True)}

    def get(vol, ip, ie):
        cls = CLASS_OF[int(vol)]
        v = cell.get((cls, int(ip), int(ie))) or cell.get((cls, -1, -1))
        c0 = v[0]
        c1 = v[1] if v[1] is not None else 0.0
        v0, v1 = res[int(vol)]
        s0 = np.hypot(v0, c0)
        s1 = np.hypot(1.0 if v1 is None else v1, c1)
        return s0, s1, (v1 is None), v0, (1.0 if v1 is None else v1)
    return get


def predict(model, u, q, r1, z1, endcap, fm, sig01):
    """helix + g_theta from an arbitrary state. Returns the corrected state.

    Sign conventions are not symmetric and getting them backwards produces a
    model that makes things exactly twice as bad, which is a suspiciously
    plausible failure:

      position   the target is b = helix - truth, so the correction is
                 SUBTRACTED from the helix point
      direction  the target is truth - helix, so it is ADDED
    """
    s, ok = solve_s(u, r1, z1, endcap, q, 2.0)
    out = helix_state(u, s, q, 2.0)
    hpos, hmom = out[:, :3], out[:, 3:]

    p = np.linalg.norm(u[:, 3:], axis=1)
    qop = q / p
    X = features_np(u, qop, r1, z1, endcap, hpos, fm)
    pred, hid = model.forward_hidden((X - model.mu) / model.sd)

    e0, e1, _ = frame(hpos, hmom, endcap)
    pb = pred[:, :2] * sig01                      # predicted bias, um
    pos = hpos - (pb[:, :1] * UM) * e0 - (pb[:, 1:2] * UM) * e1

    hp = np.linalg.norm(hmom, axis=1)
    phi_h = np.arctan2(hmom[:, 1], hmom[:, 0])
    th_h = np.arctan2(np.hypot(hmom[:, 0], hmom[:, 1]), hmom[:, 2])
    phi_c = phi_h + pred[:, 2] * model.v1_scale[0]
    th_c = th_h + pred[:, 3] * model.v1_scale[1]
    # the helix conserves |p|, so its q/p prediction is the source q/p
    qop_c = qop + pred[:, 4] * model.v1_scale[2]
    pc = np.abs(1.0 / np.where(np.abs(qop_c) < 1e-9, np.sign(qop_c) * 1e-9,
                               qop_c))
    mom = np.column_stack([pc * np.sin(th_c) * np.cos(phi_c),
                           pc * np.sin(th_c) * np.sin(phi_c),
                           pc * np.cos(th_c)])
    return np.hstack([pos, mom]), s, ok, hpos, e0, e1, hid, pred


def predict_plane(model, u, q, cen, nrm, mu0, mu1, fm, sig01):
    """`predict`, onto the module plane instead of onto a cylinder.

    Same signature in spirit and same return tuple, so a caller can swap one
    for the other. Three things differ and all three are the retarget:

      the solve   Newton onto n . (X - c) = 0, `make_teacher_pairs`'s own
                  solver, so the follower and the teacher land in the same
                  place by construction rather than by agreement.
      the frame   the correction comes back in the MODULE's local axes, which
                  is where `res_loc0_um` was measured. `frame()` is the
                  cylinder frame and the barrel modules sit 150 mrad off it,
                  so using it here would rotate the correction into the wrong
                  basis by more than the correction is worth.
      the inputs  fourteen, `features_plane`, including the two that say where
                  `mu0` points. `features_np`'s twelve are a different vector
                  and the model does not say which it was trained on, so the
                  caller has to check `model.nin`.
    """
    if model.nin != 14:
        raise ValueError(f"predict_plane needs the 14-input plane model, "
                         f"got one with {model.nin} inputs")

    hx, hy, hz, hpx, hpy, hpz, s, ok = plane_predict(
        u[:, 0], u[:, 1], u[:, 2], u[:, 3], u[:, 4], u[:, 5], q, 2.0, cen, nrm)
    hpos = np.column_stack([hx, hy, hz])
    hmom = np.column_stack([hpx, hpy, hpz])

    p = np.linalg.norm(u[:, 3:], axis=1)
    qop = q / p
    X = features_plane(u, qop, cen, nrm, mu0, mu1, hpos, fm)
    pred, hid = model.forward_hidden((X - model.mu) / model.sd)

    pb = pred[:, :2] * sig01                      # predicted bias, um
    pos = hpos - (pb[:, :1] * UM) * mu0 - (pb[:, 1:2] * UM) * mu1

    hp = np.linalg.norm(hmom, axis=1)
    phi_h = np.arctan2(hmom[:, 1], hmom[:, 0])
    th_h = np.arctan2(np.hypot(hmom[:, 0], hmom[:, 1]), hmom[:, 2])
    phi_c = phi_h + pred[:, 2] * model.v1_scale[0]
    th_c = th_h + pred[:, 3] * model.v1_scale[1]
    qop_c = qop + pred[:, 4] * model.v1_scale[2]
    pc = np.abs(1.0 / np.where(np.abs(qop_c) < 1e-9, np.sign(qop_c) * 1e-9,
                               qop_c))
    mom = np.column_stack([pc * np.sin(th_c) * np.cos(phi_c),
                           pc * np.sin(th_c) * np.sin(phi_c),
                           pc * np.cos(th_c)])
    return np.hstack([pos, mom]), s, ok, hpos, mu0, mu1, hid, pred


# ------------------------------------------------------------------- verify

def verify(model, t, fm, siglook, dump):
    """Run the forward path from the TRUE state and reproduce the dump."""
    u = np.column_stack([t[c].to_numpy() for c in
                         ("x", "y", "z", "px", "py", "pz")])
    q = np.sign(t["qop"].to_numpy())
    r1, z1 = t["r1"].to_numpy(), t["z1"].to_numpy()
    ec = t["endcap"].to_numpy()
    vol = t["surf1"].to_numpy() >> 56
    ip, ie = cells(t["pt"].to_numpy(), t["abs_eta"].to_numpy())
    sig = np.array([siglook(vol[i], ip[i], ie[i])[:2] for i in range(len(t))])

    # feature check against the original implementation, which needs the frame
    from .train_gtheta import features as features_ref
    Xref = features_ref(t, fm)
    s, ok = solve_s(u, r1, z1, ec, q, 2.0)
    hpos = helix_state(u, s, q, 2.0)[:, :3]
    p = np.linalg.norm(u[:, 3:], axis=1)
    Xnew = features_np(u, q / p, r1, z1, ec, hpos, fm)
    d = np.abs(Xref - Xnew) / np.maximum(np.abs(Xref), 1e-9)
    print(f"  features_np vs train_gtheta.features: worst relative "
          f"{np.nanmax(d):.2e}")

    xp, _, _, hp, e0, e1, _, _ = predict(model, u, q, r1, z1, ec, fm, sig)
    truth = np.column_stack([t[c].to_numpy() for c in ("x1", "y1", "z1")])
    rem0 = np.einsum("ij,ij->i", xp[:, :3] - truth, e0) / UM
    rem1 = np.einsum("ij,ij->i", xp[:, :3] - truth, e1) / UM

    D = np.load(dump)
    n = min(len(rem0), len(D["rem0"]))
    print(f"  reproduced residual, MAD sigma: this file "
          f"{robust_sigma(rem0):.2f} um   the dump {robust_sigma(D['rem0']):.2f}"
          f" um")
    print(f"  median |rem0| {np.median(np.abs(rem0)):.2f} vs "
          f"{np.median(np.abs(D['rem0'])):.2f} um")
    return rem0, rem1


# --------------------------------------------------------------- closed loop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="teacher_map_mat_logpt.parquet")
    ap.add_argument("--model", default="gtheta_mat.npz")
    ap.add_argument("--dump", default="runs/mat_v2.npz")
    ap.add_argument("--qtable", default="Q_table.parquet")
    ap.add_argument("--hits",
                    default="sim_out/runs/ch2/tracker_hits/"
                            "tracker_hits_000000-001000.parquet")
    ap.add_argument("--field", default="/tmp/oddb.npz")
    ap.add_argument("--tracks", type=int, default=1500)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--max-depth", type=int, default=12)
    ap.add_argument("--full-jacobian", action="store_true",
                    help="transport the covariance with the COMPLETE F "
                         "(helix + feature map + g_theta) instead of the "
                         "helix block alone. Every chain result so far used "
                         "the helix block, so this is the experiment that "
                         "says whether the expensive one is needed -- and "
                         "that is the whole speed claim.")
    ap.add_argument("--joint-sigma", action="store_true",
                    help="the model carries its own per-hit sigma on output 6 "
                         "(train_joint.py), so read it from there instead of "
                         "from a separately fitted head")
    ap.add_argument("--sigma-head", default=None,
                    help="use the per-hit sigma from sigma_head.py for the "
                         "loc0 component of Q instead of the table's. Scales "
                         "the loc0 row and column of Q, so the measured "
                         "correlations are kept.")
    ap.add_argument("--c0", type=float, nargs=5,
                    default=[0.0, 0.0, 0.0, 0.0, 0.0],
                    help="seed covariance as five sigmas (mm, mm, rad, rad, "
                         "1/GeV). Zero starts the chain from a perfect state, "
                         "which is optimistic; a real CKF starts from a seed.")
    ap.add_argument("--stheta", default=None,
                    help="let s_theta name the destination layer instead of "
                         "reading it off the truth. A wrong name loses the "
                         "hit whatever the gate would have done, which is the "
                         "part no number so far has included.")
    ap.add_argument("--topk", type=int, default=1,
                    help="how many layers s_theta may propose. The plan's "
                         "mitigation for a hard s_theta is top-k with a "
                         "fallback to the navigator, and k is what decides "
                         "whether that mitigation costs anything.")
    ap.add_argument("--seed-error", action="store_true",
                    help="also start the state wrong, drawn from --c0, "
                         "instead of declaring a covariance the state does "
                         "not have. Without it --c0 only tests whether the "
                         "early hole rate is a C-too-small problem; with it "
                         "the chain starts the way a real seed does.")
    ap.add_argument("--kq", type=float, default=1.0,
                    help="scale the filter's Q. The table was measured from "
                         "TRUE states, so in the closed loop it is the wrong "
                         "Q by construction -- this asks whether knowing the "
                         "shift is enough to repair the gate, or whether the "
                         "shift also changes the shape.")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--out", default="closed_loop.npz")
    args = ap.parse_args()

    model = Model(args.model)
    fm = FieldMap(args.field)
    siglook = sigma_lookup(model.cov, model.digi)
    t_all = pl.read_parquet(args.pairs)

    if args.verify:
        fold = folds(t_all["track"].to_numpy())
        print("=== verification: the same forward path, from the TRUE state "
              "===")
        verify(model, t_all.filter(pl.Series(fold == 2)), fm, siglook,
               args.dump)
        return

    fold = folds(t_all["track"].to_numpy())
    t = t_all.filter(pl.Series(fold == 0))
    uniq = np.unique(t["track"].to_numpy())
    rng = np.random.default_rng(args.seed)
    if args.tracks < len(uniq):
        keep = set(rng.choice(uniq, args.tracks, replace=False).tolist())
        t = t.filter(pl.Series([x in keep for x in t["track"].to_numpy()]))
    print(f"{len(t):,} jumps over {t['track'].n_unique():,} tracks "
          f"(training fold)")

    tracks = t["track"].to_numpy()
    surf0, surf1 = t["surf0"].to_numpy(), t["surf1"].to_numpy()
    vol = surf1 >> 56
    lay = (surf1 >> 36) & 0xFFF
    cls = np.array([CLASS_OF[int(v)] for v in vol])
    ec = t["endcap"].to_numpy()
    src_ec = ~np.isin(surf0 >> 56, BARREL_VOLUMES)
    r1, z1 = t["r1"].to_numpy(), t["z1"].to_numpy()
    truth = np.column_stack([t[c].to_numpy() for c in ("x1", "y1", "z1")])
    tmom = np.column_stack([t[c].to_numpy() for c in ("px1", "py1", "pz1")])
    u_true = np.column_stack([t[c].to_numpy() for c in
                              ("x", "y", "z", "px", "py", "pz")])
    qs = np.sign(t["qop"].to_numpy())

    tab = pl.read_parquet(args.qtable)
    d = np.load(args.dump)
    Dd = dict(rem0=d["rem0"], rem1=d["rem1"], one_d=d["one_d"],
              dphi=d["v1_rem"][:, 0], dtheta=d["v1_rem"][:, 1],
              dqop=d["v1_rem"][:, 2],
              cls=np.array([CLASS_OF[int(v)] for v in d["vol"]]))
    _, corr = correlations(Dd)
    Qf = filter_Q(tab, corr, loc1_for_1d(t))
    rho_a, rho_l, nev = occupancy(args.hits)

    link = np.zeros(len(t), dtype=bool)
    link[1:] = (tracks[1:] == tracks[:-1]) & (surf0[1:] == surf1[:-1])
    starts = np.flatnonzero(~link)
    ends = np.append(starts[1:], len(t))

    H2 = np.zeros((2, 5))
    H2[0, 0] = H2[1, 1] = 1.0

    st = None
    if args.stheta:
        sd_ = np.load(args.stheta, allow_pickle=False)
        st = dict(W=[sd_[f"W{i}"] for i in range(3)],
                  b=[sd_[f"b{i}"] for i in range(3)],
                  mu=sd_["mu"], sd=sd_["sd"], cls=sd_["cls"],
                  src_vals=sd_["src_vals"], act=ACTS[str(sd_["act"][0])])
        src_lay = layer_id(surf0)
        dst_lay = layer_id(surf1)
        st["src_idx"] = np.searchsorted(st["src_vals"], src_lay)
        st["src_ok"] = (st["src_vals"][np.clip(st["src_idx"], 0,
                        len(st["src_vals"]) - 1)] == src_lay)
        pos = np.clip(np.searchsorted(st["cls"], dst_lay), 0,
                      len(st["cls"]) - 1)
        st["y"] = np.where(st["cls"][pos] == dst_lay, pos, -1)

    head = None
    if args.sigma_head:
        hd = np.load(args.sigma_head)
        head = (hd["w"], float(hd["b"][0]))
        qtab = pl.read_parquet(args.qtable)
        cellsig = {(r["cls"], r["pt_bin"], r["eta_bin"]): r["sig_c0_um"]
                   for r in qtab.iter_rows(named=True)}

    holes, nhit, nseg = [], [], 0
    by_depth = [[] for _ in range(args.max_depth)]
    hole_depth = [[] for _ in range(args.max_depth)]
    pull_depth = [[] for _ in range(args.max_depth)]
    fake_depth = [[] for _ in range(args.max_depth)]
    w_closed, w_cell, w_depth = [], [], []
    nosol = 0

    for a, b in zip(starts, ends):
        if b - a < args.min_len:
            continue
        nseg += 1
        u = u_true[a:a + 1].copy()          # a chain starts from truth
        C = np.diag(np.array(args.c0)) ** 2
        if args.seed_error and np.any(args.c0):
            # displace the starting state by a draw from the seed covariance,
            # in the source surface's own frame
            de = rng.normal(0, np.array(args.c0))
            Ji = bound_in(u[:, :3], u[:, 3:], src_ec[a:a + 1], qs[a:a + 1])
            u = u + (Ji[0] @ de)[None, :]
        nh = 0
        for dpt, i in enumerate(range(a, b)):
            q = qs[i:i + 1]
            p = np.linalg.norm(u[:, 3:], axis=1)
            pt_est = float(np.hypot(u[0, 3], u[0, 4]))
            eta_est = float(np.abs(np.arctanh(np.clip(u[0, 5] / p[0],
                                                      -0.999999, 0.999999))))
            ip, ie = cells(np.array([pt_est]), np.array([eta_est]))
            s0, s1, one_d, v0, v1 = siglook(vol[i], ip[0], ie[0])
            sig01 = np.array([[s0, s1]])

            xp, sarc, ok, hpos, e0, e1, hid, pr = predict(
                model, u, q, r1[i:i + 1], z1[i:i + 1], ec[i:i + 1], fm, sig01)
            if not ok[0] or not np.all(np.isfinite(xp)):
                nosol += 1
                nh += 1
                if dpt < args.max_depth:
                    hole_depth[dpt].append(1.0)
                break

            # the error carried IN, in the source surface's bound frame
            Jin = bound_out(u[:, :3], u[:, 3:], src_ec[i:i + 1], q)[0]
            e_in = Jin @ np.concatenate([u[0, :3] - u_true[i, :3],
                                         u[0, 3:] - u_true[i, 3:]])
            if args.full_jacobian:
                from .jacobian_full import F_full
                F = F_full(model, u, q, r1[i:i + 1], z1[i:i + 1],
                           ec[i:i + 1], fm, sig01, src_ec[i:i + 1])[0][0]
            else:
                F = F_bound(u, r1[i:i + 1], z1[i:i + 1], ec[i:i + 1], q, 2.0,
                            s=sarc, src_endcap=src_ec[i:i + 1])[0][0]

            key = (str(cls[i]), int(ip[0]), int(ie[0]))
            Q = Qf.get(key)
            if Q is None:
                Q = Qf[(str(cls[i]), -1, -1)]
            if args.joint_sigma:
                m = float(np.exp(pr[0, 5]))
                D = np.diag([m, 1.0, 1.0, 1.0, 1.0])
                Q = D @ Q @ D
            if head is not None:
                # per-hit sigma on loc0 only: scale that row and column, which
                # keeps the measured correlations and changes only the width
                # the gate is built from
                m = float(np.exp(hid[0] @ head[0] + head[1]))
                D = np.diag([m, 1.0, 1.0, 1.0, 1.0])
                Q = D @ Q @ D
            C = F @ C @ F.T + (args.kq ** 2) * Q

            # the full prediction error on this surface, in its own frame
            dv = xp[0, :3] - truth[i]
            err = np.array([dv @ e0[0], dv @ e1[0]]) / UM      # um
            # what THIS step added, with the propagated input error removed
            w = err - (F @ e_in)[:2] / UM
            w_closed.append(w[0])
            w_cell.append(key)
            w_depth.append(dpt)

            H = H2[:1] if one_d else H2
            V = np.diag([(v0 * UM) ** 2]) if one_d else \
                np.diag([(v0 * UM) ** 2, (v1 * UM) ** 2])
            eps = rng.normal(0, np.sqrt(np.diag(V)))
            r = eps - (err * UM)[:len(eps)]
            S = H @ C @ H.T + V
            Si = np.linalg.inv(S)
            chi2 = float(r @ Si @ r)

            if dpt < args.max_depth:
                by_depth[dpt].append(abs(err[0]))
                pull_depth[dpt].append(r[0] / np.sqrt(S[0, 0]))
                # the fake budget, against the same pileup-10 occupancy
                # chain_sim uses -- a floor, and the thing that has to be held
                # equal for a holes comparison to mean anything
                # S is already in mm^2 -- C and V are both built in mm -- and
                # rho is per mm^2, so no unit conversion belongs here
                if one_d:
                    nf = (rho_l.get((int(vol[i]), int(lay[i])), 0.0)
                          * 2 * np.sqrt(CHI2_CUT * S[0, 0]))
                else:
                    nf = (rho_a.get((int(vol[i]), int(lay[i])), 0.0)
                          * np.pi * CHI2_CUT * np.sqrt(np.linalg.det(S)))
                fake_depth[dpt].append(nf)

            # s_theta names the layer BEFORE the gate sees anything: if it
            # names the wrong one the hit is lost however good the correction
            # is, so this is a veto on the gate rather than a term in it
            named_right = True
            if st is not None:
                xs, ys, zs = u[0, :3]
                pxs, pys, pzs = u[0, 3:]
                phi_s = np.arctan2(ys, xs)
                cc, ss_ = np.cos(phi_s), np.sin(phi_s)
                cont = np.array([[np.hypot(xs, ys), zs,
                                  pxs * cc + pys * ss_, -pxs * ss_ + pys * cc,
                                  pzs, q[0] / p[0], np.hypot(pxs, pys),
                                  np.arctanh(np.clip(pzs / p[0], -0.999999,
                                                     0.999999)),
                                  fm.bz(np.array([xs]), np.array([ys]),
                                        np.array([zs]))[0]]])
                oh = np.zeros((1, len(st["src_vals"])))
                if st["src_ok"][i]:
                    oh[0, st["src_idx"][i]] = 1.0
                Xs = (np.hstack([cont, oh]) - st["mu"]) / st["sd"]
                h1_, _ = st["act"](Xs @ st["W"][0] + st["b"][0])
                h2_, _ = st["act"](h1_ @ st["W"][1] + st["b"][1])
                lg = h2_ @ st["W"][2] + st["b"][2]
                topk = np.argsort(-lg[0])[:args.topk]
                named_right = st["y"][i] >= 0 and st["y"][i] in topk

            if chi2 < CHI2_CUT and named_right:
                K = C @ H.T @ Si
                dx = K @ r
                pos = xp[0, :3] + dx[0] * e0[0] + (dx[1] * e1[0]
                                                   if not one_d else 0.0)
                pmag = np.linalg.norm(xp[0, 3:])
                phi_d = np.arctan2(xp[0, 4], xp[0, 3]) + dx[2]
                th_d = np.arctan2(np.hypot(xp[0, 3], xp[0, 4]),
                                  xp[0, 5]) + dx[3]
                qop_n = q[0] / pmag + dx[4]
                pm = abs(1.0 / qop_n) if abs(qop_n) > 1e-9 else pmag
                u = np.array([[pos[0], pos[1], pos[2],
                               pm * np.sin(th_d) * np.cos(phi_d),
                               pm * np.sin(th_d) * np.sin(phi_d),
                               pm * np.cos(th_d)]])
                C = C - K @ H @ C
                if dpt < args.max_depth:
                    hole_depth[dpt].append(0.0)
            else:
                nh += 1
                u = xp.copy()
                if dpt < args.max_depth:
                    hole_depth[dpt].append(1.0)
        holes.append(nh)
        nhit.append(b - a)

    holes = np.array(holes)
    fakes = float(np.sum([np.mean(f) for f in fake_depth if len(f) >= 20]))
    print(f"  {nseg:,} chains, {nosol} abandoned for want of a helix solution")
    print(f"\n### closed loop -- g_theta run on its own filtered state"
          + ("  [per-hit sigma]" if head is not None else "  [table sigma]"))
    print(f"  holes/track {holes.mean():.3f}   >=1: {100*(holes>=1).mean():.1f}"
          f"%   >=3: {100*(holes>=3).mean():.1f}%   fakes/track {fakes:.2f}")
    print(f"  {'layer':>6}{'|loc0 err| med [um]':>22}{'p90':>10}{'pull w':>9}"
          f"{'hole rate':>11}{'fakes':>8}{'n':>8}")
    for dd, (x, p_, hd, fk) in enumerate(zip(by_depth, pull_depth, hole_depth,
                                             fake_depth)):
        if len(x) < 20:
            break
        x = np.array(x)
        print(f"  {dd:6d}{np.median(x):22.1f}{np.percentile(x, 90):10.1f}"
              f"{robust_sigma(np.array(p_)):9.2f}{100*np.mean(hd):10.1f}%"
              f"{np.mean(fk):8.2f}{len(x):8,d}")

    # --- the measurement this file exists for
    w_closed = np.array(w_closed)
    w_cell = np.array(w_cell, dtype=object)
    dc = np.array([CLASS_OF[int(v)] for v in d["vol"]])
    ipd, ied = cells(d["pt"], d["abs_eta"])
    open_key = np.array([f"{a}/{b}/{c}" for a, b, c in zip(dc, ipd, ied)],
                        dtype=object)
    closed_key = np.array([f"{k[0]}/{k[1]}/{k[2]}" for k in w_cell],
                          dtype=object)

    print(f"\n### distribution shift -- what one step adds, from truth vs "
          f"from a filtered state")
    print(f"  {'cell population':22}{'from truth':>14}{'closed loop':>14}"
          f"{'ratio':>9}{'n':>9}")
    tot = []
    for lab, sel in (("all", np.ones(len(w_closed), bool)),):
        so = robust_sigma(d["rem0"])
        sc = robust_sigma(w_closed[sel])
        print(f"  {lab:22}{so:12.1f} um{sc:12.1f} um{sc/so:9.2f}"
              f"{sel.sum():9,d}")
    for c in ("pixel", "sstrip", "lstrip"):
        mo = dc == c
        mc = np.array([k.startswith(c) for k in closed_key])
        if mc.sum() < 100:
            continue
        so, sc = robust_sigma(d["rem0"][mo]), robust_sigma(w_closed[mc])
        tot.append(sc / so)
        print(f"  {c:22}{so:12.1f} um{sc:12.1f} um{sc/so:9.2f}"
              f"{int(mc.sum()):9,d}")
    print("\n  ratio 1.00 would mean the model is as good on its own filtered "
          "state as on\n  a true one, i.e. no distribution shift. Above 1 is "
          "the cost of the shift.")

    # --- C1: does the shift grow with depth? The input state is more wrong
    # deeper in, so a shift that is a function of how wrong the input is has
    # to show up here. A flat row means the model saturates early.
    wd = np.array(w_depth)
    so_all = robust_sigma(d["rem0"])
    ro_all = np.std(d["rem0"]) / so_all
    print(f"\n### C1 -- does the shift grow with depth?")
    print(f"  {'layer':>6}{'MAD sigma':>12}{'/ from truth':>14}"
          f"{'RMS/MAD':>10}{'n':>9}")
    print(f"  {'truth':>6}{so_all:12.1f}{1.00:14.2f}{ro_all:10.2f}"
          f"{len(d['rem0']):9,d}")
    for dd in range(args.max_depth):
        k = wd == dd
        if k.sum() < 100:
            continue
        s = robust_sigma(w_closed[k])
        print(f"  {dd:6d}{s:12.1f}{s/so_all:14.2f}"
              f"{np.std(w_closed[k])/s:10.2f}{int(k.sum()):9,d}")

    # --- C2: is the shift a wider distribution, or a differently shaped one?
    # RMS/MAD is the same statistic that measured the tail in A1, so if the
    # shift only widens, this column does not move.
    rc = np.std(w_closed) / robust_sigma(w_closed)
    print(f"\n### C2 -- width or shape?")
    print(f"  RMS/MAD   from truth {ro_all:.2f}   closed loop {rc:.2f}")
    print(f"  MAD sigma ratio {robust_sigma(w_closed)/so_all:.2f}, "
          f"RMS ratio {np.std(w_closed)/np.std(d['rem0']):.2f}")
    print("  If the RMS ratio exceeds the MAD ratio, the shift is adding "
          "tail rather than\n  width -- which is what a Q inflated by the "
          "width alone cannot repair.")

    np.savez_compressed(
        args.out, holes=holes, nhit=np.array(nhit),
        w_closed=w_closed, w_depth=wd,
        fakes=np.array([fakes]),
        depth_fake=np.array([np.mean(f) if len(f) >= 20 else np.nan
                             for f in fake_depth]),
        depth_med=np.array([np.median(x) if len(x) >= 20 else np.nan
                            for x in by_depth]),
        depth_p90=np.array([np.percentile(x, 90) if len(x) >= 20 else np.nan
                            for x in by_depth]),
        depth_pull=np.array([robust_sigma(np.array(x)) if len(x) >= 20
                             else np.nan for x in pull_depth]),
        depth_hole=np.array([np.mean(x) if len(x) >= 20 else np.nan
                             for x in hole_depth]),
        depth_n=np.array([len(x) for x in by_depth]))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
