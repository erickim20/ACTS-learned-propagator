"""The last two blocks of F, and the three chained into one.

One question left open by the chained measurement:

    dg_theta/dx           done, 5e-10, 1.85x forward   (train_gtheta.MLP)
    helix Jacobian        done, 1e-14                  (jacobian_helix)
    feature-map Jacobian  MISSING -- needs grad B
    chaining the three    MISSING

Both are here. The feature map is analytic in eleven of its twelve columns; the
twelfth is B_z sampled from the ODD grid, so its gradient is a central
difference ON that grid -- which is the honest derivative of the function the
model was actually trained through, not of the underlying physical field.
Trilinear interpolation is only C0, so the exact derivative is piecewise
constant and jumps at cell walls; a central difference over one cell is the
smooth thing to hand a covariance, and it is what is used.

The composition is where the surface constraint bites twice. The prediction is

    x_pred = helix(u, s(u))  -  (g_theta(features(u, helix(u, s(u)))) . sigma) . (e0, e1)

so the network's input depends on the helix output, the correction is applied
along a frame that itself depends on the helix output, and the arc length s
depends on u through the surface. All three couplings are carried.

Verified against finite differences taken through the WHOLE prediction path in
`closed_loop.predict` -- helix solve, feature build, forward pass, correction --
so nothing in the chain can be individually plausible and jointly wrong.

Usage:
    python jacobian_full.py --n 600
"""
import argparse
import time

import numpy as np
import polars as pl

from .closed_loop import Model, features_np, predict, sigma_lookup
from .chain_sim import cells
from .helix_variants import FieldMap
from .jacobian_helix import (BARREL_VOLUMES, bound_in, bound_out, frame,
                            helix_jacobian_global, helix_state, rel, solve_s)

UM = 1e-3


def grad_bz(fm, x, y, z):
    """grad B_z by central difference on the map's own grid.

    One cell in each direction: the interpolant is C0, so its exact gradient
    is piecewise constant and discontinuous at the walls, and handing that to
    a covariance puts steps into something that is supposed to be smooth in
    the state. Same reasoning that made the activation ptanh.
    """
    h = fm.h                                  # grid spacing, mm
    out = np.zeros((len(x), 3))
    for k, (dx, dy, dz) in enumerate(((h[0], 0, 0), (0, h[1], 0),
                                      (0, 0, h[2]))):
        out[:, k] = (fm.bz(x + dx, y + dy, z + dz)
                     - fm.bz(x - dx, y - dy, z - dz)) / (2 * (dx + dy + dz))
    return out


def dfeatures(u, q, r1, z1, endcap, hpos, fm, Jh):
    """d(features)/d(u), 12x6. Jh is d(helix out)/du from helix_jacobian_global.

    Columns 8 and 9 are the two helix-derived inputs, so this is where the
    physics core's Jacobian enters the network's -- the block the plan called
    'chaining the three'.
    """
    n = len(u)
    x, y, z, px, py, pz = u.T
    r0 = np.hypot(x, y)
    phi0 = np.arctan2(y, x)
    c, s = np.cos(phi0), np.sin(phi0)
    p_r = px * c + py * s
    p_phi = -px * s + py * c
    p = np.sqrt(px ** 2 + py ** 2 + pz ** 2)

    hx, hy, hz = hpos.T
    hr = np.hypot(hx, hy)
    phih = np.arctan2(hy, hx)

    dphi0 = np.zeros((n, 6))
    dphi0[:, 0], dphi0[:, 1] = -s / r0, c / r0

    J = np.zeros((n, 12, 6))
    # 0: r0
    J[:, 0, 0], J[:, 0, 1] = c, s
    # 1: z
    J[:, 1, 2] = 1.0
    # 2: p_r, 3: p_phi -- both rotate with phi0 and pick up the momentum
    J[:, 2, :] = p_phi[:, None] * dphi0
    J[:, 2, 3], J[:, 2, 4] = c, s
    J[:, 3, :] = -p_r[:, None] * dphi0
    J[:, 3, 3], J[:, 3, 4] = -s, c
    # 4: pz
    J[:, 4, 5] = 1.0
    # 5: q/|p|
    for k, pk in enumerate((px, py, pz)):
        J[:, 5, 3 + k] = -q * pk / p ** 3
    # 6: the destination surface's own coordinate -- a constant of the surface
    # 7: dsurf = surf - (z or r0)
    J[:, 7, 2] = np.where(endcap, -1.0, 0.0)
    J[:, 7, 0] = np.where(endcap, 0.0, -c)
    J[:, 7, 1] = np.where(endcap, 0.0, -s)
    # 8: helix_free = hr (endcap) or hz (barrel)
    dhr = (hx[:, None] * Jh[:, 0, :] + hy[:, None] * Jh[:, 1, :]) / hr[:, None]
    J[:, 8, :] = np.where(endcap[:, None], dhr, Jh[:, 2, :])
    # 9: hr * wrap(phi_helix - phi0)
    dphih = (hx[:, None] * Jh[:, 1, :] - hy[:, None] * Jh[:, 0, :]) \
        / (hr ** 2)[:, None]
    dwrap = dphih - dphi0
    wrap = (phih - phi0 + np.pi) % (2 * np.pi) - np.pi
    J[:, 9, :] = wrap[:, None] * dhr + hr[:, None] * dwrap
    # 10: B_z at the source
    gb = grad_bz(fm, x, y, z)
    J[:, 10, :3] = gb
    # 11: the endcap flag, a constant of the surface
    return J


def F_full(model, u, q, r1, z1, endcap, fm, sig01, src_endcap):
    """The complete transport Jacobian, bound 5x5, for helix + g_theta."""
    n = len(u)
    s, ok = solve_s(u, r1, z1, endcap, q, 2.0)
    out = helix_state(u, s, q, 2.0)
    hpos, hmom = out[:, :3], out[:, 3:]
    Jh = helix_jacobian_global(u, s, q, 2.0, endcap)

    X = features_np(u, q / np.linalg.norm(u[:, 3:], axis=1), r1, z1, endcap,
                    hpos, fm)
    Xn = (X - model.mu) / model.sd
    pred = model.forward(Xn)

    # dpred/du = (dg/dXn) . (dX/du) / sd
    h1, d1 = model.act(Xn @ model.W[0] + model.b[0])
    _, d2 = model.act(h1 @ model.W[1] + model.b[1])
    m = d2[:, :, None] * model.W[2][None, :, :]
    m = (model.W[1] @ m) * d1[:, :, None]
    dg = np.einsum("ih,bho->boi", model.W[0], m)          # (n, 6, 12)
    dX = dfeatures(u, q, r1, z1, endcap, hpos, fm, Jh)
    dpred = np.einsum("boi,bij->boj", dg / model.sd[None, None, :], dX)

    # --- position: helix point, moved in-plane by the correction
    e0, e1, _ = frame(hpos, hmom, endcap)
    phih = np.arctan2(hpos[:, 1], hpos[:, 0])
    dphih = (hpos[:, 0:1] * Jh[:, 1, :] - hpos[:, 1:2] * Jh[:, 0, :]) \
        / (hpos[:, 0] ** 2 + hpos[:, 1] ** 2)[:, None]
    de0 = np.stack([-np.cos(phih), -np.sin(phih), np.zeros(n)], 1)
    de0 = de0[:, :, None] * dphih[:, None, :]
    de1 = np.where(endcap[:, None, None],
                   e0[:, :, None] * dphih[:, None, :], 0.0)

    pb = pred[:, :2] * sig01 * UM                          # mm
    dpb = dpred[:, :2, :] * (sig01 * UM)[:, :, None]
    dpos = (Jh[:, :3, :]
            - e0[:, :, None] * dpb[:, 0:1, :] - pb[:, 0:1, None] * de0
            - e1[:, :, None] * dpb[:, 1:2, :] - pb[:, 1:2, None] * de1)

    # --- direction and q/p: helix value plus the v1 corrections
    hpt = np.hypot(hmom[:, 0], hmom[:, 1])
    hp = np.linalg.norm(hmom, axis=1)
    dphi_h = (hmom[:, 0:1] * Jh[:, 4, :] - hmom[:, 1:2] * Jh[:, 3, :]) \
        / (hpt ** 2)[:, None]
    dth_h = (hmom[:, 2:3] * (hmom[:, 0:1] * Jh[:, 3, :]
                             + hmom[:, 1:2] * Jh[:, 4, :]) / hpt[:, None]
             - hpt[:, None] * Jh[:, 5, :]) / (hp ** 2)[:, None]
    pu = np.linalg.norm(u[:, 3:], axis=1)
    dqop_src = np.zeros((n, 6))
    for k in range(3):
        dqop_src[:, 3 + k] = -q * u[:, 3 + k] / pu ** 3

    dphi_c = dphi_h + model.v1_scale[0] * dpred[:, 2, :]
    dth_c = dth_h + model.v1_scale[1] * dpred[:, 3, :]
    dqop_c = dqop_src + model.v1_scale[2] * dpred[:, 4, :]

    phi_c = np.arctan2(hmom[:, 1], hmom[:, 0]) + pred[:, 2] * model.v1_scale[0]
    th_c = np.arctan2(hpt, hmom[:, 2]) + pred[:, 3] * model.v1_scale[1]
    qop_c = q / pu + pred[:, 4] * model.v1_scale[2]
    pc = np.abs(1.0 / qop_c)
    dpc = -(pc ** 2)[:, None] * np.sign(qop_c)[:, None] * dqop_c

    st, ct_ = np.sin(th_c), np.cos(th_c)
    sp, cp = np.sin(phi_c), np.cos(phi_c)
    dmom = np.stack([
        dpc * (st * cp)[:, None] + (pc * ct_ * cp)[:, None] * dth_c
        - (pc * st * sp)[:, None] * dphi_c,
        dpc * (st * sp)[:, None] + (pc * ct_ * sp)[:, None] * dth_c
        + (pc * st * cp)[:, None] * dphi_c,
        dpc * ct_[:, None] - (pc * st)[:, None] * dth_c], 1)

    Jglob = np.concatenate([dpos, dmom], axis=1)           # (n, 6, 6)

    pos = hpos - pb[:, 0:1] * e0 - pb[:, 1:2] * e1
    mom = np.stack([pc * st * cp, pc * st * sp, pc * ct_], 1)
    Jo = bound_out(pos, mom, endcap, q)
    Ji = bound_in(u[:, :3], u[:, 3:], src_endcap, q)
    return np.einsum("nij,njk,nkl->nil", Jo, Jglob, Ji), ok, Jglob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="teacher_map_mat_logpt.parquet")
    ap.add_argument("--model", default="gtheta_mat.npz")
    ap.add_argument("--field", default="/tmp/oddb.npz")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20260806)
    args = ap.parse_args()

    model = Model(args.model)
    fm = FieldMap(args.field)
    t = pl.read_parquet(args.pairs)
    rng = np.random.default_rng(args.seed)
    t = t[rng.choice(len(t), args.n, replace=False)]

    u = np.column_stack([t[c].to_numpy() for c in
                         ("x", "y", "z", "px", "py", "pz")])
    q = np.sign(t["qop"].to_numpy())
    r1, z1 = t["r1"].to_numpy(), t["z1"].to_numpy()
    ec = t["endcap"].to_numpy()
    src_ec = ~np.isin(t["surf0"].to_numpy() >> 56, BARREL_VOLUMES)
    vol = t["surf1"].to_numpy() >> 56
    ip, ie = cells(t["pt"].to_numpy(), t["abs_eta"].to_numpy())
    look = sigma_lookup(model.cov, model.digi)
    sig01 = np.array([look(vol[i], ip[i], ie[i])[:2] for i in range(len(t))])

    # --- feature map alone
    s, _ = solve_s(u, r1, z1, ec, q, 2.0)
    hpos = helix_state(u, s, q, 2.0)[:, :3]
    Jh = helix_jacobian_global(u, s, q, 2.0, ec)
    A = dfeatures(u, q, r1, z1, ec, hpos, fm, Jh)
    N = np.zeros_like(A)
    for j in range(6):
        h = 1e-6 * max(1.0, np.abs(u[:, j]).mean())
        up, um = u.copy(), u.copy()
        up[:, j] += h
        um[:, j] -= h
        for sgn, uu in ((1, up), (-1, um)):
            ss, _ = solve_s(uu, r1, z1, ec, q, 2.0)
            hp = helix_state(uu, ss, q, 2.0)[:, :3]
            X = features_np(uu, q / np.linalg.norm(uu[:, 3:], axis=1), r1, z1,
                            ec, hp, fm)
            N[:, :, j] += sgn * X / (2 * h)
    r = rel(A, N)
    print(f"=== d(features)/du, 12x6, against finite differences ===")
    print(f"  median {np.median(r):.2e}   p99 {np.percentile(r, 99):.2e}   "
          f"worst {r.max():.2e}")
    rb = rel(A[:, [10], :], N[:, [10], :])
    print(f"  the B_z row alone: median {np.median(rb):.2e}   worst "
          f"{rb.max():.2e}   (C0 interpolant, so this row is the loose one)")

    # --- the whole thing
    F, ok, _ = F_full(model, u, q, r1, z1, ec, fm, sig01, src_ec)
    Ji = bound_in(u[:, :3], u[:, 3:], src_ec, q)
    M = np.zeros_like(F)
    for j in range(5):
        d = Ji[:, :, j]
        nrm = np.linalg.norm(d, axis=1)
        nrm[nrm == 0] = 1.0
        dh = d / nrm[:, None]
        h = 1e-6 * np.maximum(1.0, np.abs(u).max(1))
        up = u + h[:, None] * dh
        um = u - h[:, None] * dh
        xp = predict(model, up, q, r1, z1, ec, fm, sig01)[0]
        xm = predict(model, um, q, r1, z1, ec, fm, sig01)[0]
        Jo = bound_out(0.5 * (xp[:, :3] + xm[:, :3]),
                       0.5 * (xp[:, 3:] + xm[:, 3:]), ec, q)
        M[:, :, j] = np.einsum("nij,nj->ni", Jo, xp - xm) \
            / (2 * h[:, None]) * nrm[:, None]
    r = rel(F, M)
    print(f"\n=== F_total = helix + g_theta, bound 5x5, against finite "
          f"differences\n    through the whole prediction path ===")
    print(f"  median {np.median(r):.2e}   p99 {np.percentile(r, 99):.2e}   "
          f"worst {r.max():.2e}")

    # --- what the learned block does to the transport
    from jacobian_helix import F_bound
    Fh = F_bound(u, r1, z1, ec, q, 2.0, s=s, src_endcap=src_ec)[0]
    lev_h = np.abs(Fh[:, 0, 2])
    lev_f = np.abs(F[:, 0, 2])
    print(f"\n=== what g_theta changes about F ===")
    print(f"  dloc0/dphi   helix alone median {np.median(lev_h):8.1f} mm"
          f"   with g_theta {np.median(lev_f):8.1f} mm")
    cs = np.maximum(np.abs(Fh).max(1, keepdims=True), 1e-30)
    d = (np.abs(F - Fh) / cs).reshape(len(u), -1).max(1)
    print(f"  |F_total - F_helix| / column scale: median {np.median(d):.2f}"
          f"   p90 {np.percentile(d, 90):.2f}")
    print("  A covariance transported with the helix Jacobian alone is "
          "therefore not the\n  transport unit's covariance -- the learned "
          "block is a first-order effect on F,\n  not a correction to it.")

    big = np.repeat(u, max(1, 20000 // len(u)), axis=0)
    reps = max(1, 20000 // len(u))
    args_big = (np.repeat(q, reps), np.repeat(r1, reps), np.repeat(z1, reps),
                np.repeat(ec, reps), fm, np.repeat(sig01, reps, axis=0),
                np.repeat(src_ec, reps))
    for name, fn in (("predict (forward)",
                      lambda: predict(model, big, args_big[0], args_big[1],
                                      args_big[2], args_big[3], fm,
                                      args_big[5])),
                     ("F_total", lambda: F_full(model, big, *args_big))):
        fn()
        t0 = time.perf_counter()
        for _ in range(3):
            fn()
        dt = (time.perf_counter() - t0) / 3 / len(big) * 1e9
        print(f"  {name:20}{dt:9.0f} ns/jump")
    # numpy is overhead-dominated here as it was for every other Jacobian in
    # this project; the multiply-add count is what transfers to a kernel
    nin, hid, nout = 12, 64, 6
    fwd = nin * hid + hid * hid + hid * nout
    dg = hid * hid * nout + nin * hid * nout          # dg/dXn, all six rows
    chain = nout * nin * 6                           # dg/dXn . dX/du
    helix = 420                                      # jacobian_helix's count
    print(f"\n  multiply-adds per jump:")
    print(f"    g_theta forward              {fwd:,}")
    print(f"    dg_theta/dXn, all 6 rows     {dg:,}")
    print(f"    d(features)/du + chaining    {chain + helix:,}")
    print(f"    F_total                      {dg + chain + helix:,}"
          f"  = {(dg + chain + helix) / fwd:.1f}x the forward pass")
    print(f"  The gate needs two rows, not six; at two rows dg/dXn is "
          f"{hid*hid*2 + nin*hid*2:,}\n  and F_total lands at "
          f"{(hid*hid*2 + nin*hid*2 + 2*nin*6 + helix)/fwd:.1f}x. That is the "
          f"number the speed claim\n  has to survive, and it is question 3 for "
          f"Monday.")


if __name__ == "__main__":
    main()
