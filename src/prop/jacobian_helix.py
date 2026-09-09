"""B -- the helix Jacobian, the factor of F that was missing.

`C_pred = F C_in F^T + Q`. F is a product of three blocks and only one of them
existed: `MLP.jacobian` in `train_gtheta.py` gives dg_theta/dx, verified to
5e-10 and costing 1.85x the forward pass. This file supplies the other two --
the physics core's own Jacobian, and the frame maps that put it in the
coordinates a bound track state actually uses.

The part that is easy to get wrong: the destination arc length is not a
constant. The helix is solved onto a surface, so s is an implicit function of
the source state through

    barrel   g(s, u) = x(s,u)^2 + y(s,u)^2 - r1^2 = 0
    endcap   g(s, u) = z(s,u) - z1                = 0

and the total derivative therefore carries a second term:

    dP/du = dP/du|_s  -  (dP/ds) (dg/du) / (dg/ds)

Dropping it gives a Jacobian that looks reasonable and is wrong by order one
in exactly the direction the gate measures, because moving the source along
the trajectory changes where the helix lands on the surface not at all -- the
two terms have to cancel for that mode, and only do if both are present.

Frames. The filter does not carry a global 6-vector, it carries the bound
5-vector (loc0, loc1, phi, theta, q/p) on the surface. The two in-plane
directions are the ones `chi2_gate.in_plane_residual` already uses:

    barrel   e0 = phi_hat,  e1 = z_hat
    endcap   e0 = phi_hat,  e1 = r_hat

Local *displacements* are projections onto that orthonormal pair at the
reference point, which is what a linearisation about a reference trajectory
means and avoids inventing a chart origin on the surface.

Everything is checked against finite differences through the actual bisection
solver in `make_teacher_pairs.helix_predict` -- not against a re-derivation,
since a re-derivation shares whatever mistake the analytic form has.

Usage:
    python jacobian_helix.py                    # verify + time
    python jacobian_helix.py --n 4000
"""
import argparse
import time

import numpy as np
import polars as pl

from .make_teacher_pairs import MM, helix_predict, helix_positions

BARREL_VOLUMES = (17, 24, 29)      # ODD tracker; every other volume is a disc

# ------------------------------------------------------------------ the helix


def solve_s(u, r1, z1, endcap, q, bz):
    """Arc length onto the destination surface, from the production solver."""
    x0, y0, z0, px, py, pz = u.T
    out = helix_predict(x0, y0, z0, px, py, pz, q, bz, r1, z1, endcap)
    return out[6], out[7]


def helix_state(u, s, q, bz):
    """(position, momentum) at transverse arc length s. Same formulas as the
    teacher generator, kept in one place so a fix cannot land in only one."""
    x0, y0, z0, px, py, pz = u.T
    pt = np.hypot(px, py)
    phi0 = np.arctan2(py, px)
    kappa = -0.3 * q * bz / pt * MM
    phi = phi0 + kappa * s
    a = 1.0 / kappa
    return np.column_stack([
        x0 + a * (np.sin(phi) - np.sin(phi0)),
        y0 - a * (np.cos(phi) - np.cos(phi0)),
        z0 + (pz / pt) * s,
        pt * np.cos(phi),
        pt * np.sin(phi),
        pz,
    ])


def helix_jacobian_global(u, s, q, bz, endcap, with_ds=False, plane_n=None):
    """d(position, momentum)_out / d(position, momentum)_in, 6x6, with the
    surface constraint differentiated through.

    Returned batched as (N, 6, 6). Column order is the same as u:
    (x0, y0, z0, px, py, pz).

    `with_ds` additionally returns ds/du, the derivative of the destination arc
    length wrt the source state, and d(out)/ds, the trajectory tangent in this
    parameterisation. Both are computed here rather than anywhere else because
    the constraint term they come from is the part of this file that is easy to
    get wrong, and `jacobian_free` needs them to hand ACTS the pieces its own
    covariance engine expects.

    `plane_n` replaces the constraint with the module plane n . (P - c) = 0 and
    makes `endcap` unused. Everything above the constraint is the same helix,
    so only g changes: linear in the normal, against a quadratic in r for the
    barrel cylinder and a coordinate for the disc.
    """
    x0, y0, z0, px, py, pz = u.T
    n = len(px)
    pt = np.hypot(px, py)
    phi0 = np.arctan2(py, px)
    kappa = -0.3 * q * bz / pt * MM
    a = 1.0 / kappa
    phi = phi0 + kappa * s
    sp, cp = np.sin(phi), np.cos(phi)
    sp0, cp0 = np.sin(phi0), np.cos(phi0)

    # --- derivatives of the intermediates wrt (px, py)
    dpt = np.column_stack([px / pt, py / pt])                    # (N,2)
    dphi0 = np.column_stack([-py / pt ** 2, px / pt ** 2])
    dkappa = -(kappa / pt)[:, None] * dpt
    da = -(a ** 2)[:, None] * dkappa
    dphi = dphi0 + s[:, None] * dkappa

    J = np.zeros((n, 6, 6))
    # position out wrt position in: the helix is a rigid displacement, so at
    # fixed s the source position enters additively
    J[:, 0, 0] = 1.0
    J[:, 1, 1] = 1.0
    J[:, 2, 2] = 1.0

    for k, col in enumerate((3, 4)):                # px, py
        J[:, 0, col] = (da[:, k] * (sp - sp0)
                        + a * (cp * dphi[:, k] - cp0 * dphi0[:, k]))
        J[:, 1, col] = -(da[:, k] * (cp - cp0)
                         + a * (-sp * dphi[:, k] + sp0 * dphi0[:, k]))
        J[:, 2, col] = -pz * s * dpt[:, k] / pt ** 2
        J[:, 3, col] = dpt[:, k] * cp - pt * sp * dphi[:, k]
        J[:, 4, col] = dpt[:, k] * sp + pt * cp * dphi[:, k]
    J[:, 2, 5] = s / pt                             # z wrt pz
    J[:, 5, 5] = 1.0                                # pz is conserved

    # --- the trajectory tangent in this parameterisation, d(out)/ds
    dds = np.zeros((n, 6))
    dds[:, 0] = cp
    dds[:, 1] = sp
    dds[:, 2] = pz / pt
    dds[:, 3] = -pt * sp * kappa
    dds[:, 4] = pt * cp * kappa

    # --- ds/du from the surface constraint
    x, y = (x0 + a * (sp - sp0)), (y0 - a * (cp - cp0))
    if plane_n is not None:
        dg_du = (plane_n[:, 0, None] * J[:, 0, :]
                 + plane_n[:, 1, None] * J[:, 1, :]
                 + plane_n[:, 2, None] * J[:, 2, :])
        dg_ds = (plane_n[:, 0] * dds[:, 0] + plane_n[:, 1] * dds[:, 1]
                 + plane_n[:, 2] * dds[:, 2])
    elif endcap.all() or (~endcap).all():
        ec = bool(endcap[0])
        if ec:
            dg_du = J[:, 2, :].copy()               # g = z - z1
            dg_ds = dds[:, 2]
        else:
            dg_du = 2 * (x[:, None] * J[:, 0, :] + y[:, None] * J[:, 1, :])
            dg_ds = 2 * (x * dds[:, 0] + y * dds[:, 1])
    else:
        dg_du = np.where(endcap[:, None], J[:, 2, :],
                         2 * (x[:, None] * J[:, 0, :] + y[:, None] * J[:, 1, :]))
        dg_ds = np.where(endcap, dds[:, 2],
                         2 * (x * dds[:, 0] + y * dds[:, 1]))
    ds_du = -dg_du / dg_ds[:, None]

    Jtot = J + dds[:, :, None] * ds_du[:, None, :]
    return (Jtot, ds_du, dds) if with_ds else Jtot


# ------------------------------------------------------------------- frames

def frame(pos, mom, endcap):
    """(e0, e1) on the surface and the direction basis, for one batch.

    e0 is phi_hat on both surface types, which is what makes loc0 the r*phi
    coordinate every residual in this project is quoted in.
    """
    x, y = pos[:, 0], pos[:, 1]
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    e0 = np.column_stack([-np.sin(phi), np.cos(phi), np.zeros_like(x)])
    zhat = np.column_stack([np.zeros_like(x), np.zeros_like(x),
                            np.ones_like(x)])
    rhat = np.column_stack([np.cos(phi), np.sin(phi), np.zeros_like(x)])
    e1 = np.where(endcap[:, None], rhat, zhat)
    return e0, e1, r


def bound_out(pos, mom, endcap, q, axes=None):
    """J_out: d(loc0, loc1, phi_dir, theta_dir, q/p) / d(pos, mom), 5x6.

    `axes` overrides the cylinder frame with the destination surface's own
    (e0, e1). Only rows 0 and 1 move: 2 to 4 are global direction angles and
    q/p, which no choice of in-surface axes touches.
    """
    n = len(pos)
    e0, e1 = axes if axes is not None else frame(pos, mom, endcap)[:2]
    px, py, pz = mom.T
    pt = np.hypot(px, py)
    p2 = pt ** 2 + pz ** 2
    p = np.sqrt(p2)

    J = np.zeros((n, 5, 6))
    J[:, 0, :3] = e0
    J[:, 1, :3] = e1
    # phi_dir = atan2(py, px)
    J[:, 2, 3] = -py / pt ** 2
    J[:, 2, 4] = px / pt ** 2
    # theta_dir = atan2(pt, pz)
    J[:, 3, 3] = px * pz / (p2 * pt)
    J[:, 3, 4] = py * pz / (p2 * pt)
    J[:, 3, 5] = -pt / p2
    # q/p
    J[:, 4, 3] = -q * px / p ** 3
    J[:, 4, 4] = -q * py / p ** 3
    J[:, 4, 5] = -q * pz / p ** 3
    return J


def bound_in(pos, mom, endcap, q, axes=None):
    """J_in: d(pos, mom) / d(loc0, loc1, phi_dir, theta_dir, q/p), 6x5.

    The right inverse of bound_out restricted to the surface: a bound state
    has five degrees of freedom because the sixth -- moving along the surface
    normal -- is not a state, it is a different surface.

    `axes` overrides the cylinder frame, as in `bound_out`. Here it is the
    SOURCE surface's axes, so a chained filter has to carry the module it last
    updated on rather than rebuild them from the position.
    """
    n = len(pos)
    e0, e1 = axes if axes is not None else frame(pos, mom, endcap)[:2]
    px, py, pz = mom.T
    pt = np.hypot(px, py)
    p = np.sqrt(pt ** 2 + pz ** 2)
    phi_d = np.arctan2(py, px)
    th = np.arctan2(pt, pz)

    J = np.zeros((n, 6, 5))
    J[:, :3, 0] = e0
    J[:, :3, 1] = e1
    # p = |p| (sin th cos phi, sin th sin phi, cos th)
    J[:, 3, 2] = -p * np.sin(th) * np.sin(phi_d)
    J[:, 4, 2] = p * np.sin(th) * np.cos(phi_d)
    J[:, 3, 3] = p * np.cos(th) * np.cos(phi_d)
    J[:, 4, 3] = p * np.cos(th) * np.sin(phi_d)
    J[:, 5, 3] = -p * np.sin(th)
    # d|p|/d(q/p) = -q p^2  (|p| = |q / qop|, and q = +-1)
    dp = -q * p ** 2
    J[:, 3, 4] = dp * np.sin(th) * np.cos(phi_d)
    J[:, 4, 4] = dp * np.sin(th) * np.sin(phi_d)
    J[:, 5, 4] = dp * np.cos(th)
    return J


def F_bound(u, r1, z1, endcap, q, bz, s=None, src_endcap=None):
    """The 5x5 the filter wants: bound state in -> bound state out.

    `s` may be passed in: by the time a filter wants F it has already
    propagated, so the arc length is known and re-solving it would be
    double-counting the cost.

    `src_endcap` is the SOURCE surface's type and defaults to the
    destination's, which is right only when a track stays in one region. The
    bound frame at the source has to be the frame the previous update was
    written in, so chaining across a barrel-to-endcap transition needs both.
    """
    ok = np.ones(len(u), dtype=bool)
    if s is None:
        s, ok = solve_s(u, r1, z1, endcap, q, bz)
    if src_endcap is None:
        src_endcap = endcap
    Jg = helix_jacobian_global(u, s, q, bz, endcap)
    out = helix_state(u, s, q, bz)
    Jo = bound_out(out[:, :3], out[:, 3:], endcap, q)
    Ji = bound_in(u[:, :3], u[:, 3:], src_endcap, q)
    return np.einsum("nij,njk,nkl->nil", Jo, Jg, Ji), s, ok, out


def F_bound_plane(u, s, q, bz, nrm, e0, e1, src_axes):
    """`F_bound` with the module plane as the destination.

    Two things change and both have to, or the 5x5 describes a surface the
    filter is not on. The constraint the arc length is differentiated through
    becomes n . (P - c) = 0. The destination bound frame becomes the module's
    own (e0, e1), which is the frame `res_loc0_um` is measured in and the
    frame the residual is gated in; the barrel modules sit 150 mrad off the
    cylinder tangent, so leaving the cylinder frame here rotates the position
    block of the covariance against the residual it is weighting.

    `s` is not solved for. By the time a filter wants F it has the arc length
    from the prediction, and `plane_predict` is the only solver that should
    produce it.

    `src_axes` is the frame the incoming covariance is written in, which is
    the surface of the last update rather than anything derivable from the
    current position.
    """
    dummy = np.zeros(len(u), dtype=bool)
    Jg = helix_jacobian_global(u, s, q, bz, dummy, plane_n=nrm)
    out = helix_state(u, s, q, bz)
    Jo = bound_out(out[:, :3], out[:, 3:], dummy, q, axes=(e0, e1))
    Ji = bound_in(u[:, :3], u[:, 3:], dummy, q, axes=src_axes)
    return np.einsum("nij,njk,nkl->nil", Jo, Jg, Ji), out


# ------------------------------------------------------------------- checks

def fd_global(u, r1, z1, endcap, q, bz, eps):
    """Finite differences straight through the bisection solver."""
    n, m = len(u), 6
    F = np.zeros((n, 6, 6))
    for j in range(m):
        up, um = u.copy(), u.copy()
        h = eps * max(1.0, np.abs(u[:, j]).mean())
        up[:, j] += h
        um[:, j] -= h
        sp, _ = solve_s(up, r1, z1, endcap, q, bz)
        sm, _ = solve_s(um, r1, z1, endcap, q, bz)
        F[:, :, j] = (helix_state(up, sp, q, bz)
                      - helix_state(um, sm, q, bz)) / (2 * h)
    return F


def fd_bound(u, r1, z1, endcap, q, bz, eps, src_endcap=None):
    """Finite differences in the bound coordinates, re-solving each time.

    Perturbing loc0/loc1 moves the source point along the surface, so the
    perturbed state has to be re-embedded and re-solved -- which is the only
    honest way to check J_in.

    The step walks a UNIT direction in the 6-vector and the derivative is
    divided by that direction's length. Stepping by eps * J_in column instead
    makes the displacement scale as |J_in|^2, and since d|p|/d(q/p) = -q|p|^2
    reaches 1.2e4 at |eta| = 2.5, a 1e-6 step there became a 146 GeV kick --
    the FD, not the analytic form, was what disagreed.
    """
    n = len(u)
    if src_endcap is None:
        src_endcap = endcap
    Ji = bound_in(u[:, :3], u[:, 3:], src_endcap, q)
    F = np.zeros((n, 5, 5))
    for j in range(5):
        d = Ji[:, :, j]
        nrm = np.linalg.norm(d, axis=1)
        nrm[nrm == 0] = 1.0
        dhat = d / nrm[:, None]
        h = eps * np.maximum(1.0, np.abs(u).max(1))
        up = u + h[:, None] * dhat
        um = u - h[:, None] * dhat
        sp, _ = solve_s(up, r1, z1, endcap, q, bz)
        sm, _ = solve_s(um, r1, z1, endcap, q, bz)
        op = helix_state(up, sp, q, bz)
        om = helix_state(um, sm, q, bz)
        Jo = bound_out(0.5 * (op[:, :3] + om[:, :3]),
                       0.5 * (op[:, 3:] + om[:, 3:]), endcap, q)
        F[:, :, j] = np.einsum("nij,nj->ni", Jo, (op - om)) \
            / (2 * h[:, None]) * nrm[:, None]
    return F


def rel(a, b):
    """Error relative to the size of the matrix, not of the entry.

    Per-entry normalisation reports 1.0 wherever the analytic value is exactly
    zero and the finite difference returns solver noise -- which is every
    endcap jump's z row, since the constraint pins z_out = z1 and its true
    derivative is identically zero. Normalising by the largest entry of the
    same jump asks the question that matters: is the error small compared to
    F, which is what a covariance transport actually multiplies by.
    """
    scale = np.maximum(np.abs(a), np.abs(b)).reshape(len(a), -1).max(1)
    return np.abs(a - b) / scale[:, None, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="teacher_map_nomat_logpt.parquet")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--bz", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=20260806)
    args = ap.parse_args()

    t = pl.read_parquet(args.pairs)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(t), size=min(args.n, len(t)), replace=False)
    t = t[idx]

    u = np.column_stack([t[c].to_numpy() for c in
                         ("x", "y", "z", "px", "py", "pz")])
    r1 = t["r1"].to_numpy()
    z1 = t["z1"].to_numpy()
    endcap = t["endcap"].to_numpy()
    q = np.sign(t["qop"].to_numpy())
    # the source surface's own type -- ODD barrels are volumes 17, 24, 29 and
    # everything else in the tracker is a disc
    src_endcap = ~np.isin(t["surf0"].to_numpy() >> 56, BARREL_VOLUMES)

    F5, s, ok, out = F_bound(u, r1, z1, endcap, q, args.bz,
                             src_endcap=src_endcap)
    print(f"{len(u):,} jumps  ({endcap.sum():,} endcap, {(~endcap).sum():,} "
          f"barrel), {int((~ok).sum())} with no helix solution")

    # the solver's own answer has to be the one being differentiated
    err_pos = np.abs(out[:, :3] - np.column_stack(
        [t["helix_x"], t["helix_y"], t["helix_z"]]).astype(float)).max()
    print(f"reproduces the teacher's helix prediction to {err_pos:.2e} mm")

    m = ok
    print("\n=== dF/du, global 6x6, against finite differences through the "
          "solver ===")
    Jg = helix_jacobian_global(u, s, q, args.bz, endcap)
    for eps in (1e-5, 1e-6):
        N = fd_global(u, r1, z1, endcap, q, args.bz, eps)
        r = rel(Jg[m], N[m])
        print(f"  eps {eps:.0e}   median {np.median(r):.2e}   p99 "
              f"{np.percentile(r, 99):.2e}   worst {r.max():.2e}")

    print("\n=== how big the surface term is (a naive derivation drops it) ===")
    K = helix_jacobian_no_constraint(u, s, q, args.bz)
    scale = np.abs(Jg).reshape(len(u), -1).max(1)
    drop = np.abs(Jg - K).reshape(len(u), -1).max(1) / scale
    print(f"  |constraint term| / |F|, per jump:  median {np.median(drop):.2f}"
          f"   p90 {np.percentile(drop, 90):.2f}   max {drop.max():.2f}")
    print(f"  jumps where dropping it moves F by more than 10% of its own "
          f"scale: {100 * (drop > 0.1).mean():.0f}%")
    print("  So it is not a correction to F -- in the median it is larger "
          "than F.")

    print("\n=== F, bound 5x5, against finite differences in the bound "
          "coordinates ===")
    for eps in (1e-5, 1e-6):
        N5 = fd_bound(u, r1, z1, endcap, q, args.bz, eps,
                      src_endcap=src_endcap)
        r = rel(F5[m], N5[m])
        print(f"  eps {eps:.0e}   median {np.median(r):.2e}   p99 "
              f"{np.percentile(r, 99):.2e}   worst {r.max():.2e}")
    # The frame the source is written in is the previous surface's, not this
    # one's, and the two differ whenever a track crosses between barrel and
    # disc. Finite differences cannot catch this -- perturb in the wrong frame
    # and the check agrees with itself -- so it is measured as the difference
    # between the two F's on the jumps that change region.
    Fw, *_ = F_bound(u, r1, z1, endcap, q, args.bz, s=s)
    cross = src_endcap != endcap
    if cross.any():
        # per COLUMN, because d|p|/d(q/p) scales as |p|^2 and a whole-matrix
        # norm hides everything else behind that one column
        colscale = np.maximum(np.abs(F5).max(1, keepdims=True), 1e-30)
        d = np.abs(Fw - F5) / colscale
        worst = d[cross].reshape(int(cross.sum()), -1).max(1)
        print(f"  using the destination's frame at the source moves F by "
              f"{np.median(worst):.1f}x its own column scale on the "
              f"{100 * cross.mean():.0f}% of jumps that change region,")
        print(f"  entirely in the loc1 column -- e0 is phi_hat on both "
              f"surface types and only e1 differs (z_hat vs r_hat).")

    print("\n=== what F says about chaining ===")
    print("  the (loc0, phi) entry is the lever arm: a direction error at the "
          "source\n  becomes a position error at the destination, and its "
          "size is the reason\n  a chained filter can diverge.")
    lev = F5[:, 0, 2]
    print(f"  dloc0/dphi  median {np.median(np.abs(lev)):8.1f} mm   "
          f"p90 {np.percentile(np.abs(lev), 90):8.1f} mm   "
          f"max {np.abs(lev).max():8.1f} mm")
    print(f"  dloc0/d(q/p) median {np.median(np.abs(F5[:, 0, 4])):8.3g} "
          f"mm/(1/GeV)")

    big = np.repeat(u, max(1, 20000 // len(u)), axis=0)
    r1b = np.repeat(r1, max(1, 20000 // len(u)))
    z1b = np.repeat(z1, max(1, 20000 // len(u)))
    ecb = np.repeat(endcap, max(1, 20000 // len(u)))
    qb = np.repeat(q, max(1, 20000 // len(u)))
    sb, _ = solve_s(big, r1b, z1b, ecb, qb, args.bz)
    print("\n=== cost, numpy, batched ===")
    base = None
    for name, fn in (
            ("helix forward", lambda: helix_state(big, sb, qb, args.bz)),
            ("+ solve s", lambda: solve_s(big, r1b, z1b, ecb, qb, args.bz)),
            ("F global 6x6",
             lambda: helix_jacobian_global(big, sb, qb, args.bz, ecb)),
            ("F bound 5x5, s known",
             lambda: F_bound(big, r1b, z1b, ecb, qb, args.bz, s=sb))):
        fn()
        t0 = time.perf_counter()
        for _ in range(3):
            fn()
        dt = (time.perf_counter() - t0) / 3 / len(big) * 1e9
        base = base or dt
        print(f"  {name:22}{dt:9.0f} ns/jump{dt / base:8.2f}x forward")
    print("  s is passed in for the last row because the filter has already "
          "solved it by\n  the time it wants F.")

    # numpy's ratios are overhead-dominated exactly as they were for
    # dg_theta/dx: the forward is a handful of transcendentals while F builds
    # a per-jump (5,6)(6,6)(6,5) chain. The multiply-add count is the estimate
    # that transfers to a C++ kernel, and it is the one the speed claim needs.
    mac_frames = 5 * 6 * 6 + 5 * 6 * 5          # J_out . J_global . J_in
    mac_global = 90                             # the 6x6 build, counted above
    gtheta_fwd = 12 * 64 + 64 * 64 + 64 * 6
    gtheta_F2 = 64 * 64 * 2 + 12 * 64 * 2
    print(f"\n  multiply-adds, the estimate that transfers to a kernel:")
    print(f"    helix F, global + frames   ~{mac_global + mac_frames:,}")
    print(f"    g_theta forward             {gtheta_fwd:,}")
    print(f"    dg_theta/dx, 2 rows         {gtheta_F2:,}  (1.85x its forward, "
          f"measured)")
    print(f"  So the helix Jacobian is ~"
          f"{100 * (mac_global + mac_frames) / gtheta_fwd:.0f}% of g_theta's "
          f"forward pass. The\n  covariance transport's cost is dominated by "
          f"the network's Jacobian, not this one.")


def helix_jacobian_no_constraint(u, s, q, bz):
    """F with s held fixed. Only used to size the term it omits."""
    endcap = np.zeros(len(u), dtype=bool)
    J = helix_jacobian_global(u, s, q, bz, endcap)
    # rebuild the no-constraint part by subtracting what the constraint added
    x0, y0, z0, px, py, pz = u.T
    pt = np.hypot(px, py)
    phi0 = np.arctan2(py, px)
    kappa = -0.3 * q * bz / pt * MM
    a = 1.0 / kappa
    phi = phi0 + kappa * s
    sp, cp = np.sin(phi), np.cos(phi)
    sp0, cp0 = np.sin(phi0), np.cos(phi0)
    dpt = np.column_stack([px / pt, py / pt])
    dphi0 = np.column_stack([-py / pt ** 2, px / pt ** 2])
    dkappa = -(kappa / pt)[:, None] * dpt
    da = -(a ** 2)[:, None] * dkappa
    dphi = dphi0 + s[:, None] * dkappa
    n = len(px)
    K = np.zeros((n, 6, 6))
    K[:, 0, 0] = K[:, 1, 1] = K[:, 2, 2] = 1.0
    for k, col in enumerate((3, 4)):
        K[:, 0, col] = (da[:, k] * (sp - sp0)
                        + a * (cp * dphi[:, k] - cp0 * dphi0[:, k]))
        K[:, 1, col] = -(da[:, k] * (cp - cp0)
                         + a * (-sp * dphi[:, k] + sp0 * dphi0[:, k]))
        K[:, 2, col] = -pz * s * dpt[:, k] / pt ** 2
        K[:, 3, col] = dpt[:, k] * cp - pt * sp * dphi[:, k]
        K[:, 4, col] = dpt[:, k] * sp + pt * cp * dphi[:, k]
    K[:, 2, 5] = s / pt
    K[:, 5, 5] = 1.0
    return K


if __name__ == "__main__":
    main()
