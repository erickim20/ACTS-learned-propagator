"""C -- the helix Jacobian in the form ACTS's covariance engine expects.

`jacobian_helix.py` produces F in bound coordinates, which is what the filter's
gate and gain want. ACTS never forms that directly. It keeps three things and
multiplies them itself, and the contract is written out in
`Acts/Propagator/detail/JacobianEngine.hpp`:

    jac(locA->locB) = jac(gloB->locB) (1 + pathCorrection(gloB))
                      jacTransport(gloA->gloB) jac(locA->gloA)

so what belongs in `state.jacTransport` is the transport with the destination
surface not differentiated through, and the surface constraint arrives
separately as `(I + t p)` built from

    t = state.derivative,  d(free) / d(path length), set by the stepper
    p = surface.freeToPathDerivative,  d(path length) / d(free), set by ACTS

This is the opposite of the bound picture, where the constraint is inside F.
`EigenStepper.ipp:336-339` asserts the resulting block structure, and it is not
a soft convention:

    jacTransport = [ I4   J12 ]      4x4 blocks over (pos, time), (dir, q/p)
                   [ 04   J22 ]

The unconstrained helix Jacobian satisfies both blocks exactly. At fixed path
length the endpoint moves rigidly with the source position, and in a uniform
field the outgoing direction does not depend on the source position at all.

Two things about the parameterisation.

**ACTS's path is the 3D arc length; this project's `s` is the transverse one.**
They differ by p/pt, which is a function of the state, so the Jacobian at fixed
path length is a different matrix in the two. The difference is rank one along
the trajectory tangent, and `p t = -1` exactly, so `(I + t p)` annihilates it
and the bound-to-bound answer comes out the same either way. That is measured
below rather than argued. The 3D one is still what has to be written, because
`state.derivative` and `pathAccumulated` are 3D everywhere else in ACTS and the
block assert is stated for it.

**The direction is a unit vector, not a momentum.** ACTS carries `dir` and
`q/p` separately, so `mom = (q/qop) dir` one way and `dir = mom/|mom|`,
`qop = q/|mom|` the other.

One thing that is a trap. The tangent `t` must come from the trajectory being
differentiated, not from a fresh derivation. This project's helix uses
0.3 GeV/(T m); rederiving `d(dir)/d(path)` from the Lorentz force with
0.299792458 disagrees by 7e-4 and moves the worst-case correspondence from
3e-15 to 6e-7, which is small enough to read as rounding and is not.

Units need no conversion: ACTS has mm = 1, GeV = 1, e = 1, read out of
`Acts/Definitions/Units.hpp` in the pinned image, and this project has used mm
and GeV throughout. With c = 1 a time is a length.

Usage:
    python -m prop.jacobian_free --pairs teacher_phys.parquet
"""
import argparse

import numpy as np
import polars as pl

from .jacobian_helix import (BARREL_VOLUMES, F_bound, bound_in, bound_out,
                             helix_jacobian_global,
                             helix_jacobian_no_constraint, helix_state, rel,
                             solve_s)

# ACTS free layout, Acts/Definitions/TrackParametrization.hpp
FTIME = 3
# ACTS bound layout; the first five are the order jacobian_helix already uses
BTIME = 5

PION_MASS = 0.13957039          # GeV


# --------------------------------------------------------- change of variables

def d_free_d_global(pos, mom, q):
    """(N, 8, 6): d(free) / d(x, y, z, px, py, pz).

    The time row is zero. Time is not a function of the spatial state, it is an
    independent coordinate; the transport's dependence on it is put in by the
    callers below.
    """
    n = len(pos)
    p = np.linalg.norm(mom, axis=1)
    d = mom / p[:, None]
    A = np.zeros((n, 8, 6))
    A[:, 0, 0] = A[:, 1, 1] = A[:, 2, 2] = 1.0
    A[:, 4:7, 3:6] = (np.eye(3)[None] - d[:, :, None] * d[:, None, :]) \
        / p[:, None, None]
    A[:, 7, 3:6] = -(q / p ** 2)[:, None] * d
    return A


def d_global_d_free(pos, mom, q):
    """(N, 6, 8): d(x, y, z, px, py, pz) / d(free)."""
    n = len(pos)
    p = np.linalg.norm(mom, axis=1)
    d = mom / p[:, None]
    B = np.zeros((n, 6, 8))
    B[:, 0, 0] = B[:, 1, 1] = B[:, 2, 2] = 1.0
    B[:, 3:6, 4:7] = np.eye(3)[None] * p[:, None, None]
    B[:, 3:6, 7] = (-q * p ** 2)[:, None] * d
    return B


# ------------------------------------------------- transverse s vs 3D path

def _dln_pt_over_p(mom):
    """(N, 6): d ln(pt / |p|) / d(state). Position columns are zero.

    s_transverse = path3d * pt / |p|, so this is the whole difference between
    holding one fixed and holding the other.
    """
    px, py, pz = mom.T
    pt = np.hypot(px, py)
    p2 = px ** 2 + py ** 2 + pz ** 2
    out = np.zeros((len(mom), 6))
    out[:, 3] = px / pt ** 2 - px / p2
    out[:, 4] = py / pt ** 2 - py / p2
    out[:, 5] = -pz / p2
    return out


def helix_jacobian_fixed_path(u, s, q, bz, endcap):
    """(N, 6, 6): d(out) / d(in) holding the 3D path length fixed.

    This is what `state.jacTransport` accumulates. `helix_jacobian_global`
    holds the destination surface fixed and `helix_jacobian_no_constraint`
    holds the transverse arc length fixed; neither is this.
    """
    K = helix_jacobian_no_constraint(u, s, q, bz)
    _, _, dds = helix_jacobian_global(u, s, q, bz, endcap, with_ds=True)
    return K + dds[:, :, None] * (s[:, None] * _dln_pt_over_p(u[:, 3:]))[:, None, :]


# ------------------------------------------------------------ time of flight

def _energy(mom, mass):
    return np.sqrt((mom ** 2).sum(1) + mass ** 2)


def time_of_flight(u, s, mass=PION_MASS):
    """T = s E / pt. c = 1 in ACTS units, so this is a length."""
    px, py = u[:, 3], u[:, 4]
    return s * _energy(u[:, 3:], mass) / np.hypot(px, py)


def _d_time_fixed_path(u, s, mass):
    """(N, 6): dT/d(state) at fixed 3D path length. T = path3d E / |p|."""
    mom = u[:, 3:]
    p = np.linalg.norm(mom, axis=1)
    E = _energy(mom, mass)
    path3d = s * p / np.hypot(u[:, 3], u[:, 4])
    out = np.zeros((len(u), 6))
    # d(E/|p|)/dp_i = p_i/(E |p|) - E p_i/|p|^3
    out[:, 3:6] = path3d[:, None] * (
        mom / (E * p)[:, None] - (E / p ** 3)[:, None] * mom)
    return out


# ---------------------------------------------------- what the stepper writes

def D_free(u, s, q, bz, endcap, out, mass=PION_MASS):
    """(N, 8, 8), the matrix `state.jacTransport` is multiplied by."""
    K = helix_jacobian_fixed_path(u, s, q, bz, endcap)
    A = d_free_d_global(out[:, :3], out[:, 3:], q)
    B = d_global_d_free(u[:, :3], u[:, 3:], q)
    D = np.einsum("nij,njk,nkl->nil", A, K, B)
    D[:, FTIME, :] = np.einsum("nj,njk->nk", _d_time_fixed_path(u, s, mass), B)
    D[:, FTIME, FTIME] = 1.0
    return D


def d_free_d_path(u, s, q, bz, endcap, out, mass=PION_MASS):
    """(N, 8): `state.derivative`, d(free) / d(3D path length) at the endpoint.

    EigenStepper sets head<3> to the direction and segment<3>(4) to the last
    RKN direction derivative. For a helix the same quantity is the trajectory
    tangent `helix_jacobian_global` already computes, rescaled from transverse
    to 3D path by pt/|p|.

    It is taken from there rather than re-derived from the Lorentz force on
    purpose. The helix in this project uses 0.3 GeV/(T m) and not
    0.299792458, so a re-derivation with the exact constant disagrees with the
    trajectory being differentiated by 7e-4 -- small, wrong, and invisible in
    anything but a Jacobian check.
    """
    _, _, dds = helix_jacobian_global(u, s, q, bz, endcap, with_ds=True)
    mom = out[:, 3:]
    p = np.linalg.norm(mom, axis=1)
    pt = np.hypot(mom[:, 0], mom[:, 1])
    tg = dds * (pt / p)[:, None]                     # d(pos, mom)/d(path3d)

    A = d_free_d_global(out[:, :3], mom, q)
    t = np.einsum("nij,nj->ni", A, tg)
    t[:, FTIME] = _energy(mom, mass) / p
    return t


def d_path_d_free(out, endcap):
    """(N, 8): `surface.freeToPathDerivative`, d(3D path length) / d(free).

    For a surface g(pos) = 0 reached along `dir`, the path from a displaced
    point is -g/(grad g . dir), so the derivative is -grad g / (grad g . dir).
    ACTS computes this itself from the surface; it is written out here only so
    the identity below can be checked without linking against ACTS.
    """
    pos, mom = out[:, :3], out[:, 3:]
    d = mom / np.linalg.norm(mom, axis=1)[:, None]
    r = np.hypot(pos[:, 0], pos[:, 1])
    grad = np.where(
        endcap[:, None],
        np.column_stack([np.zeros(len(out)), np.zeros(len(out)),
                         np.ones(len(out))]),
        np.column_stack([pos[:, 0] / r, pos[:, 1] / r, np.zeros(len(out))]))
    p = np.zeros((len(out), 8))
    p[:, 0:3] = -grad / (grad * d).sum(1)[:, None]
    return p


# ------------------------------------------------------- bound <-> free maps

def bound_to_free(pos, mom, endcap, q):
    """(N, 8, 6) in ACTS's bound order, time last."""
    n = len(pos)
    M = np.zeros((n, 8, 6))
    M[:, :, :5] = np.einsum("nij,njk->nik",
                            d_free_d_global(pos, mom, q),
                            bound_in(pos, mom, endcap, q))
    M[:, FTIME, BTIME] = 1.0
    return M


def free_to_bound(pos, mom, endcap, q):
    """(N, 6, 8) in ACTS's bound order, time last."""
    n = len(pos)
    M = np.zeros((n, 6, 8))
    M[:, :5, :] = np.einsum("nij,njk->nik",
                            bound_out(pos, mom, endcap, q),
                            d_global_d_free(pos, mom, q))
    M[:, BTIME, FTIME] = 1.0
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="teacher_phys.parquet")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--bz", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=20260812)
    args = ap.parse_args()

    t = pl.read_parquet(args.pairs)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(t), size=min(args.n, len(t)), replace=False)
    t = t[idx]

    u = np.column_stack([t[c].to_numpy() for c in
                         ("x", "y", "z", "px", "py", "pz")])
    r1, z1 = t["r1"].to_numpy(), t["z1"].to_numpy()
    endcap = t["endcap"].to_numpy()
    q = np.sign(t["qop"].to_numpy())
    src_endcap = ~np.isin(t["surf0"].to_numpy() >> 56, BARREL_VOLUMES)

    s, ok = solve_s(u, r1, z1, endcap, q, args.bz)
    out = helix_state(u, s, q, args.bz)
    D = D_free(u, s, q, args.bz, endcap, out)
    print(f"{len(u):,} jumps  ({endcap.sum():,} endcap, "
          f"{(~endcap).sum():,} barrel)")

    print("\n=== the change of variables ===")
    A = d_free_d_global(u[:, :3], u[:, 3:], q)
    B = d_global_d_free(u[:, :3], u[:, 3:], q)
    BA = np.einsum("nij,njk->nik", B, A)
    print(f"  B A = I6 to {np.abs(BA - np.eye(6)[None]).max():.2e}")
    AB = np.einsum("nij,njk->nik", A, B)
    idem = np.abs(np.einsum("nij,njk->nik", AB, AB) - AB).max()
    print(f"  A B is idempotent to {idem:.2e}, so it is a projector")

    print("\n=== the block structure EigenStepper.ipp:336 asserts ===")
    tl = np.abs(D[:, :4, :4] - np.eye(4)[None]).max()
    bl = np.abs(D[:, 4:, :4]).max()
    print(f"  topLeft 4x4 - I    max {tl:.2e}")
    print(f"  bottomLeft 4x4     max {bl:.2e}")
    print("  Both hold to machine precision, so the assert does not fire and "
          "the blocked")
    print("  multiply ACTS does in place is the right one for this matrix too.")

    print("\n=== the surface constraint, applied ACTS's way ===")
    tvec = d_free_d_path(u, s, q, args.bz, endcap, out)
    pvec = d_path_d_free(out, endcap)
    corr = np.eye(8)[None] + tvec[:, :, None] * pvec[:, None, :]
    Dc = np.einsum("nij,njk->nik", corr, D)

    M_in = bound_to_free(u[:, :3], u[:, 3:], src_endcap, q)
    M_out = free_to_bound(out[:, :3], out[:, 3:], endcap, q)
    F6 = np.einsum("nij,njk,nkl->nil", M_out, Dc, M_in)
    F5, *_ = F_bound(u, r1, z1, endcap, q, args.bz, s=s, src_endcap=src_endcap)
    r = rel(F6[:, :5, :5], F5)
    print(f"  freeToBound (I + t p) jacTransport boundToFree  vs  F_bound:")
    print(f"    median {np.median(r):.2e}   p99 {np.percentile(r, 99):.2e}   "
          f"worst {r.max():.2e}")
    print("  This is the correspondence item 1 asks for. Everything the "
          "stepper has to")
    print("  supply is on the left: jacTransport and the derivative t. ACTS "
          "supplies p.")

    print("\n=== why the correction can absorb the choice of path variable ===")
    pt_dot = np.einsum("ni,ni->n", pvec, tvec)
    print(f"  p . t = {pt_dot.min():.6f} to {pt_dot.max():.6f}, so (I + t p) "
          f"annihilates the tangent")
    Kwrong = helix_jacobian_no_constraint(u, s, q, args.bz)
    Dwrong = np.einsum("nij,njk,nkl->nil",
                       d_free_d_global(out[:, :3], out[:, 3:], q), Kwrong, B)
    Dwrong[:, FTIME, FTIME] = 1.0
    diff = D - Dwrong
    print(f"  the two jacTransports differ by  median "
          f"{np.median(np.abs(diff).reshape(len(u), -1).max(1)):.2e}")
    Fw = np.einsum("nij,njk,nkl->nil", M_out,
                   np.einsum("nij,njk->nik", corr, Dwrong), M_in)
    rw = rel(Fw[:, :5, :5], F5)
    print(f"  and the bound 5x5 they give agrees to  worst {rw.max():.2e}")
    trow = np.abs(Fw[:, BTIME, :] - F6[:, BTIME, :]).max()
    print(f"  what survives is the time row, which differs by {trow:.2e}")
    print("  So the choice does not reach the five components the filter "
          "gates on. The 3D")
    print("  one is written anyway: state.derivative and pathAccumulated are "
          "3D everywhere")
    print("  else in ACTS, and the block assert is stated for it.")

    print("\n=== scale, as a sanity check on units ===")
    L3d = s * np.linalg.norm(u[:, 3:], axis=1) / np.hypot(u[:, 3], u[:, 4])
    print(f"  3D path length      median {np.median(L3d):8.1f} mm")
    print(f"  |d(pos)/d(dir)|     median "
          f"{np.median(np.linalg.norm(D[:, 0:3, 4:7], axis=(1, 2)) / np.sqrt(3)):8.1f} mm")
    print(f"  time of flight      median {np.median(time_of_flight(u, s)):8.1f} mm")
    print(f"  jumps with no helix solution: {int((~ok).sum())}")


if __name__ == "__main__":
    main()
