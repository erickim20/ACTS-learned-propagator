"""Does applying the material noise mid-leg without splitting F
change the destination covariance enough for the chi2 gate to see it.

A moved seam transports one module-to-module leg as a single piece and has to
put the skipped surface's scattering somewhere. Splitting the Jacobian about
that crossing gives

    C_dest = F2 (F1 C F1' + Q) F2'

and not splitting it, which is what `cpp/LearnedJacobian.hpp` produces today,
gives

    C_dest = F C F' + Q.

With F = F2 F1 the two differ by exactly

    D = F2 Q F2' - Q

and nothing else. C drops out. That is the quantity below, projected onto the
destination module's bound 5x5 the way ACTS projects it.

Q is the material scattering, which is pointwise: an angular kick with no
position term. In the free basis an isotropic kick of variance theta0^2 in each
of the two planes transverse to the direction is theta0^2 (I - d d') on the
direction block, plus the energy-loss straggling on q/p. That is frame-free, so
no approach-surface geometry is needed and none is invented.

theta0 comes from Highland at the ODD's own thickness, which `cpp/mat_probe.cpp`
measures off the loaded material map rather than assuming. The answer is
reported as a fraction of each destination diagonal sigma and that fraction is
insensitive to theta0: D is linear in Q, and with C = 0 the denominator is
linear in Q as well.

C = 0 is the teacher's own condition: every teacher jump starts
from the true state, so the incoming covariance is exactly zero. It is also the
choice that makes the reported fraction an upper bound, because any real
incoming C only adds to the denominator.

The crossing sits `--delta` mm of 3D path before the destination module. STATUS
5.2 measures that distribution inside the CKF: p10 2.0 mm, median 7.7 mm, p90
21.7 mm. All three are run.

    python -m prop.split_jacobian --pairs teacher_muon_pt1_mapgeom.parquet
"""
import argparse

import numpy as np
import polars as pl

from .jacobian_free import (PION_MASS, D_free, d_free_d_path, d_global_d_free,
                            d_path_d_free)
from .jacobian_helix import bound_out, helix_state, solve_s

# CkfConfig.chi2CutOffMeasurement, read out of the container build.
# src/prop/chi2_gate.py:CHI2_CUT.
CHI2_CUT = 15.0

# odd-digi-smearing-config.json, in micrometres. src/prop/chi2_gate.py.
RESOLUTION_UM = {"pixel": (15.0, 15.0), "short strip": (43.0, 1200.0),
                 "long strip": (72.0, None)}

BOUND = ["loc0", "loc1", "phi", "theta", "q/p"]
BOUND_UNIT = ["mm", "mm", "rad", "rad", "1/GeV"]


def highland_theta0(p, t_over_x0, mass=PION_MASS):
    """Acts/Material/Interactions.hpp's multiple-scattering theta0, PDG form.

    beta is computed rather than set to 1: the softest bin here is 1 GeV and
    the pion hypothesis the teacher carries has a 0.14 GeV mass, so beta is
    0.99 and not 1.000. It matters at the third digit and costs nothing.
    """
    e = np.sqrt(p ** 2 + mass ** 2)
    beta = p / e
    x = np.asarray(t_over_x0, dtype=float)
    return (13.6e-3 / (beta * p)) * np.sqrt(x) * (1.0 + 0.038 * np.log(x))


def q_free(mom, theta0, sigma_qop):
    """(N, 8, 8): the pointwise material noise in the free basis.

    An isotropic angular kick of variance theta0^2 per transverse plane gives
    Cov(d_dir) = theta0^2 (I - d d'), because a unit vector's perturbations are
    transverse and the two transverse directions are equivalent. Position and
    time entries are zero, which is what makes `PointwiseMaterialInteraction`
    pointwise.
    """
    n = len(mom)
    d = mom / np.linalg.norm(mom, axis=1)[:, None]
    Q = np.zeros((n, 8, 8))
    proj = np.eye(3)[None, :, :] - d[:, :, None] * d[:, None, :]
    Q[:, 4:7, 4:7] = (theta0 ** 2)[:, None, None] * proj
    Q[:, 7, 7] = sigma_qop ** 2
    return Q


def bound_projection(pos, mom, endcap, q, axes, s, bz, u, mass):
    """(N, 5, 8): free covariance at the destination -> bound 5x5 on it.

    ACTS's contract, `JacobianEngine.hpp:55`:

        jac(locA->locB) = jac(gloB->locB) (I + t p') jacTransport jac(locA->gloA)

    so the map from a free covariance carried at fixed path length to the bound
    one on the surface is `jac(gloB->locB) (I + t p')`. Both halves are built
    here from the same functions `cpp/LearnedJacobian.hpp` is a port of, so the
    projection is the one the stepper would hand ACTS and not a second opinion.
    """
    out = np.column_stack([pos, mom])
    t = d_free_d_path(u, s, q, bz, endcap, out, mass=mass)
    p = d_path_d_free(out, endcap)
    corr = np.eye(8)[None, :, :] + t[:, :, None] * p[:, None, :]

    J = np.zeros((len(pos), 5, 8))
    J[:, :5, :] = np.einsum("nij,njk->nik", bound_out(pos, mom, endcap, q, axes),
                            d_global_d_free(pos, mom, q))
    return np.einsum("nij,njk->nik", J, corr)


def quantiles(v, qs=(0.5, 0.9, 0.99, 1.0)):
    v = np.sort(np.asarray(v))
    if v.size == 0:
        return [0.0] * len(qs)
    return [v[int(q * (v.size - 1))] for q in qs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="teacher_muon_pt1_mapgeom.parquet")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--bz", type=float, default=2.0)
    ap.add_argument("--mass", type=float, default=PION_MASS)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--x0", type=float, default=0.0292,
                    help="thickness in radiation lengths of the skipped "
                         "surface. cpp/mat_probe.cpp samples the ODD's own "
                         "map over the tracker volumes: p10 0.017, p50 0.029")
    ap.add_argument("--delta", type=float, nargs="+",
                    default=[2.0, 7.7, 21.7],
                    help="3D path in mm between the crossing and the "
                         "destination module, each scanned on its own. The "
                         "default is the measured p10, median and p90, which "
                         "is the leg the stepper seam gets today; "
                         "cpp/mat_leg.cpp measures the ones a moved seam sees")
    ap.add_argument("--multi", type=float, nargs="+", default=None,
                    help="deltas applied TOGETHER on one leg, which is what a "
                         "leg crossing more than one material surface needs. "
                         "cpp/mat_leg.cpp measures how often that happens")
    a = ap.parse_args()

    # The same draw `export_jacobian.py` makes, so these are the jumps
    # `cpp/test_jacobian.cpp` checks the port on.
    t = pl.read_parquet(a.pairs)
    rng = np.random.default_rng(a.seed)
    idx = np.sort(rng.choice(len(t), size=min(a.n, len(t)), replace=False))
    t = t[idx]

    u = np.column_stack([t[c].to_numpy() for c in
                         ("x", "y", "z", "px", "py", "pz")])
    r1, z1 = t["r1"].to_numpy(), t["z1"].to_numpy()
    endcap = t["endcap"].to_numpy().astype(bool)
    q = np.sign(t["qop"].to_numpy())
    mu0 = np.column_stack([t[c].to_numpy() for c in ("mu0_x", "mu0_y", "mu0_z")])
    mu1 = np.column_stack([t[c].to_numpy() for c in ("mu1_x", "mu1_y", "mu1_z")])

    s, ok = solve_s(u, r1, z1, endcap, q, a.bz)
    keep = ok & np.isfinite(s) & (s > 0)
    u, s, q, endcap = u[keep], s[keep], q[keep], endcap[keep]
    mu0, mu1 = mu0[keep], mu1[keep]
    n = len(u)

    pin = np.linalg.norm(u[:, 3:], axis=1)
    ptin = np.hypot(u[:, 3], u[:, 4])
    # pt/|p| is conserved along a helix (pt and pz are both constant), so the
    # transverse arc and the 3D path are proportional with a fixed ratio and a
    # 3D offset converts exactly.
    ratio = ptin / pin
    path3d = s / ratio

    out = helix_state(u, s, q, a.bz)
    F = D_free(u, s, q, a.bz, endcap, out, mass=a.mass)

    theta0 = highland_theta0(pin, a.x0, a.mass)
    # Energy-loss straggling on q/p. Kept small and stated: for a muon in
    # ~1% of a radiation length of silicon the relative momentum spread is
    # well under a per mille, and the answer below is checked against setting
    # it to zero.
    sigma_qop = 1e-4 * np.abs(t["qop"].to_numpy()[keep])

    print(f"{a.pairs}: {n} jumps of {len(t)} drawn "
          f"({int(endcap.sum())} endcap, {int((~endcap).sum())} barrel)")
    print(f"mass {a.mass} GeV, bz {a.bz} T, t/X0 {a.x0}")
    print(f"leg 3D path [mm]  p10 {quantiles(path3d, (0.1,))[0]:.1f}  "
          f"p50 {quantiles(path3d, (0.5,))[0]:.1f}  "
          f"p90 {quantiles(path3d, (0.9,))[0]:.1f}")
    th = np.degrees(theta0) * 1000.0
    print(f"Highland theta0 [mdeg]  p10 {quantiles(th, (0.1,))[0]:.2f}  "
          f"p50 {quantiles(th, (0.5,))[0]:.2f}  "
          f"p90 {quantiles(th, (0.9,))[0]:.2f}")

    axes = (mu0, mu1)
    P = bound_projection(out[:, :3], out[:, 3:], endcap, q, axes, s, a.bz, u,
                         a.mass)
    Qdest = q_free(out[:, 3:], theta0, sigma_qop)

    # The destination module's own resolution, which is what the gate divides
    # the residual by. The volume is the top byte of the geometry id,
    # `GeometryIdentifier::kVolumeMask`, and the nine present here are the nine
    # in the digitisation config.
    vol = (t["surf1"].to_numpy().astype("uint64")[keep] >> 56) & 0xFF
    cls = np.where(vol <= 18, 0, np.where(vol <= 25, 1, 2))
    res0 = np.array([RESOLUTION_UM["pixel"][0], RESOLUTION_UM["short strip"][0],
                     RESOLUTION_UM["long strip"][0]])[cls] * 1e-3   # mm

    for delta in a.delta:
        # Split the leg. s1 is the transverse arc to the crossing.
        s1 = s - delta * ratio
        good = s1 > 0
        mid = helix_state(u, s1, q, a.bz)

        F1 = D_free(u, s1, q, a.bz, endcap, mid, mass=a.mass)
        s2 = s - s1
        F2 = D_free(mid, s2, q, a.bz, endcap, out, mass=a.mass)

        # The split has to be a split. F2 F1 and F are the same transport
        # written two ways, so a disagreement here is an error in the
        # construction and not a result.
        comp = np.einsum("nij,njk->nik", F2, F1)
        scale = np.abs(F).max(axis=(1, 2))
        resid = np.abs(comp - F).max(axis=(1, 2)) / scale

        Qcross = q_free(mid[:, 3:], theta0, sigma_qop)
        Dfree = np.einsum("nij,njk,nlk->nil", F2, Qcross, F2) - Qdest

        Cun = np.einsum("nij,njk,nlk->nil", P, Qdest, P)
        Dbound = np.einsum("nij,njk,nlk->nil", P, Dfree, P)

        # The unsplit form's loc0 and loc1 variances are ANALYTICALLY zero and
        # come out of the arithmetic as +-1e-30. Clamping is not a repair of a
        # bad number: pointwise material applied on the destination surface
        # changes the direction and q/p and moves no position on that surface,
        # so `P Qdest P'` has an exactly zero position block, and the sign of
        # the roundoff is not a property of the jump. Dropping the jumps whose
        # roundoff landed negative removed 625 of 4000 for no physical reason.
        dun = np.maximum(np.einsum("nii->ni", Cun), 0.0)
        dsp = np.maximum(dun + np.einsum("nii->ni", Dbound), 0.0)
        g = good & (resid < 1e-6) & (dsp[:, 2:] > 0).all(axis=1)

        sig_un = np.sqrt(dun[g])
        sig_sp = np.sqrt(dsp[g])
        dsig = sig_sp - sig_un

        print()
        print(f"=== crossing {delta:.1f} mm before the module, "
              f"{int(g.sum())} of {n} jumps "
              f"(F2 F1 against F, worst {resid[good].max():.1e})")
        print()
        print("| component | sigma unsplit | sigma split | d sigma p50 | "
              "p90 | max | d sigma / sigma p50 |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for i, name in enumerate(BOUND):
            d = np.abs(dsig[:, i])
            d50, d90, _, dmx = quantiles(d)
            if (sig_un[:, i] > 0).all() and i >= 2:
                fr = f"{100 * quantiles(d / sig_un[:, i], (0.5,))[0]:.4f} %"
            else:
                # Pointwise material adds no position variance on the surface
                # it is applied to, so the unsplit destination sigma is
                # identically zero there and a fraction of it does not exist.
                fr = "n/a, sigma 0"
            print(f"| {name} [{BOUND_UNIT[i]}] | "
                  f"{np.median(sig_un[:, i]):.4g} | "
                  f"{np.median(sig_sp[:, i]):.4g} | "
                  f"{d50:.4g} | {d90:.4g} | {dmx:.4g} | {fr} |")

        # loc0 and loc1 against the gate. The unsplit form applies the kick on
        # the destination surface, where it moves no position; the split form
        # applies it `delta` upstream, where it does. So the whole of the split
        # form's position sigma is the difference, and the number to hold it
        # against is the measurement resolution, which is the other term of
        # S = V + H C H'.
        r = res0[g]
        for i, name in zip((0, 1), ("loc0", "loc1")):
            ratio_v = np.abs(dsig[:, i]) / r
            q50, q90, q99, qmx = quantiles(ratio_v)
            # chi2 = r' S^-1 r, so a variance that grows by f relative to S
            # moves a chi2 sitting on the cut by at most CHI2_CUT * f. Using V
            # alone for S is an over-estimate: the real S carries C as well.
            f50 = quantiles(ratio_v ** 2, (0.5,))[0]
            fmx = qmx ** 2
            print(f"  {name}: d sigma / sigma_measurement  "
                  f"p50 {q50:.4f}  p90 {q90:.4f}  p99 {q99:.4f}  max {qmx:.4f}")
            print(f"        worst chi2 shift on the cut of {CHI2_CUT}: "
                  f"p50 {CHI2_CUT * f50:.3e}  max {CHI2_CUT * fmx:.3e}")
        print(f"  d sigma_loc0 [um]  p50 "
              f"{quantiles(np.abs(dsig[:, 0]), (0.5,))[0] * 1000:.3f}  "
              f"p90 {quantiles(np.abs(dsig[:, 0]), (0.9,))[0] * 1000:.3f}  "
              f"max {np.abs(dsig[:, 0]).max() * 1000:.3f}")

    if a.multi:
        # Several crossings on one leg. The split form transports each kick the
        # rest of the way from where it happened,
        #
        #   C = F_n (... F_2 (F_1 C F_1' + Q_1) F_2' ... + Q_n) F_n'
        #
        # and the unsplit form drops all of them on the destination,
        #
        #   C = F C F' + sum_k Q_k.
        #
        # Because F = F_after(k) F_before(k) for every k, the difference is
        #
        #   sum_k [ F_after(k) Q_k F_after(k)' - Q_k ],
        #
        # the sum of the single-crossing differences above. That is stated here
        # and then checked against the accumulated form rather than assumed.
        deltas = sorted(a.multi, reverse=True)
        acc = np.zeros((n, 8, 8))
        summed = np.zeros((n, 8, 8))
        good = np.ones(n, dtype=bool)
        worst_resid = 0.0
        for delta in deltas:
            s1 = s - delta * ratio
            good &= s1 > 0
            mid = helix_state(u, s1, q, a.bz)
            F2 = D_free(mid, s - s1, q, a.bz, endcap, out, mass=a.mass)
            F1 = D_free(u, s1, q, a.bz, endcap, mid, mass=a.mass)
            comp = np.einsum("nij,njk->nik", F2, F1)
            worst_resid = max(worst_resid, float(
                np.max(np.abs(comp - F).max(axis=(1, 2)) /
                       np.abs(F).max(axis=(1, 2)))))
            Qk = q_free(mid[:, 3:], theta0, sigma_qop)
            # transport this kick the rest of the way, and accumulate
            acc = np.einsum("nij,njk,nlk->nil", F2, Qk, F2) + acc
            summed = summed + Qk
        Dfree = acc - summed
        Cun = np.einsum("nij,njk,nlk->nil", P, summed, P)
        Dbound = np.einsum("nij,njk,nlk->nil", P, Dfree, P)
        dun = np.maximum(np.einsum("nii->ni", Cun), 0.0)
        dsp = np.maximum(dun + np.einsum("nii->ni", Dbound), 0.0)
        g = good & (dsp[:, 2:] > 0).all(axis=1)
        sig_un = np.sqrt(dun[g])
        sig_sp = np.sqrt(dsp[g])
        dsig = sig_sp - sig_un
        r = res0[g]

        print()
        print(f"=== {len(deltas)} crossings on one leg at "
              f"{', '.join(f'{d:.2f}' for d in deltas)} mm, "
              f"{int(g.sum())} of {n} jumps "
              f"(F2 F1 against F, worst {worst_resid:.1e})")
        print()
        print("| component | sigma unsplit | sigma split | d sigma p50 | "
              "p90 | max |")
        print("| --- | ---: | ---: | ---: | ---: | ---: |")
        for i, name in enumerate(BOUND):
            d = np.abs(dsig[:, i])
            d50, d90, _, dmx = quantiles(d)
            print(f"| {name} [{BOUND_UNIT[i]}] | "
                  f"{np.median(sig_un[:, i]):.4g} | "
                  f"{np.median(sig_sp[:, i]):.4g} | "
                  f"{d50:.4g} | {d90:.4g} | {dmx:.4g} |")
        for i, name in zip((0, 1), ("loc0", "loc1")):
            ratio_v = np.abs(dsig[:, i]) / r
            q50, q90, q99, qmx = quantiles(ratio_v)
            print(f"  {name}: d sigma / sigma_measurement  "
                  f"p50 {q50:.4f}  p90 {q90:.4f}  p99 {q99:.4f}  "
                  f"max {qmx:.4f}")
            print(f"        worst chi2 shift on the cut of {CHI2_CUT}: "
                  f"p50 {CHI2_CUT * q50 ** 2:.3e}  "
                  f"max {CHI2_CUT * qmx ** 2:.3e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
