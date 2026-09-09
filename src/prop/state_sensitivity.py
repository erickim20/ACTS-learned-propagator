"""What the correction is worth when it is applied at a filtered state.

`g_theta` is fitted on jumps whose source state is
the simulated truth. Inside the CKF it is evaluated at the Kalman estimate,
which carries the filter's own error: median sqrt(C_00) of 64.5 µm on pixels
against a correction of 92.9 µm at 1 GeV (§6.12, §6.14). A correction evaluated
at the wrong point is not a smaller correction, it is a different one.

The test. Perturb the source position by the measured predicted covariance,
propagate the helix from the perturbed state, and ask the same question §6.14
asks: is what the model adds smaller than what the helix got wrong. Everything
downstream of the perturbation is recomputed, so the helix's own error moves
with the state and the comparison stays honest.

PRE-REGISTERED, written before the first run:

  scale 0     must reproduce §6.14's table to the digit. It is the same
              calculation on the same rows and any difference is a harness bug,
              not a result.
  scale 1     if the train-at-truth apply-at-estimate mismatch is what §6.16
              measured, the worse-than-off fractions rise materially at every
              output, and loc0 in particular crosses from about 30% toward 50%
              below 5 GeV, because §6.16 has loc0 costing 14.2 efficiency
              points in the filter while §6.14 has it helping on 70% of jumps.
  no movement if the fractions are flat in the scale, the mismatch is not the
              mechanism, and 22b should not gain an input-noise term.

Two limits on the claim, both from the table rather than from choice:

  `sigma_C` carries `sig_c0_um` and `sig_c1_um` and nothing else, so this
  perturbs position only. A direction error acts over the step's lever arm and
  would move the prediction further, so whatever this measures is a lower bound
  on the disruption.

  The perturbation is applied in the plane perpendicular to the momentum, with
  the two measured widths as its axes. A bound position error lives in the
  source surface, whose axes are not in the teacher table; the two frames
  differ by the incidence angle and the magnitudes are the measured ones either
  way.

    python -m prop.state_sensitivity
    python -m prop.state_sensitivity --scales 0,0.25,0.5,1,2
"""
import argparse

import numpy as np
import polars as pl

from .chi2_gate import measured_cov, resolutions
from .closed_loop import Model, features_plane
from .helix_variants import FieldMap
from .make_teacher_pairs import plane_predict
from .output_response import P_EDGES, p_label, table

MM_TO_UM = 1000.0


def orthonormal_to(p):
    """Two unit vectors spanning the plane perpendicular to each row of `p`."""
    z = np.zeros_like(p)
    z[:, 2] = 1.0
    a = np.cross(p, z)
    small = np.linalg.norm(a, axis=1) < 1e-9
    if small.any():
        alt = np.zeros_like(p)
        alt[:, 0] = 1.0
        a[small] = np.cross(p[small], alt[small])
    a /= np.linalg.norm(a, axis=1)[:, None]
    b = np.cross(p, a)
    b /= np.linalg.norm(b, axis=1)[:, None]
    return a, b


def evaluate(m, fm, t, u, q, cen, nrm, e0, e1, hit0, hit1, p1):
    """One pass of the deployed path from a given source state.

    Returns the helix's own error and what the model adds, both in the units
    §6.14 reports: µm for the position pair, mrad for the directions, 1/GeV for
    q/p.
    """
    hx, hy, hz, hpx, hpy, hpz, _, ok = plane_predict(
        u[:, 0], u[:, 1], u[:, 2], u[:, 3], u[:, 4], u[:, 5], q, 2.0, cen, nrm)
    hpos = np.column_stack([hx, hy, hz])
    hmom = np.column_stack([hpx, hpy, hpz])

    qop = q / np.linalg.norm(u[:, 3:], axis=1)
    X = features_plane(u, qop, cen, nrm, e0, e1, hpos, fm)
    raw = m.forward((X - m.mu) / m.sd)

    # The helix miss in the module's own axes, the frame `res_loc0_um` is
    # written in, recomputed rather than read so it follows the perturbation.
    d = hpos - cen
    b0 = ((d * e0).sum(1) - hit0) * MM_TO_UM
    b1 = ((d * e1).sum(1) - hit1) * MM_TO_UM

    phi_h = np.arctan2(hmom[:, 1], hmom[:, 0])
    th_h = np.arctan2(np.hypot(hmom[:, 0], hmom[:, 1]), hmom[:, 2])
    phi_1 = np.arctan2(p1[:, 1], p1[:, 0])
    th_1 = np.arctan2(np.hypot(p1[:, 0], p1[:, 1]), p1[:, 2])
    dphi = (phi_1 - phi_h + np.pi) % (2 * np.pi) - np.pi
    # |p_helix| == |p_source|, so the helix's q/p prediction is the source's.
    # A position perturbation does not move it, which is why this row is flat
    # in the scale by construction rather than as a result.
    dqop = q / np.linalg.norm(p1, axis=1) - qop

    tgt = np.column_stack([b0, b1, dphi * 1e3, (th_1 - th_h) * 1e3, dqop])
    return tgt, raw, ok


def worse(tgt, applied, p, keep):
    """Fraction made worse per |p| bin, and the two medians behind it."""
    left = tgt - applied
    out = []
    for lo, hi in zip(P_EDGES[:-1], P_EDGES[1:]):
        s = (p >= lo) & (p < hi) & keep
        if not s.any():
            out.append((p_label(lo, hi), 0, np.nan, np.nan, np.nan))
            continue
        out.append((p_label(lo, hi), int(s.sum()),
                    float(np.median(np.abs(tgt[s]))),
                    float(np.median(np.abs(applied[s]))),
                    float((np.abs(left[s]) > np.abs(tgt[s])).mean())))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gtheta_plane14.npz")
    ap.add_argument("--teacher", default="teacher_phys.parquet")
    ap.add_argument("--scales", default="0,0.5,1.0",
                    help="multiples of sqrt(C) to perturb the source by")
    ap.add_argument("--seed", type=int, default=20260818)
    a = ap.parse_args()

    scales = [float(s) for s in a.scales.split(",")]
    m = Model(a.model)
    fm = FieldMap(m.field)
    t = pl.read_parquet(a.teacher)
    rng = np.random.default_rng(a.seed)

    col = lambda *c: np.column_stack([t[x].to_numpy() for x in c]).astype(float)
    u0 = col("x", "y", "z", "px", "py", "pz")
    cen = col("mc_x", "mc_y", "mc_z")
    nrm = col("mn_x", "mn_y", "mn_z")
    e0 = col("mu0_x", "mu0_y", "mu0_z")
    e1 = col("mu1_x", "mu1_y", "mu1_z")
    p1 = col("px1", "py1", "pz1")
    hit0, hit1 = t["hit_loc0"].to_numpy(), t["hit_loc1"].to_numpy()
    qop0 = t["qop"].to_numpy()
    q = np.sign(qop0)
    p = 1.0 / np.abs(qop0)

    # The widths to perturb by: the CKF's own predicted covariance on the same
    # (class, pT bin, |eta| bin) cells the position outputs are scaled with.
    vol = t["surf1"].to_numpy() >> 56
    c0, c1 = measured_cov(t, vol, m.cov)
    res = resolutions(m.digi)
    one_d = np.array([res[v][1] is None for v in vol])
    sig = np.column_stack([np.hypot(np.array([res[v][0] for v in vol]), c0),
                           np.hypot(np.array([1.0 if res[v][1] is None
                                              else res[v][1] for v in vol]),
                                    c1)])

    ax, bx = orthonormal_to(u0[:, 3:])
    n0, n1 = rng.standard_normal(len(t)), rng.standard_normal(len(t))
    step = (c0 * n0)[:, None] * ax + (c1 * n1)[:, None] * bx   # µm

    print(f"# {a.model}")
    print()
    print(f"    rows        {len(t):,}")
    print(f"    sqrt(C_00)  median {np.median(c0):.1f} um   "
          f"sqrt(C_11) median {np.median(c1):.1f} um")
    print(f"    seed        {a.seed}")
    print()
    print("Position only: `sigma_C` carries no direction entry, so this is a")
    print("lower bound on what a filtered state does to the correction. The q/p")
    print("row cannot move at all under a position perturbation and is printed")
    print("as the control that says the harness is not inventing motion.")
    print()

    names = [("loc0", "um"), ("loc1", "um"), ("phi", "mrad"),
             ("theta", "mrad"), ("q/p", "1/GeV")]
    rows = {k: [] for k in range(5)}
    ref_applied = None
    response = []
    for sc in scales:
        u = u0.copy()
        u[:, :3] += sc * step * 1e-3          # µm -> mm
        tgt, raw, ok = evaluate(m, fm, t, u, q, cen, nrm, e0, e1,
                                hit0, hit1, p1)
        applied = np.column_stack([raw[:, 0] * sig[:, 0],
                                   raw[:, 1] * sig[:, 1],
                                   raw[:, 2] * m.v1_scale[0] * 1e3,
                                   raw[:, 3] * m.v1_scale[1] * 1e3,
                                   raw[:, 4] * m.v1_scale[2]])
        good = ok & np.isfinite(tgt).all(1)
        print(f"    scale {sc:>4}  solved {good.mean():.4%}  "
              f"median |shift| {np.median(np.abs(sc * step).sum(1)):.1f} um")
        for k in range(5):
            keep = good & (~one_d if k == 1 else np.ones(len(t), bool))
            rows[k].append((sc, worse(tgt[:, k], applied[:, k], p, keep)))
        if ref_applied is None:
            ref_applied = applied.copy()
            ref_tgt = tgt.copy()
        response.append((sc, np.abs(applied - ref_applied),
                         np.abs(tgt - ref_tgt), good))
    print()

    # How much the model's own output moved. This is the question underneath
    # the table above: a correction that does not respond to the state cannot
    # be damaged by getting the state wrong, and cannot be a function of it
    # either.
    print("## How far the correction itself moved")
    print()
    body = []
    for k, (name, unit) in enumerate(names):
        keep = response[0][3] & (~one_d if k == 1 else np.ones(len(t), bool))
        base = float(np.median(np.abs(ref_applied[keep, k])))
        r = [name, f"{base:.4g}"]
        for sc, da, dt, good in response[1:]:
            r.append(f"{np.median(da[keep, k]) / base:.4f}" if base else "n/a")
        for sc, da, dt, good in response[1:]:
            r.append(f"{np.median(dt[keep, k]):.4g}")
        body.append(r)
    hdr = (["output", "median |applied|"]
           + [f"|d applied|/|applied| at {s:g}" for s, _, _, _ in response[1:]]
           + [f"median |d helix error| at {s:g}" for s, _, _, _ in response[1:]])
    print(table(hdr, body))
    print()
    print("Units are each output's own. The middle columns are the model's")
    print("response to the perturbation as a fraction of what it applies; the")
    print("last are how far the thing it is supposed to remove moved under the")
    print("same perturbation.")
    print()

    for k, (name, unit) in enumerate(names):
        print(f"## {name}, made worse than switching it off [%]")
        print()
        hdr = ["|p| [GeV]", "jumps"] + [f"scale {s:g}" for s, _ in rows[k]]
        body = []
        base = rows[k][0][1]
        for i in range(len(base)):
            if base[i][1] == 0:
                continue
            r = [base[i][0], f"{base[i][1]:,}"]
            for _, w in rows[k]:
                r.append("n/a" if w[i][1] == 0 else f"{100 * w[i][4]:.1f}")
            body.append(r)
        print(table(hdr, body))
        print()
        print(f"    median helix error / applied [{unit}], by scale:")
        for s, w in rows[k]:
            tot = [x for x in w if x[1]]
            print(f"      scale {s:>4}  "
                  + "  ".join(f"{x[2]:.3g}/{x[3]:.3g}" for x in tot))
        print()


if __name__ == "__main__":
    main()
