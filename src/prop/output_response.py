"""What each of the six outputs does to the state, as a function of momentum.

A learned CKF arm has been seen reconstructing tracks at 287 TeV where stock's
hardest is 64 GeV, which can only mean the fitted curvature is
being driven towards zero. This asks the model directly, on the table it was
trained on, so there is no run, no retraining, and no extrapolation argument.

Every output is applied by `LearnedTransport.hpp` as a fixed scale times a raw
network output:

    loc0/loc1 correction  raw[0..1] * sigma(class, pT bin, |eta| bin)   [um]
    phi_c   = phi_helix   + raw[2] * v1_scale[0]
    theta_c = theta_helix + raw[3] * v1_scale[1]
    qop_c   = qop         + raw[4] * v1_scale[2]
    p_c     = |1 / qop_c|

The position scale is a table and varies with the state. `v1_scale` is not: it
is one constant per output, the standard deviation of the helix's own residual
over the whole training fold (`train_gtheta.py`, the `--v1` block). The training
sample is log-uniform in pT, so those three constants are set by the soft end of
it, and they are then added to every track at every step.

What the tables report per |p| bin is the median size of what the helix gets
wrong, the median size of what the model adds, what is left, and the fraction of
jumps where what is left is bigger than what the helix got wrong. That last
column is the whole question: a correction is only a correction if it shrinks
the thing it is aimed at.

The rows are the whole teacher table, training fold included. That is the
generous reading: these are jumps the model was fitted on.

    python -m prop.output_response
    python -m prop.output_response --model gtheta_nomat14.npz \
        --teacher teacher_nomat.parquet
"""
import argparse

import numpy as np
import polars as pl

from .chi2_gate import in_plane_residual, measured_cov, resolutions
from .closed_loop import Model
from .helix_variants import FieldMap
from .train_gtheta import features

# |p| in GeV. The top bin is where §6.13's tracks live and where the teacher is
# thinnest, which is the point.
P_EDGES = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, np.inf]


def p_label(lo, hi):
    return f"over {lo:g}" if np.isinf(hi) else f"{lo:g} to {hi:g}"


def table(header, rows):
    align = ["---"] + ["---:"] * (len(header) - 1)
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def v1_targets(t):
    """The three things outputs 2 to 4 are supervised on, in rad and 1/GeV.

    `train_gtheta.v1_targets`, recomputed here rather than imported so this
    module keeps its own record of what the numbers mean. The q/p one is the
    helix's whole error in q/p, which is the energy loss and nothing else,
    because a magnetic field does no work.
    """
    px1, py1, pz1 = (t[c].to_numpy() for c in ("px1", "py1", "pz1"))
    hpx, hpy, hpz = (t[c].to_numpy() for c in ("helix_px", "helix_py",
                                               "helix_pz"))
    qop = t["qop"].to_numpy()
    dphi = (np.arctan2(py1, px1) - np.arctan2(hpy, hpx) + np.pi) \
        % (2 * np.pi) - np.pi
    theta1 = np.arctan2(np.hypot(px1, py1), pz1)
    theta_h = np.arctan2(np.hypot(hpx, hpy), hpz)
    p_h = np.sqrt(hpx ** 2 + hpy ** 2 + hpz ** 2)
    p_1 = np.sqrt(px1 ** 2 + py1 ** 2 + pz1 ** 2)
    return np.column_stack([dphi, theta1 - theta_h, qop * p_h / p_1 - qop])


def position_sigma(t, cov_path, digi_path):
    """The per-jump sigma output 0 and 1 are written in, and the 1D mask.

    `train_gtheta.main`'s own three lines. A 1D module never measures loc1, so
    its second coordinate is a structural zero and is excluded rather than
    counted as a perfect correction.
    """
    res = resolutions(digi_path)
    vol = t["surf1"].to_numpy() >> 56
    v0 = np.array([res[v][0] for v in vol])
    v1r = [res[v][1] for v in vol]
    one_d = np.array([q is None for q in v1r])
    v1 = np.array([1.0 if q is None else q for q in v1r])
    c0, c1 = measured_cov(t, vol, cov_path)
    return np.column_stack([np.hypot(v0, c0), np.hypot(v1, c1)]), one_d


def report(name, unit, p, tgt, applied, keep=None):
    left = tgt - applied
    rows = []
    for lo, hi in zip(P_EDGES[:-1], P_EDGES[1:]):
        s = (p >= lo) & (p < hi)
        if keep is not None:
            s &= keep
        if not s.any():
            continue
        rows.append([
            p_label(lo, hi),
            f"{s.sum():,}",
            f"{np.median(np.abs(tgt[s])):.4g}",
            f"{np.median(np.abs(applied[s])):.4g}",
            f"{np.median(np.abs(left[s])):.4g}",
            f"{(np.abs(left[s]) > np.abs(tgt[s])).mean():.1%}",
        ])
    print(f"## {name}")
    print()
    print(table(["|p| [GeV]", "jumps", f"helix error [{unit}]",
                 f"applied [{unit}]", f"left [{unit}]", "made worse"], rows))
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gtheta_plane14.npz",
                    help="the weights whose outputs are read; the default is "
                         "what cpp/gtheta_weights.hpp holds")
    ap.add_argument("--teacher", default="teacher_phys.parquet",
                    help="the table the model was trained on")
    ap.add_argument("--field", default=None,
                    help="override the field map path the model records")
    ap.add_argument("--cov", default=None, help="override the model's --cov")
    ap.add_argument("--digi", default=None, help="override the model's --digi")
    args = ap.parse_args()

    m = Model(args.model)
    fm = FieldMap(args.field or m.field)
    t = pl.read_parquet(args.teacher)

    X = features(t, fm)
    if X.shape[1] != m.nin:
        raise SystemExit(f"model wants {m.nin} inputs, the table builds "
                         f"{X.shape[1]}")
    raw = m.forward((X - m.mu) / m.sd)

    qop = t["qop"].to_numpy()
    p = 1.0 / np.abs(qop)

    print(f"# {args.model}")
    print()
    print(f"    rows       {len(t):,}")
    print(f"    v1_scale   phi {m.v1_scale[0]:.6g}  theta "
          f"{m.v1_scale[1]:.6g}  q/p {m.v1_scale[2]:.6g}")
    print(f"    |p| GeV    median {np.median(p):.3f}  "
          f"p99 {np.quantile(p, 0.99):.1f}  max {p.max():.1f}")
    print(f"    over 20    {(p > 20).sum():,} rows, {(p > 20).mean():.2%}")
    print()

    # ---- outputs 0 and 1, position on the surface --------------------------
    sig, one_d = position_sigma(t, args.cov or m.cov, args.digi or m.digi)
    b0, b1 = in_plane_residual(t)
    pb = raw[:, :2] * sig
    report("Output 0, loc0", "um", p, b0, pb[:, 0])
    report("Output 1, loc1", "um", p, b1, pb[:, 1], keep=~one_d)

    # ---- outputs 2 to 4, direction and q/p ---------------------------------
    tgt = v1_targets(t)
    applied = raw[:, 2:5] * m.v1_scale
    report("Output 2, phi", "mrad", p, tgt[:, 0] * 1e3, applied[:, 0] * 1e3)
    report("Output 3, theta", "mrad", p, tgt[:, 1] * 1e3, applied[:, 1] * 1e3)
    report("Output 4, q/p", "1/GeV", p, tgt[:, 2], applied[:, 2])

    print("Medians of the absolute value. `made worse` is the fraction of jumps")
    print("whose remaining error is larger than the helix's own, which is what")
    print("switching the correction off would leave. Output 1 counts 2D modules")
    print("only. Output 5 is the sigma head and reaches the covariance, not the")
    print("state, so it is not here.")
    print()

    # ---- and what output 4 does to the momentum ----------------------------
    qop_c = qop + applied[:, 2]
    qop_c = np.where(np.abs(qop_c) < 1e-9, np.sign(qop_c) * 1e-9, qop_c)
    p_c = 1.0 / np.abs(qop_c)
    rows = []
    for lo, hi in zip(P_EDGES[:-1], P_EDGES[1:]):
        s = (p >= lo) & (p < hi)
        if not s.any():
            continue
        r = p_c[s] / p[s]
        rows.append([
            p_label(lo, hi),
            f"{s.sum():,}",
            f"{np.median(r):.3f}",
            f"{np.quantile(r, 0.99):.2f}",
            f"{np.max(r):.4g}",
            f"{(np.sign(qop_c[s]) != np.sign(qop[s])).mean():.2%}",
            f"{(p[s] > 100).mean():.2%}",
            f"{(p_c[s] > 100).mean():.2%}",
        ])
    print("## One application, on the momentum")
    print()
    print(table(["|p| [GeV]", "jumps", "median p_c/p", "p99", "max",
                 "charge flips", "was over 100 GeV", "now over 100 GeV"], rows))
    print()
    print("`p_c` is the momentum after one application of output 4, which is what")
    print("`LearnedTransport.hpp` computes. The CKF applies it once per")
    print("propagation step, so a track of twelve measurements sees it twelve")
    print("times and this is the first of them. The last two columns are the same")
    print("population before and after, so the top bin's `was` column is the")
    print("teacher's own tracks above 100 GeV and not an effect.")
    print()


if __name__ == "__main__":
    main()
