"""A per-hit sigma, on the output the kernel already emits and never used.

`train_gtheta`'s network is 12-64-64-**6**. Two outputs are the position
correction, three are direction and q/p, and the sixth has been carried since
the first speed benchmark purely so the arithmetic matched the C++ kernel. This
file trains it.

Why this and not a better table. The chained measurement showed that
99.6% of a chained filter's holes come from the residual's TAIL rather than its
width, and that the tail is predictable from run-time variables before the hit
is seen (|dz| of the step splits the within-cell width by 3.1x). A table cannot
use that -- it has already spent its resolution on (class, pT, |eta|). A per-hit
sigma can: widen the gate on the jumps that need it and leave the other 90%
alone, which is the trade A3 says you otherwise pay for globally in fakes.

**The trunk is frozen.** Only the last layer's sixth column is fitted, so the
position correction cannot move and every number already recorded stays valid.
It also sharpens the claim: if a LINEAR readout of the same 64 hidden units
predicts the residual's own scale, then the features g_theta already computes
carry the uncertainty too, and the sigma head is free in the strongest sense.

The loss is the Gaussian negative log-likelihood of the measured residual under
the predicted scale,

    L = 0.5 (r / (m sigma_cell))^2 + log(m sigma_cell),     m = exp(a)
    dL/da = 1 - (r / (m sigma_cell))^2

which has the property that matters here: it is minimised by m sigma_cell equal
to the CONDITIONAL width, so a jump the model can tell is dangerous gets a wide
sigma without every other jump paying for it.

Usage:
    python sigma_head.py
    python sigma_head.py --dump runs/nomat_v2.npz --pairs teacher_map_nomat_logpt.parquet
"""
import argparse

import numpy as np
import polars as pl

from .chi2_gate import CHI2_CUT, CLASS_OF, ETA_EDGES, PT_EDGES, resolutions
from .chain_sim import cells, folds
from .closed_loop import Model
from .helix_variants import field_source
from .qtable import robust_sigma
from .train_gtheta import features as features_ref


def cell_sigma(tab, cls, ip, ie):
    """The table's sigma for each jump -- what a lookup gets you."""
    look = {(r["cls"], r["pt_bin"], r["eta_bin"]): r["sig_c0_um"]
            for r in tab.iter_rows(named=True)}
    out = np.empty(len(cls))
    for i in range(len(cls)):
        v = look.get((cls[i], int(ip[i]), int(ie[i])))
        out[i] = v if v is not None else look[(cls[i], -1, -1)]
    return out


def hidden(model, Xn):
    h1, _ = model.act(Xn @ model.W[0] + model.b[0])
    h2, _ = model.act(h1 @ model.W[1] + model.b[1])
    return h2


def fit_head(h, r, sig, tr, va, epochs=400, lr=3e-3, batch=1024,
             seed=20260806):
    """Adam on the last layer's sixth column only."""
    rng = np.random.default_rng(seed)
    w = np.zeros(h.shape[1])
    b = 0.0
    m = [np.zeros_like(w), 0.0]
    v = [np.zeros_like(w), 0.0]
    step = 0
    best, best_p, patience = np.inf, None, 0

    def nll(idx, w, b):
        a = h[idx] @ w + b
        z = r[idx] / (np.exp(a) * sig[idx])
        return float(np.mean(0.5 * z ** 2 + a))

    for ep in range(epochs):
        perm = rng.permutation(tr)
        for k in range(0, len(perm), batch):
            idx = perm[k:k + batch]
            a = h[idx] @ w + b
            z2 = (r[idx] / (np.exp(a) * sig[idx])) ** 2
            g = (1.0 - z2) / len(idx)                 # dL/da
            gw = h[idx].T @ g
            gb = g.sum()
            step += 1
            for i, (par, grad) in enumerate(((w, gw), (b, gb))):
                m[i] = 0.9 * m[i] + 0.1 * grad
                v[i] = 0.999 * v[i] + 0.001 * grad * grad
                upd = lr * (m[i] / (1 - 0.9 ** step)) / (
                    np.sqrt(v[i] / (1 - 0.999 ** step)) + 1e-8)
                if i == 0:
                    w = w - upd
                else:
                    b = b - upd
        vl = nll(va, w, b)
        if vl < best - 1e-6:
            best, best_p, patience = vl, (w.copy(), b), 0
        else:
            patience += 1
        if patience >= 20:
            break
    return best_p


def reject_curve(r, sig, v0, grid):
    """Fraction of true hits the gate throws away, as a function of how wide
    the gate is on average. The x axis is the mean half-width, so a table and
    a per-hit sigma are compared at EQUAL cost rather than at equal k."""
    out = []
    for k in grid:
        s = np.hypot(v0, k * sig)
        lam = (r / s) ** 2
        out.append((np.mean(np.sqrt(CHI2_CUT) * s),
                    float((lam > CHI2_CUT).mean())))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="teacher_map_mat_logpt.parquet")
    ap.add_argument("--model", default="gtheta_mat.npz")
    ap.add_argument("--qtable", default="Q_table.parquet")
    ap.add_argument("--field", default="/tmp/oddb.npz",
                    help="a map npz, or `const` for a uniform 2 T. Must match "
                         "what the model was fitted with")
    ap.add_argument("--out", default="sigma_head.npz")
    args = ap.parse_args()

    model = Model(args.model)
    fm = field_source(args.field)
    t = pl.read_parquet(args.pairs)
    fold = folds(t["track"].to_numpy())
    tr = np.flatnonzero(fold == 0)
    va = np.flatnonzero(fold == 1)
    te = np.flatnonzero(fold == 2)

    X = features_ref(t, fm)
    Xn = (X - model.mu) / model.sd
    h = hidden(model, Xn)
    pred = model.forward(Xn)

    # the residual the head has to describe: what is left after the position
    # correction, in the same micrometres everything else is quoted in
    from .chi2_gate import in_plane_residual, measured_cov
    b0, b1 = in_plane_residual(t)
    vol = t["surf1"].to_numpy() >> 56
    res = resolutions(model.digi)
    v0 = np.array([res[int(v)][0] for v in vol])
    c0, c1 = measured_cov(t, vol, model.cov)
    s_train = np.hypot(v0, c0)
    rem0 = b0 - pred[:, 0] * s_train

    cls = np.array([CLASS_OF[int(v)] for v in vol])
    ip, ie = cells(t["pt"].to_numpy(), t["abs_eta"].to_numpy())
    sig_cell = cell_sigma(pl.read_parquet(args.qtable), cls, ip, ie)

    print(f"\n{len(t):,} jumps -> {len(tr):,} train / {len(va):,} val / "
          f"{len(te):,} test (split by track, same as training)")
    w, bb = fit_head(h, rem0, sig_cell, tr, va)
    a = h @ w + bb
    mscale = np.exp(a)
    sig_hit = mscale * sig_cell

    print(f"\n=== the head's output ===")
    print(f"  per-hit scale m: median {np.median(mscale[te]):.2f}   "
          f"p10 {np.percentile(mscale[te], 10):.2f}   "
          f"p90 {np.percentile(mscale[te], 90):.2f}   "
          f"max {mscale[te].max():.1f}")
    print(f"  so the gate it asks for spans "
          f"{np.percentile(sig_hit[te], 10):.0f} to "
          f"{np.percentile(sig_hit[te], 90):.0f} um where the table says "
          f"{np.percentile(sig_cell[te], 10):.0f} to "
          f"{np.percentile(sig_cell[te], 90):.0f}")

    print(f"\n=== is it calibrated? pull = residual / sigma, test fold ===")
    for lab, s in (("table sigma", sig_cell), ("per-hit sigma", sig_hit)):
        p = rem0[te] / s[te]
        print(f"  {lab:16}  MAD width {robust_sigma(p):5.2f}   "
              f"RMS {np.std(p):6.2f}   |pull|>3.9: "
              f"{100 * (np.abs(p) > np.sqrt(CHI2_CUT)).mean():5.2f}%")
    print("  RMS near 1 is the one to read: it is the tail, and it is what the "
          "gate charges for.")

    grid = np.geomspace(0.02, 30, 120)
    ct = reject_curve(rem0[te], sig_cell[te], v0[te], grid)
    ch = reject_curve(rem0[te], sig_hit[te], v0[te], grid)

    print(f"\n=== the comparison that decides it: hits lost at EQUAL average "
          f"gate width ===")
    print(f"  {'target':>8}{'table: width / lost':>24}"
          f"{'per-hit: width / lost':>26}{'ratio':>8}")
    lo = max(ct[:, 0].min(), ch[:, 0].min())
    for target in (100, 150, 200, 400, 800):
        if target < lo:
            continue
        # interpolate each curve at the SAME width rather than snapping to
        # whichever grid point happens to be near it -- the point of the
        # comparison is that the two costs are equal
        a_ = np.interp(target, ct[:, 0], ct[:, 1])
        b_ = np.interp(target, ch[:, 0], ch[:, 1])
        print(f"  {target:6d}um{target:14d}um{100*a_:8.2f}%"
              f"{target:16d}um{100*b_:8.2f}%"
              f"{(b_ / a_ if a_ else np.nan):8.2f}")
    print(f"  (curves span {lo:.0f} to "
          f"{min(ct[:, 0].max(), ch[:, 0].max()):.0f} um of mean half-width)")
    print("  Same average gate, so the same fake rate. Ratio below 1 is hits "
          "the table\n  throws away and the per-hit sigma keeps.")

    np.savez_compressed(args.out, w=w, b=np.array([bb]),
                        curve_table=ct, curve_hit=ch,
                        m_test=mscale[te], sig_cell_test=sig_cell[te],
                        sig_hit_test=sig_hit[te], rem0_test=rem0[te],
                        v0_test=v0[te])
    print(f"\nwrote {args.out}")
    print("  the head is 64 weights and one bias on the sixth output the "
          "kernel already\n  computes, so inference cost is unchanged: the "
          "456 ns / 6.0x claim stands.")


if __name__ == "__main__":
    main()
