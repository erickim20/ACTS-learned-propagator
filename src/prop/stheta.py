"""Train and save s_theta, the next-layer classifier, so it can run in a chain.

`train_gtheta.py --stheta` already trains this and scores it, but only one step
from a true state and only inside that script -- nothing is saved, so no chain
has ever had to choose its own destination surface. Section 13 of
That is the largest remaining untested
assumption, and this file is the half of it that has to exist first.

What it may use is fixed by the same rule the plan set: the state, the field,
and a one-hot of the layer it is standing on -- which is what the navigator
knows. Not surface identity of the destination, and nothing computed from the
destination hit.

The class list and the source-layer list are saved with the weights. A
destination never seen in training is unpredictable by construction and counts
as a miss, and that has to stay true at inference, so the lists are part of the
model rather than something rebuilt from whatever data is at hand.

Usage:
    python stheta.py --pairs teacher_map_mat_logpt.parquet --out stheta.npz
"""
import argparse

import numpy as np
import polars as pl

from .chain_sim import folds
from .helix_variants import FieldMap
from .train_gtheta import ACTS, MLP, layer_id, softmax_ce, state_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="teacher_map_mat_logpt.parquet")
    ap.add_argument("--field", default="/tmp/oddb.npz")
    ap.add_argument("--act", default="relu", choices=list(ACTS))
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--out", default="stheta.npz")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    t = pl.read_parquet(args.pairs)
    fold = folds(t["track"].to_numpy(), seed=args.seed)
    tr_i, va_i, te_i = (np.flatnonzero(fold == k) for k in (0, 1, 2))

    src = layer_id(t["surf0"].to_numpy())
    dst = layer_id(t["surf1"].to_numpy())
    src_vals = np.unique(src)
    src_idx = np.searchsorted(src_vals, src)
    cls = np.unique(dst[tr_i])
    pos = np.clip(np.searchsorted(cls, dst), 0, len(cls) - 1)
    seen = cls[pos] == dst
    y = np.where(seen, pos, 0)

    X = state_features(t, FieldMap(args.field), len(src_vals), src_idx)
    mu, sd = X[tr_i].mean(0), X[tr_i].std(0)
    sd[sd == 0] = 1.0
    Xn = (X - mu) / sd

    print(f"{len(t):,} jumps, {len(cls)} destination layers from "
          f"{len(src_vals)} source layers")
    net = MLP(Xn.shape[1], args.act, rng, args.hidden, len(cls))
    m = [np.zeros_like(p) for p in net.params()]
    v = [np.zeros_like(p) for p in net.params()]
    step, best, best_w, patience = 0, np.inf, None, 0
    for ep in range(args.epochs):
        perm = rng.permutation(tr_i)
        for k in range(0, len(perm), args.batch):
            idx = perm[k:k + args.batch]
            p, cache = net.forward(Xn[idx])
            _, d = softmax_ce(p, y[idx])
            gW, gb = net.backward(cache, d)
            step += 1
            for i, (par, g) in enumerate(zip(net.params(), gW + gb)):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                par -= args.lr * (m[i] / (1 - 0.9 ** step)) / (
                    np.sqrt(v[i] / (1 - 0.999 ** step)) + 1e-8)
        vl = softmax_ce(net.forward(Xn[va_i])[0], y[va_i])[0]
        if vl < best - 1e-6:
            best, best_w, patience = vl, [p.copy() for p in net.params()], 0
        else:
            patience += 1
        if patience >= 15:
            break
    for p, w in zip(net.params(), best_w):
        p[...] = w

    logits = net.forward(Xn[te_i])[0]
    rank = np.argsort(-logits, axis=1)
    print(f"\n  test fold, one step from a TRUE state:")
    for k in (1, 2, 3, 5):
        acc = ((rank[:, :k] == y[te_i][:, None]).any(1) & seen[te_i]).mean()
        print(f"    top-{k} layer accuracy {100 * acc:6.2f}%")

    p = {f"W{i}": w for i, w in enumerate(net.W)}
    p.update({f"b{i}": b for i, b in enumerate(net.b)})
    p.update(mu=mu, sd=sd, cls=cls, src_vals=src_vals,
             act=np.array([args.act]))
    np.savez(args.out, **p)
    print(f"\nsaved -> {args.out}")
    print("  A chain that uses this loses a hit whenever the named layer is "
          "wrong, whatever\n  the gate would have done -- which is the part "
          "no number so far has included.")


if __name__ == "__main__":
    main()
