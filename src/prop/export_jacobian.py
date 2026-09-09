"""Reference vectors for the C++ transport Jacobian.

`jacobian_free.py` is the verified article: its output is checked against
`F_bound`, which is itself checked against finite differences through the actual
bisection solver. `cpp/LearnedJacobian.hpp` is a port of it, and a port that is
"obviously the same formulas" is exactly the kind of thing that is silently 1%
wrong. This writes both ends so `cpp/test_jacobian.cpp` can compare.

Note which endpoint the free change of variables is evaluated at. Here it is the
HELIX endpoint, because that is what the correspondence with `F_bound` was
verified at and `F_bound` knows nothing about the network. The stepper evaluates
it at the corrected endpoint instead, which differs by the size of the correction
and is the point the covariance actually describes. `jumpJacobian` takes it as an
argument for that reason, and this file passes the helix one so the check tests
the port and not the choice.

    python -m prop.export_jacobian --pairs teacher_phys.parquet
"""
import argparse
import pathlib
import struct

import numpy as np
import polars as pl

from .jacobian_free import (PION_MASS, D_free, d_free_d_path, time_of_flight)
from .jacobian_helix import helix_state, solve_s

OUT = pathlib.Path("cpp")

# input  : x y z px py pz q bz s  hpx hpy hpz
# output : D(64, row-major) t(8) path3d dt
#
# The helix endpoint momentum is an INPUT, carried here rather than re-solved on
# the C++ side. `jumpJacobian` takes it as an argument precisely because two
# endpoints are defensible, so a test that re-derived it would be testing the
# arc-length solver and the choice of endpoint at the same time as the Jacobian,
# and a disagreement would not say which.
N_IN, N_OUT = 12, 74


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="teacher_phys.parquet")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--bz", type=float, default=2.0)
    ap.add_argument("--mass", type=float, default=PION_MASS)
    ap.add_argument("--seed", type=int, default=20260812)
    a = ap.parse_args()

    t = pl.read_parquet(a.pairs)
    rng = np.random.default_rng(a.seed)
    idx = np.sort(rng.choice(len(t), size=min(a.n, len(t)), replace=False))
    t = t[idx]

    u = np.column_stack([t[c].to_numpy() for c in
                         ("x", "y", "z", "px", "py", "pz")])
    r1, z1 = t["r1"].to_numpy(), t["z1"].to_numpy()
    endcap = t["endcap"].to_numpy().astype(bool)
    q = np.sign(t["qop"].to_numpy())

    s, ok = solve_s(u, r1, z1, endcap, q, a.bz)
    # Jumps the helix never reaches carry no Jacobian, and keeping them would
    # only measure how two languages propagate a nan.
    keep = ok & np.isfinite(s) & (s > 0)
    u, s, q, endcap = u[keep], s[keep], q[keep], endcap[keep]

    out = helix_state(u, s, q, a.bz)
    D = D_free(u, s, q, a.bz, endcap, out, mass=a.mass)
    tv = d_free_d_path(u, s, q, a.bz, endcap, out, mass=a.mass)

    p = np.linalg.norm(u[:, 3:], axis=1)
    pt = np.hypot(u[:, 3], u[:, 4])
    path3d = s * p / pt

    inp = np.column_stack([u, q, np.full(len(u), a.bz), s, out[:, 3:]])
    exp = np.column_stack([D.reshape(len(u), 64), tv, path3d,
                           time_of_flight(u, s, mass=a.mass)])
    assert inp.shape[1] == N_IN and exp.shape[1] == N_OUT

    OUT.mkdir(exist_ok=True)
    with open(OUT / "reference_jac.bin", "wb") as f:
        f.write(struct.pack("<iii", len(inp), N_IN, N_OUT))
        f.write(struct.pack("<d", a.mass))
        f.write(inp.astype("<f8").tobytes())
        f.write(exp.astype("<f8").tobytes())
    print(f"cpp/reference_jac.bin    {len(inp)} jumps "
          f"({int(endcap.sum())} endcap, {int((~endcap).sum())} barrel), "
          f"mass {a.mass} GeV")
    print(f"  dropped {int((~keep).sum())} with no helix solution")


if __name__ == "__main__":
    main()
