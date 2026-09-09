"""Write a teacher table's transports in the binary layout bench_kernel reads.

The latency benchmark runs once per pT bin against
transports drawn from that bin. The benchmark already reads a jump file;
`export_kernel.py` is the tool that writes one, but it also runs the model
forward to record expected outputs, so it needs a weights npz, a sigma head, a
covariance table and a digitization config. None of those exist for this
experiment and section 3 says none of them may be inherited.

`bench_kernel`'s loader reads the 22 input columns and stops
(bench_kernel.cpp:load). The expected-output block exists for `test_kernel`,
which is a check of the C++ port against the Python one and is not part of this
measurement. So this writes the input block, declares zero output columns, and
does not go near `cpp/gtheta_weights.hpp`.

The layout is `export_kernel.reference()`'s, column for column:

    x y z px py pz  q  cx cy cz  nx ny nz  e0(3) e1(3)  bz_map  sig0 sig1

`sig0` and `sig1` are the cell widths the network's two position outputs are
written in. They scale the output and not the arithmetic that produces it
(LearnedTransport.hpp:312-313 is two multiplies either way), so latency does not
depend on them and they are set to a fixed pair rather than looked up from a
calibration table this experiment does not have. Nothing timed reads them.

`bz_map` is the map value at the source point. It is read from `cpp/field.bin`
rather than from the npz `helix_variants.FieldMap` wants, because that is the
file the benchmark itself loads: taking both from one file is what stops the
Python and the C++ interpolating different fields. The trilinear weights below
are `FieldMap::bz` in bench_kernel.cpp, transcribed.

    python -m prop.export_jumps --pairs teacher_mu_lat.parquet \
        --out cpp/muon_jumps.bin
"""
import argparse
import pathlib
import struct

import numpy as np
import polars as pl

N_IN = 22


class FieldBin:
    """cpp/field.bin, the map bench_kernel loads.

    Layout from prop.export_fieldbin: int32 n[3], double o[3], double h[3],
    then nx*ny*nz float32 of Bz with x outer, y middle, z inner.
    """

    def __init__(self, path):
        with open(path, "rb") as f:
            self.n = np.frombuffer(f.read(12), dtype="<i4")
            self.o = np.frombuffer(f.read(24), dtype="<f8")
            self.h = np.frombuffer(f.read(24), dtype="<f8")
            total = int(self.n[0]) * int(self.n[1]) * int(self.n[2])
            self.bz_ = np.frombuffer(f.read(total * 4), dtype="<f4")
        if len(self.bz_) != total:
            raise SystemExit(f"{path}: expected {total} samples, "
                             f"got {len(self.bz_)}")
        self.grid = self.bz_.reshape(int(self.n[0]), int(self.n[1]),
                                     int(self.n[2]))

    def bz(self, x, y, z):
        p = np.stack([x, y, z], axis=-1)
        f = (p - self.o) / self.h
        i0 = np.clip(np.floor(f).astype(int), 0, self.n - 2)
        t = np.clip(f - i0, 0.0, 1.0)
        out = np.zeros(len(x))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = ((t[:, 0] if dx else 1 - t[:, 0])
                         * (t[:, 1] if dy else 1 - t[:, 1])
                         * (t[:, 2] if dz else 1 - t[:, 2]))
                    out += w * self.grid[i0[:, 0] + dx, i0[:, 1] + dy,
                                         i0[:, 2] + dz]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="a teacher pairs parquet")
    ap.add_argument("--out", default="cpp/muon_jumps.bin")
    ap.add_argument("--field", default="cpp/field.bin",
                    help="the same map bench_kernel loads")
    ap.add_argument("--sig", type=float, nargs=2, default=(50.0, 200.0),
                    help="sig0 sig1 in um. Scales the network output, not its "
                         "cost; see the module docstring")
    ap.add_argument("--pt-lo", type=float, default=None)
    ap.add_argument("--pt-hi", type=float, default=None)
    a = ap.parse_args()

    t = pl.read_parquet(a.pairs)
    if a.pt_lo is not None:
        t = t.filter(pl.col("pt") >= a.pt_lo)
    if a.pt_hi is not None:
        t = t.filter(pl.col("pt") < a.pt_hi)
    if t.height == 0:
        raise SystemExit("no jumps left after the pT filter")

    u = np.column_stack([t[c].to_numpy() for c in
                         ("x", "y", "z", "px", "py", "pz")]).astype(float)
    q = np.sign(t["qop"].to_numpy()).astype(float)
    cen = np.column_stack([t["mc_x"], t["mc_y"], t["mc_z"]]).astype(float)
    nrm = np.column_stack([t["mn_x"], t["mn_y"], t["mn_z"]]).astype(float)
    e0 = np.column_stack([t["mu0_x"], t["mu0_y"], t["mu0_z"]]).astype(float)
    e1 = np.column_stack([t["mu1_x"], t["mu1_y"], t["mu1_z"]]).astype(float)

    fm = FieldBin(a.field)
    bz = fm.bz(u[:, 0], u[:, 1], u[:, 2])

    sig = np.tile(np.asarray(a.sig, dtype=float), (t.height, 1))

    inp = np.column_stack([u, q, cen, nrm, e0, e1, bz, sig])
    assert inp.shape[1] == N_IN, inp.shape

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        # n, n_in, n_out. Zero output columns: this file is for bench_kernel,
        # which does not read them, and not for test_kernel, which does.
        f.write(struct.pack("<iii", len(inp), N_IN, 0))
        f.write(inp.astype("<f8").tobytes())

    pt = t["pt"].to_numpy()
    print(f"{out}  {len(inp)} jumps, {N_IN} in / 0 out")
    print(f"  source pT  {pt.min():.3f} to {pt.max():.3f} GeV, "
          f"median {np.median(pt):.3f}")
    print(f"  |eta|      median {np.median(np.abs(t['eta'].to_numpy())):.3f}")
    print(f"  bz at source  {bz.min():.4f} to {bz.max():.4f} T")


if __name__ == "__main__":
    main()
