"""Build the field-map npz that g_theta's 11th input needs, on this machine.

`helix_variants.FieldMap` and `train_gtheta.features` both read `oddb.npz`.
It
does not have to: the ODD image `ghcr.io/opendatadetector/sw` ships
`/opt/odd/data/odd-bfield.csv`, the image is on this Mac, and the csv is the
same grid the npz was built from. So the map is a `docker cp` and a reshape
away, and nothing about the learned transport is Windows-only.

The grid is checked rather than assumed: 201 x 201 x 301 on 100 mm spacing,
x in [-10, 10] m, y the same, z in [-15, 15] m, ordered x outer / y middle /
z inner -- which is exactly what `FieldMap.__init__` reshapes for.

Usage:
    ODD=ghcr.io/opendatadetector/sw@sha256:20e4df7b0d6befd70ce0113d852350b25cbd9aac659d1bc530036a2c85a68f89
    docker create --name oddx $ODD          # by digest, never by tag
    docker cp oddx:/opt/odd/data/odd-bfield.csv odd-bfield.csv
    docker rm oddx
    python build_fieldmap.py odd-bfield.csv --out oddb.npz
"""
import argparse

import numpy as np
import polars as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="/tmp/oddb.npz")
    args = ap.parse_args()

    d = pl.read_csv(args.csv)
    need = ["x", "y", "z", "Bx", "By", "Bz"]
    if d.columns != need:
        raise SystemExit(f"unexpected columns {d.columns}, wanted {need}")

    x, y, z = (d[c].to_numpy() for c in "xyz")
    ux, uy, uz = np.unique(x), np.unique(y), np.unique(z)
    nx, ny, nz = len(ux), len(uy), len(uz)
    if nx * ny * nz != d.height:
        raise SystemExit(f"{d.height:,} rows is not a full {nx}x{ny}x{nz} grid")
    # the reshape in FieldMap assumes this ordering; check it rather than
    # trusting the docstring, because a silently transposed field map would
    # look plausible everywhere and be wrong everywhere
    if not (np.array_equal(z[:nz], uz)
            and np.array_equal(y[:nz], np.full(nz, uy[0]))):
        raise SystemExit("csv is not ordered x outer / y middle / z inner")
    for u, nm in ((ux, "x"), (uy, "y"), (uz, "z")):
        s = np.diff(u)
        if not np.allclose(s, s[0]):
            raise SystemExit(f"{nm} grid is not regular")

    B = np.column_stack([d[c].to_numpy() for c in ("Bx", "By", "Bz")])
    np.savez_compressed(args.out, x=ux, y=uy, z=uz, B=B.astype(np.float32))
    print(f"wrote {args.out}: {nx} x {ny} x {nz}, spacing "
          f"{ux[1]-ux[0]:.0f} / {uy[1]-uy[0]:.0f} / {uz[1]-uz[0]:.0f} mm")

    # sanity: the solenoid is ~2 T on axis at the centre and falls in the
    # forward direction, which is the whole reason the helix needs correcting
    # relative, because this runs as `python -m prop.build_fieldmap`
    # from the repository root and a bare import only resolves when the cwd is
    # src/prop
    from .helix_variants import FieldMap
    fm = FieldMap(args.out)
    for zz in (0.0, 1000.0, 2000.0, 3000.0):
        bz = fm.bz(np.array([0.0]), np.array([0.0]), np.array([zz]))[0]
        print(f"  Bz(0, 0, {zz:5.0f} mm) = {bz:.3f} T")


if __name__ == "__main__":
    main()
