"""Write the field map in the flat binary form `cpp/bench_kernel.cpp` reads.

Round 9's handoff said `export_kernel.py` regenerates `field.bin`. It does not
-- it only writes `gtheta_weights.hpp` and `reference.bin`, and no writer for
`field.bin` existed anywhere in the tree. Without it the benchmark falls back
to a constant 2 T, which is the one substitution that invalidates the whole
comparison: the reason a learned jump beats an integrator is that it reads the
map ONCE while RKN4 reads it four times per step, and in a constant field that
difference disappears and RKN looks free. So the missing file was not cosmetic.

The layout is dictated by `FieldMap::load` in bench_kernel.cpp:

    int32   n[3]                     nx, ny, nz
    double  o[3]                     grid origin, mm
    double  h[3]                     spacing, mm
    float   b[nx*ny*nz*ncomp]        x outer, y middle, z inner

`--components z` is the benchmark's file and the default: only Bz is stored,
because the kernel's eleventh feature and the benchmark's RKN both use Bz alone
and carrying Bx/By would triple a 48 MB file to no purpose. Byte for byte what
this script has always written.

`--components xyz` is the detector's file. `cpp/ODDFieldMapXyz.cpp` puts the
map into the DD4hep compact so that simulation and reconstruction read one
field, and a detector needs all three components: the analytic solenoid the ODD
ships has no transverse field anywhere, and the map does.

The two differ only in the payload, so nothing distinguishes them but their
size. Both readers check it: a one-component file is exactly a third of a
three-component payload and would otherwise be read as a third of a map.

    python build_fieldmap.py odd-bfield.csv --out oddb.npz   # first
    python export_fieldbin.py --npz oddb.npz --out cpp/field.bin
    python export_fieldbin.py --npz oddb.npz --components xyz \
        --out odd-bfield-xyz.bin
"""
import argparse
import pathlib
import struct

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/tmp/oddb.npz",
                    help="the map from build_fieldmap.py")
    ap.add_argument("--out", default="cpp/field.bin")
    ap.add_argument("--components", default="z", choices=["z", "xyz"],
                    help="z: Bz alone, the benchmark's file. xyz: all three "
                         "interleaved, the file the DD4hep field plugin reads")
    a = ap.parse_args()

    d = np.load(a.npz)
    x, y, z = d["x"], d["y"], d["z"]
    nx, ny, nz = len(x), len(y), len(z)

    # build_fieldmap.py already checked regularity and ordering; re-check the
    # one property this file depends on, because a transposed grid would be
    # wrong everywhere and look plausible everywhere
    B = d["B"]
    if B.shape != (nx * ny * nz, 3):
        raise SystemExit(f"B is {B.shape}, expected {(nx * ny * nz, 3)}")
    for u, nm in ((x, "x"), (y, "y"), (z, "z")):
        s = np.diff(u)
        if not np.allclose(s, s[0]):
            raise SystemExit(f"{nm} grid is not regular")

    ncomp = 3 if a.components == "xyz" else 1
    payload = np.ascontiguousarray(B if ncomp == 3 else B[:, 2], dtype="<f4")

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(struct.pack("<iii", nx, ny, nz))
        f.write(struct.pack("<ddd", float(x[0]), float(y[0]), float(z[0])))
        f.write(struct.pack("<ddd", float(x[1] - x[0]), float(y[1] - y[0]),
                            float(z[1] - z[0])))
        f.write(payload.tobytes())

    print(f"{out}: {nx} x {ny} x {nz} x {ncomp}, "
          f"origin ({x[0]:.0f}, {y[0]:.0f}, {z[0]:.0f}) mm, "
          f"spacing {x[1]-x[0]:.0f} mm, {out.stat().st_size / 1e6:.1f} MB")

    # the size both readers check, computed here rather than trusted
    want = 12 + 24 + 24 + 4 * nx * ny * nz * ncomp
    if out.stat().st_size != want:
        raise SystemExit(f"wrote {out.stat().st_size} bytes, expected {want}")

    # read it back the way the C++ indexes it, and check the axis value the
    # note quotes: ~2 T at the origin
    i = ((nx // 2) * ny + (ny // 2)) * nz + (nz // 2)
    if ncomp == 1:
        print(f"  Bz at grid centre = {payload[i]:.3f} T")
    else:
        b = payload[i]
        print(f"  B at grid centre = ({b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}) T")


if __name__ == "__main__":
    main()
