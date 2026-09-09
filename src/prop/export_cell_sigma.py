"""Turn the training-time per-jump sigma into the binary the stepper loads.

The network's position outputs are trained in units of sigma. `train_gtheta.py`
divides each jump's target by that jump's own sigma:

    s0 = hypot(res[volume][0], sig_c0(class, pt bin, |eta| bin))
    s1 = hypot(res[volume][1], sig_c1(class, pt bin, |eta| bin))

so output 0 means "this many sigma of THIS module, at THIS pT and |eta|". The
runtime multiplied by two constants instead, `cellSig0 = 20` and
`cellSig1 = 43` micrometres, set once per run at `learned_ckf.cpp:104` and
applied to every jump. The two are not the same quantity and the ODD's own
digitisation says by how much:

    volume 16-18  pixel         15.0 / 15.0 um
    volume 23-25  short strip   43.0 / 1200.0 um
    volume 28-30  long strip    72.0 / 1D

Against 20 / 43 that is 2.9x too large on a pixel's loc1 and 28x too small on a
short strip's, so the correction the network asked for is not the correction the
filter received. Model B captures 79.5 % of the helix's position error on its
own test fold while those outputs contribute nothing inside the CKF; this is
the difference between the two
measurements, and closing it is what tests that reading.

**The resolution is constant within a class**, which is why one table per
(class, pt bin, |eta| bin) is exact rather than an approximation: all three
pixel volumes are 15/15, all three short-strip volumes are 43/1200 and all three
long-strip volumes are 72/1D. The binning is `chi2_gate.PT_EDGES` and
`ETA_EDGES`, which is what `measured_cov` keys on and what `LearnedNoise.hpp`
already implements in C++.

**Long strips get sig1 = 0.** They measure one coordinate, so `train_gtheta.py`
zeroes both the target and the weight of output 1 on them (`:652-653`) and the
model is never supervised there. The runtime applied that unsupervised output
anyway, times 43 um. A zero in the table switches the term off without a kernel
change. It is a behaviour change beyond the scaling and it is called out here
because the arm that tests the table tests both at once.

Usage:
    python -m prop.export_cell_sigma --cov sigma_C.parquet \
        --digi config/odd-digi-smearing-config.json
"""
import argparse
import json
import pathlib
import struct

import numpy as np
import polars as pl

from .chi2_gate import ETA_EDGES, PT_EDGES, resolutions

# `qtable.CLASSES` order, which is what `LearnedNoise.hpp`'s `ModuleClass`
# enumerates and what the Q table is written in. The two files are read by the
# same C++ indexing, so they must not disagree.
CLASSES = ("pixel", "sstrip", "lstrip")

# One representative volume per class. `resolutions()` is keyed on volume and
# every volume of a class carries the same smearing, which `check_uniform`
# below verifies rather than assumes.
VOLUMES = {"pixel": (16, 17, 18),
           "sstrip": (23, 24, 25),
           "lstrip": (28, 29, 30)}

N_PT = len(PT_EDGES) - 1
N_ETA = len(ETA_EDGES) - 1
MAGIC = 0x53474D41                                  # "SGMA"
OUT = pathlib.Path("cpp")


def check_uniform(res):
    """The class-level table is only exact if a class has one resolution."""
    out = {}
    for cls, vols in VOLUMES.items():
        seen = {res[v] for v in vols if v in res}
        if not seen:
            raise SystemExit(f"{cls}: none of {vols} in the digitisation config")
        if len(seen) > 1:
            raise SystemExit(
                f"{cls}: volumes {vols} carry different resolutions {seen}. "
                "A per-class table cannot represent that; key the table on "
                "volume instead.")
        out[cls] = seen.pop()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cov", default="sigma_C.parquet",
                    help="the table measured_cov reads, keyed on "
                         "(cls, pt_bin, eta_bin) with pt_bin = eta_bin = -1 "
                         "carrying the per-class fallback")
    ap.add_argument("--digi", default="config/odd-digi-smearing-config.json")
    ap.add_argument("--out", default=str(OUT / "cell_sigma.bin"))
    a = ap.parse_args()

    res = check_uniform(resolutions(a.digi))
    tab = pl.read_parquet(a.cov)
    cell = {(r["cls"], r["pt_bin"], r["eta_bin"]): (r["sig_c0_um"],
                                                    r["sig_c1_um"])
            for r in tab.iter_rows(named=True)}

    def sigma(cls, ip, ie):
        """hypot of the module resolution and the measured covariance width.

        Exactly `train_gtheta.py:600-607`. A cell the CKF sample never filled
        falls back to the class median, which is the row `measured_cov` stores
        under pt_bin = eta_bin = -1; a class with no row at all is a bug in the
        table rather than a case to guess at.
        """
        c = cell.get((cls, ip, ie)) or cell.get((cls, -1, -1))
        if c is None:
            raise SystemExit(f"{a.cov}: class {cls!r} has no cell and no "
                             "fallback row")
        r0, r1 = res[cls]
        s0 = float(np.hypot(r0, c[0]))
        # 1D module: the model is not supervised on loc1, so the runtime must
        # not apply output 1 there. See the docstring.
        s1 = 0.0 if r1 is None else float(np.hypot(r1, c[1]))
        return s0, s1, c is cell.get((cls, ip, ie))

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    nfall = 0
    with open(out, "wb") as f:
        f.write(struct.pack("<4i", MAGIC, len(CLASSES), N_PT, N_ETA))
        for cls in CLASSES:                          # per-class fallback first
            s0, s1, _ = sigma(cls, -1, -1)
            f.write(struct.pack("<2d", s0, s1))
        for cls in CLASSES:
            for ip in range(N_PT):
                for ie in range(N_ETA):
                    s0, s1, have = sigma(cls, ip, ie)
                    nfall += not have
                    f.write(struct.pack("<B", int(have)))
                    f.write(struct.pack("<2d", s0, s1))

    print(f"{out}: {len(CLASSES)}x{N_PT}x{N_ETA}, {nfall} cells on the "
          f"class fallback")
    print(f"{'class':>8} {'res loc0':>9} {'res loc1':>9}   "
          f"{'sigma at the class fallback':>28}")
    for cls in CLASSES:
        r0, r1 = res[cls]
        s0, s1, _ = sigma(cls, -1, -1)
        r1s = "1D" if r1 is None else f"{r1:.1f}"
        print(f"{cls:>8} {r0:>9.1f} {r1s:>9}   {s0:>13.2f} {s1:>13.2f}")
    print("against the constants the runtime used: 20.00 um and 43.00 um")


if __name__ == "__main__":
    main()
