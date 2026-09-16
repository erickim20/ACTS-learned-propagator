"""Turn a measured Q table into the binary `cpp/LearnedNoise.hpp` loads.

`qtable.py` writes widths and biases per (class, pt bin, |eta| bin) and the 5x5
correlation per class. The stepper needs a covariance, which is the outer
product of the widths with that correlation, so the assembly happens here rather
than in C++: it is a statement about what was measured and it belongs next to
the thing that measured it.

    sigma = (sig_c0, sig_c1, sig_phi, sig_theta, sig_qop)
    Q_ij  = R_ij sigma_i sigma_j

Three decisions are recorded in the file rather than left to the reader.

**The material flag.** The C++ REFUSES a table measured with material on. ACTS
adds scattering again at every material surface, so a material-on Q double counts
it, and the two tables are indistinguishable by inspection. The flag is written
from the source file's own name only if `--material-off` is passed, so the
default is to fail rather than to guess.

**Units.** Widths are measured in micrometres and milliradians; ACTS is mm and
rad. A variance carries the square, so the conversion is 1e-6 and not 1e-3, and
getting it wrong gives a Q a thousand times too large and a filter that accepts
everything.

**MAD, not RMS.** `sig_*` is 1.4826 x MAD, which is what a Gaussian filter wants.
The RMS is 3 to 9 times larger because the residual has a heavy tail
in the residual. Feeding the RMS in would widen every gate to cover a tail the
Gaussian update cannot represent anyway.

**The branch.** The runtime takes two transports, the network above the field
gate and the helix alone at or below it, and their errors differ by a factor of
a hundred on the pixel second coordinate. `--helix-table`
writes both measurements into one file and the stepper picks the one the jump
was made by. Without it the file holds one branch and the stepper arms it on
both, which is what every table before this one did.

**The provenance.** A Q table is a measurement of one model's error on one
population, and neither the model nor the population is recoverable from the
numbers. The model is the weights header the residuals were dumped from; the
population is decided by the field gate, because above the threshold the
network ran and at or below it the helix ran alone. Both are written into the
header and `NoiseTable::load` refuses a table whose two do not match the run
about to arm it, the way it already refuses a material-on table. It is required
rather than optional for the same reason `--material-off` is: the default has
to be to fail rather than to guess.

The model is named by a digest over the header's non-comment lines and not by
the md5 of the file. The release filter rewrites the `source:` comment on the
way out, so two copies of one model hash differently as files: a table stamped
against one could never match a build made from the other, for any model.
`cpp/incontainer_build_ckf.sh` computes the same digest in shell and stamps it
into the binary, and the two are checked against each other on every header
that matters.

Usage:
    python -m prop.export_qtable --table Q_nomat.parquet --corr Q_nomat_corr.npz \
        --material-off --weights cpp/gtheta_weights.hpp --field-gate 0.05
    python -m prop.export_qtable --table Q_fired.parquet \
        --helix-table Q_helix.parquet --corr Q_nomat_corr.npz --material-off \
        --weights cpp/gtheta_weights.hpp --field-gate 0.05
"""
import argparse
import hashlib
import pathlib
import struct

import numpy as np
import polars as pl

MAGIC = 0x51544142           # "QTAB", matched in LearnedNoise.hpp
MAGIC_BRANCH = 0x32425451    # "QTB2", the same file with a branch axis
MAGIC_PROV = 0x33425451      # "QTB3", the same file with its provenance
# The weights md5 and field gate a fixture carries. A fixture is not a
# measurement of any model on any population, and 32 zeros is a value no md5
# takes, so a fixture cannot be armed by a run that declares a real one.
FIXTURE_MD5 = "0" * 32
FIXTURE_GATE = -1.0
CLASSES = ("pixel", "sstrip", "lstrip")
N_PT, N_ETA = 5, 5
UM = 1e-3                    # micrometre -> mm
MRAD = 1e-3                  # milliradian -> rad
OUT = pathlib.Path("cpp")

# (loc0, loc1, phi, theta, q/p). The scale that takes each measured width into
# ACTS units; q/p is already absolute and needs none.
TO_ACTS = np.array([UM, UM, MRAD, MRAD, 1.0])
COLS = ("sig_c0_um", "sig_c1_um", "sig_phi_mrad", "sig_theta_mrad",
        "sig_qop_rel")


def weights_digest(path):
    """The md5 of the header's non-comment lines: a line whose first non-blank
    characters are `//` is dropped and every other line is kept byte for byte
    with a newline after it.

    One sentence, implemented twice. `cpp/incontainer_build_ckf.sh` is the
    other half and stamps the same digest into the binary; what keeps them
    honest is running both on the same headers.
    """
    lines = pathlib.Path(path).read_bytes().split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    kept = [ln + b"\n" for ln in lines
            if not ln.lstrip().startswith(b"//")]
    return hashlib.md5(b"".join(kept)).hexdigest()


def sigma_of(row):
    """The five widths of one row, in ACTS units. None means not measured."""
    s = np.zeros(5)
    for k, c in enumerate(COLS):
        v = row.get(c)
        # A 1D module never measures loc1, so its width is a structural
        # absence rather than a small number, and a zero row and column is the
        # honest encoding: no noise is added to a coordinate nothing saw.
        s[k] = 0.0 if v is None or not np.isfinite(v) else float(v)
    return s * TO_ACTS


def cov_of(row, corr):
    """Q = R_ij sigma_i sigma_j, with any unmeasured correlation set to zero.

    Zero rather than dropped: a NaN would propagate into the filter's S and
    turn a missing measurement into a crash three call frames away.
    """
    s = sigma_of(row)
    R = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(R, 1.0)
    return R * s[:, None] * s[None, :]


def write_fixture(out):
    """A table of round numbers, for checking the seam rather than the physics.

    Not a measurement. Every entry is chosen to be recognisable in a covariance
    so that "Q entered", "Q entered twice" and "Q entered in the wrong units"
    are three visibly different outcomes: 100 um on loc0, 200 um on loc1,
    1 mrad on phi, 2 mrad on theta, 1e-3 on q/p, and the -0.86
    position-direction correlation §2.6 measured, so the off-diagonal structure
    is exercised too.

    It exists because the material-off measurement does not: only the
    `_pull.npz` sidecars came across, and re-measuring needs a held-out dump
    from a material-off training run, which is item 7's business anyway since
    the target surface changes first.
    """
    s = np.array([100.0 * UM, 200.0 * UM, 1.0 * MRAD, 2.0 * MRAD, 1e-3])
    R = np.eye(5)
    R[0, 2] = R[2, 0] = -0.86
    R[1, 3] = R[3, 1] = -0.86
    q = R * s[:, None] * s[None, :]

    with open(out, "wb") as f:
        # material = 0. A fixture is by construction free of scattering, and
        # writing 1 here would only mean the C++ refused to load its own test.
        f.write(struct.pack("<6i", MAGIC_PROV, len(CLASSES), N_PT, N_ETA, 0, 1))
        f.write(FIXTURE_MD5.encode("ascii"))
        f.write(struct.pack("<d", FIXTURE_GATE))
        for _ in CLASSES:
            f.write(q.astype("<f8").tobytes())
        for _ in CLASSES:
            for _ in range(N_PT):
                for _ in range(N_ETA):
                    f.write(struct.pack("<B", 0))       # fall back everywhere
                    f.write(np.zeros((5, 5)).astype("<f8").tobytes())
    print(f"{out}  FIXTURE, not a measurement. Same value in every class:")
    print(f"  loc0 100 um  loc1 200 um  phi 1 mrad  theta 2 mrad  q/p 1e-3, "
          f"rho(loc,dir) = -0.86")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", action="store_true",
                    help="write cpp/q_table_fixture.bin instead: round numbers "
                         "for checking the seam, not a measurement")
    ap.add_argument("--table", default="Q_nomat.parquet")
    ap.add_argument("--corr", default=None,
                    help="npz with a 5x5 correlation per class; identity if "
                         "absent, which drops the -0.86 position-direction "
                         "term and is stated in the output when it happens")
    ap.add_argument("--material-off", action="store_true",
                    help="assert this table was measured with material OFF. "
                         "Without it the C++ refuses to load the result.")
    ap.add_argument("--out", default="cpp/q_table.bin")
    ap.add_argument("--weights", default=None,
                    help="the cpp/gtheta_weights.hpp the residuals in --table "
                         "were dumped from. The digest of its non-comment "
                         "lines goes in the header and NoiseTable::load "
                         "refuses a table whose digest is not the one the "
                         "binary was built from.")
    ap.add_argument("--weights-md5", default=None,
                    help="that md5 directly, for a header that is no longer on "
                         "disk. Exactly one of --weights and --weights-md5.")
    ap.add_argument("--field-gate", type=float, default=None,
                    help="the FIELD_GATE the dump was made at, in tesla. The "
                         "threshold decides which transport made each jump the "
                         "table was measured on, so a table is a measurement "
                         "at one threshold and at no other.")
    ap.add_argument("--helix-table", default=None,
                    help="a second measured table for jumps the network did "
                         "not fire on. With it, --table is the fired branch "
                         "and the file carries both; without it the file "
                         "carries one branch and the stepper arms it on "
                         "both, which is what every table before this one "
                         "did.")
    a = ap.parse_args()

    if a.fixture:
        OUT.mkdir(exist_ok=True)
        write_fixture(OUT / "q_table_fixture.bin")
        return

    # Refused rather than defaulted, the same rule as --material-off. A table
    # that cannot say what it was measured against is a measured defect, and a
    # default here would put it back.
    if (a.weights is None) == (a.weights_md5 is None):
        raise SystemExit("exactly one of --weights and --weights-md5 is "
                         "required: the header records which model's error "
                         "this table is, and nothing in the numbers does")
    if a.field_gate is None:
        raise SystemExit("--field-gate is required: the threshold decides "
                         "which transport made each jump the table was "
                         "measured on, and a table measured at one threshold "
                         "describes a different population at another")
    wmd5 = (a.weights_md5 if a.weights_md5 is not None
            else weights_digest(a.weights))
    if len(wmd5) != 32 or any(c not in "0123456789abcdef" for c in wmd5):
        raise SystemExit("--weights-md5 must be 32 lower-case hex digits, "
                         "got " + repr(wmd5))

    corr = {}
    if a.corr:
        d = np.load(a.corr)
        for c in CLASSES:
            if c in d.files:
                corr[c] = d[c]
    if not corr:
        print("no correlation file: Q will be diagonal. The -0.86")
        print("position-direction term is the chain's only brake, so this")
        print("is a real loss and not a detail.")

    def assemble(path):
        """One measured table, as (fallback, cells, have, filled cells)."""
        t = pl.read_parquet(path)
        fallback = np.zeros((len(CLASSES), 5, 5))
        cells = np.zeros((len(CLASSES), N_PT, N_ETA, 5, 5))
        have = np.zeros((len(CLASSES), N_PT, N_ETA), dtype=np.uint8)
        n_cells = 0
        for ci, cls in enumerate(CLASSES):
            R = corr.get(cls, np.eye(5))
            sub = t.filter(pl.col("cls") == cls)
            for row in sub.iter_rows(named=True):
                q = cov_of(row, R.copy())
                i, j = int(row["pt_bin"]), int(row["eta_bin"])
                if i < 0 or j < 0:
                    fallback[ci] = q
                elif i < N_PT and j < N_ETA:
                    cells[ci, i, j] = q
                    have[ci, i, j] = 1
                    n_cells += 1
            if not (fallback[ci] != 0).any():
                print(f"  {cls}: NO fallback row (pt_bin = eta_bin = -1). Cells "
                      f"the measurement did not fill will get zero noise.")
        return fallback, cells, have, n_cells

    # Branch 0 is the fired transport and branch 1 is the helix alone, which is
    # `LearnedNoise::Branch`'s own order. One table stays one branch and the
    # file it writes is byte for byte what it always was.
    tables = [assemble(a.table)]
    if a.helix_table:
        tables.append(assemble(a.helix_table))
    n_branch = len(tables)

    out = pathlib.Path(a.out)
    out.parent.mkdir(exist_ok=True)
    with open(out, "wb") as f:
        # One header for one branch and for two. The branch count was the last
        # field a reader could infer wrongly from the file length and it has
        # been written down since the branch axis; the two provenance fields
        # follow it for the same reason.
        f.write(struct.pack("<6i", MAGIC_PROV, len(CLASSES), N_PT, N_ETA,
                            0 if a.material_off else 1, n_branch))
        f.write(wmd5.encode("ascii"))
        f.write(struct.pack("<d", float(a.field_gate)))
        for fallback, _, _, _ in tables:
            for ci in range(len(CLASSES)):
                f.write(fallback[ci].astype("<f8").tobytes())
        for _, cells, have, _ in tables:
            for ci in range(len(CLASSES)):
                for i in range(N_PT):
                    for j in range(N_ETA):
                        f.write(struct.pack("<B", int(have[ci, i, j])))
                        f.write(cells[ci, i, j].astype("<f8").tobytes())

    names = ["fired", "helix alone"]
    print(f"{out}  {n_branch} branch(es), material "
          f"{'OFF' if a.material_off else 'ON (the C++ will refuse it)'}")
    print(f"  measured against weights {wmd5} at FIELD_GATE={a.field_gate:g}")
    for b, (fallback, _, _, n_cells) in enumerate(tables):
        print(f"  branch {b} ({names[b]}): {n_cells} filled cells of "
              f"{len(CLASSES) * N_PT * N_ETA}")
        for ci, cls in enumerate(CLASSES):
            d = np.sqrt(np.diag(fallback[ci]))
            print(f"    {cls:8} fallback sigma  loc0 {d[0] * 1e3:6.1f} um  "
                  f"loc1 {d[1] * 1e3:6.1f} um  phi {d[2] * 1e3:6.3f} mrad  "
                  f"theta {d[3] * 1e3:6.3f} mrad  q/p {d[4]:.3e}")


if __name__ == "__main__":
    main()
