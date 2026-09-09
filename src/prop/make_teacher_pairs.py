"""Turn Fatras SimHit CSVs into surface-to-surface teacher pairs (host side).

One row per jump:

    state on surface k  ->  state on surface k+1

State is (position, direction, q/p); the pair also carries the two surface
identifiers and the geometry of the step. Surface IDs are recorded for
bookkeeping but must not be used as model inputs: there are
28,478 distinct surface pairs with the most frequent 3,462 covering only half
the jumps, so anything that keys on identity will not generalise.

Also computes, per pair, the helix prediction and its residual. That is the
quantity g_theta has to learn, and having it in the same table means the
target distribution can be inspected before any training happens. It doubles
as a validation of the whole pipeline: these residuals should reproduce the
field-map deviations measured earlier (sub-resolution in the barrel, mm-scale
past |eta| = 1.5).

Usage:
    python make_teacher_pairs.py teacher/map_nomat --out teacher_map_nomat.parquet
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import polars as pl

from .helix_variants import field_source

MM = 1e-3
K = 0.299792458e-3          # GeV / (T mm)


def _event_of(path):
    """CSV files are per event and particle ids restart in each one, so the
    event number has to join the identity key. Without it, hits from
    different events chain into one 'particle' and the consecutive-pair
    construction silently produces jumps between unrelated tracks."""
    stem = Path(path).stem                      # event000000007-hits
    return int(stem.split("-")[0].replace("event", ""))


def load_dir(d):
    def cat(pattern):
        frames = []
        for f in sorted(glob.glob(f"{d}/{pattern}")):
            frames.append(pl.read_csv(f).with_columns(
                pl.lit(_event_of(f), dtype=pl.Int64).alias("event")))
        return pl.concat(frames, how="vertical_relaxed")
    return cat("*hits.csv"), cat("*particles.csv")


def helix_positions(x0, y0, z0, px, py, pz, q, bz, s):
    """Position on the helix at transverse arc length s."""
    pt = np.hypot(px, py)
    phi0 = np.arctan2(py, px)
    kappa = -0.3 * q * bz / pt * MM             # 1/mm
    phi = phi0 + kappa * s
    return (x0 + (np.sin(phi) - np.sin(phi0)) / kappa,
            y0 - (np.cos(phi) - np.cos(phi0)) / kappa,
            z0 + (pz / pt) * s)


def helix_momenta(px, py, pz, q, bz, s):
    """Momentum on the helix at the same arc length.

    Free once the arc length is known: the transverse momentum just rotates by
    the turning angle and pz is conserved, since a magnetic field does no work.
    That is also why there is no helix prediction for q/p to correct -- the
    helix says |p| does not change, and with material on that is exactly the
    part it gets wrong.
    """
    pt = np.hypot(px, py)
    kappa = -0.3 * q * bz / pt * MM              # same kappa as the positions
    phi = np.arctan2(py, px) + kappa * s         # the tangent turns with it
    return pt * np.cos(phi), pt * np.sin(phi), pz


def helix_predict(x0, y0, z0, px, py, pz, q, bz, r1, z1, endcap, iters=60):
    """Helix prediction on the destination surface.

    Barrel-like jumps (radius changes more than z) are intersected with the
    cylinder r = r1; endcap-like jumps with the plane z = z1, which is closed
    form. Using the cylinder for everything is what made the first version
    unusable: forward jumps move ~1 m in z while the radius barely changes,
    so the cylinder solve is ill-conditioned or has no solution at all.

    Returns positions and a validity mask; low-pT helices that curl back
    before reaching r1 have no intersection and must be dropped rather than
    silently reported at the bracket edge.

    The bisection bracket must stop at half a turn. Distance from the beam
    line varies sinusoidally with turning angle, so a helix that starts
    outward keeps growing for at most pi*R of arc and then comes back. A
    fixed 10 m bracket lets a 0.6 GeV track (R ~ 950 mm) wind 1.7 times, the
    inside/outside predicate stops being monotonic, and the bisection
    converges on whichever later crossing it happens to hit -- with z =
    z0 + (pz/pT) s that produced |dz| errors up to 6 m. Found only after
    regenerating log-uniform in pT, since the uniform sample had almost no
    tracks soft enough to wind.
    """
    pt = np.hypot(px, py)
    s_plane = np.where(np.abs(pz) > 1e-9, (z1 - z0) * pt / pz, 0.0)

    radius = pt / (0.3 * np.abs(bz) * MM)       # mm, = 1/|kappa|
    lo = np.zeros_like(pt)
    hi = np.pi * radius                         # half a turn, no further
    xh, yh, _ = helix_positions(x0, y0, z0, px, py, pz, q, bz, hi)
    reaches = np.hypot(xh, yh) >= r1            # does the helix get there?
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        x, y, _ = helix_positions(x0, y0, z0, px, py, pz, q, bz, mid)
        inside = np.hypot(x, y) < r1
        lo = np.where(inside, mid, lo)
        hi = np.where(inside, hi, mid)
    s = np.where(endcap, s_plane, 0.5 * (lo + hi))
    ok = np.where(endcap, np.isfinite(s_plane) & (s_plane > 0), reaches)
    return (helix_positions(x0, y0, z0, px, py, pz, q, bz, s)
            + helix_momenta(px, py, pz, q, bz, s) + (s, ok))


def plane_predict(x0, y0, z0, px, py, pz, q, bz, c, n, iters=8):
    """Helix intersection with the module plane n . (X - c) = 0.

    Newton on g(s) = n . (P(s) - c), where s is the same transverse arc length
    the cylinder solve uses so everything downstream keeps one definition of
    the step. g is smooth and the crossing is near normal incidence on a
    tracker module, so Newton converges in three or four iterations from the
    straight-line guess; the bisection the cylinder needs is not required and
    would cost sixty evaluations for the same answer.

    Returns the crossing, the momentum there, s, and a validity mask.

    `ok` is false for a grazing crossing, where n . dP/ds is near zero and
    Newton walks off, and for a solution behind the source. Both are dropped
    rather than reported: a module the track leaves through the edge is not a
    jump the teacher should describe.
    """
    pt = np.hypot(px, py)
    phi0 = np.arctan2(py, px)
    kappa = -0.3 * q * bz / pt * MM

    def g_and_dg(s):
        phi = phi0 + kappa * s
        gx = x0 + (np.sin(phi) - np.sin(phi0)) / kappa - c[:, 0]
        gy = y0 - (np.cos(phi) - np.cos(phi0)) / kappa - c[:, 1]
        gz = z0 + (pz / pt) * s - c[:, 2]
        g = n[:, 0] * gx + n[:, 1] * gy + n[:, 2] * gz
        dg = (n[:, 0] * np.cos(phi) + n[:, 1] * np.sin(phi)
              + n[:, 2] * (pz / pt))
        return g, dg

    g0, dg0 = g_and_dg(np.zeros_like(pt))
    # near-normal incidence means |dg| is close to |p|/pt; a small one is a
    # module the track is skimming
    grazing = np.abs(dg0) < 1e-3
    s = np.where(grazing, 0.0, -g0 / np.where(grazing, 1.0, dg0))
    for _ in range(iters):
        g, dg = g_and_dg(s)
        step = np.where(np.abs(dg) < 1e-12, 0.0, g / np.where(
            np.abs(dg) < 1e-12, 1.0, dg))
        s = s - step
    g, dg = g_and_dg(s)
    ok = (~grazing) & np.isfinite(s) & (s > 0) & (np.abs(g) < 1e-6)
    return (helix_positions(x0, y0, z0, px, py, pz, q, bz, s)
            + helix_momenta(px, py, pz, q, bz, s) + (s, ok))


def load_modules(path):
    """The geometry dump, keyed the way the hit CSV's ids can actually be
    joined on.

    `join_key` is `geometry_id` with the low 8 bits cleared. The hit CSV puts
    the endcap disc ring in those bits and the `TrackingGeometry` leaves them
    zero, so a join on the raw id keeps every barrel row and no endcap row.
    That is 27% of the teacher and it looks like a partial geometry rather than
    a key mismatch.
    """
    m = pl.read_csv(path).with_columns(pl.col("join_key").cast(pl.UInt64))
    return m.select(["join_key", "cx", "cy", "cz", "nx", "ny", "nz",
                     "u0x", "u0y", "u0z", "u1x", "u1y", "u1z",
                     "r_centre", "tilt_rad", "w_phi"]).unique(subset="join_key")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir")
    ap.add_argument("--out", default="teacher_pairs.parquet")
    ap.add_argument("--modules", default="cpp/modules.csv",
                    help="geom_dump output. The destination becomes the module "
                         "plane named by the next hit's geometry_id, which is "
                         "a surface that exists in the detector and is known "
                         "at run time. Without it the destination stays the "
                         "cylinder r = r1, which is derived from the answer.")
    ap.add_argument("--bz", default="2.0",
                    help="field the helix baseline flies at. A number or "
                         "`const` is uniform, in tesla; a map npz makes the "
                         "baseline the SOURCE core, Bz at each jump's own "
                         "source point, which is what "
                         "`LearnedTransport.hpp`'s `coreBz` runs when "
                         "`localCore` is set. Every helix_*, plane_* and "
                         "res_* column moves with it.")
    ap.add_argument("--min-dr", type=float, default=20.0,
                    help="minimum radial step; double-sided strip modules "
                         "give hit pairs 6-7 mm apart that are not layer "
                         "transitions")
    args = ap.parse_args()

    hits, parts = load_dir(args.indir)
    print(f"loaded {hits.height:,} hits, {parts.height:,} particles")

    pid = ["event", "particle_id_pv", "particle_id_sv", "particle_id_part",
           "particle_id_gen", "particle_id_subpart"]
    charge = parts.select(pid + ["q"]).unique(subset=pid)
    print(f"  {charge.height:,} distinct (event, particle) — should be "
          f"events x tracks")

    h = (hits.join(charge, on=pid, how="inner")
              .with_columns(
                  (pl.col("tx") ** 2 + pl.col("ty") ** 2).sqrt().alias("r"),
                  (pl.col("tpx") ** 2 + pl.col("tpy") ** 2).sqrt().alias("pt"),
              )
              .sort(pid + ["tt"]))

    # consecutive hits of the same particle = one jump
    nxt = {c: pl.col(c).shift(-1).over(pid) for c in
           ("tx", "ty", "tz", "tpx", "tpy", "tpz", "te", "geometry_id", "r")}
    pairs = (h.with_columns(**{f"n_{k}": v for k, v in nxt.items()})
              .drop_nulls("n_tx")
              .with_columns(
                  (pl.col("n_r") - pl.col("r")).alias("dr_"),
                  (pl.col("n_tz") - pl.col("tz")).alias("dz_"))
              # a step must move somewhere: either outward in r (barrel) or
              # along z (endcap). Double-sided strip modules give hit pairs
              # 6-7 mm apart that are not layer transitions.
              .filter((pl.col("dr_").abs() > args.min_dr)
                      | (pl.col("dz_").abs() > args.min_dr)))
    print(f"{pairs.height:,} jumps after the {args.min_dr} mm step cut")

    # --- the destination module plane
    mods = load_modules(args.modules)
    before = pairs.height
    pairs = (pairs
             .with_columns((pl.col("n_geometry_id").cast(pl.UInt64)
                            & pl.lit(0xFFFFFFFFFFFFFF00, pl.UInt64))
                           .alias("_mkey"))
             .join(mods, left_on="_mkey", right_on="join_key", how="inner"))
    print(f"  joined the module geometry on {pairs.height:,} of {before:,} "
          f"({100 * pairs.height / max(before, 1):.1f}%)")
    if pairs.height < before:
        print("  rows without a module are dropped. A large loss here means "
              "the geometry dump\n  and the hits came from different ODD "
              "builds, not that the join key is wrong.")

    d = pairs.to_dict(as_series=False)
    p_mag = np.sqrt(np.array(d["tpx"]) ** 2 + np.array(d["tpy"]) ** 2
                    + np.array(d["tpz"]) ** 2)
    q = np.array(d["q"], dtype=float)
    endcap = np.abs(np.array(d["dz_"])) > np.abs(np.array(d["dr_"]))

    # The field the baseline helix flies at, per jump. `field_source` is the
    # one place this repository resolves that argument, so `const`, `const:1.5`
    # and a map path mean here what they mean to `train_gtheta` and
    # `export_kernel`. A bare number is accepted so the old scalar spelling
    # keeps working.
    try:
        fm = field_source(f"const:{float(args.bz)}")
    except ValueError:
        fm = field_source(args.bz)
    bz = fm.bz(np.array(d["tx"]), np.array(d["ty"]), np.array(d["tz"]))
    print(f"  helix core field: {args.bz}, Bz(source) {bz.min():.4f} to "
          f"{bz.max():.4f} T, median {np.median(bz):.4f}")

    hx, hy, hz, hpx, hpy, hpz, hs, ok = helix_predict(
        np.array(d["tx"]), np.array(d["ty"]), np.array(d["tz"]),
        np.array(d["tpx"]), np.array(d["tpy"]), np.array(d["tpz"]),
        q, bz, np.array(d["n_r"]), np.array(d["n_tz"]), endcap)
    print(f"  {endcap.sum():,} endcap-like (z-plane target), "
          f"{(~endcap).sum():,} barrel-like (cylinder target); "
          f"{(~ok).sum():,} with no helix solution -> dropped")

    # --- the same jump solved onto the module plane
    #
    # The cylinder above is kept alongside rather than replaced. Everything
    # measured so far is quoted against it, `jacobian_helix.py`'s regression
    # runs on it, and having both in one table is what makes the two targets
    # comparable on identical jumps instead of across two regenerations.
    cen = np.column_stack([d["cx"], d["cy"], d["cz"]])
    nrm = np.column_stack([d["nx"], d["ny"], d["nz"]])
    u0 = np.column_stack([d["u0x"], d["u0y"], d["u0z"]])
    u1 = np.column_stack([d["u1x"], d["u1y"], d["u1z"]])
    px_, py_, pz_ = (np.array(d["tpx"]), np.array(d["tpy"]),
                     np.array(d["tpz"]))
    src = np.column_stack([d["tx"], d["ty"], d["tz"]])

    ppx, ppy, ppz, pmx, pmy, pmz, ps, pok = plane_predict(
        src[:, 0], src[:, 1], src[:, 2], px_, py_, pz_, q, bz, cen, nrm)
    print(f"  module plane: {(~pok).sum():,} with no solution")

    # local coordinates on the module, which is what a residual has to be
    # quoted in. The barrel modules are tilted 150 mrad off the cylinder
    # tangent, so these are not the r-phi and z the cylinder target
    # used, and the two numbers are not interchangeable.
    hit = np.column_stack([d["n_tx"], d["n_ty"], d["n_tz"]])
    pred = np.column_stack([ppx, ppy, ppz])
    hit_loc0 = ((hit - cen) * u0).sum(1)
    hit_loc1 = ((hit - cen) * u1).sum(1)
    pred_loc0 = ((pred - cen) * u0).sum(1)
    pred_loc1 = ((pred - cen) * u1).sum(1)

    # how far off the named plane the true hit lies. It should be zero, and it
    # is the check that the join is correct rather than merely successful.
    off_plane = ((hit - cen) * nrm).sum(1)
    print(f"  |(true hit - module centre) . normal|  max "
          f"{np.abs(off_plane).max():.2e} mm")

    # the two features that replace the truth-derived ones. Both are knowable
    # at run time from the target module alone.
    #
    # The normal's sign is a convention, not physics. Which face of a module
    # DD4hep calls the front is arbitrary and varies between modules, so
    # `d_plane` and `cos_inc` both flip with it and a network given them raw
    # would spend capacity memorising per-module conventions -- which is the
    # same failure the surface identifier was excluded for (§2.2). The two
    # combinations below are invariant under that flip, because both quantities
    # change sign together:
    #
    #   path_to_plane = d_plane / cos_inc   straight-line distance to the plane
    #                                       along the momentum, positive ahead
    #   abs_cos_inc   = |cos_inc|           obliquity of the crossing
    #
    # The raw pair is kept beside them for bookkeeping and for anyone checking
    # this reasoning.
    d_plane = ((cen - src) * nrm).sum(1)
    p_mag_ = np.sqrt(px_ ** 2 + py_ ** 2 + pz_ ** 2)
    cos_inc = (np.column_stack([px_, py_, pz_]) * nrm).sum(1) / p_mag_
    abs_cos_inc = np.abs(cos_inc)
    path_to_plane = np.where(abs_cos_inc > 1e-9,
                             d_plane / np.where(abs_cos_inc > 1e-9,
                                                cos_inc, 1.0), 0.0)
    print(f"  normal sign: {100 * (cos_inc < 0).mean():.1f}% of jumps have "
          f"p . n < 0, which is why the raw pair is not a feature")

    phi_pred = np.arctan2(hy, hx)
    phi_true = np.arctan2(np.array(d["n_ty"]), np.array(d["n_tx"]))
    dphi = (phi_pred - phi_true + np.pi) % (2 * np.pi) - np.pi

    eta = np.arctanh(np.clip(np.array(d["tpz"]) / p_mag, -0.999999, 0.999999))

    out = pl.DataFrame({
        # --- inputs: state on the source surface ---
        "x": d["tx"], "y": d["ty"], "z": d["tz"],
        "px": d["tpx"], "py": d["tpy"], "pz": d["tpz"],
        "qop": q / p_mag,
        # --- step geometry (model inputs; surface IDs are not) ---
        "r0": d["r"], "r1": d["n_r"],
        "dr": np.array(d["n_r"]) - np.array(d["r"]),
        "dz": np.array(d["n_tz"]) - np.array(d["tz"]),
        # --- targets: state on the destination surface ---
        "x1": d["n_tx"], "y1": d["n_ty"], "z1": d["n_tz"],
        "px1": d["n_tpx"], "py1": d["n_tpy"], "pz1": d["n_tpz"],
        # --- bookkeeping only ---
        "surf0": d["geometry_id"], "surf1": d["n_geometry_id"],
        # a track key, so per-hit numbers can be turned into per-track ones
        # (a gate failure is a hole, and it is holes per track that decide
        # whether the track survives, not the per-hit rate)
        "track": (np.array(d["event"], dtype=np.int64) * 1_000_000
                  + np.array(d["particle_id_part"], dtype=np.int64)),
        # --- helix baseline and the residual g_theta must learn ---
        "helix_x": hx, "helix_y": hy, "helix_z": hz,
        # v1 outputs: the helix's direction at the same arc length, and the
        # arc length itself. The helix has no q/p prediction to correct --
        # |p| is conserved in a magnetic field, so its q/p target is the
        # source q/p and every deviation from it is material.
        "helix_px": hpx, "helix_py": hpy, "helix_pz": hpz, "helix_s": hs,
        "res_rphi_um": np.array(d["n_r"]) * dphi * 1000,
        "res_z_um": (hz - np.array(d["n_tz"])) * 1000,
        # --- the module plane: the destination that exists in the detector ---
        # centre, normal and both local axes, carried so nothing downstream has
        # to re-join the geometry to know what loc0 and loc1 mean
        "mc_x": cen[:, 0], "mc_y": cen[:, 1], "mc_z": cen[:, 2],
        "mn_x": nrm[:, 0], "mn_y": nrm[:, 1], "mn_z": nrm[:, 2],
        "mu0_x": u0[:, 0], "mu0_y": u0[:, 1], "mu0_z": u0[:, 2],
        "mu1_x": u1[:, 0], "mu1_y": u1[:, 1], "mu1_z": u1[:, 2],
        "m_tilt": d["tilt_rad"], "m_wphi": d["w_phi"],
        # the two run-time-knowable features that replace inputs 7 and 8. The
        # old ones were r1 or z1 and their difference from the source, and r1
        # is the TRUE next hit's radius, so the model was being handed a
        # function of the answer. These are properties of the target module.
        "d_plane": d_plane, "cos_inc": cos_inc,
        "path_to_plane": path_to_plane, "abs_cos_inc": abs_cos_inc,
        # helix prediction on the plane, and the same in module-local
        # coordinates. loc0/loc1 replace inputs 9 and 10.
        "plane_x": ppx, "plane_y": ppy, "plane_z": ppz,
        "plane_px": pmx, "plane_py": pmy, "plane_pz": pmz,
        "plane_s": ps, "plane_ok": pok,
        "helix_loc0": pred_loc0, "helix_loc1": pred_loc1,
        "hit_loc0": hit_loc0, "hit_loc1": hit_loc1,
        # the learning target in the frame the sensor resolutions are quoted
        # in. Not res_rphi_um: the barrel modules are tilted 150 mrad off the
        # cylinder tangent, so the cylinder frame and the module frame differ
        # by more than a relabelling.
        "res_loc0_um": (pred_loc0 - hit_loc0) * 1000,
        "res_loc1_um": (pred_loc1 - hit_loc1) * 1000,
        "off_plane_mm": off_plane,
        # --- binning ---
        "pt": d["pt"], "eta": eta, "abs_eta": np.abs(eta),
        "endcap": endcap,
    }).filter(pl.Series(ok & pok))
    out.write_parquet(args.out)
    print(f"wrote {args.out}  ({out.height:,} rows, {len(out.columns)} cols)")

    b = (out.with_columns(
            pl.col("pt").cut([1, 2, 5, 10],
                             labels=["0.5-1", "1-2", "2-5", "5-10", ">10"]).alias("pt_bin"),
            pl.col("abs_eta").cut([0.5, 1.0, 1.5, 2.0],
                                  labels=["0-0.5", "0.5-1", "1-1.5", "1.5-2", ">2"]).alias("eta_bin"))
         .group_by("pt_bin", "eta_bin")
         .agg(pl.col("res_rphi_um").abs().median().round(0).alias("med"),
              pl.len().alias("n"))
         .sort("pt_bin", "eta_bin"))
    print("\n=== the two targets on the same jumps ===")
    keep = ok & pok
    for name, r0, r1_ in (("cylinder r = r1 (truth-derived)",
                           np.array(d["n_r"])[keep] * dphi[keep] * 1000,
                           (hz - np.array(d["n_tz"]))[keep] * 1000),
                          ("module plane (exists in the detector)",
                           ((pred_loc0 - hit_loc0) * 1000)[keep],
                           ((pred_loc1 - hit_loc1) * 1000)[keep])):
        print(f"  {name:40} |res| median {np.median(np.abs(r0)):8.1f} / "
              f"{np.median(np.abs(r1_)):8.1f} um   p99 "
              f"{np.percentile(np.abs(r0), 99):9.1f} / "
              f"{np.percentile(np.abs(r1_), 99):9.1f}")
    print("  The two are in DIFFERENT frames, not two estimates of one number:")
    print("  the barrel modules are tilted 150 mrad off the cylinder tangent.")

    print("\n=== |helix residual| median [um] per jump — the learning target ===")
    print(b.pivot(values="med", index="pt_bin", on="eta_bin", sort_columns=True)
           .sort("pt_bin"))
    print("\n=== rows per cell ===")
    print(b.pivot(values="n", index="pt_bin", on="eta_bin", sort_columns=True)
           .sort("pt_bin"))
    print("\nsanity: with material off these residuals are pure field-map "
          "content, so the barrel should sit below the ~200 um measurement "
          "scale and |eta| > 1.5 well above it.")


if __name__ == "__main__":
    main()
