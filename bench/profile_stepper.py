"""Q2, host half: how much work is one surface-to-surface propagation?

Runs on the bare host (no ACTS, no container). Two separable measurements:

  A. ALGORITHMIC (machine-independent) -- a faithful re-implementation of the
     ACTS EigenStepper adaptive RKN4: same k1..k4 formulation, same L1 error
     estimate, same clamp(sqrt(sqrt(tol/2|err|)), 0.25, 4) step rescale, same
     3 field lookups per attempt (B at pos, at midpoint reused by k2/k3, at
     end). Counts accepted steps / rejected attempts / field lookups to
     traverse the tracker, binned in (pT, |eta|), under constant 2 T vs a
     finite-solenoid field map. This is the multiplier in the cost model and
     it does not depend on the language it was measured in.

  B. KERNEL TIMING (this CPU) -- one RKN step, one trilinear field lookup,
     one closed-form helix solve, one MLP forward, all batched numpy so the
     interpreter overhead is amortized identically across the four. Only the
     RATIOS transfer to C++ ACTS; absolutes do not. Same caveat as the
     WSL-vs-NERSC note in HANDOFF section 4.7.

cost/jump = (steps per jump from A) x (ns per step from B), which keeps the
physics multiplier and the machine constant separate instead of entangling
them in one un-auditable number.

Field: the released ODD map lives in the container (/opt/odd/data/odd-bfield),
so this uses an analytic finite-solenoid stand-in -- exact on-axis,
first-order off-axis (div B = 0 to the order kept), B0=2 T, coil R=1.2 m,
half-length 3 m -- sampled onto a 3D grid and read back through the same
trilinear gather MagneticFieldMapXyz does. The absolute deviations are not
the ODD map's; what carries over is that a field with a real gradient forces
the adaptive stepper to shorten its steps, and where in (pT, eta) that bites.

Writes profile_stepper.parquet (per-track) + prints the tables.
"""
import time

import numpy as np
import polars as pl

K = 0.299792458e-3          # GeV / (T * mm) -- dT/ds = qop*K*(T x B)
TOL = 1e-4                  # ACTS stepTolerance, mm
H0 = 100.0                  # initial step, mm
R_MAX, Z_MAX = 1150.0, 3050.0
RADII = np.array([170.0, 360.0, 660.0, 1015.0])      # as in sweep_heatmap.py
ZPLANES = np.array([700.0, 1300.0, 2000.0, 2600.0, 2950.0])

rng = np.random.default_rng(1234)


# ---------------------------------------------------------------- fields
class ConstField:
    """Uniform 2 T along z. No memory traffic -- the baseline lookup cost."""
    n_lookup_flops = 0

    def __init__(self, bz=2.0):
        self.bz = bz

    def __call__(self, pos):
        b = np.zeros_like(pos)
        b[:, 2] = self.bz
        return b


def solenoid_analytic(r, z, b0=2.0, coil_r=1200.0, half_l=3000.0):
    """Finite solenoid: exact on-axis Bz, first-order off-axis expansion.

    Bz0(z)  = b0/2 * [(z+L)/sqrt((z+L)^2+R^2) - (z-L)/sqrt((z-L)^2+R^2)]
    Br(r,z) = -(r/2) Bz0'(z)      Bz(r,z) = Bz0(z) - (r^2/4) Bz0''(z)
    """
    def bz0(zz):
        a, b = zz + half_l, zz - half_l
        return 0.5 * b0 * (a / np.hypot(a, coil_r) - b / np.hypot(b, coil_r))

    d = 1.0
    dbz = (bz0(z + d) - bz0(z - d)) / (2 * d)
    d2bz = (bz0(z + d) - 2 * bz0(z) + bz0(z - d)) / (d * d)
    return -0.5 * r * dbz, bz0(z) - 0.25 * r * r * d2bz


class MapField:
    """Solenoid sampled on a uniform xyz grid, read back by trilinear gather.

    Deliberately sized past L2 (49*49*129*3 float64 ~ 7.4 MB) so the gather
    pays realistic cache misses, like the real MagneticFieldMapXyz does.
    """

    def __init__(self, step=50.0):
        self.x0, self.y0, self.z0 = -1200.0, -1200.0, -3200.0
        self.d = step
        self.nx = int(2 * 1200 / step) + 1
        self.ny = self.nx
        self.nz = int(2 * 3200 / step) + 1
        gx = self.x0 + self.d * np.arange(self.nx)
        gy = self.y0 + self.d * np.arange(self.ny)
        gz = self.z0 + self.d * np.arange(self.nz)
        X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
        R = np.hypot(X, Y)
        br, bz = solenoid_analytic(R, Z)
        safe = np.where(R > 1e-9, R, 1.0)
        grid = np.empty((self.nx, self.ny, self.nz, 3))
        grid[..., 0] = br * X / safe
        grid[..., 1] = br * Y / safe
        grid[..., 2] = bz
        self.grid = grid.reshape(-1, 3)
        self.mb = self.grid.nbytes / 1e6

    def __call__(self, pos):
        fx = np.clip((pos[:, 0] - self.x0) / self.d, 0, self.nx - 1.001)
        fy = np.clip((pos[:, 1] - self.y0) / self.d, 0, self.ny - 1.001)
        fz = np.clip((pos[:, 2] - self.z0) / self.d, 0, self.nz - 1.001)
        ix, iy, iz = fx.astype(np.intp), fy.astype(np.intp), fz.astype(np.intp)
        tx, ty, tz = fx - ix, fy - iy, fz - iz
        base = (ix * self.ny + iy) * self.nz + iz
        s_y, s_z = self.nz, 1
        s_x = self.ny * self.nz
        out = np.zeros((len(pos), 3))
        for dx in (0, 1):
            wx = tx if dx else 1.0 - tx
            for dy in (0, 1):
                wy = ty if dy else 1.0 - ty
                for dz in (0, 1):
                    wz = tz if dz else 1.0 - tz
                    w = (wx * wy * wz)[:, None]
                    out += w * self.grid[base + dx * s_x + dy * s_y + dz * s_z]
        return out


# ------------------------------------------------------- the ACTS RKN step
def rkn_step(pos, dirn, qop, h, field):
    """One RKN4 attempt. Returns (new_pos, new_dir, error). 3 field lookups."""
    qk = (qop * K)[:, None]
    b1 = field(pos)
    k1 = qk * np.cross(dirn, b1)
    pos1 = pos + (h[:, None] / 2) * dirn + (h[:, None] ** 2 / 8) * k1
    b2 = field(pos1)                                   # reused by k2 and k3
    k2 = qk * np.cross(dirn + (h[:, None] / 2) * k1, b2)
    k3 = qk * np.cross(dirn + (h[:, None] / 2) * k2, b2)
    pos2 = pos + h[:, None] * dirn + (h[:, None] ** 2 / 2) * k3
    b3 = field(pos2)
    k4 = qk * np.cross(dirn + h[:, None] * k3, b3)

    err = h ** 2 * np.abs(k1 - k2 - k3 + k4).sum(axis=1)
    new_pos = pos + h[:, None] * dirn + (h[:, None] ** 2 / 6) * (k1 + k2 + k3)
    new_dir = dirn + (h[:, None] / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    new_dir /= np.linalg.norm(new_dir, axis=1)[:, None]
    return new_pos, new_dir, np.maximum(err, 1e-20)


def dist_to_next_surface(pos, dirn):
    """Straight-line distance to the next un-crossed cylinder or disc.

    This is the Navigator's step-size cap: ACTS never steps past the surface
    it is targeting, so the adaptive h is always min(h_accuracy, h_surface).
    Without it the stepper flies through several layers in one step and the
    high-pT cells come out below 1 step/jump, which cannot happen in CKF.
    """
    r = np.hypot(pos[:, 0], pos[:, 1])
    vt = np.hypot(dirn[:, 0], dirn[:, 1])
    vr = np.where(r > 1e-6,
                  (pos[:, 0] * dirn[:, 0] + pos[:, 1] * dirn[:, 1]) / np.maximum(r, 1e-9),
                  vt)
    az = np.abs(pos[:, 2])
    vz = np.abs(dirn[:, 2])
    d = np.full(len(r), np.inf)
    for s in RADII:
        dd = (s - r) / np.maximum(vr, 1e-6)
        d = np.where((s > r) & (dd > 0), np.minimum(d, dd), d)
    for s in ZPLANES:
        dd = (s - az) / np.maximum(vz, 1e-6)
        d = np.where((s > az) & (dd > 0), np.minimum(d, dd), d)
    return d


def traverse(pos, dirn, qop, field, tol=TOL, max_iter=20000):
    """March every track out of the tracker volume, counting stepper work.

    Vectorised with an active mask: finished tracks are dropped from the
    working set, so counters are exact (the wasted-work concern only affects
    wall clock here, which is measured separately in bench_kernels).
    """
    n = len(qop)
    pos, dirn = pos.copy(), dirn.copy()
    h = np.full(n, H0)
    acc = np.zeros(n, np.int64)      # accepted steps
    att = np.zeros(n, np.int64)      # attempts (accepted + rejected)
    path = np.zeros(n)               # arc length, mm
    xings = np.zeros(n, np.int64)    # cylinder/disc crossings
    idx = np.arange(n)

    for _ in range(max_iter):
        if idx.size == 0:
            break
        p, d, q = pos[idx], dirn[idx], qop[idx]
        # accuracy-limited h, then the navigator's surface cap
        hh = np.minimum(h[idx], np.maximum(dist_to_next_surface(p, d), 1.0))
        np_, nd_, err = rkn_step(p, d, q, hh, field)
        att[idx] += 1

        ok = err <= tol
        scale = np.clip(np.sqrt(np.sqrt(tol / (2.0 * err))), 0.25, 4.0)
        h[idx] = np.where(ok, hh * scale, hh * np.minimum(scale, 1.0))

        a = idx[ok]
        if a.size:
            r_old = np.hypot(pos[a, 0], pos[a, 1])
            z_old = pos[a, 2]
            pos[a], dirn[a] = np_[ok], nd_[ok]
            acc[a] += 1
            path[a] += hh[ok]
            r_new = np.hypot(pos[a, 0], pos[a, 1])
            z_new = pos[a, 2]
            for s in RADII:
                xings[a] += ((r_old < s) & (r_new >= s)).astype(np.int64)
            for s in ZPLANES:
                xings[a] += ((np.abs(z_old) < s) & (np.abs(z_new) >= s)).astype(np.int64)

        r = np.hypot(pos[idx, 0], pos[idx, 1])
        done = (r > R_MAX) | (np.abs(pos[idx, 2]) > Z_MAX) | (path[idx] > 20000)
        idx = idx[~done]

    return acc, att, path, xings


# ----------------------------------------------------------- the baselines
def helix_to_cylinder(pos, dirn, qop, bz, r_target, iters=3):
    """Closed-form helix + Newton onto a cylinder -- the f_helix core."""
    kappa = qop * K * bz                       # 1/radius, signed
    ct = dirn[:, 2]
    st = np.sqrt(np.maximum(1.0 - ct * ct, 1e-12))
    phi0 = np.arctan2(dirn[:, 1], dirn[:, 0])
    r0 = np.hypot(pos[:, 0], pos[:, 1])
    s = (r_target - r0) / np.maximum(st, 1e-6)  # straight-line seed
    for _ in range(iters):
        psi = kappa * s * st
        safe = np.where(np.abs(kappa) > 1e-12, kappa, 1e-12)
        x = pos[:, 0] + (np.sin(phi0 + psi) - np.sin(phi0)) / safe
        y = pos[:, 1] - (np.cos(phi0 + psi) - np.cos(phi0)) / safe
        r = np.hypot(x, y)
        drds = st * (np.cos(phi0 + psi) * x + np.sin(phi0 + psi) * y) / np.maximum(r, 1e-9)
        s = s - (r - r_target) / np.where(np.abs(drds) > 1e-9, drds, 1e-9)
    psi = kappa * s * st
    safe = np.where(np.abs(kappa) > 1e-12, kappa, 1e-12)
    out = np.empty_like(pos)
    out[:, 0] = pos[:, 0] + (np.sin(phi0 + psi) - np.sin(phi0)) / safe
    out[:, 1] = pos[:, 1] - (np.cos(phi0 + psi) - np.cos(phi0)) / safe
    out[:, 2] = pos[:, 2] + s * ct
    return out


class MLP:
    """g_theta stand-in: 12 -> 64 -> 64 -> 6. Three GEMMs plus activations.

    Activation is a first-class cost decision here, not a detail: at this
    width the GEMMs are tiny and the elementwise activation is what the run
    time is actually spent on, so tanh vs relu moves the learned-path cost
    by more than the matrix sizes do.
    """

    def __init__(self, din=12, dh=64, dout=6, act="tanh"):
        g = np.random.default_rng(0)
        self.w1, self.b1 = g.normal(size=(din, dh)) * 0.1, np.zeros(dh)
        self.w2, self.b2 = g.normal(size=(dh, dh)) * 0.1, np.zeros(dh)
        self.w3, self.b3 = g.normal(size=(dh, dout)) * 0.1, np.zeros(dout)
        self.act = np.tanh if act == "tanh" else (lambda v: np.maximum(v, 0.0, out=v))

    def __call__(self, x):
        a = self.act(x @ self.w1 + self.b1)
        a = self.act(a @ self.w2 + self.b2)
        return a @ self.w3 + self.b3


def timeit(fn, repeat=7):
    """Min-of-repeat wall clock, in seconds. Min, not mean: the quantity
    wanted is the machine's capability, not the scheduler's noise."""
    fn()                                            # warm caches / BLAS init
    return min(_t(fn) for _ in range(repeat))


def _t(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def bench_kernels(nb=20000):
    """Per-track ns for each kernel, all batched numpy, same batch size."""
    pos = np.column_stack([rng.uniform(-800, 800, nb), rng.uniform(-800, 800, nb),
                           rng.uniform(-2500, 2500, nb)])
    d = rng.normal(size=(nb, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    qop = rng.choice([-1.0, 1.0], nb) / rng.uniform(0.5, 50.0, nb)
    h = np.full(nb, 50.0)
    cf, mf = ConstField(), MapField()
    mlp_tanh, mlp_relu = MLP(act="tanh"), MLP(act="relu")
    feats = rng.normal(size=(nb, 12))

    out = {}
    out["lookup_map"] = timeit(lambda: mf(pos)) / nb * 1e9
    out["lookup_const"] = timeit(lambda: cf(pos)) / nb * 1e9
    out["rkn_step_map"] = timeit(lambda: rkn_step(pos, d, qop, h, mf)) / nb * 1e9
    out["rkn_step_const"] = timeit(lambda: rkn_step(pos, d, qop, h, cf)) / nb * 1e9
    out["helix"] = timeit(lambda: helix_to_cylinder(pos, d, qop, 2.0, 1015.0)) / nb * 1e9
    out["mlp_tanh"] = timeit(lambda: mlp_tanh(feats)) / nb * 1e9
    out["mlp_relu"] = timeit(lambda: mlp_relu(feats.copy())) / nb * 1e9
    out["_map_mb"] = mf.mb
    return out


# ------------------------------------------------------------------- main
def main(n_tracks=4000):
    pt = 10 ** rng.uniform(np.log10(0.5), np.log10(50.0), n_tracks)   # log-uniform
    eta = rng.uniform(-3.0, 3.0, n_tracks)
    phi = rng.uniform(0, 2 * np.pi, n_tracks)
    charge = rng.choice([-1.0, 1.0], n_tracks)

    theta = 2 * np.arctan(np.exp(-eta))
    dirn = np.column_stack([np.sin(theta) * np.cos(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(theta)])
    p = pt / np.sin(theta)
    qop = charge / p
    pos = np.zeros((n_tracks, 3))

    print(f"tracks: {n_tracks}   pT 0.5-50 GeV log-uniform, |eta| < 3")
    fields = {"const2T": ConstField(), "map": MapField()}
    print(f"field map grid: {fields['map'].mb:.1f} MB "
          f"({fields['map'].nx}x{fields['map'].ny}x{fields['map'].nz} @ 50 mm)")

    cols = {"pt": pt, "eta": eta}
    for name, f in fields.items():
        t0 = time.perf_counter()
        acc, att, path, xings = traverse(pos, dirn, qop, f)
        print(f"  traversed under {name} in {time.perf_counter()-t0:.1f} s")
        cols[f"steps_{name}"] = acc
        cols[f"attempts_{name}"] = att
        cols[f"path_{name}"] = path
        cols[f"xings_{name}"] = xings

    df = pl.DataFrame(cols)
    # a "jump" = one surface-to-surface hop; tracks that cross <2 surfaces
    # (very low pT curlers, dead-end trajectories) have no well-defined jump
    df = df.with_columns([
        (pl.col(f"steps_{n}") / pl.col(f"xings_{n}").clip(1)).alias(f"steps_per_jump_{n}")
        for n in fields
    ] + [
        (3 * pl.col(f"attempts_{n}") / pl.col(f"xings_{n}").clip(1)).alias(f"lookups_per_jump_{n}")
        for n in fields
    ] + [
        (1 - pl.col(f"steps_{n}") / pl.col(f"attempts_{n}")).alias(f"reject_{n}")
        for n in fields
    ])
    df = df.filter((pl.col("xings_const2T") >= 2) & (pl.col("xings_map") >= 2))
    df.write_parquet("profile_stepper.parquet")
    print(f"tracks with >=2 surface crossings: {df.height}")

    print("\n=== A. algorithmic: RKN steps per surface-to-surface jump ===")
    b = df.with_columns(
        pl.col("pt").cut([1, 2, 5, 10], labels=["0.5-1", "1-2", "2-5", "5-10", ">10"]).alias("pt_bin"),
        pl.col("eta").abs().cut([0.5, 1.0, 1.5, 2.0], labels=["0-0.5", "0.5-1", "1-1.5", "1.5-2", ">2"]).alias("eta_bin"),
    )
    for n in fields:
        agg = (b.group_by("pt_bin", "eta_bin")
               .agg(pl.col(f"steps_per_jump_{n}").median().round(1).alias("m"))
               .sort("pt_bin", "eta_bin"))
        print(f"\n-- {n} --")
        print(agg.pivot(values="m", index="pt_bin", on="eta_bin", sort_columns=True).sort("pt_bin"))

    print("\n=== whole-track totals (median) ===")
    for n in fields:
        print(f"  {n:8s}  steps={df[f'steps_{n}'].median():7.1f}  "
              f"field lookups={3*df[f'attempts_{n}'].median():8.1f}  "
              f"reject frac={df[f'reject_{n}'].median():.3f}")

    print("\n=== B. kernel timing on this CPU (batched numpy, ns/track) ===")
    k = bench_kernels()
    for key in ("lookup_const", "lookup_map", "rkn_step_const", "rkn_step_map",
                "helix", "mlp_tanh", "mlp_relu"):
        print(f"  {key:16s} {k[key]:9.1f} ns")
    share = 3 * k["lookup_map"] / k["rkn_step_map"]
    print(f"\n  field lookups = {share*100:.0f}% of one RKN step under the map")

    print("\n=== C. cost model: ns per surface-to-surface jump ===")
    spj = {n: float(df[f"steps_per_jump_{n}"].median()) for n in fields}
    rkn_map = spj["map"] * k["rkn_step_map"]
    rkn_const = spj["const2T"] * k["rkn_step_const"]
    print(f"  RKN under map      {spj['map']:5.1f} steps x {k['rkn_step_map']:6.1f} = {rkn_map:9.1f} ns")
    print(f"  RKN under const 2T {spj['const2T']:5.1f} steps x {k['rkn_step_const']:6.1f} = {rkn_const:9.1f} ns")
    for a in ("tanh", "relu"):
        learned = k["helix"] + k[f"mlp_{a}"]
        print(f"  helix + g_theta ({a:4s})                       = {learned:9.1f} ns"
              f"   (helix {k['helix']:.1f} + mlp {k[f'mlp_{a}']:.1f})"
              f"  -> {rkn_map/learned:5.2f}x vs map, {rkn_const/learned:5.2f}x vs const")

    print("\n  high-|eta| slice, where the learning target actually lives:")
    hi = df.filter(pl.col("eta").abs() > 1.5)
    spj_hi = float(hi["steps_per_jump_map"].median())
    learned = k["helix"] + k["mlp_relu"]
    print(f"    |eta|>1.5, map: {spj_hi:.1f} steps/jump x {k['rkn_step_map']:.1f} = "
          f"{spj_hi*k['rkn_step_map']:.0f} ns  -> {spj_hi*k['rkn_step_map']/learned:.2f}x "
          f"vs helix+g_theta(relu)")

    np.savez("profile_kernels.npz", **{a: b for a, b in k.items()})
    print("\nwrote profile_stepper.parquet, profile_kernels.npz")


if __name__ == "__main__":
    main()
