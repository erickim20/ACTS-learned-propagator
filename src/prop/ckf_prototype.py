"""v1 prototype: track following on real events, with real hit competition.

Every number in this project so far is a proxy. `chain_sim.py` and
`closed_loop.py` chain the transport unit, but against a single muon's own
hits -- there is nothing else on the layer to pick up by mistake, so "fakes"
had to be estimated from an occupancy rather than counted. This runs the same
loop over a real event, where the layer the filter lands on carries every
other particle's hits too, and counts what it actually picks.

What it is:

  seed        three hits of a true particle, circle-fit in the transverse
              plane and line-fit in (s, z). No truth momentum is used, so the
              seed carries a realistic error.
  follow      s_theta names the next layer, helix + g_theta predicts the
              position on it, C = F C F^T + Q with the helix Jacobian, gate on
              chi2 < 15 against EVERY hit on that layer in the event, take the
              best, Kalman update.
  score       against the truth labels the sample carries: a picked hit is
              right if its particle_id is the seed's.

What it is not, and these bound the claim:

  * no branching. One candidate is followed greedily by best chi2, so this
    measures track FOLLOWING, not the combinatorial part of a CKF.
  * seeding is truth-assisted -- the three hits are known to belong together.
    Seeding efficiency is therefore not measured, only track following given
    a correct seed.
  * the sample is the local `higgs_portal` run at pileup 10, B = 2 T. The
    field matches the teacher (checked by circle fit: 2.00 T). The pT spectrum
    does not: median 0.28 GeV here against a teacher sampled log-uniform over
    0.5-20, so most of these tracks are below anything g_theta was trained on.
    That is a generalisation test as much as an integration one.

Usage:
    python ckf_prototype.py --tracks 300
    python ckf_prototype.py --tracks 300 --topk 2 --no-gtheta   # helix only
"""
import argparse
import collections

import numpy as np
import polars as pl

from .chi2_gate import CHI2_CUT, CLASS_OF, resolutions
from .chain_sim import cells, filter_Q, loc1_for_1d
from .closed_loop import Model, predict, predict_plane, sigma_lookup
from .helix_variants import FieldMap
from .jacobian_helix import (BARREL_VOLUMES, F_bound, F_bound_plane, bound_out,
                            frame, helix_state, solve_s)
from .make_teacher_pairs import plane_predict
from .qtable import correlations, robust_sigma
from .train_gtheta import ACTS

UM = 1e-3
MM = 1e-3


# --------------------------------------------------------------------- data

def load_event_hits(path):
    """Per (event, volume, layer): hit positions, particle ids, and the
    surface's own coordinate. Barrels are pinned by r, discs by z.

    Grouped by a sort rather than a loop over rows. The ColliderML sample this
    was written against is pileup 10; the ttbar baseline is pileup 200 and 24
    events, which is 5.5 million rows, and a per-row loop does not finish.
    """
    d = pl.read_parquet(path)
    lc = [c for c, t in zip(d.columns, d.dtypes) if t == pl.List]
    h = d.explode(lc) if lc else d
    ev = h["event_id"].to_numpy().astype(np.int64)
    vol = h["volume_id"].to_numpy().astype(int)
    lay = h["layer_id"].to_numpy().astype(int)
    xyz = np.column_stack([h[c].to_numpy() for c in ("x", "y", "z")]).astype(float)
    txyz = np.column_stack([h[c].to_numpy()
                            for c in ("true_x", "true_y", "true_z")]).astype(float)
    pid = h["particle_id"].to_numpy()

    keep = np.isin(vol, list(CLASS_OF))
    ev, vol, lay = ev[keep], vol[keep], lay[keep]
    xyz, txyz, pid = xyz[keep], txyz[keep], pid[keep]

    order = np.lexsort((lay, vol, ev))
    ev, vol, lay = ev[order], vol[order], lay[order]
    xyz, txyz, pid = xyz[order], txyz[order], pid[order]

    cut = np.flatnonzero((np.diff(ev) != 0) | (np.diff(vol) != 0)
                         | (np.diff(lay) != 0)) + 1
    starts = np.concatenate([[0], cut])
    stops = np.concatenate([cut, [len(ev)]])

    out = {}
    for a, b in zip(starts, stops):
        out[(int(ev[a]), int(vol[a]), int(lay[a]))] = (
            xyz[a:b], pid[a:b], txyz[a:b])

    # the surface each (volume, layer) is: r for a barrel, z for a disc
    surf = {}
    for (e, v, la), (p, _, _t) in out.items():
        key = (v, la)
        if key in surf:
            continue
        r = np.hypot(p[:, 0], p[:, 1])
        surf[key] = (float(np.median(r)), float(np.median(p[:, 2])),
                     v not in BARREL_VOLUMES)
    return out, surf, len(np.unique(ev))


def true_tracks(hits, min_hits):
    """(event, particle) -> its hits, ordered outward by |position|.

    The `min_hits` cut is applied to the group sizes before any tuple is built.
    At pileup 200 most of the 5.5 million hits belong to particles that leave
    fewer than six, and materialising those first is what makes this the
    expensive step.
    """
    keys, cells = list(hits), list(hits.values())
    n = np.array([len(c[1]) for c in cells])
    ev = np.repeat(np.array([k[0] for k in keys], dtype=np.int64), n)
    vol = np.repeat(np.array([k[1] for k in keys], dtype=np.int32), n)
    lay = np.repeat(np.array([k[2] for k in keys], dtype=np.int32), n)
    xyz = np.concatenate([c[0] for c in cells])
    pid = np.concatenate([c[1] for c in cells])
    txyz = np.concatenate([c[2] for c in cells])

    rad = np.linalg.norm(xyz, axis=1)
    order = np.lexsort((rad, pid, ev))
    ev, vol, lay = ev[order], vol[order], lay[order]
    xyz, pid, txyz = xyz[order], pid[order], txyz[order]

    cut = np.flatnonzero((np.diff(ev) != 0) | (np.diff(pid) != 0)) + 1
    starts = np.concatenate([[0], cut])
    stops = np.concatenate([cut, [len(ev)]])
    long = (stops - starts) >= min_hits

    out = {}
    for a, b in zip(starts[long], stops[long]):
        out[(int(ev[a]), int(pid[a]))] = [
            (xyz[i], int(vol[i]), int(lay[i]), txyz[i]) for i in range(a, b)]
    return out


# ------------------------------------------------------------------ modules

def load_layer_modules(path):
    """`geom_dump`'s table, grouped by the (volume, layer) `s_theta` names.

    Also returns each layer's nominal surface, taken from the module centres
    rather than from the event's hits. It is only a pre-filter, but taking it
    from geometry keeps the plane arm from reading the sample it is measuring.
    """
    m = pl.read_csv(path)
    out, nominal = {}, {}
    for (v, la), g in m.group_by(["volume", "layer"]):
        v, la = int(v), int(la)
        cen = np.column_stack([g[c].to_numpy() for c in ("cx", "cy", "cz")])
        out[(v, la)] = {
            "cen": cen,
            "nrm": np.column_stack([g[c].to_numpy() for c in ("nx", "ny", "nz")]),
            "u0": np.column_stack([g[c].to_numpy() for c in ("u0x", "u0y", "u0z")]),
            "u1": np.column_stack([g[c].to_numpy() for c in ("u1x", "u1y", "u1z")]),
            "hx": g["hx"].to_numpy(),
            "hy": g["hy"].to_numpy(),
        }
        nominal[(v, la)] = (float(np.median(np.hypot(cen[:, 0], cen[:, 1]))),
                            float(np.median(cen[:, 2])),
                            v not in BARREL_VOLUMES)
    return out, nominal


def module_at(cell, nom, u, q, bz, pre_mm=200.0):
    """Which module of a layer the helix crosses, from the state alone.

    Two steps, because solving every module in a layer is wasteful when the
    biggest layer holds 3,360 of them. First the helix onto the layer's
    nominal cylinder or disc, which costs one solve and puts the answer within
    a module of the truth. Then the real plane solve against the modules whose
    centre is within `pre_mm` of that point.

    The bounds test uses the enclosing rectangle. `hx` for a trapezoid is the
    WIDER of its two half-widths (`geom_dump.cpp:140`), so an endcap crossing
    near the narrow end can fall inside two neighbours' boxes. That is what
    the score is for: the module the track is most interior to wins.

    Returns (index into the layer, how far outside the bounds it landed) or
    None. A score above 1 means the crossing is off the module, which is a
    real thing for a track through a gap between two of them.
    """
    s0, ok0 = solve_s(u, np.array([nom[0]]), np.array([nom[1]]),
                      np.array([nom[2]]), q, bz)
    if not ok0[0] or not np.isfinite(s0[0]):
        return None
    p0 = helix_state(u, s0, q, bz)[0, :3]

    near = np.flatnonzero(np.sum((cell["cen"] - p0) ** 2, axis=1) < pre_mm ** 2)
    if near.size == 0:
        return None

    cen, nrm = cell["cen"][near], cell["nrm"][near]
    n = len(near)
    uu = np.repeat(u, n, axis=0)
    px, py, pz, *_rest, s, ok = plane_predict(
        uu[:, 0], uu[:, 1], uu[:, 2], uu[:, 3], uu[:, 4], uu[:, 5],
        np.repeat(q, n), bz, cen, nrm)
    p = np.column_stack([px, py, pz]) - cen
    loc0 = (p * cell["u0"][near]).sum(1)
    loc1 = (p * cell["u1"][near]).sum(1)
    score = np.maximum(np.abs(loc0) / cell["hx"][near],
                       np.abs(loc1) / cell["hy"][near])
    score = np.where(ok, score, np.inf)
    j = int(np.argmin(score))
    if not np.isfinite(score[j]):
        return None
    return int(near[j]), float(score[j])


# --------------------------------------------------------------------- seed

def seed_from_three(p0, p1, p2, bz):
    """Circle in (x, y) and a line in (s, z) through three hits.

    Returns (position, momentum) at the third hit, or None if the fit is
    degenerate. No truth momentum enters, so the seed error is the real one.
    """
    x = np.array([p0[0], p1[0], p2[0]])
    y = np.array([p0[1], p1[1], p2[1]])
    A = np.column_stack([x, y, np.ones(3)])
    b = x ** 2 + y ** 2
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    cx, cy = c[0] / 2, c[1] / 2
    R2 = c[2] + cx ** 2 + cy ** 2
    if R2 <= 0:
        return None
    R = np.sqrt(R2)
    if not np.isfinite(R) or R < 20.0 or R > 1e6:
        return None

    # turning direction fixes the charge
    cross = ((x[1] - x[0]) * (y[2] - y[1]) - (y[1] - y[0]) * (x[2] - x[1]))
    q = -np.sign(cross) * np.sign(bz)
    if q == 0:
        return None

    pt = 0.3 * abs(bz) * R * MM
    # tangent at the third hit, pointing outward along the track
    rad = np.array([x[2] - cx, y[2] - cy])
    tan = np.array([-rad[1], rad[0]]) * (-q * np.sign(bz))
    n = np.linalg.norm(tan)
    if n == 0:
        return None
    tan /= n
    if np.dot(tan, np.array([x[2] - x[1], y[2] - y[1]])) < 0:
        tan = -tan

    # arc length between hits, for the (s, z) slope
    def arc(pa, pb):
        va = np.array([pa[0] - cx, pa[1] - cy])
        vb = np.array([pb[0] - cx, pb[1] - cy])
        d = np.clip(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)),
                    -1, 1)
        return R * np.arccos(d)
    s = np.array([0.0, arc(p0, p1), arc(p0, p1) + arc(p1, p2)])
    z = np.array([p0[2], p1[2], p2[2]])
    if s[2] <= 0:
        return None
    slope = np.polyfit(s, z, 1)[0]          # dz/ds_transverse

    mom = np.array([pt * tan[0], pt * tan[1], pt * slope])
    return np.array(p2, dtype=float), mom, q


# ------------------------------------------------------------------ s_theta

class STheta:
    def __init__(self, path, fm):
        d = np.load(path, allow_pickle=False)
        self.W = [d[f"W{i}"] for i in range(3)]
        self.b = [d[f"b{i}"] for i in range(3)]
        self.mu, self.sd = d["mu"], d["sd"]
        self.cls, self.src_vals = d["cls"], d["src_vals"]
        self.act = ACTS[str(d["act"][0])]
        self.fm = fm

    def topk(self, u, q, vol, lay, k):
        lid = vol * 0x1000 + lay
        i = int(np.searchsorted(self.src_vals, lid))
        x, y, z = u[:3]
        px, py, pz = u[3:]
        p = np.linalg.norm(u[3:])
        phi = np.arctan2(y, x)
        c, s = np.cos(phi), np.sin(phi)
        cont = np.array([[np.hypot(x, y), z, px * c + py * s,
                          -px * s + py * c, pz, q / p, np.hypot(px, py),
                          np.arctanh(np.clip(pz / p, -0.999999, 0.999999)),
                          self.fm.bz(np.array([x]), np.array([y]),
                                     np.array([z]))[0]]])
        oh = np.zeros((1, len(self.src_vals)))
        if 0 <= i < len(self.src_vals) and self.src_vals[i] == lid:
            oh[0, i] = 1.0
        X = (np.hstack([cont, oh]) - self.mu) / self.sd
        h1, _ = self.act(X @ self.W[0] + self.b[0])
        h2, _ = self.act(h1 @ self.W[1] + self.b[1])
        lg = (h2 @ self.W[2] + self.b[2])[0]
        return [int(self.cls[j]) for j in np.argsort(-lg)[:k]]


# ------------------------------------------------------------------- follow

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits", default="sim_out/runs/ch2/tracker_hits/"
                                      "tracker_hits_000000-001000.parquet")
    ap.add_argument("--model", default="gtheta_mat.npz")
    ap.add_argument("--stheta", default="stheta.npz")
    ap.add_argument("--qtable", default="Q_table.parquet")
    ap.add_argument("--dump", default="runs/mat_v2.npz",
                    help="residual dump from train_gtheta.py --dump, for the "
                         "Q correlations. Must match --model")
    ap.add_argument("--pairs", default="teacher_map_mat_logpt.parquet")
    ap.add_argument("--field", default="/tmp/oddb.npz")
    ap.add_argument("--bz", type=float, default=2.0)
    ap.add_argument("--tracks", type=int, default=300)
    ap.add_argument("--min-hits", type=int, default=6)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--max-holes", type=int, default=3)
    ap.add_argument("--max-layers", type=int, default=12)
    ap.add_argument("--c0", type=float, nargs=5,
                    default=[0.30, 0.60, 0.006, 0.006, 0.05],
                    help="seed covariance as five sigmas (mm, mm, rad, rad, "
                         "1/GeV). A three-hit fit is wrong by ~250 um in "
                         "loc0, so declaring C = 0 shuts the gate on the first "
                         "layer -- the same mistake section 9.4 of the chained "
                         "report already recorded.")
    ap.add_argument("--particles",
                    default="sim_out/runs/ch2/particles/"
                            "particles_000000-001000.parquet",
                    help="truth table, used ONLY to select the population -- "
                         "never inside the loop. Selecting on truth is honest "
                         "for a diagnostic that asks 'does it work where the "
                         "teacher lived', and dishonest for anything else.")
    ap.add_argument("--truth-pt-min", type=float, default=0.0)
    ap.add_argument("--truth-pt-max", type=float, default=1e9)
    ap.add_argument("--primary-only", action="store_true")
    ap.add_argument("--no-electrons", action="store_true",
                    help="the teacher dropped electrons -- brems spirals")
    ap.add_argument("--true-seed", action="store_true",
                    help="DIAGNOSTIC: build the seed from the unsmeared hit "
                         "positions. A three-hit fit over a short lever arm is "
                         "noise-dominated, and this says how much of the "
                         "inefficiency is the seed rather than the transport.")
    ap.add_argument("--seed-span", type=int, default=2,
                    help="fit the seed over hits 0..N instead of 0..2. A wider "
                         "lever arm is a better momentum estimate, so sweeping "
                         "this separates seed error from population mismatch")
    ap.add_argument("--pdg", type=int, default=None,
                    help="keep only this |pdg|. 13 is the muon the teacher was "
                         "made of; hadrons carry nuclear interactions no muon "
                         "sample can contain.")
    ap.add_argument("--min-pt", type=float, default=0.0,
                    help="cut on the SEED's fitted pT, not on truth. The "
                         "teacher was sampled log-uniform over 0.5-20 GeV and "
                         "this sample's median is 0.39, so most tracks here "
                         "are outside anything g_theta was trained on.")
    ap.add_argument("--target", choices=("cylinder", "plane"),
                    default="cylinder",
                    help="what the step aims at. `cylinder` is the layer's "
                         "median radius or z, which is what this file was "
                         "written against, and it is the train/deploy gap: "
                         "0.81 mm in the barrel, 5.00 mm on "
                         "the discs. `plane` aims at the module the state "
                         "says it will cross, which is the surface the round10 "
                         "model was trained on and needs a 14-input model")
    ap.add_argument("--modules", default="cpp/modules.csv",
                    help="geom_dump output, for --target plane")
    ap.add_argument("--no-gtheta", action="store_true",
                    help="helix only, so the learned correction can be priced "
                         "against its own baseline on the same events")
    ap.add_argument("--seed", type=int, default=20260806)
    args = ap.parse_args()

    fm = FieldMap(args.field)
    model = Model(args.model)
    st = STheta(args.stheta, fm)
    siglook = sigma_lookup(model.cov, model.digi)
    res = resolutions(model.digi)

    layer_mods, nominal = (None, None)
    if args.target == "plane":
        layer_mods, nominal = load_layer_modules(args.modules)
        print(f"module geometry: {sum(len(c['hx']) for c in layer_mods.values()):,}"
              f" modules over {len(layer_mods)} (volume, layer) cells")

    hits, surf, nev = load_event_hits(args.hits)
    tracks = true_tracks(hits, args.min_hits)
    print(f"{nev} events, {len(hits):,} (event, layer) cells, "
          f"{len(tracks):,} particles with >= {args.min_hits} hits")

    tab = pl.read_parquet(args.qtable)
    t = pl.read_parquet(args.pairs)
    # The residual dump the Q correlations are measured from must come from
    # the same model as --model, which
    # silently pairs a v2 network with v1's process noise -- and the whole
    # teacher-v2 experiment is a comparison between exactly those two, so the
    # one thing it cannot afford is to mix them.
    d = np.load(args.dump)
    D = dict(rem0=d["rem0"], rem1=d["rem1"], one_d=d["one_d"],
             dphi=d["v1_rem"][:, 0], dtheta=d["v1_rem"][:, 1],
             dqop=d["v1_rem"][:, 2],
             cls=np.array([CLASS_OF[int(v)] for v in d["vol"]]))
    _, corr = correlations(D)
    Qf = filter_Q(tab, corr, loc1_for_1d(t))

    # --- select the population, from truth, before anything runs
    if (args.truth_pt_min > 0 or args.truth_pt_max < 1e8
            or args.primary_only or args.no_electrons
            or args.pdg is not None):
        pt_tab = pl.read_parquet(args.particles)
        lp = [c for c, ty in zip(pt_tab.columns, pt_tab.dtypes)
              if ty == pl.List]
        pt_tab = pt_tab.explode(lp)
        sel = set()
        for e, i, px, py, pdg, prim, ch in zip(
                pt_tab["event_id"], pt_tab["particle_id"], pt_tab["px"],
                pt_tab["py"], pt_tab["pdg_id"], pt_tab["primary"],
                pt_tab["charge"]):
            if ch == 0:
                continue
            if args.primary_only and not prim:
                continue
            if args.no_electrons and abs(int(pdg)) == 11:
                continue
            if args.pdg is not None and abs(int(pdg)) != args.pdg:
                continue
            q = float(np.hypot(px, py))
            if args.truth_pt_min <= q <= args.truth_pt_max:
                sel.add((int(e), int(i)))
        before = len(tracks)
        tracks = {k: v for k, v in tracks.items() if k in sel}
        print(f"  population cut: {before:,} -> {len(tracks):,} particles "
              f"(pT {args.truth_pt_min}-{args.truth_pt_max} GeV"
              f"{', primary' if args.primary_only else ''}"
              f"{', no electrons' if args.no_electrons else ''})")

    rng = np.random.default_rng(args.seed)
    keys = sorted(tracks)
    if args.tracks < len(keys):
        keys = [keys[i] for i in
                rng.choice(len(keys), args.tracks, replace=False)]

    n_seed, n_true, n_found, n_wrong, n_hole = 0, 0, 0, 0, 0
    n_empty, n_gated, n_wrongl, n_nomod = 0, 0, 0, 0
    per_track, chi2s, m_scores = [], [], []

    for (ev, pid) in keys:
        lst = tracks[(ev, pid)]
        j = 3 if args.true_seed else 0
        # Lever arm. Three ADJACENT hits span a few cm, and a circle fit over
        # that is noise-dominated for a soft track -- which matters here beyond
        # the seed itself, because g_theta is a correction learned AT THE TRUE
        # STATE and applied to whatever state the filter is actually in. If the
        # deficit shrinks as this span grows, the failure is state error rather
        # than the population the model was trained on.
        sp = min(args.seed_span, len(lst) - 1)
        i0, i1, i2 = 0, sp // 2, sp
        if i1 == i0 or i2 == i1:
            continue
        s = seed_from_three(lst[i0][j], lst[i1][j], lst[i2][j], args.bz)
        if s is None:
            continue
        pos, mom, q = s
        if np.hypot(mom[0], mom[1]) < args.min_pt:
            continue
        n_seed += 1
        u = np.concatenate([pos, mom])[None, :]
        C = np.diag(np.array(args.c0)) ** 2
        # the follow starts where the seed ends, which is hit i2, not hit 2
        vol, lay = lst[i2][1], lst[i2][2]
        # The seed covariance is a declared diagonal, so the frame it is
        # written in is a choice rather than a measurement. It is taken as the
        # cylinder frame at the seed on both targets, which keeps the two arms
        # starting from the same number; from the first update on, the plane
        # arm carries the module it actually landed on.
        e0s, e1s, _ = frame(u[:, :3], u[:, 3:],
                            np.array([vol not in BARREL_VOLUMES]))
        src_axes = (e0s, e1s)
        true_layers = collections.Counter((int(h[1]), int(h[2]))
                                          for h in lst[i2 + 1:])
        true_rest = len(lst) - (i2 + 1)
        n_true += true_rest

        got, bad, holes = 0, 0, 0
        for _ in range(args.max_layers):
            if holes > args.max_holes:
                break
            cand = st.topk(u[0], q, vol, lay, args.topk)
            picked = None
            best = CHI2_CUT
            any_hits = False
            on_a_true_layer = any((lid >> 12, lid & 0xFFF) in true_layers
                                  for lid in cand)
            for lid in cand:
                v2, l2 = lid >> 12, lid & 0xFFF
                cell = hits.get((ev, int(v2), int(l2)))
                if cell is None or (int(v2), int(l2)) not in surf:
                    continue
                any_hits = True
                pt_e = float(np.hypot(u[0, 3], u[0, 4]))
                pmag = float(np.linalg.norm(u[0, 3:]))
                eta_e = abs(np.arctanh(np.clip(u[0, 5] / pmag, -0.999999,
                                               0.999999)))
                ip, ie = cells(np.array([pt_e]), np.array([eta_e]))
                s0, s1, one_d, v0, v1 = siglook(v2, ip[0], ie[0])
                sig01 = np.array([[s0, s1]])

                if layer_mods is not None:
                    mk = (int(v2), int(l2))
                    mcell = layer_mods.get(mk)
                    if mcell is None:
                        continue
                    hit_mod = module_at(mcell, nominal[mk], u,
                                        np.array([q]), args.bz)
                    if hit_mod is None:
                        n_nomod += 1
                        continue
                    mi, mscore = hit_mod
                    m_scores.append(mscore)
                    xp, sarc, ok, hp, e0, e1, _, _ = predict_plane(
                        model, u, np.array([q]),
                        mcell["cen"][mi:mi + 1], mcell["nrm"][mi:mi + 1],
                        mcell["u0"][mi:mi + 1], mcell["u1"][mi:mi + 1],
                        fm, sig01)
                    if args.no_gtheta:
                        xp = np.concatenate([hp, xp[:, 3:]], axis=1)
                    if not ok[0] or not np.all(np.isfinite(xp)):
                        continue
                    F = F_bound_plane(u, sarc, np.array([q]), args.bz,
                                      mcell["nrm"][mi:mi + 1], e0, e1,
                                      src_axes)[0][0]
                else:
                    r1, z1, ec = surf[(int(v2), int(l2))]
                    xp, sarc, ok, hp, e0, e1, _, _ = predict(
                        model, u, np.array([q]), np.array([r1]),
                        np.array([z1]), np.array([ec]), fm, sig01)
                    if args.no_gtheta:
                        xp = np.concatenate([hp, xp[:, 3:]], axis=1)
                    if not ok[0] or not np.all(np.isfinite(xp)):
                        continue
                    F = F_bound(u, np.array([r1]), np.array([z1]),
                                np.array([ec]), np.array([q]), args.bz,
                                s=sarc)[0][0]
                key = (CLASS_OF[int(v2)], int(ip[0]), int(ie[0]))
                Q = Qf.get(key)
                if Q is None:
                    Q = Qf[(CLASS_OF[int(v2)], -1, -1)]
                Cp = F @ C @ F.T + Q
                V = np.diag([(v0 * UM) ** 2]) if one_d else \
                    np.diag([(v0 * UM) ** 2, (v1 * UM) ** 2])
                nd = 1 if one_d else 2
                H = np.zeros((nd, 5))
                for j in range(nd):
                    H[j, j] = 1.0
                S = H @ Cp @ H.T + V
                Si = np.linalg.inv(S)
                dv = cell[0] - xp[0, :3]
                r = np.column_stack([dv @ e0[0], dv @ e1[0]])[:, :nd]
                chi2 = np.einsum("ij,jk,ik->i", r, Si, r)
                j = int(np.argmin(chi2))
                if chi2[j] < best:
                    best = float(chi2[j])
                    picked = (v2, l2, j, cell, Cp, H, Si, r[j], e0, e1, xp)
            if picked is None:
                holes += 1
                n_hole += 1
                if not any_hits:
                    n_empty += 1        # s_theta named a layer with no hits
                elif not on_a_true_layer:
                    n_wrongl += 1       # hits, but not this particle's layer
                else:
                    n_gated += 1        # the hit was there and the gate refused
                continue
            v2, l2, j, cell, Cp, H, Si, rj, e0, e1, xp = picked
            chi2s.append(best)
            if cell[1][j] == pid:
                got += 1
            else:
                bad += 1
            K = Cp @ H.T @ Si
            dx = K @ rj
            newpos = xp[0, :3] + dx[0] * e0[0] + (
                dx[1] * e1[0] if len(dx) > 1 and H.shape[0] > 1 else 0.0)
            pm = np.linalg.norm(xp[0, 3:])
            phi_d = np.arctan2(xp[0, 4], xp[0, 3]) + dx[2]
            th_d = np.arctan2(np.hypot(xp[0, 3], xp[0, 4]), xp[0, 5]) + dx[3]
            qop = q / pm + dx[4]
            pm2 = abs(1.0 / qop) if abs(qop) > 1e-9 else pm
            u = np.array([[newpos[0], newpos[1], newpos[2],
                           pm2 * np.sin(th_d) * np.cos(phi_d),
                           pm2 * np.sin(th_d) * np.sin(phi_d),
                           pm2 * np.cos(th_d)]])
            C = Cp - K @ H @ Cp
            vol, lay = int(v2), int(l2)
            # the updated covariance is written in the surface just used, so
            # the next step's J_in has to be built on that surface and not on
            # whatever the cylinder frame would be at the new position
            if layer_mods is not None:
                src_axes = (e0, e1)

        n_found += got
        n_wrong += bad
        if got + bad:
            per_track.append((got, bad, true_rest))

    print(f"\n=== track following, {n_seed:,} seeded tracks "
          f"({'helix only' if args.no_gtheta else 'helix + g_theta'}, "
          f"s_theta top-{args.topk}) ===")
    picked = n_found + n_wrong
    print(f"  true hits available (after the 3-hit seed) : {n_true:,}")
    print(f"  hits picked                                : {picked:,}")
    print(f"  of those, CORRECT                          : {n_found:,}"
          f"   ({100*n_found/max(picked,1):.1f}%)")
    print(f"  of those, from another particle (FAKE)     : {n_wrong:,}"
          f"   ({100*n_wrong/max(picked,1):.1f}%)")
    print(f"  hit efficiency  (correct / available)      : "
          f"{100*n_found/max(n_true,1):.1f}%")
    print(f"  layers with no hit taken                   : {n_hole:,}")
    print(f"      s_theta named a layer with no hits     : {n_empty:,}"
          f"   ({100*n_empty/max(n_hole,1):.0f}%)")
    print(f"      s_theta named a layer this particle    : {n_wrongl:,}"
          f"   ({100*n_wrongl/max(n_hole,1):.0f}%)")
    print(f"        never crossed (hits were other tracks')")
    print(f"      our own hit was there, gate refused it : {n_gated:,}"
          f"   ({100*n_gated/max(n_hole,1):.0f}%)")
    if chi2s:
        print(f"  chi2 of accepted hits: median "
              f"{np.median(chi2s):.2f}, p90 {np.percentile(chi2s, 90):.2f}")
    if m_scores:
        ms = np.array(m_scores)
        print(f"  module solves                              : {len(ms):,}"
              f"   ({n_nomod:,} layers with no module reached)")
        print(f"      crossing inside the module's bounds    : "
              f"{100*(ms <= 1).mean():.1f}%   (score median "
              f"{np.median(ms):.2f}, p90 {np.percentile(ms, 90):.2f})")

    if per_track:
        a = np.array(per_track, dtype=float)
        pure = (a[:, 1] == 0).mean()
        frac = a[:, 0] / np.maximum(a[:, 2], 1)
        print(f"\n  per track: {100*pure:.1f}% picked no wrong hit at all;"
              f"  median {100*np.median(frac):.0f}% of its true hits found")

    print("\n  Greedy, single-candidate following with a truth-assisted seed. "
          "Not a full CKF:\n  no branching and no seeding efficiency. The pT "
          "spectrum here is far softer\n  than the teacher's, so this is a "
          "generalisation test as much as an integration one.")


if __name__ == "__main__":
    main()
