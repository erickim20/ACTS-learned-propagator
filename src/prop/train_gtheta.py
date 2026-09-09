"""Train g_theta and score it with the gate the CKF actually applies.

This does not report a loss. It reports points of hit efficiency, the same
number the helix baseline is quoted in.

Four choices here are forced by earlier measurements rather than picked:

  split by track   jumps from one track share surfaces and states, so a
                   per-jump split leaks. That is what turned the s_theta
                   module accuracy from 33% (held out) into 59% (in-sample).

  no surface IDs   28,478 distinct (source, destination) pairs with the top
                   3,462 covering only half the jumps. Anything keyed on
                   surface identity memorises the detector.

  loss on b/sigma  not on b in micrometres. (b/sigma)^2 is the bias's
                   contribution to the chi2 the gate tests, so this is the
                   right variable. It also handles pixels being 5x tighter
                   than long strips, and sigma_C growing towards low pT, with
                   no hand-tuned tail weighting.

  Huber, not MSE   the gate saturates and MSE does not. Plain MSE on b/sigma
                   put 81% of the gradient on the 1.7% of jumps above
                   b/sigma = 20, which are lost whatever the model does,
                   and the first run duly made things worse (-630%). Huber
                   with delta = sqrt(15), the gate boundary itself, keeps the
                   quadratic region exactly where the decision is still in
                   play.

  12-64-64-6       the size that measured 456 ns and 6.0x. Growing it
                   invalidates the speed claim, so fit at this size first and
                   re-measure if it underfits.

numpy rather than a framework: the model is ~5k parameters, and the forward
pass here is the same arithmetic the benchmarked C++ kernel runs. Backprop is
checked against finite differences at startup rather than trusted.

Usage:
    python train_gtheta.py teacher_map_nomat_logpt.parquet
    python train_gtheta.py teacher_map_nomat_logpt.parquet --act ptanh
    python train_gtheta.py teacher_map_nomat_logpt.parquet --model pure \
        --lr 3e-4 --epochs 2000          # the ablation, at its best settings

The ablation answered its question: the physics core earns its place. A pure
MLP on the same inputs minus the helix loses 37-40 points of hit efficiency
against the helix's own 9.9 and the residual model's 0.2 -- i.e. four times
worse than not learning anything. It is not a capacity limit (64 -> 256 -> 512
gives 37.9 -> 36.7 -> 38.0) and not an optimiser one (swept). It is the range:
the destination is ~2 cm away and the gate wants ~100 um, so a pure model has
to be right to 1 part in 200 everywhere, while the residual model starts from
a helix that is already right to 8.6 um in the median.
"""
import argparse

import numpy as np
import polars as pl

from .chi2_gate import (CLASS_OF, ETA_EDGES, PT_EDGES, gate, in_plane_residual,
                        measured_cov, resolutions)
from .helix_variants import field_source

# `qtable.CLASSES` order, which is `LearnedNoise.hpp`'s `ModuleClass` and the
# order the Q table and the cell sigma table are both written in. The v1 scale
# table is indexed by the same C++ code, so it must not disagree.
V1_CLASSES = ("pixel", "sstrip", "lstrip")


def v1_cell_scale(v1, tr_i, vol, pt, abs_eta):
    """The spread of the (phi, theta, q/p) target per (class, pT, |eta|) cell.

    `v1[tr_i].std(0)` is three numbers for the whole detector and the whole
    momentum range, and that is the measured defect. The
    target falls as 1/pT and a constant does not, so the correction stayed the
    size of a soft track's while the error it corrected shrank, and in the
    barrel above 20 GeV, where the helix is already right, it cost 12.68
    efficiency points.

    Three separate reasons a single constant cannot be right, and only the
    first is about pT:

      * phi and theta are the helix's constant-field approximation failing, and
        that scales as 1/pT.
      * q/p is not an approximation error at all. The helix conserves |p|
        exactly, so the residual is the energy loss, and dq/p goes as 1/p^2.
      * both depend on where in the detector the jump is, because the field is
        within 5 % of 2 T in the barrel and runs 0.532 to 1.970 T in the
        endcap.

    So the scale is fitted per cell, on the same binning `measured_cov` and the
    Q table use, and the network sees the same normalised target everywhere
    instead of one that is 20x smaller in the bins it is asked to leave alone.
    That is also what puts every jump on an equal footing in the loss: dividing
    the whole fold by one number made the high-pT jumps contribute ~400x less
    to a squared loss, so the model had almost no incentive to get them right.

    A cell the training fold did not populate falls back to the class median
    and then to the global spread, which is what `measured_cov` does with its
    pt_bin = eta_bin = -1 row. Interpolating between cells would be inventing a
    measurement. Cells are reported by the caller so a fit standing on mostly
    fallbacks is visible rather than silent.
    """
    nC, nP, nE = len(V1_CLASSES), len(PT_EDGES) - 1, len(ETA_EDGES) - 1
    glob = v1[tr_i].std(0)
    glob[glob == 0] = 1.0                         # q/p with material off

    cls = np.array([V1_CLASSES.index(CLASS_OF[int(v)]) for v in vol])
    ip = np.clip(np.digitize(pt, PT_EDGES) - 1, 0, nP - 1)
    ie = np.clip(np.digitize(abs_eta, ETA_EDGES) - 1, 0, nE - 1)

    table = np.tile(glob, (nC, nP, nE, 1))
    have = np.zeros((nC, nP, nE), dtype=bool)
    # At least this many training jumps before a cell's own spread is used. A
    # std over a handful of jumps is noise, and a scale that is too small turns
    # into an output the model has to make large, which is the failure mode
    # this whole function exists to remove.
    kMin = 50
    for c in range(nC):
        inC = cls[tr_i] == c
        med = v1[tr_i][inC].std(0) if inC.sum() >= kMin else glob
        med[med == 0] = 1.0
        for i in range(nP):
            for j in range(nE):
                m = inC & (ip[tr_i] == i) & (ie[tr_i] == j)
                if m.sum() >= kMin:
                    s = v1[tr_i][m].std(0)
                    s[s == 0] = 1.0
                    table[c, i, j] = s
                    have[c, i, j] = True
                else:
                    table[c, i, j] = med
    per_jump = table[cls, ip, ie]
    return table, per_jump, have, glob


def v1_at(per_jump, idx):
    """The scale for a subset of jumps, whichever form `per_jump` is in.

    Per cell it is an (n, 3) array and the subset is a row selection. Under
    `--v1-global` it is the one triple every jump shares, and broadcasting is
    what reproduces the old arithmetic exactly.
    """
    a = np.asarray(per_jump)
    return a[idx] if a.ndim == 2 else np.tile(a, (len(idx), 1))


H = 64
N_OUT = 6
DELTA = 15.0 ** 0.5        # Huber knee = the gate boundary in units of sigma          # only the first two are supervised in v0; the rest are
                   # carried so the arithmetic matches the benchmarked kernel


# ---------------------------------------------------------------- activations

def relu(x):
    return np.maximum(x, 0.0), (x > 0).astype(x.dtype)


def ptanh(x):
    """Pade 3/2 approximant to tanh, clamped where it stops saturating.

    tanh(x) ~ x(27+x^2)/(27+9x^2) is accurate to ~1e-3 on |x|<3 and costs a
    divide instead of an exp. Past |x|=3 the approximant's derivative tends to
    1/9 rather than 0, so it is clamped there -- and the clamp is C1, since
    the approximant reaches exactly 1 with derivative exactly 0 at x=3.

    Worth having because a relu Jacobian jumps at cell boundaries, and the
    covariance transport in the integration phase needs F = df/dx.
    """
    c = np.clip(x, -3.0, 3.0)
    x2 = c * c
    y = c * (27.0 + x2) / (27.0 + 9.0 * x2)
    d = (x2 - 9.0) ** 2 / (9.0 * (3.0 + x2) ** 2)
    d = np.where(np.abs(x) >= 3.0, 0.0, d)
    return y, d


ACTS = {"relu": relu, "ptanh": ptanh}


# ---------------------------------------------------------------------- model

class MLP:
    def __init__(self, nin, act, rng, hidden=H, nout=N_OUT):
        self.act = ACTS[act]
        s = [nin, hidden, hidden, nout]
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (a, b))
                  for a, b in zip(s[:-1], s[1:])]
        self.b = [np.zeros(b) for b in s[1:]]

    def forward(self, x):
        h1, d1 = self.act(x @ self.W[0] + self.b[0])
        h2, d2 = self.act(h1 @ self.W[1] + self.b[1])
        return h2 @ self.W[2] + self.b[2], (x, h1, d1, h2, d2)

    def backward(self, cache, dout):
        x, h1, d1, h2, d2 = cache
        gW2 = h2.T @ dout
        gb2 = dout.sum(0)
        g2 = (dout @ self.W[2].T) * d2
        gW1 = h1.T @ g2
        gb1 = g2.sum(0)
        g1 = (g2 @ self.W[1].T) * d1
        gW0 = x.T @ g1
        gb0 = g1.sum(0)
        return [gW0, gW1, gW2], [gb0, gb1, gb2]

    def params(self):
        return self.W + self.b

    def jacobian(self, x, nout=None):
        """dout/dx, exactly, for the covariance transport in integration.

        The filter needs F = df_theta/dx for both the chi2 gate and the Kalman
        gain, and this is the g_theta half of it: chain the three weight
        matrices with the activation derivatives that the forward pass already
        computes. It is why the activation had to be C1 -- a relu Jacobian is
        piecewise constant and jumps at cell boundaries, which puts steps into
        a covariance that is supposed to be smooth in the state.

        Returned as (batch, nout, nin), or a subset of the outputs.

        Association order matters and is not free to choose carelessly. The
        chain is W0 D1 W1 D2 W2 with shapes (12,64)(64,64)(64,6); multiplying
        left to right costs nin*H*H = 49k multiply-adds per jump, right to
        left costs H*H*nout = 25k for nout = 6 and 8k for nout = 2. Since the
        gate only needs the position rows, the cheap end of that range is the
        one that matters. Left to right was the first thing written here and
        it measured 4.8x the forward pass; right to left is 2.6x.
        """
        h1, d1 = self.act(x @ self.W[0] + self.b[0])
        _, d2 = self.act(h1 @ self.W[1] + self.b[1])
        w2 = self.W[2] if nout is None else self.W[2][:, :nout]
        m = d2[:, :, None] * w2[None, :, :]                 # (B, H, nout)
        m = (self.W[1] @ m) * d1[:, :, None]                # (B, H, nout)
        return np.einsum("ih,bho->boi", self.W[0], m)


def huber(pred, targ, w=None, sw=None):
    """Value and dL/dpred, elementwise, mean-reduced.

    w rescales the residual into units of sigma before the knee is applied, so
    that the same loss can be used whatever the output is parameterised in. It
    is 1 for the residual model, which already predicts in units of sigma, and
    (target scale)/sigma for the pure model, which predicts a standardised
    displacement. w = 0 drops a term -- used for the second coordinate of 1D
    modules, which is never measured.

    sw is a SAMPLE weight and is a different quantity. It multiplies the value
    and the gradient AFTER the knee, so how much a jump counts changes and
    where its loss stops being quadratic does not. Putting a sample weight in
    w instead would scale the residual going in, which moves the knee by the
    same factor and breaks the one property DELTA = sqrt(15) is chosen for:
    that the loss is quadratic exactly as far out as the gate's decision is
    still in play. Broadcasts, so a (batch, 1) column weights whole jumps.
    """
    r = pred - targ
    if w is not None:
        r = r * w
    a = np.abs(r)
    quad = a <= DELTA
    val = np.where(quad, 0.5 * r * r, DELTA * (a - 0.5 * DELTA))
    grad = np.where(quad, r, DELTA * np.sign(r))
    if w is not None:
        grad = grad * w
    if sw is not None:
        val = val * sw
        grad = grad * sw
    return val.mean(), grad / r.size


def huber_value(r):
    """The elementwise Huber value on an already-scaled residual.

    Separate from `huber` because the weighting needs the loss a jump carries
    with the model predicting nothing, which is the helix's own residual in
    loss units and is a property of the sample rather than of a fit.
    """
    a = np.abs(r)
    return np.where(a <= DELTA, 0.5 * r * r, DELTA * (a - 0.5 * DELTA))


def mse(pred, targ, w=None, sw=None):
    """Warm-up loss for the pure model, on the standardised target.

    w is used only as a mask here (w = 0 means the coordinate is not
    measured); using it as a weight would put the target's own scale back into
    the loss, which is the thing the warm-up exists to avoid. sw is the sample
    weight and multiplies the squared error, as it does in huber.
    """
    r = pred - targ
    if w is not None:
        r = np.where(w == 0.0, 0.0, r)
    if sw is None:
        return float((r * r).mean()), 2.0 * r / r.size
    return float((sw * r * r).mean()), 2.0 * sw * r / r.size


def cell_weights(vol, pt, abs_eta, tr_i, base_loss, mode="loss", power=1.0,
                 k_min=50):
    """One weight per jump, so that a (module class, pT, |eta|) cell's share
    of the loss stops being set by the size of the residual in it.

    The cells are the ones the Q table, the cell sigma table and the v1 scale
    table already use, so a weight is not a fourth binning to keep in step.

    Two things decide what a cell contributes today: how many jumps it holds,
    and how large a loss each of them carries. The second is the one that
    matters here. A barrel jump 15 um out sits at 0.15 sigma and is deep in
    the quadratic region, so it carries about a twentieth of the loss of a
    transition-region jump at 3 sigma, and the barrel gets whatever gradient
    is left over.

      mode="loss"    weight = 1 / (the cell's total helix loss). Every cell
                     then contributes the same amount, which is what removes
                     the residual-size factor.
      mode="count"   weight = 1 / (the cell's jump count). This equalises
                     POPULATION, which is a different quantity and moves the
                     barrel the wrong way -- the dense cells are the small
                     residuals, so equalising counts takes weight off them.
                     Kept because it is the reading that was measured.

    `base_loss` is the per-jump Huber value with the model predicting nothing,
    i.e. what the helix alone costs. It is a property of the sample, not of a
    fit, so the weights are fixed before the first step and do not chase the
    model.

    Cells below k_min training jumps are left at their natural share rather
    than equalised. Exact equalisation over every occupied cell puts a weight
    of 1,717 on a cell holding one jump -- fifteen of the seventy-five hold
    fewer than fifty -- and roughly a fifth of the gradient would then come
    from about fifteen jumps. That is fitting noise, and k_min = 50 is the
    same threshold `v1_cell_scale` uses above for the same reason.

    Normalised to mean 1 over the training fold, so the loss keeps its scale
    and the learning rate means what it meant before. Returns the per-jump
    weight, the counts and the equalised mask, which the caller reports: a
    weighting standing on mostly fallbacks should be visible rather than
    silent.
    """
    nP, nE = len(PT_EDGES) - 1, len(ETA_EDGES) - 1
    cls = np.array([V1_CLASSES.index(CLASS_OF[int(v)]) for v in vol])
    ip = np.clip(np.digitize(pt, PT_EDGES) - 1, 0, nP - 1)
    ie = np.clip(np.digitize(abs_eta, ETA_EDGES) - 1, 0, nE - 1)
    cell = (cls * nP + ip) * nE + ie

    n = len(V1_CLASSES) * nP * nE
    cnt = np.bincount(cell[tr_i], minlength=n)
    if mode == "count":
        share = cnt.astype(float)
    else:
        share = np.bincount(cell[tr_i], weights=base_loss[tr_i], minlength=n)
    big = (cnt >= k_min) & (share > 0)
    per = np.ones(n)
    if big.any():
        # each equalised cell gets the same total, and that total is the mean
        # the equalised cells hold today, so the un-equalised remainder is not
        # scaled up or down relative to them.
        #
        # `power` tempers it. 0 is the unweighted loss and 1 is full
        # equalisation, and the two ends are measured to be a straight trade:
        # full equalisation reaches the helix in the barrel and gives up the
        # transition region entirely. Anything between says whether that is a
        # wall or a frontier.
        per[big] = (share[big].mean() / share[big]) ** power
    w = per[cell]
    w = w / w[tr_i].mean()
    return w[:, None], cnt, big


def softmax_ce(logits, idx):
    """Cross-entropy and dL/dlogits for s_theta. Mean-reduced."""
    e = np.exp(logits - logits.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    n = len(idx)
    loss = float(-np.log(p[np.arange(n), idx] + 1e-12).mean())
    g = p.copy()
    g[np.arange(n), idx] -= 1.0
    return loss, g / n


def grad_check(model, x, y, w, rng):
    """Finite differences on a few random weights. Hand-written backprop that
    is subtly wrong trains to a plausible-looking loss, which is exactly the
    failure mode this project keeps hitting."""
    def loss_of():
        p, _ = model.forward(x)
        return float(huber(p[:, :y.shape[1]], y, w)[0])

    p, cache = model.forward(x)
    dout = np.zeros_like(p)
    dout[:, :y.shape[1]] = huber(p[:, :y.shape[1]], y, w)[1]
    gW, gb = model.backward(cache, dout)

    worst = 0.0
    for li in range(3):
        for _ in range(4):
            i = rng.integers(model.W[li].shape[0])
            j = rng.integers(model.W[li].shape[1])
            eps = 1e-6
            model.W[li][i, j] += eps
            lp = loss_of()
            model.W[li][i, j] -= 2 * eps
            lm = loss_of()
            model.W[li][i, j] += eps
            num = (lp - lm) / (2 * eps)
            ana = gW[li][i, j]
            denom = max(abs(num), abs(ana), 1e-12)
            worst = max(worst, abs(num - ana) / denom)
    return worst


# --------------------------------------------------------------------- inputs

def v1_targets(t):
    """What the helix gets wrong about direction and q/p at the destination.

    The gate does not test these -- it tests position on the surface -- so
    they cannot be scored in points of hit efficiency directly. They matter
    because they are the launch conditions of the *next* jump, which is why
    chaining needs them. Reported in mrad and in relative q/p, plus the
    position error a given direction error would induce over the median step,
    which is the only way to put them on the same axis as everything else.

    q/p is the interesting one: the helix conserves |p| exactly, because a
    magnetic field does no work. So its q/p residual is not an approximation
    error at all -- it is the energy loss, in full, and it is zero by
    construction when material is off.
    """
    px1, py1, pz1 = (t[c].to_numpy() for c in ("px1", "py1", "pz1"))
    hpx, hpy, hpz = (t[c].to_numpy() for c in ("helix_px", "helix_py",
                                               "helix_pz"))
    dphi = (np.arctan2(py1, px1) - np.arctan2(hpy, hpx) + np.pi) \
        % (2 * np.pi) - np.pi
    theta1 = np.arctan2(np.hypot(px1, py1), pz1)
    theta_h = np.arctan2(np.hypot(hpx, hpy), hpz)
    qop = t["qop"].to_numpy()                    # = q/|p| at the source
    # |p_helix| == |p_source|, so the helix's q/p prediction *is* qop
    qop1 = qop * np.hypot(np.hypot(hpx, hpy), hpz) / np.hypot(
        np.hypot(px1, py1), pz1)
    return dphi, theta1 - theta_h, qop1 - qop


def layer_id(g):
    """volume + layer bits, i.e. the thing s_theta is allowed to predict.

    Not the module: 28,478 distinct (source, destination) module pairs with the
    top 3,462 covering half the jumps, and 33% top-1 held out against 59%
    in-sample. The plan says the module must come from intersecting the
    predicted position with the named layer, which is what the ACTS navigator
    does anyway once it knows the layer.
    """
    return ((g >> 56) & 0xFF) * 0x1000 + ((g >> 36) & 0xFFF)


def state_features(t, fm, n_src, src_idx):
    """Inputs for s_theta: the state, and nothing about the destination.

    g_theta is allowed the destination surface's own coordinate because by the
    time it runs the surface has been named. s_theta is what names it, so it
    gets the state, the field, and a one-hot of the layer it is standing on --
    which is what the navigator knows too.
    """
    x, y, z = (t[c].to_numpy() for c in ("x", "y", "z"))
    px, py, pz = (t[c].to_numpy() for c in ("px", "py", "pz"))
    r0 = t["r0"].to_numpy()
    phi0 = np.arctan2(y, x)
    c, s = np.cos(phi0), np.sin(phi0)
    cont = np.column_stack([
        r0, z,
        px * c + py * s, -px * s + py * c, pz,
        t["qop"].to_numpy(),
        np.hypot(px, py), t["eta"].to_numpy(),
        fm.bz(x, y, z),
    ])
    onehot = np.zeros((len(t), n_src))
    onehot[np.arange(len(t)), src_idx] = 1.0
    return np.hstack([cont, onehot])


def absolute_target(t):
    """The whole displacement to the destination hit, in [um].

    This is what the helix computes and what a pure MLP has to learn instead.
    Resolved in exactly the two in-surface coordinates chi2_gate uses, and
    measured from the source point, so that

        (displacement the helix predicts) - (this) == in_plane_residual(t)

    identically -- which is what makes the two models scoreable against the
    same gate. The range is the point of the ablation: the residual model has
    to cover 524 um of p90, this one has to cover the whole step.
    """
    x, y, z = (t[c].to_numpy() for c in ("x", "y", "z"))
    x1, y1, z1 = (t[c].to_numpy() for c in ("x1", "y1", "z1"))
    r0, r1 = t["r0"].to_numpy(), t["r1"].to_numpy()
    ec = t["endcap"].to_numpy()

    dphi = (np.arctan2(y1, x1) - np.arctan2(y, x) + np.pi) % (2*np.pi) - np.pi
    a_rphi = r1 * dphi * 1000.0
    a_other = np.where(ec, (r1 - r0) * 1000.0, (z1 - z) * 1000.0)
    return a_rphi, a_other


def features(t, fm, pure=False, legacy=False):
    """12 inputs, rotated so the source sits at phi = 0.

    pure=True drops the two helix-derived inputs and returns 10. That is what
    the ablation means: the network keeps the state and the geometry of the
    step, and loses the physics core's answer.

    The field map is symmetric about z, so feeding global x and y would make
    the network learn azimuthal invariance from data instead of being handed
    it. Rotating costs two trig calls and removes a degree of freedom.

    Only one of (r1, z1) may be given, and which one depends on the module.
    A barrel surface is a cylinder: r1 defines it, and z1 is part of where the
    track landed -- i.e. truth. An endcap disc is the other way round. The
    first version of this function passed both and scored 97.8% of the loss
    captured; that number was reading the answer, since the pair (r1, z1)
    pins the arc length and the arc length all but determines the azimuth.

    What is legitimate on top of the surface's own coordinate is the helix
    prediction itself: the model is a correction to it, and it is computable
    at run time. So the helix's arc length and its estimate of the
    *unconstrained* coordinate go in, and the truth version of that
    coordinate stays out.
    """
    x, y, z = (t[c].to_numpy() for c in ("x", "y", "z"))
    px, py, pz = (t[c].to_numpy() for c in ("px", "py", "pz"))
    r0 = t["r0"].to_numpy()
    ec = t["endcap"].to_numpy()

    phi0 = np.arctan2(y, x)
    c, s = np.cos(phi0), np.sin(phi0)

    # `legacy` forces the cylinder inputs on a table that has both column sets,
    # which is the only way to separate two changes that landed together: the
    # target surface moved to the module plane, and input 7 stopped being the
    # true next hit's own coordinate. Run with the plane target and the legacy
    # inputs and the difference is attributable.
    plane = "path_to_plane" in t.columns and not legacy
    if plane:
        # THE MODULE-PLANE TARGET. Inputs 7 to 10 and 12 change with it, and
        # the reason is input 7: the old one was r1 or z1, i.e. the TRUE next
        # hit's own coordinate, so the model was handed a function of the
        # answer and could not be run at inference without one. Every input
        # below is a property of the target module and the source state.
        #
        # 7 and 8 are the invariant pair, not the raw (d_plane, cos_inc).
        # Which face of a module DD4hep calls the front is a convention that
        # varies between modules and flips both, so the raw pair would teach
        # the network per-module conventions -- the same failure the surface
        # identifier was excluded for. See make_teacher_pairs.
        surf = t["path_to_plane"].to_numpy()
        dsurf = t["abs_cos_inc"].to_numpy()
        # 9 and 10: the helix's answer in the module's own local coordinates,
        # which is the frame the target is written in. The old pair was the
        # helix's guess at the unconstrained cylinder coordinate and its
        # azimuthal move, and neither has a meaning on a tilted plane.
        helix_free = t["helix_loc0"].to_numpy()
        dphi_h = t["helix_loc1"].to_numpy()
        # 12: how forward-facing the module is. Continuous, because with a
        # plane target there is no longer a barrel/disc switch to encode, and
        # |n_z| carries the same information without a discontinuity.
        last = np.abs(t["mn_z"].to_numpy())
        # 13 and 14: WHERE loc0 POINTS INSIDE THE PLANE.
        #
        # The target is a vector in each module's own local axes, and those
        # axes rotate from module to module. The cylinder frame did not: e0 was
        # phi_hat everywhere, so one convention covered the whole detector. A
        # network given the module frame's residual but not the module frame's
        # orientation is being asked to predict components of a vector in a
        # basis it cannot see, and it can only learn the average basis.
        #
        # e0 resolved in the cylindrical frame at the module centre. Two
        # numbers fix the in-plane rotation given the normal, and both come
        # from the geometry, so both are knowable at run time.
        cx, cy = t["mc_x"].to_numpy(), t["mc_y"].to_numpy()
        rc = np.hypot(cx, cy)
        rc = np.where(rc > 0, rc, 1.0)
        e0_phi = (-t["mu0_x"].to_numpy() * cy
                  + t["mu0_y"].to_numpy() * cx) / rc
        e0_z = t["mu0_z"].to_numpy()
    else:
        hx, hy, hz = (t[c].to_numpy() for c in ("helix_x", "helix_y",
                                                "helix_z"))
        hr = np.hypot(hx, hy)
        # the coordinate the destination surface fixes, and the step in it
        surf = np.where(ec, t["z1"].to_numpy(), t["r1"].to_numpy())
        dsurf = surf - np.where(ec, z, r0)
        # the helix's guess at the coordinate the surface does *not* fix
        helix_free = np.where(ec, hr, hz)
        # the helix's own azimuthal displacement from the source, as a length
        dphi_h = hr * ((np.arctan2(hy, hx) - phi0 + np.pi)
                       % (2 * np.pi) - np.pi)
        last = ec.astype(float)
        e0_phi = e0_z = None

    cols = [
        r0, z,
        px * c + py * s,            # p_r
        -px * s + py * c,           # p_phi
        pz,
        t["qop"].to_numpy(),
        surf, dsurf,
    ]
    if not pure:
        cols += [helix_free, dphi_h]
    cols += [fm.bz(x, y, z), last]
    if e0_phi is not None:
        cols += [e0_phi, e0_z]
    return np.column_stack(cols)


def combined(t, args, rng, tr_i, va_i, te_i, keep_perfect, keep_helix,
             keep_model, order, edges):
    """Train s_theta on the same split and score the transport unit whole.

    Every gate number elsewhere in this file reads the destination surface off
    the true trajectory. That is the right way to measure g_theta on its own
    and the wrong way to claim a transport unit, because naming the wrong
    layer loses the hit no matter how good the correction is. Here a hit
    survives only if s_theta names the true layer *and* the chi2 gate passes.

    Top-k is reported alongside because PLAN's mitigation for a hard s_theta is
    exactly top-k with a fallback to the navigator, and k is the thing that
    decides whether that mitigation costs anything.
    """
    src = layer_id(t["surf0"].to_numpy())
    dst = layer_id(t["surf1"].to_numpy())
    src_vals = np.unique(src)
    src_idx = np.searchsorted(src_vals, src)

    # classes from the training fold only; a destination never seen in
    # training is unpredictable by construction and counts as a miss
    cls = np.unique(dst[tr_i])
    pos = np.clip(np.searchsorted(cls, dst), 0, len(cls) - 1)
    seen = cls[pos] == dst
    y = np.where(seen, pos, 0)

    Xs = state_features(t, field_source(args.field), len(src_vals), src_idx)
    mu, sd = Xs[tr_i].mean(0), Xs[tr_i].std(0)
    sd[sd == 0] = 1.0
    Xs = (Xs - mu) / sd

    print(f"\n=== s_theta: next layer, {len(cls)} classes from "
          f"{len(src_vals)} source layers ===")
    net = MLP(Xs.shape[1], args.act, rng, args.hidden, len(cls))
    m = [np.zeros_like(p) for p in net.params()]
    v = [np.zeros_like(p) for p in net.params()]
    step, best, best_w, patience = 0, np.inf, None, 0
    for ep in range(args.epochs):
        perm = rng.permutation(tr_i)
        for k in range(0, len(perm), args.batch):
            idx = perm[k:k + args.batch]
            p, cache = net.forward(Xs[idx])
            _, d = softmax_ce(p, y[idx])
            gW, gb = net.backward(cache, d)
            step += 1
            for i, (par, g) in enumerate(zip(net.params(), gW + gb)):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                par -= args.lr * (m[i] / (1 - 0.9 ** step)) / (
                    np.sqrt(v[i] / (1 - 0.999 ** step)) + 1e-8)
        vl = softmax_ce(net.forward(Xs[va_i])[0], y[va_i])[0]
        if vl < best - 1e-6:
            best, best_w, patience = vl, [p.copy() for p in net.params()], 0
        else:
            patience += 1
        if patience >= 15:
            break
    for p, w in zip(net.params(), best_w):
        p[...] = w

    logits = net.forward(Xs[te_i])[0]
    rank = np.argsort(-logits, axis=1)
    hit_at = {k: (rank[:, :k] == y[te_i][:, None]).any(1) & seen[te_i]
              for k in (1, 2, 3, 5)}
    for k in (1, 2, 3, 5):
        print(f"  top-{k} layer accuracy {100*hit_at[k].mean():6.2f}%")

    def holes(keep):
        return np.array([q.sum() for q in
                         np.split((1 - keep)[order], edges)]).mean()

    print(f"\n  {'':30}{'hits lost':>12}{'holes/track':>14}")
    print(f"  {'helix alone, surface given':30}"
          f"{100*(keep_perfect.mean()-keep_helix.mean()):12.2f}"
          f"{holes(keep_helix):14.2f}")
    print(f"  {'helix + g_theta, given':30}"
          f"{100*(keep_perfect.mean()-keep_model.mean()):12.2f}"
          f"{holes(keep_model):14.2f}")
    for k in (1, 3):
        kc = keep_model * hit_at[k]
        print(f"  {'+ s_theta top-' + str(k):30}"
              f"{100*(keep_perfect.mean()-kc.mean()):12.2f}"
              f"{holes(kc):14.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("--legacy-features", action="store_true",
                    help="force the cylinder inputs on a table that carries "
                         "both. Input 7 is then r1 or z1, the TRUE next hit's "
                         "own coordinate, so this is not a runnable model -- "
                         "it exists to attribute the change between the two "
                         "targets to the surface or to the inputs.")
    ap.add_argument("--legacy-target", action="store_true",
                    help="force the cylinder residual on a table that carries "
                         "both, so the old and new targets can be compared on "
                         "the SAME rows and the same split. Also not a "
                         "runnable model: that target is the true next hit's "
                         "own radius.")
    ap.add_argument("--cov", default="sigma_C.parquet")
    ap.add_argument("--digi", default="odd-digi-smearing-config.json")
    ap.add_argument("--field", default="/tmp/oddb.npz",
                    help="where the field feature comes from: a map npz, or "
                         "`const` for the uniform 2 T the reconstruction runs "
                         "and `gen_teacher.py --field const` generates")
    ap.add_argument("--act", default="relu", choices=list(ACTS))
    ap.add_argument("--model", default="residual",
                    choices=["residual", "pure"],
                    help="residual = helix + g_theta; pure = MLP predicts the "
                         "destination outright (ablation: does the physics "
                         "core earn its place?)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--pure-warmup", type=int, default=None,
                    help="epochs of plain MSE on the standardised target "
                         "before switching to the gate-aligned Huber (pure "
                         "model only). Defaults to all of them, i.e. never "
                         "switch, which is what measured best: the Huber knee "
                         "sits at sqrt(15) sigma -- a few hundred um -- and "
                         "the pure model's residual never gets near it, so "
                         "the switch only turns the loss into an L1 that "
                         "destabilises training (40 -> 62 points lost). The "
                         "gate-aligned loss is the right one for a model "
                         "whose errors are already gate-sized; it is the "
                         "wrong one for a model whose errors are not.")
    ap.add_argument("--dump", default=None,
                    help="write the test-fold arrays to this .npz so the "
                         "figures are made from the run and not from numbers "
                         "retyped out of a table")
    ap.add_argument("--dump-train", default=None,
                    help="the same arrays on the TRAINING fold. A held-out "
                         "residual alone cannot tell a model that does not "
                         "fit a region from one that fits it and does not "
                         "generalise, and those have different fixes: the "
                         "first is the loss, the second is capacity or the "
                         "teacher. Written after everything reported, so a "
                         "run with this flag reports the same numbers as one "
                         "without it.")
    ap.add_argument("--save", default=None,
                    help="write the trained parameters, the input "
                         "normalisation and the v1 scales to this .npz. "
                         "Without it a run is only reproducible by retraining, "
                         "which is why no model existed to feed its own "
                         "filtered state back into -- the thing chaining needs.")
    ap.add_argument("--jacobian", action="store_true",
                    help="check F = dg_theta/dx against finite differences "
                         "and time it against the forward pass. This is the "
                         "measurement that decides the covariance route.")
    ap.add_argument("--v1", action="store_true",
                    help="also supervise direction and q/p, which is what "
                         "chaining jumps needs. Uses outputs 2-4 of the six "
                         "the kernel already emits, so it costs nothing at "
                         "inference.")
    ap.add_argument("--v1-global", action="store_true",
                    help="scale outputs 2-4 by one triple for the whole "
                         "training fold, which is what every model before this "
                         "was fitted with, and which measures -12.68 "
                         "efficiency points in the barrel above 20 GeV. "
                         "Here so the two normalisations can be fitted on the "
                         "same teacher and the same seed and compared; not for "
                         "shipping.")
    ap.add_argument("--stheta", action="store_true",
                    help="also train the next-layer classifier and score the "
                         "two together, so the gate stops assuming the "
                         "destination surface is known")
    ap.add_argument("--field-gate", type=float, default=None,
                    help="the runtime's own routing condition, in tesla: a "
                         "transport belongs to the gated population when "
                         "|bz - 2 T| at its source exceeds this. Read off the "
                         "same map --field names, which is the map the "
                         "features already read. It selects nothing on its "
                         "own; --gate-fit and --gate-share are what use it.")
    ap.add_argument("--gate-fit", action="store_true",
                    help="fit on the gated population alone. The split is "
                         "taken on the WHOLE sample first and only the "
                         "training and validation folds are then restricted, "
                         "so a model fitted this way is held out on the "
                         "identical rows as one fitted on everything and the "
                         "two are comparable without a second split.")
    ap.add_argument("--gate-share", type=float, default=None,
                    help="reweight instead of restricting: scale the gated "
                         "rows so they carry this share of the training "
                         "fold's Huber loss at zero prediction. That is the "
                         "quantity `--cell-weight loss` equalises, over two "
                         "groups rather than seventy-five cells, and the "
                         "direction is set by hand here because the gated "
                         "rows already carry more of the loss than their "
                         "share of the population. A SAMPLE weight: it is "
                         "`sw` and multiplies after the Huber knee.")
    ap.add_argument("--cell-weight", default=None, choices=["loss", "count"],
                    help="give every (module class, pT, |eta|) cell the same "
                         "share of the loss. `loss` equalises what each cell "
                         "contributes, which is the count times the size of "
                         "the residual in it; `count` equalises population "
                         "only and leaves the residual-size factor in. Either "
                         "way this is a SAMPLE weight, multiplied after the "
                         "Huber knee -- not the loss's `w`, which rescales "
                         "the residual before the knee and would move it.")
    ap.add_argument("--cell-weight-power", type=float, default=1.0,
                    help="temper the cell weight: 0 is the unweighted loss, "
                         "1 is full equalisation. The two ends are a straight "
                         "trade between the barrel and the transition region, "
                         "so this is the axis that says whether that is a "
                         "wall or a frontier.")
    ap.add_argument("--hidden", type=int, default=H,
                    help="width. 64 is the deployed size, pinned by the speed "
                         "budget (456 ns, 6.0x). Only worth changing to ask "
                         "whether a result is limited by capacity -- any "
                         "answer found above 64 costs a re-measurement.")
    ap.add_argument("--seed", type=int, default=20260730,
                    help="also reshuffles the track split, so a spread over "
                         "seeds is the noise floor for any comparison")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    t = pl.read_parquet(args.pairs)

    # --- the gate's own scale, which is both the loss normaliser and the
    # --- yardstick the result is reported in
    res = resolutions(args.digi)
    vol = t["surf1"].to_numpy() >> 56
    v0 = np.array([res[v][0] for v in vol])
    v1r = [res[v][1] for v in vol]
    one_d = np.array([q is None for q in v1r])
    v1 = np.array([1.0 if q is None else q for q in v1r])
    c0, c1 = measured_cov(t, vol, args.cov)
    s0, s1 = np.hypot(v0, c0), np.hypot(v1, c1)

    pure = args.model == "pure"
    if args.legacy_target and "res_loc0_um" in t.columns:
        # force the cylinder residual on a table that carries both. Same rows,
        # same split, same seed, so the only thing that moves is the target.
        b0, b1 = in_plane_residual(t.drop("res_loc0_um", "res_loc1_um"))
    else:
        b0, b1 = in_plane_residual(t)             # what the helix gets wrong
    X = features(t, field_source(args.field), pure=pure,
                 legacy=args.legacy_features)

    # target in um, and what the gate's sigma is for each coordinate
    tgt0, tgt1 = absolute_target(t) if pure else (b0, b1)
    sig = np.column_stack([s0, s1])

    # --- split by track
    tracks = t["track"].to_numpy()
    uniq = rng.permutation(np.unique(tracks))
    n = len(uniq)
    lut = {}
    for i, tr in enumerate(uniq):
        lut[tr] = 0 if i < 0.70 * n else (1 if i < 0.85 * n else 2)
    fold = np.array([lut[x] for x in tracks])
    tr_i, va_i, te_i = (np.flatnonzero(fold == k) for k in (0, 1, 2))
    print(f"\n{len(t):,} jumps, {n:,} tracks -> "
          f"{len(tr_i):,} train / {len(va_i):,} val / {len(te_i):,} test")

    # --- the population the deployed configuration actually asks about.
    # `LearnedStepper.hpp` branches per transport on |bzMap - 2 T| against the
    # field gate and runs the helix alone below it, so at 0.10 T the network
    # is switched off over the whole barrel. Restricting or reweighting on
    # THIS condition is not an |eta| cut: in |eta| 1.2 to 1.8 the two differ
    # by a factor of thirteen.
    gmask = None
    if args.field_gate is not None:
        gx, gy, gz = (t[c].to_numpy() for c in ("x", "y", "z"))
        gmask = np.abs(field_source(args.field).bz(gx, gy, gz)
                       - 2.0) > args.field_gate
        print(f"field gate |bz - 2 T| > {args.field_gate:g} T at the source: "
              f"{gmask.sum():,} of {len(t):,} jumps "
              f"({100 * gmask.mean():.2f} %), {gmask[tr_i].sum():,} of "
              f"{len(tr_i):,} on the training fold")
    if args.gate_fit:
        if gmask is None:
            raise SystemExit("--gate-fit needs --field-gate")
        # The test fold is deliberately left whole. It is then the same
        # held-out sample either way and the gated subset of it is selected at
        # scoring time, so a restricted fit and a full one are read on the
        # identical rows.
        tr_i, va_i = tr_i[gmask[tr_i]], va_i[gmask[va_i]]
        print(f"  fitting on the gated population alone -> {len(tr_i):,} "
              f"train / {len(va_i):,} val, test fold left whole")

    # --- output parameterisation.
    # The residual model predicts in units of sigma directly, which is the
    # right scale because that is what the gate measures. The pure model
    # cannot: its target spans ~1e5 um while sigma is ~1e2, so an output in
    # units of sigma would have to reach 1e3 from an initialisation near zero.
    # It gets a standardised output instead -- an affine reparameterisation,
    # with the loss left alone, so the two models are still trained on the
    # identical objective (Huber on residual/sigma with the knee at the gate).
    tgt = np.column_stack([tgt0, tgt1])
    if pure:
        off = tgt[tr_i].mean(0)
        scale = tgt[tr_i].std(0)
    else:
        off, scale = np.zeros(2), None            # scale is per-sample: sigma
    Y = (tgt - off) / (scale if pure else sig)
    # weight carrying (output scale)/sigma, and dropping the unmeasured
    # second coordinate of 1D modules
    W = (scale / sig) if pure else np.ones_like(sig)
    W[:, 1] = np.where(one_d, 0.0, W[:, 1])
    Y[:, 1] = np.where(one_d, 0.0, Y[:, 1])

    # --- v1: direction and q/p, on outputs 2-4.
    # These have no sigma from the gate, since the gate tests position on a
    # surface. Each is normalised by the spread of the helix's own residual on
    # the training fold, so the model is asked to shrink each term relative to
    # the baseline it is replacing, and the Huber knee keeps its meaning
    # (sqrt(15) baseline-widths out is as lost as sqrt(15) sigma is).
    n_sup = 2
    v1_cell = None
    if args.v1:
        v1 = np.column_stack(v1_targets(t))
        # Per cell, not per fold. `v1_cell_scale` carries the argument; the one
        # line version is that the target falls as 1/pT and a constant does
        # not. `--v1-global` reproduces the old behaviour, which is the arm
        # this is measured against and the only way to read an older model's
        # numbers against a new one.
        if args.v1_global:
            v1_scale = v1[tr_i].std(0)
            v1_scale[v1_scale == 0] = 1.0         # q/p with material off
            v1_cell = np.tile(v1_scale, (len(V1_CLASSES), len(PT_EDGES) - 1,
                                         len(ETA_EDGES) - 1, 1))
            per_jump = v1_scale
            print("v1 scale: one global triple (the known defect), "
                  f"{np.array2string(v1_scale, precision=4)}")
        else:
            v1_cell, per_jump, have, v1_scale = v1_cell_scale(
                v1, tr_i, vol, t["pt"].to_numpy(), t["abs_eta"].to_numpy())
            n_cell = have.size
            print(f"v1 scale: per (class, pT, |eta|) cell, {have.sum()} of "
                  f"{n_cell} cells fitted and {n_cell - have.sum()} on a "
                  "fallback")
            print(f"  global spread, for reference: "
                  f"{np.array2string(v1_scale, precision=4)}")
            print(f"  cell range phi   {v1_cell[..., 0].min():.4e} to "
                  f"{v1_cell[..., 0].max():.4e}")
            print(f"  cell range theta {v1_cell[..., 1].min():.4e} to "
                  f"{v1_cell[..., 1].max():.4e}")
            print(f"  cell range q/p   {v1_cell[..., 2].min():.4e} to "
                  f"{v1_cell[..., 2].max():.4e}")
        Y = np.hstack([Y, v1 / per_jump])
        W = np.hstack([W, np.ones((len(t), 3))])
        n_sup = 5

    # --- the sample weight. `W` above is the mask and the output-scale
    # --- carrier and keeps doing only that; this is a separate argument.
    SW = None
    if args.cell_weight:
        # what a jump costs with the model predicting nothing: the helix's own
        # residual in the loss's units, which is the quantity the weighting
        # divides out. Position only, because that is the variable the gate
        # tests and the one the argument is about; the v1 targets are already
        # normalised per cell, so at zero prediction they cost about the same
        # everywhere and would only dilute the ratio being equalised.
        base = huber_value(-Y[:, :2] * W[:, :2]).sum(1)
        SW, cnt, big = cell_weights(vol, t["pt"].to_numpy(),
                                    t["abs_eta"].to_numpy(), tr_i, base,
                                    mode=args.cell_weight,
                                    power=args.cell_weight_power)
        print(f"cell weight, mode {args.cell_weight}, power "
              f"{args.cell_weight_power:g}: {big.sum()} of "
              f"{cnt.size} cells equalised, {(cnt > 0).sum() - big.sum()} "
              "occupied but under the 50-jump floor and left at their natural "
              "share")
        print(f"  weight range {SW[tr_i].min():.4f} to {SW[tr_i].max():.4f}, "
              f"mean {SW[tr_i].mean():.4f} on the training fold")

    if args.gate_share is not None:
        if gmask is None:
            raise SystemExit("--gate-share needs --field-gate")
        # what a jump costs with the model predicting nothing: the helix's own
        # residual in the loss's units. A property of the sample and not of a
        # fit, so the weight is fixed before the first step and does not chase
        # the model. Position only, for the reason `--cell-weight` is.
        base = huber_value(-Y[:, :2] * W[:, :2]).sum(1)
        Lg = base[tr_i][gmask[tr_i]].sum()
        Ln = base[tr_i][~gmask[tr_i]].sum()
        f = args.gate_share / (1.0 - args.gate_share) * (Ln / Lg)
        sw = np.where(gmask, f, 1.0)[:, None]
        SW = sw if SW is None else SW * sw
        SW = SW / SW[tr_i].mean()
        print(f"gate weight: the gated rows carry {100 * Lg / (Lg + Ln):.2f} "
              f"% of the training fold's Huber loss unweighted and "
              f"{100 * args.gate_share:.2f} % after a factor {f:.4f} on them")
        print(f"  weight range {SW[tr_i].min():.4f} to {SW[tr_i].max():.4f}, "
              f"mean {SW[tr_i].mean():.4f} on the training fold")

    mu, sd = X[tr_i].mean(0), X[tr_i].std(0)
    sd[sd == 0] = 1.0
    Xn = (X - mu) / sd

    model = MLP(X.shape[1], args.act, rng, args.hidden)
    err = grad_check(model, Xn[tr_i[:64]], Y[tr_i[:64]], W[tr_i[:64]], rng)
    print(f"gradient check (finite differences): worst relative error {err:.2e}"
          + ("  OK" if err < 1e-4 else "  *** BACKPROP IS WRONG ***"))
    if err >= 1e-4:
        raise SystemExit(1)

    # --- Adam
    m = [np.zeros_like(p) for p in model.params()]
    v = [np.zeros_like(p) for p in model.params()]
    step = 0
    best, best_w, patience = np.inf, None, 0
    curve = []
    warm = 0 if not pure else (args.epochs if args.pure_warmup is None
                               else args.pure_warmup)
    for ep in range(args.epochs):
        loss_fn = mse if ep < warm else huber
        if pure and ep == warm and warm:
            print(f"  epoch {ep:4d}   switching from MSE warm-up to Huber")
            best, patience = np.inf, 0            # the two are not comparable
        order = rng.permutation(tr_i)
        for k in range(0, len(order), args.batch):
            idx = order[k:k + args.batch]
            p, cache = model.forward(Xn[idx])
            d = np.zeros_like(p)
            d[:, :n_sup] = loss_fn(p[:, :n_sup], Y[idx], W[idx],
                                   None if SW is None else SW[idx])[1]
            gW, gb = model.backward(cache, d)
            step += 1
            for i, (par, g) in enumerate(zip(model.params(), gW + gb)):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                mh = m[i] / (1 - 0.9 ** step)
                vh = v[i] / (1 - 0.999 ** step)
                par -= args.lr * mh / (np.sqrt(vh) + 1e-8)

        # the same objective the steps are taken on, weight included -- an
        # unweighted validation loss would early-stop on the objective the
        # weighting exists to replace
        pv, _ = model.forward(Xn[va_i])
        vl = float(loss_fn(pv[:, :n_sup], Y[va_i], W[va_i],
                           None if SW is None else SW[va_i])[0])
        if vl < best - 1e-6:
            best, best_w, patience = vl, [p.copy() for p in model.params()], 0
        else:
            patience += 1
        ptr, _ = model.forward(Xn[tr_i])
        tl = float(loss_fn(ptr[:, :n_sup], Y[tr_i], W[tr_i],
                           None if SW is None else SW[tr_i])[0])
        curve.append((ep, tl, vl, ep >= warm))
        if ep % 20 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:4d}   train {tl:10.4f}   val {vl:10.4f}")
        if patience >= 25:
            print(f"  early stop at epoch {ep}")
            break
    for p, w in zip(model.params(), best_w):
        p[...] = w

    # --- score on the test fold, in points of hit efficiency
    print(f"\n=== test fold, {len(te_i):,} jumps ===")
    pred, _ = model.forward(Xn[te_i])
    if pure:
        # the network's own answer, in um, with no helix anywhere in it
        pb = pred[:, :2] * scale + off
    else:
        # a correction, to be added to the helix
        pb = pred[:, :2] * sig[te_i]
    rem0 = tgt[te_i, 0] - pb[:, 0]
    rem1 = np.where(one_d[te_i], 0.0, tgt[te_i, 1] - pb[:, 1])

    g = rng
    _, keep_perfect = gate(np.zeros(len(te_i)), np.zeros(len(te_i)),
                           s0[te_i], s1[te_i], one_d[te_i], g)
    _, keep_helix = gate(b0[te_i], b1[te_i], s0[te_i], s1[te_i],
                         one_d[te_i], g)
    _, keep_model = gate(rem0, rem1, s0[te_i], s1[te_i], one_d[te_i], g)

    tr_t = tracks[te_i]
    order = np.argsort(tr_t, kind="stable")
    edges = np.flatnonzero(np.diff(tr_t[order])) + 1

    def holes(keep):
        return np.array([q.sum() for q in
                         np.split((1 - keep)[order], edges)]).mean()

    lh = 100 * (keep_perfect.mean() - keep_helix.mean())
    lm = 100 * (keep_perfect.mean() - keep_model.mean())
    print(f"  {'':22}{'|b| median':>12}{'p90':>10}{'hits lost':>12}"
          f"{'holes/track':>14}")
    print(f"  {'helix alone':22}{np.median(np.abs(b0[te_i])):12.1f}"
          f"{np.percentile(np.abs(b0[te_i]), 90):10.1f}{lh:12.2f}"
          f"{holes(keep_helix):14.2f}")
    label = "pure MLP (no helix)" if pure else "helix + g_theta"
    print(f"  {label:22}{np.median(np.abs(rem0)):12.1f}"
          f"{np.percentile(np.abs(rem0), 90):10.1f}{lm:12.2f}"
          f"{holes(keep_model):14.2f}")
    print(f"\n  captured {100*(1 - lm/lh):5.1f}% of the loss "
          f"({args.act}, {args.model}, {X.shape[1]}-{args.hidden}-"
          f"{args.hidden}-{N_OUT}, seed {args.seed})")
    if pure:
        # the range the two models have to cover, which is the whole point
        print(f"  target span: |a| p90 = "
              f"{np.percentile(np.abs(tgt[te_i, 0]), 90):,.0f} um azimuthal, "
              f"against the helix's {np.percentile(np.abs(b0[te_i]), 90):,.0f} um")

    if args.dump:
        # everything a figure could want, per jump on the test fold, so the
        # plots are made from the run rather than from numbers retyped out of
        # a table. make_figures.py reads these.
        d = dict(b0=b0[te_i], b1=b1[te_i], rem0=rem0, rem1=rem1,
                 s0=s0[te_i], s1=s1[te_i], one_d=one_d[te_i], vol=vol[te_i],
                 keep_perfect=keep_perfect, keep_helix=keep_helix,
                 keep_model=keep_model, track=tracks[te_i],
                 abs_eta=t["abs_eta"].to_numpy()[te_i],
                 pt=t["pt"].to_numpy()[te_i],
                 endcap=t["endcap"].to_numpy()[te_i],
                 curve=np.array(curve, dtype=float),
                 hits_lost=np.array([lh, lm]),
                 meta=np.array([args.hidden, X.shape[1], args.seed,
                                float(pure), args.lr]))
        if args.v1:
            d["v1_helix"] = v1[te_i]
            # per jump, since the scale is now per cell. `per_jump` is
            # broadcast when --v1-global reproduces the old behaviour.
            d["v1_rem"] = v1[te_i] - pred[:, 2:5] * v1_at(per_jump, te_i)
            d["arm"] = np.median(np.hypot(t["dr"].to_numpy(),
                                          t["dz"].to_numpy()))
        np.savez_compressed(args.dump, **d)
        print(f"\n  dumped test-fold arrays -> {args.dump}")

    if args.save:
        # everything needed to run the forward pass on a state this script
        # never saw: the weights, the input normalisation they were fitted
        # with, and the output scales that turn the six raw outputs back into
        # micrometres, radians and 1/GeV.
        p = {f"W{i}": w for i, w in enumerate(model.W)}
        p.update({f"b{i}": b for i, b in enumerate(model.b)})
        p.update(mu=mu, sd=sd,
                 v1_scale=(v1_scale if args.v1 else np.ones(3)),
                 v1_scale_cell=(v1_cell if v1_cell is not None else
                                np.ones((len(V1_CLASSES), len(PT_EDGES) - 1,
                                         len(ETA_EDGES) - 1, 3))),
                 meta=np.array([args.hidden, X.shape[1], N_OUT, args.seed,
                                float(pure), float(args.v1)]),
                 act=np.array([args.act]), cov=np.array([args.cov]),
                 digi=np.array([args.digi]), field=np.array([args.field]))
        if pure:
            p.update(off=off, scale=scale)
        np.savez(args.save, **p)
        print(f"  saved parameters -> {args.save}")

    if args.jacobian:
        import time
        xs = Xn[te_i]
        J = model.jacobian(xs[:256])
        worst = 0.0
        for j in rng.integers(0, xs.shape[1], 6):
            eps = 1e-5
            xp, xm = xs[:256].copy(), xs[:256].copy()
            xp[:, j] += eps
            xm[:, j] -= eps
            num = (model.forward(xp)[0] - model.forward(xm)[0]) / (2 * eps)
            den = np.maximum(np.abs(num), np.abs(J[:, :, j])).clip(1e-9)
            worst = max(worst, np.abs(num - J[:, :, j]).max()
                        / den.max())
        print(f"\n=== F = dg_theta/dx ===")
        print(f"  analytic vs finite differences: worst relative error "
              f"{worst:.2e}" + ("  OK" if worst < 1e-5 else "  *** WRONG ***"))

        big = np.repeat(xs, max(1, 20000 // len(xs)), axis=0)
        base = None
        for name, fn in (("forward", lambda: model.forward(big)),
                         ("F, 2 rows", lambda: model.jacobian(big, 2)),
                         ("F, all 6", lambda: model.jacobian(big))):
            fn()
            t0 = time.perf_counter()
            for _ in range(5):
                fn()
            dt = (time.perf_counter() - t0) / 5 / len(big) * 1e9
            base = base or dt
            print(f"  {name:12}{dt:8.0f} ns/jump   {dt/base:5.2f}x forward")
        # numpy's ratio is overhead-dominated for the small case -- the
        # forward is one clean GEMM while F builds a (batch, H, nout) tensor
        # per jump. For a C++ kernel the multiply-add count is the honest
        # estimate, so print both and say which is which.
        nin, hid = X.shape[1], args.hidden
        fwd = nin * hid + hid * hid + hid * N_OUT
        print(f"  numpy, batch {len(big):,}. Those ratios are "
              f"overhead-dominated: the forward is one\n  GEMM while F builds "
              f"a per-jump tensor. For the C++ kernel count multiply-adds:")
        for k in (2, N_OUT):
            j = hid * hid * k + nin * hid * k
            print(f"    F, {k} rows: {j:,} MACs against the forward's "
                  f"{fwd:,} = {j/fwd:.2f}x")
        print(f"  So the deployed {456} ns becomes roughly "
              f"{456*(1+ (hid*hid*2 + nin*hid*2)/fwd):.0f} ns for the two "
              f"position rows and\n  {456*(1+(hid*hid*N_OUT+nin*hid*N_OUT)/fwd):.0f}"
              f" ns for all six, against RKN's ~2.7 us. The 6.0x speed-up\n"
              f"  survives the first and does not survive the second, so how "
              f"many rows of F\n  the filter actually needs is a real design "
              f"question and not a detail.")
        print("  This is only g_theta's block. The helix's own Jacobian and "
              "the feature\n  map's still have to be chained onto it, and "
              "neither is measured yet.")

    if args.v1:
        # The gate cannot score these -- it tests position on a surface -- so
        # they are reported on their own scales, plus the position error the
        # direction error would produce over the median step. That last column
        # is the one to read: it is in the same micrometres as everything else
        # and it says what the direction error would cost the *next* jump.
        arm = np.median(np.hypot(t["dr"].to_numpy(), t["dz"].to_numpy()))
        names = ["phi [mrad]", "theta [mrad]", "q/p [rel]"]
        units = [1e3, 1e3, 1.0]
        print(f"\n=== v1 outputs (median step {arm:.0f} mm as lever arm) ===")
        print(f"  {'':16}{'helix med':>11}{'+g_th med':>11}"
              f"{'helix p90':>11}{'+g_th p90':>11}{'-> um over the step':>22}")
        for j, (nm, u) in enumerate(zip(names, units)):
            h = v1[te_i, j]                  # what the helix gets wrong
            # per jump: one scale per cell, so the reported residual has to
            # use each jump's own or it is not what the runtime applies
            r = h - pred[:, 2 + j] * v1_at(per_jump, te_i)[:, j]
            extra = ""
            if j < 2:                        # an angle, so arc = angle * arm
                extra = (f"{np.percentile(np.abs(h), 90) * arm * 1e3:9.0f}"
                         f" -> {np.percentile(np.abs(r), 90) * arm * 1e3:<9.0f}")
            print(f"  {nm:16}"
                  f"{np.median(np.abs(h)) * u:11.4f}"
                  f"{np.median(np.abs(r)) * u:11.4f}"
                  f"{np.percentile(np.abs(h), 90) * u:11.4f}"
                  f"{np.percentile(np.abs(r), 90) * u:11.4f}{extra:>22}")
        rel = np.abs(v1[:, 2] / t["qop"].to_numpy()).max()
        if rel < 1e-5:
            print(f"  q/p residual is zero to {rel:.0e} relative -- material "
                  f"is off and the helix conserves |p| exactly, so that "
                  f"output has nothing to learn here.")

    if args.stheta:
        combined(t, args, rng, tr_i, va_i, te_i, keep_perfect, keep_helix,
                 keep_model, order, edges)

    for lo, hi in [(0, 1), (1, 1.5), (1.5, 2), (2, 3)]:
        ae = t["abs_eta"].to_numpy()[te_i]
        k = (ae >= lo) & (ae < hi)
        if k.sum() < 50:
            continue
        print(f"    |eta| {lo}-{hi}: helix p90 "
              f"{np.percentile(np.abs(b0[te_i][k]), 90):8.1f} -> "
              f"{np.percentile(np.abs(rem0[k]), 90):8.1f} um   "
              f"kept {100*keep_helix[k].mean():5.1f}% -> "
              f"{100*keep_model[k].mean():5.1f}%")

    if args.dump_train:
        # Last, so that every number above is produced by the identical code
        # path whether or not this flag is given: the gate draws from `rng`,
        # and a second fold's worth of draws taken earlier would move the
        # test-fold keeps.
        ti = tr_i
        ptr, _ = model.forward(Xn[ti])
        pbt = (ptr[:, :2] * scale + off) if pure else ptr[:, :2] * sig[ti]
        trem0 = tgt[ti, 0] - pbt[:, 0]
        trem1 = np.where(one_d[ti], 0.0, tgt[ti, 1] - pbt[:, 1])
        _, tkp = gate(np.zeros(len(ti)), np.zeros(len(ti)), s0[ti], s1[ti],
                      one_d[ti], rng)
        _, tkh = gate(b0[ti], b1[ti], s0[ti], s1[ti], one_d[ti], rng)
        _, tkm = gate(trem0, trem1, s0[ti], s1[ti], one_d[ti], rng)
        d = dict(b0=b0[ti], b1=b1[ti], rem0=trem0, rem1=trem1,
                 s0=s0[ti], s1=s1[ti], one_d=one_d[ti], vol=vol[ti],
                 keep_perfect=tkp, keep_helix=tkh, keep_model=tkm,
                 track=tracks[ti],
                 abs_eta=t["abs_eta"].to_numpy()[ti],
                 pt=t["pt"].to_numpy()[ti],
                 endcap=t["endcap"].to_numpy()[ti],
                 curve=np.array(curve, dtype=float),
                 hits_lost=np.array([
                     100 * (tkp.mean() - tkh.mean()),
                     100 * (tkp.mean() - tkm.mean())]),
                 meta=np.array([args.hidden, X.shape[1], args.seed,
                                float(pure), args.lr]))
        if args.v1:
            d["v1_helix"] = v1[ti]
            d["v1_rem"] = v1[ti] - ptr[:, 2:5] * v1_at(per_jump, ti)
            d["arm"] = np.median(np.hypot(t["dr"].to_numpy(),
                                          t["dz"].to_numpy()))
        np.savez_compressed(args.dump_train, **d)
        print(f"\n  dumped training-fold arrays -> {args.dump_train} "
              f"({len(ti):,} jumps)")


if __name__ == "__main__":
    main()
