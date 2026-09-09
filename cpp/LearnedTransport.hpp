// The deployable form of the learned transport: helix + g_theta, surface to
// surface, in ~5,376 multiply-adds of Eigen and no inference runtime.
//
// Why not ONNX Runtime, which is the sanctioned ACTS path (ActsPlugins/Onnx,
// used by ActsExamples::NeuralCalibrator): ORT costs 1788 + 98.0*b ns and the
// CKF's findTracks() is depth-first over an explicit stack, so b = 1 always.
// The fixed term alone is 1,886 ns, against 2,076 ns for the whole ACTS
// EigenStepper it would replace, so the plugin route spends the entire budget
// before the helix solve. Written out by hand the same network is 1,023 ns.
//
// This header has no ACTS dependency on purpose: it is checkable against the
// Python that produced every number in the report (see test_kernel.cpp) before
// any of it is exposed to a filter. LearnedStepper.hpp is the ACTS adapter.
#pragma once

#include <Eigen/Dense>

#include <cmath>

#include "FeatureDump.hpp"
#include "gtheta_weights.hpp"

namespace collider_ml {

namespace w = weights;

inline constexpr int kNIn = w::kNIn;
inline constexpr int kNH = w::kNHidden;
inline constexpr int kNOut = w::kNOut;

using Vec3 = Eigen::Vector3d;
using VecIn = Eigen::Matrix<double, kNIn, 1>;
using VecH = Eigen::Matrix<double, kNH, 1>;
using VecOut = Eigen::Matrix<double, kNOut, 1>;

/// Everything one jump needs. `sig0`/`sig1` are the cell widths the network's
/// position outputs are written in; in ACTS they come from the calibration
/// context rather than from this header, so they are an input rather than a
/// table here. The destination is the module plane, not a cylinder or a
/// z-plane.
///
/// The teacher's old target was the cylinder r = r1 with r1 the true next
/// hit's radius: a surface that does not exist in the detector and is derived
/// from the answer. Every sensitive surface in the ODD is a planar module
/// (0 sensitive cylinders, 0 sensitive discs, 18,824 planes), so the old target
/// and the set of surfaces the detector measures on did not intersect at all.
struct Jump {
  Vec3 pos;        ///< mm
  Vec3 mom;        ///< GeV
  double q = 1.0;  ///< +-1
  Vec3 c;          ///< destination module centre, mm
  Vec3 n;          ///< its normal, unit
  Vec3 e0, e1;     ///< its local axes, unit and orthogonal to n
  double bzMap = 2.0;  ///< tesla, the map value at the SOURCE point
  double sig0 = 1.0;   ///< um
  double sig1 = 1.0;   ///< um

  /// Run the helix core at `bzMap` instead of the fixed 2 T of `kBHelix`.
  ///
  /// A switch and not a replacement. `kBHelix` stays what it is, so one binary
  /// carries both cores and the comparison is one build. Default false, which
  /// is the core every measurement in this repository was taken on, so the
  /// three sites below are bit-identical to what they were unless a caller
  /// asks otherwise.
  ///
  /// Per jump rather than a file-scope flag because `bzMap` is per jump and
  /// because everything that reads this header off the runtime path
  /// (`bench_kernel`, `test_kernel`, `cov_probe`, `nav_truth`) builds its own
  /// `Jump` and keeps the fixed core by construction. Those are measurement
  /// code and are left alone.
  ///
  /// It moves the core alone. The supervised target, inputs 8 and 9, the
  /// header's `mu` and `sd` and the `v1_scale` cells were all written at 2 T,
  /// so a build that sets this and also runs the network is
  /// a model read off a core it was not fitted on. `new-helix` has no network.
  bool localCore = false;

  /// The scale outputs 2 to 4 are applied with: (phi, theta, q/p), in rad,
  /// rad and 1/GeV. An input for the same reason `sig0` is: it is a property of
  /// where the jump is and how stiff the track is, and the caller is the only
  /// one that knows both.
  ///
  /// Defaulted to the header's global triple, which is what a jump on a volume
  /// the runtime cannot classify gets, and what every model fitted before the
  /// per-cell table used everywhere, which cost efficiency in the barrel at
  /// high pT.
  double v1s[3] = {w::v1_scale[0], w::v1_scale[1], w::v1_scale[2]};
};

struct Prediction {
  Vec3 pos;              ///< corrected position on the destination surface, mm
  Vec3 mom;              ///< corrected momentum, GeV
  double s = 0.0;        ///< transverse arc length used, mm
  bool ok = false;       ///< false if the helix never reaches the surface
  double m = 1.0;        ///< per-hit sigma scale, exp(sixth output)
  VecOut raw;            ///< the network's six outputs, unscaled
  Vec3 helixPos;         ///< the physics core's answer, before the correction
  Vec3 helixMom;         ///< ditto for momentum; LearnedJacobian.hpp reads it
  Vec3 e0, e1;           ///< the surface frame the position correction is in
};

namespace detail {

inline constexpr double kMM = 1e-3;
inline constexpr double kBHelix = 2.0;  ///< the helix core runs at fixed 2 T
inline constexpr int kNewton = 8;       ///< identical to the Python solver

/// The field one jump's helix core runs at.
///
/// The three runtime sites go through here:
/// `solveArcLength`, `transport` and `LearnedJacobian::jumpJacobian`. Nothing
/// else does. `LearnedStepper.hpp:589`'s field gate keeps comparing `bzMap`
/// against `kBHelix`, because nominal is what that gate means and not what the
/// core happens to be running at.
inline double coreBz(const Jump& j) {
  return j.localCore ? j.bzMap : kBHelix;
}

/// Pade 3/2 approximant to tanh, clamped at |x| = 3.
///
/// Chosen over relu because the covariance transport needs F = df/dx and a
/// relu Jacobian is piecewise constant, which puts steps into a covariance
/// that is supposed to be smooth in the state. The clamp is C1: the
/// approximant reaches exactly 1 with derivative exactly 0 at x = 3.
inline double ptanh(double x) {
  const double c = x < -3.0 ? -3.0 : (x > 3.0 ? 3.0 : x);
  const double x2 = c * c;
  return c * (27.0 + x2) / (27.0 + 9.0 * x2);
}

/// d(ptanh)/dx, for the Jacobian path.
inline double dptanh(double x) {
  if (std::abs(x) >= 3.0) {
    return 0.0;
  }
  const double x2 = x * x;
  const double a = x2 - 9.0;
  const double b = 3.0 + x2;
  return (a * a) / (9.0 * b * b);
}

inline double wrapPi(double d) {
  constexpr double kTwoPi = 2.0 * M_PI;
  double r = std::fmod(d + M_PI, kTwoPi);
  if (r < 0.0) {
    r += kTwoPi;
  }
  return r - M_PI;
}

inline Vec3 helixPos(const Vec3& p0, double phi0, double pt, double pzOverPt,
                     double kappa, double s) {
  const double phi = phi0 + kappa * s;
  return {p0.x() + (std::sin(phi) - std::sin(phi0)) / kappa,
          p0.y() - (std::cos(phi) - std::cos(phi0)) / kappa,
          p0.z() + pzOverPt * s};
}

}  // namespace detail

/// Transverse arc length onto the module plane n . (X - c) = 0.
///
/// Newton on g(s) = n . (P(s) - c), from the straight-line guess. A tracker
/// crossing is near normal incidence, so this converges in three or four
/// iterations; the cylinder target needed sixty bisection steps because its
/// inside/outside predicate is only monotonic for half a turn. Same solver as
/// `make_teacher_pairs.plane_predict`, so the kernel and the teacher agree by
/// construction rather than by inspection.
///
/// `ok` is false for a grazing crossing, where n . dP/ds is near zero and
/// Newton walks off, and for a solution behind the source.
inline double solveArcLength(const Jump& j, bool* ok) {
  const double pt = std::hypot(j.mom.x(), j.mom.y());
  const double phi0 = std::atan2(j.mom.y(), j.mom.x());
  const double kappa = -0.3 * j.q * detail::coreBz(j) / pt * detail::kMM;
  const double pzOverPt = j.mom.z() / pt;

  auto gAndDg = [&](double s, double* dg) {
    const double phi = phi0 + kappa * s;
    const Vec3 p = detail::helixPos(j.pos, phi0, pt, pzOverPt, kappa, s);
    *dg = j.n.x() * std::cos(phi) + j.n.y() * std::sin(phi) +
          j.n.z() * pzOverPt;
    return j.n.dot(p - j.c);
  };

  double dg0 = 0.0;
  const double g0 = gAndDg(0.0, &dg0);
  if (std::abs(dg0) < 1e-3) {   // skimming the module edge on
    *ok = false;                // rather than crossing it
    return 0.0;
  }
  double s = -g0 / dg0;
  for (int i = 0; i < detail::kNewton; ++i) {
    double dg = 0.0;
    const double g = gAndDg(s, &dg);
    if (std::abs(dg) < 1e-12) {
      break;
    }
    s -= g / dg;
  }
  double dg = 0.0;
  const double g = gAndDg(s, &dg);
  *ok = std::isfinite(s) && s > 0.0 && std::abs(g) < 1e-6;
  return s;
}

/// The fourteen features, in the order the network was trained on.
///
/// Everything that names the destination is expressed against the module's own
/// plane and axes, and nothing is derived from the answer. There is deliberately
/// no surface identifier -- 28,478 distinct (source, destination) pairs with the
/// top 3,462 covering half the jumps, so a model given the identifier memorises
/// the detector instead of learning the physics.
inline VecIn features(const Jump& j, const Vec3& helixPos, double qop) {
  const double r0 = std::hypot(j.pos.x(), j.pos.y());
  const double phi0 = std::atan2(j.pos.y(), j.pos.x());
  const double c = std::cos(phi0);
  const double s = std::sin(phi0);

  // 7 and 8, the pair that is INVARIANT under the module normal's sign.
  // Which face DD4hep calls the front is a per-module convention and 36.4% of
  // real jumps have p . n < 0, so both d_plane and cos_inc flip. Their ratio
  // and the absolute cosine do not, and the raw pair would teach the network
  // conventions instead of physics.
  const double pAbs = j.mom.norm();
  const double cosInc = j.mom.dot(j.n) / pAbs;
  const double dPlane = (j.c - j.pos).dot(j.n);
  const double pathToPlane =
      std::abs(cosInc) > 1e-9 ? dPlane / cosInc : 0.0;

  // 9 and 10, the helix's answer in the MODULE's own local axes, which is the
  // frame the correction is written in and the frame the sensor resolutions
  // are quoted in.
  const Vec3 d = helixPos - j.c;

  // 13, e0 resolved against phi_hat at the module centre. Together with e0.z
  // this fixes the in-plane rotation given the normal.
  const double rc = std::hypot(j.c.x(), j.c.y());
  const double e0Phi =
      rc > 0.0 ? (-j.e0.x() * j.c.y() + j.e0.y() * j.c.x()) / rc : 0.0;

  VecIn x;
  x << r0, j.pos.z(),
      j.mom.x() * c + j.mom.y() * s,   // p along r_hat at the source
      -j.mom.x() * s + j.mom.y() * c,  // p along phi_hat
      j.mom.z(), qop, pathToPlane, std::abs(cosInc), d.dot(j.e0), d.dot(j.e1),
      j.bzMap,
      // 12: how forward-facing the module is. Continuous, because with a plane
      // target there is no barrel/disc switch left to encode.
      std::abs(j.n.z()),
      // 13 and 14: WHERE e0 POINTS INSIDE THE PLANE, resolved in the
      // cylindrical frame at the module centre.
      //
      // Not optional. The correction is a vector in this module's own axes and
      // those axes rotate from module to module, while the old cylinder frame
      // had e0 = phi_hat everywhere. Without these two the network is asked
      // for components in a basis it cannot see and can only learn the average
      // basis: measured, the same model captures 6.3% of the loss without them
      // and 37.3% with them.
      e0Phi, j.e0.z();
  return x;
}

// The surface frame is no longer constructed here. It is the module's own
// transform, carried in on the Jump, because the residual has to be written in
// the frame the sensor resolutions are quoted in. Rebuilding a cylinder frame
// from the crossing point instead would be 150 mrad off on every ODD barrel
// module, which is not a relabelling: on identical jumps the two frames give
// residuals differing by 142 um in the median, the size of the residual.

/// x -> (six outputs, last hidden layer).
///
/// The hidden layer is returned rather than recomputed because the sigma head
/// reads it, and computing the forward pass twice would misrepresent the cost
/// of the thing whose whole claim is that it costs nothing.
inline void forward(const VecIn& xRaw, VecOut* out, VecH* hidden) {
  using Eigen::Map;
  using Eigen::RowMajor;
  const Map<const Eigen::Matrix<double, kNIn, kNH, RowMajor>> W0(w::W0);
  const Map<const Eigen::Matrix<double, kNH, kNH, RowMajor>> W1(w::W1);
  const Map<const Eigen::Matrix<double, kNH, kNOut, RowMajor>> W2(w::W2);
  const Map<const VecH> b0(w::b0), b1(w::b1);
  const Map<const VecOut> b2(w::b2);
  const Map<const VecIn> mu(w::mu), sd(w::sd);

  const VecIn x = (xRaw - mu).cwiseQuotient(sd);

  VecH h1 = W0.transpose() * x + b0;
  for (int i = 0; i < kNH; ++i) {
    h1[i] = detail::ptanh(h1[i]);
  }
  VecH h2 = W1.transpose() * h1 + b1;
  for (int i = 0; i < kNH; ++i) {
    h2[i] = detail::ptanh(h2[i]);
  }
  *out = W2.transpose() * h2 + b2;
  *hidden = h2;
}

/// The per-hit sigma scale: a linear readout of the existing hidden units.
///
/// That a *linear* readout suffices is itself the result -- the features
/// g_theta computes in order to make the correction already carry the
/// correction's uncertainty. 64 weights and one bias on an output the kernel
/// already emits, so inference cost is unchanged.
inline double sigmaScale(const VecH& hidden) {
  const Eigen::Map<const VecH> hw(w::head_w);
  return std::exp(hidden.dot(hw) + w::head_b);
}

/// helix + g_theta from an arbitrary state.
///
/// The two sign conventions are not symmetric and reversing them produces a
/// model that makes things exactly twice as bad, which is a suspiciously
/// plausible failure mode:
///   position   the target is b = helix - truth, so the correction is
///              SUBTRACTED from the helix point
///   direction  the target is truth - helix, so it is ADDED
/// `useNetwork == false` is the helix bypass: the forward pass is skipped and
/// the six outputs are taken as exactly zero. Nothing below the branch changes,
/// which is deliberate. A build with `W2`'s columns and `b2` zeroed produces
/// `raw` that is exactly 0.0 in IEEE too, so the two must agree bit for bit on
/// every jump and the acceptance test for the bypass is that they do. Writing
/// `r.mom = helixMom` here instead would be the same number to a few ulp and
/// not the same bits, because the line below rebuilds it through atan2 and
/// sin/cos.
inline Prediction transport(const Jump& j, bool useNetwork = true) {
  Prediction r;
  r.s = solveArcLength(j, &r.ok);

  const double pt = std::hypot(j.mom.x(), j.mom.y());
  const double phi0 = std::atan2(j.mom.y(), j.mom.x());
  const double kappa = -0.3 * j.q * detail::coreBz(j) / pt * detail::kMM;
  r.helixPos = detail::helixPos(j.pos, phi0, pt, j.mom.z() / pt, kappa, r.s);

  const double phiH = phi0 + kappa * r.s;
  const Vec3 helixMom(pt * std::cos(phiH), pt * std::sin(phiH), j.mom.z());
  r.helixMom = helixMom;

  const double qop = j.q / j.mom.norm();
  if (useNetwork) {
    VecH hidden;
    const VecIn x = features(j, r.helixPos, qop);
    // The acceptance check's only tap, and it is here rather than in the
    // stepper because this is the last point at which the vector is still
    // the one `forward()` will read. Off unless FEATURE_DUMP names a file.
    if (auto& dump = FeatureDump::instance(); dump.on()) {
      dump.record(x.data(), kNIn);
    }
    forward(x, &r.raw, &hidden);
    r.m = sigmaScale(hidden);
  } else {
    r.raw.setZero();
    r.m = 1.0;
    // The helix arm never builds the input vector, because it never runs the
    // forward pass. It builds one here when the check is on, and only then, so
    // that no timed run pays for it. Without this arm the check can say that an
    // input is outside its training range and cannot say who put it there: the
    // stepper state the features are read off is the state the previous
    // corrections left behind, so an input can be driven out of range by the
    // model's own output. The helix arm is the same transport with the network
    // removed, so it is the control that separates the two.
    if (auto& dump = FeatureDump::instance(); dump.on()) {
      const VecIn xOff = features(j, r.helixPos, qop);
      dump.record(xOff.data(), kNIn);
    }
  }

  // The module's own axes, not a frame rebuilt from the crossing point. The
  // correction is in the frame the residual was trained in, and that is the
  // module's; a cylinder frame would be 150 mrad off on every ODD barrel
  // module.
  r.e0 = j.e0;
  r.e1 = j.e1;
  r.pos = r.helixPos - (r.raw[0] * j.sig0 * detail::kMM) * r.e0 -
          (r.raw[1] * j.sig1 * detail::kMM) * r.e1;

  // direction: the helix conserves |p|, so its q/p prediction is the source
  // q/p and the correction is applied to that rather than to a propagated one
  // `j.v1s`, not `w::v1_scale`. The scale is per (class, pT bin, |eta| bin) and
  // the caller looked it up, for a measured reason: one
  // triple for the whole detector and the whole momentum range is the size of a
  // soft track's error, and applying it where the helix is already right cost
  // 12.68 efficiency points in the barrel above 20 GeV.
  const double phiHel = std::atan2(helixMom.y(), helixMom.x());
  const double thHel =
      std::atan2(std::hypot(helixMom.x(), helixMom.y()), helixMom.z());
  const double phiC = phiHel + r.raw[2] * j.v1s[0];
  const double thC = thHel + r.raw[3] * j.v1s[1];
  double qopC = qop + r.raw[4] * j.v1s[2];
  if (std::abs(qopC) < 1e-9) {
    qopC = std::copysign(1e-9, qopC);
  }
  const double pC = std::abs(1.0 / qopC);
  r.mom = Vec3(pC * std::sin(thC) * std::cos(phiC),
               pC * std::sin(thC) * std::sin(phiC), pC * std::cos(thC));
  return r;
}

}  // namespace collider_ml
