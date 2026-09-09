// The transport Jacobian of one learned jump, in the free 8x8 form ACTS's
// covariance engine consumes.
//
// This is the C++ side of `src/prop/jacobian_free.py`, and `test_jacobian.cpp`
// checks it against that file's output on real jumps. Like LearnedTransport.hpp
// it has no ACTS dependency, so the numerics can be verified before any of it
// is exposed to a filter.
//
// What ACTS wants.
//
// `state.jacTransport` is not the Jacobian of the surface-to-surface map. The
// contract is written out in `Acts/Propagator/detail/JacobianEngine.hpp:55`:
//
//     jac(locA->locB) = jac(gloB->locB) (1 + pathCorrection(gloB))
//                       jacTransport(gloA->gloB) jac(locA->gloA)
//
// so `jacTransport` holds the transport with the destination surface not
// differentiated through, at fixed 3D path length, and the surface constraint
// arrives separately as `(I + t p')` built from
//
//     t = state.derivative        d(free)/d(path), set by the stepper
//     p = freeToPathDerivative    d(path)/d(free), set by ACTS from the surface
//
// This is the opposite of the bound picture the rest of this project works in,
// where the constraint is inside F. Getting it backwards produces a matrix that
// is the right size, has plausible entries, and double-counts the constraint.
//
// The block structure is asserted, not assumed (`EigenStepper.ipp:338`):
//
//     jacTransport = [ I4   J12 ]     4x4 blocks over (pos, time), (dir, q/p)
//                    [ 04   J22 ]
//
// The unconstrained helix Jacobian satisfies both blocks exactly, to 0.0 and
// not to 1e-16. At fixed path length the endpoint moves rigidly with the source
// position, and in a uniform field the outgoing direction does not depend on the
// source position at all. That the assert holds exactly is the strongest signal
// available that this correspondence is the right one.
//
// Two parameterisation facts.
//
// ACTS's path is the 3D arc length; this project's `s` is the transverse one.
// They differ by pt/|p|, which is a function of the state, so the Jacobian at
// fixed path is a different matrix in the two. The difference is rank one along
// the tangent and `p.t = -1` exactly, so `(I + t p')` annihilates it and the
// bound 5x5 comes out the same either way. The 3D one is written anyway:
// `derivative` and `pathAccumulated` are 3D everywhere else in ACTS and the
// block assert is stated for it.
//
// The tangent `t` must come from the trajectory being differentiated. This
// project's helix uses 0.3 GeV/(T m); rederiving d(dir)/d(path) from the Lorentz
// force with 0.299792458 disagrees by 7e-4 and moves the worst-case
// correspondence from 3e-15 to 6e-7, which is small enough to read as rounding
// and is not.
//
// What is deliberately not in here.
//
// dg_theta/dx. F is the helix Jacobian alone, which is the approximate F that
// is measured and shipped: the exact one costs 30,348 multiply-adds
// against 5,668 and 2637 ns against 492, for 9% fewer holes and fakes. Because
// the filter updates at every layer the incoming covariance is repeatedly
// squashed, so how accurately it was carried matters less than how much noise
// was added. The switch is not offered here; if the exact F is ever wanted it is
// a second function, not a flag on this one.
//
// Units are ACTS's already: mm = 1, GeV = 1, e = 1, and with c = 1 a time is a
// length. No conversion appears below and none is missing.
#pragma once

#include <Eigen/Dense>

#include <cmath>

#include "LearnedTransport.hpp"

namespace collider_ml {

using Mat66 = Eigen::Matrix<double, 6, 6>;
using Mat86 = Eigen::Matrix<double, 8, 6>;
using Mat68 = Eigen::Matrix<double, 6, 8>;
using Mat88 = Eigen::Matrix<double, 8, 8>;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Vec8 = Eigen::Matrix<double, 8, 1>;

/// The free layout, `Acts/Definitions/TrackParametrization.hpp`. Written out
/// rather than included so this header stays ACTS-free and testable alone.
inline constexpr int kFPos = 0;
inline constexpr int kFTime = 3;
inline constexpr int kFDir = 4;
inline constexpr int kFQop = 7;

/// Everything one jump owes the covariance engine.
struct JumpJacobian {
  Mat88 D = Mat88::Identity();  ///< state.jacTransport <- D * jacTransport
  Vec8 t = Vec8::Zero();        ///< state.derivative
  double path3d = 0.0;          ///< 3D arc length, for pathAccumulated
  double dt = 0.0;              ///< time of flight, for pars[eFreeTime]
};

namespace detail {

/// d(free) / d(x, y, z, px, py, pz), 8x6.
///
/// The time row is zero. Time is not a function of the spatial state, it is an
/// independent coordinate, and the transport's dependence on it is put in by
/// `jumpJacobian` below rather than smuggled in here.
inline Mat86 dFreeDGlobal(const Vec3& mom, double q) {
  const double p = mom.norm();
  const Vec3 d = mom / p;
  Mat86 A = Mat86::Zero();
  A.block<3, 3>(kFPos, 0).setIdentity();
  A.block<3, 3>(kFDir, 3) =
      (Eigen::Matrix3d::Identity() - d * d.transpose()) / p;
  A.block<1, 3>(kFQop, 3) = -(q / (p * p)) * d.transpose();
  return A;
}

/// d(x, y, z, px, py, pz) / d(free), 6x8.
inline Mat68 dGlobalDFree(const Vec3& mom, double q) {
  const double p = mom.norm();
  const Vec3 d = mom / p;
  Mat68 B = Mat68::Zero();
  B.block<3, 3>(0, kFPos).setIdentity();
  B.block<3, 3>(3, kFDir) = p * Eigen::Matrix3d::Identity();
  B.block<3, 1>(3, kFQop) = (-q * p * p) * d;
  return B;
}

/// d ln(pt / |p|) / d(pos, mom). Position entries are zero.
///
/// s_transverse = path3d * pt/|p|, so this one row is the entire difference
/// between holding one path variable fixed and holding the other fixed.
inline Vec6 dLogPtOverP(const Vec3& mom) {
  const double px = mom.x(), py = mom.y(), pz = mom.z();
  const double pt = std::hypot(px, py);
  const double pt2 = pt * pt;
  const double p2 = pt2 + pz * pz;
  Vec6 w = Vec6::Zero();
  w(3) = px / pt2 - px / p2;
  w(4) = py / pt2 - py / p2;
  w(5) = -pz / p2;
  return w;
}

/// d(pos, mom)_out / d(pos, mom)_in at fixed TRANSVERSE arc length, plus the
/// trajectory tangent d(out)/ds in the same parameterisation.
///
/// Same expressions as `jacobian_helix.helix_jacobian_no_constraint`, which is
/// the piece of that file the surface constraint is later added to. Here the
/// constraint is never added, because ACTS applies it itself.
inline void helixJacobianFixedS(const Vec3& mom0, double q, double bz, double s,
                                Mat66* K, Vec6* dds) {
  const double px = mom0.x(), py = mom0.y(), pz = mom0.z();
  const double pt = std::hypot(px, py);
  const double pt2 = pt * pt;
  const double phi0 = std::atan2(py, px);
  const double kappa = -0.3 * q * bz / pt * kMM;
  const double a = 1.0 / kappa;
  const double phi = phi0 + kappa * s;
  const double sp = std::sin(phi), cp = std::cos(phi);
  const double sp0 = std::sin(phi0), cp0 = std::cos(phi0);

  const double dpt[2] = {px / pt, py / pt};
  const double dphi0[2] = {-py / pt2, px / pt2};
  const double dkappa[2] = {-(kappa / pt) * dpt[0], -(kappa / pt) * dpt[1]};
  const double da[2] = {-(a * a) * dkappa[0], -(a * a) * dkappa[1]};
  const double dphi[2] = {dphi0[0] + s * dkappa[0], dphi0[1] + s * dkappa[1]};

  K->setZero();
  (*K)(0, 0) = (*K)(1, 1) = (*K)(2, 2) = 1.0;
  for (int k = 0; k < 2; ++k) {
    const int col = 3 + k;  // px, py. pz is handled separately below
    (*K)(0, col) = da[k] * (sp - sp0) + a * (cp * dphi[k] - cp0 * dphi0[k]);
    (*K)(1, col) = -(da[k] * (cp - cp0) + a * (-sp * dphi[k] + sp0 * dphi0[k]));
    (*K)(2, col) = -pz * s * dpt[k] / pt2;
    (*K)(3, col) = dpt[k] * cp - pt * sp * dphi[k];
    (*K)(4, col) = dpt[k] * sp + pt * cp * dphi[k];
  }
  (*K)(2, 5) = s / pt;  // z wrt pz
  (*K)(5, 5) = 1.0;     // pz is conserved

  dds->setZero();
  (*dds)(0) = cp;
  (*dds)(1) = sp;
  (*dds)(2) = pz / pt;
  (*dds)(3) = -pt * sp * kappa;
  (*dds)(4) = pt * cp * kappa;
}

}  // namespace detail

/// The Jacobian of one solved jump.
///
/// `outMom` is where the free change of variables at the far end is evaluated.
/// Two points are defensible and they differ by the size of the correction
/// itself: the helix endpoint, which is what `jacobian_free.py` uses because it
/// checks against a pure-helix F, and the corrected endpoint, which is the state
/// the covariance actually describes. The caller chooses; the stepper passes the
/// corrected one and the test passes the helix one. Either way the larger
/// approximation by far is dropping dg_theta/dx, which is deliberate and
/// measured.
///
/// `mass` is the particle hypothesis's, and enters only the time row. In ACTS
/// the same quantity appears as dtds = sqrt(1 + m^2/p^2)
/// (`EigenStepperDefaultExtension.hpp:120`), which is E/|p|.
inline JumpJacobian jumpJacobian(const Jump& j, const Vec3& outMom, double s,
                                 double mass) {
  Mat66 K;
  Vec6 dds;
  // The core's own field, which is `kBHelix` unless the jump asks for the
  // map value at its source. `helixJacobianFixedS` already took `bz` as an
  // argument, so this site needed no new expression.
  detail::helixJacobianFixedS(j.mom, j.q, detail::coreBz(j), s, &K, &dds);

  // transverse arc length -> 3D path length: one rank-one term along the
  // tangent, and the only place the two parameterisations differ.
  const Mat66 Kp = K + dds * (s * detail::dLogPtOverP(j.mom)).transpose();

  const Mat86 A = detail::dFreeDGlobal(outMom, j.q);
  const Mat68 B = detail::dGlobalDFree(j.mom, j.q);

  const double pIn = j.mom.norm();
  const double ptIn = std::hypot(j.mom.x(), j.mom.y());
  const double eIn = std::sqrt(pIn * pIn + mass * mass);

  JumpJacobian r;
  r.D = A * Kp * B;
  r.path3d = s * pIn / ptIn;
  r.dt = r.path3d * eIn / pIn;

  // The time row. T = path3d * E/|p| at fixed path, so only the momentum
  // enters; the diagonal is 1 because a shift of the source clock shifts the
  // destination clock by the same amount.
  Vec6 dT = Vec6::Zero();
  dT.tail<3>() = r.path3d * (j.mom / (eIn * pIn) -
                             (eIn / (pIn * pIn * pIn)) * j.mom);
  r.D.row(kFTime) = dT.transpose() * B;
  r.D(kFTime, kFTime) = 1.0;

  // state.derivative. The trajectory tangent, rescaled from transverse to 3D
  // path by pt/|p|, and not re-derived from the field. See the header.
  const double pOut = outMom.norm();
  const double ptOut = std::hypot(outMom.x(), outMom.y());
  r.t = A * (dds * (ptOut / pOut));
  r.t(kFTime) = std::sqrt(pOut * pOut + mass * mass) / pOut;
  return r;
}

}  // namespace collider_ml
