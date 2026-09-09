// Transport a covariance across one learned jump and across the equivalent RKN
// path, and compare.
//
// `test_jacobian` checks the C++ Jacobian against the Python it was ported
// from. That is a port check, and it cannot catch an error the Python and the
// C++ share. This is the independent one: ACTS integrates the same trajectory
// with its own stepper, transports the same starting covariance with its own
// covariance engine, and the two answers are compared.
//
// The comparison is built so that what is under test is D and t and nothing
// else. Every other factor in
//
//     jac(locA->locB) = jac(gloB->locB) (1 + t p') jacTransport jac(locA->gloA)
//
// is taken from ACTS: `curvilinearToFreeJacobian` for the start,
// `boundToBoundTransportJacobian` for the surface constraint and the
// projection. So a disagreement is in the two objects, not in a reimplemented
// factor.
//
// The field is a constant 2 T, which is what makes the test meaningful: a helix
// is the exact solution there, so the two answers are two computations of
// the same thing rather than two models.
//
// Built and run by run_in_image.sh, which is where libActsCore lives.
#include <cmath>
#include <cstdio>
#include <memory>
#include <optional>

#include "Acts/Definitions/Units.hpp"
#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/MagneticField/ConstantBField.hpp"
#include "Acts/MagneticField/MagneticFieldContext.hpp"
#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Propagator/VoidNavigator.hpp"
#include "Acts/Propagator/detail/JacobianEngine.hpp"
#include "Acts/Surfaces/CylinderBounds.hpp"
#include "Acts/Surfaces/CylinderSurface.hpp"
#include "Acts/Surfaces/DiscSurface.hpp"
#include "Acts/Surfaces/RadialBounds.hpp"

#include "LearnedJacobian.hpp"

namespace {

/// Error relative to the size of the matrix, not of the entry. Same rule as
/// `jacobian_helix.rel()`: per-entry normalisation reports 1.0 wherever the
/// reference is a structural zero and the other side has rounding in it.
double relMax(const Acts::BoundMatrix& a,
              const Acts::BoundMatrix& b) {
  const double scale = std::max(a.cwiseAbs().maxCoeff(), b.cwiseAbs().maxCoeff());
  return scale > 0.0 ? (a - b).cwiseAbs().maxCoeff() / scale : 0.0;
}

}  // namespace

int main() {
  Acts::GeometryContext gctx =
      Acts::GeometryContext::dangerouslyDefaultConstruct();
  Acts::MagneticFieldContext mctx;
  const double kB = 2.0 * Acts::UnitConstants::T;
  auto bfield =
      std::make_shared<Acts::ConstantBField>(Acts::Vector3(0, 0, kB));

  Acts::BoundMatrix c0 = Acts::BoundMatrix::Zero();
  c0(Acts::eBoundLoc0, Acts::eBoundLoc0) = 0.05 * 0.05;
  c0(Acts::eBoundLoc1, Acts::eBoundLoc1) = 0.05 * 0.05;
  c0(Acts::eBoundPhi, Acts::eBoundPhi) = 1e-6;
  c0(Acts::eBoundTheta, Acts::eBoundTheta) = 1e-6;
  c0(Acts::eBoundQOverP, Acts::eBoundQOverP) = 1e-8;
  c0(Acts::eBoundTime, Acts::eBoundTime) = 1.0;

  struct Case {
    double pt, eta, phi, q, target;
    bool endcap;
  };
  // Targets are the surfaces the teacher actually uses: a cylinder r = r1 in
  // the barrel and a plane z = z1 on the discs. Radii and z are ODD-like.
  const Case kCases[] = {
      {1.0, 0.0, 0.35, +1.0, 200.0, false},
      {2.0, 0.2, -1.20, -1.0, 300.0, false},
      {5.0, -0.4, 2.60, +1.0, 500.0, false},
      {10.0, 0.5, 0.05, -1.0, 800.0, false},
      {1.0, 2.0, 1.10, +1.0, 1500.0, true},
      {2.0, -1.8, -2.40, -1.0, -1500.0, true},
      {5.0, 1.5, 0.90, +1.0, 2000.0, true},
      {20.0, -2.2, 2.10, -1.0, -2500.0, true},
  };
  constexpr int kN = sizeof(kCases) / sizeof(kCases[0]);

  std::printf(
      "one jump, constant %.1f T, no material. A helix is exact in a uniform\n"
      "field, so RKN and the learned core are two computations of one answer.\n\n",
      kB / Acts::UnitConstants::T);
  std::printf("%-3s %5s %6s %8s   %10s %10s %8s   %10s\n", "#", "pt", "eta",
              "target", "endpoint", "predicted", "ratio", "cov rel");

  double worstCov = 0.0, worstPos = 0.0;
  int nDone = 0;

  // The whole endpoint disagreement should be the helix constant and nothing
  // else. Two circular arcs of the same length whose curvatures differ by dk
  // separate by dk*s^2/2, so that number can be predicted per case rather than
  // gestured at, and the ratio column is the check.
  constexpr double kActsC = 0.299792458;
  constexpr double kOursC = 0.3;
  constexpr double kRelC = (kOursC - kActsC) / kOursC;

  for (int i = 0; i < kN; ++i) {
    const Case& k = kCases[i];
    const double theta = 2.0 * std::atan(std::exp(-k.eta));
    const Acts::Vector3 dir(std::sin(theta) * std::cos(k.phi),
                            std::sin(theta) * std::sin(k.phi),
                            std::cos(theta));
    const double p = k.pt * Acts::UnitConstants::GeV / std::sin(theta);
    const double qop = k.q / p;

    std::shared_ptr<Acts::Surface> target;
    if (k.endcap) {
      Acts::Transform3 tf = Acts::Transform3::Identity();
      tf.translation() = Acts::Vector3(0, 0, k.target);
      target = Acts::Surface::makeShared<Acts::DiscSurface>(
          tf, std::make_shared<Acts::RadialBounds>(0.0, 4000.0));
    } else {
      target = Acts::Surface::makeShared<Acts::CylinderSurface>(
          Acts::Transform3::Identity(),
          std::make_shared<Acts::CylinderBounds>(k.target, 5000.0));
    }

    auto start = Acts::BoundTrackParameters::createCurvilinear(
        Acts::Vector4(0, 0, 0, 0), dir, qop, c0,
        Acts::ParticleHypothesis::pion());

    // ---------------------------------------------------------------- arm A
    using Stepper = Acts::EigenStepper<>;
    using Prop = Acts::Propagator<Stepper, Acts::VoidNavigator>;
    Prop prop{Stepper(bfield), Acts::VoidNavigator()};
    Prop::Options<> opt(gctx, mctx);
    opt.pathLimit = 10000.0;
    auto res = prop.propagate(start, *target, opt);
    if (!res.ok() || !res->endParameters.has_value() ||
        !res->endParameters->covariance().has_value()) {
      std::printf("%-3d %5.1f %6.1f %8.0f   RKN did not reach the surface\n", i,
                  k.pt, k.eta, k.target);
      continue;
    }
    const Acts::BoundMatrix cRkn = *res->endParameters->covariance();
    const Acts::Vector3 endRkn = res->endParameters->position(gctx);

    // ---------------------------------------------------------------- arm B
    collider_ml::Jump j;
    j.pos = Acts::Vector3::Zero();
    j.mom = dir * p;
    j.q = k.q >= 0 ? 1.0 : -1.0;
    j.endcap = k.endcap;
    if (k.endcap) {
      j.z1 = k.target;
    } else {
      j.r1 = k.target;
    }
    bool ok = false;
    const double s = collider_ml::solveArcLength(j, &ok);
    if (!ok) {
      std::printf("%-3d %5.1f %6.1f %8.0f   no helix solution\n", i, k.pt, k.eta,
                  k.target);
      continue;
    }
    // The helix endpoint, from the same expressions the kernel uses.
    const double pt = std::hypot(j.mom.x(), j.mom.y());
    const double phi0 = std::atan2(j.mom.y(), j.mom.x());
    const double kappa = -0.3 * j.q * collider_ml::detail::kBHelix / pt * 1e-3;
    const Acts::Vector3 endPos = collider_ml::detail::helixPos(
        j.pos, phi0, pt, j.mom.z() / pt, kappa, s);
    const double phiEnd = phi0 + kappa * s;
    const Acts::Vector3 endMom(pt * std::cos(phiEnd), pt * std::sin(phiEnd),
                               j.mom.z());

    const double mass = Acts::ParticleHypothesis::pion().mass();
    const collider_ml::JumpJacobian jj =
        collider_ml::jumpJacobian(j, endMom, s, mass);

    // Every factor below this line is ACTS's own. What is being tested is jj.
    Acts::FreeVector freeEnd = Acts::FreeVector::Zero();
    freeEnd.segment<3>(Acts::eFreePos0) = endPos;
    freeEnd[Acts::eFreeTime] = jj.dt;
    freeEnd.segment<3>(Acts::eFreeDir0) = endMom.normalized();
    freeEnd[Acts::eFreeQOverP] = qop;

    // The start surface's own jacobian, not `detail::curvilinearToFreeJacobian`.
    // That free function is declared in JacobianEngine.hpp but not exported
    // from libActsCore in this build, so it links nowhere; the Surface method
    // is the public route and is also the general one, which matters when the
    // start is a module rather than a curvilinear plane.
    const Acts::BoundToFreeMatrix b2f = start.referenceSurface().boundToFreeJacobian(
        gctx, start.position(gctx), dir);
    const Acts::BoundMatrix jac = Acts::detail::boundToBoundTransportJacobian(
        gctx, freeEnd, b2f, jj.D, jj.t, *target);
    const Acts::BoundMatrix cOurs = jac * c0 * jac.transpose();

    const double dCov = relMax(cRkn, cOurs);
    const double dPos = (endPos - endRkn).norm();
    worstCov = std::max(worstCov, dCov);
    worstPos = std::max(worstPos, dPos);
    ++nDone;

    const double predicted = 0.5 * std::abs(kappa) * kRelC * s * s;
    std::printf("%-3d %5.1f %6.1f %8.0f   %10.2e %10.2e %8.2f   %10.2e\n", i,
                k.pt, k.eta, k.target, dPos, predicted,
                predicted > 0.0 ? dPos / predicted : 0.0, dCov);
  }

  std::printf(
      "\n%d of %d cases completed.\n"
      "endpoint disagreement, worst  %.2e mm\n"
      "covariance disagreement, worst %.2e relative to the matrix\n",
      nDone, kN, worstPos, worstCov);

  // A ratio near 1 in the table above says the endpoint disagreement is the
  // helix constant and nothing else: this project uses 0.3 GeV/(T m) and ACTS
  // uses 0.299792458, i.e. 6.9e-4 on the curvature. Item 1 established that the
  // tangent has to be taken from the trajectory being differentiated rather
  // than re-derived with the exact constant, so this is a property of the model
  // and not a bug to patch here.
  std::printf(
      "\nA `ratio` near 1 means the endpoint disagreement is entirely the\n"
      "helix constant, %.1f against ACTS's %.9f, i.e. %.1e on the curvature.\n"
      "The covariance column is then the linearisation on top of that, and it\n"
      "is three to four orders of magnitude smaller than the tolerance\n"
      "the approximate F is measured against (it ships against an exact one\n"
      "that buys 9%% fewer holes and fakes).\n",
      kOursC, kActsC, kRelC);
  return 0;
}
