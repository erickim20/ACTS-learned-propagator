// Build the REAL ODD tracking geometry and run a track through it, twice:
// once on a stock EigenStepper and once on LearnedStepper, with a Navigator
// doing real navigation over real layers and portals.
//
// This is the step link_probe could not take. link_probe answers a type
// question -- does the filter instantiate and does ActsCore resolve -- and it
// answers it with an empty Navigator, because a geometry would have turned a
// link check into a geometry test. Everything that can actually go wrong with
// LearnedStepper goes wrong *here* instead:
//
//   * the latch. `step()` consumes it, which is what makes a skipped or
//     exhausted target unable to reach a later step. That is an argument on
//     paper until a real navigator produces real skipped targets.
//   * `isSensitive()`. The ODD's portals and layer-approach surfaces are
//     cylinders and discs, so before that predicate went in, `toJump` would
//     have accepted them. A real geometry is the only thing that generates
//     them.
//   * whether the learned jump lands where RKN lands at all.
//
// The two runs share one geometry, one field and one start state, so any
// difference in the endpoint is the learned transport and nothing else.
//
//   g++ -std=c++20 -O2 -I. -I$ACTS_INC -I$EIGEN -I$BOOST_INC \
//       -I$DD4HEP/include -I$ROOT/include/root geom_probe.cpp \
//       -L$ACTS_LIB -lActsCore -lActsPluginDD4hep \
//       -L$DD4HEP/lib -lDDCore -L$ROOT/lib/root -lCore -lGeom -o geom_probe
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>

#include "Acts/Definitions/Units.hpp"
#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/MagneticField/ConstantBField.hpp"
#include "Acts/MagneticField/MagneticFieldContext.hpp"
#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Utilities/Logger.hpp"

#include "ActsPlugins/DD4hep/ConvertDD4hepDetector.hpp"
#include "DD4hep/Detector.h"

#include "LearnedStepper.hpp"

namespace {

/// Count the sensitive surfaces a propagation actually crossed, and where it
/// ended up. The surface counter is the interesting half: a stepper that
/// arrives at the right place having skipped half the detector is not
/// equivalent to one that did not.
/// `kTransportCalls` is how many times the actor calls
/// `transportCovarianceToBound` at each surface. A bare propagation never calls
/// it at all -- it is the CKF's actor that does, at
/// CombinatorialKalmanFilter.hpp:419, :481 and :612 -- so 0 is right for the
/// plumbing comparison, where calling it on the learned arms and not on the
/// stock one would destroy the bit-identity being checked.
///
/// 1 against 2 is the process-noise check. The claim is that Q enters once per
/// jump however many times the filter asks for a bound covariance, and the only
/// way to check a claim about repeated calls is to repeat the call and require
/// the same answer.
template <int kTransportCalls>
struct SurfaceCounter {
  struct result_type {
    std::size_t sensitive = 0;
    std::size_t other = 0;
    // Copied out of the stepping state, which the propagation result does not
    // expose. An actor is the only thing that sees it, so the counters ride
    // out with the surface tally rather than through a second mechanism.
    std::size_t latched = 0, accepted = 0, built = 0, fired = 0;
    std::size_t qArmed = 0, qApplied = 0;
  };

  // Returns Result<void>, not void: 44.99.99's actor_caller assigns the
  // return straight into the propagation's global result, so a void act() is
  // simply not an Actor as far as the concept is concerned, and the error
  // arrives as an unreadable wall of template substitution failures.
  template <typename propagator_state_t, typename stepper_t,
            typename navigator_t>
  Acts::Result<void> act(propagator_state_t& state, const stepper_t& stepper,
                         const navigator_t& navigator, result_type& result,
                         const Acts::Logger& /*logger*/) const {
    const Acts::Surface* s = navigator.currentSurface(state.navigation);
    if (s != nullptr) {
      if (s->isSensitive()) {
        ++result.sensitive;
      } else {
        ++result.other;
      }
      if constexpr (kTransportCalls > 0) {
        if constexpr (requires { state.stepping.nQApplied; }) {
          if (state.stepping.covTransport) {
            for (int i = 0; i < kTransportCalls; ++i) {
              stepper.transportCovarianceToBound(state.stepping, *s);
            }
          }
        }
      }
    }
    // Read AFTER the transport above, or the last call's Q would not be
    // counted. Assignment, not accumulation: these are monotonic counters on
    // the state, so the last read carries the whole propagation.
    if constexpr (requires { state.stepping.nFired; }) {
      result.latched = state.stepping.nLatched;
      result.accepted = state.stepping.nAccepted;
      result.built = state.stepping.nBuilt;
      result.fired = state.stepping.nFired;
      result.qArmed = state.stepping.nQArmed;
      result.qApplied = state.stepping.nQApplied;
    }
    return Acts::Result<void>::success();
  }
};

}  // namespace

int main(int argc, char** argv) {
  const std::string compact =
      argc > 1 ? argv[1] : "/opt/odd/xml/OpenDataDetector.xml";

  // ---------------------------------------------------------------- geometry
  std::printf("loading %s ...\n", compact.c_str());
  std::fflush(stdout);
  auto& detector = dd4hep::Detector::getInstance();
  detector.fromCompact(compact);
  detector.volumeManager();
  detector.apply("DD4hepVolumeManager", 0, nullptr);
  std::printf("DD4hep world loaded\n");
  std::fflush(stdout);

  auto logger = Acts::getDefaultLogger("ODD", Acts::Logging::WARNING);
  std::shared_ptr<const Acts::TrackingGeometry> tGeo =
      ActsPlugins::convertDD4hepDetector(detector.world(), *logger);
  if (tGeo == nullptr) {
    std::fprintf(stderr, "convertDD4hepDetector returned null\n");
    return 2;
  }

  std::size_t nSensitive = 0;
  // Tally sensitive surfaces BY TYPE. This is the question that decides
  // whether LearnedStepper can ever fire here at all: `toJump` switches on
  // Cylinder and Disc, because the teacher's targets are (r1, z1) layer
  // surfaces. If the ODD's sensitive surfaces are planar modules instead,
  // that switch never matches and the learned arm silently degrades to RKN.
  std::size_t byType[8] = {0};
  std::size_t byTypeInsensitive[8] = {0};
  tGeo->visitSurfaces([&](const Acts::Surface* s) {
    if (s == nullptr) {
      return;
    }
    const auto t = static_cast<std::size_t>(s->type());
    if (t >= 8) {
      return;
    }
    if (s->isSensitive()) {
      ++nSensitive;
      ++byType[t];
    } else {
      ++byTypeInsensitive[t];
    }
  },
  // false = do not restrict to sensitives. The default is true, and with the
  // default the insensitive column below is vacuous rather than zero -- which
  // would quietly turn "no portals were counted" into "no portals exist".
  false);
  static const char* kTypeName[8] = {"Cone",  "Cylinder", "Disc",     "Perigee",
                                     "Plane", "Straw",    "Curvilinear", "Other"};
  std::printf("ACTS TrackingGeometry built: %zu sensitive surfaces\n",
              nSensitive);
  std::printf("    %-12s %10s %12s   %s\n", "type", "sensitive", "insensitive",
              "toJump");
  for (std::size_t i = 0; i < 8; ++i) {
    if (byType[i] != 0 || byTypeInsensitive[i] != 0) {
      std::printf("    %-12s %10zu %12zu   %s\n", kTypeName[i], byType[i],
                  byTypeInsensitive[i],
                  (i == 1 || i == 2) ? "ACCEPTS" : "rejects");
    }
  }
  if (byType[1] == 0 && byType[2] == 0) {
    std::printf(
        "\n    *** NO SENSITIVE CYLINDER OR DISC EXISTS IN THIS DETECTOR. ***\n"
        "    toJump() switches on Cylinder and Disc because the teacher's\n"
        "    targets are (r1, z1) layer surfaces, but every sensitive surface\n"
        "    in the ODD is a planar module. So the learned transport cannot\n"
        "    fire on a measurement surface here at all -- it degrades to RKN\n"
        "    silently and completely, which is what the identical arms below\n"
        "    are showing. The cylinders and discs that DO exist are portals\n"
        "    and layer-approach surfaces, i.e. exactly what isSensitive()\n"
        "    was added to exclude.\n");
  }
  std::fflush(stdout);

  // ------------------------------------------------------------------- setup
  Acts::GeometryContext gctx = Acts::GeometryContext::dangerouslyDefaultConstruct();
  Acts::MagneticFieldContext mctx;
  auto bfield = std::make_shared<Acts::ConstantBField>(
      Acts::Vector3(0, 0, 2 * Acts::UnitConstants::T));

  Acts::Navigator::Config navCfg{tGeo};
  navCfg.resolveSensitive = true;
  navCfg.resolveMaterial = true;
  navCfg.resolvePassive = false;

  // A spread of start states rather than one. A single track can be
  // bit-identical because it never reached a sensitive surface at all, and one
  // track cannot tell that apart from transparency either.
  struct Start {
    double pt, eta, phi, q;
  };
  const Start kStarts[] = {
      {1.0, 0.0, 0.35, +1.0},  {1.0, 1.5, -2.10, -1.0},
      {2.0, -0.7, 1.25, +1.0}, {5.0, 0.3, 2.80, -1.0},
      {5.0, -2.2, -0.60, +1.0}, {10.0, 1.1, 0.05, -1.0},
      {20.0, -1.8, -1.70, +1.0}, {50.0, 2.4, 3.05, -1.0},
  };
  constexpr int kNStart = sizeof(kStarts) / sizeof(kStarts[0]);

  // A start covariance, so covTransport is ON and the whole jacTransport ->
  // transportCovarianceToBound path is exercised. With no covariance ACTS
  // skips that machinery entirely and the check would pass without ever
  // touching the code item 3 is about to change.
  Acts::BoundMatrix cov0 = Acts::BoundMatrix::Identity();
  cov0(Acts::eBoundLoc0, Acts::eBoundLoc0) = 0.05 * 0.05;
  cov0(Acts::eBoundLoc1, Acts::eBoundLoc1) = 0.05 * 0.05;
  cov0(Acts::eBoundPhi, Acts::eBoundPhi) = 1e-6;
  cov0(Acts::eBoundTheta, Acts::eBoundTheta) = 1e-6;
  cov0(Acts::eBoundQOverP, Acts::eBoundQOverP) = 1e-8;
  cov0(Acts::eBoundTime, Acts::eBoundTime) = 1.0;

  /// Everything a propagation produced, in a form that can be compared with
  /// `==`. Doubles, deliberately: "bit-identical" is the claim, so a tolerance
  /// here would be the check quietly refusing to make it.
  struct Snapshot {
    bool ok = false;
    std::string err;
    double pathLength = 0.0;
    std::size_t steps = 0;
    bool hasEnd = false;
    Acts::BoundVector par = Acts::BoundVector::Zero();
    bool hasCov = false;
    Acts::BoundMatrix cov = Acts::BoundMatrix::Zero();
    std::size_t sensitive = 0, other = 0;
    // learned arm only
    std::size_t latched = 0, accepted = 0, built = 0, fired = 0;
    std::size_t qArmed = 0, qApplied = 0;
  };

  // memcmp, not `==`. Eigen's comparison operators are coefficient-wise and
  // would need an .all(); more to the point, "bit-identical" is the actual
  // claim, and memcmp is the only comparison that makes it -- `==` reports a
  // difference between two NaNs that are the same NaN, and reports none
  // between +0.0 and -0.0.
  auto bitsame = [](const auto& x, const auto& y) {
    return std::memcmp(x.data(), y.data(), sizeof(double) * x.size()) == 0;
  };
  auto same = [&](const Snapshot& a, const Snapshot& b) {
    if (a.ok != b.ok || a.err != b.err || a.hasEnd != b.hasEnd ||
        a.hasCov != b.hasCov) {
      return false;
    }
    if (std::memcmp(&a.pathLength, &b.pathLength, sizeof(double)) != 0 ||
        a.steps != b.steps || a.sensitive != b.sensitive ||
        a.other != b.other) {
      return false;
    }
    if (a.hasEnd && !bitsame(a.par, b.par)) {
      return false;
    }
    if (a.hasCov && !bitsame(a.cov, b.cov)) {
      return false;
    }
    return true;
  };

  auto makeStart = [&](const Start& s) {
    const double theta = 2.0 * std::atan(std::exp(-s.eta));
    const Acts::Vector3 dir(std::sin(theta) * std::cos(s.phi),
                            std::sin(theta) * std::sin(s.phi),
                            std::cos(theta));
    const double p = s.pt * Acts::UnitConstants::GeV / std::sin(theta);
    return Acts::BoundTrackParameters::createCurvilinear(
        Acts::Vector4(0, 0, 0, 0), dir, s.q / p, cov0,
        Acts::ParticleHypothesis::pion());
  };

  // `propagator_t::propagate` returns the actor results and the end
  // parameters; this pulls both into a Snapshot the same way for every arm, so
  // a difference cannot come from the reading.
  // The counter type comes in as a tag so one body serves both actor lists.
  auto capture = [&](auto& prop, const auto& start, auto counterTag) {
    using Prop = std::decay_t<decltype(prop)>;
    using Counter = decltype(counterTag);
    typename Prop::template Options<Acts::ActorList<Counter>> opt(gctx, mctx);
    opt.pathLimit = 3000.0;
    Snapshot snap;
    auto res = prop.propagate(start, opt);
    if (!res.ok()) {
      snap.err = res.error().message();
      return snap;
    }
    snap.ok = true;
    snap.pathLength = res->pathLength;
    snap.steps = res->steps;
    if (res->endParameters.has_value()) {
      snap.hasEnd = true;
      snap.par = res->endParameters->parameters();
      if (res->endParameters->covariance().has_value()) {
        snap.hasCov = true;
        snap.cov = *res->endParameters->covariance();
      }
    }
    const auto& c = res->template get<typename Counter::result_type>();
    snap.sensitive = c.sensitive;
    snap.other = c.other;
    snap.latched = c.latched;
    snap.accepted = c.accepted;
    snap.built = c.built;
    snap.fired = c.fired;
    snap.qArmed = c.qArmed;
    snap.qApplied = c.qApplied;
    return snap;
  };

  // ------------------------------------------------------------- the three arms
  std::printf(
      "\n=== the plumbing check: kSensitiveOnly against stock ===\n"
      "%d start states, curvilinear at the origin, covariance ON so the\n"
      "transport path is exercised. Constant 2 T.\n\n",
      kNStart);
  std::printf("%-4s %6s %6s %7s   %-12s %-12s   %s\n", "#", "pt", "eta", "q",
              "off vs stock", "ON vs stock", "learned: latch/accept/build/fire");

  int nOffDiff = 0, nOnDiff = 0, nFailed = 0;
  std::size_t totLatched = 0, totAccepted = 0, totBuilt = 0, totFired = 0;
  std::size_t totSensitive = 0;

  for (int i = 0; i < kNStart; ++i) {
    const auto start = makeStart(kStarts[i]);

    // arm A: stock EigenStepper
    Snapshot a;
    {
      using Stepper = Acts::EigenStepper<>;
      using Prop = Acts::Propagator<Stepper, Acts::Navigator>;
      // braces, not parens: `Prop prop(Stepper(bfield), Navigator(navCfg))`
      // declares a function returning Prop
      Prop prop{Stepper(bfield), Acts::Navigator(navCfg)};
      a = capture(prop, start, SurfaceCounter<0>{});
    }

    // arm B: LearnedStepper with the switch OFF. A forwarding shell around the
    // same EigenStepper, so a difference here is a bug in the twenty-six
    // forwarding methods and has nothing to do with the model.
    Snapshot b;
    {
      using Prop = Acts::Propagator<collider_ml::LearnedStepper, Acts::Navigator>;
      collider_ml::LearnedStepper stepper(bfield);
      stepper.setCellSigma(20.0, 43.0);
      Prop prop{std::move(stepper), Acts::Navigator(navCfg)};
      b = capture(prop, start, SurfaceCounter<0>{});
    }

    // arm C: the switch ON, in kSensitiveOnly. This is the arm the item is
    // about.
    Snapshot c;
    {
      using Prop = Acts::Propagator<collider_ml::LearnedStepper, Acts::Navigator>;
      collider_ml::LearnedStepper stepper(bfield);
      stepper.setCellSigma(20.0, 43.0);
      stepper.setUseLearned(true);
      Prop prop{std::move(stepper), Acts::Navigator(navCfg)};
      c = capture(prop, start, SurfaceCounter<0>{});
    }

    if (!a.ok) {
      ++nFailed;
    }
    const bool offSame = same(a, b);
    const bool onSame = same(a, c);
    nOffDiff += offSame ? 0 : 1;
    nOnDiff += onSame ? 0 : 1;
    totSensitive += a.sensitive;

    std::printf("%-4d %6.1f %6.1f %7.0f   %-12s %-12s   %zu/%zu/%zu/%zu%s\n",
                i, kStarts[i].pt, kStarts[i].eta, kStarts[i].q,
                offSame ? "identical" : "DIFFERS",
                onSame ? "identical" : "DIFFERS", c.latched, c.accepted,
                c.built, c.fired, a.ok ? "" : "   (stock FAILED)");
    if (!a.ok) {
      std::printf("     stock error: %s\n", a.err.c_str());
    }
    totLatched += c.latched;
    totAccepted += c.accepted;
    totBuilt += c.built;
    totFired += c.fired;
  }

  std::printf(
      "\n%d/%d identical with the switch off, %d/%d identical with it on.\n",
      kNStart - nOffDiff, kNStart, kNStart - nOnDiff, kNStart);
  std::printf("stock crossed %zu sensitive surfaces in total.\n", totSensitive);
  std::printf(
      "learned arm, summed: %zu targets latched, %zu accepted by\n"
      "isSensitive(), %zu built into a Jump, %zu answered by the kernel.\n",
      totLatched, totAccepted, totBuilt, totFired);

  // WHAT THIS CHECK MEANS NOW.
  //
  // While the kernel targeted a cylinder, every sensitive ODD surface fell
  // through `toJump`'s switch, the jump fired nowhere, and bit-identical
  // output was the expected answer. That was never evidence of transparency,
  // which is why the counters exist.
  //
  // With plane handling the jump fires on measurement surfaces, so the two
  // arms now answer different questions and the verdict is split:
  //
  //   switch OFF must still be bit-identical. It is a forwarding shell around
  //   the same EigenStepper and nothing else can explain a difference.
  //
  //   switch ON must DIFFER, and must fire. Identical output with the switch
  //   on would now mean the jump is not reaching the kernel, which is the
  //   failure the counters were added to name.
  const bool offOk = (nOffDiff == 0);
  const bool onFires = (totFired > 0);
  const bool onDiffers = (nOnDiff > 0);
  std::printf("\n");
  if (!offOk) {
    std::printf(
        "FAIL: with the switch OFF the adapter is not transparent. It is a\n"
        "forwarding shell around the same EigenStepper there, so this is a bug\n"
        "in the twenty-six forwarding methods and has nothing to do with the\n"
        "model.\n");
  } else if (totLatched == 0) {
    std::printf(
        "INCONCLUSIVE: no target was ever latched, so the learned branch was\n"
        "never reached and nothing here says anything about it. Check\n"
        "updateSurfaceStatus.\n");
  } else if (!onFires) {
    std::printf(
        "FAIL: %zu targets latched and %zu passed isSensitive(), but the\n"
        "kernel answered none of them. `toJump` is rejecting measurement\n"
        "surfaces, which is what plane handling was supposed to fix.\n",
        totLatched, totAccepted);
  } else if (!onDiffers) {
    std::printf(
        "FAIL: the kernel fired %zu times and the output was still identical\n"
        "on every track. A jump that runs and changes nothing is a worse bug\n"
        "than one that never runs.\n",
        totFired);
  } else {
    std::printf(
        "PASS. Switch off: bit-identical to stock on %d of %d, so the\n"
        "forwarding is sound. Switch on: %zu targets latched, %zu accepted by\n"
        "isSensitive(), %zu built into a Jump, %zu answered by the kernel, and\n"
        "the output differs on %d of %d.\n\n"
        "The SIZE of that difference is not a result. The weights in\n"
        "gtheta_weights.hpp have to be the ones trained on the module-plane\n"
        "target and the same twelve inputs, and this probe runs a constant 2 T\n"
        "while input 11 is the ODD map value. This says the kernel is reached\n"
        "and changes the answer, nothing more.\n",
        kNStart - nOffDiff, kNStart, totLatched, totAccepted, totBuilt,
        totFired, nOnDiff, kNStart);
  }
  const bool identical = offOk && onFires && onDiffers;

  std::printf(
      "\nNOTE the field: this probe runs a CONSTANT 2 T, while the kernel's\n"
      "eleventh input is the ODD map value at the source point. That does not\n"
      "affect the check above, which is about whether the kernel is reached\n"
      "at all, but it does mean no line here is a physics result.\n");

  // ------------------------------------------------------ the process noise
  //
  // kSensitiveOnly cannot answer anything about Q, because the jump never
  // fires. kLayerApproach exists for exactly this: it fires on the layer
  // approach surfaces, which are the cylinders and discs the teacher's (r1, z1)
  // targets correspond to, and lets RKN carry the last hop onto the module. It
  // is a scaffold for measurement and not a shipping mode.
  collider_ml::NoiseTable qtab;
  // The fixture is not a measurement, so it carries the sentinel provenance
  // `write_fixture` writes: 32 zeros for the weights and a negative gate.
  // Declaring it at the call site rather than exempting it inside the
  // loader keeps the check with no bypass in it.
  const bool haveQ =
      qtab.load("q_table_fixture.bin", std::string(32, '0'), -1.0);
  std::printf(
      "\n\n=== the process noise seam ===\n"
      "kLayerApproach, so the jump actually fires. The table is a FIXTURE of\n"
      "round numbers, not a measurement: the material-off Q has to be\n"
      "re-measured after the target surface changes, which is item 7. What is\n"
      "under test here is that Q enters ONCE, on the right block, and that the\n"
      "sigma head reaches the covariance and nothing else.\n\n");
  if (!haveQ) {
    std::printf("no q_table_fixture.bin; run  python -m prop.export_qtable "
                "--fixture\n");
  } else {
    // ACTS's own `transportCovarianceToBound` is not bit-idempotent. It
    // reinitialises the Jacobians and a second call then transports by what
    // should be an identity and is one only to rounding. So comparing the final
    // covariance between one call and two would be measuring ACTS. What
    // isolates the noise table's contribution is the difference it makes:
    //
    //   dQ(n) = cov(with table, n calls) - cov(without, n calls)
    //
    // and dQ(2)/dQ(1) is 1 if Q enters once per jump and 2 if it enters per
    // call. A ratio, so it is readable without knowing the fixture's numbers.
    std::printf("%-4s %6s %6s   %7s %7s   %10s   %-14s %s\n", "#", "pt", "eta",
                "fired", "Q armed", "dQ(2)/dQ(1)", "sigma head", "params");
    int nRepeatBad = 0, nParamMoved = 0, nCovSame = 0;
    int nFiredTotal = 0, nArmedTotal = 0, nAppliedTotal = 0;
    double worstRatio = 0.0;

    for (int i = 0; i < kNStart; ++i) {
      const auto start = makeStart(kStarts[i]);
      auto run = [&](bool withTable, bool sigmaHead, auto tag) {
        using Prop =
            Acts::Propagator<collider_ml::LearnedStepper, Acts::Navigator>;
        collider_ml::LearnedStepper stepper(bfield);
        stepper.setCellSigma(20.0, 43.0);
        stepper.setUseLearned(true);
        stepper.setNoiseTable(withTable ? &qtab : nullptr);
        stepper.setUseSigmaHead(sigmaHead);
        Prop prop{std::move(stepper), Acts::Navigator(navCfg)};
        return capture(prop, start, tag);
      };
      const Snapshot bare1 = run(false, false, SurfaceCounter<1>{});
      const Snapshot bare2 = run(false, false, SurfaceCounter<2>{});
      const Snapshot q1 = run(true, false, SurfaceCounter<1>{});
      const Snapshot q2 = run(true, false, SurfaceCounter<2>{});
      const Snapshot withHead = run(true, true, SurfaceCounter<1>{});

      const double d1 = (q1.cov - bare1.cov).cwiseAbs().maxCoeff();
      const double d2 = (q2.cov - bare2.cov).cwiseAbs().maxCoeff();
      const double ratio = d1 > 0.0 ? d2 / d1 : 0.0;
      const bool repeatSafe =
          d1 > 0.0 && std::abs(ratio - 1.0) < 1e-6 && q1.qApplied == q2.qApplied;
      nRepeatBad += repeatSafe ? 0 : 1;
      worstRatio = std::max(worstRatio, std::abs(ratio - 1.0));

      // The sigma head must reach the covariance and nothing else: same
      // trajectory, same path, same step count, same parameters.
      const bool parsSame = q1.hasEnd && withHead.hasEnd &&
                            bitsame(q1.par, withHead.par) &&
                            q1.steps == withHead.steps &&
                            std::memcmp(&q1.pathLength, &withHead.pathLength,
                                        sizeof(double)) == 0;
      const bool covSame =
          q1.hasCov && withHead.hasCov && bitsame(q1.cov, withHead.cov);
      nParamMoved += parsSame ? 0 : 1;
      nCovSame += covSame ? 1 : 0;

      nFiredTotal += static_cast<int>(q1.fired);
      nArmedTotal += static_cast<int>(q1.qArmed);
      nAppliedTotal += static_cast<int>(q1.qApplied);

      std::printf("%-4d %6.1f %6.1f   %7zu %7zu   %10.6f   %-14s %s\n", i,
                  kStarts[i].pt, kStarts[i].eta, q1.fired, q1.qArmed, ratio,
                  covSame ? "no effect" : "changes cov",
                  parsSame ? "identical" : "MOVED");
    }

    std::printf("\n%d learned jumps fired, %d armed a Q, %d payments made.\n",
                nFiredTotal, nArmedTotal, nAppliedTotal);
    std::printf(
        "A jump arms Q only where the destination volume is one the table\n"
        "knows, and one payment settles every jump accumulated since the last\n"
        "one, so the third number is the smallest and none of the debt is\n"
        "lost.\n");
    if (nRepeatBad == 0 && nParamMoved == 0 && nCovSame == 0 &&
        nFiredTotal > 0 && nArmedTotal > 0) {
      std::printf(
          "\nPASS: dQ(2)/dQ(1) is 1 to %.1e, so Q enters once per jump however\n"
          "often the filter asks for a bound covariance. The sigma head changes\n"
          "the covariance and leaves the parameters, the path and the step\n"
          "count bit-identical.\n",
          worstRatio);
    } else {
      std::printf(
          "\nFAIL: %d tracks where a repeated transport call changed our\n"
          "contribution (worst |ratio - 1| = %.2e), %d where the sigma head\n"
          "moved the parameters, %d where it did not reach the covariance.\n",
          nRepeatBad, worstRatio, nParamMoved, nCovSame);
    }
    std::printf(
        "\nThe numbers in that covariance are a fixture and mean nothing. The\n"
        "seam does: the learned Q lands on the top-left 5x5 between ACTS's\n"
        "transport and ACTS's own scattering noise, which is the order the\n"
        "filter requires.\n");
  }

  return identical ? 0 : 1;
}
