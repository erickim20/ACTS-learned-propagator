// Does the moved seam's material plan match what the leg actually crosses.
//
//   ./run_in_image.sh --map plan_probe
//
// `cpp/mat_leg.cpp` measured where the material sits along a leg by recording
// every surface a stock propagation makes current. `cpp/nav_walk.hpp` now
// carries a plan, built from the same three geometry calls the walk already
// makes, and `cpp/LearnedStepper.hpp` applies it. Nothing so far says the two
// are the same set.
//
// A CKF run cannot say it either. It reports efficiency, and an over-applied
// slab and a missing one both lower it. So this compares them directly, over
// the same 5,400 start states and the same 50,519 transitions the walk uses,
// with the same stock `Propagator<EigenStepper<>, Navigator>`.
//
// What is compared, per leg, and it is three different questions:
//
//   The set. Which material surfaces the stock navigator stopped at between
//   the two modules, against which ones the plan names. A surface in one and
//   not the other is a miss or an extra, and the two are opposite bugs: a miss
//   under-applies the scattering, an extra applies material that is not there.
//
//   THE THICKNESS. `evaluateMaterialSlab` scales the map's slab by
//   `pathCorrection(gctx, position, direction)` at the point it is given, and
//   that factor is 1/cos of the incidence angle, so it diverges at grazing
//   incidence. Stock evaluates it standing exactly on the surface. The seam
//   evaluates it from wherever the step landed, after re-intersecting. The
//   ratio of the two thicknesses is what says whether that substitution is
//   safe.
//
//   THE MODE. Stock applies `PreUpdate` at `CombinatorialKalmanFilter.hpp:481`
//   and `PostUpdate` at `:612`, which is the whole slab in two calls. The seam
//   makes one `FullUpdate` call. Whether those agree depends on the map's split
//   factor, which is not in the headers, so it is measured here rather than
//   assumed.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <utility>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "Acts/Definitions/Common.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Geometry/Layer.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Geometry/TrackingVolume.hpp"
#include "Acts/MagneticField/MagneticFieldContext.hpp"
#include "Acts/Material/MaterialSlab.hpp"
#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Propagator/detail/PointwiseMaterialInteraction.hpp"
#include "Acts/Surfaces/BoundaryTolerance.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Utilities/Result.hpp"

#include "ActsPlugins/DD4hep/ConvertDD4hepDetector.hpp"
#include "ActsPlugins/DD4hep/DD4hepFieldAdapter.hpp"
#include "ActsPlugins/Root/RootMaterialDecorator.hpp"
#include "DD4hep/Detector.h"

#include "nav_helix.hpp"
#include "nav_walk.hpp"

namespace {

// ---------------------------------------------------------------- the sample
//
// Identical to cpp/nav_truth.cpp and cpp/mat_leg.cpp, so the three address the
// same transitions and the counts can be compared directly.
constexpr int kNPt = 5;
constexpr double kPtEdges[kNPt + 1] = {1.0, 2.0, 4.0, 8.0, 20.0, 50.0};
constexpr int kNEta = 5;
constexpr double kEtaEdges[kNEta] = {0.6, 1.2, 1.8, 2.4, 3.0};
constexpr double kInside[3] = {0.25, 0.5, 0.75};
constexpr int kNPhi = 6;
constexpr double kPhi0 = 0.17;

struct Start {
  double pt = 0.0;
  double eta = 0.0;
  double phi = 0.0;
  double q = 1.0;
  int ptBin = 0;
  int etaBin = 0;
};

struct Arrival {
  const Acts::Surface* surface = nullptr;
  double path = 0.0;
  Acts::Vector3 pos = Acts::Vector3::Zero();
  Acts::Vector3 dir = Acts::Vector3::Zero();
  double pAbs = 0.0;
  double q = 1.0;
  bool sensitive = false;
  bool material = false;
};

/// Every surface the propagation makes current, in order, with the state there.
///
/// `act` returns `Result<void>` and not void: 44.99.99's actor_caller assigns
/// the return into the propagation's global result, so a void `act` fails the
/// Actor concept with a wall of substitution errors.
struct AllArrivalRecorder {
  struct result_type {
    std::vector<Arrival> arrivals;
    const Acts::Surface* last = nullptr;
  };

  template <typename propagator_state_t, typename stepper_t,
            typename navigator_t>
  Acts::Result<void> act(propagator_state_t& state, const stepper_t& stepper,
                         const navigator_t& navigator, result_type& result,
                         const Acts::Logger& /*logger*/) const {
    const Acts::Surface* s = navigator.currentSurface(state.navigation);
    if (s == nullptr || s == result.last) {
      return Acts::Result<void>::success();
    }
    result.last = s;
    Arrival a;
    a.surface = s;
    a.path = state.stepping.pathAccumulated;
    a.pos = stepper.position(state.stepping);
    a.dir = stepper.direction(state.stepping);
    a.pAbs = stepper.absoluteMomentum(state.stepping);
    a.q = stepper.charge(state.stepping) >= 0 ? 1.0 : -1.0;
    a.sensitive = s->isSensitive();
    a.material = s->surfaceMaterial() != nullptr;
    result.arrivals.push_back(a);
    return Acts::Result<void>::success();
  }
};

double quantile(std::vector<double>& v, double p) {
  if (v.empty()) {
    return std::nan("");
  }
  std::sort(v.begin(), v.end());
  return v[static_cast<std::size_t>(p * static_cast<double>(v.size() - 1))];
}

void report(const char* what, std::vector<double> v) {
  const std::size_t n = v.size();
  std::printf("  %-34s n %7zu   p10 %9.4f  p50 %9.4f  p90 %9.4f  p99 %9.4f  "
              "max %9.4f\n",
              what, n, quantile(v, 0.10), quantile(v, 0.50), quantile(v, 0.90),
              quantile(v, 0.99), quantile(v, 1.0));
}

double pct(std::size_t a, std::size_t b) {
  return b == 0 ? 0.0 : 100.0 * static_cast<double>(a) / static_cast<double>(b);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string compact =
      argc > 1 ? argv[1] : "/opt/odd/xml/OpenDataDetector.xml";
  std::printf("compact           %s\n", compact.c_str());
  std::fflush(stdout);

  auto& detector = dd4hep::Detector::getInstance();
  detector.fromCompact(compact);
  detector.volumeManager();
  detector.apply("DD4hepVolumeManager", 0, nullptr);

  auto logger = Acts::getDefaultLogger("ODD", Acts::Logging::WARNING);

  std::string mapFile;
  {
    const std::string tail = "/xml/OpenDataDetector.xml";
    if (compact.size() > tail.size() &&
        compact.compare(compact.size() - tail.size(), tail.size(), tail) == 0) {
      mapFile = compact.substr(0, compact.size() - tail.size()) +
                "/data/odd-material-maps.root";
    }
  }
  std::FILE* probe = mapFile.empty() ? nullptr : std::fopen(mapFile.c_str(), "r");
  if (probe == nullptr) {
    // The decorator argument defaults to nullptr and builds a
    // geometry with no material at all, without warning. Every number below
    // would then be zero for a reason that is not the geometry.
    std::printf("material map NOT FOUND at %s; stopping.\n",
                mapFile.empty() ? "(not derivable)" : mapFile.c_str());
    return 2;
  }
  std::fclose(probe);
  std::shared_ptr<const Acts::IMaterialDecorator> matDec;
  {
    ActsPlugins::RootMaterialDecorator::Config cfg;
    cfg.fileName = mapFile;
    matDec = std::make_shared<const ActsPlugins::RootMaterialDecorator>(
        cfg, Acts::Logging::WARNING);
  }
  std::printf("material map      %s\n", mapFile.c_str());

  std::shared_ptr<const Acts::TrackingGeometry> tGeo =
      ActsPlugins::convertDD4hepDetector(
          detector.world(), *logger, Acts::equidistant, Acts::equidistant,
          Acts::equidistant, Acts::UnitConstants::mm, Acts::UnitConstants::mm,
          Acts::UnitConstants::fm, ActsPlugins::sortDetElementsByID,
          Acts::GeometryContext::dangerouslyDefaultConstruct(), matDec);
  if (tGeo == nullptr) {
    std::fprintf(stderr, "convertDD4hepDetector returned null\n");
    return 2;
  }

  Acts::GeometryContext gctx =
      Acts::GeometryContext::dangerouslyDefaultConstruct();
  Acts::MagneticFieldContext mctx;

  std::size_t nMat = 0;
  tGeo->visitSurfaces(
      [&](const Acts::Surface* s) {
        if (s != nullptr && s->surfaceMaterial() != nullptr) {
          ++nMat;
        }
      },
      false);
  std::printf("surfaces carrying material %zu\n", nMat);
  if (nMat == 0) {
    std::printf("\nNo surface carries material. Stopping.\n");
    return 2;
  }

  auto bfield =
      std::make_shared<ActsPlugins::DD4hepFieldAdapter>(detector.field());
  auto fieldCache = bfield->makeCache(mctx);
  {
    // A run in the analytic solenoid is not a run in the map, and the two are
    // not distinguishable from anything else this prints.
    const Acts::Vector3 probes[3] = {{0, 0, 0}, {800, 0, 0}, {0, 0, 2000}};
    double bz0 = 0.0, spread = 0.0;
    for (int i = 0; i < 3; ++i) {
      auto r = bfield->getField(probes[i], fieldCache);
      if (!r.ok()) {
        std::fprintf(stderr, "field lookup failed\n");
        return 2;
      }
      const double bz = (*r).z() / Acts::UnitConstants::T;
      if (i == 0) {
        bz0 = bz;
      }
      spread = std::max(spread, std::abs(bz - bz0));
    }
    std::printf("field             Bz spread over the tracker %.4f T, %s\n",
                spread, spread < 1e-6 ? "the analytic solenoid" : "the map");
  }

  // ------------------------------------------------ the helix sign, as a
  // round trip
  //
  // `nav_walk::helixAdvance` turns the direction it is given at the rate the
  // MOMENTUM turns, and `NewNavigator` hands it the direction of TRAVEL,
  // which on the CKF's backward pass is the momentum reversed. This is the
  // test that separates the two conventions and it needs no propagation:
  // advance by L, reverse the landing direction, advance by L again, and the
  // trip closes only if the second leg's curvature has the opposite sign.
  //
  // Both arms are printed. The wrong one misses by `2R(1 - cos(kappa L))`,
  // which is 30 mm for a 1 GeV track over 225 mm and 0.3 mm at 20 GeV and
  // \|eta\| 1.5, so a test run only at high pT and high \|eta\| would call both
  // of them correct.
  {
    std::printf("\nhelix round trip over 225 mm, |closure| in mm\n");
    std::printf("%8s %10s %12s %12s\n", "pT[GeV]", "|eta|", "sign flipped",
                "sign kept");
    for (double pt : {1.0, 5.0, 20.0}) {
      for (double absEta : {0.0, 1.5}) {
        const double theta = 2.0 * std::atan(std::exp(-absEta));
        const Acts::Vector3 d(std::sin(theta), 0.0, std::cos(theta));
        const Acts::Vector3 p(300.0, 0.0, 0.0);
        const double pAbs = pt / std::sin(theta);
        const double L = 225.0;
        Acts::Vector3 p1, d1, p2a, d2a, p2b, d2b;
        nav_walk::helixAdvance(p, d, pAbs, 1.0, 2.0, L, &p1, &d1);
        // Retrace: the travel direction is the momentum reversed.
        nav_walk::helixAdvance(p1, -d1, pAbs, -1.0, 2.0, L, &p2a, &d2a);
        nav_walk::helixAdvance(p1, -d1, pAbs, 1.0, 2.0, L, &p2b, &d2b);
        std::printf("%8.1f %10.1f %12.6f %12.6f\n", pt, absEta,
                    (p2a - p).norm(), (p2b - p).norm());
      }
    }
  }

  // The local map value in tesla, which is the unit `helixAdvance` and the
  // channel both use.
  auto bzAt = [&](const Acts::Vector3& p) {
    auto r = bfield->getField(p, fieldCache);
    return r.ok() ? (*r).z() / Acts::UnitConstants::T : 2.0;
  };

  Acts::Navigator::Config navCfg{tGeo};
  navCfg.resolveSensitive = true;
  navCfg.resolveMaterial = true;
  navCfg.resolvePassive = false;

  using Stepper = Acts::EigenStepper<>;
  using Prop = Acts::Propagator<Stepper, Acts::Navigator>;
  // Braces, not parentheses: with parentheses this parses as a function
  // declaration. geom_probe.cpp:354.
  Prop prop{Stepper(bfield), Acts::Navigator(navCfg)};

  std::vector<Start> starts;
  for (int ip = 0; ip < kNPt; ++ip) {
    for (double fp : kInside) {
      const double pt =
          kPtEdges[ip] * std::pow(kPtEdges[ip + 1] / kPtEdges[ip], fp);
      for (int ie = 0; ie < kNEta; ++ie) {
        const double lo = ie == 0 ? 0.0 : kEtaEdges[ie - 1];
        const double hi = kEtaEdges[ie];
        for (double fe : kInside) {
          const double absEta = lo + fe * (hi - lo);
          for (int sgn = -1; sgn <= 1; sgn += 2) {
            for (int iq = -1; iq <= 1; iq += 2) {
              for (int k = 0; k < kNPhi; ++k) {
                Start s;
                s.pt = pt;
                s.eta = sgn * absEta;
                s.phi = kPhi0 + 2.0 * M_PI * k / kNPhi;
                s.q = iq;
                s.ptBin = ip;
                s.etaBin = ie;
                starts.push_back(s);
              }
            }
          }
        }
      }
    }
  }
  std::printf("start states      %zu\n", starts.size());

  Acts::BoundMatrix cov0 = Acts::BoundMatrix::Identity();
  cov0(Acts::eBoundLoc0, Acts::eBoundLoc0) = 0.05 * 0.05;
  cov0(Acts::eBoundLoc1, Acts::eBoundLoc1) = 0.05 * 0.05;
  cov0(Acts::eBoundPhi, Acts::eBoundPhi) = 1e-6;
  cov0(Acts::eBoundTheta, Acts::eBoundTheta) = 1e-6;
  cov0(Acts::eBoundQOverP, Acts::eBoundQOverP) = 1e-8;
  cov0(Acts::eBoundTime, Acts::eBoundTime) = 1.0;

  auto makeStart = [&](const Start& s) {
    const double theta = 2.0 * std::atan(std::exp(-s.eta));
    const Acts::Vector3 dir(std::sin(theta) * std::cos(s.phi),
                            std::sin(theta) * std::sin(s.phi),
                            std::cos(theta));
    const double p = s.pt * Acts::UnitConstants::GeV / std::sin(theta);
    return Acts::BoundTrackParameters::createCurvilinear(
        Acts::Vector4(0, 0, 0, 0), dir, s.q / p, cov0,
        Acts::ParticleHypothesis::muon());
  };

  nav_walk::Options walkOpts;

  std::size_t nLegs = 0, nSameLayer = 0, nWalkFailed = 0, nWrongModule = 0;
  std::size_t nActual = 0, nPlanned = 0, nMatched = 0, nMissing = 0, nExtra = 0;
  std::map<std::size_t, std::size_t> missPerLeg, extraPerLeg;
  std::vector<double> ratioAll, ratioMatched, x0Stock, x0Seam;
  std::vector<double> corrStock, corrSeam;
  std::vector<double> x0Missing, x0Extra;
  std::size_t nVacuumStock = 0, nVacuumSeam = 0;
  std::size_t nPropFailed = 0;

  // Section 3. Where a missing surface came from, on three axes at once,
  // because one axis alone does not name a branch of the walk.
  //
  //   Its ROLE, off the geometry identifier, which says which of the three
  //   collection points owed it: an approach or representing surface comes out
  //   of `Layer::compatibleSurfaces`, a boundary out of `compatibleBoundaries`.
  //
  //   Which LAYER owns it, against the layers this walk actually asked. A
  //   surface on a layer the walk never asked is a hole in the layer walk. One
  //   on a layer it did ask is a hole in the filter applied to the answer.
  //
  //   Where it sits along the STRAIGHT RAY, against `nearestPath`.
  //   `finalizeMaterial` drops everything at or past the named module, so a
  //   surface the propagation crossed before its module and the ray places
  //   after it is that trim firing on the difference between a ray and a helix.
  const char* kRoleName[5] = {"module", "approach", "representing", "boundary",
                              "other"};
  const char* kWhereName[6] = {"the start layer",
                               "the resolving layer",
                               "a layer ahead in the resolving volume",
                               "a layer the walk never asked",
                               "no layer (boundary or volume-level)",
                               "unknown"};
  std::size_t missRole[5] = {0, 0, 0, 0, 0};
  std::size_t missWhere[6] = {0, 0, 0, 0, 0, 0};
  std::size_t missPastNearest = 0, missBeforeNearest = 0, missNoRay = 0;
  std::size_t missPastAndAsked = 0;
  std::map<std::string, std::size_t> missVolume;
  std::vector<double> missRayOver;

  // Where the seam evaluates the slab, which is where its own trajectory
  // reaches the crossing and not a re-intersection of the surface.
  //
  // `LearnedStepper::applyMaterial` uses `position(state)`: the stepper arm
  // constrains the step to land on the crossing and the learned arm advances
  // the helix to the crossing's planned path. Both are the same point to
  // within the integrator's tolerance, and it is the helix of the walk's
  // second stage, so this is the one model that covers both arms.
  //
  // Re-intersecting was the earlier model here and it is not what the seam
  // does. On a surface the track grazes, the root `closest()` picks can be a
  // long way from the crossing, which put a 16x tail on the thickness ratio
  // that belonged to the probe rather than to the seam.
  auto seamPoint = [&](const Acts::Vector3& from, const Acts::Vector3& dir,
                       double pAbs, double q, double bz, double path) {
    Acts::Vector3 p1 = from, d1 = dir;
    nav_walk::helixAdvance(from, dir, pAbs, q, bz, path, &p1, &d1);
    return std::pair<Acts::Vector3, Acts::Vector3>(p1, d1);
  };

  for (const Start& s : starts) {
    const auto start = makeStart(s);
    Prop::Options<Acts::ActorList<AllArrivalRecorder>> opt(gctx, mctx);
    opt.pathLimit = 6000.0;
    opt.maxSteps = 10000;
    auto res = prop.propagate(start, opt);
    if (!res.ok()) {
      ++nPropFailed;
      continue;
    }
    const std::vector<Arrival>& arrivals =
        res->get<AllArrivalRecorder::result_type>().arrivals;

    for (std::size_t i = 0; i < arrivals.size(); ++i) {
      if (!arrivals[i].sensitive) {
        continue;
      }
      // Find the next sensitive arrival; everything between is what the
      // navigator stopped at on this leg.
      std::size_t j = i + 1;
      while (j < arrivals.size() && !arrivals[j].sensitive) {
        ++j;
      }
      if (j >= arrivals.size()) {
        break;
      }
      const Arrival& a = arrivals[i];
      const Arrival& b = arrivals[j];
      const Acts::Layer* startLayer = a.surface->associatedLayer();
      if (startLayer == nullptr) {
        continue;
      }
      // Outside the seam's domain: the walk excludes the layer the track is
      // standing on and `NewNavigator` declines those legs.
      if (b.surface->associatedLayer() == startLayer) {
        ++nSameLayer;
        continue;
      }
      ++nLegs;

      const nav_walk::Result r =
          nav_walk::walk(*tGeo, gctx, a.pos, a.dir, startLayer, a.surface,
                         walkOpts, *logger);
      if (!r.ok) {
        ++nWalkFailed;
        continue;
      }
      if (r.nearest != b.surface) {
        ++nWrongModule;
      }

      std::unordered_set<const Acts::Surface*> actual;
      for (std::size_t k = i + 1; k < j; ++k) {
        if (arrivals[k].material) {
          actual.insert(arrivals[k].surface);
        }
      }
      std::unordered_set<const Acts::Surface*> planned;
      for (const nav_walk::Crossing& x : r.material) {
        planned.insert(x.surface);
      }
      nActual += actual.size();
      nPlanned += planned.size();

      std::size_t nMiss = 0, nExt = 0;
      for (const Acts::Surface* sf : actual) {
        if (planned.count(sf) == 0) {
          ++nMiss;
        }
      }
      for (const Acts::Surface* sf : planned) {
        if (actual.count(sf) == 0) {
          ++nExt;
        }
      }
      nMissing += nMiss;
      nExtra += nExt;
      nMatched += actual.size() - nMiss;
      ++missPerLeg[nMiss];
      ++extraPerLeg[nExt];

      // Section 3's classification, one pass over the ones that are missing.
      for (const Acts::Surface* sf : actual) {
        if (planned.count(sf) != 0) {
          continue;
        }
        ++missRole[static_cast<int>(nav_walk::kindOf(*sf))];

        const Acts::Layer* owner = sf->associatedLayer();
        int where = 5;
        if (owner == nullptr) {
          where = 4;
        } else if (owner == startLayer) {
          where = 0;
        } else if (owner == r.layer) {
          where = 1;
        } else {
          where = 3;
          for (const Acts::Layer* l : r.layersAhead) {
            if (l == owner) {
              where = 2;
              break;
            }
          }
        }
        ++missWhere[where];

        const Acts::TrackingVolume* vol =
            tGeo->findVolume(Acts::GeometryIdentifier().withVolume(
                sf->geometryId().volume()));
        ++missVolume[vol == nullptr ? std::string("unknown")
                                    : vol->volumeName()];

        // `Infinite()` deliberately: the question is where the ray meets the
        // surface's plane, not whether it lands inside the bounds. A surface
        // the exact test rejects and the propagation still stopped at is one of
        // the cases this is trying to see.
        const auto is = sf->intersect(gctx, a.pos, a.dir,
                                      Acts::BoundaryTolerance::Infinite())
                            .closest();
        if (!is.isValid()) {
          ++missNoRay;
        } else if (is.pathLength() >= r.nearestPath) {
          ++missPastNearest;
          missRayOver.push_back(is.pathLength() - r.nearestPath);
          if (where != 3) {
            ++missPastAndAsked;
          }
        } else {
          ++missBeforeNearest;
        }
      }

      // The thickness the two ways, per surface the navigator stopped at.
      for (std::size_t k = i + 1; k < j; ++k) {
        if (!arrivals[k].material) {
          continue;
        }
        const Acts::Surface& sf = *arrivals[k].surface;
        // Stock: standing exactly on the surface, PreUpdate then PostUpdate,
        // which is the two halves the CKF applies at :481 and :612.
        const Acts::MaterialSlab pre = Acts::detail::evaluateMaterialSlab(
            gctx, sf, Acts::Direction::Forward(), arrivals[k].pos,
            arrivals[k].dir, Acts::MaterialUpdateMode::PreUpdate);
        const Acts::MaterialSlab post = Acts::detail::evaluateMaterialSlab(
            gctx, sf, Acts::Direction::Forward(), arrivals[k].pos,
            arrivals[k].dir, Acts::MaterialUpdateMode::PostUpdate);
        const double stock = pre.thicknessInX0() + post.thicknessInX0();

        // Seam: one FullUpdate call where its own helix reaches the crossing.
        // A surface the plan does not name has no planned path, so the ray's
        // own intersection stands in; those are counted separately as the
        // missing ones and never enter `ratioMatched`.
        double planPath = std::nan("");
        for (const nav_walk::Crossing& x : r.material) {
          if (x.surface == &sf) {
            planPath = x.path;
            break;
          }
        }
        if (!std::isfinite(planPath)) {
          const auto is = sf.intersect(gctx, a.pos, a.dir,
                                       Acts::BoundaryTolerance::Infinite())
                              .closest();
          planPath = is.isValid() ? is.pathLength() : 0.0;
        }
        const auto [p, pd] =
            seamPoint(a.pos, a.dir, a.pAbs, a.q, bzAt(a.pos), planPath);
        const Acts::MaterialSlab full = Acts::detail::evaluateMaterialSlab(
            gctx, sf, Acts::Direction::Forward(), p, pd,
            Acts::MaterialUpdateMode::FullUpdate);
        const double seam = full.thicknessInX0();

        if (stock <= 0.0) {
          ++nVacuumStock;
        }
        if (seam <= 0.0) {
          ++nVacuumSeam;
        }
        x0Stock.push_back(stock);
        x0Seam.push_back(seam);
        corrStock.push_back(sf.pathCorrection(gctx, arrivals[k].pos,
                                              arrivals[k].dir));
        corrSeam.push_back(sf.pathCorrection(gctx, p, pd));
        if (stock > 0.0) {
          ratioAll.push_back(seam / stock);
          if (planned.count(&sf) != 0) {
            ratioMatched.push_back(seam / stock);
          } else {
            x0Missing.push_back(stock);
          }
        }
      }
      // What the plan carries that the navigator never stopped at.
      for (const nav_walk::Crossing& x : r.material) {
        if (actual.count(x.surface) != 0) {
          continue;
        }
        const auto [p, pd] =
            seamPoint(a.pos, a.dir, a.pAbs, a.q, bzAt(a.pos), x.path);
        const Acts::MaterialSlab full = Acts::detail::evaluateMaterialSlab(
            gctx, *x.surface, Acts::Direction::Forward(), p, pd,
            Acts::MaterialUpdateMode::FullUpdate);
        x0Extra.push_back(full.thicknessInX0());
      }
      i = j - 1;  // the next leg starts at this leg's destination
    }
  }

  std::printf("\npropagations failed %zu\n", nPropFailed);
  std::printf("legs outside the seam domain (successor on the start layer) "
              "%zu\n",
              nSameLayer);
  std::printf("legs in the seam domain            %zu\n", nLegs);
  std::printf("  walk named nothing               %zu  (%.2f %%)\n",
              nWalkFailed, pct(nWalkFailed, nLegs));
  std::printf("  walk named a different module    %zu  (%.2f %%)\n",
              nWrongModule, pct(nWrongModule, nLegs));

  std::printf(
      "\n================================================================\n"
      "1. The set of surfaces\n"
      "================================================================\n\n");
  std::printf("material surfaces the navigator stopped at  %zu\n", nActual);
  std::printf("material surfaces the plan names            %zu\n", nPlanned);
  std::printf("  in both                                   %zu  (%.2f %% of "
              "stopped)\n",
              nMatched, pct(nMatched, nActual));
  std::printf("  stopped at and not planned                %zu  (%.2f %%)\n",
              nMissing, pct(nMissing, nActual));
  std::printf("  planned and not stopped at                %zu  (%.2f %% of "
              "planned)\n",
              nExtra, pct(nExtra, nPlanned));

  std::printf("\nper leg:\n");
  std::printf("%8s %12s %12s\n", "n", "missing", "extra");
  for (std::size_t k = 0; k <= 6; ++k) {
    const std::size_t m = missPerLeg.count(k) ? missPerLeg[k] : 0;
    const std::size_t e = extraPerLeg.count(k) ? extraPerLeg[k] : 0;
    if (m == 0 && e == 0) {
      continue;
    }
    std::printf("%8zu %12zu %12zu\n", k, m, e);
  }

  std::printf(
      "\n================================================================\n"
      "2. The thickness, in radiation lengths\n"
      "================================================================\n\n");
  std::printf("empty slab at the evaluation point: stock %zu, seam %zu, of "
              "%zu\n",
              nVacuumStock, nVacuumSeam, x0Stock.size());
  report("stock, PreUpdate + PostUpdate", x0Stock);
  report("seam, FullUpdate", x0Seam);
  report("seam over stock, every surface", ratioAll);
  report("seam over stock, planned ones", ratioMatched);
  report("stock pathCorrection", corrStock);
  report("seam pathCorrection", corrSeam);
  report("stock t/X0 of the missing ones", x0Missing);
  report("seam t/X0 of the extra ones", x0Extra);

  {
    double sumStock = 0.0, sumSeam = 0.0;
    for (double v : x0Stock) {
      sumStock += v;
    }
    for (double v : x0Seam) {
      sumSeam += v;
    }
    double sumMissing = 0.0, sumExtra = 0.0;
    for (double v : x0Missing) {
      sumMissing += v;
    }
    for (double v : x0Extra) {
      sumExtra += v;
    }
    std::printf("\ntotal t/X0 the navigator stopped at   %.4f\n", sumStock);
    std::printf("total t/X0 the seam would apply       %.4f\n",
                sumSeam - sumMissing + sumExtra);
    std::printf("  of which never stopped at           %.4f\n", sumExtra);
    std::printf("  and never applied                   %.4f\n", sumMissing);
  }

  std::printf(
      "\n================================================================\n"
      "3. Where the missing ones come from\n"
      "================================================================\n\n");
  std::printf("by role:\n");
  for (int k = 0; k < 5; ++k) {
    if (missRole[k] != 0) {
      std::printf("  %-38s %7zu  (%.2f %%)\n", kRoleName[k], missRole[k],
                  pct(missRole[k], nMissing));
    }
  }
  std::printf("\nby the layer that owns it:\n");
  for (int k = 0; k < 6; ++k) {
    if (missWhere[k] != 0) {
      std::printf("  %-38s %7zu  (%.2f %%)\n", kWhereName[k], missWhere[k],
                  pct(missWhere[k], nMissing));
    }
  }
  std::printf("\nagainst the straight ray's own module distance:\n");
  std::printf("  %-38s %7zu  (%.2f %%)\n", "at or past it, so the trim dropped",
              missPastNearest, pct(missPastNearest, nMissing));
  std::printf("  %-38s %7zu\n", "of those, on a layer the walk asked",
              missPastAndAsked);
  std::printf("  %-38s %7zu  (%.2f %%)\n", "before it, so never collected",
              missBeforeNearest, pct(missBeforeNearest, nMissing));
  std::printf("  %-38s %7zu\n", "the ray does not meet it at all", missNoRay);
  report("mm past the named module", missRayOver);
  std::printf("\nby volume:\n");
  for (const auto& [name, n] : missVolume) {
    std::printf("  %-38s %7zu  (%.2f %%)\n", name.c_str(), n,
                pct(n, nMissing));
  }

  std::printf(
      "\nThe seam's own evaluation point is the surface intersected from the "
      "module\nthe leg starts at, which is what `LearnedStepper::applyMaterial` "
      "does. The\ndirection is the one at the start of the leg, because a "
      "stepper flying a leg\nhas no later one to use.\n");
  return 0;
}
