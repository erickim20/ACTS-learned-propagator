// Where the material sits along a module-to-module leg, measured off the
// propagation rather than off the geometry.
//
//   ./run_in_image.sh --map mat_leg
//
// The split-Jacobian error has been scanned at the percentiles that describe
// the last few millimetres the stepper seam gets today. A moved seam flies the
// whole leg, so the deltas it
// has to handle are the distances from the destination module back to every
// material surface the leg crosses, and there is more than one of them.
//
// The ground truth is the same stock Propagator<EigenStepper<>, Navigator> over
// the same 5,400 start states that cpp/nav_truth.cpp uses, so the transitions
// counted here are its transitions. Nothing here calls nav_walk, so the walk's
// own numbers cannot move: this measures the surfaces the propagation makes
// current, and the walk is not in the loop.
//
// The recorder differs from nav_truth's in one way and it is the point of this
// file: nav_truth's ArrivalRecorder keeps only sensitive surfaces, and this one
// keeps every surface the navigator makes current. A material-only surface is
// exactly a surface that is current and carries material and is not sensitive,
// which is CombinatorialKalmanFilter.hpp:456-458's own split.
//
// convertDD4hepDetector's material decorator argument defaults to nullptr and
// builds a geometry with no material at all, without warning.
// Every count here would then be zero, so the coverage is printed and a zero
// stops the run.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "Acts/Definitions/Units.hpp"
#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/MagneticField/MagneticFieldContext.hpp"
#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Propagator/MaterialInteractor.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Utilities/Result.hpp"

#include "ActsPlugins/DD4hep/ConvertDD4hepDetector.hpp"
#include "ActsPlugins/DD4hep/DD4hepFieldAdapter.hpp"
#include "ActsPlugins/Root/RootMaterialDecorator.hpp"
#include "DD4hep/Detector.h"

namespace {

// ---------------------------------------------------------------- the sample
//
// Copied value for value from cpp/nav_truth.cpp so the two run the same 5,400
// start states. The pT edges are the muon table's; TransportCensus.hpp:135
// carries the |eta| edges.
constexpr double kPtEdges[6] = {1.0, 2.0, 4.0, 8.0, 20.0, 50.0};
constexpr int kNPt = 5;
constexpr double kEtaEdges[5] = {0.6, 1.2, 1.8, 2.4, 3.0};
constexpr int kNEta = 5;
constexpr double kInside[3] = {1.0 / 6.0, 3.0 / 6.0, 5.0 / 6.0};
constexpr int kNPhi = 6;
constexpr double kPhi0 = 0.1234;

struct Start {
  double pt = 0.0;
  double eta = 0.0;
  double phi = 0.0;
  double q = 1.0;
  int ptBin = 0;
  int etaBin = 0;
};

// ------------------------------------------------------------------ arrivals

/// One surface the navigator made current, with what it is.
struct Arrival {
  const Acts::Surface* surface = nullptr;
  double path = 0.0;
  bool sensitive = false;
  bool material = false;
  /// Which of the three material-carrying kinds this is. Read off the
  /// geometry identifier: `sensitive()` non-zero is a module, `approach()`
  /// non-zero is an approach-descriptor surface, `boundary()` non-zero is a
  /// volume boundary, and layer-scoped with none of those is the layer's own
  /// representing surface.
  int kind = 0;
};

constexpr int kKindModule = 0;
constexpr int kKindApproach = 1;
constexpr int kKindRepresenting = 2;
constexpr int kKindBoundary = 3;
constexpr int kKindOther = 4;
constexpr int kNKind = 5;
const char* kKindName[kNKind] = {"module", "approach descriptor",
                                 "layer representing", "volume boundary",
                                 "other"};

int classify(const Acts::Surface& s) {
  const Acts::GeometryIdentifier id = s.geometryId();
  if (id.sensitive() != 0) {
    return kKindModule;
  }
  if (id.approach() != 0) {
    return kKindApproach;
  }
  if (id.boundary() != 0) {
    return kKindBoundary;
  }
  if (id.layer() != 0) {
    return kKindRepresenting;
  }
  return kKindOther;
}

/// Every surface the propagation makes current, in order.
///
/// `act` returns `Result<void>` and not void: 44.99.99's actor_caller assigns
/// the return into the propagation's global result, so a void `act` fails the
/// Actor concept with a wall of substitution errors. nav_truth.cpp says the
/// same and for the same reason.
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
    // A surface stays current for more than one step, so comparing against the
    // last recorded one makes this one entry per arrival.
    if (s == nullptr || s == result.last) {
      return Acts::Result<void>::success();
    }
    result.last = s;
    Arrival a;
    a.surface = s;
    a.path = state.stepping.pathAccumulated;
    a.sensitive = s->isSensitive();
    a.material = s->surfaceMaterial() != nullptr;
    a.kind = classify(*s);
    result.arrivals.push_back(a);
    return Acts::Result<void>::success();
  }
};

// --------------------------------------------------------------- the counting

double quantile(std::vector<double>& v, double p) {
  if (v.empty()) {
    return 0.0;
  }
  std::sort(v.begin(), v.end());
  const auto i =
      static_cast<std::size_t>(p * static_cast<double>(v.size() - 1));
  return v[i];
}

void report(const char* what, std::vector<double> v) {
  if (v.empty()) {
    std::printf("| %-22s | %8s | %8s | %8s | %8s | %8s |\n", what, "0", "-",
                "-", "-", "-");
    return;
  }
  std::printf("| %-22s | %8zu | %8.2f | %8.2f | %8.2f | %8.2f |\n", what,
              v.size(), quantile(v, 0.10), quantile(v, 0.50),
              quantile(v, 0.90), quantile(v, 1.0));
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
  std::shared_ptr<const Acts::IMaterialDecorator> matDec;
  std::FILE* mapProbe =
      mapFile.empty() ? nullptr : std::fopen(mapFile.c_str(), "r");
  if (mapProbe == nullptr) {
    std::printf("material map NOT FOUND at %s; stopping.\n",
                mapFile.empty() ? "(not derivable)" : mapFile.c_str());
    return 2;
  }
  std::fclose(mapProbe);
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

  std::size_t nSurf = 0, nSurfWithMat = 0, nSens = 0, nSensWithMat = 0;
  std::size_t byKind[kNKind] = {};
  tGeo->visitSurfaces(
      [&](const Acts::Surface* s) {
        if (s == nullptr) {
          return;
        }
        ++nSurf;
        const bool hasMat = s->surfaceMaterial() != nullptr;
        nSurfWithMat += hasMat ? 1 : 0;
        if (hasMat) {
          ++byKind[classify(*s)];
        }
        if (s->isSensitive()) {
          ++nSens;
          nSensWithMat += hasMat ? 1 : 0;
        }
      },
      false);
  std::printf("surfaces          %zu, %zu carry material\n", nSurf,
              nSurfWithMat);
  std::printf("  sensitive       %zu, %zu carry material\n", nSens,
              nSensWithMat);
  for (int k = 0; k < kNKind; ++k) {
    if (byKind[k] != 0) {
      std::printf("  %-22s %zu\n", kKindName[k], byKind[k]);
    }
  }
  if (nSurfWithMat == 0) {
    std::printf("\nNo surface carries material; every count below is zero for "
                "a reason that is not the geometry. Stopping.\n");
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
                spread,
                spread < 1e-6 ? "the analytic solenoid" : "the map");
  }

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

  for (int mat = 0; mat < 2; ++mat) {
    std::size_t nPropFailed = 0, nArrivals = 0;
    std::size_t nLegs = 0, nSameLayer = 0;
    // Crossings per leg, as a histogram rather than a mean.
    std::map<std::size_t, std::size_t> perLeg, perLegSameLayerExcluded;
    // Distance from the destination module, in mm of 3D path.
    std::vector<double> dist, distByKind[kNKind];
    std::vector<double> legLen;
    // The nearest and the farthest crossing of each leg, which are the two ends
    // of the range a covariance loop has to span.
    std::vector<double> nearest, farthest;
    std::size_t nBoundaryMat = 0, nApproachMat = 0, nRepMat = 0;

    for (const Start& s : starts) {
      const auto start = makeStart(s);
      std::vector<Arrival> arrivals;
      if (mat == 0) {
        Prop::Options<Acts::ActorList<AllArrivalRecorder>> opt(gctx, mctx);
        opt.pathLimit = 6000.0;
        opt.maxSteps = 10000;
        auto res = prop.propagate(start, opt);
        if (!res.ok()) {
          ++nPropFailed;
          continue;
        }
        arrivals = res->get<AllArrivalRecorder::result_type>().arrivals;
      } else {
        Prop::Options<
            Acts::ActorList<AllArrivalRecorder, Acts::MaterialInteractor>>
            opt(gctx, mctx);
        opt.pathLimit = 6000.0;
        opt.maxSteps = 10000;
        auto& mi = opt.actorList.get<Acts::MaterialInteractor>();
        mi.multipleScattering = true;
        mi.energyLoss = true;
        auto res = prop.propagate(start, opt);
        if (!res.ok()) {
          ++nPropFailed;
          continue;
        }
        arrivals = res->get<AllArrivalRecorder::result_type>().arrivals;
      }
      nArrivals += arrivals.size();

      // A leg runs from one sensitive arrival to the next. Everything between
      // them that carries material is a crossing the moved seam has to apply
      // itself, because the seam flies the whole leg in one piece.
      std::size_t prevSens = 0;
      bool havePrev = false;
      for (std::size_t i = 0; i < arrivals.size(); ++i) {
        if (!arrivals[i].sensitive) {
          continue;
        }
        if (havePrev) {
          const Arrival& a = arrivals[prevSens];
          const Arrival& b = arrivals[i];
          ++nLegs;
          legLen.push_back(b.path - a.path);

          const bool sameLayer =
              a.surface->associatedLayer() != nullptr &&
              a.surface->associatedLayer() == b.surface->associatedLayer();
          if (sameLayer) {
            ++nSameLayer;
          }

          std::size_t n = 0;
          double near = -1.0, far = -1.0;
          for (std::size_t k = prevSens + 1; k < i; ++k) {
            if (!arrivals[k].material) {
              continue;
            }
            ++n;
            const double d = b.path - arrivals[k].path;
            dist.push_back(d);
            distByKind[arrivals[k].kind].push_back(d);
            if (near < 0.0 || d < near) {
              near = d;
            }
            if (d > far) {
              far = d;
            }
            if (arrivals[k].kind == kKindBoundary) {
              ++nBoundaryMat;
            } else if (arrivals[k].kind == kKindApproach) {
              ++nApproachMat;
            } else if (arrivals[k].kind == kKindRepresenting) {
              ++nRepMat;
            }
          }
          ++perLeg[n];
          if (!sameLayer) {
            ++perLegSameLayerExcluded[n];
          }
          if (n > 0) {
            nearest.push_back(near);
            farthest.push_back(far);
          }
        }
        prevSens = i;
        havePrev = true;
      }
    }

    std::printf(
        "\n================================================================\n"
        "Material %s\n"
        "================================================================\n\n",
        mat == 0 ? "off in the probe's actor list"
                 : "on in the probe's actor list");
    std::printf("propagations failed   %zu of %zu\n", nPropFailed,
                starts.size());
    std::printf("surface arrivals      %zu\n", nArrivals);
    std::printf("module-to-module legs %zu\n", nLegs);
    std::printf("  of which same-layer %zu, so %zu are the transitions\n",
                nSameLayer, nLegs - nSameLayer);
    std::printf("material crossings    %zu  (approach %zu, representing %zu, "
                "boundary %zu)\n",
                dist.size(), nApproachMat, nRepMat, nBoundaryMat);

    std::printf("\nMaterial-carrying surfaces crossed per leg.\n\n");
    std::printf("| crossings | legs | %% | legs, same-layer excluded | %% |\n");
    std::printf("| ---: | ---: | ---: | ---: | ---: |\n");
    const std::size_t denom = nLegs == 0 ? 1 : nLegs;
    const std::size_t denom2 =
        (nLegs - nSameLayer) == 0 ? 1 : (nLegs - nSameLayer);
    std::size_t maxN = 0;
    for (const auto& e : perLeg) {
      maxN = std::max(maxN, e.first);
    }
    for (std::size_t k = 0; k <= maxN; ++k) {
      const std::size_t a = perLeg.count(k) != 0 ? perLeg.at(k) : 0;
      const std::size_t b = perLegSameLayerExcluded.count(k) != 0
                                ? perLegSameLayerExcluded.at(k)
                                : 0;
      if (a == 0) {
        continue;
      }
      std::printf("| %zu | %zu | %.2f | %zu | %.2f |\n", k, a,
                  100.0 * static_cast<double>(a) / static_cast<double>(denom), b,
                  100.0 * static_cast<double>(b) / static_cast<double>(denom2));
    }

    std::printf("\nDistance from the destination module, mm of 3D path.\n\n");
    std::printf("| | n | p10 | p50 | p90 | max |\n");
    std::printf("| --- | ---: | ---: | ---: | ---: | ---: |\n");
    report("every crossing", dist);
    for (int k = 0; k < kNKind; ++k) {
      if (!distByKind[k].empty()) {
        report(kKindName[k], distByKind[k]);
      }
    }
    report("nearest of each leg", nearest);
    report("farthest of each leg", farthest);
    report("leg length", legLen);
  }

  return 0;
}
