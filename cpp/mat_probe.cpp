// Where the ODD's 123 material surfaces sit, and which of them
// Navigator::Config::resolveMaterial can actually remove.
//
//   ./run_in_image.sh --map mat_probe
//
// Turning resolveMaterial off is a proposed proxy for a seam that
// stops reporting the material surfaces. The source says that reaches only
// part of the channel, and this counts the parts.
//
// Layer::compatibleSurfaces gates its approach-descriptor section (A) at
// Layer.cpp:185 and its representing-surface section (C) at :214 on
// resolveMaterial || resolvePassive, so with the flag off neither contributes
// to navSurfaces.
//
// Layer::surfaceOnApproach does not gate on it. resolvePS = resolveSensitive
// || resolvePassive at Layer.cpp:232 is true under this project's
// configuration, so the approach descriptor is still consulted and the layer
// target is still one of the approach surfaces. Navigator::handleSurfaceReached
// then writes state.currentSurface = &surface unconditionally at
// Navigator.cpp:309, ahead of the stage tests, so that surface's material is
// applied and its track state added whatever resolveMaterial says.
//
// So the flag removes a layer's OTHER approach surfaces and its representing
// surface, and not the one the track approaches on. This counts them per layer
// so the proxy's reach is a number rather than an argument.
#include <algorithm>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "Acts/Geometry/ApproachDescriptor.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Geometry/Layer.hpp"
#include "Acts/Material/ISurfaceMaterial.hpp"
#include "Acts/Material/MaterialSlab.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Geometry/TrackingVolume.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Surfaces/SurfaceArray.hpp"
#include "Acts/Utilities/BinnedArray.hpp"
#include "Acts/Utilities/Logger.hpp"

#include "ActsPlugins/DD4hep/ConvertDD4hepDetector.hpp"
#include "ActsPlugins/Root/RootMaterialDecorator.hpp"
#include "DD4hep/Detector.h"

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

  // ------------------------------------------- the 123, split by identifier
  //
  // A GeometryIdentifier says which of a layer's roles a surface plays:
  // sensitive() non-zero is a module, approach() non-zero is one of the
  // approach descriptor's surfaces, and layer-scoped with neither is the
  // layer's own representing surface. boundary() non-zero is a volume boundary
  // and belongs to no layer.
  std::size_t nAll = 0, nMat = 0;
  std::size_t mSens = 0, mAppr = 0, mRepr = 0, mBound = 0, mOther = 0;
  std::map<Acts::GeometryIdentifier::Value, std::size_t> apprIndex;
  tGeo->visitSurfaces(
      [&](const Acts::Surface* s) {
        if (s == nullptr) {
          return;
        }
        ++nAll;
        if (s->surfaceMaterial() == nullptr) {
          return;
        }
        ++nMat;
        const Acts::GeometryIdentifier id = s->geometryId();
        if (id.sensitive() != 0) {
          ++mSens;
        } else if (id.approach() != 0) {
          ++mAppr;
          ++apprIndex[id.approach()];
        } else if (id.boundary() != 0) {
          ++mBound;
        } else if (id.layer() != 0) {
          ++mRepr;
        } else {
          ++mOther;
        }
      },
      false);

  std::printf("\nsurfaces %zu, %zu carry material\n", nAll, nMat);
  std::printf("  sensitive (module)              %zu\n", mSens);
  std::printf("  approach descriptor             %zu\n", mAppr);
  std::printf("  layer representing surface      %zu\n", mRepr);
  std::printf("  volume boundary                 %zu\n", mBound);
  std::printf("  other                           %zu\n", mOther);
  std::printf("\napproach index of the material approach surfaces:\n");
  for (const auto& entry : apprIndex) {
    std::printf("  approach()=%llu   %zu\n",
                static_cast<unsigned long long>(entry.first), entry.second);
  }

  // ------------------------------------ per layer, what the flag can remove
  //
  // Walk the layers the navigator walks: every volume's confined layer array.
  std::printf(
      "\n================================================================\n"
      "Layers, and the material the flag reaches\n"
      "================================================================\n\n");
  std::printf("%-30s %6s %8s %9s %8s %7s\n", "volume", "layer", "modules",
              "approach", "apprMat", "repMat");

  std::size_t nLayersWithArray = 0, nLayersNoArray = 0;
  std::size_t apprTotal = 0, apprMatTotal = 0, repMatTotal = 0;
  std::size_t apprMatOnLayersWithArray = 0, apprMatOnLayersNoArray = 0;

  tGeo->visitVolumes([&](const Acts::TrackingVolume* v) {
    if (v == nullptr || v->confinedLayers() == nullptr) {
      return;
    }
    for (const auto& lptr : v->confinedLayers()->arrayObjects()) {
      const Acts::Layer* l = lptr.get();
      if (l == nullptr) {
        continue;
      }
      const bool hasArray = l->surfaceArray() != nullptr;
      const std::size_t nMod =
          hasArray ? l->surfaceArray()->surfaces().size() : std::size_t{0};
      std::size_t nAppr = 0, nApprMat = 0;
      if (l->approachDescriptor() != nullptr) {
        for (const Acts::Surface* a :
             l->approachDescriptor()->containedSurfaces()) {
          if (a == nullptr) {
            continue;
          }
          ++nAppr;
          if (a->surfaceMaterial() != nullptr) {
            ++nApprMat;
          }
        }
      }
      const bool repMat =
          l->surfaceRepresentation().surfaceMaterial() != nullptr;

      // A layer with neither modules nor material is a NavigationLayer and
      // never resolves under any flag; skip it so the table is the detector.
      if (!hasArray && nApprMat == 0 && !repMat) {
        continue;
      }
      if (hasArray) {
        ++nLayersWithArray;
        apprMatOnLayersWithArray += nApprMat;
      } else {
        ++nLayersNoArray;
        apprMatOnLayersNoArray += nApprMat;
      }
      apprTotal += nAppr;
      apprMatTotal += nApprMat;
      repMatTotal += repMat ? 1 : 0;

      std::printf("%-30s %6llu %8zu %9zu %8zu %7s\n", v->volumeName().c_str(),
                  static_cast<unsigned long long>(
                      l->surfaceRepresentation().geometryId().layer()),
                  nMod, nAppr, nApprMat, repMat ? "yes" : "-");
    }
  });

  // ------------------------------------------- how thick the material is
  //
  // Run 3 needs a scattering angle to put on the skipped surface, and an
  // assumed one would make the answer an assumption. This is the ODD's own
  // thickness in radiation lengths, sampled at each material surface's centre.
  // `materialSlab` is the binned map's value there and not an average over the
  // surface, which is what a track crossing at that point would see.
  //
  // Sampled in LOCAL coordinates and not at `center()`. A cylinder's centre is
  // a point on its axis, which is not on the surface, and the binned map
  // returns an empty slab there: sampling that way found a non-zero slab on 19
  // of the 123 and the 19 were the calorimeter.
  //
  // Split into the tracker's own volumes and the rest, because the scattering
  // Run 3 needs is the tracker's. The volume ids are `chi2_gate.CLASS_OF`.
  {
    auto isTracker = [](unsigned v) {
      return (v >= 16 && v <= 18) || (v >= 23 && v <= 25) ||
             (v >= 28 && v <= 30);
    };
    std::vector<double> trk, rest;
    std::size_t nZero = 0;
    tGeo->visitSurfaces(
        [&](const Acts::Surface* s) {
          if (s == nullptr || s->surfaceMaterial() == nullptr) {
            return;
          }
          double best = 0.0;
          // A few local points, because a binned map is not constant over the
          // surface and one sample can land in an empty bin.
          for (double a : {0.0, 0.25, 0.5, 0.75}) {
            const Acts::Vector2 lp(a * 100.0, a * 100.0);
            best = std::max(
                best, static_cast<double>(
                          s->surfaceMaterial()->materialSlab(lp).thicknessInX0()));
          }
          if (best <= 0.0) {
            ++nZero;
            return;
          }
          (isTracker(static_cast<unsigned>(s->geometryId().volume())) ? trk
                                                                     : rest)
              .push_back(best);
        },
        false);
    auto report = [](const char* what, std::vector<double> v) {
      std::sort(v.begin(), v.end());
      auto q = [&](double f) {
        return v.empty() ? 0.0
                         : v[static_cast<std::size_t>(
                               f * static_cast<double>(v.size() - 1))];
      };
      std::printf("  %-22s n %3zu   p10 %.5f  p50 %.5f  p90 %.5f  max %.5f\n",
                  what, v.size(), q(0.10), q(0.50), q(0.90), q(1.0));
    };
    std::printf("\nthickness in X0, sampled at four local points per surface\n");
    report("tracker volumes", trk);
    report("everything else", rest);
    std::printf("  %-22s n %3zu\n", "no slab at any sample", nZero);
  }

  std::printf("\nlayers with a surface array           %zu\n",
              nLayersWithArray);
  std::printf("layers without one, carrying material %zu\n", nLayersNoArray);
  std::printf("approach surfaces                     %zu\n", apprTotal);
  std::printf("  carrying material                   %zu\n", apprMatTotal);
  std::printf("    on layers with a surface array    %zu\n",
              apprMatOnLayersWithArray);
  std::printf("    on layers without one             %zu\n",
              apprMatOnLayersNoArray);
  std::printf("representing surfaces with material   %zu\n", repMatTotal);

  return 0;
}
