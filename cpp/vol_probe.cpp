// Two counts over the ODD's tracking volumes.
//
//   ./run_in_image.sh --map vol_probe
//
// 1. Volumes with a non-null `volumeMaterial()`. `Navigator.cpp:56-62` is where
//    the propagation reads it, through `Navigator::currentVolumeMaterial`. If
//    any volume carries it there is a second material channel besides the 123
//    surfaces, and the surface-only picture is incomplete.
//
// 2. Volumes with a non-null `navigationPolicy()` (`TrackingVolume.hpp:531`),
//    which is what `Navigator::checkTargetValid` consults at
//    `Navigator.cpp:286-295`. Eight of the navigator's twelve members are
//    trivial only if this is empty.
//
// The geometry is built exactly as `nav_truth.cpp` builds it, material map
// included: `convertDD4hepDetector`'s `matDecorator` argument defaults to
// nullptr and a two-argument call gives a geometry with no material at all,
// without warning, which would make count 1 come out zero for the wrong
// reason.
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Geometry/TrackingVolume.hpp"
#include "Acts/Material/IVolumeMaterial.hpp"
#include "Acts/Navigation/INavigationPolicy.hpp"
#include "Acts/Surfaces/Surface.hpp"
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
  if (mapProbe != nullptr) {
    std::fclose(mapProbe);
    ActsPlugins::RootMaterialDecorator::Config cfg;
    cfg.fileName = mapFile;
    matDec = std::make_shared<const ActsPlugins::RootMaterialDecorator>(
        cfg, Acts::Logging::WARNING);
    std::printf("material map      %s\n", mapFile.c_str());
  } else {
    std::printf("material map      NOT FOUND at %s\n",
                mapFile.empty() ? "(path not derivable from the compact)"
                                : mapFile.c_str());
    std::printf("\nWithout the map every material count below is zero for a\n"
                "reason that has nothing to do with the geometry. Stopping.\n");
    return 2;
  }

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

  const bool gen1 = tGeo->geometryVersion() ==
                    Acts::TrackingGeometry::GeometryVersion::Gen1;
  std::printf("geometry version  %s\n", gen1 ? "Gen1" : "Gen3");

  // The surface coverage, as the check that the decoration took at all. The
  // second argument of `visitSurfaces` is `restrictToSensitives` and defaults
  // to true; the ODD's map lands on approach surfaces and not on modules, so
  // the default finds zero on a correctly decorated geometry.
  std::size_t nSurf = 0, nSurfWithMat = 0, nSens = 0, nSensWithMat = 0;
  tGeo->visitSurfaces(
      [&](const Acts::Surface* s) {
        if (s == nullptr) {
          return;
        }
        ++nSurf;
        const bool hasMat = s->surfaceMaterial() != nullptr;
        nSurfWithMat += hasMat ? 1 : 0;
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
  if (nSurfWithMat == 0) {
    std::printf(
        "\nNo surface carries material, so the decoration did not take and the\n"
        "volume counts below are not a measurement of the geometry. Stopping.\n");
    return 2;
  }

  // ---------------------------------------------------------- the two counts
  std::size_t nVol = 0, nVolMat = 0, nVolPolicy = 0;
  std::vector<std::string> matNames, policyNames;
  tGeo->visitVolumes([&](const Acts::TrackingVolume* v) {
    if (v == nullptr) {
      return;
    }
    ++nVol;
    if (v->volumeMaterial() != nullptr) {
      ++nVolMat;
      matNames.push_back(v->volumeName());
    }
    if (v->navigationPolicy() != nullptr) {
      ++nVolPolicy;
      policyNames.push_back(v->volumeName());
    }
  });

  std::printf(
      "\n================================================================\n"
      "Volumes\n"
      "================================================================\n\n");
  std::printf("tracking volumes                    %zu\n", nVol);
  std::printf("  with volumeMaterial()             %zu\n", nVolMat);
  std::printf("  with navigationPolicy()           %zu\n", nVolPolicy);

  if (!matNames.empty()) {
    std::printf("\nvolumeMaterial() non-null on:\n");
    for (const std::string& n : matNames) {
      std::printf("  %s\n", n.c_str());
    }
  }
  if (!policyNames.empty()) {
    std::printf("\nnavigationPolicy() non-null on:\n");
    for (const std::string& n : policyNames) {
      std::printf("  %s\n", n.c_str());
    }
  }

  // Every volume by name, so a count of zero can be read against a list that is
  // plainly the whole detector rather than against a visit that found nothing.
  std::printf(
      "\nEvery volume visited, name / volumeMaterial / navigationPolicy:\n");
  tGeo->visitVolumes([&](const Acts::TrackingVolume* v) {
    if (v == nullptr) {
      return;
    }
    std::printf("  %-28s %-4s %-4s\n", v->volumeName().c_str(),
                v->volumeMaterial() != nullptr ? "yes" : "-",
                v->navigationPolicy() != nullptr ? "yes" : "-");
  });

  return 0;
}
