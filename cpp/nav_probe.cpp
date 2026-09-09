// Can the next module be named from where the track already is, without first
// flying to that layer's approach surface?
//
// Why this matters. Inside the CKF a module becomes the
// stepper's target only for the last few millimetres of a leg, because
// `Navigator::resolveSurfaces` will not build a candidate list before the track
// has arrived at the layer. The stepper seam therefore cannot be widened. A
// propagator seam only helps if the module can be named EARLIER, and the walk
// that tries is `nav_walk.hpp`, shared with `nav_truth`.
//
// This runs that walk over the whole detector and reports whether a sensitive
// surface comes back and at what path length. It does not check that the module
// it names is the one a propagation reaches; that is `nav_truth`.
//
// What a start point is. Every sensitive module centre in the geometry, with
// the direction a straight track from the origin would have there. That is the
// situation the navigator is in when it has just finished with a module and
// switches to `layerTarget`: standing on a module, about to fly to the next
// layer's approach surface. `compatibleSurfaces` intersects along the straight
// direction it is handed, so a straight ray is not an approximation of the
// query, only of the trajectory.
//
// What is reported. Per start point, the path to the nearest layer's approach
// surface and the path to the nearest sensitive surface resolved on that layer.
// The gap between those two is the millimetres the current design gives to
// `EigenStepper` and a propagator seam would not have to. Split by whether the
// module was found in the volume the track stands in or past a boundary,
// because those are different amounts of work for a propagator to do.
//
// Also the geometry version, because Gen1 is taken from the build
// path (`convertDD4hepDetector` constructs no `Acts::Portal`) and not from a
// run. `geometryVersion()` is the navigator's own test.
//
//   ./run_in_image.sh nav_probe            # the ODD as shipped
//   ./run_in_image.sh nav_probe --map      # the tree that carries the map
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "Acts/Definitions/Units.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Utilities/Logger.hpp"

#include "ActsPlugins/DD4hep/ConvertDD4hepDetector.hpp"
#include "DD4hep/Detector.h"

#include "nav_walk.hpp"

namespace {

double quantile(std::vector<double>& v, double p) {
  if (v.empty()) {
    return std::nan("");
  }
  std::sort(v.begin(), v.end());
  const auto i =
      static_cast<std::size_t>(p * static_cast<double>(v.size() - 1));
  return v[i];
}

void report(const char* name, std::vector<double> v) {
  if (v.empty()) {
    std::printf("%-28s  no entries\n", name);
    return;
  }
  double sum = 0.0;
  for (double x : v) {
    sum += x;
  }
  std::printf("%-28s  n %6zu   p10 %8.1f   median %8.1f   p90 %8.1f   "
              "mean %8.1f\n",
              name, v.size(), quantile(v, 0.10), quantile(v, 0.50),
              quantile(v, 0.90), sum / static_cast<double>(v.size()));
}

}  // namespace

int main(int argc, char** argv) {
  const std::string compact =
      argc > 1 ? argv[1] : "/opt/odd/xml/OpenDataDetector.xml";

  auto& detector = dd4hep::Detector::getInstance();
  detector.fromCompact(compact);
  detector.volumeManager();
  detector.apply("DD4hepVolumeManager", 0, nullptr);

  auto logger = Acts::getDefaultLogger("ODD", Acts::Logging::WARNING);
  std::shared_ptr<const Acts::TrackingGeometry> tGeo =
      ActsPlugins::convertDD4hepDetector(detector.world(), *logger);
  if (tGeo == nullptr) {
    std::fprintf(stderr, "convertDD4hepDetector returned null\n");
    return 2;
  }

  Acts::GeometryContext gctx =
      Acts::GeometryContext::dangerouslyDefaultConstruct();

  // The navigator's own test, run here so Gen1 is read out of the geometry
  // rather than inferred from which builders assembled it.
  const bool gen1 = tGeo->geometryVersion() ==
                    Acts::TrackingGeometry::GeometryVersion::Gen1;
  std::printf("compact           %s\n", compact.c_str());
  std::printf("geometry version  %s\n", gen1 ? "Gen1" : "Gen3");
  if (!gen1) {
    std::printf(
        "\nGen3. `getNextTargetGen1` is not the branch that runs, so nothing "
        "below\nis the measurement it claims.\n");
  }

  const nav_walk::Options opts;

  // Start points: every sensitive module centre inside the tracker.
  std::vector<Acts::Vector3> starts;
  std::vector<const Acts::Surface*> startSurfaces;
  tGeo->visitSurfaces([&](const Acts::Surface* s) {
    if (s == nullptr || !s->isSensitive()) {
      return;
    }
    const Acts::Vector3 c = s->localToGlobalTransform(gctx).translation();
    if (c.head<2>().norm() > opts.rMax || std::abs(c.z()) > opts.zMax) {
      return;
    }
    starts.push_back(c);
    startSurfaces.push_back(s);
  });
  std::printf("start points      %zu sensitive module centres\n\n",
              starts.size());

  std::vector<double> approach0, module0, gap0;
  std::vector<double> approach1, module1, gap1, boundary1;
  std::size_t noVolume = 0, noStartLayer = 0;
  std::size_t sameNoLayer = 0, sameNoModule = 0;

  // Where the failures are, by volume, because a count alone does not say
  // whether the unresolved fraction is the geometry or the query. Each entry is
  // { starts, resolved in this volume, resolved after a crossing }.
  struct VolumeTally {
    std::size_t n = 0, same = 0, hop = 0;
  };
  std::map<std::string, VolumeTally> byVolume;
  // The crossings themselves, as `from -> to`, with how many of them then found
  // a module. A destination that never resolves is a gap volume and says the
  // walk has to continue rather than that it failed.
  std::map<std::string, std::pair<std::size_t, std::size_t>> crossings;
  /// How many boundary crossings the resolved ones needed.
  std::map<std::size_t, std::size_t> hopHist;
  /// Why the walk stopped, for the ones that never resolved.
  std::map<std::string, std::size_t> stopReason;
  std::set<std::string> seen;
  std::size_t printed = 0;

  for (std::size_t i = 0; i < starts.size(); ++i) {
    const Acts::Vector3 p = starts[i];
    // A straight track from the origin. `compatibleSurfaces` intersects along
    // whatever direction it is given, so this fixes the query and not the
    // physics.
    const Acts::Vector3 d = p.normalized();

    const Acts::TrackingVolume* volume = tGeo->lowestTrackingVolume(gctx, p);
    if (volume == nullptr) {
      ++noVolume;
      continue;
    }
    VolumeTally& tally = byVolume[volume->volumeName()];
    ++tally.n;

    const Acts::Layer* startLayer = startSurfaces[i]->associatedLayer();
    if (startLayer == nullptr) {
      ++noStartLayer;
      continue;
    }

    const nav_walk::Result r = nav_walk::walk(*tGeo, gctx, p, d, startLayer,
                                              startSurfaces[i], opts, *logger);

    if (r.firstVolume == nav_walk::FirstVolume::NoLayer) {
      ++sameNoLayer;
    } else if (r.firstVolume == nav_walk::FirstVolume::NoModule) {
      ++sameNoModule;
    }

    for (std::size_t k = 0; k < r.crossed.size(); ++k) {
      const std::string key = r.crossed[k].first->volumeName() + "  ->  " +
                              r.crossed[k].second->volumeName();
      ++crossings[key].first;
      if (r.ok && k + 1 == r.crossed.size()) {
        ++crossings[key].second;
      }
    }

    if (r.ok) {
      if (r.crossings == 0) {
        approach0.push_back(r.layerPath);
        module0.push_back(r.nearestPath);
        gap0.push_back(r.nearestPath - r.layerPath);
        ++tally.same;
      } else {
        approach1.push_back(r.layerPath);
        module1.push_back(r.nearestPath);
        gap1.push_back(r.nearestPath - r.layerPath);
        boundary1.push_back(r.crossingPath);
        ++tally.hop;
        ++hopHist[r.crossings];
      }
      continue;
    }

    ++stopReason[r.stopped];
    // One example per (start volume, reason). Without the key this prints the
    // same module twelve times, because the modules of one ring differ only in
    // phi and the straight ray treats them identically.
    const Acts::TrackingVolume* last =
        r.crossed.empty() ? volume : r.crossed.back().second;
    const std::string example =
        std::string(r.stopped) + " | " + volume->volumeName();
    if (printed < 12 && seen.insert(example).second) {
      std::printf(
          "  [%-17s] r %7.1f z %8.1f  %-24s after %zu crossing(s), stopped in "
          "%s\n",
          r.stopped, p.head<2>().norm(), p.z(), volume->volumeName().c_str(),
          r.crossings, last->volumeName().c_str());
      ++printed;
    }
  }

  const std::size_t resolved = module0.size() + module1.size();
  std::printf("resolved          %zu of %zu start points\n", resolved,
              starts.size());
  std::printf("  in the same volume                %zu\n", module0.size());
  std::printf("  after crossing a volume boundary  %zu\n\n", module1.size());

  std::printf("crossings needed, of the %zu that took any\n", module1.size());
  for (const auto& [n, count] : hopHist) {
    std::printf("  %zu  %zu\n", n, count);
  }
  std::printf("\nunresolved, by where the walk stopped\n");
  std::printf("  no volume at the start point      %zu\n", noVolume);
  std::printf("  no start layer                    %zu\n", noStartLayer);
  for (const auto& [why, count] : stopReason) {
    std::printf("  %-32s %zu\n", why.c_str(), count);
  }
  std::printf(
      "\n`left the tracker` is the ray reaching r > %.0f mm or |z| > %.0f mm.\n"
      "There is no next measurement there, so those are tracks ending rather\n"
      "than queries failing. The straight ray from the origin makes that the\n"
      "outermost module of whatever it passes through, which a curved track of\n"
      "the same start would not always be.\n\n",
      opts.rMax, opts.zMax);

  // Kept apart because they mean opposite things. The first is a track on the
  // outermost layer of its volume and is the geometry answering correctly. The
  // second is every layer of the volume having been asked and none of them
  // holding a module along this ray, which is a ray through the gaps.
  std::printf("the volume the track stands in had nothing\n");
  std::printf("  no layer ahead in it              %zu\n", sameNoLayer);
  std::printf("  layers ahead, none with a module  %zu\n\n", sameNoModule);

  std::printf("By the volume the track starts in.\n\n");
  std::printf("%-32s %8s %8s %8s\n", "volume", "starts", "same", "crossed");
  for (const auto& [name, t] : byVolume) {
    std::printf("%-32s %8zu %8zu %8zu\n", name.c_str(), t.n, t.same, t.hop);
  }
  std::printf("\nCrossings tried, and how many of them found a module.\n\n");
  for (const auto& [name, c] : crossings) {
    std::printf("%-64s %8zu %8zu\n", name.c_str(), c.first, c.second);
  }
  std::printf("\n");

  std::vector<double> approachAll = approach0, moduleAll = module0,
                      gapAll = gap0;
  approachAll.insert(approachAll.end(), approach1.begin(), approach1.end());
  moduleAll.insert(moduleAll.end(), module1.begin(), module1.end());
  gapAll.insert(gapAll.end(), gap1.begin(), gap1.end());

  std::printf("Path length [mm] from a module centre, straight ray.\n\n");
  report("to approach surface", approachAll);
  report("to the next module", moduleAll);
  report("approach to module", gapAll);
  std::printf("\n");
  report("same volume, module", module0);
  report("after a crossing, module", module1);
  report("the crossing itself", boundary1);

  std::printf(
      "\n`to approach surface` is what ACTS flies before a module is named\n"
      "today and `approach to module` is what is left for the stepper once it\n"
      "is. `to the next module` is what one call would cover if the propagator\n"
      "resolved the module itself. `resolved` against `start points` is whether\n"
      "that is possible at all.\n");
  return 0;
}
