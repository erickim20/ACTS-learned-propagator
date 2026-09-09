// Write out the real ODD module geometry, keyed by `geometry_id`.
//
// The teacher's destination is the cylinder r = r1 where r1 is the true next
// hit's radius. That surface is not in the detector and is derived from the
// answer. Retargeting it onto the module plane needs each module's centre,
// normal, local axes and bounds, and those live in the `TrackingGeometry`
// rather than in the hit CSV, which carries only the id.
//
// `surf1` in the teacher table is already the next hit's `geometry_id`
// (`make_teacher_pairs.py:198`), so this table joins onto it and no new
// simulation is needed. It does not join on the raw value, and the way it
// fails is quiet:
//
//   The hit CSV's endcap ids carry a non-zero EXTRA field in the low 8 bits,
//   the disc ring index 1 to 3. `TrackingGeometry` surfaces have extra = 0.
//   Barrel ids have extra = 0 on both sides and join directly. So a naive join
//   on the raw id keeps 100% of barrel rows and 0% of endcap rows, which is
//   27% of the teacher and looks like a partial geometry rather than a key
//   mismatch. Masking the low 8 bits takes it to 100%.
//
// `join_key` below is that masked value, so the join is right by construction
// rather than by whoever writes it remembering.
//
// Two distributions come out alongside, because nobody has measured either and
// each decides something:
//
//   MODULE TILT against the cylinder tangent. `chi2_gate.py:96` projects
//   residuals onto cylinder axes while comparing them against resolutions that
//   are module-local. The tilt is how wrong that is.
//
//   HALF-WIDTH and RADIUS. A module is a chord of the cylinder it sits on, so
//   the two surfaces separate by w^2/(2R) at the module edge. That is the size
//   of the error the current cylinder target makes on every jump.
//
// Built and run by run_in_image.sh.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "Acts/Definitions/Units.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Utilities/Logger.hpp"

#include "ActsPlugins/DD4hep/ConvertDD4hepDetector.hpp"
#include "DD4hep/Detector.h"

namespace {

/// Quantiles of a sample, by sorting. The samples here are 18,824 doubles at
/// most and this runs once.
struct Quantiles {
  double q(std::vector<double>& v, double p) const {
    if (v.empty()) {
      return 0.0;
    }
    std::sort(v.begin(), v.end());
    const auto i = static_cast<std::size_t>(
        p * static_cast<double>(v.size() - 1));
    return v[i];
  }
};

}  // namespace

int main(int argc, char** argv) {
  const std::string compact =
      argc > 1 ? argv[1] : "/opt/odd/xml/OpenDataDetector.xml";
  const std::string out = argc > 2 ? argv[2] : "modules.csv";

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

  std::FILE* f = std::fopen(out.c_str(), "w");
  if (f == nullptr) {
    std::fprintf(stderr, "cannot write %s\n", out.c_str());
    return 2;
  }
  // The local axes are the transform's own columns rather than anything
  // reconstructed from a normal. A plane's chart is its transform, so taking
  // the axes from it is the only way to be sure loc0 and loc1 here mean what
  // they mean inside ACTS.
  std::fprintf(f,
               "geometry_id,join_key,volume,boundary,layer,approach,sensitive,"
               "cx,cy,cz,"
               "u0x,u0y,u0z,u1x,u1y,u1z,nx,ny,nz,"
               "bounds_type,hx,hy,r_centre,tilt_rad,w_phi,sagitta_mm\n");

  std::vector<double> tiltBarrel, halfWidth, radius, sagitta;
  std::size_t nWritten = 0, nNonPlane = 0;

  tGeo->visitSurfaces(
      [&](const Acts::Surface* s) {
        if (s == nullptr || !s->isSensitive()) {
          return;
        }
        if (s->type() != Acts::Surface::SurfaceType::Plane) {
          ++nNonPlane;
          return;
        }
        // `localToGlobalTransform`, not `transform`. In 44.99.99 the accessor
        // is named for what it does and `transform` is the protected member,
        // so the obvious call compiles into a visibility error that reads like
        // a missing method.
        const auto& tf = s->localToGlobalTransform(gctx);
        const Acts::Vector3 c = tf.translation();
        const Acts::Vector3 u0 = tf.rotation().col(0);
        const Acts::Vector3 u1 = tf.rotation().col(1);
        const Acts::Vector3 n = tf.rotation().col(2);

        const auto v = s->bounds().values();
        // Reduced to a half-width and a half-length so one pair of columns
        // describes both shapes, with the raw values still recoverable from
        // bounds_type if anyone needs them.
        //
        // Switch on the TYPE, never on values().size(). Both of the ODD's
        // sensitive shapes carry four values and they mean different things:
        // RectangleBounds is (minX, minY, maxX, maxY) and TrapezoidBounds is
        // (halfXnegY, halfXposY, halfY, rotationAngle). Reading a trapezoid
        // as a rectangle produces a half-length out of an angle, and the
        // result is a plausible small number rather than an error.
        double hx = 0.0, hy = 0.0;
        const auto bt = static_cast<int>(s->bounds().type());
        if (s->bounds().type() == Acts::SurfaceBounds::eTrapezoid &&
            v.size() >= 3) {
          // the wider of the two ends, so the pair bounds the shape
          hx = std::max(v[0], v[1]);
          hy = v[2];
        } else if (v.size() >= 4) {
          hx = 0.5 * std::abs(v[2] - v[0]);
          hy = 0.5 * std::abs(v[3] - v[1]);
        }

        const double rc = std::hypot(c.x(), c.y());
        // Tilt against the cylinder tangent plane: 0 when the module normal is
        // radial, which is what a module tangent to the cylinder looks like.
        // Taken as an unsigned angle because the normal's sign is a convention
        // of the transform and not a property of the module.
        const Acts::Vector3 rhat =
            rc > 0 ? Acts::Vector3(c.x() / rc, c.y() / rc, 0.0)
                   : Acts::Vector3(1, 0, 0);
        const double tilt =
            std::acos(std::min(1.0, std::abs(n.dot(rhat))));
        // How far the module plane and the cylinder through its centre
        // separate at the module edge.
        //
        // The half-width that matters is the one that runs around phi, and
        // which of the two local axes that is cannot be assumed: it is a
        // convention of the DD4hep description, not of the geometry. So it is
        // chosen by projecting both axes onto phi_hat. Guessing wrong swaps a
        // 24 mm half-width for a 48 mm half-length and changes the answer by
        // four.
        const Acts::Vector3 phihat(-rhat.y(), rhat.x(), 0.0);
        const double wPhi = std::abs(u0.dot(phihat)) >= std::abs(u1.dot(phihat))
                                ? hx
                                : hy;
        const double sag = rc > 0 ? wPhi * wPhi / (2.0 * rc) : 0.0;

        const auto gid = s->geometryId();
        std::fprintf(f,
                     "%llu,%llu,%u,%u,%u,%u,%u,"
                     "%.6f,%.6f,%.6f,"
                     "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,"
                     "%d,%.6f,%.6f,%.6f,%.9f,%.6f,%.6f\n",
                     static_cast<unsigned long long>(gid.value()),
                     // the low 8 bits are `extra`, which the hit CSV uses for
                     // the endcap ring and the geometry leaves at zero
                     static_cast<unsigned long long>(gid.value() &
                                                     ~0xFFULL),
                     static_cast<unsigned>(gid.volume()),
                     static_cast<unsigned>(gid.boundary()),
                     static_cast<unsigned>(gid.layer()),
                     static_cast<unsigned>(gid.approach()),
                     static_cast<unsigned>(gid.sensitive()), c.x(), c.y(),
                     c.z(), u0.x(), u0.y(), u0.z(), u1.x(), u1.y(), u1.z(),
                     n.x(), n.y(), n.z(), bt, hx, hy, rc, tilt, wPhi, sag);
        ++nWritten;

        // Barrel modules only for the tilt distribution: an endcap module's
        // normal is along z and its angle to r_hat is 90 degrees by
        // construction, which would swamp the barrel number with a constant.
        if (std::abs(n.z()) < 0.5) {
          tiltBarrel.push_back(tilt);
          halfWidth.push_back(wPhi);
          radius.push_back(rc);
          sagitta.push_back(sag);
        }
      },
      false);
  std::fclose(f);

  std::printf("%s: %zu sensitive plane modules\n", out.c_str(), nWritten);
  if (nNonPlane != 0) {
    std::printf("  %zu sensitive surfaces were NOT planes and were skipped\n",
                nNonPlane);
  }

  Quantiles Q;
  auto line = [&](const char* name, std::vector<double> v, double scale,
                  const char* unit) {
    if (v.empty()) {
      std::printf("  %-22s (none)\n", name);
      return;
    }
    const double med = Q.q(v, 0.5);
    std::printf("  %-22s min %8.3f   median %8.3f   p90 %8.3f   max %8.3f  %s\n",
                name, Q.q(v, 0.0) * scale, med * scale, Q.q(v, 0.90) * scale,
                Q.q(v, 1.0) * scale, unit);
  };

  std::printf("\n=== barrel modules, %zu of them ===\n", tiltBarrel.size());
  line("tilt vs tangent", tiltBarrel, 1e3, "mrad");
  line("half-width", halfWidth, 1.0, "mm");
  line("radius", radius, 1.0, "mm");
  line("edge sagitta w^2/2R", sagitta, 1e3, "um");
  std::printf(
      "\nThe tilt is how far chi2_gate.py:96's cylinder axes are from the\n"
      "module-local axes the sensor resolutions are quoted in. The sagitta is\n"
      "how far the current cylinder target sits from the module plane at the\n"
      "module edge, which is the error the retarget removes.\n"
      "\nJoin the teacher on `join_key`, not on `geometry_id`. The hit CSV's\n"
      "endcap ids carry the disc ring in the low 8 bits and the geometry's do\n"
      "not, so the raw id keeps every barrel row and no endcap row, which is\n"
      "27%% of the teacher and reads as a partial geometry.\n");
  return 0;
}
