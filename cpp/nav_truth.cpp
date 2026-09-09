// Does the module the geometry walk names equal the module a propagation
// actually reaches next, and if not, why not.
//
// `nav_probe` measures that the walk of `nav_walk.hpp` names a module at 13,496
// of 18,824 start points. It does not check that the module is the right one. A
// seam built on a walk that names the wrong module transports to the wrong
// surface every time, and nothing above it recovers that, so this runs before
// anything is built.
//
// Two halves, one binary.
//
// The ground truth is a stock `Propagator<EigenStepper<>, Navigator>` over
// single muons from the origin, with an actor recording every sensitive surface
// it reaches, in order, and the momentum it had there. That is the sequence the
// CKF's own transport produces.
//
// The prediction is the walk, run from each arrival's own position and
// direction, compared against the next arrival's `geometryId`.
//
// Six arms. The first two are one-stage: they ask for the module from the
// module the track is standing on, 225 mm out. The last four
// are two-stage: they pick the layer from there and then ask for the module
// from where a helix advanced to that layer lands, which is a few millimetres
// out and is where the stock navigator asks from.
//
//   1-stage         the walk unchanged: the module whose straight-line
//                   intersection from the previous module is nearest.
//   1-stage helix   the same candidate list, ordered by solving the helix
//                   against each candidate plane at the fixed 2 T the model's
//                   own core runs at (`LearnedTransport.hpp`, `kBHelix`).
//                   This changes the answer 0 times; it is kept so that the
//                   run measures that rather than assuming it.
//   2-stage i0 2T   advance the helix from the arrival by the straight-ray path
//                   to the layer's approach surface, then ask that layer for its
//                   modules from there. Zero Newton iterations: the straight-ray
//                   path is not the helix's, so the landing point is near the
//                   approach surface and not on it.
//   2-stage i1 2T   one Newton iteration. Re-intersect the approach surface with
//                   a straight ray from the landing point and advance the helix
//                   again by that path.
//   2-stage i0 @B   the same at the field the detector actually has at the
//                   arrival, read from the field provider the propagation runs
//   2-stage i1 @B   in. Over 225 mm a 2 T helix and the map's ~1.9 T differ by
//                   about a millimetre of sagitta at 1 GeV, and the two-stage
//                   arms use the helix to MOVE rather than to order, which is a
//                   different use of it, so the difference is measured.
//
// The helix runs at the LOCAL momentum, read off the stepper at each arrival,
// not at the start state's. With material on the momentum falls along the track
// and a stale momentum bends the helix wrongly.
//
// WHAT THE WALK CANNOT NAME, BY CONSTRUCTION. It excludes the layer the track
// is standing on, because keeping it resolves that layer's own modules again
// and reproduces the stepper seam rather than measuring a propagator seam. So
// an arrival whose true successor is on the SAME layer is outside the walk's
// domain, not a miss it could fix. Those are counted and reported separately
// rather than folded into the match fraction.
//
// Why a miss is a miss. `Layer::compatibleSurfaces` takes its module candidates
// from `m_surfaceArray->neighbors(gctx, position, direction)`, a binned subset
// of the layer and not the whole layer, and then intersects each one with
// `surface.intersect(gctx, position, direction, boundaryTolerance)` at the
// navigator's default `BoundaryTolerance::None()` (`Layer.cpp`, section (B)).
// A miss is therefore one of two things and they need different fixes: the true
// module was never returned by `neighbors` and so was never intersected, or it
// was returned and the exact edge test rejected it. This asks `neighbors`
// directly for every miss, and separately re-runs the whole walk with the edge
// test relaxed, so the two are separated by measurement.
//
// Sample. The five pT bins of the muon table and the |eta| edges the transport
// census uses (`TransportCensus.hpp:135`), three values inside each bin, both
// signs of eta, both charges, six values of phi. No random number generator.
//
// That is not on its own enough for two runs to agree, and for a while they did
// not. `SurfaceArray::populateNeighborCache` sorts each bin's surface pack on
// the POINTERS, so `compatibleSurfaces` returns its candidates in an order that
// depends on the heap and changes between runs, and a non-stable sort on path
// length alone then breaks an exact tie differently each time. `nav_walk` sorts
// on `(path, geometryId)` for that reason. Two runs of this binary now produce
// byte-identical tables; before the tie-break they disagreed on 2 of 50,519
// transitions in the `Infinite()` row of section 5 and nowhere else.
//
// Run twice, material off and on. With material off the trajectory is exactly
// the field's own curve and a mismatch can only be geometry. With it on,
// scattering is in the answer, and the difference between the two is how much
// of the problem is scattering rather than the straight-ray approximation.
//
//   ./run_in_image.sh --map nav_truth      # the tree that carries the map
//   ./run_in_image.sh nav_truth            # the ODD as shipped, uniform 2 T
//
// The field is printed at four points before anything else. The map tree loads
// through three things that each fail silently (`run_in_image.sh`), and a
// uniform 2 T everywhere is what all three look like.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <vector>

#include "Acts/Definitions/Units.hpp"
#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Geometry/Layer.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Geometry/TrackingVolume.hpp"
#include "Acts/MagneticField/MagneticFieldContext.hpp"
#include "Acts/MagneticField/MagneticFieldProvider.hpp"
#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Propagator/MaterialInteractor.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Surfaces/BoundaryTolerance.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Surfaces/SurfaceArray.hpp"
#include "Acts/Utilities/AxisDefinitions.hpp"
#include "Acts/Utilities/IAxis.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Utilities/Result.hpp"

#include "ActsPlugins/DD4hep/ConvertDD4hepDetector.hpp"
#include "ActsPlugins/DD4hep/DD4hepFieldAdapter.hpp"
#include "ActsPlugins/Root/RootMaterialDecorator.hpp"
#include "DD4hep/Detector.h"

#include "LearnedTransport.hpp"
#include "nav_walk.hpp"

namespace {

// ---------------------------------------------------------------- the sample

/// The bin edges the whole muon table is taken in.
constexpr double kPtEdges[6] = {1.0, 2.0, 4.0, 8.0, 20.0, 50.0};
constexpr int kNPt = 5;

/// `TransportCensus.hpp:135`, the performance writer's own \|eta\| edges, so a
/// row here can be put beside one of its rows.
constexpr double kEtaEdges[5] = {0.6, 1.2, 1.8, 2.4, 3.0};
constexpr int kNEta = 5;

/// Three values inside each bin. Not the edges: an edge value sits on the
/// boundary of two rows and the assignment then depends on the comparison.
constexpr double kInside[3] = {1.0 / 6.0, 3.0 / 6.0, 5.0 / 6.0};

/// Six phi. The offset is there so that no track is fired along a module
/// boundary or through a ring's own symmetry axis, which would make the
/// straight and helix picks agree for a reason that is not physics.
constexpr int kNPhi = 6;
constexpr double kPhi0 = 0.1234;

/// `src/prop/chi2_gate.py`'s own edges, which every offline table in this
/// project is binned on. They are not the edges the sample is built on, so a
/// cell here holds whatever the three values inside each `kEtaEdges` band
/// happen to fall into and the cells are unevenly filled by construction. Both
/// binnings are dumped and the difference is the point.
constexpr double kGatePtEdges[6] = {0.0, 1.0, 2.0, 5.0, 10.0, 1e9};
constexpr double kGateEtaEdges[6] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0};
constexpr int kNGate = 5;

/// Which cell of a 5-edge-pair table a value falls in, clamped at both ends.
int binOf(double x, const double (&edges)[6]) {
  for (int i = 0; i < kNGate; ++i) {
    if (x < edges[i + 1]) {
      return i;
    }
  }
  return kNGate - 1;
}

struct Start {
  double pt = 0.0;   ///< GeV
  double eta = 0.0;
  double phi = 0.0;
  double q = 1.0;
  int ptBin = 0;
  int etaBin = 0;
  int gatePtBin = 0;
  int gateEtaBin = 0;
};

// ------------------------------------------------------------------ the arms

constexpr int kNArm = 6;
constexpr int kArmOneStage = 0;
constexpr int kArmFirstTwoStage = 2;

const char* kArmName[kNArm] = {"1-stage",       "1-stage helix",
                               "2-stage i0 2T", "2-stage i1 2T",
                               "2-stage i0 @B", "2-stage i1 @B"};

/// Nominal field and Newton iteration count for the four two-stage arms, in the
/// order of `kArmName`. `useLocal` takes the field from the provider at the
/// arrival instead of the nominal one.
struct TwoStageArm {
  int iterations;
  bool useLocal;
};
constexpr TwoStageArm kTwoStage[4] = {{0, false}, {1, false},
                                      {0, true},  {1, true}};

/// The floor on the path length of a module in the SECOND-stage query.
///
/// Stage one can use a positive floor because the query point is a module the
/// track has already crossed, so everything of interest is ahead. Stage two
/// cannot: the helix is advanced by a path length that is not its own, so it
/// lands near the approach surface and can land a little past it. A positive
/// floor would then drop the very modules the stage exists to find.
///
/// The size is set by the geometry, not by taste. The approach surface to
/// first module distance measures p10 10.3 mm, so a window this
/// deep cannot reach back past a module the helix has genuinely crossed except
/// in the thinnest tenth of the detector. The landing residual is reported
/// below, and it is what says whether the window was ever used.
constexpr double kStage2NearLimit = -10.0;

// ------------------------------------------------------------------ the truth

struct Arrival {
  const Acts::Surface* surface = nullptr;
  Acts::Vector3 pos = Acts::Vector3::Zero();
  Acts::Vector3 dir = Acts::Vector3::Zero();
  double path = 0.0;
  /// |p| in GeV and the charge in units of e, AT THIS ARRIVAL. Taken off the
  /// stepper rather than from the start state, because with material on the
  /// momentum falls along the track and a helix built on a stale momentum has
  /// the wrong radius.
  double pAbs = 0.0;
  double q = 0.0;
};

/// Every sensitive surface the propagation reaches, in order.
///
/// `act` returns `Result<void>`, not void: 44.99.99's actor_caller assigns the
/// return straight into the propagation's global result, so a void `act` is not
/// an Actor as far as the concept is concerned and the error arrives as a wall
/// of template substitution failures. `geom_probe.cpp:83`.
struct ArrivalRecorder {
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
    // The actor is called on every step, and a surface stays current for more
    // than one of them. Comparing against the last recorded one makes this one
    // entry per arrival rather than one per step.
    if (s == nullptr || !s->isSensitive() || s == result.last) {
      return Acts::Result<void>::success();
    }
    result.last = s;
    result.arrivals.push_back(
        {s, stepper.position(state.stepping), stepper.direction(state.stepping),
         state.stepping.pathAccumulated,
         stepper.absoluteMomentum(state.stepping) / Acts::UnitConstants::GeV,
         stepper.charge(state.stepping)});
    return Acts::Result<void>::success();
  }
};

// ------------------------------------------------------------------ the helix

/// Transverse arc length onto the plane n . (X - c) = 0 at an arbitrary field.
///
/// `collider_ml::solveArcLength` is the same Newton solve with `kBHelix` fixed
/// at 2 T by a compile-time constant, so it cannot answer the arms that run at
/// the local field. This is that function with the field as an argument, and
/// `checkSolver` below runs both on the same inputs and reports the largest
/// disagreement at B = 2, so the copy is verified against the original rather
/// than trusted.
///
/// NOTE THE UNIT OF `s`. dz/ds is pz/pt, so the parameter is the TRANSVERSE arc
/// length and the three-dimensional path along the helix is s * |p| / pt. Every
/// path length that comes out of ACTS is the three-dimensional one, so the two
/// are converted at every boundary between them and never mixed.
double solveArcLengthAtB(const Acts::Vector3& pos, const Acts::Vector3& mom,
                         double q, const Acts::Vector3& n,
                         const Acts::Vector3& c, double bz, bool* ok) {
  const double pt = std::hypot(mom.x(), mom.y());
  const double phi0 = std::atan2(mom.y(), mom.x());
  const double kappa = -0.3 * q * bz / pt * 1e-3;
  const double pzOverPt = mom.z() / pt;

  auto helixPos = [&](double s) {
    const double phi = phi0 + kappa * s;
    return Acts::Vector3(pos.x() + (std::sin(phi) - std::sin(phi0)) / kappa,
                         pos.y() - (std::cos(phi) - std::cos(phi0)) / kappa,
                         pos.z() + pzOverPt * s);
  };
  auto gAndDg = [&](double s, double* dg) {
    const double phi = phi0 + kappa * s;
    *dg = n.x() * std::cos(phi) + n.y() * std::sin(phi) + n.z() * pzOverPt;
    return n.dot(helixPos(s) - c);
  };

  double dg0 = 0.0;
  const double g0 = gAndDg(0.0, &dg0);
  if (std::abs(dg0) < 1e-3) {  // skimming the module edge on rather than
    *ok = false;               // crossing it
    return 0.0;
  }
  double s = -g0 / dg0;
  for (int i = 0; i < 8; ++i) {
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

/// The candidate whose helix intersection comes first. Returns nullptr when the
/// helix reaches none of them, which is itself an answer.
const Acts::Surface* helixPick(const std::vector<nav_walk::Candidate>& cands,
                               const Acts::GeometryContext& gctx,
                               const Acts::Vector3& pos,
                               const Acts::Vector3& mom, double q, double bz) {
  const Acts::Surface* best = nullptr;
  double bestS = 0.0;
  for (const nav_walk::Candidate& cand : cands) {
    // The plane's own chart. `localToGlobalTransform`, not `transform`: in
    // 44.99.99 the accessor is named for what it does and `transform` is the
    // protected member, so the obvious call is a visibility error that reads
    // like a missing method. geom_dump.cpp.
    const auto& tf = cand.surface->localToGlobalTransform(gctx);
    const Acts::Vector3 c = tf.translation();
    const Acts::Vector3 n = tf.rotation().col(2);
    bool ok = false;
    const double s = solveArcLengthAtB(pos, mom, q, n, c, bz, &ok);
    if (!ok) {
      continue;
    }
    if (best == nullptr || s < bestS) {
      best = cand.surface;
      bestS = s;
    }
  }
  return best;
}

/// Move along the helix by a THREE-DIMENSIONAL path length and return where it
/// ends up and which way it is then pointing.
///
/// This is the same helix `solveArcLengthAtB` solves against, evaluated instead
/// of solved, and the conversion between the two parameters is in one place.
/// The polar angle is unchanged by a solenoid, so the direction differs from
/// the starting one only in phi.
void helixAdvance(const Acts::Vector3& pos, const Acts::Vector3& dir,
                  double pAbs, double q, double bz, double path3D,
                  Acts::Vector3* outPos, Acts::Vector3* outDir) {
  const double sinTheta = std::hypot(dir.x(), dir.y());
  const double pt = pAbs * sinTheta;
  const double phi0 = std::atan2(dir.y(), dir.x());
  const double kappa = -0.3 * q * bz / pt * 1e-3;

  // A straight line, and the expression below divides by kappa. This is not
  // reachable in the tracker; it is here so the self-check can drive bz to zero
  // and compare against a straight ray.
  if (!std::isfinite(kappa) || std::abs(kappa) < 1e-14) {
    *outPos = pos + path3D * dir;
    *outDir = dir;
    return;
  }

  const double s = path3D * sinTheta;  // transverse arc length
  const double phi = phi0 + kappa * s;
  *outPos = Acts::Vector3(pos.x() + (std::sin(phi) - std::sin(phi0)) / kappa,
                          pos.y() - (std::cos(phi) - std::cos(phi0)) / kappa,
                          pos.z() + dir.z() / sinTheta * s);
  *outDir = Acts::Vector3(sinTheta * std::cos(phi), sinTheta * std::sin(phi),
                          dir.z());
}

// --------------------------------------------------------------- the counting

/// One arm's tally in one cell.
struct Cell {
  std::size_t n = 0;     ///< comparisons made
  std::size_t match = 0;
};

double pct(std::size_t a, std::size_t b) {
  return b == 0 ? std::nan("") : 100.0 * static_cast<double>(a) /
                                     static_cast<double>(b);
}

double quantile(std::vector<double>& v, double p) {
  if (v.empty()) {
    return std::nan("");
  }
  std::sort(v.begin(), v.end());
  const auto i =
      static_cast<std::size_t>(p * static_cast<double>(v.size() - 1));
  return v[i];
}

/// A miss category. The three are exhaustive and they need different fixes:
/// `kInList` is a wrong choice among the right candidates, `kRightLayer` is the
/// right layer failing to offer the module, `kWrongLayer` is a wrong layer.
constexpr int kNCat = 3;
constexpr int kInList = 0;
constexpr int kRightLayer = 1;
constexpr int kWrongLayer = 2;
const char* kCatName[kNCat] = {"in list", "not in list, right layer",
                               "not in list, wrong layer"};

/// Everything asked of one miss category. `neighborsHadIt` is the whole point
/// of Part A: it separates a module that `SurfaceArray::neighbors` never
/// returned, which no boundary tolerance can recover, from one it returned and
/// the edge test then threw away.
struct MissStats {
  std::size_t n = 0;
  std::size_t neighborsHadIt = 0;      ///< on the layer the walk chose
  std::size_t trueLayerHadIt = 0;      ///< on the true module's own layer
  std::size_t trueLayerAhead = 0;      ///< true layer was among the layers ahead
  std::size_t noPick = 0;              ///< the arm named nothing at all
  std::vector<double> dist;            ///< predicted to true module centre [mm]
  std::vector<double> nNeighbors;      ///< size of the returned bin pack
  std::vector<double> pt;              ///< start state pT [GeV]
  std::vector<double> aeta;            ///< start state |eta|
};

/// Is `truth` in the bin pack `SurfaceArray::neighbors` returns at this point.
/// `nPack` is written even when the answer is no, because an empty pack and a
/// pack that simply does not hold the module are different failures.
bool inNeighbors(const Acts::Layer* layer, const Acts::GeometryContext& gctx,
                 const Acts::Vector3& p, const Acts::Vector3& d,
                 const Acts::Surface* truth, std::size_t* nPack) {
  *nPack = 0;
  if (layer == nullptr) {
    return false;
  }
  const Acts::SurfaceArray* sa = layer->surfaceArray();
  if (sa == nullptr) {
    return false;
  }
  const std::span<const Acts::Surface* const> pack = sa->neighbors(gctx, p, d);
  *nPack = pack.size();
  return std::ranges::find(pack, truth) != pack.end();
}

// ------------------------------------------------------- the geometry readout

/// One layer's surface array, reduced to the numbers that decide whether a bin
/// neighbourhood covers a given transverse error.
struct LayerBinning {
  std::string volume;
  Acts::GeometryIdentifier::Value volumeId = 0;
  Acts::GeometryIdentifier::Value layerId = 0;
  std::string rep;                ///< the representative surface's own name
  std::size_t nModules = 0;
  double rMean = 0.0;             ///< mean module radius [mm]
  double zMean = 0.0;             ///< mean module |z| [mm]
  /// The module pitch along each binning axis, as the span of the module
  /// centres divided by how many DISTINCT positions they take along it. Not a
  /// nearest-neighbour centre distance: the strip layers are stereo pairs whose
  /// two sensors sit a few millimetres apart at the same azimuth, and a
  /// nearest-neighbour distance there reports the stereo separation rather than
  /// the spacing between modules.
  double pitch[2] = {0.0, 0.0};
  std::size_t nDistinct[2] = {0, 0};
  std::string axis[2];            ///< the binning direction, per axis
  std::size_t nBins[2] = {0, 0};
  double binSize[2] = {0.0, 0.0}; ///< in the axis's own unit
  double binSizeMm[2] = {0.0, 0.0};
  int maxNeighbor = 0;
};

/// How many distinct values a set takes, and the mean gap between adjacent
/// ones, at a given resolution. `wrap` closes the range, which azimuth needs
/// and a length does not.
void spacing(std::vector<double> v, double resolution, bool wrap, double period,
             std::size_t* nDistinct, double* pitch) {
  std::sort(v.begin(), v.end());
  std::vector<double> uniq;
  for (double x : v) {
    if (uniq.empty() || x - uniq.back() > resolution) {
      uniq.push_back(x);
    }
  }
  *nDistinct = uniq.size();
  if (uniq.size() < 2) {
    *pitch = std::nan("");
    return;
  }
  *pitch = wrap ? period / static_cast<double>(uniq.size())
                : (uniq.back() - uniq.front()) /
                      static_cast<double>(uniq.size() - 1);
}

/// Read every layer that holds sensitive surfaces out of the geometry.
///
/// The layers are reached through the surfaces rather than through the volume
/// tree, because `Surface::associatedLayer` is public and walking the confined
/// layer arrays is not.
std::vector<LayerBinning> readBinning(const Acts::TrackingGeometry& tGeo,
                                      const Acts::GeometryContext& gctx) {
  std::map<const Acts::Layer*, std::vector<Acts::Vector3>> centres;
  tGeo.visitSurfaces(
      [&](const Acts::Surface* s) {
        if (s == nullptr || !s->isSensitive()) {
          return;
        }
        const Acts::Layer* l = s->associatedLayer();
        if (l == nullptr || l->surfaceArray() == nullptr) {
          return;
        }
        centres[l].push_back(s->center(gctx));
      },
      false);

  std::vector<LayerBinning> out;
  for (const auto& [layer, pts] : centres) {
    LayerBinning b;
    const Acts::TrackingVolume* vol = layer->trackingVolume();
    b.volume = vol != nullptr ? vol->volumeName() : "(none)";
    b.volumeId = layer->geometryId().volume();
    b.layerId = layer->geometryId().layer();
    b.nModules = pts.size();

    for (const Acts::Vector3& p : pts) {
      b.rMean += p.head<2>().norm();
      b.zMean += std::abs(p.z());
    }
    b.rMean /= static_cast<double>(pts.size());
    b.zMean /= static_cast<double>(pts.size());

    const Acts::SurfaceArray* sa = layer->surfaceArray();
    b.maxNeighbor = sa->maxNeighborDistance();
    const Acts::Surface* rep = sa->surfaceRepresentation();
    b.rep = rep != nullptr ? rep->name() : "(none)";

    // What the axes are is not asked of `binningValues()`. In this build it
    // returns an empty vector for every layer: `SurfaceArray::
    // makeSurfaceGridLookup` builds the lookup with
    // `std::vector<AxisDirection>()` and nothing ever fills it. That is
    // upstream and not a wrong call here, but reading every axis as "unknown"
    // would lose the whole comparison, so the meaning is taken from the
    // representative surface's own chart, which is what `surfaceToGridLocal`
    // works in:
    //
    //   cylinder   grid local = (phi [rad], z [mm])
    //   disc       grid local = (r [mm], phi [rad])
    //
    // `surfaceToGridLocal` divides local 0 by the cylinder radius and leaves a
    // disc's local alone, and the bin sizes printed below agree: a barrel
    // axis 0 is a tenth of a radian and a disc axis 0 is a hundred millimetres.
    const bool isCylinder =
        rep != nullptr && rep->type() == Acts::Surface::Cylinder;
    b.axis[0] = isCylinder ? "phi [rad]" : "r [mm]";
    b.axis[1] = isCylinder ? "z [mm]" : "phi [rad]";

    const std::vector<const Acts::IAxis*> axes = sa->getAxes();
    for (std::size_t k = 0; k < axes.size() && k < 2; ++k) {
      b.nBins[k] = axes[k]->getNBins();
      b.binSize[k] = b.nBins[k] == 0 ? 0.0
                                     : (axes[k]->getMax() - axes[k]->getMin()) /
                                           static_cast<double>(b.nBins[k]);
      // A phi bin is an angle; what matters is the arc it subtends at the
      // radius the modules sit at. Everything else is already a length.
      b.binSizeMm[k] =
          b.axis[k][0] == 'p' ? b.binSize[k] * b.rMean : b.binSize[k];
    }

    // The module spacing along those same two axes, so a bin width can be read
    // in modules rather than only in millimetres.
    std::vector<double> phis, others;
    phis.reserve(pts.size());
    others.reserve(pts.size());
    for (const Acts::Vector3& q : pts) {
      phis.push_back(std::atan2(q.y(), q.x()));
      others.push_back(isCylinder ? q.z() : q.head<2>().norm());
    }
    std::size_t nPhi = 0, nOther = 0;
    double pitchPhi = 0.0, pitchOther = 0.0;
    // 1 mrad separates two module azimuths; 0.5 mm separates two z or r rows.
    spacing(phis, 1e-3, true, 2.0 * M_PI, &nPhi, &pitchPhi);
    spacing(others, 0.5, false, 0.0, &nOther, &pitchOther);
    const int iPhi = isCylinder ? 0 : 1;
    const int iOther = isCylinder ? 1 : 0;
    b.nDistinct[iPhi] = nPhi;
    b.pitch[iPhi] = pitchPhi * b.rMean;  // an arc, so a length like the rest
    b.nDistinct[iOther] = nOther;
    b.pitch[iOther] = pitchOther;

    out.push_back(std::move(b));
  }

  std::sort(out.begin(), out.end(), [](const LayerBinning& a,
                                       const LayerBinning& c) {
    return a.volumeId != c.volumeId ? a.volumeId < c.volumeId
                                    : a.layerId < c.layerId;
  });
  return out;
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

  // The material map, which is a separate ROOT file and not in the compact.
  //
  // `convertDD4hepDetector`'s `matDecorator` argument defaults to nullptr
  // (`ConvertDD4hepDetector.hpp`), so the obvious two-argument call builds a
  // geometry whose surfaces carry no material at all. Nothing says so, and the
  // failure is not a crash: `MaterialInteractor` runs, finds nothing to apply,
  // and the material-on rows come out bit-identical to the material-off rows.
  // That is what the first run of this probe produced.
  //
  // The map sits beside the compact in every tree, so its path is derived from
  // the compact rather than given, and both are reported.
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

  Acts::GeometryContext gctx =
      Acts::GeometryContext::dangerouslyDefaultConstruct();
  Acts::MagneticFieldContext mctx;

  // The detector's own field, not a constant. Whether the map is actually in it
  // is printed rather than assumed: `run_in_image.sh` lists the three ways the
  // map tree fails to load without saying so, and all three end in a uniform
  // 2 T.
  auto bfield =
      std::make_shared<ActsPlugins::DD4hepFieldAdapter>(detector.field());
  auto fieldCache = bfield->makeCache(mctx);

  const bool gen1 = tGeo->geometryVersion() ==
                    Acts::TrackingGeometry::GeometryVersion::Gen1;
  std::printf("geometry version  %s\n", gen1 ? "Gen1" : "Gen3");

  // Whether the decoration took. A count of zero here is what "material on"
  // silently means when it means nothing.
  //
  // The second argument is `restrictToSensitives` and it defaults to TRUE
  // (`TrackingGeometry.hpp:123`). Counting with the default finds zero even on
  // a correctly decorated ODD, because the map puts its material on the layers'
  // approach surfaces and not on the modules, and zero then reads as a
  // decoration that failed.
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
        "\nNo surface carries material. The material-on rows below are not a\n"
        "measurement of material: `MaterialInteractor` has nothing to apply and\n"
        "they will reproduce the material-off rows exactly.\n");
  }

  std::printf("\nField the propagation will run in [T].\n\n");
  std::printf("%-28s %10s %10s %10s\n", "point [mm]", "Bx", "By", "Bz");
  const Acts::Vector3 probes[4] = {{0, 0, 0},
                                   {800, 0, 0},
                                   {0, 0, 2000},
                                   {600, 600, -1500}};
  double bzSpread = 0.0, bzFirst = 0.0;
  for (int i = 0; i < 4; ++i) {
    auto r = bfield->getField(probes[i], fieldCache);
    if (!r.ok()) {
      std::printf("  field lookup failed at probe %d\n", i);
      return 2;
    }
    const Acts::Vector3 b = *r / Acts::UnitConstants::T;
    char where[32];
    std::snprintf(where, sizeof(where), "(%.0f, %.0f, %.0f)", probes[i].x(),
                  probes[i].y(), probes[i].z());
    std::printf("%-28s %10.4f %10.4f %10.4f\n", where, b.x(), b.y(), b.z());
    if (i == 0) {
      bzFirst = b.z();
    }
    bzSpread = std::max(bzSpread, std::abs(b.z() - bzFirst));
  }
  const bool uniform = bzSpread < 1e-6;
  std::printf(
      "\n%s\n",
      uniform ? "Bz is the same at all four points. This is the analytic "
                "solenoid, not the map."
              : "Bz varies across the tracker, so this is the map.");

  // ------------------------------------------------------- the binning table
  //
  // Static geometry, so it is read once and does not belong inside either run.
  {
    const std::vector<LayerBinning> bins = readBinning(*tGeo, gctx);
    std::printf(
        "\n================================================================\n"
        "1. What the surface array of each layer is binned on\n"
        "================================================================\n\n"
        "`Layer::compatibleSurfaces` takes its module candidates from\n"
        "`m_surfaceArray->neighbors(gctx, position, direction)`, which\n"
        "intersects the layer's REPRESENTATIVE surface with the straight ray,\n"
        "converts the hit to a grid bin, and returns that bin's surfaces plus\n"
        "those of its neighbours out to `nb` bins. Modules are flood-filled\n"
        "into every bin they overlap, so a bin holds more than one.\n\n"
        "A transverse error smaller than `nb` bin widths is covered. One\n"
        "larger than that is not, and the module is never intersected at all.\n\n"
        "`bin` is the bin width in the axis's own unit and `bin mm` the same at\n"
        "the mean module radius. `pitch` is the module spacing along that axis,\n"
        "as the span of the module centres over the number of DISTINCT\n"
        "positions `nd` they take along it. `bin mm / pitch` is how many modules\n"
        "wide one bin is, and `nb` bins of that on each side is what a\n"
        "transverse error has to stay inside.\n\n"
        "`SurfaceArray::binningValues()` is not the source of the axis names.\n"
        "It returns an empty vector for every layer in this build, because\n"
        "`makeSurfaceGridLookup` constructs the lookup with an empty\n"
        "`std::vector<AxisDirection>`. The names come from the representative\n"
        "surface instead, which is the chart `surfaceToGridLocal` works in.\n\n");
    std::printf(
        "%-27s %4s %6s %8s %8s  %-9s %5s %8s %8s %5s %8s %7s  %-9s %5s %8s "
        "%8s %5s %8s %7s  %3s\n",
        "volume", "lay", "n", "r [mm]", "|z| [mm]", "axis 0", "bins", "bin",
        "bin mm", "nd", "pitch", "bins/mod", "axis 1", "bins", "bin", "bin mm",
        "nd", "pitch", "bins/mod", "nb");
    for (const LayerBinning& b : bins) {
      std::printf(
          "%-27s %4llu %6zu %8.1f %8.1f  %-9s %5zu %8.4f %8.2f %5zu %8.2f "
          "%7.2f  %-9s %5zu %8.4f %8.2f %5zu %8.2f %7.2f  %3d\n",
          b.volume.c_str(), static_cast<unsigned long long>(b.layerId),
          b.nModules, b.rMean, b.zMean, b.axis[0].c_str(), b.nBins[0],
          b.binSize[0], b.binSizeMm[0], b.nDistinct[0], b.pitch[0],
          b.binSizeMm[0] / b.pitch[0], b.axis[1].c_str(), b.nBins[1],
          b.binSize[1], b.binSizeMm[1], b.nDistinct[1], b.pitch[1],
          b.binSizeMm[1] / b.pitch[1], b.maxNeighbor);
    }
    std::printf(
        "\nThe representative surface per volume, which is what the ray is\n"
        "intersected against before the bin is looked up:\n\n");
    std::map<std::string, std::map<std::string, std::size_t>> reps;
    for (const LayerBinning& b : bins) {
      ++reps[b.volume][b.rep];
    }
    for (const auto& [vol, byRep] : reps) {
      for (const auto& [rep, n] : byRep) {
        std::printf("  %-22s %-32s %zu layers\n", vol.c_str(), rep.c_str(), n);
      }
    }
  }

  Acts::Navigator::Config navCfg{tGeo};
  navCfg.resolveSensitive = true;
  navCfg.resolveMaterial = true;
  navCfg.resolvePassive = false;

  using Stepper = Acts::EigenStepper<>;
  using Prop = Acts::Propagator<Stepper, Acts::Navigator>;
  // braces, not parens: with parentheses this parses as a function declaration.
  // geom_probe.cpp:354.
  Prop prop{Stepper(bfield), Acts::Navigator(navCfg)};

  // ----------------------------------------------------------- the sample
  std::vector<Start> starts;
  for (int ip = 0; ip < kNPt; ++ip) {
    for (double fp : kInside) {
      // Inside the bin in log pT, because the bins themselves are geometric.
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
                s.gatePtBin = binOf(pt, kGatePtEdges);
                s.gateEtaBin = binOf(absEta, kGateEtaEdges);
                starts.push_back(s);
              }
            }
          }
        }
      }
    }
  }
  std::printf("\nstart states      %zu\n", starts.size());

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

  // --------------------------------------------------------- solver checks
  //
  // Two things are checked before anything is measured, because both would be
  // silent and both would move every number in Part B.
  {
    // (1) The arbitrary-field solve is `collider_ml::solveArcLength` with the
    // field lifted out of a compile-time constant, not a re-derivation of it.
    double worst = 0.0;
    int checked = 0;
    for (int i = 0; i < 64; ++i) {
      collider_ml::Jump j;
      const double a = 0.11 * i;
      j.pos = collider_ml::Vec3(30.0 + i, 10.0 * std::sin(a), 5.0 * i);
      j.mom = collider_ml::Vec3(1.0 + 0.1 * i, 0.5 * std::cos(a), 0.7 * i);
      j.q = (i % 2 == 0) ? 1.0 : -1.0;
      j.c = j.pos + collider_ml::Vec3(50.0, 20.0 * std::cos(a), 10.0);
      j.n = collider_ml::Vec3(1.0, 0.3 * std::sin(a), 0.1).normalized();
      bool okA = false, okB = false;
      const double sA = collider_ml::solveArcLength(j, &okA);
      const double sB = solveArcLengthAtB(j.pos, j.mom, j.q, j.n, j.c,
                                          collider_ml::detail::kBHelix, &okB);
      if (okA != okB) {
        worst = std::numeric_limits<double>::infinity();
        break;
      }
      if (okA) {
        worst = std::max(worst, std::abs(sA - sB));
        ++checked;
      }
    }
    std::printf("solver check      %d cases, largest |s - s| = %.3g mm\n",
                checked, worst);
    if (!(worst < 1e-9)) {
      std::fprintf(stderr,
                   "the arbitrary-field solver does not reproduce "
                   "collider_ml::solveArcLength at 2 T; stopping\n");
      return 2;
    }

    // (2) `helixAdvance` takes a THREE-DIMENSIONAL path and `solveArcLengthAtB`
    // returns a TRANSVERSE arc length. Getting that conversion backwards does
    // not fail, it lands the helix short by a factor of sin(theta), which is a
    // 20 to 90 % error in the lever arm and would be read as the two-stage idea
    // not working. So: drive the field to zero and require a straight line, and
    // solve against a plane and require the advance to land on it.
    double worstLine = 0.0, worstPlane = 0.0;
    for (int i = 0; i < 64; ++i) {
      const double a = 0.09 * i;
      const double theta = 0.15 + 0.04 * (i % 60);
      const Acts::Vector3 d(std::sin(theta) * std::cos(a),
                            std::sin(theta) * std::sin(a), std::cos(theta));
      const Acts::Vector3 p(20.0 + i, 5.0 * std::sin(a), 3.0 * i);
      const double pAbs = 1.0 + 0.3 * (i % 20);
      const double q = (i % 2 == 0) ? 1.0 : -1.0;
      const double L = 50.0 + 4.0 * i;

      Acts::Vector3 p1, d1;
      helixAdvance(p, d, pAbs, q, 0.0, L, &p1, &d1);
      worstLine = std::max(worstLine, (p1 - (p + L * d)).norm());

      const Acts::Vector3 n =
          Acts::Vector3(1.0, 0.4 * std::sin(a), 0.2 * std::cos(a)).normalized();
      const Acts::Vector3 c = p + 220.0 * d;
      bool ok = false;
      const double sT = solveArcLengthAtB(p, pAbs * d, q, n, c, 2.0, &ok);
      if (!ok) {
        continue;
      }
      // transverse arc -> three-dimensional path
      const double path3D = sT / std::hypot(d.x(), d.y());
      helixAdvance(p, d, pAbs, q, 2.0, path3D, &p1, &d1);
      worstPlane = std::max(worstPlane, std::abs(n.dot(p1 - c)));
    }
    std::printf(
        "advance check     straight line at B=0 off by %.3g mm; landing on the "
        "solved plane off by %.3g mm\n\n",
        worstLine, worstPlane);
    if (!(worstLine < 1e-9) || !(worstPlane < 1e-6)) {
      std::fprintf(stderr, "helixAdvance is not the helix solveArcLengthAtB "
                           "solves against; stopping\n");
      return 2;
    }
  }

  // ------------------------------------------------------------- the runs
  const nav_walk::Options walkOpts;

  /// The two relaxed-edge scans are seven extra walks and seven extra layer
  /// resolves per transition and they dominate the run. `NAV_TRUTH_TOL=0`
  /// leaves them out. They share no counter with the match tables, so leaving
  /// them out cannot move a match fraction; their own tables then read zero and
  /// say so.
  const char* tolEnv = std::getenv("NAV_TRUTH_TOL");
  const bool doTol = !(tolEnv != nullptr && std::string(tolEnv) == "0");
  std::printf("relaxed-edge scan %s\n", doTol ? "on" : "OFF (NAV_TRUTH_TOL=0)");

  Cell byCell[2][kNArm][kNPt][kNEta];
  Cell total[2][kNArm];
  std::size_t nArrivals[2] = {0, 0};
  std::size_t nSameLayer[2] = {0, 0};
  std::size_t nWalkFailed[2] = {0, 0};
  std::map<std::string, std::size_t> walkStop[2];
  std::size_t nPropFailed[2] = {0, 0};

  /// The three pooled counts above, split by the pT and \|eta\| bin of the
  /// transition's own start state. `offeredCell` counts every consecutive pair
  /// of arrivals the loop reaches, so it is the denominator the same-layer and
  /// walk-failed fractions are taken against, and
  /// `offered = sameLayer + walkFailed + n` holds cell by cell.
  std::size_t offeredCell[2][kNPt][kNEta] = {};
  std::size_t sameLayerCell[2][kNPt][kNEta] = {};
  std::size_t walkFailedCell[2][kNPt][kNEta] = {};
  /// The same five counts on `chi2_gate`'s edges instead.
  Cell byGate[2][kNArm][kNGate][kNGate];
  std::size_t offeredGate[2][kNGate][kNGate] = {};
  std::size_t sameLayerGate[2][kNGate][kNGate] = {};
  std::size_t walkFailedGate[2][kNGate][kNGate] = {};
  std::size_t fellBackGate[2][kNArm][kNGate][kNGate] = {};

  MissStats miss[2][kNArm][kNCat];

  /// How many modules the chosen layer offered, over ALL comparisons. An arm
  /// can only differ from another where this is above one.
  std::map<std::size_t, std::size_t> candHist[2];
  /// How often an arm picked something other than the one-stage arm did.
  std::size_t armDisagree[2][kNArm] = {};
  /// The second stage returned nothing and the arm fell back to the first.
  std::size_t nFellBack[2][kNArm] = {};
  /// The same, per cell. Only the four two-stage arms ever fall back.
  std::size_t fellBackCell[2][kNArm][kNPt][kNEta] = {};

  /// Where the helix landed, per two-stage arm. `sagitta` is how far the
  /// landing point is from the straight ray's own point at the same path
  /// length, which is estimated at 3.8 mm.
  /// `residual` is the signed straight-ray path from the landing point to the
  /// approach surface, so it is what one more Newton iteration would remove.
  std::vector<double> sagitta[2][4];
  std::vector<double> residual[2][4];
  std::size_t nNegativePath[2][4] = {};
  /// The same sagitta split by pT bin, for the local-field arm with no
  /// iteration, material off. The sample is log-uniform in pT and the overall
  /// median is therefore not the 1 GeV number.
  std::vector<double> sagittaByPt[kNPt];

  // The relaxed-boundary pass, material off only, because the 414 it addresses
  // is the material-off split.
  struct TolArm {
    const char* name;
    Acts::BoundaryTolerance tol;
  };
  const std::vector<TolArm> tolArms = {
      {"1 mm", Acts::BoundaryTolerance::AbsoluteEuclidean(1.0)},
      {"2 mm", Acts::BoundaryTolerance::AbsoluteEuclidean(2.0)},
      {"5 mm", Acts::BoundaryTolerance::AbsoluteEuclidean(5.0)},
      {"10 mm", Acts::BoundaryTolerance::AbsoluteEuclidean(10.0)},
      {"20 mm", Acts::BoundaryTolerance::AbsoluteEuclidean(20.0)},
      {"50 mm", Acts::BoundaryTolerance::AbsoluteEuclidean(50.0)},
      {"infinite", Acts::BoundaryTolerance::Infinite()}};
  const std::size_t nTol = tolArms.size();
  std::vector<std::size_t> tolN(nTol, 0), tolMatch(nTol, 0);
  std::vector<std::size_t> tolNewWrong(nTol, 0), tolRecovered(nTol, 0);
  // Restricted to the baseline's `not in list, right layer` misses.
  std::vector<std::size_t> tolRlOffered(nTol, 0), tolRlMatched(nTol, 0);
  std::size_t nRightLayerBase = 0;
  // How long the candidate list gets. A tolerance that recovers a module by
  // admitting six others it does not cross is doing something visible here.
  std::vector<std::vector<double>> tolCands(nTol);
  // And how often the walk came out on a different LAYER. `compatibleLayers`
  // is untouched, but `resolveIn` stops at the first layer ahead that offers a
  // module, and a relaxed edge test can make a nearer layer offer one where it
  // offered nothing before. That is a second way the tolerance changes the
  // answer and it is not the edge test on the layer in question.
  std::vector<std::size_t> tolLayerMoved(nTol, 0);

  // The same scan on the SECOND stage of the best two-stage arm, which is the
  // configuration a seam would actually run. The first-stage scan measures the
  // edge test at a 225 mm lever arm; this one measures it at the lever arm the
  // second stage leaves, and the two are different questions. Arm index 2 of
  // `kTwoStage`, the local field with no iteration.
  std::vector<std::size_t> t2N(nTol, 0), t2Match(nTol, 0);
  std::vector<std::size_t> t2NewWrong(nTol, 0), t2Recovered(nTol, 0);
  std::vector<std::size_t> t2RlOffered(nTol, 0), t2RlMatched(nTol, 0);
  std::vector<std::vector<double>> t2Cands(nTol);
  std::size_t n2RightLayerBase = 0;

  for (int mat = 0; mat < 2; ++mat) {
    for (const Start& s : starts) {
      const auto start = makeStart(s);

      std::vector<Arrival> arrivals;
      if (mat == 0) {
        Prop::Options<Acts::ActorList<ArrivalRecorder>> opt(gctx, mctx);
        opt.pathLimit = 6000.0;
        opt.maxSteps = 10000;
        auto res = prop.propagate(start, opt);
        if (!res.ok()) {
          ++nPropFailed[mat];
          continue;
        }
        arrivals = res->get<ArrivalRecorder::result_type>().arrivals;
      } else {
        Prop::Options<Acts::ActorList<ArrivalRecorder, Acts::MaterialInteractor>>
            opt(gctx, mctx);
        opt.pathLimit = 6000.0;
        opt.maxSteps = 10000;
        // Defaults are multipleScattering and energyLoss both on. Named here
        // because "material on" has to mean something specific.
        auto& mi = opt.actorList.get<Acts::MaterialInteractor>();
        mi.multipleScattering = true;
        mi.energyLoss = true;
        auto res = prop.propagate(start, opt);
        if (!res.ok()) {
          ++nPropFailed[mat];
          continue;
        }
        arrivals = res->get<ArrivalRecorder::result_type>().arrivals;
      }

      nArrivals[mat] += arrivals.size();
      if (arrivals.empty()) {
        continue;
      }

      for (std::size_t i = 0; i + 1 < arrivals.size(); ++i) {
        const Arrival& a = arrivals[i];
        const Acts::Surface* truth = arrivals[i + 1].surface;

        const Acts::Layer* startLayer = a.surface->associatedLayer();
        if (startLayer == nullptr) {
          continue;
        }
        ++offeredCell[mat][s.ptBin][s.etaBin];
        ++offeredGate[mat][s.gatePtBin][s.gateEtaBin];
        // Outside the walk's domain by construction: it excludes the layer the
        // track stands on, so a successor there is not a module it could name.
        if (truth->associatedLayer() == startLayer) {
          ++nSameLayer[mat];
          ++sameLayerCell[mat][s.ptBin][s.etaBin];
          ++sameLayerGate[mat][s.gatePtBin][s.gateEtaBin];
          continue;
        }

        const nav_walk::Result r = nav_walk::walk(*tGeo, gctx, a.pos, a.dir,
                                                  startLayer, a.surface,
                                                  walkOpts, *logger);
        if (!r.ok) {
          ++nWalkFailed[mat];
          ++walkFailedCell[mat][s.ptBin][s.etaBin];
          ++walkFailedGate[mat][s.gatePtBin][s.gateEtaBin];
          ++walkStop[mat][r.stopped];
          continue;
        }

        const double localBz = [&] {
          auto fr = bfield->getField(a.pos, fieldCache);
          return fr.ok() ? (*fr).z() / Acts::UnitConstants::T : 2.0;
        }();
        const Acts::Vector3 mom = a.dir * a.pAbs;

        // ------------------------------------------------ the six arms
        const Acts::Surface* picks[kNArm] = {};
        const std::vector<nav_walk::Candidate>* used[kNArm] = {};
        Acts::Vector3 qpos[kNArm], qdir[kNArm];

        picks[0] = r.nearest;
        picks[1] = helixPick(r.candidates, gctx, a.pos, mom, a.q,
                             collider_ml::detail::kBHelix);
        for (int k = 0; k < 2; ++k) {
          used[k] = &r.candidates;
          qpos[k] = r.queryPos;
          qdir[k] = a.dir;
        }

        // The second stage's own candidate lists have to outlive the loop that
        // fills them, because `used[]` points into them.
        nav_walk::LayerResolve s2[4];
        for (int k = 0; k < 4; ++k) {
          const int arm = kArmFirstTwoStage + k;
          const double bz =
              kTwoStage[k].useLocal ? localBz : collider_ml::detail::kBHelix;

          Acts::Vector3 p1 = a.pos, d1 = a.dir;
          double L = r.layerPath;
          for (int it = 0; it <= kTwoStage[k].iterations; ++it) {
            helixAdvance(p1, d1, a.pAbs, a.q, bz, L, &p1, &d1);
            if (it == kTwoStage[k].iterations) {
              break;
            }
            // Re-intersect the approach surface from where the helix landed.
            // `Infinite()`, because this is a Newton step onto the surface and
            // not a navigation decision: an edge test here would abandon the
            // iteration whenever the landing point drifted off the bounds.
            const auto is =
                r.layerSurface
                    ->intersect(gctx, p1, d1, Acts::BoundaryTolerance::Infinite())
                    .closest();
            if (!is.isValid()) {
              L = 0.0;
              break;
            }
            L = is.pathLength();
          }

          const double sag = (p1 - (a.pos + r.layerPath * a.dir)).norm();
          sagitta[mat][k].push_back(sag);
          if (mat == 0 && k == 2) {  // local field, no iteration
            sagittaByPt[s.ptBin].push_back(sag);
          }
          {
            const auto is =
                r.layerSurface
                    ->intersect(gctx, p1, d1, Acts::BoundaryTolerance::Infinite())
                    .closest();
            residual[mat][k].push_back(is.isValid() ? is.pathLength()
                                                    : std::nan(""));
          }

          s2[k] = nav_walk::resolveOnLayer(*r.layer, gctx, p1, d1,
                                           kStage2NearLimit);
          if (s2[k].ok) {
            picks[arm] = s2[k].nearest;
            used[arm] = &s2[k].candidates;
            if (s2[k].nearestPath < 0.0) {
              ++nNegativePath[mat][k];
            }
          } else {
            // The chosen layer offered nothing from the landing point. Falling
            // back to the first stage's answer is what a deployed version would
            // do; it is counted so the arm is not credited with an answer the
            // second stage did not produce.
            ++nFellBack[mat][arm];
            ++fellBackCell[mat][arm][s.ptBin][s.etaBin];
            ++fellBackGate[mat][arm][s.gatePtBin][s.gateEtaBin];
            picks[arm] = r.nearest;
            used[arm] = &r.candidates;
          }
          qpos[arm] = p1;
          qdir[arm] = d1;
        }
        // Where the local-field, no-iteration arm ended up, for the second-stage
        // tolerance scan below.
        const Acts::Vector3 bestPos = qpos[kArmFirstTwoStage + 2];
        const Acts::Vector3 bestDir = qdir[kArmFirstTwoStage + 2];

        ++candHist[mat][r.candidates.size()];

        for (int arm = 0; arm < kNArm; ++arm) {
          Cell& c = byCell[mat][arm][s.ptBin][s.etaBin];
          Cell& g = byGate[mat][arm][s.gatePtBin][s.gateEtaBin];
          ++c.n;
          ++g.n;
          ++total[mat][arm].n;
          if (picks[arm] != picks[kArmOneStage]) {
            ++armDisagree[mat][arm];
          }
          if (picks[arm] == truth) {
            ++c.match;
            ++g.match;
            ++total[mat][arm].match;
            continue;
          }

          // ------------------------------------------- the miss, classified
          const bool inList =
              std::ranges::any_of(*used[arm],
                                  [&](const nav_walk::Candidate& k) {
                                    return k.surface == truth;
                                  });
          const int cat = inList ? kInList
                                 : (truth->associatedLayer() == r.layer
                                        ? kRightLayer
                                        : kWrongLayer);
          MissStats& m = miss[mat][arm][cat];
          ++m.n;
          m.pt.push_back(s.pt);
          m.aeta.push_back(std::abs(s.eta));
          if (picks[arm] == nullptr) {
            ++m.noPick;
          } else {
            m.dist.push_back(
                (picks[arm]->center(gctx) - truth->center(gctx)).norm());
          }

          // The binary question. Was the true module ever returned by the
          // binned lookup, asked at the same point and along the same direction
          // the arm's own `compatibleSurfaces` call used.
          std::size_t nPack = 0;
          if (inNeighbors(r.layer, gctx, qpos[arm], qdir[arm], truth, &nPack)) {
            ++m.neighborsHadIt;
          }
          m.nNeighbors.push_back(static_cast<double>(nPack));

          // The same question of the module's own layer, which is a different
          // surface array whenever the layer was wrong.
          const Acts::Layer* tl = truth->associatedLayer();
          std::size_t nPack2 = 0;
          if (inNeighbors(tl, gctx, qpos[arm], qdir[arm], truth, &nPack2)) {
            ++m.trueLayerHadIt;
          }
          if (std::ranges::find(r.layersAhead, tl) != r.layersAhead.end()) {
            ++m.trueLayerAhead;
          }
        }

        // ------------------------------------------ the relaxed-edge pass
        if (mat == 0 && doTol) {
          const bool baseMatched = picks[kArmOneStage] == truth;
          const bool baseRightLayer =
              !baseMatched &&
              !std::ranges::any_of(r.candidates,
                                   [&](const nav_walk::Candidate& k) {
                                     return k.surface == truth;
                                   }) &&
              truth->associatedLayer() == r.layer;
          if (baseRightLayer) {
            ++nRightLayerBase;
          }
          for (std::size_t t = 0; t < nTol; ++t) {
            nav_walk::Options o = walkOpts;
            o.surfaceTolerance = tolArms[t].tol;
            const nav_walk::Result rt = nav_walk::walk(
                *tGeo, gctx, a.pos, a.dir, startLayer, a.surface, o, *logger);
            ++tolN[t];
            const bool matched = rt.ok && rt.nearest == truth;
            if (matched) {
              ++tolMatch[t];
            }
            if (baseMatched && !matched) {
              ++tolNewWrong[t];
            }
            if (!baseMatched && matched) {
              ++tolRecovered[t];
            }
            if (baseRightLayer) {
              if (rt.ok && std::ranges::any_of(
                               rt.candidates,
                               [&](const nav_walk::Candidate& k) {
                                 return k.surface == truth;
                               })) {
                ++tolRlOffered[t];
              }
              if (matched) {
                ++tolRlMatched[t];
              }
            }
            tolCands[t].push_back(
                rt.ok ? static_cast<double>(rt.candidates.size()) : 0.0);
            if (rt.ok && rt.layer != r.layer) {
              ++tolLayerMoved[t];
            }
          }

          // The second stage of the best arm, at the same tolerances.
          const int bestArm = kArmFirstTwoStage + 2;
          const bool base2Matched = picks[bestArm] == truth;
          const bool base2RightLayer =
              !base2Matched &&
              !std::ranges::any_of(*used[bestArm],
                                   [&](const nav_walk::Candidate& k) {
                                     return k.surface == truth;
                                   }) &&
              truth->associatedLayer() == r.layer;
          if (base2RightLayer) {
            ++n2RightLayerBase;
          }
          for (std::size_t t = 0; t < nTol; ++t) {
            const nav_walk::LayerResolve q =
                nav_walk::resolveOnLayer(*r.layer, gctx, bestPos, bestDir,
                                         kStage2NearLimit, tolArms[t].tol);
            ++t2N[t];
            const bool matched = q.ok && q.nearest == truth;
            if (matched) {
              ++t2Match[t];
            }
            if (base2Matched && !matched) {
              ++t2NewWrong[t];
            }
            if (!base2Matched && matched) {
              ++t2Recovered[t];
            }
            if (base2RightLayer) {
              if (q.ok && std::ranges::any_of(q.candidates,
                                              [&](const nav_walk::Candidate& k) {
                                                return k.surface == truth;
                                              })) {
                ++t2RlOffered[t];
              }
              if (matched) {
                ++t2RlMatched[t];
              }
            }
            t2Cands[t].push_back(q.ok ? static_cast<double>(q.candidates.size())
                                      : 0.0);
          }
        }
      }
    }
  }

  // ------------------------------------------------------------- the report
  const char* matName[2] = {"material off", "material on"};

  std::printf("\n");
  for (int mat = 0; mat < 2; ++mat) {
    std::printf("%-14s  arrivals %7zu   propagations failed %zu\n",
                matName[mat], nArrivals[mat], nPropFailed[mat]);
    std::printf("                successor on the same layer, outside the "
                "walk's domain  %zu\n",
                nSameLayer[mat]);
    std::printf("                walk named nothing  %zu", nWalkFailed[mat]);
    for (const auto& [why, n] : walkStop[mat]) {
      std::printf("   [%s %zu]", why.c_str(), n);
    }
    std::printf("\n");
  }

  std::printf(
      "\n================================================================\n"
      "2. Match fraction, by arm\n"
      "================================================================\n\n"
      "The module the arm names against the module the propagation reaches\n"
      "next. Rows are pT, columns are |eta|, both of the START state. n is\n"
      "the same for every arm, because every arm answers the same\n"
      "transitions: the one-stage baseline is this run's own.\n");

  auto printGrid = [&](int mat, int arm) {
    std::printf("\n%s, %s\n\n", kArmName[arm], matName[mat]);
    std::printf("%-12s", "pT [GeV]");
    for (int ie = 0; ie < kNEta; ++ie) {
      char h[16];
      std::snprintf(h, sizeof(h), "%g-%g", ie == 0 ? 0.0 : kEtaEdges[ie - 1],
                    kEtaEdges[ie]);
      std::printf(" %9s", h);
    }
    std::printf(" %9s %9s\n", "all", "n");
    for (int ip = 0; ip < kNPt; ++ip) {
      char pt[16];
      std::snprintf(pt, sizeof(pt), "%g-%g", kPtEdges[ip], kPtEdges[ip + 1]);
      std::printf("%-12s", pt);
      std::size_t rn = 0, rm = 0;
      for (int ie = 0; ie < kNEta; ++ie) {
        const Cell& c = byCell[mat][arm][ip][ie];
        std::printf(" %9.2f", pct(c.match, c.n));
        rn += c.n;
        rm += c.match;
      }
      std::printf(" %9.2f %9zu\n", pct(rm, rn), rn);
    }
    std::printf("%-12s", "all");
    for (int ie = 0; ie < kNEta; ++ie) {
      std::size_t cn = 0, cm = 0;
      for (int ip = 0; ip < kNPt; ++ip) {
        cn += byCell[mat][arm][ip][ie].n;
        cm += byCell[mat][arm][ip][ie].match;
      }
      std::printf(" %9.2f", pct(cm, cn));
    }
    std::printf(" %9.2f %9zu\n", pct(total[mat][arm].match, total[mat][arm].n),
                total[mat][arm].n);
  };

  for (int arm = 0; arm < kNArm; ++arm) {
    for (int mat = 0; mat < 2; ++mat) {
      printGrid(mat, arm);
    }
  }

  // Every cell as counts rather than as a percentage, so a rate can be taken
  // over any grouping of cells without reading it back off a rounded table.
  // `offered` is every consecutive pair of arrivals in the cell,
  // `sameLayer` those whose successor is on the layer the track stands on and
  // which the walk excludes by construction, `walkFailed` those where the walk
  // named nothing, `n` those compared and `match` those named correctly.
  // `offered = sameLayer + walkFailed + n` holds in every cell.
  std::printf(
      "\nCELL,mat,arm,ptLo,ptHi,etaLo,etaHi,offered,sameLayer,walkFailed,n,"
      "match,fellBack\n");
  for (int mat = 0; mat < 2; ++mat) {
    for (int arm = 0; arm < kNArm; ++arm) {
      for (int ip = 0; ip < kNPt; ++ip) {
        for (int ie = 0; ie < kNEta; ++ie) {
          const Cell& c = byCell[mat][arm][ip][ie];
          std::printf("CELL,%d,%s,%g,%g,%g,%g,%zu,%zu,%zu,%zu,%zu,%zu\n", mat,
                      kArmName[arm], kPtEdges[ip], kPtEdges[ip + 1],
                      ie == 0 ? 0.0 : kEtaEdges[ie - 1], kEtaEdges[ie],
                      offeredCell[mat][ip][ie], sameLayerCell[mat][ip][ie],
                      walkFailedCell[mat][ip][ie], c.n, c.match,
                      fellBackCell[mat][arm][ip][ie]);
        }
      }
    }
  }

  // The same counts on `chi2_gate`'s edges. The sample is built on
  // `kEtaEdges`, three |eta| values inside each band, so these cells hold
  // whatever those discrete values fall into and are unevenly filled by
  // construction; `n` is printed so that is visible rather than assumed.
  std::printf(
      "\nGCELL,mat,arm,ptLo,ptHi,etaLo,etaHi,offered,sameLayer,walkFailed,n,"
      "match,fellBack\n");
  for (int mat = 0; mat < 2; ++mat) {
    for (int arm = 0; arm < kNArm; ++arm) {
      for (int ip = 0; ip < kNGate; ++ip) {
        for (int ie = 0; ie < kNGate; ++ie) {
          const Cell& c = byGate[mat][arm][ip][ie];
          std::printf("GCELL,%d,%s,%g,%g,%g,%g,%zu,%zu,%zu,%zu,%zu,%zu\n", mat,
                      kArmName[arm], kGatePtEdges[ip], kGatePtEdges[ip + 1],
                      kGateEtaEdges[ie], kGateEtaEdges[ie + 1],
                      offeredGate[mat][ip][ie], sameLayerGate[mat][ip][ie],
                      walkFailedGate[mat][ip][ie], c.n, c.match,
                      fellBackGate[mat][arm][ip][ie]);
        }
      }
    }
  }

  std::printf("\nOverall, side by side.\n\n%-16s %10s %10s %14s %14s %14s\n",
              "arm", "off [%]", "on [%]", "misses off", "misses on",
              "fell back off");
  for (int arm = 0; arm < kNArm; ++arm) {
    std::printf("%-16s %10.2f %10.2f %14zu %14zu %14zu\n", kArmName[arm],
                pct(total[0][arm].match, total[0][arm].n),
                pct(total[1][arm].match, total[1][arm].n),
                total[0][arm].n - total[0][arm].match,
                total[1][arm].n - total[1][arm].match, nFellBack[0][arm]);
  }

  std::printf(
      "\nWhere an arm picked something other than the one-stage arm did.\n\n");
  for (int mat = 0; mat < 2; ++mat) {
    for (int arm = 1; arm < kNArm; ++arm) {
      std::printf("  %-14s %-16s %zu of %zu\n", matName[mat], kArmName[arm],
                  armDisagree[mat][arm], total[mat][arm].n);
    }
  }

  std::printf(
      "\nHow many modules the chosen layer offered in the FIRST stage, over\n"
      "all comparisons. An arm that reorders this list can only differ from\n"
      "another where it is above one.\n\n");
  std::printf("%-14s", "candidates");
  for (int mat = 0; mat < 2; ++mat) {
    std::printf(" %14s", matName[mat]);
  }
  std::printf("\n");
  std::set<std::size_t> sizes;
  for (int mat = 0; mat < 2; ++mat) {
    for (const auto& [k, v] : candHist[mat]) {
      sizes.insert(k);
    }
  }
  for (std::size_t k : sizes) {
    std::printf("%-14zu", k);
    for (int mat = 0; mat < 2; ++mat) {
      const auto it = candHist[mat].find(k);
      std::printf(" %14zu", it == candHist[mat].end() ? 0 : it->second);
    }
    std::printf("\n");
  }

  std::printf(
      "\n================================================================\n"
      "3. Where the helix lands\n"
      "================================================================\n\n"
      "`sagitta` is the distance from the landing point to the point the\n"
      "straight ray reaches at the same path length, which is what the second\n"
      "stage moves the query by. `residual` is the signed straight-ray path\n"
      "from the landing point to the layer's approach surface, so it is what\n"
      "the next Newton iteration would remove; a negative one is a landing\n"
      "past the surface. Both over every transition the arm answered.\n\n");
  std::printf("%-14s %-16s %9s %9s %9s %9s %11s %11s %11s %10s\n", "material",
              "arm", "sag p50", "sag p90", "sag p99", "sag max", "res p10",
              "res p50", "res p90", "back path");
  for (int mat = 0; mat < 2; ++mat) {
    for (int k = 0; k < 4; ++k) {
      // Into locals first. `quantile` sorts its argument in place and the order
      // in which a call's arguments are evaluated is unspecified, so reading a
      // maximum off the same vector inside the same `printf` is a coin toss.
      std::vector<double>& sg = sagitta[mat][k];
      std::vector<double>& rs = residual[mat][k];
      const double s50 = quantile(sg, 0.50);
      const double s90 = quantile(sg, 0.90);
      const double s99 = quantile(sg, 0.99);
      const double smax = sg.empty() ? std::nan("") : sg.back();  // now sorted
      const double r10 = quantile(rs, 0.10);
      const double r50 = quantile(rs, 0.50);
      const double r90 = quantile(rs, 0.90);
      std::printf(
          "%-14s %-16s %9.3f %9.3f %9.3f %9.3f %11.4f %11.4f %11.4f %10zu\n",
          matName[mat], kArmName[kArmFirstTwoStage + k], s50, s90, s99, smax,
          r10, r50, r90, nNegativePath[mat][k]);
    }
  }

  std::printf(
      "\nThe same sagitta by pT, material off, for the arm at the local field\n"
      "and no iteration. It is 3.8 mm for a 1 GeV track over\n"
      "a 225 mm leg; the sample is log-uniform in pT and most legs are shorter,\n"
      "so the overall median above is not that number.\n\n");
  std::printf("%-12s %9s %9s %9s %9s %9s\n", "pT [GeV]", "n", "p50", "p90",
              "p99", "max");
  for (int ip = 0; ip < kNPt; ++ip) {
    char pt[16];
    std::snprintf(pt, sizeof(pt), "%g-%g", kPtEdges[ip], kPtEdges[ip + 1]);
    std::vector<double>& v = sagittaByPt[ip];
    const double q50 = quantile(v, 0.50);
    const double q90 = quantile(v, 0.90);
    const double q99 = quantile(v, 0.99);
    const double mx = v.empty() ? std::nan("") : v.back();
    std::printf("%-12s %9zu %9.3f %9.3f %9.3f %9.3f\n", pt, v.size(), q50, q90,
                q99, mx);
  }
  std::printf(
      "\n`back path` is how often the module the second stage named lay at a\n"
      "NEGATIVE path from the landing point, which is the only case in which\n"
      "the %.0f mm floor of the second-stage query changed the answer.\n",
      kStage2NearLimit);

  std::printf(
      "\n================================================================\n"
      "4. The misses, split three ways and asked of the binned lookup\n"
      "================================================================\n\n"
      "`in list` is a wrong choice among the right candidates. The other two\n"
      "are nomination failures and no reordering reaches them.\n\n"
      "`nbr had it` is the binary question: was the true module returned by\n"
      "`SurfaceArray::neighbors` on the layer the walk chose, asked from the\n"
      "point and along the direction that arm's own `compatibleSurfaces` call\n"
      "used. If it was, the module was intersected and the exact edge test\n"
      "rejected it, and a boundary tolerance can recover it. If it was not,\n"
      "the module was never intersected and no tolerance can.\n\n"
      "`own lay` asks the same of the TRUE module's own layer, which is a\n"
      "different surface array whenever the layer was wrong. `ahead` is how\n"
      "often that layer was among the layers the walk had in front of it.\n"
      "`pack` is the median number of surfaces the bin lookup returned.\n\n");
  std::printf(
      "%-14s %-16s %-26s %7s %10s %8s %7s %7s %8s %8s %7s %7s %7s %7s %7s "
      "%7s\n",
      "material", "arm", "category", "n", "nbr had it", "own lay", "ahead",
      "pack", "dist p50", "dist p90", "pT p10", "pT p50", "pT p90", "eta p10",
      "eta p50", "eta p90");
  for (int mat = 0; mat < 2; ++mat) {
    for (int arm = 0; arm < kNArm; ++arm) {
      for (int cat = 0; cat < kNCat; ++cat) {
        MissStats& m = miss[mat][arm][cat];
        if (m.n == 0) {
          continue;
        }
        const double nb50 = quantile(m.nNeighbors, 0.50);
        const double d50 = quantile(m.dist, 0.50);
        const double d90 = quantile(m.dist, 0.90);
        const double pt10 = quantile(m.pt, 0.10);
        const double pt50 = quantile(m.pt, 0.50);
        const double pt90 = quantile(m.pt, 0.90);
        const double e10 = quantile(m.aeta, 0.10);
        const double e50 = quantile(m.aeta, 0.50);
        const double e90 = quantile(m.aeta, 0.90);
        std::printf(
            "%-14s %-16s %-26s %7zu %10zu %8zu %7zu %7.0f %8.1f %8.1f %7.2f "
            "%7.2f %7.2f %7.2f %7.2f %7.2f\n",
            matName[mat], kArmName[arm], kCatName[cat], m.n, m.neighborsHadIt,
            m.trueLayerHadIt, m.trueLayerAhead, nb50, d50, d90, pt10, pt50,
            pt90, e10, e50, e90);
      }
    }
  }

  std::printf(
      "\n================================================================\n"
      "5. The same walk with the per-module edge test relaxed\n"
      "================================================================\n\n"
      "`Layer::compatibleSurfaces` hands `options.boundaryTolerance` straight\n"
      "to `surface.intersect`. This is the one-stage walk re-run with that\n"
      "relaxed and nothing else changed. `compatibleLayers` keeps `None()`, so\n"
      "the layers offered are the same ones; `layer moved` is how often the\n"
      "walk nonetheless came out on a different layer, because `resolveIn`\n"
      "stops at the first layer ahead that offers a module and a relaxed edge\n"
      "test can make a nearer one offer where it offered nothing.\n"
      "Material off, %zu transitions.\n\n"
      "`recovered` and `new wrong` are against this run's own baseline. A\n"
      "tolerance that recovers 400 and creates 600 is not a fix.\n"
      "`rl offered` and `rl matched` are restricted to the %zu baseline\n"
      "misses whose true module was on the right layer and not in the list.\n\n",
      tolN.empty() ? 0 : tolN[0], nRightLayerBase);
  std::printf("%-12s %10s %12s %12s %12s %12s %10s %12s\n", "tolerance",
              "match [%]", "recovered", "new wrong", "rl offered", "rl matched",
              "cands p50", "layer moved");
  {
    std::vector<double> base;
    for (const auto& [k, v] : candHist[0]) {
      for (std::size_t i = 0; i < v; ++i) {
        base.push_back(static_cast<double>(k));
      }
    }
    const double b50 = quantile(base, 0.50);
    std::printf("%-12s %10.2f %12s %12s %12s %12s %10.0f %12s\n", "none (base)",
                pct(total[0][kArmOneStage].match, total[0][kArmOneStage].n),
                "-", "-", "0", "0", b50, "-");
  }
  for (std::size_t t = 0; t < nTol; ++t) {
    const double c50 = quantile(tolCands[t], 0.50);
    std::printf("%-12s %10.2f %12zu %12zu %12zu %12zu %10.0f %12zu\n",
                tolArms[t].name, pct(tolMatch[t], tolN[t]), tolRecovered[t],
                tolNewWrong[t], tolRlOffered[t], tolRlMatched[t], c50,
                tolLayerMoved[t]);
  }

  std::printf(
      "\nThe same tolerances on the SECOND stage of `%s`, which is the query a\n"
      "seam would actually make. Same %zu transitions, same baseline arm, and\n"
      "`rl offered` and `rl matched` are now restricted to that arm's own %zu\n"
      "right-layer misses rather than the one-stage walk's.\n\n",
      kArmName[kArmFirstTwoStage + 2], t2N.empty() ? 0 : t2N[0],
      n2RightLayerBase);
  std::printf("%-12s %10s %12s %12s %12s %12s %10s\n", "tolerance",
              "match [%]", "recovered", "new wrong", "rl offered", "rl matched",
              "cands p50");
  std::printf("%-12s %10.2f %12s %12s %12s %12s %10s\n", "none (base)",
              pct(total[0][kArmFirstTwoStage + 2].match,
                  total[0][kArmFirstTwoStage + 2].n),
              "-", "-", "0", "0", "-");
  for (std::size_t t = 0; t < nTol; ++t) {
    const double c50 = quantile(t2Cands[t], 0.50);
    std::printf("%-12s %10.2f %12zu %12zu %12zu %12zu %10.0f\n",
                tolArms[t].name, pct(t2Match[t], t2N[t]), t2Recovered[t],
                t2NewWrong[t], t2RlOffered[t], t2RlMatched[t], c50);
  }

  std::printf(
      "\nWhat is NOT in these tables. |p| and the charge at each arrival are\n"
      "read off the stepper, so the helix arms run at the local momentum with\n"
      "material on as well as off. The second stage asks the layer the FIRST\n"
      "stage chose and no other, so a wrong layer stays wrong however close\n"
      "the query is made from. Arrivals whose successor lies on the same layer\n"
      "are outside the walk's domain and are excluded, not counted as misses.\n");
  return 0;
}
