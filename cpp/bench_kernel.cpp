// Latency of one learned jump, measured the way a CKF actually consumes it,
// against a Runge-Kutta-Nystrom stepper compiled and timed on the same machine.
//
// Three rules the measurement is built around:
//
//  1. Throughput is not latency. Independent calls thrown at the CPU overlap in
//     the pipeline and read 1.8x fast. A CKF transport chain cannot overlap,
//     because jump k+1 starts from jump k's output, so every loop below feeds
//     the previous result into the next input and the compiler has no choice
//     but to serialise.
//
//  2. Do not borrow a divisor or a baseline. Quoting a kernel measured here
//     against an RKN measured on another machine is not a comparison, so RKN is
//     implemented here and timed in the same loop, and ACTS's own EigenStepper
//     runs in the same executable.
//
//  3. Measure what the filter pays for. LearnedStepper advances jacTransport
//     and derivative on every jump, so both sides are timed twice, once without
//     a covariance and once with, and the with-covariance pair is the one a CKF
//     actually pays.
//
// The destination is the module plane, not a cylinder r = r1 at the true next
// hit's radius: that is a surface the detector does not have and it is derived
// from the answer. So the RKN arm integrates to n . (X - c) = 0 and the ACTS
// arm targets a PlaneSurface built from the module's own frame, and neither is
// timed against a surface the other cannot see. A plane's intersection
// predicate is also smooth, so the solver that reproduces the Python bit for
// bit is the eight-step Newton that would ship, and there is no
// faithful-but-slow variant to subtract.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "LearnedJacobian.hpp"
#include "LearnedTransport.hpp"

// The ACTS arms. Guarded because the kernel and the reference RKN stand on
// their own and must keep building on a machine with no ACTS. Without it this
// benchmark can only compare one local implementation against another, which
// is not a baseline.
#ifdef WITH_ACTS
#include "Acts/Definitions/TrackParametrization.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/MagneticField/BFieldMapUtils.hpp"
#include "Acts/MagneticField/MagneticFieldContext.hpp"
#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Propagator/VoidNavigator.hpp"
#include "Acts/Propagator/detail/CovarianceEngine.hpp"
#include "Acts/Surfaces/CurvilinearSurface.hpp"
#include "Acts/Surfaces/PlaneSurface.hpp"
#include "Acts/Surfaces/RectangleBounds.hpp"
#include <memory>
#include <optional>
#endif

namespace {

constexpr double kOrtFixedNs = 1788.0;   // measured, ONNX Runtime
constexpr double kOrtPerItemNs = 98.0;
constexpr double kPionMass = 0.13957039;  // GeV, the teacher's hypothesis

using collider_ml::Jump;
using collider_ml::Vec3;

/// Trilinear lookup on the real ODD grid, 201x201x301.
///
/// This is here because a benchmark against RKN in a CONSTANT field measures
/// the wrong thing. The reason a learned jump can beat an integrator is not
/// that its arithmetic is cheaper -- it is that it reads the field ONCE per
/// jump while RKN4 reads it at three distinct points per step. With a constant
/// field that difference is invisible and RKN looks free; with the real 48 MB
/// map it is most of the cost, and it is a cache-miss cost that no amount of
/// SIMD helps.
class FieldMap {
 public:
  bool load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
      return false;
    }
    f.read(reinterpret_cast<char*>(&n_[0]), 12);
    f.read(reinterpret_cast<char*>(&o_[0]), 24);
    f.read(reinterpret_cast<char*>(&h_[0]), 24);
    const std::size_t total = static_cast<std::size_t>(n_[0]) * n_[1] * n_[2];
    bz_.resize(total);
    f.read(reinterpret_cast<char*>(bz_.data()),
           static_cast<std::streamsize>(total * sizeof(float)));
    return static_cast<bool>(f);
  }

  bool loaded() const { return !bz_.empty(); }

  /// `bz`, counted. Only the untimed replay that produces the kernel arms'
  /// field-read count calls this; the timed loops call `bz` so that no arm
  /// pays for a counter another arm does not.
  double bzCounted(const Vec3& p) const {
    ++counted_;
    return bz(p);
  }
  std::size_t counts() const { return counted_; }
  void resetCounts() const { counted_ = 0; }

  // The ACTS arm builds its own InterpolatedBFieldMap out of THESE numbers
  // rather than re-reading the csv, so that neither side can be reading a
  // different field from the other. That is the whole point of item 1.
  const int* n() const { return n_; }
  const double* origin() const { return o_; }
  const double* spacing() const { return h_; }
  const std::vector<float>& bz() const { return bz_; }

  double bz(const Vec3& p) const {
    if (bz_.empty()) {
      return 2.0;
    }
    double t[3];
    int i0[3];
    for (int k = 0; k < 3; ++k) {
      const double f = (p[k] - o_[k]) / h_[k];
      int i = static_cast<int>(std::floor(f));
      i = std::clamp(i, 0, n_[k] - 2);
      i0[k] = i;
      t[k] = std::clamp(f - i, 0.0, 1.0);
    }
    const std::size_t sy = static_cast<std::size_t>(n_[2]);
    const std::size_t sx = sy * static_cast<std::size_t>(n_[1]);
    double out = 0.0;
    for (int dx = 0; dx < 2; ++dx) {
      for (int dy = 0; dy < 2; ++dy) {
        for (int dz = 0; dz < 2; ++dz) {
          const double w = (dx ? t[0] : 1 - t[0]) * (dy ? t[1] : 1 - t[1]) *
                           (dz ? t[2] : 1 - t[2]);
          out += w * bz_[(i0[0] + dx) * sx + (i0[1] + dy) * sy + i0[2] + dz];
        }
      }
    }
    return out;
  }

 private:
  int n_[3]{};
  double o_[3]{};
  double h_[3]{};
  std::vector<float> bz_;
  mutable std::size_t counted_ = 0;
};

FieldMap g_field;  // one map, shared by both sides of the comparison

/// The 22-column layout `export_kernel.py` writes: position, momentum, charge,
/// then the destination module's centre, normal and two in-plane axes, then the
/// field value at the source and the two cell widths. `test_kernel.cpp` reads
/// the same file and the same columns, so a benchmark cannot end up timing a
/// jump the correctness check never saw.
std::vector<Jump> load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::int32_t n = 0;
  std::int32_t nin = 0;
  std::int32_t nout = 0;
  f.read(reinterpret_cast<char*>(&n), 4);
  f.read(reinterpret_cast<char*>(&nin), 4);
  f.read(reinterpret_cast<char*>(&nout), 4);
  if (nin != 22) {
    std::fprintf(stderr,
                 "%s has %d input columns, expected 22. This is the pre-plane\n"
                 "layout; rerun python -m prop.export_kernel.\n",
                 path.c_str(), nin);
    return {};
  }
  std::vector<double> in(static_cast<std::size_t>(n) * nin);
  f.read(reinterpret_cast<char*>(in.data()),
         static_cast<std::streamsize>(in.size() * sizeof(double)));

  std::vector<Jump> out;
  out.reserve(n);
  for (std::int32_t i = 0; i < n; ++i) {
    const double* x = &in[static_cast<std::size_t>(i) * nin];
    Jump j;
    j.pos = {x[0], x[1], x[2]};
    j.mom = {x[3], x[4], x[5]};
    j.q = x[6];
    j.c = {x[7], x[8], x[9]};
    j.n = {x[10], x[11], x[12]};
    j.e0 = {x[13], x[14], x[15]};
    j.e1 = {x[16], x[17], x[18]};
    j.bzMap = x[19];
    j.sig0 = x[20];
    j.sig1 = x[21];
    out.push_back(j);
  }
  return out;
}

// ------------------------------------------------------------ a real RKN4
//
// dr/ds = T,  dT/ds = (q/p) T x B, integrated with the classical
// Runge-Kutta-Nystrom scheme ACTS's EigenStepper uses, with the same
// embedded error estimate driving the step size.

struct RknResult {
  Vec3 pos;
  Vec3 dir;
  int steps = 0;
  bool ok = false;
};

inline Vec3 field(const Vec3& p) { return {0.0, 0.0, g_field.bz(p)}; }

/// One RKN4 step of length h, returning the embedded error estimate.
///
/// dT/ds = (q/p) * c * (T x B). With p in GeV, B in tesla and s in mm the
/// constant is 0.299792458e-3, so that |dT/ds| = 1/R for T perpendicular to B.
inline double rknStep(Vec3* pos, Vec3* dir, double qop, double h) {
  const double lambda = qop * 0.299792458e-3;  // 1/(mm*T)
  // NOTE the explicit Vec3 return type. With `auto`, the deduced type is an
  // Eigen expression template holding a reference to the temporary returned by
  // field(p), which is destroyed before the expression is evaluated -- every
  // step then comes out NaN. Eigen's documented `auto` hazard, and it costs an
  // afternoon every time it is rediscovered.
  auto acc = [&](const Vec3& p, const Vec3& t) -> Vec3 {
    return lambda * t.cross(field(p));
  };
  // THREE distinct field points per step, not four. k2 and k3 sit at the same
  // midpoint p2 and differ only in the direction handed to the cross product,
  // so the map is read at pos, at p2 and at p4. This lambda reads p2 twice,
  // which is one redundant lookup per step: it makes the RKN floor slightly
  // slower than it needs to be and changes nothing else. Quote 3 reads per
  // step; the printouts below say "distinct field points" for that reason.
  const Vec3 k1 = acc(*pos, *dir);
  const Vec3 p2 = *pos + 0.5 * h * *dir + 0.125 * h * h * k1;
  const Vec3 k2 = acc(p2, *dir + 0.5 * h * k1);
  const Vec3 k3 = acc(p2, *dir + 0.5 * h * k2);
  const Vec3 p4 = *pos + h * *dir + 0.5 * h * h * k3;
  const Vec3 k4 = acc(p4, *dir + h * k3);

  // ACTS's error estimate: h^2 * |k1 - k2 - k3 + k4|
  const double err = h * h * (k1 - k2 - k3 + k4).template lpNorm<1>();

  *pos += h * *dir + (h * h / 6.0) * (k1 + k2 + k3);
  *dir += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
  dir->normalize();
  return err;
}

/// Integrate to the destination module plane with an adaptive step, the way the
/// production stepper does: propose, estimate, accept or shrink.
///
/// The step constraint is the distance ALONG THE PATH to the plane,
/// n . (c - r) / (n . T), not the perpendicular distance. That is what the
/// navigator hands the stepper, and it is what makes the last step land on the
/// surface rather than approach it geometrically.
RknResult rknToSurface(const Jump& j, double tol = 1e-4) {
  RknResult r;
  r.pos = j.pos;
  r.dir = j.mom.normalized();
  const double p = j.mom.norm();
  const double qop = j.q / p;

  double h = 100.0;  // mm, the usual initial guess

  for (int i = 0; i < 200; ++i) {
    const double along = j.n.dot(r.dir);
    if (std::abs(along) < 1e-9) {  // grazing the module edge on
      break;
    }
    const double toSurface = j.n.dot(j.c - r.pos) / along;
    // 1e-4 mm is Acts::s_onSurfaceTolerance, the same bar the real navigator
    // declares arrival at. A tighter one would charge this arm for iterations
    // ACTS never runs.
    if (std::abs(toSurface) < 1e-4) {
      r.ok = true;
      break;
    }
    // Overshoot is normal and is not a failure. The constraint is computed
    // from the TANGENT, so a curved step of exactly that length lands slightly
    // past the plane; a real stepper then steps back. Treating a negative
    // constraint as "missed" instead cost this arm 8.1% of its jumps and, worse,
    // made it look fast, because the ones it abandoned were abandoned early.
    h = std::copysign(std::min(std::abs(h), std::abs(toSurface)), toSurface);

    // propose, estimate, shrink until accepted -- rejections are not steps
    Vec3 tp;
    Vec3 td;
    double err = 0.0;
    for (int t = 0; t < 12; ++t) {
      tp = r.pos;
      td = r.dir;
      err = rknStep(&tp, &td, qop, h);
      if (err <= 4.0 * tol) {
        break;
      }
      h *= 0.5;
    }
    r.pos = tp;
    r.dir = td;
    ++r.steps;
    if (err < 0.25 * tol) {
      h *= 2.0;
    }
  }
  return r;
}

/// The two step counters, plus the attempted and rejected split that says why
/// the step count moves. Declared outside the
/// ACTS guard so the CSV writer compiles on a machine with no ACTS, where it
/// refuses to write anything anyway.
struct BenchCounters {
  double steps = 0.0;         ///< PropagatorResult::steps, mean per transport
  double attempted = 0.0;     ///< statistics.stepping.nAttemptedSteps
  double rejected = 0.0;      ///< statistics.stepping.nRejectedSteps
  double fieldLookups = 0.0;  ///< getField() calls, mean per transport
  std::size_t counted = 0;    ///< transports that reached the surface
  std::size_t failed = 0;     ///< ... that did not, and are excluded above
  double reached = 0.0;       ///< counted / attempted, the condition on both
};

// ------------------------------------------- ACTS's own stepper, same binary
//
// Everything above this line is local code, which is exactly the problem this
// arm exists to fix: a kernel timed here against an EigenStepper timed
// somewhere else is not a measurement, and the reference RKN above is a floor.
// This arm runs Acts::Propagator<Acts::EigenStepper<>, ...> over the same
// jumps, in the SAME executable, through the SAME timeIt() dependency chain.
//
// Three things are held equal on purpose:
//
//  * THE FIELD. ACTS's InterpolatedBFieldMap is built from the very floats
//    this file already loaded out of field.bin, with Bx = By = 0, because the
//    reference RKN and the kernel both use Bz alone. Handing ACTS the full
//    three-component map would let it integrate a different field and the
//    difference would be charged to the algorithm.
//
//  * THE SURFACE. The destination is a PlaneSurface whose transform is the
//    module's own (e0, e1, n) frame about its centre -- the same three vectors
//    the kernel's Newton solve uses. Both sides therefore aim at the same
//    plane, and neither is credited for solving an easier geometry.
//
//  * THE ALLOCATIONS. Destination surfaces are built ONCE, before timing.
//    In production they come out of the tracking geometry already built, so
//    paying makeShared() per jump inside the loop would be an artefact. What
//    cannot be hoisted is the start parameters, which genuinely are rebuilt
//    per transport in a CKF -- so that cost is measured on its own and
//    subtracted.
//
// Covariance is run both ways, because LearnedStepper transports one and the
// comparable pair is the one with it on.
#ifdef WITH_ACTS
/// The same field, counting every lookup that goes through it.
///
/// Field reads per transport are counted rather than derived from "three per
/// step". RKN4 evaluates the acceleration
/// at four points but three distinct positions, and whether the implementation
/// asks the provider three times or four is a property of this ACTS build, not
/// of the method, so multiplying a step count would be asserting the answer.
///
/// This never appears in a timed loop. The counting pass runs on its own
/// propagator so that the timed arm keeps calling the plain field with no
/// virtual indirection or counter that the kernel arms do not also pay.
class CountingField : public Acts::MagneticFieldProvider {
 public:
  explicit CountingField(std::shared_ptr<const Acts::MagneticFieldProvider> in)
      : m_inner(std::move(in)) {}

  Cache makeCache(const Acts::MagneticFieldContext& mctx) const override {
    return m_inner->makeCache(mctx);
  }

  Acts::Result<Acts::Vector3> getField(const Acts::Vector3& position,
                                       Cache& cache) const override {
    ++m_count;
    return m_inner->getField(position, cache);
  }

  std::size_t count() const { return m_count; }
  void reset() { m_count = 0; }

 private:
  std::shared_ptr<const Acts::MagneticFieldProvider> m_inner;
  mutable std::size_t m_count = 0;
};

class ActsArm {
 public:
  using Stepper = Acts::EigenStepper<>;
  using Prop = Acts::Propagator<Stepper, Acts::VoidNavigator>;

  /// Returns false if there is no field map, in which case this arm is
  /// skipped rather than quietly run against a constant 2 T.
  bool init(const FieldMap& fm, const std::vector<Jump>& jumps) {
    if (!fm.loaded()) {
      return false;
    }
    const int nx = fm.n()[0];
    const int ny = fm.n()[1];
    const int nz = fm.n()[2];

    std::vector<double> xs(nx);
    std::vector<double> ys(ny);
    std::vector<double> zs(nz);
    for (int i = 0; i < nx; ++i) {
      xs[i] = fm.origin()[0] + i * fm.spacing()[0];
    }
    for (int i = 0; i < ny; ++i) {
      ys[i] = fm.origin()[1] + i * fm.spacing()[1];
    }
    for (int i = 0; i < nz; ++i) {
      zs[i] = fm.origin()[2] + i * fm.spacing()[2];
    }

    // same order the file is written in: x outer, y middle, z inner
    std::vector<Acts::Vector3> b;
    b.reserve(fm.bz().size());
    for (const float v : fm.bz()) {
      b.emplace_back(0.0, 0.0, static_cast<double>(v));
    }
    auto localToGlobal = [](std::array<std::size_t, 3> bins,
                            std::array<std::size_t, 3> nBins) {
      return (bins[0] * nBins[1] + bins[1]) * nBins[2] + bins[2];
    };
    using Map = decltype(Acts::fieldMapXYZ(localToGlobal, xs, ys, zs, b,
                                           Acts::UnitConstants::mm,
                                           Acts::UnitConstants::T, false));
    m_field = std::make_shared<Map>(
        Acts::fieldMapXYZ(localToGlobal, xs, ys, zs, b, Acts::UnitConstants::mm,
                          Acts::UnitConstants::T, false));

    m_prop = std::make_unique<Prop>(Stepper(m_field), Acts::VoidNavigator{});
    // The same stepper without a propagator around it, for the per-step arm.
    m_stepper = std::make_unique<Stepper>(m_field);

    // A second propagator over the same map, wrapped in a counter. Used only
    // by counters() below, never by anything timed.
    m_counting = std::make_shared<CountingField>(m_field);
    m_countProp =
        std::make_unique<Prop>(Stepper(m_counting), Acts::VoidNavigator{});

    // one destination surface per jump, built up front. The bounds are
    // deliberately much larger than a real ODD module: the aborter only has to
    // accept the crossing the teacher recorded, and a rectangle sized to the
    // true sensor would reject the ones ACTS lands a few hundred microns
    // outside, turning a geometry question into a timing artefact.
    auto rect = std::make_shared<const Acts::RectangleBounds>(kHalfX, kHalfY);
    m_targets.reserve(jumps.size());
    for (const Jump& j : jumps) {
      Acts::Transform3 t = Acts::Transform3::Identity();
      Acts::RotationMatrix3 R;
      R.col(0) = Acts::Vector3(j.e0.x(), j.e0.y(), j.e0.z());
      R.col(1) = Acts::Vector3(j.e1.x(), j.e1.y(), j.e1.z());
      R.col(2) = Acts::Vector3(j.n.x(), j.n.y(), j.n.z());
      // (e0, e1, n) comes out of DD4hep and is orthonormal but not always
      // right-handed. A left-handed rotation is not a valid Transform3 and
      // ACTS's local-to-global would silently mirror the plane, so flip the
      // in-plane axis rather than the normal: the plane is the same either way
      // and the normal is what the kernel's solve uses.
      if (R.determinant() < 0.0) {
        R.col(1) = -R.col(1);
      }
      t.linear() = R;
      t.translation() = Acts::Vector3(j.c.x(), j.c.y(), j.c.z());
      m_targets.push_back(Acts::Surface::makeShared<Acts::PlaneSurface>(t, rect));
    }
    return true;
  }

  /// The start parameters for jump `j`, and nothing else. Subtracted from the
  /// full number below so the quoted ACTS cost is transport, not bookkeeping.
  Acts::BoundTrackParameters start(const Jump& j, bool withCov) const {
    const double p = j.mom.norm();
    std::optional<Acts::BoundMatrix> cov;
    if (withCov) {
      cov = m_startCov;
    }
    return Acts::BoundTrackParameters::createCurvilinear(
        Acts::Vector4(j.pos.x(), j.pos.y(), j.pos.z(), 0.0), j.mom / p,
        j.q / p, std::move(cov), Acts::ParticleHypothesis::pion());
  }

  /// One transport to the destination surface. Returns the arc length, or a
  /// negative number if ACTS did not reach the surface.
  double propagate(const Jump& j, std::size_t idx, bool withCov) const {
    Prop::Options<> opt(m_gctx, m_mctx);
    opt.stepping.maxStepSize = 100.0;
    const auto res = m_prop->propagate(start(j, withCov), *m_targets[idx], opt);
    if (!res.ok()) {
      return -1.0;
    }
    return res->pathLength;
  }

  std::size_t size() const { return m_targets.size(); }

  using Counters = BenchCounters;

  /// One untimed pass with covariance, reading the counters off the result.
  ///
  /// `steps` is `Acts::PropagatorResult::steps`
  /// (Propagator/PropagatorResult.hpp:36) and the attempted and rejected
  /// counts are `Acts::StepperStatistics` (StepperStatistics.hpp:22-27). None
  /// of this instruments the stepper; the propagator already carries it.
  /// Steps and field reads share one denominator, and it is the transports
  /// that reached the surface.
  ///
  /// An earlier version divided the reads by every transport attempted while
  /// dividing the steps by the ones that reached, and the two are not the same
  /// population. A propagation that never arrives runs to the step limit, so it
  /// contributes hundreds of field reads and no steps, and at the 0.7% failure
  /// rate of the soft bins that alone moved the read mean by more than the
  /// physics did. The counter is reset per propagation here and its reads are
  /// kept only when the propagation is kept, so the ratio of the two columns is
  /// a real reads-per-step.
  Counters counters(const std::vector<Jump>& jumps) {
    Counters c;
    for (std::size_t i = 0; i < jumps.size(); ++i) {
      Prop::Options<> opt(m_gctx, m_mctx);
      opt.stepping.maxStepSize = 100.0;
      m_counting->reset();
      const auto res =
          m_countProp->propagate(start(jumps[i], true), *m_targets[i], opt);
      const double reads = static_cast<double>(m_counting->count());
      if (!res.ok()) {
        ++c.failed;
        continue;
      }
      ++c.counted;
      c.steps += static_cast<double>(res->steps);
      c.attempted +=
          static_cast<double>(res->statistics.stepping.nAttemptedSteps);
      c.rejected +=
          static_cast<double>(res->statistics.stepping.nRejectedSteps);
      c.fieldLookups += reads;
    }
    const double n = std::max<std::size_t>(c.counted, 1);
    c.steps /= n;
    c.attempted /= n;
    c.rejected /= n;
    c.fieldLookups /= n;
    c.reached = static_cast<double>(c.counted) /
                std::max<std::size_t>(jumps.size(), 1);
    return c;
  }

  /// One `EigenStepper::step()` call. HANDOFF item 2c.
  ///
  /// The unit the census counts. `Propagator.ipp`'s stepping loop calls
  /// `step()` once per iteration and increments `state.steps` once per
  /// iteration, and `EigenStepper::step` runs its own error control inside a
  /// `while (true)` (EigenStepper.ipp:277-372) incrementing `nAttemptedSteps`
  /// per trial and `nSuccessfulSteps` once at the end. So one call is one
  /// accepted step, `PropagatorResult::steps` is the number of calls, and the
  /// census's step column and this arm are the same quantity.
  ///
  /// `h` is the step the navigator would ask for, which is the distance to the
  /// destination. Passing the whole jump lets the error control reject and
  /// retry exactly as it does in production rather than being handed a step it
  /// is certain to accept.
  ///
  /// Covariance is on, because the CKF's is. `n == 0` is the same body with
  /// the steps removed, and the difference is what gets quoted: a real
  /// propagation builds the state once and steps many times, so the state
  /// construction belongs to neither arm.
  ///
  /// `n == 2` exists because the first step off a fresh state pays a cold
  /// magnetic-field cell. `EigenStepper::State` carries a
  /// `MagneticFieldProvider::Cache` that interpolates inside one cell until it
  /// leaves it, so a step taken after another step is cheaper than the first
  /// one, and every step inside a CKF is of the second kind: by the time the
  /// navigator hands over a module the stepper has already been flying. The
  /// difference between `n == 2` and `n == 1` is that warm step, and it is the
  /// one to multiply by the census.
  double nSteps(const Jump& j, double h, int n) const {
    Stepper::Options sopt(m_gctx, m_mctx);
    sopt.maxStepSize = 100.0;
    auto st = m_stepper->makeState(sopt);
    m_stepper->initialize(st, start(j, true));
    double acc = st.pars[0];
    for (int k = 0; k < n; ++k) {
      m_stepper->updateStepSize(st, h, Acts::ConstrainedStep::Type::Navigator);
      const auto r = m_stepper->step(st, Acts::Direction::Forward(), nullptr);
      if (!r.ok()) {
        break;
      }
      acc += *r;
    }
    return acc;
  }

  /// What one such call costs in trials, untimed. Same three counters as
  /// `counters()` above and the same reason for reporting them.
  Counters stepCounters(const std::vector<Jump>& jumps,
                        const std::vector<double>& h) const {
    Counters c;
    for (std::size_t i = 0; i < jumps.size(); ++i) {
      Stepper::Options sopt(m_gctx, m_mctx);
      sopt.maxStepSize = 100.0;
      auto st = m_stepper->makeState(sopt);
      m_stepper->initialize(st, start(jumps[i], true));
      m_stepper->updateStepSize(st, h[i], Acts::ConstrainedStep::Type::Navigator);
      const auto r = m_stepper->step(st, Acts::Direction::Forward(), nullptr);
      if (!r.ok()) {
        ++c.failed;
        continue;
      }
      ++c.counted;
      c.steps += 1.0;
      c.attempted += static_cast<double>(st.statistics.nAttemptedSteps);
      c.rejected += static_cast<double>(st.statistics.nRejectedSteps);
    }
    const double n = std::max<std::size_t>(c.counted, 1);
    c.steps /= n;
    c.attempted /= n;
    c.rejected /= n;
    c.reached = static_cast<double>(c.counted) /
                std::max<std::size_t>(jumps.size(), 1);
    return c;
  }

  // For the matched-work arm below: the same destination surface, geometry
  // context and start covariance this arm's own propagations use, so the two
  // covariance arms are charged against identical inputs.
  const Acts::Surface& target(std::size_t i) const { return *m_targets[i]; }
  const Acts::GeometryContext& gctx() const { return m_gctx; }
  const Acts::BoundMatrix& startCov() const { return m_startCov; }

 private:
  // Half-extents of the destination rectangle. See the note in init().
  static constexpr double kHalfX = 500.0;  // mm
  static constexpr double kHalfY = 500.0;  // mm

  /// A plausible seed covariance, in ACTS's bound order
  /// (loc0, loc1, phi, theta, q/p, time). The values do not change the timing
  /// -- the cost of transporting a covariance is the matrix products, not the
  /// entries -- but a singular or zero matrix would let the compiler or the
  /// library take a shortcut that production never gets.
  static Acts::BoundMatrix seedCov() {
    Acts::BoundMatrix c = Acts::BoundMatrix::Zero();
    c(0, 0) = 0.05 * 0.05;      // mm
    c(1, 1) = 0.30 * 0.30;      // mm
    c(2, 2) = 1e-3 * 1e-3;      // rad
    c(3, 3) = 1e-3 * 1e-3;      // rad
    c(4, 4) = 1e-4 * 1e-4;      // 1/GeV
    c(5, 5) = 1.0;              // mm, c = 1
    return c;
  }

  Acts::GeometryContext m_gctx{};
  Acts::MagneticFieldContext m_mctx{};
  Acts::BoundMatrix m_startCov = seedCov();
  std::shared_ptr<const Acts::MagneticFieldProvider> m_field;
  std::unique_ptr<Prop> m_prop;
  std::unique_ptr<Stepper> m_stepper;
  // The counting pair. Separate from the timed propagator on purpose.
  std::shared_ptr<CountingField> m_counting;
  std::unique_ptr<Prop> m_countProp;
  std::vector<std::shared_ptr<Acts::Surface>> m_targets;
};
#endif  // WITH_ACTS

template <typename F>
double timeIt(const std::vector<Jump>& jumps, int reps, F&& body) {
  double sink = 0.0;
  for (const Jump& j : jumps) {
    body(j, &sink);
  }
  double best = 1e300;
  for (int rep = 0; rep < 5; ++rep) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
      for (const Jump& jj : jumps) {
        Jump j = jj;
        j.pos.x() += sink * 1e-18;  // a true dependency, too small to drift
        body(j, &sink);
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    best = std::min(best,
                    std::chrono::duration<double, std::nano>(t1 - t0).count() /
                        (reps * static_cast<double>(jumps.size())));
  }
  if (sink == 12345.6789) {
    std::printf(" ");
  }
  return best;
}

/// timeIt, but the body also gets the jump's index -- the ACTS arm needs it to
/// reach its pre-built destination surface. Identical chaining and identical
/// best-of-5, so the numbers stay comparable with the ones above.
template <typename F>
double timeItIdx(const std::vector<Jump>& jumps, int reps, F&& body) {
  double sink = 0.0;
  for (std::size_t i = 0; i < jumps.size(); ++i) {
    body(jumps[i], i, &sink);
  }
  double best = 1e300;
  for (int rep = 0; rep < 5; ++rep) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
      for (std::size_t i = 0; i < jumps.size(); ++i) {
        Jump j = jumps[i];
        j.pos.x() += sink * 1e-18;
        body(j, i, &sink);
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    best = std::min(best,
                    std::chrono::duration<double, std::nano>(t1 - t0).count() /
                        (reps * static_cast<double>(jumps.size())));
  }
  if (sink == 12345.6789) {
    std::printf(" ");
  }
  return best;
}

double median(std::vector<double> v) {
  if (v.empty()) {
    return -1.0;
  }
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  // Positional: the jump file then the field. The options after them give one
  // run per pT bin instead of one run over a mixed set, repeated so the number
  // carries a spread, and a CSV of the result.
  //
  //   bench_kernel muon_jumps.bin field.bin --pt 1 2 --repeats 5 \
  //       --csv speed_by_pt.csv
  //
  // --csv appends, so the five bins accumulate into one file.
  std::string path = "reference.bin";
  std::string fieldPath = "field.bin";
  std::string csvPath;
  double ptLo = -1.0;
  double ptHi = -1.0;
  int repeats = 1;
  {
    int pos = 0;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--pt" && i + 2 < argc) {
        ptLo = std::atof(argv[i + 1]);
        ptHi = std::atof(argv[i + 2]);
        i += 2;
      } else if (a == "--csv" && i + 1 < argc) {
        csvPath = argv[++i];
      } else if (a == "--repeats" && i + 1 < argc) {
        repeats = std::max(1, std::atoi(argv[++i]));
      } else if (a.rfind("--", 0) == 0) {
        std::fprintf(stderr, "unknown option %s\n", a.c_str());
        return 2;
      } else if (pos == 0) {
        path = a;
        ++pos;
      } else if (pos == 1) {
        fieldPath = a;
        ++pos;
      }
    }
  }

  std::vector<Jump> jumps = load(path);
  if (jumps.empty()) {
    std::fprintf(stderr, "no usable jumps in %s\n", path.c_str());
    return 2;
  }

  // Bin by the SOURCE state's pT. The transport's momentum is a property of
  // the track and the helix conserves it, so source and destination agree;
  // the source is the one both arms are handed.
  const std::size_t nAll = jumps.size();
  if (ptLo >= 0.0) {
    std::vector<Jump> keep;
    keep.reserve(jumps.size());
    for (const Jump& j : jumps) {
      const double pt = std::hypot(j.mom.x(), j.mom.y());
      if (pt >= ptLo && pt < ptHi) {
        keep.push_back(j);
      }
    }
    jumps.swap(keep);
    if (jumps.empty()) {
      std::fprintf(stderr, "no jumps with %g <= source pT < %g in %s\n", ptLo,
                   ptHi, path.c_str());
      return 2;
    }
  }
  const int kReps = 200;
  const bool haveField = g_field.load(fieldPath);

  // --- how much work does each side actually do, and do they land on the
  //     same plane? Two separate questions that a single "worst gap" number
  //     conflates, so they are asked separately.
  //
  //  offPlane   how far off the destination plane each arm finished. This is
  //             a correctness check on the arms and must be at the surface
  //             tolerance for both.
  //  gap        how far apart the two landing points are IN the plane. This is
  //             not an error in either arm: the kernel's core is a helix in a
  //             fixed 2 T and the RKN integrates the real map, so this is the
  //             size of the constant-field approximation g_theta is trained to
  //             correct, measured on the same jumps.
  double meanRknSteps = 0.0;
  int nSolved = 0;
  int nRknReached = 0;
  double offPlaneHelix = 0.0;
  double offPlaneRkn = 0.0;
  double meanPath = 0.0;
  std::vector<double> gap;
  gap.reserve(jumps.size());
  for (const Jump& j : jumps) {
    bool ok = false;
    const double s = collider_ml::solveArcLength(j, &ok);
    nSolved += ok ? 1 : 0;
    if (ok) {
      // 3D arc length, which is what ACTS reports as pathLength. The two are
      // compared below: if the arms disagree on how far the track went, the
      // ratio is between two different problems.
      meanPath += s * j.mom.norm() / std::hypot(j.mom.x(), j.mom.y());
    }
    const RknResult r = rknToSurface(j);
    meanRknSteps += r.steps;
    nRknReached += r.ok ? 1 : 0;
    if (ok && r.ok) {
      const collider_ml::Prediction p = collider_ml::transport(j);
      offPlaneHelix =
          std::max(offPlaneHelix, std::abs(j.n.dot(p.helixPos - j.c)));
      offPlaneRkn = std::max(offPlaneRkn, std::abs(j.n.dot(r.pos - j.c)));
      gap.push_back((p.helixPos - r.pos).norm());
    }
  }
  meanRknSteps /= static_cast<double>(jumps.size());
  meanPath /= std::max(nSolved, 1);
  std::sort(gap.begin(), gap.end());
  auto q = [&](double f) {
    return gap.empty() ? 0.0
                       : gap[static_cast<std::size_t>(f * (gap.size() - 1))];
  };

  std::printf("%zu jumps, dependency-chained (jump k+1 waits on jump k)\n",
              jumps.size());
  std::printf("target: the destination MODULE PLANE, n . (X - c) = 0, for both "
              "arms\n");
  std::printf("field: %s\n",
              haveField ? "the real ODD map, 201x201x301, trilinear"
                        : "*** field.bin missing -- constant 2 T, RKN is "
                          "flattered ***");
  // Which SIMD this was built for changes the kernel's ns/jump by a lot, so
  // the number carries it rather than relying on anyone remembering. With the
  // ACTS arm in, this is deliberately the generic baseline that libActsCore
  // was built with -- see build.sh.
  std::printf("build: %s\n\n",
#if defined(__AVX512F__)
              "AVX-512"
#elif defined(__AVX2__)
              "AVX2"
#elif defined(__AVX__)
              "AVX"
#else
              "x86-64 baseline (SSE2), matching libActsCore"
#endif
  );

  const double tRkn = timeIt(jumps, kReps, [](const Jump& j, double* sink) {
    const RknResult r = rknToSurface(j);
    *sink += r.pos.x() + r.dir.z();
  });
  const double tSolve = timeIt(jumps, kReps, [](const Jump& j, double* sink) {
    bool ok = false;
    *sink += collider_ml::solveArcLength(j, &ok);
  });
  // the kernel pays for its own single field read, so the two sides are
  // charged for the map on the same terms
  const double tFull = timeIt(jumps, kReps, [](const Jump& jj, double* sink) {
    Jump j = jj;
    j.bzMap = g_field.bz(j.pos);
    const collider_ml::Prediction p = collider_ml::transport(j);
    *sink += p.pos.x() + p.mom.z() + p.m;
  });
  // and the same thing again with the 8x8 transport Jacobian and the tangent
  // that LearnedStepper::writeBack hands to ACTS's covariance engine. This is
  // the number to compare against ACTS-with-covariance, not tFull.
  const double tFullCov = timeIt(jumps, kReps, [](const Jump& jj, double* sink) {
    Jump j = jj;
    j.bzMap = g_field.bz(j.pos);
    const collider_ml::Prediction p = collider_ml::transport(j);
    const collider_ml::JumpJacobian jj8 =
        collider_ml::jumpJacobian(j, p.mom, p.s, kPionMass);
    *sink += p.pos.x() + p.mom.z() + jj8.D(4, 4) + jj8.t[3];
  });

  // The helix bypass: the same transport with the forward pass skipped, which
  // is what `--stepper helix` runs. That configuration reproduces the stock
  // CKF's four track-finding numbers, so this is the arm whose time can be
  // quoted against RKN without also quoting a correction that costs
  // efficiency.
  const double tHelix = timeIt(jumps, kReps, [](const Jump& jj, double* sink) {
    Jump j = jj;
    j.bzMap = g_field.bz(j.pos);
    const collider_ml::Prediction p = collider_ml::transport(j, false);
    *sink += p.pos.x() + p.mom.z() + p.m;
  });
  const double tHelixCov = timeIt(jumps, kReps, [](const Jump& jj, double* sink) {
    Jump j = jj;
    j.bzMap = g_field.bz(j.pos);
    const collider_ml::Prediction p = collider_ml::transport(j, false);
    const collider_ml::JumpJacobian jj8 =
        collider_ml::jumpJacobian(j, p.mom, p.s, kPionMass);
    *sink += p.pos.x() + p.mom.z() + jj8.D(4, 4) + jj8.t[3];
  });

  // --- HANDOFF item 2c. The per-STEP arms.
  //
  // Everything above prices one whole surface-to-surface transport, ending in
  // `transportCovarianceToBound`. Inside a CKF that is the wrong unit on the
  // ACTS side and the right one on this one, and the census is what says so: the
  // switch-off arm spends 1.05 to 1.54 `step()` calls per sensitive
  // destination while this benchmark's ACTS arm spends 2.2 to 4.4, because the
  // benchmark flies the whole gap between two hits in one propagation while a
  // navigator hands the destination over only for the last part of it.
  //
  // `transportCovarianceToBound` is paid once per bound state by both arms and
  // therefore cancels out of the difference. What does not cancel is the step:
  // the switch-off arm pays a Runge-Kutta step per call and the learned arm
  // pays one jump for the whole destination, so the arms to compare are one
  // `EigenStepper::step()` against one learned jump, both without the bound
  // state neither of them forms per step.
  const double tKernelJump =
      timeIt(jumps, kReps, [](const Jump& jj, double* sink) {
        Jump j = jj;
        j.bzMap = g_field.bz(j.pos);
        const collider_ml::Prediction p = collider_ml::transport(j);
        const collider_ml::JumpJacobian jj8 =
            collider_ml::jumpJacobian(j, p.mom, p.s, kPionMass);
        // the left-multiply LearnedStepper::writeBack does per jump. Named
        // as the kernel's own 8x8 rather than as Acts::FreeMatrix, which is
        // the same Eigen type but only exists in a WITH_ACTS build.
        collider_ml::Mat88 jacTransport = collider_ml::Mat88::Identity();
        jacTransport = (jj8.D * jacTransport).eval();
        *sink += p.pos.x() + jacTransport(4, 4) + jj8.t[3];
      });
  const double tHelixJump =
      timeIt(jumps, kReps, [](const Jump& jj, double* sink) {
        Jump j = jj;
        j.bzMap = g_field.bz(j.pos);
        const collider_ml::Prediction p = collider_ml::transport(j, false);
        const collider_ml::JumpJacobian jj8 =
            collider_ml::jumpJacobian(j, p.mom, p.s, kPionMass);
        collider_ml::Mat88 jacTransport = collider_ml::Mat88::Identity();
        jacTransport = (jj8.D * jacTransport).eval();
        *sink += p.pos.x() + jacTransport(4, 4) + jj8.t[3];
      });

  // --- HANDOFF item 2c's miss arm.
  //
  // A destination the track does not reach, built by reversing the source
  // momentum against the same real module plane: `solveArcLength` then solves
  // for a crossing behind the source and returns `ok == false`. The count of
  // failures is printed rather than assumed, because an arm that quietly
  // succeeded would be timing the arriving case again.
  //
  // What it costs is asymmetric and the asymmetry is in the source, not in the
  // timing. `LearnedStepper::step` calls `transport()` and only then consults
  // `p.ok` (LearnedStepper.hpp:262), and nothing inside `transport()` exits
  // early: the Newton loop runs its eight iterations whether or not it is
  // converging, and the forward pass runs before the flag is read. So a
  // destination the learned arm fails to reach costs the whole kernel and the
  // Runge-Kutta step it then falls back to, while the stock arm costs the step
  // alone. `jumpJacobian` is not in this arm because the failed branch never
  // reaches it.
  std::vector<Jump> missJumps;
  missJumps.reserve(jumps.size());
  for (const Jump& j : jumps) {
    Jump m = j;
    m.mom = Vec3(-j.mom.x(), -j.mom.y(), -j.mom.z());
    missJumps.push_back(m);
  }
  std::size_t nMissOk = 0;
  for (const Jump& j : missJumps) {
    bool ok = false;
    collider_ml::solveArcLength(j, &ok);
    nMissOk += ok ? 1 : 0;
  }
  const double tKernelMiss =
      timeIt(missJumps, kReps, [](const Jump& jj, double* sink) {
        Jump j = jj;
        j.bzMap = g_field.bz(j.pos);
        const collider_ml::Prediction p = collider_ml::transport(j);
        *sink += p.pos.x() + p.mom.z() + (p.ok ? 1.0 : 0.0);
      });
  const double tHelixMiss =
      timeIt(missJumps, kReps, [](const Jump& jj, double* sink) {
        Jump j = jj;
        j.bzMap = g_field.bz(j.pos);
        const collider_ml::Prediction p = collider_ml::transport(j, false);
        *sink += p.pos.x() + p.mom.z() + (p.ok ? 1.0 : 0.0);
      });

  const double net = tFull - tSolve;      // the learned part, on its own
  const double jac = tFullCov - tFull;    // the Jacobian, on its own
  const double ort = kOrtFixedNs + kOrtPerItemNs;

  // --- the ACTS arms
  double tActs = -1.0;
  double tActsCov = -1.0;
  double tFairCov = -1.0;
  double tFairCovHelix = -1.0;
  double tActsSetup = 0.0;
  std::vector<double> repActsCov;
  std::vector<double> repFairCov;
  std::vector<double> repFairCovHelix;
  BenchCounters actsCounters;
  bool haveCounters = false;
  // HANDOFF item 2c's per-step arm.
  double tActsStep = -1.0;
  double tActsStepWarm = -1.0;
  std::vector<double> repActsStep;
  std::vector<double> repActsStepWarm;
  BenchCounters stepCounters;
  bool haveStepCounters = false;
  double actsReached = 0.0;
  double actsPath = 0.0;
#ifdef WITH_ACTS
  ActsArm acts;
  if (acts.init(g_field, jumps)) {
    std::size_t reached = 0;
    for (std::size_t i = 0; i < jumps.size(); ++i) {
      const double s = acts.propagate(jumps[i], i, false);
      if (s >= 0.0) {
        ++reached;
        actsPath += s;
      }
    }
    actsPath /= std::max<std::size_t>(reached, 1);
    actsReached = 100.0 * static_cast<double>(reached) / jumps.size();

    // ACTS gets a smaller rep count: one propagation is ~1000x a helix solve
    // and best-of-5 over 200 reps of it would run for minutes to no purpose.
    const int kActsReps = std::max(1, kReps / 20);
    const double tActsFull =
        timeItIdx(jumps, kActsReps,
                  [&](const Jump& j, std::size_t i, double* sink) {
                    *sink += acts.propagate(j, i, false);
                  });
    // Repeated `repeats` times, because one invocation gives a number with no
    // error on it. `timeIt` already takes a best-of-5 inside itself, so what
    // the outer loop measures is the spread between independent best-of-5
    // measurements, which is what section 14 asks to be reported.
    for (int rep = 0; rep < repeats; ++rep) {
      const double full =
          timeItIdx(jumps, kActsReps,
                    [&](const Jump& j, std::size_t i, double* sink) {
                      *sink += acts.propagate(j, i, true);
                    });
      const double setup =
          timeItIdx(jumps, kActsReps,
                    [&](const Jump& j, std::size_t, double* sink) {
                      *sink += acts.start(j, false).parameters()[0];
                    });
      repActsCov.push_back(full - setup);
      tActsSetup = setup;
    }
    tActs = tActsFull - tActsSetup;
    tActsCov = median(repActsCov);

    // --- the same-work covariance arm.
    //
    // `tFullCov` above builds D and the tangent and stops, while `tActsCov`
    // additionally applies its Jacobian and forms the bound state, so those two
    // are not charged for the same work and the imbalance favours the kernel.
    // This arm charges the kernel side for the rest of what production runs:
    //
    //   1. `jacTransport = D * jacTransport` -- the left-multiply
    //      `LearnedStepper::writeBack` does (LearnedStepper.hpp:462). Written
    //      as a real 8x8 product into a running matrix, because that is what
    //      writeBack pays even when the running matrix is the post-reset
    //      identity.
    //   2. `Acts::detail::transportCovarianceToBound(...)` -- verbatim the
    //      function `EigenStepper::transportCovarianceToBound` forwards to
    //      (EigenStepper.ipp:159-166), on the same destination surface, the
    //      same start covariance and the same curvilinear start frame the
    //      ACTS arm uses. `LearnedStepper::transportCovarianceToBound`
    //      forwards to the inner EigenStepper (LearnedStepper.hpp:333), so
    //      this is the call a CKF on the learned stepper really ends with.
    //
    // The start's boundToFree Jacobian is computed inside the loop rather
    // than hoisted: the ACTS arm rebuilds it per propagation too, inside
    // `initialize`, whose cost sits in `tActsSetup`... except the covariance
    // half of initialize is not in tActsSetup (start(j,false) there), so
    // keeping it timed here errs against the kernel rather than for it.
    for (int rep = 0; rep < repeats; ++rep)
    repFairCov.push_back(timeItIdx(
        jumps, kReps, [&](const Jump& jj, std::size_t i, double* sink) {
          Jump j = jj;
          j.bzMap = g_field.bz(j.pos);
          const collider_ml::Prediction p = collider_ml::transport(j);
          const collider_ml::JumpJacobian jj8 =
              collider_ml::jumpJacobian(j, p.mom, p.s, kPionMass);

          // 1. what writeBack does
          Acts::FreeMatrix jacTransport = Acts::FreeMatrix::Identity();
          jacTransport = (jj8.D * jacTransport).eval();
          Acts::FreeVector derivative = jj8.t;

          // 2. what the inner EigenStepper's bound state does
          const Acts::Vector3 dir0 = j.mom.normalized();
          Acts::BoundToFreeMatrix jacToGlobal =
              Acts::CurvilinearSurface(j.pos, dir0).boundToFreeJacobian();
          Acts::FreeVector freeParams;
          const double pAbs = p.mom.norm();
          freeParams[Acts::eFreePos0] = p.pos.x();
          freeParams[Acts::eFreePos1] = p.pos.y();
          freeParams[Acts::eFreePos2] = p.pos.z();
          freeParams[Acts::eFreeTime] = jj8.dt;
          freeParams[Acts::eFreeDir0] = p.mom.x() / pAbs;
          freeParams[Acts::eFreeDir1] = p.mom.y() / pAbs;
          freeParams[Acts::eFreeDir2] = p.mom.z() / pAbs;
          freeParams[Acts::eFreeQOverP] = j.q / pAbs;

          Acts::BoundMatrix cov = acts.startCov();
          Acts::BoundMatrix fullJacobian = Acts::BoundMatrix::Zero();
          Acts::detail::transportCovarianceToBound(
              acts.gctx(), acts.target(i), cov, fullJacobian, jacTransport,
              derivative, jacToGlobal, std::nullopt, freeParams,
              Acts::FreeToBoundCorrection{});

          *sink += cov(0, 0) + fullJacobian(4, 4) + jacTransport(4, 4) +
                   p.pos.x();
        }));
    tFairCov = median(repFairCov);

    // The same matched arm with the network skipped, which is the
    // configuration NOTES 6.16 has reproducing the stock CKF. Without
    // this row the helix arm could only be compared to ACTS on the
    // unmatched pair, where the applying step is charged to ACTS alone.
    for (int rep = 0; rep < repeats; ++rep)
    repFairCovHelix.push_back(timeItIdx(
        jumps, kReps, [&](const Jump& jj, std::size_t i, double* sink) {
          Jump j = jj;
          j.bzMap = g_field.bz(j.pos);
          const collider_ml::Prediction p = collider_ml::transport(j, false);
          const collider_ml::JumpJacobian jj8 =
              collider_ml::jumpJacobian(j, p.mom, p.s, kPionMass);

          // 1. what writeBack does
          Acts::FreeMatrix jacTransport = Acts::FreeMatrix::Identity();
          jacTransport = (jj8.D * jacTransport).eval();
          Acts::FreeVector derivative = jj8.t;

          // 2. what the inner EigenStepper's bound state does
          const Acts::Vector3 dir0 = j.mom.normalized();
          Acts::BoundToFreeMatrix jacToGlobal =
              Acts::CurvilinearSurface(j.pos, dir0).boundToFreeJacobian();
          Acts::FreeVector freeParams;
          const double pAbs = p.mom.norm();
          freeParams[Acts::eFreePos0] = p.pos.x();
          freeParams[Acts::eFreePos1] = p.pos.y();
          freeParams[Acts::eFreePos2] = p.pos.z();
          freeParams[Acts::eFreeTime] = jj8.dt;
          freeParams[Acts::eFreeDir0] = p.mom.x() / pAbs;
          freeParams[Acts::eFreeDir1] = p.mom.y() / pAbs;
          freeParams[Acts::eFreeDir2] = p.mom.z() / pAbs;
          freeParams[Acts::eFreeQOverP] = j.q / pAbs;

          Acts::BoundMatrix cov = acts.startCov();
          Acts::BoundMatrix fullJacobian = Acts::BoundMatrix::Zero();
          Acts::detail::transportCovarianceToBound(
              acts.gctx(), acts.target(i), cov, fullJacobian, jacTransport,
              derivative, jacToGlobal, std::nullopt, freeParams,
              Acts::FreeToBoundCorrection{});

          *sink += cov(0, 0) + fullJacobian(4, 4) + jacTransport(4, 4) +
                   p.pos.x();
        }));
    tFairCovHelix = median(repFairCovHelix);

    actsCounters = acts.counters(jumps);
    haveCounters = true;

    // --- HANDOFF item 2c. One EigenStepper step, against one learned jump.
    //
    // The step size handed in is the jump's own 3D arc length, which is what
    // the navigator constrains the step to when it aims the stepper at the
    // destination. Jumps whose helix solve fails get the mean, so that this
    // arm runs on the same population as every other arm in the row rather
    // than on a subset selected by the kernel's own success.
    std::vector<double> stepH(jumps.size(), 0.0);
    double hSum = 0.0;
    std::size_t hN = 0;
    for (std::size_t i = 0; i < jumps.size(); ++i) {
      bool ok = false;
      const double s = collider_ml::solveArcLength(jumps[i], &ok);
      if (ok) {
        const Jump& j = jumps[i];
        stepH[i] = s * j.mom.norm() / std::hypot(j.mom.x(), j.mom.y());
        hSum += stepH[i];
        ++hN;
      }
    }
    const double hMean = hSum / std::max<std::size_t>(hN, 1);
    for (double& h : stepH) {
      if (h <= 0.0) {
        h = hMean;
      }
    }
    for (int rep = 0; rep < repeats; ++rep) {
      const double t0 =
          timeItIdx(jumps, kReps, [&](const Jump& j, std::size_t i,
                                      double* sink) {
            *sink += acts.nSteps(j, stepH[i], 0);
          });
      const double t1 =
          timeItIdx(jumps, kReps, [&](const Jump& j, std::size_t i,
                                      double* sink) {
            *sink += acts.nSteps(j, stepH[i], 1);
          });
      const double t2 =
          timeItIdx(jumps, kReps, [&](const Jump& j, std::size_t i,
                                      double* sink) {
            *sink += acts.nSteps(j, stepH[i], 2);
          });
      repActsStep.push_back(t1 - t0);
      repActsStepWarm.push_back(t2 - t1);
    }
    tActsStep = median(repActsStep);
    tActsStepWarm = median(repActsStepWarm);
    stepCounters = acts.stepCounters(jumps, stepH);
    haveStepCounters = true;
  }
#endif

  auto row = [&](const char* name, double ns, const char* note) {
    std::printf("%-38s %9.1f ns   %s\n", name, ns, note);
  };

  std::printf("--- no covariance ------------------------------------------\n");
  row("RKN to the plane", tRkn, "our reference impl, a FLOOR");
  std::printf("%40s mean %.1f steps, so %.0f distinct field points/jump; "
              "reached %.1f%%\n", "", meanRknSteps, 3.0 * meanRknSteps,
              100.0 * nRknReached / static_cast<double>(jumps.size()));
  if (tActs > 0.0) {
    row("ACTS EigenStepper to the plane", tActs, "same binary, same machine");
    std::printf("%40s reached the surface in %.1f%% of jumps; "
                "start-parameter setup %.1f ns subtracted\n",
                "", actsReached, tActsSetup);
    // A propagation that never arrives still costs time -- it runs to the step
    // limit and then reports a failure -- so a low reach fraction does not
    // show up as a missing number, it shows up as a large one. That is the
    // shape of error this project keeps having to catch, so it is loud.
    if (actsReached < 95.0) {
      std::printf(
          "%40s *** ONLY %.1f%% REACHED. The ACTS number above is inflated by\n"
          "%40s     failed propagations running to their step limit, and must\n"
          "%40s     not be quoted. Wrong jumps, or bounds too small. ***\n",
          "", actsReached, "", "");
    }
  } else {
    std::printf("%-38s %9s      %s\n", "ACTS EigenStepper", "--",
                "not built with -DWITH_ACTS (or no field.bin)");
  }
  row("helix + g_theta", tFull, "what test_kernel verified, and what ships");
  row("helix alone, network skipped", tHelix,
      "--stepper helix: NOTES 6.16 has this arm reproducing stock's four "
      "track-finding numbers");
  std::printf("\n");

  std::printf("--- with covariance ----------------------------------------\n");
  if (tActsCov > 0.0) {
    row("ACTS EigenStepper, cov transport", tActsCov, "jacTransport + bound");
  } else {
    std::printf("%-38s %9s\n", "ACTS EigenStepper, cov transport", "--");
  }
  row("helix + g_theta + jumpJacobian", tFullCov,
      "builds D and t, then STOPS: not the same work as the ACTS row");
  row("helix alone + jumpJacobian", tHelixCov,
      "same, with the network skipped");
  if (tFairCov > 0.0) {
    row("  + jacTransport update + boundState", tFairCov,
        "the matched arm: writeBack's left-multiply, then the same "
        "transportCovarianceToBound the ACTS arm ends with");
  }
  if (tFairCovHelix > 0.0) {
    row("  the same, network skipped", tFairCovHelix,
        "the matched arm for --stepper helix; quote this against the ACTS "
        "covariance row");
  }
  std::printf("\n");

  std::printf("--- inside the kernel --------------------------------------\n");
  row("plane solve, Newton x8", tSolve, "the whole intersection");
  std::printf("%40s %d iterations always; %d of %zu jumps intersect\n", "",
              collider_ml::detail::kNewton, nSolved, jumps.size());
  row("g_theta alone (feat+net+apply+1 read)", net, "");
  row("  of which the forward pass", tFull - tHelix,
      "what --stepper helix does not pay");
  row("the 8x8 transport Jacobian", jac, "");
  row("the same net under ORT, batch 1", ort + tSolve,
      "1788 + 98*b, measured");
  std::printf("\n");

  std::printf("--- do the two arms land on the same plane? ----------------\n");
  std::printf("  mean 3D path length:  helix %.1f mm", meanPath);
  if (actsPath > 0.0) {
    std::printf(",  ACTS %.1f mm  (%.2f%% apart)", actsPath,
                100.0 * std::abs(actsPath - meanPath) / meanPath);
  }
  std::printf("\n");
  std::printf("  off the destination plane, worst:  helix %.2e mm, "
              "RKN %.2e mm\n", offPlaneHelix, offPlaneRkn);
  std::printf("  distance between the two landings:  median %.3f mm, "
              "p99 %.2f mm, max %.1f mm\n", q(0.5), q(0.99), q(1.0));
  std::printf("  The second line is the constant-2T approximation, not an "
              "error in either\n  arm. It is the quantity g_theta is trained "
              "to remove.\n");
  std::printf("\n");

  // Speedup, in the same direction the notes quote it: baseline / learned, so
  // ">1 means the learned jump is that many times faster". The 6.0x this
  // project has been carrying since July is a number of this shape, and
  // printing the reciprocal next to it would guarantee the two get mixed up.
  std::printf("speedup vs our RKN floor:       %.2fx  (%s)\n", tRkn / tFull,
              tRkn > tFull ? "learned is faster"
                           : "the FLOOR is faster than the kernel");
  if (tActs > 0.0) {
    std::printf("speedup vs ACTS EigenStepper:   %.2fx  (%s)\n", tActs / tFull,
                tActs > tFull ? "learned is faster" : "ACTS is faster");
    std::printf("  with covariance on both:      %.2fx   (UNMATCHED arms: the "
                "applying step is charged to ACTS alone)\n",
                tActsCov / tFullCov);
    if (tFairCov > 0.0) {
      std::printf("  with covariance, matched:     %.2fx   (both arms end in "
                  "transportCovarianceToBound; quote THIS one)\n",
                  tActsCov / tFairCov);
    }
    std::printf("ACTS / our RKN floor:           %.2fx  "
                "(how much the floor understates)\n", tActs / tRkn);
  }
  std::printf("hand-written kernel over ORT:   %.2fx\n", (ort + tSolve) / tFull);
  std::printf("field points, RKN : learned  =  %.0f : 1\n",
              3.0 * meanRknSteps);

  if (tActs > 0.0) {
    std::printf(
        "\nWHAT THE ACTS NUMBER DOES AND DOES NOT INCLUDE.\n"
        "  Included: ACTS's own EigenStepper, its adaptive RKN with error\n"
        "  control, a real PlaneSurface intersection through the\n"
        "  SurfaceReached aborter, the same interpolated field map this file\n"
        "  loaded, and in the second table its covariance transport.\n"
        "  Excluded: the Navigator (volume, layer and portal traversal) and\n"
        "  material effects. Both cost real time inside a CKF, so production\n"
        "  ACTS costs MORE than the number above -- the ratio quoted here is\n"
        "  a conservative one for the learned side, which is the direction an\n"
        "  honest error should point. How much more is not measured anywhere\n"
        "  in this repository.\n");
  } else {
    std::printf(
        "\nREAD THIS BEFORE QUOTING THE RATIO.\n"
        "  The RKN above is a reference implementation, not ACTS's\n"
        "  EigenStepper. It has no material interaction, no boundary or\n"
        "  portal surfaces, no navigator, and no covariance. So this number\n"
        "  is a FLOOR on RKN's cost and the learned side is not entitled to\n"
        "  the difference. Rebuild with -DWITH_ACTS on the Linux box to get\n"
        "  the arm that settles it.\n");
  }

  // --- the output contract.
  //
  // Blank and zero are different statements here. `rk_steps` is left empty on
  // the two kernel arms because they do not integrate: a zero would claim they
  // took no steps, which is a measurement, while a blank says the quantity does
  // not exist for that arm.
  if (!csvPath.empty()) {
    if (tActsCov <= 0.0) {
      std::fprintf(stderr,
                   "\n--csv asked for but the ACTS arm did not run; the CSV\n"
                   "would carry the kernel arms with no baseline. Rebuild with\n"
                   "-DWITH_ACTS.\n");
      return 3;
    }

    // The kernel arms' field reads, counted rather than assumed. This replays
    // the timed body once, untimed, through the counting accessor. `transport`
    // and `jumpJacobian` never touch the map, so every read the arm makes is
    // the one below.
    g_field.resetCounts();
    double guard = 0.0;
    for (const Jump& jj : jumps) {
      Jump j = jj;
      j.bzMap = g_field.bzCounted(j.pos);
      const collider_ml::Prediction p = collider_ml::transport(j);
      const collider_ml::JumpJacobian j8 =
          collider_ml::jumpJacobian(j, p.mom, p.s, kPionMass);
      guard += p.pos.x() + j8.D(0, 0);
    }
    const double kernelLookups =
        static_cast<double>(g_field.counts()) /
        static_cast<double>(std::max<std::size_t>(jumps.size(), 1));
    if (guard == 12345.6789) {
      std::printf(" ");
    }

    const bool exists = std::ifstream(csvPath).good();
    std::ofstream csv(csvPath, std::ios::app);
    if (!csv) {
      std::fprintf(stderr, "cannot write %s\n", csvPath.c_str());
      return 3;
    }
    // Two columns past section 14's contract: the attempted and rejected step
    // counts. `rk_steps` is the accepted count, so a reader given only that
    // cannot divide the field reads by it and recover the reads-per-step
    // number, and a table whose third column cannot be checked against its
    // second is not readable. plot_speed.py names the columns it wants and
    // ignores the rest, so the superset costs it nothing.
    if (!exists) {
      csv << "pt_lo,pt_hi,arm,n_transport,ns_median,ns_lo,ns_hi,rk_steps,"
             "field_lookups,rk_steps_attempted,rk_steps_rejected,"
             "reached_frac\n";
    }

    // `ctr` rather than a bool, because there is now more than one stepping
    // arm and they do not share a counter: the whole-transport arm counts the
    // steps of a whole propagation and the per-step arm counts the trials
    // inside one call.
    auto emit = [&](const char* arm, const std::vector<double>& reps,
                    const BenchCounters* ctr, double lookups) {
      std::vector<double> v = reps;
      std::sort(v.begin(), v.end());
      csv << ptLo << "," << ptHi << "," << arm << "," << jumps.size() << ","
          << median(v) << ",";
      if (v.size() > 1) {
        csv << v.front() << "," << v.back();
      } else {
        csv << ",";
      }
      csv << ",";
      if (ctr != nullptr) {
        csv << ctr->steps;
      }
      csv << ",";
      if (lookups >= 0.0) {
        csv << lookups;
      }
      csv << ",";
      if (ctr != nullptr) {
        csv << ctr->attempted;
      }
      csv << ",";
      if (ctr != nullptr) {
        csv << ctr->rejected;
      }
      csv << ",";
      if (ctr != nullptr) {
        // The condition both counter columns are taken under: they are means
        // over the transports that reached, and this says how many did.
        csv << ctr->reached;
      }
      csv << "\n";
    };

    emit("acts_eigen_cov", repActsCov, haveCounters ? &actsCounters : nullptr,
         haveCounters ? actsCounters.fieldLookups : -1.0);
    emit("kernel_matched", repFairCov, nullptr, kernelLookups);
    emit("helix_matched", repFairCovHelix, nullptr, kernelLookups);
    // HANDOFF item 2c's arms. The unit here is one step or one jump, not one
    // transport; `n_transport` still says how many jumps the mean is over.
    // `acts_step` gets no field-read count: the counting propagator counts a
    // whole propagation and there is no propagation here, so the cell is left
    // blank rather than filled from a different arm.
    if (tActsStep > 0.0) {
      emit("acts_step", repActsStep,
           haveStepCounters ? &stepCounters : nullptr, -1.0);
      // The same call with a warm field cell. No counters of its own: the
      // trial counts belong to the cold measurement they were taken in.
      emit("acts_step_warm", repActsStepWarm, nullptr, -1.0);
    }
    emit("kernel_jump", {tKernelJump}, nullptr, kernelLookups);
    emit("helix_jump", {tHelixJump}, nullptr, kernelLookups);
    emit("kernel_miss", {tKernelMiss}, nullptr, kernelLookups);
    emit("helix_miss", {tHelixMiss}, nullptr, kernelLookups);
    csv.close();

    std::printf("\n--- section 14 row ------------------------------------\n");
    std::printf("pT %g to %g GeV, %zu of %zu transports, %d repeats\n", ptLo,
                ptHi, jumps.size(), nAll, repeats);
    std::printf("  acts_eigen_cov  %9.1f ns  steps %.2f  field reads %.2f\n",
                tActsCov, actsCounters.steps, actsCounters.fieldLookups);
    std::printf("  kernel_matched  %9.1f ns  steps -     field reads %.2f\n",
                tFairCov, kernelLookups);
    std::printf("  helix_matched   %9.1f ns  steps -     field reads %.2f\n",
                tFairCovHelix, kernelLookups);
    std::printf("  ratio acts / kernel  %.2fx   acts / helix  %.2fx\n",
                tActsCov / tFairCov, tActsCov / tFairCovHelix);
    std::printf("  attempted %.2f, rejected %.2f steps per transport; "
                "reads per attempted step %.2f\n",
                actsCounters.attempted, actsCounters.rejected,
                actsCounters.attempted > 0.0
                    ? actsCounters.fieldLookups / actsCounters.attempted
                    : 0.0);
    std::printf("  counters are means over the %.2f%% that reached; "
                "%zu of %zu did not\n",
                100.0 * actsCounters.reached, actsCounters.failed,
                jumps.size());

    // HANDOFF item 2c. The unit changes here: everything above is per
    // transport, everything below is per step or per jump.
    std::printf("\n--- per step, not per transport (HANDOFF 2c) ----------\n");
    if (tActsStep > 0.0) {
      std::printf("  acts_step       %9.1f ns  one EigenStepper::step(), "
                  "%.2f trials, %.2f rejected\n",
                  tActsStep, stepCounters.attempted, stepCounters.rejected);
      std::printf("  acts_step_warm  %9.1f ns  the same after another step, "
                  "field cell already loaded\n", tActsStepWarm);
    }
    std::printf("  kernel_jump     %9.1f ns  one learned jump, no bound "
                "state\n", tKernelJump);
    std::printf("  helix_jump      %9.1f ns  the same with the network "
                "skipped\n", tHelixJump);
    std::printf("  kernel_miss     %9.1f ns  destination not reachable, "
                "%zu of %zu solves still succeeded\n",
                tKernelMiss, nMissOk, missJumps.size());
    std::printf("  helix_miss      %9.1f ns  the same with the network "
                "skipped\n", tHelixMiss);
    if (tActsStep > 0.0) {
      std::printf("  a missed destination costs the learned arm "
                  "%.1f ns and the stock arm %.1f ns\n",
                  tHelixMiss + tActsStepWarm, tActsStepWarm);
    }
    std::printf("appended to %s\n", csvPath.c_str());
  }
  return 0;
}
