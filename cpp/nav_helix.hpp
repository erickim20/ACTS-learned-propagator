// Move along a helix by a three-dimensional path length.
//
// This is the second stage of the walk, which is what takes the
// module naming from 98.48 % to 99.73 %. The layer is chosen from where the
// track is, along the straight ray; a helix is then advanced from there by the
// straight ray's own path length to that layer's approach surface, and the
// layer is asked for its modules from where the helix landed.
//
// A copy, and deliberately so. The original is `cpp/nav_truth.cpp:342`, which
// is a measurement binary and is frozen: its tables are the
// regression test for this whole seam and a file that produces them should not
// also be a dependency of the thing being tested. That file carries its own
// self-check, printed on every run, and it is what validates this formula:
//
//   advance check     straight line at B=0 off by 0 mm; landing on the
//                     solved plane off by 3.06e-12 mm
//
// The first arm drives `bz` to zero and compares against a straight ray; the
// second advances by the arc length `solveArcLengthAtB` solved for a plane and
// checks that the landing point is on it. Neither can be run from here without
// the propagator harness, so the check stays there and this carries the note.
//
// NOTE THE UNIT of the parameter. dz/ds is pz/pt, so the helix parameter is the
// TRANSVERSE arc length, while every path length ACTS produces is the
// three-dimensional one. The two are converted at the one boundary below and
// never mixed.
#pragma once

#include <cmath>

#include "Acts/Definitions/Algebra.hpp"

namespace nav_walk {

/// Advance `(pos, dir)` along the helix of a track with |p| = `pAbs` and charge
/// sign `q` in a solenoid of `bz` tesla, by `path3D` millimetres of
/// three-dimensional path.
///
/// The polar angle is unchanged by a solenoid, so the direction differs from
/// the starting one only in phi.
///
/// `dir` is the direction of travel and `q` is signed against it. This turns
/// `dir` at `kappa = -0.3 q bz / pt`, which is the rate the momentum turns, so
/// a caller retracing a helix backwards -- travelling along the reversed
/// momentum -- has to hand over `-q`. `NewNavigator::walkFrom` does that
/// off the propagation direction. Negating `path3D` instead does not work: the
/// same argument advances z, and z still advances along the direction of
/// travel. `cpp/plan_probe.cpp` prints the round trip that separates the two.
inline void helixAdvance(const Acts::Vector3& pos, const Acts::Vector3& dir,
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

/// The path floor the second stage asks its layer with.
///
/// Negative, for a measured reason: the helix is advanced by a path
/// length that is not its own, so it can land past the approach surface, and a
/// positive floor would drop the modules the stage exists to find. 5.5 puts the
/// approach-surface to first-module distance at p10 10.3 mm, which is what sets
/// the size. In 50,519 transitions the named module lay at a negative path 0
/// times, so this has never actually been used.
inline constexpr double kStage2NearLimit = -10.0;

}  // namespace nav_walk
