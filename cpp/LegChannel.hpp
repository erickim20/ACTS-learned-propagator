// The one thing the navigator has to tell the stepper, and the one thing the
// stepper has to tell the navigator, for which ACTS has no channel.
//
// Why this exists.
//
// `Propagator` holds a stepper and a navigator and keeps their states apart:
// `state.stepping` and `state.navigation` are separate members and neither
// member function ever sees the other's state (`Propagator.ipp:92-180`). Only
// two things cross between them in the whole loop, and neither is enough:
//
//   `navigator.currentVolumeMaterial(state.navigation)` is handed to
//   `stepper.step` (`Propagator.ipp:95-96`). One `const IVolumeMaterial*`.
//
//   `nextTarget.surface()` is handed to `stepper.updateSurfaceStatus`
//   (`:49-53`, `:122-126`). One `const Surface&`.
//
// The moved seam needs two more, in opposite directions:
//
//   Navigator to stepper. Which material surfaces the leg crosses, and how far
//   along it each one sits, measured at 1 to 4 per leg. The
//   navigator is the only side that knows, because the walk is what finds them,
//   and `step()` is the only side that can act, because the covariance and the
//   momentum live on the stepping state. That is why the moved seam is a
//   change to the stepper as well as to the navigator.
//
//   Stepper to navigator. The momentum, the charge and the local field. The
//   second stage of the walk advances a helix, and a helix needs
//   all three; `nextTarget(state, position, direction)` carries none of them.
//   Without this the seam is the one-stage walk, which names the reached module
//   98.48 % of the time against the two-stage walk's 99.73 %, so the fallback
//   rate would be six times higher.
//
// Why a thread-local is sound here, and it is the part to be suspicious of.
//
//   One propagation is one stepping loop on one thread. `nextTarget`, `step`
//   and `handleSurfaceReached` are called from that loop in a fixed order and
//   never concurrently, so within a propagation this is ordinary sequential
//   code with the writes and reads interleaved by `Propagator::propagate`.
//
//   `ActsExamples` runs `TrackFindingAlgorithm::execute` on several threads.
//   Each has its own propagation and its own instance of this, which is what
//   `thread_local` buys and what a plain global would not.
//
//   The CKF branches but does not nest propagations. `findTracks` calls
//   `m_propagator.propagate` once per pass and the branching happens inside one
//   stepping loop (`CombinatorialKalmanFilter.hpp:825` fixes the actor list to
//   one entry and there is no inner propagator).
//
//   A stale plan cannot be consumed. The plan carries the destination surface
//   it was made for, and `LearnedStepper::step` acts on it only while the
//   target it was handed matches (`LearnedStepper.hpp`, `onPlan`). A plan left
//   behind by a propagation that ended abruptly is therefore inert rather than
//   wrong, which is the same property `LearnedStepper`'s latch has and for the
//   same reason.
//
// This is a channel and not a cache. Nothing here is read for anything except
// the current leg, and both sides clear what they wrote.
#pragma once

#include <cstddef>
#include <vector>

#include "Acts/Surfaces/Surface.hpp"

namespace collider_ml {

class LegChannel {
 public:
  /// One material surface on the plan, with its distance from the start of the
  /// leg along the straight ray. `nav_walk::Crossing` is where it comes from;
  /// it is copied into a plain pair so that `LearnedStepper` does not have to
  /// include the geometry walk.
  struct Crossing {
    const Acts::Surface* surface = nullptr;
    double path = 0.0;
  };

  /// What the navigator needs off the stepper to advance a helix.
  ///
  /// `bz` is in TESLA, not in ACTS's native units. `getField` answers in the
  /// native ones, where 1 T is 0.000299792458 (`Definitions/Units.hpp:173`),
  /// and `nav_walk::helixAdvance` is written in tesla like every other helix in
  /// this repository. Shipping the raw value once cost 18 to 80 points of
  /// tracking efficiency through a different door, so the
  /// unit is stated here rather than left to the call site.
  struct Track {
    bool valid = false;
    double pAbs = 0.0;
    double q = 0.0;
    double bz = 0.0;
  };

  static LegChannel& current() {
    thread_local LegChannel c;
    return c;
  }

  // ------------------------------------------------ stepper to navigator

  void publish(double pAbs, double q, double bz) {
    m_track.valid = true;
    m_track.pAbs = pAbs;
    m_track.q = q;
    m_track.bz = bz;
  }
  const Track& track() const { return m_track; }

  // ------------------------------------------------ navigator to stepper

  /// Hand over one leg's material. `destination` is what makes it consumable:
  /// the stepper acts on it only while that surface is the target it was given.
  void setPlan(const Acts::Surface* destination) {
    m_destination = destination;
    m_items.clear();
  }
  void addCrossing(const Acts::Surface* surface, double path) {
    m_items.push_back({surface, path});
  }
  void clearPlan() {
    m_destination = nullptr;
    m_items.clear();
  }

  const Acts::Surface* destination() const { return m_destination; }
  const std::vector<Crossing>& items() const { return m_items; }

 private:
  LegChannel() { m_items.reserve(8); }

  Track m_track;
  const Acts::Surface* m_destination = nullptr;
  std::vector<Crossing> m_items;
};

}  // namespace collider_ml
