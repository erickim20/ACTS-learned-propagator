// The ACTS adapter on the other side of the seam: a Navigator that names the
// next sensitive module from where the track already is, instead of naming the
// approach surface a few millimetres ahead of it, and delegates everything else
// to a stock `Acts::Navigator`.
//
// Why a navigator and not a stepper. A module becomes the stepper's target only
// for the last few millimetres of a leg, because `Navigator::resolveSurfaces`
// returns nothing while `currentLayer` is null (`Navigator.cpp:669`) and
// `currentLayer` is set only on arrival at the layer's own approach surface
// (`:398`). A stepper seam is therefore handed 18.1 mm of a 225 mm leg, and the
// learned transport never sees the leg it was fitted on. Moving the seam to the
// navigator is what gives it the leg.
//
// What the move costs, each measured before this file was written:
//
//   The material. A leg crosses 1 to 4 material-carrying surfaces and the
//   navigator no longer stops at them, so the CKF no longer applies them, which
//   costs 19.7 efficiency points in the softest pT bin. The seam therefore owes
//   the material itself, which is `applyMaterial` below.
//
//   The material-only track states. Those go, and nothing reads them: the chain
//   from `CombinatorialKalmanFilter.hpp:752-764` through
//   `TrackHelpers.hpp:399-414` and the branch stopper at
//   `acts_examples_src/TrackFindingAlgorithm.cpp:172-247` to every consumer
//   after track finding filters on `isMeasurement()`
//   first. The state count itself is reported in three files and will fall by
//   about a third.
//
//   The inner navigator's position-dependent state. That is the hard part and
//   it has its own section below.
//
// Two constraints come from ACTS rather than from the design:
//
//  1. `Acts::Navigator` is final (`Navigator.hpp:79`), exactly as
//     `Acts::EigenStepper` is. Inheritance is not available; this is
//     composition plus twelve members, ten of which forward.
//     `cpp/LearnedStepper.hpp` is the same move on `stepper_t` and needed
//     twenty-six.
//
//  2. `handleSurfaceReached` cannot be forwarded for a target this class named.
//     The stock body dereferences `state.navSurface()`, `state.navLayer()` or
//     `state.navBoundary()` depending on `navigationStage`
//     (`Navigator.cpp:381-430`), and each of those is
//     `navSurfaces.at(navSurfaceIndex.value())` on an `std::optional`. A module
//     named here is in none of those lists, so forwarding it throws
//     `std::bad_optional_access` out of ACTS, or falls through to the
//     `ACTS_ERROR("Surface reached but unknown state.")` at `:432`.
//
// Resynchronisation. Skipping a 225 mm leg invalidates `currentVolume`,
// `currentLayer`, `navigationStage` and three candidate lists, all of which
// describe where the track was. There are two ways out and this file takes the
// first:
//
//   `Navigator::initialize(state, position, direction, propDir)`
//   (`Navigator.cpp:80`) re-derives volume and layer from a position and calls
//   `state.resetForRenavigation()`. It is ACTS's own primitive and the CKF
//   itself uses it this way at `CombinatorialKalmanFilter.hpp:407-415`, where
//   it sets `state.navigation.options.startSurface` to the surface it is
//   restarting from and then calls it. `resync` below is that call.
//
//   Setting `currentVolume`, `currentLayer` and `navigationStage` by hand from
//   what the walk already knows is the other way, and is cheaper. It is not
//   taken. The walk knows the layer and the volumes it crossed, but the reset
//   also has to clear `navSurfaces`, `navLayers`, `navBoundaries`, their three
//   `std::optional` indices, the policy state and the navigation stream, and
//   getting one of those wrong produces a stale candidate rather than a crash.
//   `initialize` is one call, it is the code ACTS maintains, and it is paid
//   once per leg against a `compatibleLayers` and a `compatibleSurfaces` that
//   the walk pays anyway.
//
//
// Three things `initialize` does that have to be handled, all read out of
// `Navigator.cpp:80-210` rather than assumed:
//
//   It takes its starting point from `state.options.startSurface` (`:118`),
//   not from the `position` argument. Left alone that is the surface the
//   propagation started on, which is now far behind, and the surface check at
//   `:197-206` then fails with `NotOnExpectedSurface`. So `options.startSurface`
//   is pointed at the module actually landed on, for the duration of the call.
//
//   `resetForRenavigation` does not clear `startVolume` (`Navigator.hpp
//   :251-268`). With `options.startSurface` null, `initialize` takes the middle
//   branch at `:135` and resolves the layer inside the old start volume, then
//   fails the `inside` check at `:171-177`. So `startVolume` is nulled to force
//   the `lowestTrackingVolume` branch at `:149` whenever there is no surface to
//   start from.
//
//   It overwrites `startSurface`, `startVolume` and `startLayer`, which are the
//   propagation's own and are read by `determineMaterialUpdateMode`
//   (`PointwiseMaterialInteraction.hpp:55-57`) to decide whether a surface is
//   the start or the target. All three are saved and restored around the call,
//   so the seam does not silently redefine where the propagation began.
//
//
// Three members are overridden, not two: `nextTarget`, `handleSurfaceReached`
// and `checkTargetValid`. The third is forced by the resynchronisation above:
// `initialize` leaves `navigationStage` at `Stage::initial`
// (`Navigator.hpp:260`), and `Navigator::checkTargetValid` returns false in that
// state unconditionally (`Navigator.cpp:282-284`). The propagator polls it after
// every step and drops the target when it is false (`Propagator.ipp:158-163`),
// so with all twelve forwarding, every leg named here would be abandoned after
// exactly one step and the seam would fall back 100 % of the time while still
// reproducing `stock`. A target this navigator produced itself is valid until it
// is reached or missed, which is what the override says. The alternative,
// writing `navigationStage = Stage::surfaceTarget` after each resync, was not
// taken: it changes what the inner navigator's own first target does on every
// leg the seam declines, and that is a behaviour change on the path this class
// is supposed to leave alone.
//
// Checked at both gates. With `setUseWalk(false)` every one of the twelve
// members forwards, which is bit-identical to
// `Propagator<LearnedStepper, Acts::Navigator>` over 3,170,201 track-summary
// values on the muon sample. With it true the far module is the target and
// `cpp/LearnedStepper.hpp` carries the material. The counters in
// `cpp/NavCensus.hpp` are what separate "the wrapper is transparent" from "the
// wrapper is not in the loop", which no amount of identical output can.
#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

#include "Acts/Definitions/Direction.hpp"
#include "Acts/Geometry/TrackingVolume.hpp"
#include "Acts/Propagator/NavigationTarget.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/NavigatorConcept.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Utilities/Result.hpp"

#include "LegChannel.hpp"
#include "NavCensus.hpp"
#include "nav_helix.hpp"
#include "nav_walk.hpp"

namespace collider_ml {

class NewNavigator {
 public:
  using Inner = Acts::Navigator;
  using Config = Inner::Config;
  using Options = Inner::Options;

  /// The stock state, plus what the seam needs to know between two calls.
  ///
  /// Deriving rather than holding is what lets the ten forwarding members pass
  /// `State&` straight through: the inner navigator takes `Inner::State&` and
  /// the reference upcasts. Same shape as `LearnedStepper::State`.
  struct State : public Inner::State {
    using Inner::State::State;

    /// The propagation direction `initialize` was given. `handleSurfaceReached`
    /// does not receive it and `resync` needs it, so it is latched here.
    /// `Navigator::initialize` ignores it (`Navigator.cpp:82`), so this is
    /// carried for the signature rather than for the geometry.
    Acts::Direction propDir = Acts::Direction::Forward();

    /// The module this navigator named and the propagator has not yet reached
    /// or missed. Non-null means such a leg is in flight, and it is the one
    /// piece of state the three overridden members share.
    ///
    /// Set in `nextTarget`, cleared in `handleSurfaceReached` on arrival and in
    /// `nextTarget` on a fallback. `checkTargetValid` reads it and nothing
    /// else.
    const Acts::Surface* pending = nullptr;

    /// Per-propagation counters, in the shape `LearnedStepper::State` uses.
    /// They are folded into `NavCensus` and are here as well because a probe
    /// holding one state can read them and a CKF run cannot.
    std::uint64_t nOffered = 0;
    std::uint64_t nAnswered = 0;
    std::uint64_t nReturned = 0;
    std::uint64_t nCompleted = 0;
    std::uint64_t nFellBack = 0;
  };

  explicit NewNavigator(Config cfg,
                            std::shared_ptr<const Acts::Logger> logger =
                                Acts::getDefaultLogger(
                                    "NewNavigator",
                                    Acts::Logging::Level::INFO))
      : m_geo(cfg.trackingGeometry),
        m_logger(logger),
        m_inner(std::move(cfg), std::move(logger)) {
    // The inner navigator keeps its `Config` private, and the walk needs the
    // geometry on every leg. Held rather than reached for, so there is one
    // geometry and not two.
    if (m_geo == nullptr) {
      throw std::runtime_error("NewNavigator: no tracking geometry");
    }
  }

  State makeState(const Options& options) const { return State(options); }

  /// Arm the far-module target and the material loop.
  ///
  /// Off by default, and the same reasoning as `LearnedStepper::setUseLearned`:
  /// a navigator that silently changes which surfaces the filter sees is not
  /// something anyone should have to notice. With it off this class forwards
  /// all twelve members.
  ///
  void setUseWalk(bool on) { m_useWalk = on; }

  /// The material ablation.
  ///
  /// With the walk on and this off, the seam still names the far module and
  /// still flies the whole leg, and the material of the surfaces it flew past
  /// is applied by nobody. That is an arm no stock flag can build: the only
  /// stock flag that removes the material-only track states removes
  /// the material with them, and this removes the material and keeps
  /// everything else the moved seam does.
  ///
  /// An ablation has to switch the thing off, not substitute
  /// part of it. Nothing is planned, so nothing is applied, and the counters
  /// say so rather than the arithmetic quietly cancelling.
  void setUseMaterial(bool on) { m_useMaterial = on; }

  // ------------------------------------------------------------- forwarding

  const Acts::Surface* currentSurface(const State& s) const {
    return m_inner.currentSurface(s);
  }
  const Acts::TrackingVolume* currentVolume(const State& s) const {
    return m_inner.currentVolume(s);
  }
  const Acts::IVolumeMaterial* currentVolumeMaterial(const State& s) const {
    return m_inner.currentVolumeMaterial(s);
  }
  const Acts::Surface* startSurface(const State& s) const {
    return m_inner.startSurface(s);
  }
  const Acts::Surface* targetSurface(const State& s) const {
    return m_inner.targetSurface(s);
  }
  bool endOfWorldReached(const State& s) const {
    return m_inner.endOfWorldReached(s);
  }
  bool navigationBreak(const State& s) const {
    return m_inner.navigationBreak(s);
  }

  // ---------------------------------------------------- the four that do not

  /// Forwarded, and it clears what the seam carries.
  ///
  /// The CKF calls this at every branch reset (`CombinatorialKalmanFilter.hpp
  /// :409-415`), not only at the start of a propagation, and a reset abandons
  /// whatever leg was in flight. Leaving `pending` set across one would make
  /// the next `nextTarget` count a fallback for a leg that was never dropped,
  /// and would leave a plan in the channel that the stepper could still match.
  [[nodiscard]] Acts::Result<void> initialize(
      State& s, const Acts::Vector3& position, const Acts::Vector3& direction,
      Acts::Direction propagationDirection) const {
    s.propDir = propagationDirection;
    s.pending = nullptr;
    LegChannel::current().clearPlan();
    return m_inner.initialize(s, position, direction, propagationDirection);
  }

  /// Name the module the walk names, or hand back to ACTS.
  ///
  /// Four ways out and each is counted, because "the run reproduced stock" is
  /// the expected answer for all four and for a seam that never fired.
  Acts::NavigationTarget nextTarget(State& s, const Acts::Vector3& position,
                                    const Acts::Vector3& direction) const {
    NavCensus& c = NavCensus::instance();
    // Counted before anything else and with no condition on it. This is the
    // number that says the wrapper is in the propagator loop at all, which is
    // the entire content of Gate 1 and is invisible in the output.
    NavCensus::bump(c.wrapped);
    if (!m_useWalk) {
      return m_inner.nextTarget(s, position, direction);
    }

    // ------------------------------------------------------- the fallback
    //
    // Being asked again with a leg still in flight means the propagator dropped
    // it, and this is the only signal there is. `Propagator.ipp` produces it
    // three ways and the seam does not need to tell them apart: the pre-step
    // status came back `onSurface` and the target was skipped (`:54-62`), the
    // post-step status came back neither `onSurface` nor `reachable` and the
    // target was dropped (`:131-133`), or `checkTargetValid` said no (`:158`).
    //
    // The decision lives here because this is where the propagator states the
    // outcome. The alternative,
    // deciding in `handleSurfaceReached`, cannot work: that is called only on
    // arrival, so the one case that needs a fallback is the one case it is
    // never called for.
    //
    // The inner navigator still describes the module the leg started from and
    // the stepper has flown some distance since. Renavigating from the current
    // position before handing back is what stops it returning a candidate
    // resolved at a point the track has left.
    if (s.pending != nullptr) {
      ++s.nFellBack;
      NavCensus::bump(c.fellback);
      if (s.propDir == Acts::Direction::Backward()) {
        NavCensus::bump(c.fellbackBwd);
      }
      s.pending = nullptr;
      LegChannel::current().clearPlan();
      resync(s, position, direction, nullptr);
      return m_inner.nextTarget(s, position, direction);
    }

    // ------------------------------------------------------- is a leg offered
    //
    // The walk names the next module from a module. Anywhere else -- mid-flight
    // with no current surface, or standing on an approach surface or a boundary
    // because the seam declined the previous leg -- belongs to the stock
    // navigator.
    const Acts::Surface* here = m_inner.currentSurface(s);
    if (here == nullptr || !here->isSensitive() ||
        here->associatedLayer() == nullptr) {
      return m_inner.nextTarget(s, position, direction);
    }
    ++s.nOffered;
    NavCensus::bump(c.offered);
    if (s.propDir == Acts::Direction::Backward()) {
      NavCensus::bump(c.offeredBwd);
    }

    const Acts::Surface* pick = nullptr;
    const nav_walk::Result r = walkFrom(s, position, direction, *here, &pick);
    if (pick == nullptr) {
      NavCensus::bump(r.startLayerNearest != nullptr ? c.sameLayer
                                                     : c.walkFailed);
      return m_inner.nextTarget(s, position, direction);
    }
    ++s.nAnswered;
    NavCensus::bump(c.answered);

    // The target, built the way `Layer::compatibleSurfaces` builds one
    // (`Layer.cpp:167-172`): the closest intersection with its index, at the
    // navigator own exact edge test.
    const auto [isec, index] =
        pick->intersect(s.options.geoContext, position, direction,
                        Acts::BoundaryTolerance::None(),
                        s.options.surfaceTolerance)
            .closestWithIndex();
    if (!isec.isValid() || isec.pathLength() <= m_walkOpts.nearLimit) {
      // The walk named a module the ray does not reach from here. Counted as a
      // walk failure and not as a fallback: nothing was offered to the
      // propagator, so nothing was dropped.
      NavCensus::bump(c.walkFailed);
      return m_inner.nextTarget(s, position, direction);
    }

    // ------------------------------------------- the successor on this layer
    //
    // The walk excludes the layer the track is standing on, and that exclusion
    // is measured: 17,824 of 68,343 transitions have their successor on that
    // layer and are outside the walk domain.
    // Outside the walk domain is not outside the CKF: the stock navigator names
    // those out of the candidate list it resolved when it entered the layer,
    // and a seam that flies past them loses one measurement each. At 3.3 such
    // transitions per track against 13 measurements that is a quarter of the
    // multitrajectory, and it is a measurement loss rather than a bookkeeping
    // one.
    //
    // So the seam declines them. The leg is short, the stock navigator has
    // always named it, and the CKF applies its material at the stop as before.
    // The seam fires on the 74 % the walk was measured on and on nothing else.
    if (r.startLayerNearest != nullptr &&
        r.startLayerPath < isec.pathLength()) {
      NavCensus::bump(c.sameLayer);
      return m_inner.nextTarget(s, position, direction);
    }

    LegChannel& ch = LegChannel::current();
    ch.setPlan(pick);
    if (m_useMaterial) {
      for (const nav_walk::Crossing& x : r.material) {
        ch.addCrossing(x.surface, x.path);
      }
      NavCensus::bump(c.matPlanned, r.material.size());
    }

    s.pending = pick;
    ++s.nReturned;
    NavCensus::bump(c.returned);
    if (s.propDir == Acts::Direction::Backward()) {
      NavCensus::bump(c.returnedBwd);
    }
    // `currentSurface` is cleared here and not by the inner navigator, because
    // the inner navigator is not being called. `Navigator::nextTarget` does it
    // at `:216` and the CKF actor tests it on every step
    // (`CombinatorialKalmanFilter.hpp:278`), so leaving it set would filter the
    // same module once per step for the whole leg.
    s.currentSurface = nullptr;
    return Acts::NavigationTarget(isec, index, *pick,
                                  Acts::BoundaryTolerance::None());
  }

  /// A target this navigator named stays valid until it is reached or missed.
  ///
  /// See the header comment. With this forwarded, `initialize` having left
  /// `navigationStage` at `initial` makes the inner one return false on the
  /// first poll after every resync, and every leg is abandoned after one step.
  bool checkTargetValid(State& s, const Acts::Vector3& position,
                        const Acts::Vector3& direction) const {
    if (s.pending != nullptr) {
      return true;
    }
    return m_inner.checkTargetValid(s, position, direction);
  }

  /// Arrival. Handled here when the target is this navigator's, and not
  /// forwardable.
  ///
  /// `Navigator::handleSurfaceReached` dereferences `state.navSurface()`,
  /// `navLayer()` or `navBoundary()` according to `navigationStage`
  /// (`Navigator.cpp:381-430`), each of which is `.at(index.value())` on an
  /// `std::optional`. A module named by the walk is in none of those lists, so
  /// forwarding it throws `std::bad_optional_access` out of ACTS.
  void handleSurfaceReached(State& s, const Acts::Vector3& position,
                            const Acts::Vector3& direction,
                            const Acts::Surface& surface) const {
    if (s.pending == nullptr || &surface != s.pending) {
      m_inner.handleSurfaceReached(s, position, direction, surface);
      return;
    }
    s.pending = nullptr;
    LegChannel::current().clearPlan();
    ++s.nCompleted;
    NavCensus::bump(NavCensus::instance().completed);
    // Everything the inner navigator knows about where the track is describes
    // the module the leg started from, 225 mm back. This is where that is
    // repaired, and `resync` is the whole of it.
    resync(s, position, direction, &surface);
  }

 private:
  /// The two-stage walk, which names the reached module 99.73 % of the time
  /// against the one-stage walk's 98.48 %.
  ///
  /// Stage one chooses the layer from where the track is, along the straight
  /// ray. Stage two advances a helix to that layer approach surface by the
  /// straight ray own path length and asks the layer for its modules from where
  /// the helix landed, which removes the 225 mm lever arm from both
  /// `SurfaceArray::neighbors` and the per-module edge test.
  ///
  /// Zero Newton iterations and the local field, which is the arm kept: one
  /// iteration moves the landing point by p50 0.0001 mm and
  /// names the same module at every one of 50,519 transitions, and a nominal
  /// 2 T leaves 159 misses where the field the detector has leaves 134.
  ///
  /// Falls back to stage one answer when stage two finds nothing on the chosen
  /// layer, which happened 57 times in 50,519. That is not a fallback to the
  /// stock navigator: a module is still named and the leg still flies.
  nav_walk::Result walkFrom(const State& s, const Acts::Vector3& position,
                            const Acts::Vector3& direction,
                            const Acts::Surface& here,
                            const Acts::Surface** pick) const {
    *pick = nullptr;
    nav_walk::Result r =
        nav_walk::walk(*m_geo, s.options.geoContext, position, direction,
                       here.associatedLayer(), &here, m_walkOpts, *m_logger);
    if (!r.ok || r.nearest == nullptr) {
      return r;
    }
    *pick = r.nearest;

    const LegChannel::Track& tk = LegChannel::current().track();
    if (!tk.valid || r.layer == nullptr) {
      // No stepper has published yet, which is the first target of a
      // propagation. Stage one answer is what there is.
      return r;
    }
    // THE CURVATURE TURNS THE OTHER WAY ON A BACKWARD PASS.
    //
    // What arrives here is the direction of TRAVEL, not of the momentum:
    // `Propagator.ipp:106-107` sets `state.direction = state.options.direction
    // * stepper.direction(state.stepping)` and hands that to `nextTarget`
    // (`:44-45`). `helixAdvance` turns the direction it is given at
    // `kappa = -0.3 q Bz / pt` per unit of transverse arc, which is the rate
    // the MOMENTUM turns.
    //
    // Retracing the same helix backwards, that is the wrong sign. At forward
    // arc `-u` the momentum azimuth is `phi0 - kappa u`, and the travel
    // direction is the momentum reversed, so the travel azimuth is
    // `(phi0 + pi) - kappa u`: the same rate with the opposite sign. Negating
    // the path instead does not work, because the same argument also advances
    // z and z must still advance along the travel direction.
    //
    // The second CKF pass is backwards -- `TrackFindingAlgorithm.cpp:359`
    // inverts the first pass's direction and `sim/fatras_reco.py:259` sets
    // `twoWay=True` -- so this fires on every track that reaches it.
    const double qTravel = s.propDir.sign() * tk.q;
    Acts::Vector3 p1 = position, d1 = direction;
    nav_walk::helixAdvance(position, direction, tk.pAbs, qTravel, tk.bz,
                           r.layerPath, &p1, &d1);
    const nav_walk::LayerResolve s2 = nav_walk::resolveOnLayer(
        *r.layer, s.options.geoContext, p1, d1, nav_walk::kStage2NearLimit);
    if (s2.ok && s2.nearest != nullptr) {
      *pick = s2.nearest;
    }
    return r;
  }

  /// Put the inner navigator back in step with where the track now is.
  ///
  /// The header comment carries the argument for `initialize` over the partial
  /// resets, and the three things it does that have to be handled. This is
  /// that, and nothing more.
  void resync(State& s, const Acts::Vector3& position,
              const Acts::Vector3& direction, const Acts::Surface* on) const {
    NavCensus& c = NavCensus::instance();
    NavCensus::bump(c.resync);

    const Acts::Surface* savedStartSurface = s.startSurface;
    const Acts::TrackingVolume* savedStartVolume = s.startVolume;
    const Acts::Layer* savedStartLayer = s.startLayer;
    const Acts::Surface* savedOption = s.options.startSurface;

    auto attempt = [&](const Acts::Surface* from) {
      s.options.startSurface = from;
      // Nulled, not left. `resetForRenavigation` does not clear these
      // (`Navigator.hpp:253-268`), so with no surface to start from
      // `initialize` would resolve the layer inside the volume the PROPAGATION
      // started in and then fail its own `inside` check.
      s.startVolume = nullptr;
      s.startLayer = nullptr;
      return m_inner.initialize(s, position, direction, s.propDir);
    };

    Acts::Result<void> res = attempt(on);
    if (!res.ok()) {
      NavCensus::bump(c.resyncFailed);
      // Second attempt with no surface hint, which takes the
      // `lowestTrackingVolume` branch and cannot fail the surface check. The
      // arrival surface is written back afterwards because the CKF actor reads
      // `currentSurface` on the very next call
      // (`CombinatorialKalmanFilter.hpp:278`), and losing it here would drop
      // the measurement rather than the navigation.
      res = attempt(nullptr);
      s.currentSurface = on;
    }
    static_cast<void>(res);

    // Restored, because these are the PROPAGATION own and are read by
    // `determineMaterialUpdateMode` (`PointwiseMaterialInteraction.hpp:55-57`)
    // to decide whether a surface is the start or the target. A seam that
    // redefined where the propagation began would change which half of a slab
    // the CKF applies at the first and last surfaces.
    s.startSurface = savedStartSurface;
    s.startVolume = savedStartVolume;
    s.startLayer = savedStartLayer;
    s.options.startSurface = savedOption;
  }

  std::shared_ptr<const Acts::TrackingGeometry> m_geo;
  std::shared_ptr<const Acts::Logger> m_logger;
  Inner m_inner;
  bool m_useWalk = false;
  bool m_useMaterial = true;
  /// The walk's own options, at their defaults. Relaxing the boundary
  /// tolerance has no usable value: 1 mm
  /// recovers 251 misses and creates 3,276.
  nav_walk::Options m_walkOpts{};
};

static_assert(Acts::NavigatorConcept<NewNavigator>,
              "NewNavigator must remain substitutable for Acts::Navigator, "
              "or Propagator<LearnedStepper, NewNavigator> will not "
              "instantiate");

}  // namespace collider_ml
