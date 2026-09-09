// A Stepper that answers one sensitive-surface crossing with the learned
// kernel instead of a Runge-Kutta integration, and delegates everything else
// to a stock EigenStepper.
//
// Why a stepper. CombinatorialKalmanFilter is templated on `propagator_t`
// (CombinatorialKalmanFilter.hpp:161) and Propagator on `stepper_t`, so the
// hook exists at the type level. The CKF actor touches the stepper only
// through initialize / position / direction / boundState /
// transportCovarianceToBound / transportCovarianceToCurvilinear /
// releaseStepSize / particleHypothesis.
//
// Two alternative hooks were checked against the ODD image's headers and
// rejected:
//
//   MeasurementCalibrator is where ActsExamples::NeuralCalibrator puts a
//     learned model, but it produces V, the measurement covariance, while the
//     sigma head scales the process noise that goes into C. The two agree in
//     the chi2 gate, S = HCH' + V, and oppose each other in the gain,
//     K = CH'S^-1: inflating V makes the filter trust its prediction more,
//     inflating C makes it trust the measurement more.
//
//   An extra Actor cannot be added: the CKF hard-codes
//     `using Actors = ActorList<CombinatorialKalmanFilterActor>;`
//     (CombinatorialKalmanFilter.hpp:825), which would need an ACTS patch.
//
// Three constraints come from ACTS rather than from the design:
//
//  1. `Acts::EigenStepper` is final, so this is composition plus twenty-six
//     forwarding methods rather than inheritance.
//
//  2. `step(State&, Direction, const IVolumeMaterial*)` does not receive the
//     destination surface; the stepper sets step size and the target belongs
//     to the navigator. `updateSurfaceStatus` does receive it, so the target
//     is latched there and consumed on the next step.
//
//     updateSurfaceStatus has two call sites, Propagator.ipp:49 inside
//     `getNextTarget()` and Propagator.ipp:122 after the step, and
//     `getNextTarget()` runs only when the current target is exhausted, so
//     between consecutive steps it is the post-step call that refreshes the
//     latch. Two hazards follow:
//
//       a) `step()` is called unconditionally at the top of the loop,
//          including when `nextTarget.isNone()`, where a latch left set would
//          still be live.
//       b) `getNextTarget()` calls updateSurfaceStatus on candidates it then
//          skips (`continue` on `onSurface`), so a surface can be latched and
//          then rejected by the navigator.
//
//     `step()` consumes the latch, taking it and clearing it, so a latch is
//     used at most once and a stale one cannot survive into a later step.
//
//  3. `nextTarget.surface()` is not necessarily a measurement surface. It also
//     carries portal and layer-approach surfaces, which are cylinders and
//     discs too. The kernel was trained on sensitive-to-sensitive crossings,
//     so firing it at a portal is out of distribution with no signal that
//     anything is wrong. `toJump` requires `surface.isSensitive()`, the same
//     predicate the CKF actor uses to decide whether to expect measurements
//     (CombinatorialKalmanFilter.hpp:456). Everything else falls through to
//     Runge-Kutta.
//
// Checked inside a real `Propagator` over the ODD `TrackingGeometry`, with a
// `Navigator` and covariance transport on: bit-identical to a stock
// `EigenStepper` on 8 of 8 start states with the switch off and on. The switch
// changes nothing in that setup because `toJump` fires on nothing there; the
// counters on `State` below are what separate the two cases. See
// `cpp/geom_probe.cpp`.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>

#include "Acts/Definitions/Units.hpp"
#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Propagator/StepperConcept.hpp"
#include "Acts/Surfaces/BoundaryTolerance.hpp"
#include "Acts/Surfaces/CylinderBounds.hpp"
#include "Acts/Surfaces/CylinderSurface.hpp"
#include "Acts/Surfaces/DiscSurface.hpp"
#include "Acts/Surfaces/Surface.hpp"

#include "Acts/Definitions/Common.hpp"
#include "Acts/Definitions/Tolerance.hpp"
#include "Acts/Material/MaterialSlab.hpp"
#include "Acts/Propagator/ConstrainedStep.hpp"
#include "Acts/Propagator/detail/PointwiseMaterialInteraction.hpp"
#include "Acts/Utilities/MathHelpers.hpp"

#include "CellSigma.hpp"
#include "LearnedJacobian.hpp"
#include "LearnedNoise.hpp"
#include "LearnedTransport.hpp"
#include "LegChannel.hpp"
#include "NavCensus.hpp"
#include "TransportCensus.hpp"
#include "nav_helix.hpp"

namespace collider_ml {

class LearnedStepper {
 public:
  using Inner = Acts::EigenStepper<>;
  using Options = Inner::Options;
  using Jacobian = Inner::Jacobian;
  using Covariance = Inner::Covariance;
  using BoundState = Inner::BoundState;
  // Not `Inner::BoundParameters`. That typedef exists in ACTS 44.99.99 but not
  // in the main builds shipped in the perf and ODD images, where the compile
  // dies on this line alone. StepperConcept asks for
  // `s.initialize(state, std::tuple_element_t<0, BoundState>)`, so taking the
  // type from BoundState is what the concept actually requires and it holds on
  // both versions.
  using BoundParameters = std::tuple_element_t<0, BoundState>;

  /// The stock state, plus the latched target and the switch.
  struct State : public Inner::State {
    using Inner::State::State;

    /// Set by updateSurfaceStatus, consumed by step, cleared after use.
    const Acts::Surface* target = nullptr;
    /// The last sensitive destination the census counted on this propagation.
    /// Held so that a destination reached over several RKN steps is counted
    /// once, which is what makes the census count transports and not steps.
    /// HANDOFF item 2.
    const Acts::Surface* censusTarget = nullptr;
    /// Which census bin that destination went into, so arrival can be credited
    /// to the bin the transport was counted in rather than to the bin its pT
    /// falls in a few steps later. Cleared once credited, so no destination is
    /// marked as arriving twice. HANDOFF item 2c.
    int censusSlot = TransportCensus::kNone;
    /// Steps aimed at that destination so far.
    std::uint64_t censusSteps = 0;
    /// Millimetres flown at that destination so far, and the two bins to
    /// credit them to.
    ///
    /// The bins are held separately from `censusSlot` because arrival clears
    /// that slot to stop a second credit, and the path is retired at arrival
    /// too. Keeping its own copy means the two do not have to be ordered.
    double censusPath = 0.0;
    int censusPathSlot = TransportCensus::kNone;
    int censusEta = TransportCensus::kNone;
    /// Whether `censusPath` belongs to a destination that has not been retired.
    bool censusPathOpen = false;
    /// Off by default. A stepper that silently changes the physics is not
    /// something anyone should have to notice.
    bool useLearned = false;
    /// The last jump's per-hit sigma scale, for whoever inflates Q with it.
    double sigmaScale = 1.0;

    /// Process noise owed to the covariance but not yet added to it.
    ///
    /// A matrix and not a flag. Two things have to hold at once and a boolean
    /// gets only the first:
    ///
    ///   `transportCovarianceToBound` is called MORE than once per jump. The
    ///   CKF alone reaches it at CombinatorialKalmanFilter.hpp:419, :481 and
    ///   :612. So the second and later calls must add nothing, or Q enters as
    ///   many times as the filter happens to ask. Adding it twice inflates the
    ///   covariance in the direction that makes a filter look stable, which is
    ///   the worst way for a bug to fail.
    ///
    ///   More than one jump can happen between two of those calls. RKN steps
    ///   and learned jumps interleave, and a flag would let the second jump
    ///   overwrite the first jump's debt, silently losing it. Two jumps owe two
    ///   Q, so this accumulates and is zeroed when paid.
    ///
    /// It is evaluated at the jump, not at the transport: by then `state.pars`
    /// has moved to the destination and the sigma head belongs to a different
    /// jump.
    Mat55 qPending = Mat55::Zero();
    /// Counted for the same reason the jump counters are: "the covariance
    /// changed" does not say whether Q entered once, twice or not at all.
    std::size_t nQArmed = 0;
    std::size_t nQApplied = 0;

    /// What the learned arm actually did, narrowing at each stage.
    ///
    /// These exist because "the output was bit-identical to stock" is evidence
    /// of nothing on its own. It is the expected answer when the adapter is
    /// transparent and when the jump never fires, and those are opposite
    /// facts that a bit-identical run cannot tell apart. With these, the two
    /// readings are distinguishable from the outside: `latched > 0` says the
    /// navigator is handing targets over, `accepted > 0` says the sensitivity
    /// guard passes them, and `fired == 0` after that is the mismatch and
    /// nothing else.
    std::size_t nLatched = 0;   ///< targets updateSurfaceStatus handed us
    std::size_t nAccepted = 0;  ///< ... that accepts() said yes to
    std::size_t nBuilt = 0;     ///< ... that toJump turned into a Jump
    std::size_t nFired = 0;     ///< ... the kernel answered and we wrote back

    /// The moved seam material plan, as this stepper is walking it.
    ///
    /// `LegChannel` holds the plan the navigator made and this holds where the
    /// stepper has got to in it. The two are separate because the plan is one
    /// per leg and shared, and the position in it is one per stepping state:
    /// the CKF branches, and two branches of one propagation share a plan while
    /// having flown different distances along it.
    ///
    /// `planStart` is `pathAccumulated` when this leg began, so the distance
    /// flown is `propDir * (pathAccumulated - planStart)`. Signed, because
    /// `EigenStepper.ipp:368` accumulates the signed step and the second CKF
    /// pass runs backward (`TrackFindingAlgorithm.cpp:524-531`).
    const Acts::Surface* planTarget = nullptr;
    double planStart = 0.0;
    std::size_t planNext = 0;

    /// The state at the start of the leg, latched with the plan.
    ///
    /// A LEARNED JUMP CANNOT LAND ON A CROSSING. It flies from the module to
    /// the module in one call, so the step-size constraint above has nothing
    /// to act on and `applyDueMaterial` is never reached: the plan would be
    /// made, never consumed, and the leg would carry no material at all,
    /// which costs 19.7 efficiency points in the softest pT bin.
    ///
    /// `applyPlanRest` applies the crossings after the jump instead, and it
    /// needs a point on each surface to evaluate the slab at. The straight ray
    /// is not that point: `pathCorrection` is one over the cosine of the
    /// incidence angle and reaches 8.7 here, so the second-stage helix is
    /// advanced from these five numbers to each
    /// crossing's own path, which is the same trajectory the navigator used to
    /// name the module.
    Acts::Vector3 planPos = Acts::Vector3::Zero();
    Acts::Vector3 planDir = Acts::Vector3::UnitZ();
    double planPAbs = 0.0;
    double planQ = 1.0;
    double planBz = 0.0;

    /// The transport the seam closed mid-leg and the filter never saw.
    ///
    /// No counter reads a material-only state, which is true but not the
    /// whole of it. Each track state
    /// stores the transport jacobian from the previous track state
    /// (`CombinatorialKalmanFilter.hpp:746`), and the smoother reads exactly
    /// that: `GainMatrixSmoother.hpp:78-88` asserts it and hands it to
    /// `calculate`, `KalmanGlobalCovariance.hpp:76` shows the same
    /// `C_filt F' (C_pred)^-1` product in the open, and the mBF variant takes it
    /// as `F` outright (`MbfSmootherImpl.hpp:20`). In stock a material surface
    /// gets its own
    /// state, so the chain module -> material -> module is two jacobians and
    /// is complete.
    ///
    /// Applying material mid-leg requires closing the transport there:
    /// `transportCovarianceToCurvilinear` is what puts the covariance at the
    /// crossing so the noise lands in the right place, and it resets
    /// `jacTransport` to the identity and overwrites `state.jacobian`
    /// (`EigenStepper.ipp:150-156`). With no track state created at the
    /// crossing that first jacobian is dropped on the floor, and the state at
    /// the destination module then claims the previous state is only the last
    /// few millimetres away. Measured: the smoother survives it, the perigee
    /// extrapolation fails 439 times in 10,000 events, and the track selector
    /// rejects two thirds of the candidates.
    ///
    /// So the seam keeps the product. Each mid-leg reset multiplies its
    /// jacobian in on the left, and the next call that closes a segment for
    /// real hands the filter the whole module-to-module transport.
    Acts::BoundMatrix legJacobian = Acts::BoundMatrix::Identity();
    bool legJacobianPending = false;
    /// Slabs actually applied, against `NavCensus::matPlanned`, which the
    /// navigator counts in a different file on a different state. They are
    /// expected to agree and are printed side by side because the failure that
    /// matters, a plan nothing consumed, moves them apart and moves nothing
    /// else.
    std::size_t nMatApplied = 0;
  };

  explicit LearnedStepper(std::shared_ptr<const Acts::MagneticFieldProvider> b)
      : m_inner(std::move(b)) {}

  /// The cell widths the position outputs are written in. In ACTS these belong
  /// to the calibration context rather than to the stepper; until that is
  /// wired, they are set here or the correction comes out scaled wrong.
  void setCellSigma(double sig0, double sig1) {
    m_sig0 = sig0;
    m_sig1 = sig1;
  }

  /// The per-jump sigma, which is what the position outputs were TRAINED in.
  ///
  /// `setCellSigma` above is two constants for a whole run and the training
  /// divisor is not: `train_gtheta.py:600-607` divides each jump's target by
  /// `hypot(module resolution, measured covariance width)`, keyed on the
  /// module's class and the track's pT and |eta| bin. On the ODD that is 15/15
  /// on pixels against 43/1200 on short strips, so two constants cannot be
  /// right for both and were 2.9x too large on one and 28x too small on the
  /// other. `cpp/CellSigma.hpp` says the rest.
  ///
  /// A borrowed pointer, owned by the caller for the life of the stepper, the
  /// same shape as the Q table. Null keeps the constants, which is what a run
  /// with no table gets and is the arm the table is measured against.
  void setCellSigmaTable(const CellSigmaTable* t) { m_cellSigmaTable = t; }

  /// Arm the learned transport for states this stepper makes.
  ///
  /// `useLearned` lives on the State, which is the right place for it -- but
  /// inside a Propagator the State is made by `makeState()` and never handed
  /// out before stepping starts, so with only the State field there is no
  /// reachable way to switch the thing on in a real propagation. This is the
  /// same shape as setCellSigma: a knob on the stepper that seeds the state.
  /// Still off by default, so nothing changes physics without being asked.
  void setUseLearned(bool on) { m_useLearned = on; }

  /// The per-hit sigma head, which scales Q and nothing else.
  ///
  /// A switch rather than always-on, because "turning the sigma head on
  /// changes the covariance and nothing else" is a claim that can only be
  /// checked by running both ways. With it off the scale is pinned at 1 and
  /// the network's sixth output is simply not used; the state, the path and
  /// the Jacobian are untouched either way, which is the thing being checked.
  void setUseSigmaHead(bool on) { m_useSigmaHead = on; }

  /// The helix bypass: take the learned jump but skip the network entirely.
  ///
  /// Not the same as a build with the weights zeroed. That build still runs
  /// the forward pass and multiplies by zero, so every timing taken on it
  /// carries the network's cost while showing none of its effect, which makes
  /// its wall times unreadable. This skips the pass. The physics is identical
  /// by construction and the acceptance test is that it is identical in the
  /// output as well.
  void setUseNetwork(bool on) { m_useNetwork = on; }

  /// Fire the learned transport only on the transport the walk named.
  ///
  /// The gate above it is `accepts()`, which tests the surface type and not
  /// who asked for it; this adds the second half. Off by default, so a build
  /// from this tree still reproduces the earlier arms, and the counter it
  /// reads is written either way.
  void setPlannedOnly(bool on) { m_plannedOnly = on; }

  /// Skip the network where the map is within `tesla` of the helix's own 2 T.
  ///
  /// Zero is off, which is the default, and the arm is
  /// then what it was. Independent of `setPlannedOnly` and measured apart from
  /// it, because both change when the network runs and in one configuration
  /// their effects are not separable.
  void setFieldGate(double tesla) { m_fieldGate = tesla; }

  /// Run the helix core at the map value at each jump's source.
  ///
  /// Measured offline at 1.957 of the helix's 7.606 points of `hits lost` on
  /// the held-out fold, and 7.886 of its 26.012 on the
  /// population the field gate hands to the network. Off by default, so a
  /// build from this tree reproduces every arm taken before it.
  ///
  /// No field lookup is added. `toJump` already reads `bzMap` from the inner
  /// stepper's own cache on every jump, for the network's input 10.
  ///
  /// Not for an arm with the network on. The fit's target, its inputs 8 and 9,
  /// the header's `mu` and `sd` and the `v1_scale` cells were all written at
  /// 2 T, so a model above a moved core is a model read
  /// off a core it was not fitted on. `reco_ckf.py` refuses that combination.
  void setLocalCore(bool on) { m_localCore = on; }

  /// Stop publishing to `LegChannel` on every step. A measurement, not a fix.
  ///
  /// `publishTrack` runs after every `m_inner.step`, and `NewNavigator` reads
  /// what it wrote once per surface-to-surface transport. The wrapper as a
  /// whole measures 0.30 s of the 0.42 s the moved seam costs, unsplit, so
  /// which share is this call and which is the twelve forwarding members is
  /// not measured. This switch is the split: it is the
  /// only thing inside the wrapper that does arithmetic per step.
  ///
  /// It is only meaningful with the walk off. With the walk on the navigator
  /// reads a channel nothing is writing to and the arm is not a comparison, it
  /// is a different and wrong physics, so `learned_ckf.cpp` refuses that
  /// combination rather than producing a number someone could quote.
  void setPublishTrack(bool on) { m_publishTrack = on; }

  State makeState(const Options& options) const {
    State s{options, m_inner.makeState(options).fieldCache};
    s.useLearned = m_useLearned;
    return s;
  }

  // ------------------------------------------------------ the two overrides

  Acts::IntersectionStatus updateSurfaceStatus(
      State& state, const Acts::Surface& surface, std::uint8_t index,
      Acts::Direction propDir, const Acts::BoundaryTolerance& tolerance,
      double surfaceTolerance, Acts::ConstrainedStep::Type stype,
      const Acts::Logger& logger = Acts::getDummyLogger()) const {
    state.target = &surface;  // the only place ACTS hands us the destination
    const Acts::IntersectionStatus status = m_inner.updateSurfaceStatus(
        state, surface, index, propDir, tolerance, surfaceTolerance, stype,
        logger);
    // HANDOFF item 2c. This is the arrival signal and the only one: after each
    // step the propagator polls the same target here, and `onSurface` is what
    // sends it into `handleSurfaceReached` (Propagator.ipp:122-131). A status
    // that is neither `onSurface` nor `reachable` drops the target, and the
    // destination is then simply never credited, which is what makes
    // `n_transport - n_reached` the transports that were paid for and not
    // completed.
    //
    // No arrival can be seen before its own transport is counted. The pre-step
    // poll in `getNextTarget` (Propagator.ipp:43-71) skips a target the stepper
    // is already on, so a destination that reaches `step()` was `reachable`
    // when it was latched and `state.censusTarget` is set by the step before
    // the first poll that can return `onSurface`.
    if (auto& census = TransportCensus::instance();
        census.on() && status == Acts::IntersectionStatus::onSurface &&
        &surface == state.censusTarget) {
      if (state.censusSlot >= 0) {
        census.recordReached(state.censusSlot, state.censusSteps);
        state.censusSlot = TransportCensus::kNone;
      }
      // Arrival is where a destination's path is complete. Retiring it here
      // rather than waiting for the next latch is what keeps the path records
      // in step with the arrivals.
      retirePath(&state);
    }
    return status;
  }

  Acts::Result<double> step(State& state, Acts::Direction propDir,
                            const Acts::IVolumeMaterial* material) const {
    // CONSUME the latch: take it and clear it, whatever happens next. A latch
    // is then usable at most once, which is what makes hazards (a) and (b) in
    // the header comment unreachable -- a target that the navigator latched and
    // then skipped, or a step taken with no target at all, cannot pick up a
    // surface left over from an earlier iteration.
    const Acts::Surface* target = state.target;
    state.target = nullptr;

    // The counters narrow one stage at a time, and they are counted even with
    // the switch off. `latched` under `useLearned == false` is what proves the
    // latch mechanism itself works, which no amount of identical output can.
    if (target != nullptr) {
      ++state.nLatched;
      NavCensus::bump(NavCensus::instance().jumpLatched);
    }

    // HANDOFF item 2. Counted before the switch is consulted and before any
    // arm-specific work, so every arm sees the same population: what is being
    // counted is what the NAVIGATOR asked for, which does not depend on which
    // stepper answers. Recorded once per destination, so the count is one per
    // surface-to-surface transport and not one per RKN step.
    // Whether the step about to be taken is aimed at the destination being
    // censused, so that its length can be credited once it is known. The step
    // returns that length and the census cannot be closed before it does.
    bool censusAimed = false;
    if (auto& census = TransportCensus::instance(); census.on()) {
      if (target == nullptr || !accepts(*target)) {
        // A step toward a portal, a layer approach, or nothing. The learned
        // jump never fires on these, so both arms pay them and they belong
        // outside the bins rather than in one.
        census.recordOtherStep();
      } else {
        if (target != state.censusTarget) {
          // A different destination: the previous one is finished, whether it
          // arrived or was dropped, so its path is retired before the state is
          // overwritten.
          retirePath(&state);
          state.censusTarget = target;
          const Acts::Vector3 p = direction(state) * absoluteMomentum(state);
          state.censusSlot = census.record(std::hypot(p.x(), p.y()));
          state.censusSteps = 0;
          state.censusPath = 0.0;
          state.censusPathSlot = state.censusSlot;
          state.censusEta = TransportCensus::etaBin(absEta(direction(state)));
          state.censusPathOpen = true;
        }
        // The step about to be taken, counted before it is taken and before
        // the switch decides who takes it. One transport can cost many of
        // these, and how many is what HANDOFF item 2c needs beside the count.
        ++state.censusSteps;
        census.recordStep(state.censusSlot);
        censusAimed = true;
        // The map at the SOURCE of this step. `getField` is the inner
        // stepper's own cache, so this is a lookup and not an integration, and
        // it is taken in every arm rather than only where `toJump` runs.
        //
        // DIVIDED BY Acts::UnitConstants::T. `getField` answers in ACTS's
        // native units, where 1 T is 0.000299792458 (`Definitions/Units.hpp`
        // :173), so the raw z component of a 2 T solenoid is 5.996e-4. The
        // census reports tesla because that is the unit the map, the teacher
        // and `kBHelix` are all in.
        if (auto b = m_inner.getField(state, position(state)); b.ok()) {
          census.recordField(state.censusPathSlot, state.censusEta,
                             b->z() / Acts::UnitConstants::T);
        }
      }
    }
    // ------------------------------------------ the moved seam material
    //
    // The navigator no longer stops the stepper at the 1 to 4 material
    // surfaces a leg crosses, so the CKF no longer applies them, and losing
    // that costs 19.7 efficiency points in the softest pT bin. This is where
    // they are paid.
    //
    // The plan is matched on its destination, so a plan left behind by a
    // propagation that ended abruptly is inert rather than wrong. Same property
    // as the latch above and for the same reason.
    LegChannel& channel = LegChannel::current();
    // `planned` is the whole of the match: this target is the one the walk
    // named for this transport. `onPlan` adds only that the walk found
    // material to pay on the way, and the two are separate because a planned
    // transport that crosses no material is still a planned transport.
    // `NewNavigator.hpp:377` publishes the destination outside the
    // `m_useMaterial` guard, so `planned` is set on every transport the walk
    // names and `onPlan` is not.
    const bool planned = target != nullptr && channel.destination() == target;
    const bool onPlan = planned && !channel.items().empty();
    if (onPlan) {
      if (state.planTarget != target) {
        state.planTarget = target;
        state.planStart = state.pathAccumulated;
        state.planNext = 0;
        state.planPos = position(state);
        state.planDir = direction(state);
        state.planPAbs = absoluteMomentum(state);
        state.planQ = charge(state) >= 0 ? 1.0 : -1.0;
        state.planBz = 0.0;
        if (auto b = m_inner.getField(state, state.planPos); b.ok()) {
          state.planBz = b->z() / Acts::UnitConstants::T;
        }
      }
      // Land on the next crossing rather than somewhere past it.
      //
      // Without this the material goes in at the end of whichever step happened
      // to cross the surface, and the covariance is then split at that point
      // instead of at the surface. A misplaced split costs 0.028 of a module
      // resolution at 4.7 mm, so an
      // overshoot of a few millimetres is small, and it is free to remove.
      //
      // This is a step-size constraint and not a navigation stop. The navigator
      // still does not report the surface, so `currentSurface` stays null, the
      // CKF actor does nothing and no track state is added
      // (`CombinatorialKalmanFilter.hpp:278`). The material is applied and the
      // multitrajectory entry is not, which is the whole content of the moved
      // seam.
      //
      // `Type::Actor` and a positive magnitude, which is what
      // `SteppingHelper.hpp:71-72` passes: `ConstrainedStep` stores magnitudes
      // and `EigenStepper.ipp:272` applies the sign. The propagator releases
      // this constraint after every step (`Propagator.ipp:116`), so it binds
      // one step and is re-made from scratch on the next.
      if (state.planNext < channel.items().size()) {
        const double flown = propDir * (state.pathAccumulated - state.planStart);
        const double togo = channel.items()[state.planNext].path - flown;
        if (togo > 0.0) {
          m_inner.updateStepSize(state, togo,
                                 Acts::ConstrainedStep::Type::Actor);
        }
      }
    }

    // ------------------------------------------- who named this destination
    //
    // `accepts` is a surface-type test and nothing else, so on its own the
    // learned transport also fires on targets the walk never named. When
    // `NewNavigator` declines a leg it hands back to `Acts::Navigator`, which
    // targets the layer's approach surface and then the module, and that
    // module is sensitive. What reaches the stepper is then a
    // transport of about 10 mm rather than a whole surface-to-surface leg, and
    // over 18 mm a helix is already exact to 9 um against
    // tens of microns of hit resolution. There is nothing there to correct and
    // whatever the network adds is its own error on top of a correct answer.
    //
    // `jumpUnplanned` counts those whether or not they are suppressed, so
    // their number is a measurement rather than the difference of two
    // counters. `m_plannedOnly` is what suppresses them.
    const bool sensitive =
        state.useLearned && target != nullptr && accepts(*target);
    if (sensitive) {
      // `accepted` keeps the meaning it has always had, a destination
      // `accepts()` said yes to, so it reads the same on both arms and
      // `unplanned` is a subset of it rather than a different denominator.
      ++state.nAccepted;
      NavCensus::bump(NavCensus::instance().jumpAccepted);
      if (!planned) {
        NavCensus::bump(NavCensus::instance().jumpUnplanned);
      }
    }
    if (sensitive && (planned || !m_plannedOnly)) {
      Jump jump;
      if (toJump(state, *target, &jump)) {
        ++state.nBuilt;
        NavCensus::bump(NavCensus::instance().jumpBuilt);
        // ------------------------------- nothing to correct at nominal field
        //
        // The helix core runs at a fixed 2 T (`LearnedTransport.hpp:73`) and
        // the map inside the barrel is within 5 % of that at every module, so
        // there
        // the helix is already right and any correction is noise. `bzMap` is
        // the map at the source of this transport, which is the network's own
        // input 10. Above the threshold the network runs; at or below it the
        // helix answer is kept and nothing else changes, so the arm is
        // `new-helix` in the barrel and `new-learned` outside it.
        //
        // Zero is off and is the default. It is a threshold in tesla, not a
        // fraction: 0.1 T is the 5 % of 2 T the barrel and endcap line is
        // drawn at.
        const bool nearNominal =
            m_fieldGate > 0.0 &&
            std::abs(jump.bzMap - detail::kBHelix) <= m_fieldGate;
        if (nearNominal) {
          NavCensus::bump(NavCensus::instance().jumpNearField);
        }
        // The branch, named once. The gate is one of the two ways the network
        // is off; `--stepper new-helix` is the other, and Q has the same
        // problem on both, so the table is selected on whether the network
        // ran and not on the gate alone.
        const bool netOn = m_useNetwork && !nearNominal;
        const Prediction p = transport(jump, netOn);
        if (p.ok) {
          ++state.nFired;
          NavCensus::bump(NavCensus::instance().jumpFired);
          // The return value is the step LENGTH, which is what EigenStepper
          // returns (`EigenStepper.ipp:368` accumulates the same h). The chord
          // |p.pos - position| is a different and always smaller number.
          const double h = writeBack(&state, jump, p);
          state.sigmaScale = m_useSigmaHead ? p.m : 1.0;
          armNoise(&state, *target, jump, !netOn);
          // The whole leg has just been flown, so every crossing on the plan
          // is behind the state now. `applyPlanRest` is where they are paid.
          if (onPlan) {
            applyPlanRest(&state, propDir, channel, std::abs(h));
          }
          creditPath(&state, censusAimed, h);
          publishTrack(&state);
          return h;
        }
      }
    }
    auto inner = m_inner.step(state, propDir, material);
    if (inner.ok()) {
      creditPath(&state, censusAimed, *inner);
      if (onPlan) {
        applyDueMaterial(&state, propDir, channel);
      }
    }
    // Published after the step, so what the navigator reads is the state at the
    // arrival and not the state one step before it. `LegChannel.hpp` says why
    // there is no other way to get it there.
    publishTrack(&state);
    return inner;
  }

  // ------------------------------------------------------------- forwarding

  /// Forwarded, and it publishes.
  ///
  /// The channel is `thread_local` and outlives a propagation, so without this
  /// the first `nextTarget` of a track would advance its helix at the PREVIOUS
  /// track momentum. The CKF makes that the common case rather than a corner
  /// one: it re-initialises the stepper and then the navigator at every branch
  /// reset (`CombinatorialKalmanFilter.hpp:400-415`), always in that order, so
  /// publishing here means the navigator never reads a momentum belonging to
  /// something else.
  ///
  /// The plan is not cleared here. `NewNavigator::initialize` does that,
  /// and it is the side that made it.
  void initialize(State& s, const BoundParameters& par) const {
    m_inner.initialize(s, par);
    resetPlan(&s);
    publishTrack(&s);
  }
  void initialize(State& s, const Acts::BoundVector& boundParams,
                  const std::optional<Acts::BoundMatrix>& cov,
                  Acts::ParticleHypothesis hypothesis,
                  const Acts::Surface& surface) const {
    m_inner.initialize(s, boundParams, cov, hypothesis, surface);
    resetPlan(&s);
    publishTrack(&s);
  }
  Acts::Result<Acts::Vector3> getField(State& s,
                                       const Acts::Vector3& pos) const {
    return m_inner.getField(s, pos);
  }
  Acts::Vector3 position(const State& s) const { return m_inner.position(s); }
  Acts::Vector3 direction(const State& s) const { return m_inner.direction(s); }
  double qOverP(const State& s) const { return m_inner.qOverP(s); }
  double absoluteMomentum(const State& s) const {
    return m_inner.absoluteMomentum(s);
  }
  Acts::Vector3 momentum(const State& s) const { return m_inner.momentum(s); }
  double charge(const State& s) const { return m_inner.charge(s); }
  const Acts::ParticleHypothesis& particleHypothesis(const State& s) const {
    return m_inner.particleHypothesis(s);
  }
  double time(const State& s) const { return m_inner.time(s); }
  void updateStepSize(State& s, const Acts::NavigationTarget& target,
                      Acts::Direction dir,
                      Acts::ConstrainedStep::Type stype) const {
    m_inner.updateStepSize(s, target, dir, stype);
  }
  void updateStepSize(State& s, double stepSize,
                      Acts::ConstrainedStep::Type stype) const {
    m_inner.updateStepSize(s, stepSize, stype);
  }
  double getStepSize(const State& s, Acts::ConstrainedStep::Type stype) const {
    return m_inner.getStepSize(s, stype);
  }
  void releaseStepSize(State& s, Acts::ConstrainedStep::Type stype) const {
    m_inner.releaseStepSize(s, stype);
  }
  std::string outputStepSize(const State& s) const {
    return m_inner.outputStepSize(s);
  }
  /// Forwarded, and it closes the leg.
  ///
  /// The CKF reads the jacobian out of the returned `BoundState` and stores it
  /// on the track state (`CombinatorialKalmanFilter.hpp:493` and `:746`), so
  /// the value returned here is the one the smoother will use. When this
  /// actually transports it is the call that closes the segment; when it does
  /// not (`transportCov` false, which is what `filter()` passes) the segment
  /// was closed by `transportCovarianceToBound` just before and there is
  /// nothing left pending.
  Acts::Result<BoundState> boundState(
      State& s, const Acts::Surface& surface, bool transportCov = true,
      const Acts::FreeToBoundCorrection& c =
          Acts::FreeToBoundCorrection(false)) const {
    if (!(s.covTransport && transportCov)) {
      return m_inner.boundState(s, surface, transportCov, c);
    }
    auto res = m_inner.boundState(s, surface, transportCov, c);
    if (res.ok()) {
      composeLegJacobian(&s);
      std::get<1>(*res) = s.jacobian;
    }
    return res;
  }
  bool prepareCurvilinearState(State& s) const {
    return m_inner.prepareCurvilinearState(s);
  }
  BoundState curvilinearState(State& s, bool transportCov = true) const {
    if (!(s.covTransport && transportCov)) {
      return m_inner.curvilinearState(s, transportCov);
    }
    BoundState out = m_inner.curvilinearState(s, transportCov);
    composeLegJacobian(&s);
    std::get<1>(out) = s.jacobian;
    return out;
  }
  /// Forwarded, and it drops a pending leg jacobian rather than carrying it.
  ///
  /// This re-seats the covariance and `jacToGlobal` at a surface from the
  /// filtered state (`EigenStepper.ipp:129-137`), which the CKF does after an
  /// update (`CombinatorialKalmanFilter.hpp:600`). Everything before that point
  /// has already been closed into a track state, so a jacobian still pending
  /// here would belong to a segment that no longer exists.
  void update(State& s, const Acts::FreeVector& freeParams,
              const Acts::BoundVector& boundParams, const Covariance& cov,
              const Acts::Surface& surface) const {
    m_inner.update(s, freeParams, boundParams, cov, surface);
    s.legJacobian = Acts::BoundMatrix::Identity();
    s.legJacobianPending = false;
  }
  void update(State& s, const Acts::Vector3& pos, const Acts::Vector3& dir,
              double qop, double t) const {
    m_inner.update(s, pos, dir, qop, t);
  }
  /// Forwarded, and it closes the leg.
  ///
  /// The CKF calls this at a material-only surface it stopped at
  /// (`CombinatorialKalmanFilter.hpp:475`). The seam declines some legs, so
  /// the stock navigator still reports material surfaces on those and this
  /// path is live.
  void transportCovarianceToCurvilinear(State& s) const {
    m_inner.transportCovarianceToCurvilinear(s);
    if (s.covTransport) {
      composeLegJacobian(&s);
    }
  }
  /// The seam where the measured process noise enters.
  ///
  /// The CKF's sequence at each hit is fixed and this is the first of its three
  /// steps:
  ///
  ///   stepper.transportCovarianceToBound(...)   <- C = F C F', and then Q
  ///   detail::performMaterialInteraction(..., addNoise, ...)
  ///   stepper.boundState(...)
  ///
  /// So delegating and then adding puts the measured Q between the transport
  /// and ACTS's own scattering noise, which is the right order and the only
  /// place a
  /// stepper method can reach. The hard-coded actor list at
  /// CombinatorialKalmanFilter.hpp:825 does not have to be worked around.
  void transportCovarianceToBound(
      State& s, const Acts::Surface& surface,
      const Acts::FreeToBoundCorrection& c =
          Acts::FreeToBoundCorrection(false)) const {
    m_inner.transportCovarianceToBound(s, surface, c);
    if (s.covTransport) {
      composeLegJacobian(&s);
    }

    if (!s.covTransport || s.qPending.isZero()) {
      return;
    }
    // Top-left 5x5 only. The bound layout is (loc0, loc1, phi, theta, q/p,
    // time) and Q was measured over the first five; the time row and column are
    // left alone because nothing measured them, not because they are zero.
    s.cov.topLeftCorner<5, 5>() += s.qPending;
    // PAID. Everything after this point in the same jump adds nothing, which is
    // what makes a second call harmless.
    s.qPending.setZero();
    ++s.nQApplied;
  }

  /// The measured process noise. Not owned, and not defaulted to anything: with
  /// no table set the stepper transports the covariance and adds nothing, which
  /// is the honest behaviour and the one the plumbing check in `geom_probe`
  /// expects.
  void setNoiseTable(const NoiseTable* q) { m_q = q; }

  /// Apply the seam material plan, or only fly it.
  ///
  /// The plan does two things to a leg and they are separable. It says WHERE
  /// the leg has to pause, which the step-size constraint in `step()` acts on
  /// whether or not any material follows, and it says WHAT to apply there.
  /// `NewNavigator::setUseMaterial(false)` removes both, so an arm run that
  /// way prices the pair and not either one.
  ///
  /// An ablation has to switch the thing off, not substitute
  /// part of it. This is the second half of the pair, so the three arms are
  /// no plan, plan without material, and plan with material.
  void setUseSeamMaterial(bool on) { m_useSeamMaterial = on; }

 private:
  /// Forget where the propagation was in a leg, because the leg is gone.
  ///
  /// `EigenStepper::initialize` sets `pathAccumulated` back to zero
  /// (`EigenStepper.ipp:48`) and the CKF calls it at every branch reset on the
  /// SAME stepping state (`CombinatorialKalmanFilter.hpp:400`). `planStart` is
  /// a `pathAccumulated` from before that reset, so a plan left open across one
  /// measures the distance flown against an origin that no longer exists. When
  /// the new leg happens to have the same destination the reset in `step()`
  /// does not fire, and the material of that leg is then applied at the wrong
  /// point or not at all. Silent either way.
  void resetPlan(State* state) const {
    state->planTarget = nullptr;
    state->planStart = 0.0;
    state->planNext = 0;
    state->legJacobian = Acts::BoundMatrix::Identity();
    state->legJacobianPending = false;
  }

  /// Hand the navigator what a helix needs and `nextTarget` does not carry.
  ///
  /// `nextTarget(state, position, direction)` has no momentum, no charge and no
  /// field, and the second stage of the walk needs all three. Costs
  /// one cached field lookup per step.
  ///
  /// Divided by `Acts::UnitConstants::T`. `getField` answers in ACTS native
  /// units where 1 T is 0.000299792458 (`Definitions/Units.hpp:173`), and every
  /// helix in this repository is written in tesla. Shipping the raw value once
  /// cost 18 to 80 points of tracking efficiency through the network feature,
  /// which is the same mistake one door along.
  void publishTrack(State* state) const {
    // The guard is first, before the field lookup, because pricing this call
    // is the whole point of the switch and a lookup outside the guard would
    // leave the largest part of it in the arm it was meant to remove.
    if (!m_publishTrack) {
      return;
    }
    const auto b = m_inner.getField(*state, position(*state));
    if (!b.ok()) {
      return;
    }
    LegChannel::current().publish(absoluteMomentum(*state),
                                  charge(*state) >= 0 ? 1.0 : -1.0,
                                  b->z() / Acts::UnitConstants::T);
  }

  /// Hand back the whole module-to-module transport, once.
  ///
  /// Called from every member that closes a segment the filter will store.
  /// Consuming rather than only reading is what makes a second call on the
  /// same surface harmless, which matters because `filter()` reaches
  /// `transportCovarianceToBound` and then `boundState` for one track state
  /// (`CombinatorialKalmanFilter.hpp:477` and `:488`).
  void composeLegJacobian(State* state) const {
    if (!state->legJacobianPending) {
      return;
    }
    state->jacobian = state->jacobian * state->legJacobian;
    state->legJacobian = Acts::BoundMatrix::Identity();
    state->legJacobianPending = false;
  }

  /// Apply every crossing the step just flew past, in order.
  ///
  /// A loop and not one call: 28.5 % of transitions cross more than one
  /// material surface. The tolerance is
  /// `Acts::s_onSurfaceTolerance`, because the constrained step lands on the
  /// crossing to within the integrator own tolerance and an exact test would
  /// defer every crossing by one step.
  void applyDueMaterial(State* state, Acts::Direction propDir,
                        const LegChannel& channel) const {
    const auto& items = channel.items();
    const double flown = propDir * (state->pathAccumulated - state->planStart);
    while (state->planNext < items.size() &&
           flown >= items[state->planNext].path - Acts::s_onSurfaceTolerance) {
      const Acts::Surface* surface = items[state->planNext].surface;
      ++state->planNext;
      if (surface != nullptr) {
        applyMaterial(state, propDir, *surface);
      }
    }
  }

  /// Every crossing still on the plan, applied at the end of a learned jump.
  ///
  /// This is not the same arithmetic as `applyDueMaterial`. The stepper arm
  /// lands on each crossing, so `transportCovarianceToCurvilinear` there puts
  /// the noise where it belongs and the remaining transport carries it:
  /// `C = F2 (F1 C F1' + Q) F2'`. A learned jump has already flown the whole
  /// leg by the time this runs, so the noise is added at the destination and
  /// carried back to the crossing by the lever arm in `applyMaterial` rather
  /// than by `F2`. That is close enough to try because the three angular
  /// components are unchanged by the split to 1e-8 and the position block is
  /// the whole of it, which is what a lever arm is.
  ///
  /// A full Jacobian for a partial jump is not built. The lever arm against
  /// no lever arm measures 0.5 efficiency points in the softest bin, against a
  /// transport error of
  /// 21 points, so on this evidence the full split would not change the
  /// reading.
  ///
  /// The slab is evaluated where the leg's own helix reaches each crossing,
  /// not where the jump landed and not on the straight ray. That is the point
  /// the stepper arm lands on, so the two arms apply the same slab; the
  /// difference between them is where the noise goes and nothing else.
  void applyPlanRest(State* state, Acts::Direction propDir,
                     const LegChannel& channel, double legPath) const {
    const auto& items = channel.items();
    while (state->planNext < items.size()) {
      const Acts::Surface* surface = items[state->planNext].surface;
      const double path = items[state->planNext].path;
      ++state->planNext;
      if (surface == nullptr) {
        continue;
      }
      Acts::Vector3 p = state->planPos, d = state->planDir;
      nav_walk::helixAdvance(state->planPos, state->planDir, state->planPAbs,
                             state->planQ, state->planBz, path, &p, &d);
      // The plan's path is the straight ray's and `legPath` is the jump's own
      // three-dimensional one. Over a 225 mm leg the two differ by less than a
      // part in a thousand, which is far below what a lever arm is sensitive
      // to; the floor at zero is for a crossing the plan puts past the module,
      // which `nav_walk::detail::finalizeMaterial` trims but the ray-against-
      // helix difference can still leave marginal.
      applyMaterial(state, propDir, *surface, &p, &d,
                    std::max(0.0, legPath - path));
    }
  }

  /// One material surface, applied the way the CKF applies one.
  ///
  /// The stock sequence at a material-only surface is two lines of
  /// `CombinatorialKalmanFilter.hpp`:
  ///
  ///   `stepper.transportCovarianceToCurvilinear(state.stepping)`   (`:471`)
  ///   `detail::performMaterialInteraction(..., PreUpdate, ...)`    (`:481`)
  ///
  /// and a `PostUpdate` call at `:612` on the way out. Pre plus Post is the
  /// whole slab by construction of the split factor, so this makes one
  /// `FullUpdate` call in place of the two. With the ODD split factor of 1 the
  /// Post half is empty and `performMaterialInteraction` returns immediately on
  /// `slab.isVacuum()`, so the two are the same arithmetic; where a split
  /// factor is not 1 they differ at second order through the log term in
  /// Highland formula and through the energy loss between the halves.
  ///
  /// The transport is the point, and it is the covariance ordering.
  /// `transportCovarianceToCurvilinear` moves `cov` to here and resets
  /// `jacTransport` to the identity, so the noise is added at
  /// the crossing and the remaining transport carries it the rest of the way:
  /// `C = F2 (F1 C F1' + Q) F2'`, and with k crossings it nests k deep for
  /// free. At the distances a moved seam sees, the unsplit form costs 0.33 of
  /// a module resolution at the median crossing, which would allow splitting
  /// at the farthest crossing only. Landing on
  /// each crossing makes the full loop cost nothing extra, so the
  /// simplification is not taken.
  ///
  /// `determineMaterialUpdateMode` is not consulted. It restricts the mode only
  /// when the surface is the propagation own start or target surface
  /// (`PointwiseMaterialInteraction.cpp:16-28`); the start surface is the
  /// module the CKF restarted from and the target surface is null in forward
  /// filtering (`CombinatorialKalmanFilter.hpp:407`), so no crossing on a leg
  /// can be either. The stepper has no route to those two pointers in any case.
  ///
  /// `evalPos` and `evalDir` override the point the SLAB is evaluated at and
  /// nothing else. They are null on the stepper arm, where the step landed on
  /// the crossing and the state's own position is the right point, and set on
  /// the learned arm, where it did not. Everything after the slab -- the
  /// transport, the energy loss, the noise -- still happens at the state.
  ///
  /// `leverArm` is the path from the crossing to the state, in millimetres,
  /// and is zero on the stepper arm because there is none: the step landed on
  /// the crossing, so the transport that follows carries the noise itself.
  void applyMaterial(State* state, Acts::Direction propDir,
                     const Acts::Surface& surface,
                     const Acts::Vector3* evalPos = nullptr,
                     const Acts::Vector3* evalDir = nullptr,
                     double leverArm = 0.0) const {
    if (!m_useSeamMaterial) {
      return;
    }
    const Acts::GeometryContext& gctx = state->options.geoContext;
    const Acts::Vector3 dir = direction(*state);

    // The point the step landed on, and not a re-intersection of the surface.
    //
    // This is what `evaluateMaterialSlab(state, stepper, surface, mode)` uses
    // (`PointwiseMaterialInteraction.hpp:95-101`) and therefore what stock
    // applies, and the step-size constraint above is what makes it the right
    // point: the step is aimed at the crossing, so the position is on the
    // surface to within the integrator own tolerance.
    //
    // Re-intersecting is wrong. `pathCorrection` is one over the cosine of the
    // incidence angle and reaches 8.7 in this detector (`cpp/plan_probe.cpp`),
    // so on a surface the track grazes, a solution picked by `closest()` on the
    // far root moves the evaluation point a long way and scales the slab by a
    // factor that has nothing to do with the crossing.
    //
    // Measured against stock, with the probe evaluating where this evaluates:
    // the thickness ratio is 1.0000 through the ninetieth percentile, 1.0001 at
    // the ninety-ninth and 1.78 at its worst, and `pathCorrection` agrees to
    // four decimals at every percentile.
    const Acts::Vector3 pos =
        evalPos != nullptr ? *evalPos : position(*state);
    const Acts::Vector3 sdir = evalDir != nullptr ? *evalDir : dir;

    const Acts::MaterialSlab slab = Acts::detail::evaluateMaterialSlab(
        gctx, surface, propDir, pos, sdir, Acts::MaterialUpdateMode::FullUpdate);
    if (slab.isVacuum()) {
      // The surface carries a map and the map is empty here. Counted, because
      // "the material did nothing" and "the plan was never consumed" are
      // otherwise the same observation.
      NavCensus::bump(NavCensus::instance().matVacuum);
      return;
    }

    if (state->covTransport) {
      m_inner.transportCovarianceToCurvilinear(*state);
      // The segment just closed, kept because nothing else will keep it. See
      // `State::legJacobian`.
      state->legJacobian = state->jacobian * state->legJacobian;
      state->legJacobianPending = true;
    }

    const Acts::detail::PointwiseMaterialEffects effects =
        Acts::detail::computeMaterialEffects(
            slab, particleHypothesis(*state), dir,
            static_cast<float>(qOverP(*state)), m_matScattering, m_matEnergyLoss,
            state->covTransport);

    // The rest is `performMaterialInteraction`
    // (`PointwiseMaterialInteraction.hpp:185-221`) with the propagator state
    // taken out of it. That overload is templated on a type with `.stepping`
    // and `.options.direction` and `step()` has neither, so the arithmetic is
    // written out rather than called.
    const Acts::ParticleHypothesis& hypothesis = particleHypothesis(*state);
    const double mass = hypothesis.mass();
    const double absQ = hypothesis.absoluteCharge();
    const double momentum = absoluteMomentum(*state);
    const double qop = qOverP(*state);

    const double nextE =
        Acts::fastHypot(mass, momentum) - effects.eLoss * propDir;
    double nextP = (mass < nextE) ? Acts::fastCathetus(nextE, mass) : 0.0;
    static constexpr double kMinP = 10 * Acts::UnitConstants::MeV;
    nextP = std::max(kMinP, nextP);
    const double nextQOverP =
        hypothesis.qOverP(nextP, std::copysign(absQ, qop));

    m_inner.update(*state, position(*state), dir, nextQOverP, time(*state));

    if (state->covTransport) {
      auto add = [](double variance, double change) {
        return std::max(0.0, variance + change);
      };
      state->cov(Acts::eBoundPhi, Acts::eBoundPhi) =
          add(state->cov(Acts::eBoundPhi, Acts::eBoundPhi),
              effects.variancePhi);
      state->cov(Acts::eBoundTheta, Acts::eBoundTheta) =
          add(state->cov(Acts::eBoundTheta, Acts::eBoundTheta),
              effects.varianceTheta);
      state->cov(Acts::eBoundQOverP, Acts::eBoundQOverP) =
          add(state->cov(Acts::eBoundQOverP, Acts::eBoundQOverP),
              effects.varianceQoverP);

      // The noise has to arrive at the right place, not only in the right
      // amount.
      //
      // The three variances above are the scattering as seen at the crossing.
      // The stepper arm stops there, so `jacTransport` carries them the rest
      // of the way on its own and the block below is not reached. A learned
      // jump has already flown past, so adding them here gives
      // `F C F' + sum Q_i` where the filter needs
      // `F C F' + sum F2_i Q_i F2_i'`, and the position block is the whole
      // difference: the three angular components are unchanged by the split
      // to 1e-8 and sigma_loc0 is low by 0.33 of a module resolution at the
      // median crossing.
      //
      // `F2_i` restricted to what it does to a scattering angle is a lever
      // arm. In the curvilinear frame at the state, with
      // `U = (Z x T)/|Z x T|` and `V = T x U`,
      //
      //   dT/dphi   = +sin(theta) U   so  d loc0 / d phi   = +L sin(theta)
      //   dT/dtheta = -V             so  d loc1 / d theta = -L
      //
      // and the variance and the covariance follow. The `q/p` to position
      // coupling is second order in the sagitta over one crossing and is not
      // taken; a slab of this detector puts p99 0.21 in `t/X0`, so the energy
      // straggling over one crossing is far below the scattering.
      if (leverArm > 0.0) {
        const Acts::Vector3 t = direction(*state);
        const double sinTheta = std::hypot(t.x(), t.y());
        const double a = leverArm * sinTheta;
        const double b = -leverArm;
        auto bump = [&](Acts::BoundIndices i, Acts::BoundIndices j, double v) {
          state->cov(i, j) += v;
          if (i != j) {
            state->cov(j, i) += v;
          }
        };
        bump(Acts::eBoundLoc0, Acts::eBoundLoc0, a * a * effects.variancePhi);
        bump(Acts::eBoundLoc1, Acts::eBoundLoc1, b * b * effects.varianceTheta);
        bump(Acts::eBoundLoc0, Acts::eBoundPhi, a * effects.variancePhi);
        bump(Acts::eBoundLoc1, Acts::eBoundTheta, b * effects.varianceTheta);
      }
    }

    ++state->nMatApplied;
    NavCensus& census = NavCensus::instance();
    NavCensus::bump(census.matApplied);
    // Magnitude guards. The arithmetic above mirrors ACTS line for line, so a
    // difference from stock is a wrong slab rather than a wrong formula, and a
    // wrong slab is invisible in a count of applications. `cpp/plan_probe.cpp`
    // has the honest distribution to compare against.
    if (slab.thicknessInX0() > 1.0) {
      NavCensus::bump(census.matThick);
    }
    if (effects.varianceTheta > 1e-4) {
      NavCensus::bump(census.matBigTheta);
    }
    if (nextP <= kMinP) {
      NavCensus::bump(census.matFloored);
    }
  }

 public:

 private:
  /// Is this a surface the kernel was actually trained to jump to?
  ///
  /// `isSensitive()` alone, which is now both correct and non-empty. THE ODD
  /// CENSUS over the real TrackingGeometry:
  ///
  ///                  sensitive   insensitive
  ///       Cylinder           0           190
  ///       Disc               0           250
  ///       Plane         18,824             0
  ///
  /// Every measurement surface is a planar module and every cylinder and disc
  /// is a portal or a layer-approach surface. While the teacher's target was
  /// the cylinder r = r1, the set of surfaces it described and the set the
  /// detector measures on did not intersect at all, and the sensitivity guard
  /// made the jump fire nowhere. The teacher's destination is now the module
  /// plane, so the two sets coincide.
  ///
  /// It is the same predicate the CKF's own actor uses to decide whether to
  /// expect measurements (CombinatorialKalmanFilter.hpp:456).
  bool accepts(const Acts::Surface& surface) const {
    return surface.isSensitive();
  }

  /// \|eta\| of a unit direction, on the performance writer's own variable.
  ///
  /// `atanh(cos theta)` rather than `-log(tan(theta/2))`: the two agree, and
  /// this one has no pole to guard at theta = 0 beyond the clamp, which is
  /// reached by a track along the beam that the navigator can still step.
  static double absEta(const Acts::Vector3& d) {
    const double c = std::clamp(d.z() / d.norm(), -0.999999, 0.999999);
    return std::abs(std::atanh(c));
  }

  /// Add one step's length to whatever the census is currently accumulating.
  ///
  /// A step aimed at a censused destination goes to that destination's own
  /// total; everything else goes to the portal and layer-approach accumulator,
  /// which is the part every arm pays identically.
  static void creditPath(State* state, bool aimed, double h) {
    auto& census = TransportCensus::instance();
    if (!census.on()) {
      return;
    }
    const double mm = std::abs(h);
    if (aimed) {
      state->censusPath += mm;
    } else {
      census.recordOtherPath(mm);
    }
  }

  /// Hand a finished destination's path to the census, once.
  static void retirePath(State* state) {
    if (!state->censusPathOpen) {
      return;
    }
    TransportCensus::instance().recordPath(
        state->censusPathSlot, state->censusEta, state->censusPath);
    state->censusPathOpen = false;
    state->censusPath = 0.0;
  }

  bool toJump(const State& state, const Acts::Surface& surface,
              Jump* out) const {
    if (!accepts(surface)) {
      return false;
    }
    // A general plane, from the surface's own transform. Not from `center` and
    // a reconstructed normal: a plane's chart is its transform, and taking the
    // axes from it is the only way loc0 and loc1 here mean what they mean
    // inside ACTS and in the module table the teacher was retargeted onto.
    //
    // The geometry context is the stepper state's, not a default-constructed
    // one, so a misaligned detector does not get silently propagated as a
    // nominal one.
    if (surface.type() != Acts::Surface::SurfaceType::Plane) {
      return false;
    }
    const auto& tf = surface.localToGlobalTransform(state.options.geoContext);
    out->c = tf.translation();
    out->e0 = tf.rotation().col(0);
    out->e1 = tf.rotation().col(1);
    out->n = tf.rotation().col(2);
    out->pos = position(state);
    out->mom = direction(state) * absoluteMomentum(state);
    out->q = charge(state) >= 0 ? 1.0 : -1.0;

    // the field feature (input 10, counting from zero) is the map value at the
    // SOURCE point, and the stepper already owns a field cache, so this is one
    // lookup -- against the four per RKN step that the benchmark counts.
    //
    // DIVIDED BY Acts::UnitConstants::T. `getField` answers in ACTS's native
    // units, where 1 T is 0.000299792458 (`Definitions/Units.hpp:173`), so a
    // 2 T solenoid comes back as 5.996e-4. The feature was fitted in tesla:
    // `train_gtheta.py` builds it from `fm.bz(x, y, z)` off the map npz, whose
    // values are around 2, and the model's own `mu` for this input is 1.907
    // with `sd` 0.191. Shipping the raw value put the input 9.99 sd below its
    // training mean on every jump, measured, and it cost 18 to 80 points of
    // tracking efficiency. `src/prop/feature_check.py` is the
    // check that would have caught it, and it is what has to pass before a
    // learned-on number is quoted.
    State& mutableState = const_cast<State&>(state);
    const auto b = m_inner.getField(mutableState, out->pos);
    if (!b.ok()) {
      return false;
    }
    out->bzMap = b->z() / Acts::UnitConstants::T;

    // The same number, used a second way when the switch is on: as the field
    // the helix core itself runs at rather than only as an input to the
    // network and to the gate. `setLocalCore` says what that costs and what it
    // invalidates.
    out->localCore = m_localCore;

    // The scale the position outputs are written in. The constants are the
    // fallback and not the intent: they are what a run with no table gets, and
    // what every run before this one got. `setCellSigmaTable` says why.
    //
    // pT and |eta| off the jump's own momentum rather than the seed's, because
    // that is what `chi2_gate.measured_cov` binned on: the table's `pt` column
    // is the track's pT at the source of the jump.
    const double jumpPt = std::hypot(out->mom.x(), out->mom.y());
    const double jumpEta = absEta(direction(state));

    out->sig0 = m_sig0;
    out->sig1 = m_sig1;
    if (m_cellSigmaTable != nullptr) {
      CellSigmaTable::Sigma s{};
      if (m_cellSigmaTable->lookup(surface.geometryId().volume(), jumpPt,
                                   jumpEta, &s)) {
        out->sig0 = s.s0;
        out->sig1 = s.s1;
      }
    }

    // The scale outputs 2 to 4 are applied with, from the header's own table.
    // Compiled in rather than loaded, unlike the sigma above, because it is a
    // property of the FIT: the model was trained on the target divided by these
    // numbers, so a model and a scale from different runs is not a model. The
    // `Jump` default is the global triple and is what a volume `classOf` does
    // not know keeps.
    ModuleClass v1cls{};
    if (classOf(surface.geometryId().volume(), &v1cls)) {
      const std::size_t k =
          ((static_cast<std::size_t>(v1cls) * kNPt + ptBin(jumpPt)) * kNEta +
           etaBin(jumpEta)) * 3;
      out->v1s[0] = w::v1_scale_cell[k];
      out->v1s[1] = w::v1_scale_cell[k + 1];
      out->v1s[2] = w::v1_scale_cell[k + 2];
    }
    return true;
  }

  /// Put the learned answer back into the state. Returns the 3D path length,
  /// which is what `step()` owes its caller.
  ///
  /// A step owes the state four things and every one of them is silent when
  /// wrong, so each is written out here rather than left to whoever reads the
  /// diff.
  ///
  ///   `jacTransport` is the covariance transport. If a 20 cm jump leaves it
  ///   alone, the covariance is transported as though nothing had moved: S
  ///   comes out too small, the chi2 gate too tight and the gain too small,
  ///   with no crash and no warning. `JumpJacobian::D` is the jump's own. It
  ///   LEFT-multiplies, because jacTransport accumulates forward from the last
  ///   surface and the new step composes on the outside.
  ///
  ///   `derivative` is d(free)/d(path) at the new point. ACTS does not use it
  ///   for the transport itself; it uses it to build the surface constraint
  ///   that `JumpJacobian::D` deliberately omits. Leaving it stale silently
  ///   reapplies the previous surface's constraint to this one, and is as
  ///   invisible as a frame mix-up. See LearnedJacobian.hpp.
  ///
  ///   `pathAccumulated` is the 3D arc length, not the transverse one. ACTS
  ///   adds the RKN step length at `EigenStepper.ipp:368` and everything
  ///   downstream reads it as 3D. The two differ by pt/|p|, which is 0.16 at
  ///   |eta| = 2.5, so the wrong one is a factor six on the quantity path
  ///   limits and `boundState` both use.
  ///
  ///   The free time. ACTS advances it by `h * dtds` with
  ///   dtds = sqrt(1 + m^2/p^2) (`EigenStepperDefaultExtension.hpp:121`), and
  ///   that product is `JumpJacobian::dt`. Without it the particle moves 20 cm
  ///   and its clock does not tick.
  ///
  /// The covariance itself is still not touched, on purpose. `jacTransport` is
  /// the transport; C = F C F' + Q is applied by `transportCovarianceToBound`,
  /// which is also the seam the measured Q goes in at.
  double writeBack(State* state, const Jump& jump, const Prediction& p) const {
    const JumpJacobian jj =
        jumpJacobian(jump, p.mom, p.s, particleHypothesis(*state).mass());

    // The path and the clock advance on every step. Only the transport is
    // conditional, and on the same flag ACTS uses (`EigenStepper.ipp:337`,
    // `:362`): with covTransport off, jacTransport and derivative are not
    // maintained by anyone, and writing them would be a claim about state the
    // propagator has been told not to keep.
    if (state->covTransport) {
      state->jacTransport = jj.D * state->jacTransport;
      state->derivative = jj.t;
    }
    state->pathAccumulated += jj.path3d;
    state->pars[Acts::eFreeTime] += jj.dt;

    const double pAbs = p.mom.norm();
    state->pars.template segment<3>(Acts::eFreePos0) = p.pos;
    state->pars.template segment<3>(Acts::eFreeDir0) = p.mom / pAbs;
    state->pars[Acts::eFreeQOverP] = (charge(*state) >= 0 ? 1.0 : -1.0) / pAbs;
    return jj.path3d;
  }

  /// Add what this jump owes the covariance to the pending debt.
  ///
  /// The bin is taken from the jump's SOURCE momentum. pt and |eta| are what
  /// `qtable.py` binned by, they are properties of the track rather than of an
  /// endpoint, and a helix conserves both, so source and destination give the
  /// same cell. It is evaluated here rather than at the transport because by
  /// then `state.pars` has moved and the sigma head belongs to a later jump.
  ///
  /// A volume the table does not know adds NOTHING. There is no sane default
  /// for a module class, and someone else's noise is worse than none: it would
  /// be a number with no measurement behind it, in a covariance that everything
  /// downstream treats as measured.
  ///
  /// `helixAlone` is the field gate's own decision for this jump, taken twenty
  /// lines earlier and passed rather than recomputed. It selects the branch of
  /// the table. The two branches are two different transports and the error of
  /// one is not a measurement of the other; a one-branch table ignores it and
  /// arms the same Q on both, which is what every table before this one did.
  void armNoise(State* state, const Acts::Surface& surface, const Jump& jump,
                bool helixAlone) const {
    if (m_q == nullptr || !state->covTransport) {
      return;
    }
    ModuleClass cls;
    if (!classOf(surface.geometryId().volume(), &cls)) {
      return;
    }
    const double pt = std::hypot(jump.mom.x(), jump.mom.y());
    // |eta| = atanh(|pz|/|p|), clamped. The last bin edge is 3.0 and the ODD
    // stops well before it, so the clamp guards a division rather than making a
    // physics choice.
    const double c =
        std::min(std::abs(jump.mom.z()) / jump.mom.norm(), 1.0 - 1e-12);
    Mat55 q;
    if (!m_q->lookup(cls, pt, std::atanh(c), state->sigmaScale,
                     helixAlone ? Branch::kHelixAlone : Branch::kFired, &q)) {
      return;
    }
    state->qPending += q;
    ++state->nQArmed;
  }

  Inner m_inner;
  const NoiseTable* m_q = nullptr;
  /// What the CKF asks for. `CombinatorialKalmanFilter::Config` defaults both
  /// true and `TrackFindingAlgorithmFunction.cpp` does not change them, so the
  /// seam applies the same two terms the stock filter applies.
  bool m_matScattering = true;
  bool m_matEnergyLoss = true;
  bool m_useSeamMaterial = true;
  bool m_useLearned = false;
  bool m_useSigmaHead = true;
  bool m_useNetwork = true;
  bool m_plannedOnly = false;
  double m_fieldGate = 0.0;
  bool m_localCore = false;
  double m_sig0 = 1.0;
  double m_sig1 = 1.0;
  const CellSigmaTable* m_cellSigmaTable = nullptr;
  bool m_publishTrack = true;
};

// The header's v1 scale table and this build's binning have to be the same
// binning or `toJump` indexes a different table than the one that was fitted,
// silently and with plausible numbers. A header from a different binning fails
// to compile instead.
static_assert(w::kV1Class == kNClass && w::kV1Pt == kNPt &&
                  w::kV1Eta == kNEta,
              "gtheta_weights.hpp was exported against a different "
              "(class, pT, |eta|) binning than LearnedNoise.hpp uses. "
              "Re-export with src/prop/export_kernel.py.");

static_assert(Acts::StepperConcept<LearnedStepper>,
              "LearnedStepper must remain substitutable for EigenStepper, or "
              "Propagator<LearnedStepper, Navigator> will not instantiate");

}  // namespace collider_ml
