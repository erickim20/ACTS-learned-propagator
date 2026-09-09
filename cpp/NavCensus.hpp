#pragma once

/// What the moved seam did, counted, and printed whether or not anyone asked.
///
/// The trap for this job: a run that reproduces `stock` is the expected answer
/// both when the seam works and when the seam never fired, and nothing in the
/// performance tables tells those apart. A bit-identical run has already been
/// read once as a working switch when it was a surface-type mismatch and the
/// jump fired nowhere.
///
/// So these are unconditional. There is no environment variable and no off
/// switch: a run whose log carries no `[navcensus]` block was not taken with
/// this build, and a run whose `offered` is zero did not consult the seam.
///
/// The counters narrow one stage at a time, in the shape
/// `cpp/LearnedStepper.hpp` uses:
///
///   wrapped     nextTarget calls that reached this navigator at all. Nonzero
///               proves the wrapper is in the propagator's loop, which is the
///               whole content of Gate 1 and cannot be shown by output.
///   offered     legs where the seam was consulted, meaning the track was
///               standing on a sensitive surface with an associated layer.
///   answered    ... that the walk named a module for.
///   returned    ... that were handed to the propagator as a target.
///   completed   ... the stepper actually arrived on.
///   fellback    ... re-asked without an arrival, so the target was dropped and
///               the leg went back to the stock navigator.
///   resync      inner-state resynchronisations, and how many failed.
///   matplanned  material surfaces the walk put on a leg's plan.
///   matapplied  ... the stepper actually applied.
///
/// `matplanned` and `matapplied` are counted in two different translation units
/// on two different state objects and are expected to agree. They are printed
/// side by side because the failure that matters, a plan the stepper never
/// consumed, moves them apart and moves nothing else.
///
/// Atomic and relaxed, because ActsExamples runs the finder on several threads
/// and the numbers are totals rather than an ordering.

#include <atomic>
#include <cstdint>
#include <cstdio>

namespace collider_ml {

class NavCensus {
 public:
  static NavCensus& instance() {
    static NavCensus c;
    return c;
  }

  using Counter = std::atomic<std::uint64_t>;

  Counter wrapped{0};
  Counter offered{0};
  Counter answered{0};
  Counter returned{0};
  Counter completed{0};
  Counter fellback{0};
  Counter walkFailed{0};
  Counter sameLayer{0};
  /// The same three, restricted to a propagation running backwards.
  ///
  /// The CKF's second pass is one (`TrackFindingAlgorithm.cpp:359`,
  /// `digi_and_reco.py:409`), and it is the only place the seam's helix is
  /// asked to retrace rather than to advance. Pooling the two passes hides a
  /// sign error in the curvature: the backward legs are a minority and a
  /// fallback rate that doubles on them moves the total by less than the bin
  /// to bin spread. Split, it is the measurement that distinguishes the two
  /// signs on the real sample.
  Counter offeredBwd{0};
  Counter returnedBwd{0};
  Counter fellbackBwd{0};
  /// The learned transport, narrowed the same way.
  ///
  /// The same trap on the other side of the seam: an arm that
  /// reproduces another one has either a transparent transport or a transport
  /// that never fired, and the tables cannot tell those apart.
  /// `LearnedStepper::State` has counted these since it was written and no
  /// aggregate ever read them, so a learned-on run reported nothing about
  /// whether the kernel ran.
  ///
  ///   latched   destinations `updateSurfaceStatus` handed the stepper
  ///   accepted  ... that `accepts()` said yes to, which is `isSensitive()`
  ///   built     ... that `toJump` turned into a Jump
  ///   fired     ... the kernel answered and the state was written back
  ///   unplanned ... of `accepted`, those whose destination the walk never
  ///             named for this transport. Counted whether or not they are
  ///             suppressed, so it reads the same on both arms of part A.
  Counter jumpLatched{0};
  Counter jumpAccepted{0};
  Counter jumpBuilt{0};
  Counter jumpFired{0};
  Counter jumpUnplanned{0};
  /// ... of `built`, those the field gate took the network off, so the helix
  /// answered alone. Zero unless the gate is set.
  Counter jumpNearField{0};
  Counter resync{0};
  Counter resyncFailed{0};
  Counter matPlanned{0};
  Counter matApplied{0};
  Counter matVacuum{0};
  /// Magnitude guards on what the seam actually applied. A slab whose
  /// thickness or whose scattering angle is far outside what this detector
  /// holds is a wrong evaluation point or a wrong mode, and neither shows up
  /// in a count of applications. `cpp/plan_probe.cpp` has the honest
  /// distribution: t/X0 is p50 0.018 and max 0.86 over every material surface
  /// a leg crosses, so anything above 1 is not this detector.
  Counter matThick{0};     ///< t/X0 > 1
  Counter matBigTheta{0};  ///< variance in theta > 1e-4 rad^2
  Counter matFloored{0};   ///< the 10 MeV momentum floor was hit

  static void bump(Counter& c, std::uint64_t n = 1) {
    c.fetch_add(n, std::memory_order_relaxed);
  }

  ~NavCensus() { dump(); }

  void dump() const {
    // Printed even when every counter is zero. A zero block says the build was
    // this one and the seam did not fire; no block at all says the build was
    // something else. Those are different facts and a conditional print would
    // make them the same one.
    std::fprintf(stderr,
                 "[navcensus] wrapped=%llu offered=%llu answered=%llu "
                 "returned=%llu completed=%llu fellback=%llu\n"
                 "[navcensus] walk_failed=%llu same_layer=%llu resync=%llu "
                 "resync_failed=%llu\n"
                 "[navcensus] backward offered=%llu returned=%llu "
                 "fellback=%llu\n"
                 "[navcensus] jump_latched=%llu jump_accepted=%llu "
                 "jump_built=%llu jump_fired=%llu jump_unplanned=%llu\n"
                 "[navcensus] jump_nearfield=%llu\n"
                 "[navcensus] mat_planned=%llu mat_applied=%llu mat_vacuum=%llu "
                 "mat_thick=%llu mat_bigtheta=%llu mat_floored=%llu\n",
                 u(wrapped), u(offered), u(answered), u(returned), u(completed),
                 u(fellback), u(walkFailed), u(sameLayer), u(resync),
                 u(resyncFailed), u(offeredBwd), u(returnedBwd), u(fellbackBwd),
                 u(jumpLatched), u(jumpAccepted), u(jumpBuilt), u(jumpFired),
                 u(jumpUnplanned), u(jumpNearField),
                 u(matPlanned), u(matApplied), u(matVacuum), u(matThick),
                 u(matBigTheta), u(matFloored));
    std::fflush(stderr);
  }

 private:
  static unsigned long long u(const Counter& c) {
    return static_cast<unsigned long long>(c.load(std::memory_order_relaxed));
  }

  NavCensus() = default;
};

}  // namespace collider_ml
