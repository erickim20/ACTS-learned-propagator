// Does a CombinatorialKalmanFilter built on LearnedStepper compile and link
// against the ACTS that is already installed in the ODD image?
//
// This is the question that decides where the integration work has to happen.
// The image ships an ACTS install but no ACTS source, so anything that needed
// a rebuild of ActsExamples would be a many-hour job on four emulated cores --
// i.e. the Linux box. But CombinatorialKalmanFilter, Propagator and Navigator
// are all templates, so they are header-only, and the only thing that has to
// come out of a library is ActsCore's non-template support code. If this file
// links, integration CORRECTNESS can be done here and only TIMING needs native
// hardware.
//
//   g++ -std=c++20 -I$ACTS/include -I$EIGEN -I$BOOST link_probe.cpp \
//       -L$ACTS/lib -lActsCore -o probe
#include <cstdio>
#include <exception>
#include <memory>

#include "Acts/EventData/VectorMultiTrajectory.hpp"
#include "Acts/EventData/VectorTrackContainer.hpp"
#include "Acts/MagneticField/ConstantBField.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/TrackFinding/CombinatorialKalmanFilter.hpp"

#include "LearnedStepper.hpp"

int main() {
  using Stepper = collider_ml::LearnedStepper;
  using Prop = Acts::Propagator<Stepper, Acts::Navigator>;
  using TC = Acts::TrackContainer<Acts::VectorTrackContainer,
                                  Acts::VectorMultiTrajectory>;
  using CKF = Acts::CombinatorialKalmanFilter<Prop, TC>;

  static_assert(Acts::StepperConcept<Stepper>);

  auto bfield = std::make_shared<Acts::ConstantBField>(Acts::Vector3(0, 0, 2));
  Stepper stepper(bfield);
  stepper.setCellSigma(20.0, 43.0);

  // Report first. Everything above is the question this file exists to answer
  // and all of it is settled by the time main() starts: if the types did not
  // instantiate and ActsCore did not resolve, there would be no binary to run.
  std::printf("LINKED. sizeof(CKF<Propagator<LearnedStepper,Navigator>>) = %zu\n",
              sizeof(CKF));
  std::printf("The learned stepper is substitutable all the way up to the "
              "filter, against the installed ACTS.\n");

  // Now the runtime half. The navigator is left without a geometry on purpose
  // -- building the ODD here would turn a link check into a geometry test --
  // so recent ACTS throws from the Navigator constructor. That throw is the
  // expected end of this probe, not a failure, and it is caught so the probe
  // exits 0; an UNCAUGHT abort here would be indistinguishable from a real one.
  try {
    Prop prop(std::move(stepper), Acts::Navigator({}));
    CKF ckf(std::move(prop));
    std::printf("Constructed CKF with an empty Navigator (no geometry).\n");
  } catch (const std::exception& e) {
    std::printf("Navigator declined an empty geometry, as expected: %s\n",
                e.what());
  }
  return 0;
}
