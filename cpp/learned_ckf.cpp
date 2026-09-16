// A TrackFinderFunction over Propagator<LearnedStepper, Navigator>, exposed to
// Python, so `digi_and_reco.py`'s own reconstruction chain can run the CKF on
// the learned stepper by swapping one factory and changing nothing else.
//
// The image cannot rebuild ActsExamples (no spack driver, no source tree), and
// it does not need
// to: `TrackFindingAlgorithm::Config::findTracks` is a type-erased
// `std::shared_ptr<TrackFinderFunction>` that `addCKFTracks` fills from Python
// (`reconstruction.py:1708`), so a second factory in a second module is a
// complete integration. The stock arm stays the prebuilt
// `makeTrackFinderFunction` in `libActsExamplesTrackFinding.so`, which is what
// makes the acceptance test reproduce by construction.
//
// `acts_examples_src/TrackFindingAlgorithm.hpp` is the class declaration at
// the exact commit the image was built from (7cba36b, branch
// feat/colliderml-arrow-tracks of murnanedaniel/acts; recorded by
// `.spack/spec.json` in the image and fetched by the build script). Only the
// abstract `TrackFinderFunction` interface and the `Config` layout are taken
// from it; nothing from `libActsExamplesTrackFinding` is linked.
//
// Cross-module pybind11 requirements, each checked before this was written:
// the image's bindings carry internals key
// `__pybind11_internals_v11_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1__`,
// which is what spack's py-pybind11 3.0.0 under the image's g++ 13.3 produces,
// so a module built in-container with those headers shares the type registry.
// No -march=native: the image's libActsCore is built for a generic baseline,
// and mixing the two crashes in a way that looks like an ACTS bug.

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Acts/MagneticField/MagneticFieldProvider.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/TrackFinding/CombinatorialKalmanFilter.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "ActsExamples/EventData/Track.hpp"

#include "acts_examples_src/TrackFindingAlgorithm.hpp"
#include "NewNavigator.hpp"
#include "CellSigma.hpp"
#include "LearnedStepper.hpp"

namespace py = pybind11;

namespace {

using Stepper = collider_ml::LearnedStepper;
using TrackFinderFunction =
    ActsExamples::TrackFindingAlgorithm::TrackFinderFunction;

/// The two navigators this module can put behind the same stepper.
///
/// `Acts::Navigator` is the stock one and is what every measurement before the
/// moved seam was taken with. `collider_ml::NewNavigator` is the
/// moved seam. They are separate `Propagator` instantiations because
/// `Propagator` is templated on the navigator (`Propagator.hpp:87-88`), which is
/// the whole reason no ACTS source is patched; `TrackFinderFunction` is
/// type-erased, so both reach Python through one factory.
template <typename navigator_t>
using PropagatorFor = Acts::Propagator<Stepper, navigator_t>;
template <typename navigator_t>
using CKFFor = Acts::CombinatorialKalmanFilter<PropagatorFor<navigator_t>,
                                               ActsExamples::TrackContainer>;

// Same shape as the stock TrackFinderFunctionImpl in
// TrackFindingAlgorithmFunction.cpp, plus ownership of the noise table:
// `setNoiseTable` stores a raw pointer, so the table has to live exactly as
// long as the CKF that reads it.
template <typename navigator_t>
struct LearnedTrackFinderFunctionImpl : public TrackFinderFunction {
  std::unique_ptr<collider_ml::NoiseTable> qtab;
  std::unique_ptr<collider_ml::CellSigmaTable> sigtab;
  CKFFor<navigator_t> trackFinder;

  LearnedTrackFinderFunctionImpl(
      std::unique_ptr<collider_ml::NoiseTable> q,
      std::unique_ptr<collider_ml::CellSigmaTable> s, CKFFor<navigator_t>&& f)
      : qtab(std::move(q)), sigtab(std::move(s)), trackFinder(std::move(f)) {}

  ActsExamples::TrackFindingAlgorithm::TrackFinderResult operator()(
      const ActsExamples::TrackParameters& initialParameters,
      const ActsExamples::TrackFindingAlgorithm::TrackFinderOptions& options,
      ActsExamples::TrackContainer& tracks,
      ActsExamples::TrackProxy rootBranch) const override {
    return trackFinder.findTracks(initialParameters, options, tracks,
                                  rootBranch);
  }
};

// The md5 of the weights header this extension was compiled from, put here by
// `cpp/incontainer_build_ckf.sh`. Empty is a build that did not record it, and
// `NoiseTable::load` refuses a measured table on such a build rather than
// arming a covariance nothing can check.
#ifndef GTHETA_WEIGHTS_MD5
#define GTHETA_WEIGHTS_MD5 ""
#endif
const std::string kWeightsMd5 = GTHETA_WEIGHTS_MD5;

std::shared_ptr<TrackFinderFunction> makeLearnedTrackFinderFunction(
    std::shared_ptr<const Acts::TrackingGeometry> trackingGeometry,
    std::shared_ptr<const Acts::MagneticFieldProvider> magneticField,
    int level, bool learned, bool sigmaHead, const std::string& qtablePath,
    const std::string& cellSigmaPath,
    double cellSig0, double cellSig1, bool network, bool resolveMaterial,
    bool navSeam, bool navWalk, bool navMaterial, bool seamMaterial,
    bool plannedOnly, double fieldGate, bool publishTrack, bool localCore,
    double sigmaConst, double sigmaHelix) {
  auto logger = Acts::getDefaultLogger(
      "LearnedCKF", static_cast<Acts::Logging::Level>(level));

  Stepper stepper(std::move(magneticField));
  stepper.setCellSigma(cellSig0, cellSig1);
  stepper.setUseLearned(learned);
  stepper.setUseSigmaHead(sigmaHead);
  stepper.setUseNetwork(network);
  stepper.setUseSeamMaterial(seamMaterial);
  stepper.setPlannedOnly(plannedOnly);
  stepper.setFieldGate(fieldGate);
  // One scale per branch on the declared covariance. One is off for both,
  // and off is the identity.
  stepper.setSigmaConst(sigmaConst);
  stepper.setSigmaHelix(sigmaHelix);
  // The three runtime sites, behind one switch. False is
  // the fixed 2 T core every arm on disk was taken on; true reads the map
  // at each jump's source, which `toJump` already has in hand.
  stepper.setLocalCore(localCore);
  // Part B's split. Only a comparison while the walk is off: with the
  // walk on, `NewNavigator` would read a channel nothing writes to and
  // fly a helix at a stale momentum and a stale field, which is not a
  // faster arm but a different and wrong one.
  if (!publishTrack && navSeam && navWalk) {
    throw std::runtime_error(
        "publishTrack=False with the walk on: NewNavigator reads what "
        "publishTrack writes. Use it with navWalk=False, which is the "
        "arm Part B priced the wrapper in.");
  }
  stepper.setPublishTrack(publishTrack);

  std::unique_ptr<collider_ml::NoiseTable> qtab;
  if (!qtablePath.empty()) {
    qtab = std::make_unique<collider_ml::NoiseTable>();
    if (!qtab->load(qtablePath, kWeightsMd5, fieldGate)) {
      // load() already said why on stderr. Refuse to run half-configured: a
      // silent no-noise arm would be indistinguishable from a measured one.
      throw std::runtime_error("NoiseTable refused " + qtablePath);
    }
    stepper.setNoiseTable(qtab.get());
  }

  // The scale the position outputs were trained in. Empty keeps `cellSig0` and
  // `cellSig1`, which is what every run before this one used and is the arm
  // this is measured against. Refusing a table that will not load, rather than
  // falling back quietly, for the same reason as the Q table: an arm that
  // silently used the constants would be indistinguishable from the baseline.
  std::unique_ptr<collider_ml::CellSigmaTable> sigtab;
  if (!cellSigmaPath.empty()) {
    sigtab = std::make_unique<collider_ml::CellSigmaTable>();
    if (!sigtab->load(cellSigmaPath)) {
      throw std::runtime_error("CellSigmaTable refused " + cellSigmaPath);
    }
    stepper.setCellSigmaTable(sigtab.get());
  }

  // Identical to the stock factory from here down, except that
  // `resolveMaterial` is settable. `true` is the stock value and the value
  // every measurement before the moved seam was taken at.
  // `Acts::Navigator::Config` is what both navigators take:
  // `NewNavigator::Config` is an alias for it,
  // because the moved seam changes which surface is reported and not which
  // surfaces the geometry resolves.
  Acts::Navigator::Config cfg{std::move(trackingGeometry)};
  cfg.resolvePassive = false;
  cfg.resolveMaterial = resolveMaterial;
  cfg.resolveSensitive = true;
  // Printed unconditionally rather than logged. The run's logging level is
  // whatever `addCKFTracks` passes and an INFO line can be below it, so a
  // silent factory would leave no record in the log of which arm ran. This is
  // the line that says which navigator this run put behind the stepper.
  std::printf(
      "[learned_ckf] resolveMaterial=%s navSeam=%s plannedOnly=%s "
      "fieldGate=%.3f cellSigma=%s publishTrack=%s coreBz=%s\n",
      resolveMaterial ? "true" : "false", navSeam ? "true" : "false",
      plannedOnly ? "true" : "false", fieldGate,
      cellSigmaPath.empty() ? "constants" : cellSigmaPath.c_str(),
      publishTrack ? "true" : "false", localCore ? "source" : "fixed");
  std::printf("[learned_ckf] sigmaConst=%.6g sigmaHelix=%.6g\n",
              sigmaConst, sigmaHelix);
  // The provenance this run and this table agreed on, in the run's own log.
  const std::string prov =
      qtab == nullptr ? std::string("none")
                      : (qtab->weightsMd5() + " measured at gate " +
                         std::to_string(qtab->fieldGate()));
  std::printf("[learned_ckf] weights %s  qtable %s\n",
              kWeightsMd5.empty() ? "NOT RECORDED BY THIS BUILD"
                                  : kWeightsMd5.c_str(),
              prov.c_str());
  std::fflush(stdout);

  // The one type substitution this whole job is. Both arms are built the same
  // way from the same config and the same stepper; only the navigator's type
  // differs, and `TrackFinderFunction` erases it again on the way out.
  if (!navSeam && navWalk) {
    // The walk lives on the navigator, so asking for it without the navigator
    // is a request this factory cannot honour. Refusing is the point: a run
    // that accepted the flag and ignored it would be labelled as an arm it is
    // not, which is the same rule reco_ckf.py applies to
    // --no-resolve-material.
    throw std::runtime_error("navWalk=True needs navSeam=True");
  }

  if (navSeam) {
    using Nav = collider_ml::NewNavigator;
    Nav navigator(cfg, logger->cloneWithSuffix("Navigator"));
    navigator.setUseWalk(navWalk);
    navigator.setUseMaterial(navMaterial);
    PropagatorFor<Nav> propagator(std::move(stepper), std::move(navigator),
                                  logger->cloneWithSuffix("Propagator"));
    CKFFor<Nav> trackFinder(std::move(propagator),
                            logger->cloneWithSuffix("Finder"));
    return std::make_shared<LearnedTrackFinderFunctionImpl<Nav>>(
        std::move(qtab), std::move(sigtab), std::move(trackFinder));
  }

  using Nav = Acts::Navigator;
  Nav navigator(cfg, logger->cloneWithSuffix("Navigator"));
  PropagatorFor<Nav> propagator(std::move(stepper), std::move(navigator),
                                logger->cloneWithSuffix("Propagator"));
  CKFFor<Nav> trackFinder(std::move(propagator),
                          logger->cloneWithSuffix("Finder"));
  return std::make_shared<LearnedTrackFinderFunctionImpl<Nav>>(
      std::move(qtab), std::move(sigtab), std::move(trackFinder));
}

}  // namespace

PYBIND11_MODULE(learned_ckf, m) {
  m.doc() =
      "TrackFinderFunction over Propagator<LearnedStepper, Navigator>. "
      "Drop-in for TrackFindingAlgorithm.makeTrackFinderFunction.";

  m.def("make_track_finder", &makeLearnedTrackFinderFunction,
        py::arg("trackingGeometry"), py::arg("magneticField"),
        py::arg("level"), py::arg("learned") = true,
        py::arg("sigmaHead") = false, py::arg("qtable") = std::string(),
        py::arg("cellSigma") = std::string(),
        py::arg("cellSig0") = 20.0, py::arg("cellSig1") = 43.0,
        py::arg("network") = true, py::arg("resolveMaterial") = true,
        py::arg("navSeam") = false, py::arg("navWalk") = false,
        py::arg("navMaterial") = true, py::arg("seamMaterial") = true,
        py::arg("plannedOnly") = false, py::arg("fieldGate") = 0.0,
        py::arg("publishTrack") = true, py::arg("localCore") = false,
        py::arg("sigmaConst") = 1.0, py::arg("sigmaHelix") = 1.0,
        "Build the type-erased track finder. `learned=False` is the switch-off "
        "arm: the inner EigenStepper runs and the output is bit-identical to a "
        "stock EigenStepper build. `qtable` is the binary "
        "export_qtable.py writes; empty adds no process noise. "
        "`network=False` is the helix bypass: the learned jump is taken and "
        "the forward pass is skipped, so the corrections are exactly zero and "
        "nothing pays for computing them. "
        "`navSeam=True` puts collider_ml::NewNavigator behind the same "
        "stepper instead of Acts::Navigator: the moved seam, whose "
        "nextTarget names the far module and whose step carries the material "
        "the skipped surfaces would have applied. With the walk off this "
        "wrapper forwards all twelve NavigatorConcept members, bit-identical "
        "to navSeam=False; the [navcensus] block on stderr is what "
        "distinguishes a transparent wrapper from one that never ran. "
        "`sigmaConst` scales the declared covariance on the branch the "
        "network fired on, with the head off, and `sigmaHelix` scales it on "
        "the branch that ran when it did not. Both are 1 by default and 1 is "
        "the identity. "
        "`localCore=True` runs the helix core at the map value at each "
        "jump's source instead of the fixed 2 T of LearnedTransport.hpp's "
        "kBHelix: the three runtime sites, behind one "
        "switch, in the same binary as the fixed core. It adds no field "
        "lookup, because toJump already reads that value for the network's "
        "input 10. It is for an arm with the network OFF: the fit's target "
        "and two of its inputs were written at 2 T. "
        "`resolveMaterial=False` sets Navigator::Config::resolveMaterial "
        "false, which is a proxy for a seam that stops reporting "
        "the material surfaces; read cpp/mat_probe.cpp for how much of the "
        "material channel it actually reaches. The stock SympyStepper arm is "
        "NOT here: use the untouched "
        "TrackFindingAlgorithm.makeTrackFinderFunction for that.");
}
