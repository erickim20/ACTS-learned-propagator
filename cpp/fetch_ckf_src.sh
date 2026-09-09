#!/usr/bin/env bash
# Fetch the TrackFinding and Navigator sources at the exact commit the image's
# ACTS was built from, into cpp/acts_examples_src/. Not committed (they are
# ACTS's code, MPL-2.0 headers intact); this script is the record of where they
# came from.
#
# `cpp/acts_headers.sh` pulls the installed headers out of the image and is the
# authority when a signature is in question. It needs GNU tar, so it extracts
# nothing on macOS, and it carries no .cpp: the navigator's whole state machine
# is in `Core/src/Navigation/Navigator.cpp` and none of it is in the header.
# This is the route to the implementation, at the commit the spec records.
#
# The commit is not guessed: the image's spack manifest
# (.spack/spec.json in the acts prefix) records
#   version 44.99.99-colliderml-arrow
#   commit  7cba36b173dd73a14342cc65043a43179a4b1dbd
#   from    https://github.com/murnanedaniel/acts.git  branch feat/colliderml-arrow-tracks
# and `cpp/probe_ckf_seam2.sh` prints it. The prefix's ActsVersion.hpp says
# "-dirty"; the spec records no patches, so the dirt is spack's own build-time
# touch-ups. The acceptance test is what checks that nothing semantic drifted.
set -euo pipefail
cd "$(dirname "$0")"

C=7cba36b173dd73a14342cc65043a43179a4b1dbd
B="https://raw.githubusercontent.com/murnanedaniel/acts/$C"
mkdir -p acts_examples_src

fetch() {
  local src=$1 dst=acts_examples_src/$2
  [[ -s "$dst" ]] && { echo "have $dst"; return; }
  curl -sf "$B/$src" -o "$dst"
  echo "fetched $dst"
}

fetch "Examples/Algorithms/TrackFinding/include/ActsExamples/TrackFinding/TrackFindingAlgorithm.hpp" TrackFindingAlgorithm.hpp
fetch "Examples/Algorithms/TrackFinding/src/TrackFindingAlgorithm.cpp" TrackFindingAlgorithm.cpp
fetch "Examples/Algorithms/TrackFinding/src/TrackFindingAlgorithmFunction.cpp" TrackFindingAlgorithmFunction.cpp

# The navigator, which is what decides when a module is offered to the stepper.
# Note the path: the header is under Propagator/ and the
# implementation is under Navigation/, so a guess at Core/src/Propagator/
# returns 404.
fetch "Core/include/Acts/Propagator/Navigator.hpp" Navigator.hpp
fetch "Core/src/Navigation/Navigator.cpp" Navigator.cpp
fetch "Core/include/Acts/Propagator/NavigationTarget.hpp" NavigationTarget.hpp
fetch "Core/include/Acts/Geometry/Layer.hpp" Layer.hpp
fetch "Core/src/Geometry/TrackingGeometry.cpp" TrackingGeometry.cpp
fetch "Plugins/DD4hep/src/ConvertDD4hepDetector.cpp" ConvertDD4hepDetector.cpp

# Where a module becomes a candidate, and where it stops being one. Layer.cpp
# takes the candidates from `m_surfaceArray->neighbors` at :206, a binned subset
# of the layer, and then applies `options.boundaryTolerance` to each at :169.
# SurfaceArray.cpp is that binned lookup: which bin the straight ray lands in,
# how far the neighbourhood reaches, and the pointer-order sort in
# `populateNeighborCache` that makes the returned order depend on the heap.
# TrackingVolume.cpp:443 is `compatibleLayers`, which builds its targets out of
# `Layer::surfaceOnApproach`, so a layer target's `surface()` is the approach
# surface. Intersection.cpp carries `checkPathLength`, which is where a negative
# `nearLimit` does or does not mean what it looks like.
fetch "Core/src/Geometry/Layer.cpp" Layer.cpp
fetch "Core/src/Surfaces/SurfaceArray.cpp" SurfaceArray.cpp
fetch "Core/src/Geometry/TrackingVolume.cpp" TrackingVolume.cpp
fetch "Core/src/Utilities/Intersection.cpp" Intersection.cpp

# Where the seam would sit, and what it would owe. The two concepts are the
# constraints on `Propagator.hpp:87-88`; everything else is detail on top of
# them, and both are headers, so `cpp/acts_headers.sh` is the authority for
# them and for `Propagator.ipp`, `CombinatorialKalmanFilter.hpp` and
# `PointwiseMaterialInteraction.hpp`, which are header-only in this build.
#
# These three are the parts that are not in a header. `evaluateMaterialSlab` and
# `computeMaterialEffects` are what turns an `ISurfaceMaterial` into a slab and
# a slab into scattering and energy loss, and they decide whether a transport
# that flies past a surface can still have that surface's material applied.
# `updateSingleSurfaceStatus`, which is what makes a reported navigation target
# a step-size constraint, is not here: it is defined inline in
# `Acts/Propagator/detail/SteppingHelper.hpp` and there is no
# `Core/src/Propagator/detail/SteppingHelper.cpp` at this commit. The two
# engines are the covariance transport a propagator seam would have to
# reproduce.
#
# NOTE the paths. The headers are under `Core/include/Acts/Propagator/detail/`
# and the sources under `Core/src/Propagator/detail/`, which mirror here, but
# `Navigator.hpp` and `Navigator.cpp` do not. Probe before assuming.
fetch "Core/src/Propagator/detail/PointwiseMaterialInteraction.cpp" PointwiseMaterialInteraction.cpp
fetch "Core/src/Propagator/detail/CovarianceEngine.cpp" CovarianceEngine.cpp
fetch "Core/src/Propagator/detail/JacobianEngine.cpp" JacobianEngine.cpp

# What reads a track after track finding. The live question is whether a
# material-only track state is visible to any of them, and the answer has to
# come from the code that runs rather than from the Core interfaces it
# implements. `GreedyAmbiguityResolution.cpp` is the greedy solver
# `digi_and_reco.py` selects by default; `RootTrackFinderPerformanceWriter.cpp`
# with `EffPlotTool.cpp` and `FakePlotTool.cpp` produce the efficiency and fake
# ratio that is reported; `RootTrackSummaryWriter.cpp` writes the
# `nStates` branch; `TrackTruthMatcher.cpp` decides which
# tracks are matched and which are fake.
#
# NOTE the Examples layout. Headers and sources sit side by side under
# `Examples/Algorithms/<X>/ActsExamples/<X>/` for TruthTracking, and split into
# `ActsExamples/` and `src/` for AmbiguityResolution and Io/Root. Both were
# probed; a guess at one layout returns 404 for the other.
fetch "Core/src/AmbiguityResolution/GreedyAmbiguityResolution.cpp" GreedyAmbiguityResolution.cpp
fetch "Examples/Algorithms/AmbiguityResolution/src/GreedyAmbiguityResolutionAlgorithm.cpp" GreedyAmbiguityResolutionAlgorithm.cpp
fetch "Examples/Algorithms/TruthTracking/ActsExamples/TruthTracking/TrackTruthMatcher.cpp" TrackTruthMatcher.cpp
fetch "Examples/Io/Root/src/RootTrackFinderPerformanceWriter.cpp" RootTrackFinderPerformanceWriter.cpp
fetch "Examples/Io/Root/src/RootTrackSummaryWriter.cpp" RootTrackSummaryWriter.cpp
fetch "Examples/Framework/src/Validation/EffPlotTool.cpp" EffPlotTool.cpp
fetch "Examples/Framework/src/Validation/FakePlotTool.cpp" FakePlotTool.cpp
fetch "Examples/Framework/src/Validation/TrackClassification.cpp" TrackClassification.cpp
fetch "Examples/Framework/src/Validation/TrackFinderPerformanceCollector.cpp" TrackFinderPerformanceCollector.cpp
fetch "Examples/Framework/src/Validation/TrackSummaryPlotTool.cpp" TrackSummaryPlotTool.cpp
fetch "Examples/Framework/src/Validation/TrackQualityPlotTool.cpp" TrackQualityPlotTool.cpp

echo "$C" > acts_examples_src/COMMIT
