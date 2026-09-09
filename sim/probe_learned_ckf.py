"""The cross-module seam, tested end to end without an event.

Three things have to hold before the learned CKF driver is worth running, and
each fails in its own way if the ABI story is wrong:

  1. `import learned_ckf` next to `acts.examples` -- same pybind11 internals,
     or the import itself dies.
  2. `make_track_finder(...)` returns an object Python sees as the SAME type
     the stock bindings registered (`TrackFindingAlgorithm.TrackFinderFunction`)
     -- typeid unification across the two .so, or the return raises
     "unregistered type".
  3. The stock `TrackFindingAlgorithm` constructor ACCEPTS it as `findTracks`
     -- the round trip back into C++, which is the seam the driver uses.

Runs INSIDE the image:  sim/run_in_odd.sh sim/probe_learned_ckf.py
"""
import sys

import acts
import acts.examples
from acts.examples.odd import getOpenDataDetector, getOpenDataDetectorDirectory

sys.path.insert(0, "/data/cpp")
import learned_ckf  # noqa: E402  (after sys.path)

print("1. imported learned_ckf next to acts.examples")

geoDir = getOpenDataDetectorDirectory()
deco = acts.IMaterialDecorator.fromFile(
    geoDir / "data/odd-material-maps.root", level=acts.logging.WARNING)
detector = getOpenDataDetector(odd_dir=geoDir, materialDecorator=deco)
trackingGeometry = detector.trackingGeometry()
field = detector.field

f = learned_ckf.make_track_finder(
    trackingGeometry, field, int(acts.logging.WARNING), learned=False)
stock_t = acts.examples.TrackFindingAlgorithm.TrackFinderFunction
print(f"2. factory returned {type(f).__module__}.{type(f).__qualname__}; "
      f"isinstance(stock TrackFinderFunction) = {isinstance(f, stock_t)}")

alg = acts.examples.TrackFindingAlgorithm(
    level=acts.logging.WARNING,
    measurementSelectorCfg=acts.MeasurementSelector.Config(
        [(acts.GeometryIdentifier(), ([], [15.0], [25.0], [1]))]),
    inputMeasurements="measurement_subset",
    inputInitialTrackParameters="estimatedparameters",
    inputSeeds="",
    outputTracks="ckf_tracks",
    findTracks=f,
    trackingGeometry=trackingGeometry,
    magneticField=field,
)
print("3. stock TrackFindingAlgorithm accepted the learned findTracks")

q = learned_ckf.make_track_finder(
    trackingGeometry, field, int(acts.logging.WARNING), learned=True,
    qtable="/data/cpp/q_table.bin")
print("4. learned=True with the measured material-off q_table.bin loads")

print("\nseam OK on all four counts")
