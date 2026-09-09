"""Bucket a perf profile of the reconstruction into what the time is spent on.

`timing.tsv` divides the event by algorithm and stops there.
This divides the samples by symbol, so that track finding's own 6.5 s per event
can be read as integrator, navigator, surface intersection, track container and
Kalman update rather than as one number.

Four things this has to get right, and every one of them is quiet when wrong.

**The stepper is not named `EigenStepper`.** This build runs the code generated
`SympyStepper`, and the hot symbol is `rk4_vacuum` instantiated on a lambda
inside it. A bucket keyed on the name expected undercounts the integrator badly.

**A template argument is not a caller.**
`Propagator<SympyStepper, VoidNavigator>::propagate` names the stepper and is
not stepper time, so matching on the whole mangled signature charges propagator
glue to the integrator. Symbols are matched on the function name with the
template arguments stripped, and only then on the full signature.

**Not all propagation is track finding.** That `VoidNavigator` propagator lives
in `libActsExamplesIoRoot.so`: it is the track writers extrapolating to the
perigee. On a profile taken with the writers on it is 3.5% of samples, and
`LineSurface::intersect` follows it. Take the core numbers off a writers-off
run, which is what `sim/profile/digitization_noio.yaml` is for.

**A symbol that does not resolve is a hole, not a small bucket.** Those are
reported separately and the accounted fraction is printed with every table.
`--audit` prints them by library rather than by address, because an address
says nothing and the library says everything: on the arms here the hole is
`libc` and `libm`, and `libm`'s share triples between `stock` and `pa_cell`.

**A library rule can match the wrong library by its file name.** This
repository's CKF is a Python extension and its file is
`learned_ckf.cpython-313-x86_64-linux-gnu.so`, so the `cpython` key that was
there for the interpreter's own modules matched it and charged every symbol in
the arm's whole CKF that no symbol rule caught to the bindings. It read as 6.0%
of `new-off` and 9.6% of `pa_cell`. The extension has its own rule now, ahead
of the interpreter's.

**Eigen's kernels have no caller in this profile.** `gebp_kernel`,
`general_matrix_vector_product` and the assignment loops are called from the
integrator, from the learned transport and from ACTS's own matrix work, and
frame pointers are not there, so which one is not recoverable from these
samples. They are their own bucket and are not charged to a caller. On the
arms this repository builds it is the bucket the network's forward pass lands
in, and it moves with the transport.

    python3 sim/profile/bucket_perf.py <dir holding perf_dso_sym.txt> [...]
"""
import argparse
import collections
import re
from pathlib import Path

# Matched against the symbol with template arguments stripped, then against the
# full signature. Order matters, first match wins.
SYMBOL_RULES = [
    # This repository's transport, ahead of the stepper: it replaces the
    # integrator over a leg rather than adding to it, so a rule that let
    # `collider_ml::transport` fall through to the stepper would hide exactly
    # the swap the profile is taken to see.
    ("learned transport", (
        "collider_ml::forward", "collider_ml::transport",
        "collider_ml::jumpJacobian", "LearnedTransport", "LearnedJacobian",
        "LearnedNoise", "CellSigma", "collider_ml::weights",
        "LearnedStepper::toJump", "LearnedStepper::armQ",
    )),
    ("stepper", (
        "rk4_vacuum", "SympyStepper::step", "EigenStepper::step",
        "SympyStepper::transport", "EigenStepper::transport",
        "SympyStepper::boundState", "SympyStepper::curvilinearState",
        "transportCovarianceToBound", "transportCovarianceToCurvilinear",
        "transformFreeToBound", "transformBoundToFree", "boundToFreeJacobian",
        "freeToBoundJacobian", "JacobianEngine", "transportJacobian",
        "SympyStepper::update", "estimateStepSize",
        "TransportJacobianImpl", "reinitializeJacobians", "sympy::boundState",
        "sympy::curvilinearState",
        # This build. `EigenStepper::step` is instantiated in this
        # repository's own translation unit and the wrapper's members around
        # it are the same work: `LearnedStepper` forwards to the inner
        # `EigenStepper` whenever the transport does not fire.
        "EigenStepper::direction", "EigenStepper::State",
        "EigenStepper::update", "EigenStepper::boundState",
        "EigenStepper::curvilinearState", "LearnedStepper::step",
        "LearnedStepper::State", "LearnedStepper::transportCovarianceToBound",
        "LearnedStepper::publishTrack",
        # The RK stages' own cross products. It is in the integrator's bucket
        # because it moves with the integrator: 0.30% of `new-off` against
        # 0.11% of `pa_cell`, where two thirds of the transports no longer
        # step.
        "VectorHelpers::cross",
    )),
    ("field", (
        "fieldComponents", "getField", "getFieldAndGradient", "SolenoidField",
        "InterpolatedBFieldMap", "ConstantBField", "MagneticFieldProvider",
        "DD4hepFieldAdapter",
    )),
    ("navigator", (
        "Navigator::", "updateSingleSurfaceStatus", "compatibleSurfaces",
        "compatibleLayers", "compatibleBoundaries", "resolveSurfaces",
        "resolveLayers", "resolveBoundaries", "NavigationState",
        "TrackingVolume::", "lowestTrackingVolume", "ApproachDescriptor",
        "SurfaceArray", "Layer::", "Portal", "BinUtility", "BinnedArray",
        "updateCandidates", "initializeVolume", "SurfaceGridLookup",
        # This build. `Navigator::` already matches `NewNavigator::`; the walk
        # is a namespace of its own and matched nothing.
        "nav_walk::", "NewNavigator",
    )),
    ("surface intersect", (
        "::intersect", "intersectionSolver", "Bounds::inside", "checkPathLength",
        "BoundaryTolerance", "RectangleBounds", "TrapezoidBounds",
        "AnnulusBounds", "DiscBounds", "CylinderBounds", "RadialBounds",
        "ConvexPolygonBounds", "SurfaceBounds", "insideBounds",
    )),
    # After the intersection rules, not before them: `intersectionSolver` takes
    # an `Eigen::Transform` and would otherwise be charged to the frame work it
    # is doing the intersection in.
    ("geometry transform", (
        "getSharedPtr", "referenceFrame", "localToGlobalTransform",
        "localToGlobal", "globalToLocal", "Surface::normal", "Surface::center",
        "Surface::transform", "CurvilinearSurface", "DetectorElement",
        "Eigen::Transform",
    )),
    ("material", (
        "MaterialSlab", "ISurfaceMaterial", "MaterialInteract", "materialSlab",
        "interactionLength", "BinnedSurfaceMaterial", "MaterialEffects",
        "HomogeneousSurfaceMaterial", "MaterialInteractor",
        # This build. The seam applies the material the walk planned.
        "LearnedStepper::applyMaterial", "applyPlanRest", "finalizeMaterial",
    )),
    ("measurement selection", (
        "MeasurementSelector", "calculateChi2", "createSourceLinkTrackStates",
        "selectMeasurements",
    )),
    ("kalman update", (
        "GainMatrixUpdater", "GainMatrixSmoother", "KalmanFitter",
        "calculateFilteredResidual", "visit_measurement", "updateStep",
    )),
    ("calibration", (
        "Calibrator", "calibrate", "MeasurementCalibrator",
    )),
    # Its own bucket rather than part of the container: this is the CKF's own
    # per-surface loop over source links, and it calls the calibrator and the
    # selector as well as writing the track state.
    ("track state creation", (
        "TrackStateCreator", "createTrackStates",
    )),
    ("track container", (
        "MultiTrajectory", "VectorTrackContainer", "TrackContainer",
        "TrackStateProxy", "TrackProxy", "addTrackState", "component_impl",
        "VectorMultiTrajectory", "allocateCalibrated", "trackStateContainer",
    )),
    # After every rule that names a caller, because these have none in this
    # profile. Eigen's compiled kernels are reached from the integrator, from
    # the learned transport and from ACTS's own matrix code, and with no frame
    # pointers the sample does not say which. Reported rather than attributed.
    ("eigen kernels", (
        "Eigen::internal::gebp_kernel", "Eigen::internal::gemm_pack",
        "general_matrix_vector_product", "general_matrix_matrix_product",
        "dense_assignment_loop", "copy_using_evaluator",
        "real_2x2_jacobi_svd", "JacobiSVD", "partial_lu_impl",
        "Eigen::internal::triangular_solve", "Eigen::internal::scalar_sum_op",
        "Eigen::DenseBase", "Eigen::MatrixBase",
    )),
    ("ckf glue", (
        "CombinatorialKalmanFilter", "TrackFindingAlgorithm", "Propagator::",
        "PropagatorState", "SurfaceReached", "ActorList", "AbortList",
        "visitSeedIdentifiers", "BranchStopper", "SourceLinkAccessor",
        "SourceLinkAdapterIterator",
    )),
    ("seeding", (
        "SeedFinder", "SeedFilter", "GridTriplet", "Seeding", "SpacePoint",
        "createTripletTopCandidates", "createDoublets", "DoubletsForMiddleSp",
        "CandidatesForMiddleSp", "MiddleSpInfo", "sortByCotTheta",
        "CylindricalSpacePointGrid", "TrackParamsEstimation",
        "estimateTrackParamsFromSeed", "SeedsToProtoTracks",
    )),
    ("digitization", (
        "Digitization", "Smear", "Cluster", "Segmentation", "channelize",
        "DigiComponents", "GeometricConfig",
    )),
    ("truth / selection", (
        "TruthMatcher", "ParticleSelector", "TruthMatching", "SimHit",
        "SimParticle", "TruthSeed",
    )),
    ("root io", (
        "TTree", "TBranch", "TBasket", "TBuffer", "TFile", "TStreamer", "TClass",
        "TObject", "TH1", "TDirectory", "deflate", "inflate", "compress",
        "ZSTD", "LZ4", "lzma", "crc32", "adler32",
    )),
    ("edm4hep input", (
        "EDM4hep", "edm4hep", "podio", "Podio",
    )),
    ("alloc / runtime", (
        "malloc", "free", "cfree", "operator new", "operator delete",
        "_int_free", "_int_malloc", "memcpy", "memset", "memmove", "tcache",
        "unlink_chunk", "arena", "_Rb_tree", "__memcmp", "strlen",
    )),
    ("python / bindings", (
        "PyObject", "_Py", "PyEval", "pybind11", "cpython",
    )),
    ("framework", (
        "Logging", "FilterPolicy", "PrintPolicy", "WhiteBoard", "Sequencer",
        "AlgorithmContext",
    )),
]

# Used only when no symbol rule matched. A library the reconstruction only
# enters from one algorithm answers the question on its own.
DSO_RULES = [
    # `libActsExamplesIoJson` holds the smearing functions the digitization
    # configuration names, so its time is digitization rather than input.
    ("digitization", ("libActsExamplesDigitization", "libActsExamplesIoJson")),
    # `libActsExamplesTrackFinding` is deliberately not here. It holds the
    # seeding algorithm as well as `TrackFindingAlgorithm`, so the library does
    # not say which of the two a symbol belongs to and only the name does.
    ("truth / selection", ("libActsExamplesTruthTracking", "libActsExamplesFatras")),
    ("ambiguity", ("libActsExamplesAmbiguityResolution",)),
    ("edm4hep input", ("libActsExamplesIoEDM4hep", "libpodio", "libedm4hep")),
    ("root io", ("libActsExamplesIoRoot", "libCore.so", "libTree", "libRIO",
                 "libHist", "libNet", "libz.", "libzstd", "liblzma", "libMathCore")),
    ("framework", ("libActsExamplesFramework",)),
    ("geometry / dd4hep", ("libDDCore", "libDDG4", "libActsPluginDD4hep",
                           "libDD4hep")),
    ("kernel", ("[kernel",)),
    # BEFORE the interpreter's rule and not after it. This repository's CKF is
    # a Python extension, so its file name carries `cpython` and the rule
    # below matched it: every symbol of the arm's own propagator, navigator,
    # stepper and transport that no symbol rule caught was charged to the
    # bindings, 6.0% of `new-off` and 9.6% of `pa_cell`. The translation unit
    # holds the CKF and nothing else, so what is left in it after the symbol
    # rules is the filter's own glue.
    ("ckf glue", ("learned_ckf",)),
    ("python / bindings", ("libpython", "PythonBindings", "cpython")),
    ("alloc / runtime", ("libc.so", "libm.so", "libstdc++", "ld-linux",
                         "libgcc", "[unknown]")),
]

# What is inside track finding on this chain. Nothing else in it propagates,
# navigates, calibrates a measurement or writes a track state.
CORE = ("stepper", "learned transport", "eigen kernels", "field",
        "navigator", "surface intersect", "geometry transform", "material",
        "measurement selection", "kalman update", "calibration",
        "track state creation", "track container", "ckf glue")

# `(.*)` and an explicit rstrip rather than `(.*?)\s*$`: the non-greedy group
# backtracks over every position of the line, and one instantiated template
# symbol from this repository's translation unit is several kilobytes long, so
# the lazy form takes minutes per file where this takes a second.
ROW = re.compile(r"^\s+(\d+\.\d+)%\s+(\S+)\s+\[[.k?]\]\s+(.*)$")
ROW_NO_DSO = re.compile(r"^\s+(\d+\.\d+)%\s+\[[.k?]\]\s+(.*)$")


def strip_templates(sym):
    """Drop balanced <...> so that a template argument cannot match a rule."""
    out, depth = [], 0
    for ch in sym:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def bucket_of(sym, dso):
    if sym.startswith("0x") or sym.startswith("[unknown]"):
        return "kernel" if dso.startswith("[kernel") else "unresolved"
    name = strip_templates(sym)
    for stage in (name, sym):
        for bucket, keys in SYMBOL_RULES:
            if any(k in stage for k in keys):
                return bucket
    for bucket, keys in DSO_RULES:
        if any(k in dso for k in keys):
            return bucket
    return "other"


EVENT_COUNT = re.compile(r"^#\s+Event count \(approx\.\):\s+(\d+)")


def cpu_ns(path):
    """The report's own `Event count`, which for `cpu-clock` is nanoseconds.

    perf's header carries the total the percentages are shares of. Without it
    a share is comparable only between two runs of the same length, and the
    arms here are not the same length: that is the subject.
    """
    for line in path.read_text(errors="replace").splitlines():
        m = EVENT_COUNT.match(line)
        if m:
            return int(m.group(1))
        if not line.startswith("#") and line.strip():
            break
    return None


def read(path):
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = ROW.match(line)
        if m:
            rows.append((float(m.group(1)), m.group(2), m.group(3).rstrip()))
            continue
        m = ROW_NO_DSO.match(line)
        if m:
            rows.append((float(m.group(1)), "", m.group(2).rstrip()))
    return rows


def report(path, audit):
    rows = read(path)
    totals = collections.Counter()
    members = collections.defaultdict(list)
    for pct, dso, sym in rows:
        b = bucket_of(sym, dso)
        totals[b] += pct
        members[b].append((pct, sym))
    total = sum(totals.values())
    core = sum(totals.get(b, 0.0) for b in CORE)

    print(f"\n=== {path}")
    print(f"    {len(rows)} symbols, {total:.1f}% of samples above the report's limit")
    print(f"\n    {'bucket':24s} {'% of all':>9s} {'% of track finding':>19s}")
    for name, pct in totals.most_common():
        share = f"{pct / core * 100:18.1f}%" if name in CORE and core else " " * 19
        print(f"    {name:24s} {pct:8.2f}% {share}")
    print(f"\n    track finding, the {len(CORE)} buckets marked above: {core:.2f}% of samples")

    if audit:
        print("\n    the three largest symbols in each bucket")
        for name, _ in totals.most_common():
            if name == "unresolved":
                # An address is not a name. The library it is in is the only
                # thing about an unresolved sample that can be read.
                by_dso = collections.Counter()
                for pct, dso, sym in rows:
                    if bucket_of(sym, dso) == "unresolved":
                        by_dso[dso] += pct
                for dso, pct in by_dso.most_common(3):
                    print(f"      {name:22s} {pct:6.2f}%  (no symbols) {dso}")
                continue
            for pct, sym in sorted(members[name], reverse=True)[:3]:
                print(f"      {name:22s} {pct:6.2f}%  {strip_templates(sym)[:88]}")
    return totals


def per_candidate(paths, totals, candidates):
    """Every bucket in nanoseconds per candidate, one column per run.

    A share of a run's own samples cannot be compared between two arms that
    neither take the same time nor build the same number of candidates, and
    none of the arms here do. The share times the report's own `Event count`,
    which is the run's CPU nanoseconds, over the candidate count from that
    run's `TrackFindingAlgorithm::finalize` line, can.

    It carries the caveat every number in this file carries. The buckets are
    flat self time, so a bucket says where the time was spent and not what
    asked for it.
    """
    names = [q.parent.name for q in paths]
    ns = [cpu_ns(q) for q in paths]
    if any(n is None for n in ns):
        print("\nno `Event count` header, so no per-candidate table")
        return
    seen = []
    for t in totals:
        for b in t:
            if b not in seen:
                seen.append(b)
    order = sorted(seen, key=lambda b: -max(t.get(b, 0.0) for t in totals))

    print("\n=== nanoseconds per candidate, by bucket")
    print()
    print("| bucket | " + " | ".join(names) + " |")
    print("| --- | " + " | ".join("---:" for _ in names) + " |")
    for b in order:
        row = [f"{t.get(b, 0.0) / 100.0 * n / c:,.0f}"
               for t, n, c in zip(totals, ns, candidates)]
        mark = "**" if b in CORE else ""
        print(f"| {mark}{b}{mark} | " + " | ".join(row) + " |")
    core = [sum(t.get(b, 0.0) for b in CORE) / 100.0 * n / c
            for t, n, c in zip(totals, ns, candidates)]
    print("| **the bold buckets together** | "
          + " | ".join(f"{v:,.0f}" for v in core) + " |")
    print("| the whole run | "
          + " | ".join(f"{n / c:,.0f}" for n, c in zip(ns, candidates)) + " |")
    print()
    print("`Event count` over the candidate count, per run. The buckets in "
          "bold are the ones inside track finding; the rest of the run is in "
          "the table so the two totals can be read against each other.")
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dirs", nargs="+", type=Path,
                   help="directories holding perf_dso_sym.txt, or the files themselves")
    p.add_argument("--audit", action="store_true",
                   help="print the largest symbols in each bucket, to check the rules")
    p.add_argument("--candidates", nargs="+", type=int, default=None,
                   help="one candidate count per directory, in the same "
                        "order, from each run's "
                        "`TrackFindingAlgorithm::finalize` line. With it every "
                        "bucket is also printed in nanoseconds per candidate, "
                        "which is the only form in which two arms that build "
                        "different amounts of filter work can be compared")
    args = p.parse_args()
    paths, totals = [], []
    for d in args.dirs:
        path = d
        if d.is_dir():
            path = d / "perf_dso_sym.txt"
            if not path.exists():
                path = d / "perf_flat.txt"
        if not path.exists():
            print(f"\n=== {path}: not there")
            continue
        paths.append(path)
        totals.append(report(path, args.audit))
    if args.candidates:
        if len(args.candidates) != len(paths):
            raise SystemExit(f"{len(args.candidates)} candidate counts for "
                             f"{len(paths)} profiles")
        per_candidate(paths, totals, args.candidates)


if __name__ == "__main__":
    main()
