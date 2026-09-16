"""digi_and_reco.py's own chain, with the CKF's stepper chosen at run time.

The chain is not copied and not patched on disk: the one
thing `addCKFTracks` takes from outside is the type-erased factory it reads
off `acts.examples.TrackFindingAlgorithm.makeTrackFinderFunction`
(`reconstruction.py:1708`), so this driver swaps that attribute and then runs
the unmodified script. On `--stepper stock` nothing is swapped at all and the
prebuilt `libActsExamplesTrackFinding` factory runs, which is what makes the
acceptance test a check of the plumbing rather than of a reimplementation.

    --stepper stock         the prebuilt SympyStepper CKF, untouched
    --stepper learned-off   Propagator<LearnedStepper, Navigator>, switch off:
                            the inner EigenStepper runs (bit-identical to a
                            stock EigenStepper build), so the
                            difference to `stock` is EigenStepper vs
                            SympyStepper and nothing else
    --stepper learned-on    the learned jump fires; add --qtable for the
                            measured material-off process noise
    --stepper helix         the learned jump fires with the network skipped:
                            the physics core alone, and the only arm whose
                            wall time prices the helix rather than the helix
                            plus a forward pass that is multiplied by zero
    --stepper new-off       the moved seam: Propagator<LearnedStepper,
                            NewNavigator> with the learned transport off,
                            so the transport is Runge-Kutta over a whole leg
                            and the material of the surfaces the navigator no
                            longer stops at is applied by the stepper.
                            `--nav-walk-off` turns the walk off inside it, so
                            every navigator member forwards and nothing else
                            changes.
    --stepper new-helix     the moved seam with the learned transport on and
                            the network skipped, so one helix call replaces a
                            whole leg of Runge-Kutta rather than the 8 mm
                            fragment of --stepper helix
    --stepper new-learned   the same with the network in, which is the
                            configuration the project was built for: the leg
                            the transport flies is now the leg the teacher was
                            fitted on

The three `new-*` arms differ from each other in the stepper alone. They differ
from `helix` and `learned-on` in that the navigator names the far module, so
the transport covers the whole 225 mm leg instead of the last 8 mm of it.

Runs inside the image via sim/run_reco_ckf.sh. Arguments it does not know are
handed to digi_and_reco.py unchanged.
"""
import argparse
import os
import runpy
import sys


def arm_algorithm_timing(outdir):
    """Make the Sequencer write its per-algorithm timing next to the run.

    `Sequencer::Config` carries `outputDir` and `outputTimingFile`
    (`Framework/Sequencer.hpp:72-75`) and `digi_and_reco.py:118` sets neither,
    so the file goes to the working directory, which is `/workspace` and is
    mounted read only. Setting `outputDir` is the only change and it adds one
    file write at the end of the run.

    Why it is wanted: digitisation, seeding and ambiguity resolution are
    identical in every arm and are most of a run's wall clock, so their
    variance lands on every comparison while carrying no signal. The
    per-algorithm line is the part with any resolution in it. The whole stage
    stays the headline number.

    Patched the same way the CKF factory is: `digi_and_reco.py` does
    `from acts.examples import Sequencer` at line 6, and runpy executes that
    import after this runs, so replacing the module attribute reaches it.
    """
    import acts.examples as ae

    stock = ae.Sequencer

    def sequencer(*a, **kw):
        kw.setdefault("outputDir", outdir)
        return stock(*a, **kw)

    ae.Sequencer = sequencer
    print(f"[reco_ckf] per-algorithm timing -> {outdir}/timing.csv", flush=True)


def raise_algo_log_level(name):
    """Let the TrackFindingAlgorithm print its own finalize() statistics.

    `digi_and_reco.py:43` pins `LOG_LEVEL = acts.logging.FATAL` for the whole
    chain, which hides the eleven lines `TrackFindingAlgorithm::finalize`
    writes at INFO: total and deduplicated seeds, failed seeds, failed
    smoothing, failed extrapolation, found and selected tracks, stopped
    branches, and skipped second passes. Those are the only place the
    reconstruction says WHERE it lost a track, and no performance file carries
    them.

    Wrapping the class rather than editing the script keeps the run the same
    run: `addCKFTracks` constructs it with `config=` and `level=` keywords
    (`reconstruction.py`), so only the level changes and every other argument
    is passed through untouched.
    """
    import acts
    import acts.examples as ae

    level = getattr(acts.logging, name)
    stock = ae.TrackFindingAlgorithm

    class LoudTrackFinding:
        def __call__(self, *a, **kw):
            if len(a) >= 2:
                a = (a[0], level) + tuple(a[2:])
            else:
                kw["level"] = level
            return stock(*a, **kw)

        def __getattr__(self, name_):
            return getattr(stock, name_)

    ae.TrackFindingAlgorithm = LoudTrackFinding()
    print(f"[reco_ckf] TrackFindingAlgorithm log level -> {name}", flush=True)
    return ae.TrackFindingAlgorithm


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--stepper", required=True,
                    choices=["stock", "learned-off", "learned-on", "helix",
                             "new-off", "new-helix", "new-learned"])
    ap.add_argument("--nav-walk-off", action="store_true",
                    help="with a --stepper new-* arm, keep NewNavigator in "
                         "the propagator and switch its walk off. That is "
                         "Every NavigatorConcept member then "
                         "forwards and the output must be bit-identical to "
                         "--stepper learned-off. Meaningless on any other arm "
                         "and refused there")
    ap.add_argument("--qtable", default="",
                    help="binary from export_qtable.py, path inside the "
                         "container; empty adds no process noise")
    ap.add_argument("--no-publish-track", action="store_true",
                    help="stop LearnedStepper publishing to LegChannel on "
                         "every step. A measurement and not a fix: the "
                         "wrapper measures 0.30 s of the moved seam's "
                         "0.42 s unsplit, and this is the only "
                         "thing inside the wrapper that does arithmetic per "
                         "step. Meaningful only with --nav-walk-off, since "
                         "the walk is what reads the channel; the factory "
                         "refuses the other combination")
    ap.add_argument("--cell-sigma", default="",
                    help="binary from export_cell_sigma.py, path inside the "
                         "container: the per-jump sigma the position outputs "
                         "were trained in, keyed on (module class, pT bin, "
                         "|eta| bin). Empty keeps the two constants 20 um and "
                         "43 um that every run before this one used, which are "
                         "2.9x too large on a pixel's loc1 and 28x too small "
                         "on a short strip's. cpp/CellSigma.hpp says the rest")
    ap.add_argument("--sigma-head", action="store_true")
    ap.add_argument("--no-resolve-material", action="store_true",
                    help="set Navigator::Config::resolveMaterial false in the "
                         "navigator this factory builds. A seam proxy. "
                         "Only the non-stock arms build a navigator, so this "
                         "cannot be combined with --stepper stock")
    ap.add_argument("--no-seam-material", action="store_true",
                    help="with a --stepper new-* arm, the moved seam names the far "
                         "module and flies the whole leg but applies none of "
                         "the material it flew past. An ablation no stock "
                         "navigator flag can build, because no flag "
                         "removes the material without also removing the "
                         "material-only track states. Refused on any other arm")
    ap.add_argument("--no-seam-slab", action="store_true",
                    help="with a --stepper new-* arm, plan the crossings and stop "
                         "the step on each of them, and apply no material "
                         "there. Splits the pair --no-seam-material removes: "
                         "that one takes away both the plan and the slab, this "
                         "one takes away the slab alone")
    ap.add_argument("--planned-only", action="store_true",
                    help="with a --stepper new-helix or new-learned arm, fire "
                         "the learned transport only on the destination the "
                         "walk named for that transport. "
                         "Without it the gate is LearnedStepper::accepts "
                         "alone, which tests the surface type and not who "
                         "asked, so the transport also fires on the short "
                         "module-to-module targets the stock navigator names "
                         "on a declined leg. Refused on an arm whose "
                         "transport never fires, which would label a run as "
                         "an arm it is not")
    ap.add_argument("--field-gate", type=float, default=0.0,
                    help="with a --stepper new-learned arm, keep the helix and "
                         "skip the network wherever the map at the source of "
                         "the transport is within this many tesla of the 2 T "
                         "the helix core runs at. The threshold used is 0.1, "
                         "the 5 percent of 2 T the barrel and endcap line "
                         "is drawn at. "
                         "Zero is off. Refused on an arm with no network")
    ap.add_argument("--sigma-const", type=float, default=1.0,
                    help="a blanket scale on the declared covariance, on the "
                         "branch the network fired on, with the head off. "
                         "An earlier arm put this at 1.789; below one it "
                         "NARROWS what the "
                         "filter is told the transport got wrong. One is off "
                         "and is the default. Refused with --sigma-head, "
                         "which writes that scale itself")
    ap.add_argument("--sigma-helix", type=float, default=1.0,
                    help="the same scale on the branch that runs when the "
                         "network did not fire. That branch "
                         "carries 4.166 of the deployed arm's 4.964 points of "
                         "fake excess over stock at pileup 200 and nothing "
                         "has ever been armed on it but the tail calibration "
                         "and the blanket factor of two. One is off and is "
                         "the default")
    ap.add_argument("--core-bz", default="fixed",
                    choices=["fixed", "source"],
                    help="which field the helix core of the learned "
                         "transport runs at. `fixed` is 2 T, which is "
                         "LearnedTransport.hpp's kBHelix and what every "
                         "arm on disk was taken on. `source` is the map "
                         "value at each jump's source point, which the "
                         "stepper already reads for the network's input "
                         "10, so no field lookup is added. Measured at "
                         "1.957 offline points of hits lost. Refused on an "
                         "arm whose learned transport never fires, and "
                         "refused with the network on: the fit's target, its "
                         "inputs 8 and 9, the header's mu and sd and the "
                         "v1_scale cells all written at 2 T, so a model "
                         "above a moved core is a model read off a core "
                         "it was not fitted on. --net-core-fit lifts the "
                         "second refusal")
    ap.add_argument("--net-core-fit", action="store_true",
                    help="assert that the weights compiled into this binary "
                         "were fitted on the source core, which is what "
                         "`make_teacher_pairs --bz <map>` produces and what "
                         "the header's `source:` line records. Only meaningful "
                         "beside --core-bz source with the network on, and it "
                         "is an assertion the caller makes: nothing in this "
                         "process can read which header `learned_ckf` was "
                         "built with")
    ap.add_argument("--algo-log-level", default="",
                    help="raise the log level of the TrackFindingAlgorithm "
                         "this driver constructs, e.g. INFO. digi_and_reco.py "
                         "pins LOG_LEVEL to FATAL at its line 43, which hides "
                         "the algorithm's own finalize() statistics: total "
                         "seeds, failed seeds, found and selected tracks, "
                         "stopped branches and skipped second passes. Those "
                         "are what say WHERE an arm lost a track, and the "
                         "per-bin table cannot. Off by default so a measured "
                         "run writes what it wrote before")
    ap.add_argument("--module-dir", default="/repo/cpp")
    ap.add_argument("--reco-script",
                    default="/workspace/scripts/simulation/digi_and_reco.py")
    args, rest = ap.parse_known_args()

    # Off unless ALGO_TIMING names a directory, so a normal run writes exactly
    # what it wrote before. Same convention as TRANSPORT_CENSUS.
    if timing_dir := os.environ.get("ALGO_TIMING", ""):
        arm_algorithm_timing(timing_dir)

    if args.no_seam_slab and not args.stepper.startswith("new-"):
        raise SystemExit(
            "--no-seam-slab needs one of the --stepper new-* arms")

    if args.no_seam_material and not args.stepper.startswith("new-"):
        raise SystemExit(
            "--no-seam-material needs one of the --stepper new-* arms; no "
            "other arm applies material from the seam")

    if args.field_gate and args.stepper != "new-learned":
        # `new-helix` has no network to take off and every other arm has no
        # walk. Same rule as the flags below: an arm that accepted the flag and
        # ignored it would be labelled as an arm it is not.
        raise SystemExit(
            "--field-gate needs --stepper new-learned; it takes the network "
            "off where the field is nominal and no other arm has one to take "
            "off")

    # Both scales act inside `LearnedStepper::step`, which only a
    # `new-learned` arm reaches. Same rule as the flags around them: an arm
    # that accepted a flag and ignored it would be labelled as an arm it is
    # not.
    for flag, val in (("--sigma-const", args.sigma_const),
                      ("--sigma-helix", args.sigma_helix)):
        if val != 1.0 and args.stepper != "new-learned":
            raise SystemExit(
                f"{flag} needs --stepper new-learned; it scales the noise on "
                "one branch of the two-branch table and no other arm has two")
        if val <= 0.0:
            raise SystemExit(f"{flag} must be positive; it enters the "
                             "covariance squared")
    if args.sigma_const != 1.0 and args.sigma_head:
        raise SystemExit(
            "--sigma-const is refused with --sigma-head: the head writes the "
            "scale on that branch and the constant is for the head OFF")

    if args.core_bz == "source":
        # Same rule as the flags around it. On `stock`, `learned-off` and
        # `new-off` there is no learned transport to move the core of, and a
        # run that accepted the flag and ignored it would be labelled as an arm
        # it is not.
        if args.stepper not in ("learned-on", "helix", "new-helix",
                                "new-learned"):
            raise SystemExit(
                "--core-bz source needs an arm whose learned transport fires: "
                "learned-on, helix, new-helix or new-learned")
        # And refused with the network on unless the operator states that the
        # header in this binary was fitted on the source core. The refusal
        # exists because the supervised target, inputs
        # 8 and 9, the header's mu and sd and the v1_scale cells are all
        # written at whatever field `make_teacher_pairs --bz` named, so a
        # network above a moved core is read off a core it was not fitted on.
        #
        # Nothing here can see which header `learned_ckf` was built with --
        # the weights are compiled in and `cpp/gtheta_weights.hpp`'s `source:`
        # line is not carried into the module. So this is an assertion the
        # caller makes and the factory prints, not a check.
        if args.stepper in ("learned-on", "new-learned") \
                and not args.net_core_fit:
            raise SystemExit(
                "--core-bz source needs the network off, or --net-core-fit to "
                "say the header in this binary was fitted on the source core. "
                "The supervised target, inputs 8 and 9, the "
                "header's mu and sd and the v1_scale cells are all written at "
                "one field, so a network above a moved core is otherwise read "
                "off a core it was not fitted on")

    if args.planned_only and args.stepper not in ("new-helix", "new-learned"):
        # Same rule as the three flags below. On any other arm the learned
        # transport either never fires or fires behind the stock navigator,
        # where there is no plan to match against.
        raise SystemExit(
            "--planned-only needs --stepper new-helix or new-learned; no "
            "other arm has both a walk and a learned transport")

    if args.nav_walk_off and not args.stepper.startswith("new-"):
        # The same rule as --no-resolve-material below and for the same reason:
        # an arm that accepted a flag it cannot act on would be labelled as an
        # arm it is not.
        raise SystemExit(
            "--nav-walk-off needs one of the --stepper new-* arms; no other "
            "arm builds NewNavigator")

    if args.stepper == "stock" and args.no_resolve_material:
        # ACTS's own factory builds the stock navigator and nothing here can
        # reach its config. Failing is the point: a run that accepted the flag
        # and ignored it would be labelled as an arm it is not.
        raise SystemExit(
            "--no-resolve-material needs an arm whose navigator this "
            "repository builds; --stepper stock builds none")

    if args.stepper != "stock":
        sys.path.insert(0, args.module_dir)
        import acts.examples as ae
        import learned_ckf

        if args.algo_log_level:
            raise_algo_log_level(args.algo_log_level)
        stock_cls = ae.TrackFindingAlgorithm
        # `helix` takes the learned jump and skips the forward pass. It is not
        # a build with the weights zeroed: that one still computes the network
        # and multiplies by zero, so it prices the transport wrongly while
        # showing the same physics.
        # The seam is the navigator and the learned transport is the stepper,
        # and the two are independent: `new-off` is NewNavigator with an
        # EigenStepper inside, `helix` is the learned transport behind the
        # stock navigator, and `new-helix` and `new-learned` are both.
        learned = args.stepper in ("learned-on", "helix", "new-helix",
                                   "new-learned")
        network = args.stepper not in ("helix", "new-helix")
        nav_seam = args.stepper.startswith("new-")
        nav_walk = nav_seam and not args.nav_walk_off
        nav_material = not args.no_seam_material
        seam_material = not args.no_seam_slab

        class FactorySwap:
            """The stock class with one static method swapped.

            An instance attribute shadows __getattr__, so
            `makeTrackFinderFunction` resolves to the learned factory while
            construction and everything else fall through to the stock class.
            Swapping the module attribute rather than setattr on the pybind11
            type keeps this independent of whether the extension type allows
            monkeypatching.
            """

            def __call__(self, *a, **kw):
                return stock_cls(*a, **kw)

            def __getattr__(self, name):
                return getattr(stock_cls, name)

        swap = FactorySwap()
        swap.makeTrackFinderFunction = (
            lambda trackingGeometry, field, level:
            learned_ckf.make_track_finder(
                trackingGeometry, field, int(level), learned=learned,
                sigmaHead=args.sigma_head, qtable=args.qtable,
                cellSigma=args.cell_sigma,
                network=network,
                resolveMaterial=not args.no_resolve_material,
                navSeam=nav_seam, navWalk=nav_walk,
                navMaterial=nav_material,
                seamMaterial=seam_material,
                plannedOnly=args.planned_only,
                fieldGate=args.field_gate,
                sigmaConst=args.sigma_const, sigmaHelix=args.sigma_helix,
                publishTrack=not args.no_publish_track,
                localCore=args.core_bz == "source"))
        ae.TrackFindingAlgorithm = swap
        print(f"[reco_ckf] factory swapped: stepper={args.stepper} "
              f"qtable={args.qtable or 'none'} sigma_head={args.sigma_head} "
              f"cell_sigma={args.cell_sigma or 'constants'} "
              f"network={network} "
              f"resolve_material={not args.no_resolve_material} "
              f"nav_seam={nav_seam} nav_walk={nav_walk} "
              f"nav_material={nav_material} "
              f"seam_material={seam_material} "
              f"planned_only={args.planned_only} "
              f"field_gate={args.field_gate} "
              f"sigma_const={args.sigma_const} "
              f"sigma_helix={args.sigma_helix} "
              f"publish_track={not args.no_publish_track} "
              f"core_bz={args.core_bz} "
              f"net_core_fit={args.net_core_fit}", flush=True)
    else:
        if args.algo_log_level:
            import acts.examples  # noqa: F401  (imported for the side effect)
            raise_algo_log_level(args.algo_log_level)
        print("[reco_ckf] stock chain, nothing swapped", flush=True)

    sys.argv = [args.reco_script] + rest
    runpy.run_path(args.reco_script, run_name="__main__")


if __name__ == "__main__":
    main()
