# ACTS learned propagator

Learned surface-to-surface transport inside CKF track finding, built against
ACTS 44.99.99 and the Open Data Detector.

A combinatorial Kalman filter spends most of its time on one question: a
particle is on this detector layer, where does it cross the next one. This
replaces the answer with a closed-form helix plus a small network, and leaves
the recursive filter itself unchanged.

One Kalman step moves two things, and both have to be supplied:

    x_pred = f(x)                 f = helix + g_theta
    C_pred = F C F^T + Q          F computed from the helix, Q measured

The learned pieces are `g_theta`, the correction to the helix; a per-jump sigma
read off a spare network output; and `s_theta`, which layer comes next. `F` is
not learned. `Q` is measured.

## Layout

    src/prop/      the Python library. Every module has its own CLI entry
                   point, so `python -m prop.<name> --help` works throughout
    cpp/           the deployable kernel and the ACTS adapters. See cpp/README.md
    sim/           everything that runs inside the ODD image
    sim/baseline/  the stock CKF baseline: configs, runners and what produced
                   the number. See its README
    sim/profile/   where reconstruction time goes. See its README
    bench/         timing and profiling on the host
    config/        detector configuration

The two ACTS integration points are `cpp/LearnedStepper.hpp`, a `StepperConcept`
model wrapping `EigenStepper`, and `cpp/NewNavigator.hpp`, which names the next
sensitive module from where the track already is. Each carries its own header
comment explaining what ACTS forces and why.

## Environment

An x86_64 Linux host with Docker. `sim/` runs inside the ColliderML ODD image,
because ACTS there is built against a spack Python 3.13 that a host virtualenv
cannot import. Everything else runs on the host.

    python3 -m venv .venv && . .venv/bin/activate
    pip install -e .

    # by digest, never by tag. :latest is arm64 and has no /opt/odd in it
    docker pull ghcr.io/opendatadetector/sw@sha256:20e4df7b0d6befd70ce0113d852350b25cbd9aac659d1bc530036a2c85a68f89
    cpp/acts_headers.sh          # ACTS, Eigen and Boost out of the image

Run scripts from the repository root. The file paths in the argparse defaults
are relative to the working directory.

## Running

Generate a teacher set, fit, and score against the gate the CKF applies:

    sim/run_in_odd.sh sim/gen_teacher.py --events 120 --tracks 100 \
        --field map --material --pdg pion --tag phys_pion --out /data/teacher

    python -m prop.make_teacher_pairs teacher/phys_pion --out teacher_pion.parquet
    python -m prop.mix_teacher --out teacher_phys.parquet
    python -m prop.train_gtheta teacher_phys.parquet --act ptanh --v1 \
        --save gtheta.npz --dump runs/phys.npz
    python -m prop.ckf_prototype --tracks 300 --true-seed --model gtheta.npz

`--act ptanh` because transporting the covariance needs a C1 Jacobian. `--v1`
supervises the direction and q/p outputs; a score based only on one-step
position is blind to them, and a model can win that score by trading direction
away.

Export the weights and build the C++ side:

    python -m prop.export_kernel        # weights header + reference vectors
    eval "$(cpp/acts_headers.sh)" && cpp/build.sh

Run the CKF with a chosen stepper:

    sim/run_reco_ckf.sh <source run> <output subdir> <arm>

where the arm is one of `stock`, `learned-off`, `learned-on`, `helix`,
`new-off`, `new-helix`, `new-learned`. `stock` is ACTS's own prebuilt
`SympyStepper` CKF, so a difference measured against it is never a single
change.

## Generated inputs

The data these scripts read is not in the repository. It is reproducible:

| input | produced by |
| --- | --- |
| teacher tables | `sim/gen_teacher.py` then `prop.make_teacher_pairs` |
| `oddb.npz`, the field map grid | `prop.build_fieldmap` from the image's `odd-bfield.csv` |
| `cpp/field.bin` | `prop.export_fieldbin` |
| `cpp/reference.bin` | `prop.export_kernel` |
| `cpp/muon_jumps*.bin`, the latency benchmark input | `prop.export_jumps` |
| `sigma_C.parquet` | a CKF run, read back by `prop.export_qtable` |

Two generated files are tracked, because nothing else records them:

| file | |
| --- | --- |
| `cpp/gtheta_weights.hpp` | the trained model. The `source:` line at the top is the only record of which model it holds |
| `cpp/cell_sigma.bin` | the per-cell sigma the position outputs were trained in |

`cpp/acts_examples_src/` holds ACTS sources at the image's build commit and is
not tracked either. `cpp/fetch_ckf_src.sh` retrieves them.

## License

MPL-2.0, the same license as ACTS. See `LICENSE`.
