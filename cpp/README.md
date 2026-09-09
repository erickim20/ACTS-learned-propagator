# cpp/ — the deployable kernel, and where it plugs into ACTS

Everything here was checked against ACTS **44.99.99**, read out of the ODD
image already on this machine (`ghcr.io/opendatadetector/sw`). File and line
references below are to that tree; `./acts_headers.sh` reproduces it.

```
./acts_headers.sh                 # ACTS + Eigen 3.4.0 + Boost 1.88 out of the image
python ../export_kernel.py        # weights header + reference vectors
eval "$(./acts_headers.sh)" && ./build.sh
```

| file | |
| --- | --- |
| `LearnedTransport.hpp` | the kernel. helix + g_θ + the σ head, no ACTS dependency |
| `LearnedStepper.hpp` | the ACTS adapter: a `StepperConcept` model wrapping `EigenStepper` |
| `gtheta_weights.hpp` | generated. **Tracked** — the port means nothing without the weights it was checked against, and the `source:` line on line 2 is the only thing that says which model it holds |
| `test_kernel.cpp` | compares against the Python that produced the report's numbers |
| `bench_kernel.cpp` | latency, dependency-chained, against ACTS's own `EigenStepper` and an RKN both compiled here. Build it with `run_in_image.sh` for the ACTS arms |
| `LearnedJacobian.hpp` | the jump's transport Jacobian in ACTS's free 8×8 form, plus `state.derivative`. No ACTS dependency either |
| `test_jacobian.cpp` | compares that against `src/prop/jacobian_free.py` |
| `geom_probe.cpp` | the adapter inside a real `Propagator` over the real ODD geometry |
| `cov_probe.cpp` | one jump's transported covariance against ACTS's own |
| `geom_dump.cpp` | one row per sensitive module out of the real geometry, for retargeting the teacher. Join it on `join_key`, not on `geometry_id` |
| `acts_headers.sh`, `build.sh` | the two commands above |
| `run_in_image.sh` | builds and runs the two probes inside the image, where `libActsCore`, DD4hep and ROOT live |

---

## 1. The port is numerically correct

`export_kernel.py` runs `closed_loop.predict` — the same function every chained
result in this project came from — on 4,000 teacher jumps and records both ends.
`test_kernel` replays them.

| | median | p99 | max |
| --- | ---: | ---: | ---: |
| position | 0 | 2.8e-16 | 3.2e-15 |
| momentum | 0 | 2.9e-16 | 1.1e-13 |
| arc length | 0 | 1.9e-15 | 1.8e-14 |
| raw outputs | 0 | 1.1e-15 | 1.8e-14 |
| σ scale m | 6.0e-17 | 1.1e-15 | 7.6e-14 |

**Worst 1.1e-13**, no `ok`-flag mismatches, 100% surface reach, against the
1e-10 bar the Jacobian work used. Errors are relative to the RMS of each quantity, not to each entry —
dividing by `|want|` per entry is the trap `jacobian_helix.rel()` documents,
where a near-zero component reports a huge relative error for nothing.

## 2. Why this is hand-written and not ONNX

ONNX Runtime is the sanctioned ACTS path (`ActsPlugins/Onnx`, used by
`ActsExamples::NeuralCalibrator`). It is the wrong runtime here:

- ORT costs a **measured 1788 + 98·b ns**.
- `findTracks()` is depth-first over an explicit stack, so **b = 1, always**.
- 1,886 ns for the inference alone — against ACTS's **2,076 ns** without a
  covariance (§3), so ORT alone eats 91% of what it would replace and leaves
  nothing for the helix solve.

The network is
5,376 multiply-adds, so writing it out is a morning's work, and it is written
out. Two further facts from the image: `libonnxruntime.so.1.21.0` is present but
**`libActsPluginOnnx` is not built**, and `Onnx/NeuralCalibrator.hpp` therefore
does not currently compile there — using the plugin route at all would need
ACTS rebuilt with `ACTS_BUILD_PLUGIN_ONNX=ON`.

## 3. Latency

One binary, one machine, 4,000 real ODD jumps onto their own module planes,
dependency-chained so jump k+1 waits on jump k. Built at the generic x86-64
baseline `libActsCore` was built at, so all arms get identical codegen. Median
of five runs; the spread across runs is under 2% on every line.

| arm | ns/jump | |
| --- | ---: | --- |
| the reference RKN *(the floor)* | 674.6 | understates ACTS by **3.08×** |
| **ACTS `EigenStepper`** | **2,076.3** | the real baseline |
| **helix + g_θ** | **1,272.3** | what ships |
| **ACTS `EigenStepper`, covariance on** | **3,624.6** | what a CKF actually calls |
| **helix + g_θ + `jumpJacobian`** | **1,607.2** | what `writeBack` does |
| ORT, batch 1 | 2,134.9 | slower than the integrator it would replace |

**2.26× carrying a covariance, 1.63× without.** The first is the one that
applies: `CombinatorialKalmanFilter` calls `transportCovarianceToBound` at every
surface it considers.

The two rows are the result, not just two numbers. A covariance costs ACTS
1,548 ns and costs the kernel 335 ns, because ACTS propagates the 8×8 through
every Runge-Kutta step while the learned jump forms it once in closed form.

| | no covariance | with | cost |
| --- | ---: | ---: | ---: |
| ACTS `EigenStepper` | 2,076.3 | 3,624.6 | 1.75× |
| the kernel | 1,272.3 | 1,607.2 | **1.26×** |

Inside the kernel: plane intersection 248.9 ns, network and its single field
read 1,023.4 ns, the 8×8 transport Jacobian 337.7 ns.

**Both arms were checked to be solving one problem before any ratio was taken.**
Mean 3D path length 237.5 mm by the helix against 237.6 mm by ACTS, 0.05% apart.
Both finish on the destination plane, the helix to 2e-12 mm and the RKN to 8e-5
mm against ACTS's own 1e-4 mm surface tolerance. ACTS reaches the plane on 99.8%
of jumps, the RKN on 100%.

**The reference RKN is faster than the kernel**, at 0.53×. It reaches the plane
in 5.2 steps, and 5.2 steps of RKN4 is less arithmetic than 5,376 multiply-adds.
It is a floor and not a baseline: no material, no portals, no navigator, no
covariance, and ACTS's own stepper on the same jumps costs 3.08× more.

**A confound that had to be measured.** `libActsCore.so` is a spack build
targeting generic x86_64 with no AVX-512, so the whole binary must be built at
that baseline, and that is not neutral:

| | SSE2 | AVX-512 | gain |
| --- | ---: | ---: | ---: |
| the reference RKN | 675.9 | 586.5 | 1.15× |
| plane solve | 251.5 | 249.1 | 1.01× |
| g_θ alone | 1,064.8 | 659.3 | **1.62×** |
| the 8×8 Jacobian | 301.5 | 226.2 | 1.33× |
| the kernel | 1,316.4 | 908.8 | 1.45× |

Dense MACs vectorise; branchy adaptive integration does not. So 2.26× and 1.63×
are the matched-codegen measurements, and an ACTS rebuilt at the kernel's
baseline would raise both. That is an *estimate* and is not quoted.

**A trap worth recording.** Building the kernel `-march=native` against a
generic-baseline `libActsCore` does not give a wrong number — it gives
`std::bad_variant_access` out of the propagator on the first jump, because
Eigen's alignment assumptions change across the library boundary. It reads
exactly like an ACTS bug and is not one.

**The field-access ratio.** An RKN4 step evaluates the acceleration at four
points but at only three distinct positions, since `k2` and `k3` share the
midpoint and differ only in the direction handed to the cross product. Over 5.2
steps that is **16 : 1** against the learned jump's single read. That is where
any advantage has to come from, and it is a cache-miss cost that no amount of
SIMD removes. `bench_kernel.cpp` reads the midpoint twice rather than reusing
it, so its RKN arm pays 21 lookups where 16 would do, which makes the floor
slightly slower than it needs to be.

**How far apart the two trajectories land**, in the plane: median 19 µm, p99
9.1 mm, max 61 mm. That is not an error in either arm. It is the constant-2 T
approximation the kernel's core makes, measured on the same jumps, and it is the
quantity g_θ is trained to remove.

## 4. Where it plugs into ACTS

`CombinatorialKalmanFilter<propagator_t, track_container_t>` (line 161) is
templated on the propagator, and `Propagator` on the stepper, so the hook is at
the type level. The CKF actor reaches the stepper at:

```
:475  stepper.transportCovarianceToCurvilinear(...)
:477  stepper.transportCovarianceToBound(...)      <- C
:488  stepper.boundState(...)                      <- x
```

Three alternatives were checked in the image's own headers and rejected:

| candidate | verdict |
| --- | --- |
| `MeasurementCalibrator` (what `NeuralCalibrator` uses) | **wrong hook.** It produces **V**, the measurement covariance. The σ head scales the process noise in **C**. Same for the gate, `S = HCHᵀ + V`; **opposite** for the gain, `K = CHᵀS⁻¹` — inflating V trusts the prediction more, inflating C trusts the measurement more |
| an extra Actor | **not available.** `using Actors = ActorList<CombinatorialKalmanFilterActor>;` is hard-coded at line 825. Needs an ACTS patch |
| a custom Propagator | possible, but larger, and unnecessary given the below |

### Two things the compiler said that the design did not

1. **`Acts::EigenStepper` is `final`.** No inheritance. `LearnedStepper` is
   composition plus twenty-six forwarding methods.

2. **`step(State&, Direction, const IVolumeMaterial*)` never sees the
   destination surface.** ACTS's stepper is a step-size machine; the target
   belongs to the navigator. The target is latched out of
   `updateSurfaceStatus`, which has two call sites —
   `Propagator.ipp:49` inside `getNextTarget()`, which only runs
   when the target is exhausted, and `Propagator.ipp:122` after the step, which
   is the one that refreshes the latch between consecutive steps. `step()` now
   **consumes** the latch, which is what makes a stale one unreachable. See
   `LearnedStepper.hpp`. But `updateSurfaceStatus(State&, const Surface&,
   …)` does receive it and the Propagator calls it immediately before every
   step to constrain the step size (`Propagator.ipp:50`). So the target is
   latched there and consumed in the next `step`. **This is not a documented
   contract** and is the first thing to re-check against a newer ACTS.

### What holds today

```
static_assert(Acts::StepperConcept<collider_ml::LearnedStepper>);       // passes
using P = Acts::Propagator<collider_ml::LearnedStepper, Acts::Navigator>;  // instantiates
```

Both are checked by `build.sh` whenever `ACTS_INC` is set.

## 4b. The covariance is transported

`writeBack` advances `state.jacTransport` by `LearnedJacobian.hpp`'s `D` and sets
`state.derivative`. Two checks, and they fail differently, which is why there are
two.

`test_jacobian` compares the C++ against `src/prop/jacobian_free.py` on 4,000
real teacher jumps: **worst 4.0e-13**, and the `EigenStepper.ipp:338` block
asserts hold at exactly 0.00e+00 on the C++ matrix. That is a port check and
cannot catch anything the Python and the C++ get wrong together.

`cov_probe` is the independent one. One jump in a constant 2 T field, against
ACTS's own stepper and its own covariance engine, with every factor other than
the local `jacTransport` and `derivative` taken from ACTS.

| | worst of 8 cases |
| --- | ---: |
| transported covariance, relative to the matrix | 8.1e-6 |
| endpoint position | 3.7e-2 mm |
| endpoint / predicted from the helix constant | 1.00 |

The endpoint residual is the helix constant and nothing else: 0.3 GeV/(T m) here
against ACTS's 0.299792458 is 6.9e-4 on the curvature, and two equal-length arcs
whose curvatures differ by δκ separate by δκ s²/2. That constant belongs to the
model rather than being patched.

## 4c. The process noise seam

`transportCovarianceToBound` delegates and then adds the jump's `Q` to the
top-left 5×5, which puts it between ACTS's transport and ACTS's own material
noise. Checked in `kLayerApproach`, 406 learned jumps over the
real geometry: ΔQ(2 transport calls)/ΔQ(1) = 1 to 9.5e-12, and the σ head
changes the covariance while leaving the parameters, path and step count
bit-identical.

The debt is a matrix rather than a flag, because two jumps can fall between two
transport calls and a flag would let the second overwrite the first jump's debt.

**The table is a fixture.** `Q_nomat.parquet` is not on this box and item 7
produces it. `LearnedNoise.hpp` refuses a table whose material flag is set,
because a material-on `Q` double counts scattering with ACTS's own and the two
are indistinguishable by inspection.

## 5. What is deliberately not done

- **`sig0`/`sig1` are a setter, not a lookup.** In ACTS they belong to the
  calibration context. Until that is wired the position correction is scaled by
  whatever the caller passed.
- **`useLearned` defaults to false.** A stepper that silently changes the
  physics is not something anyone should have to notice.

## 5b. It has been run inside a propagator

`geom_probe.cpp`, real ODD `TrackingGeometry`, real `Navigator`, covariance
transport on, eight start states from 1 to 50 GeV and |η| up to 2.4.
`./run_geom_probe.sh` builds and runs it inside the image, which is where
DD4hep, ROOT and `libActsPluginDD4hep` live.

| | |
| --- | --- |
| bit-identical to stock, switch off | 8 of 8 |
| bit-identical to stock, switch on | 8 of 8 |
| targets latched | 513 |
| accepted by `isSensitive()` | 153 |
| built into a `Jump` | 0 |
| answered by the kernel | 0 |

`memcmp` over end parameters, covariance, path length and step count, not a
tolerance. The zeros in the last two rows are why the first two rows are what
they are: every sensitive ODD surface is a `Plane` and `toJump` switches on
`Cylinder` and `Disc`. Transparency is not demonstrated by this and cannot be
until the kernel handles planes.

Two failures worth recording, because each names the wrong culprit. Without
`source $(spack dd4hep)/bin/thisdd4hep.sh` the binary links and runs and then
dies inside `fromCompact` with *Failed to locate plugin to interprete files of
type "lccdd"*, which reads as a broken detector description. And `-march=native`
against the image's generic `libActsCore` gives `std::bad_variant_access` out of
the propagator rather than a wrong number.

## 6. And the thing that comes first

None of this is worth running until the teacher is fixed. On a real physics
event the learned correction is **worse** than plain helix — 29.2% against
33.1% hit efficiency on a chained measurement. Porting weights that
do not transfer just moves the problem into C++.

The kernel is ready. The model is not.
