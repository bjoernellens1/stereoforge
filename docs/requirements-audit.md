# Requirements audit against the approved seven-stage plan

Audit baseline: `88a9e372c908816b891a6158725bc1ab07b02e70`, branch
`implementation/stereoforge`, reviewed on 2026-10-27. The binding specification is
the user-approved **StereoForge: Vulkan stereo, wide RGB-D, and temporal
reconstruction** plan. This audit evaluates that plan, including its acceptance
gates, rather than treating the currently implemented subset as the full goal.
References below identify baseline source locations; subsequent edits can move
their line numbers.

**The plan is not complete.** At the baseline, required Vulkan RGB behaviors,
production integration, controlled quality evidence, and sustained qualification
remained open. Follow-up changes below close some implementation gaps. Implemented
building blocks and successful numerical tests do not close those gates.

## Current follow-up

The shared Vulkan compositor now implements exposure matching, confidence seams,
compatible blending and marked preview through both Python and C++ wrappers.
Python uses its stereo device for views and packs host readback into one transfer.
Current full Python verification passes 221 tests; native Vulkan CTest passes 4/4
and CPU CTest 1/1. Native stereo, compositor and temporal contract fixtures also
pass bounded positive Khronos core validation. These updates are documented with
measured costs and limitations in fusion-evidence.md and vulkan-validation-evidence.md.

Real-camera recording/replay now preserves decoded IMU samples exactly, but
strict frame-interval association still fails for late arrivals; see imu-evidence.md.
Real-camera RGB replay exposed and traced a few float32 half-pixel rounding
differences from the float64 CPU oracle. Exact mask parity at those boundaries
is not claimed. Python virtual-view identity/age export, the integrated native
chain, controlled temporal/adaptive quality evidence, sustained throughput,
physical metric truth, and full pipeline validation remain required open work.
The standalone C++/Slang RHI build and matched 720p benchmark are now implemented
and verified without Python/SGL linkage. See standalone-rhi-comparison.md: output
correctness matches the research implementation, but retained GPU-output wall
time is still 22–25 ms and complete CPU readback is slower. Next work must remove
per-frame host inputs/metadata copies and form GPU-to-GPU stage handoff, then
measure and optimize stereo kernels against the full 16.67 ms frame budget.
This does not qualify a 60 Hz live pipeline.

## Requirements matrix

| Stage or contract | Status at the baseline | Evidence and remaining work |
|---|---|---|
| Shared Slang/Vulkan architecture | Complete for implemented kernels | Python loads shared SGM source and selects Vulkan explicitly (`python/stereoforge/gpu.py:13-18`); native loads matching entry points (`src/runtime.cpp:53-72`) through the shared shader tree (`CMakeLists.txt:37-41`). Four/eight-path cross-host parity tests exist (`tests/test_native_parity.py:7-21`). CPU rectification, consistency, and RGB photometric operations are not shared GPU algorithms yet. |
| **1. Capture and CPU reference** — ZED Open Capture, calibration cache, timestamps, deterministic recording/replay, Python access, SGBM | Implemented; qualification partial | Calibration download/cache and rectification are implemented (`python/stereoforge/calibration.py:51-95`); recording preserves checksums, metadata, and input bytes (`README.md:59-69`). A 600-frame capture measured 59.6973 FPS (`docs/hardware-evidence.md:7-11`). Rectification screening and numerical output do not establish physical metric-depth accuracy (`docs/hardware-evidence.md:30-36`). IMU delivery follow-up is described below. |
| **2. Classical Vulkan stereo** — Census, four/eight-path SGM, subpixel, consistency, confidence, depth, scalar references | Research implementation substantially present; acceptance partial | Scalar and Vulkan stage comparisons cover Census, matching costs, aggregation, disparity, and validity (`tests/test_stereo.py:5-47`). Python performs both stereo directions, consistency, and metric conversion (`python/stereoforge/pipeline.py:107-117`); native metric conversion exists (`src/runtime.cpp:156-173`). These checks establish numerical behavior, not held-out accuracy, completeness by operating condition, or metric surface qualification. |
| **3. Wide RGB-D** — virtual camera, visibility, seams, exposure matching, strict/preview outputs | Partial; required GPU behaviors missing | CPU synthesis implements exposure compensation, confidence seams, compatible-depth blending, and explicitly marked preview (`python/stereoforge/fusion.py:218-250`; `tests/test_fusion.py:108-164`). Baseline Vulkan resolves one nearest owner (`python/stereoforge/shaders/fusion/project.slang:95-118`); GPU preview is rejected (`python/stereoforge/pipeline.py:38`), and native rejects unsupported preview/photometric capabilities (`src/geometry.cpp:180`). GPU tests establish geometric projection and visibility, not the full RGB contract (`tests/test_gpu_geometry.py:27-62`). |
| **4. Temporal reconstruction** — motion, reprojection, occlusion rejection, bounded history, fusion, reset, stabilization | Partial | Supplied motion drives reprojection and bounded fusion (`python/stereoforge/pipeline.py:97-104`, `:120-129`). Tests cover visibility, expiry, ancestry, and resets (`tests/test_temporal.py:12-55`, `:68-90`). Stabilization validates external pose time and clock (`python/stereoforge/pipeline.py:83-86`, `:135-137`). Controlled sequence evidence for temporal benefit, artifacts, recovery, and estimated-motion accuracy remains missing. External motion is consistent with scope; replacing VIO is not required (`README.md:131-135`). |
| **5. Adaptive stereo** — reliable priors, unrestricted uncertain regions, refresh, work accounting, four ablations | Partial; benefit gate unmet | Bounds and periodic refresh exist (`python/stereoforge/temporal.py:116-135`). The runner implements four variants, fixed truth-support completeness, and common-valid accuracy (`python/stereoforge/benchmark.py:57-69`, `:109-136`). Tests use a two-frame static fixture (`tests/test_benchmark.py:7-29`). A held-out sequence campaign with development-frozen quality-loss limits remains absent. Existing microbenchmarks show temporal/adaptive processing slower than independent stereo while aggregation stays dense (`docs/performance-evidence.md:15-21`). |
| **6. Production runtime** — integrated C++ execution, persistent rings, async completion, live applications, profiling, GPU outputs | Partial; integrated pipeline missing | Native stages provide retained GPU buffers, bounded slots, and optional metric depth (`src/runtime.cpp:87-100`, `:159-173`); tests cover queue bounds and retained lifetime (`tests/native_geometry.cpp:29-38`, `:103-108`). Native stereo accepts host image spans (`include/stereoforge/runtime.hpp:30-33`); geometry creates a separate device and accepts host planes (`src/geometry.cpp:74-77`; `include/stereoforge/geometry.hpp:26-38`, `:71-77`). There is no complete same-device native camera/rectification-to-stereo/consistency-to-temporal-to-wide-view chain or demonstrated downstream GPU consumer. Native stereo dispatch lacks the Python pipeline's LR check (`src/runtime.cpp:147-157`; `python/stereoforge/pipeline.py:108-114`). |
| **7. Extensions** — portable neural inference, bindings, ROS 2, interoperability, other vendors | Missing; optional later work | Neural status explicitly reports no qualified model (`python/stereoforge/inference.py:8-12`). Additional integrations/vendors remain open (`docs/implementation-status.md:65-67`). These do not excuse missing required work in stages 3–6. |

The baseline preserves original lens canvases and distinguishes source disparity
from virtual-view depth, strict holes from approximate color, and axial Z from
cylindrical range (`README.md:101-120`; `docs/architecture.md:43-48`). Confidence is
treated as a score and ancestry is conservatively retained rather than claiming
calibrated independent evidence (`docs/architecture.md:19-30`).

**Additional required identity gap:** Python `FusedView` carries virtual depth and
provenance but no transformed per-pixel `source_ids` or `age_ns`
(`python/stereoforge/fusion.py:56-69`). CLI output saves identity/age only for the
source `DepthFrame`, not for virtual-view pixels (`python/stereoforge/cli.py:30-34`).
Camera-source attribution and provenance alone cannot preserve the original depth
observation identity or age after warping. Native geometry's selected-owner metadata
gathering does not close this Python/output contract gap.

## Verification and evidence boundaries

This audit ran **74 targeted CPU tests**, covering fusion, temporal behavior,
pipeline contracts, ablations, calibration, recording/replay, and capture adapters.
It did not run camera or GPU tests. Existing GPU/native test results and physical
measurements cited above are retained evidence, not new executions by this audit.

The baseline performance report contains three measured synthetic frames per
case; GPU timestamps cover only the last stereo pass, not the whole pipeline
(`docs/performance-evidence.md:11-23`). Research adapters still perform transfers,
readbacks, host geometry, and metadata gathering (`docs/performance-evidence.md:27-34`).
Sustained processing, peak-memory evidence, frame-age/drop accounting, and complete
stage timings are not established by those measurements.

The baseline validation test verifies honest rejection when Khronos layers are
unavailable; ordinary native smokes do not establish successful driver validation
(`CMakeLists.txt:47-57`; `docs/implementation-status.md:68-81`). The follow-up below
narrows this remaining gap without qualifying every stage.

## Follow-ups after the baseline

- **Raw IMU delivery is now observed.** The local artifact
  `artifacts/imu-access-recheck-1793102478048966698.json:3-11` records hardware sync
  enabled, 120 camera frames, 1,020 unique increasing IMU timestamps, and zero
  sensor-queue drops. Lines 42–43 report approximately 396 Hz and explicitly limit
  the result to raw delivery. The baseline ledger's IMU-blocked statement
  (`docs/implementation-status.md:54-55`) is stale. IMU extrinsics, calibrated
  motion, and the running IMU recording/replay qualification are not thereby passed.
- **A narrow native stereo driver-validation run has succeeded.** The coordinating
  agent reports obtaining the official Fedora 44
  `vulkan-validation-layers-1.4.341.0-2.fc44.x86_64` RPM with digest/signature checks,
  extracting it locally under `build/validation-layer` without a host installation,
  and successfully running `native_stereo --validation --self-test` on AMD 8060S
  without reported validation warnings. A dedicated artifact is being added.
  This is a follow-up result, not a baseline audit execution or blanket validation
  of geometry, temporal, compositor, resource stress, or the integrated pipeline.
- **Shared GPU compositor implementation is in progress.** This audit does not
  mark its deliverables or parity gates complete before implementation and review.

## Remaining gates, in order

1. **Complete Stage 3's shared GPU compositor.** Add exposure matching,
   confidence-based deterministic seams, compatible-surface blending, and marked
   infinity-plane preview through shared Slang passes and both GPU adapters.
   Preserve depth ordering, unsupported holes, masks, provenance, and requested
   versus observed coverage. Compare against the CPU oracle on exposure,
   confidence, occlusion, thin-structure, missing-depth, and projection fixtures;
   measure incremental transfer-inclusive cost separately from stereo.
2. **Preserve virtual-depth observation identity and age.** Extend Python CPU/GPU
   `FusedView` and exported virtual-view arrays with the selected depth owner's
   original identity and age. Color blending must retain the chosen depth's
   metadata, while unsupported and preview-only pixels must not acquire metric
   identity. Test transformed views, overlapping observations, temporal depth,
   strict holes, and approximate preview without substituting source-grid indices.
3. **Build the Stage 6 same-device native chain.** Compose calibrated capture and
   preprocessing, both stereo directions/LR consistency, metric depth, temporal
   processing, and view synthesis with retained GPU resources and completion.
   Materialize final identity/age metadata without compulsory readback and prove a
   downstream GPU consumer. Preserve stage parity and explicit backpressure;
   standalone host-upload stage APIs are not this integration gate.
4. **Run controlled known-motion and held-out experiments for stages 4–5.** Freeze
   configurations and quality-loss limits on development data before evaluation.
   Run all four ablations on identical inputs/motion and matched evaluation support,
   including dynamic objects, occlusion/disocclusion, thin geometry, low texture,
   rapid motion, and motion loss/recovery. Establish quality or compute benefit;
   candidate-count reduction alone is insufficient. Validate estimated-motion use
   separately from known motion and raw IMU delivery.
5. **Complete sustained performance, memory, and driver-validation qualification.**
   Measure both stereo passes and temporal/view stages, transfers, waits, CPU
   overhead, throughput, frame age, drops, and peak memory on the intended camera
   configuration. Stress resource reuse and lifetimes under successful Khronos
   validation. Extend the narrow stereo validation follow-up to every required
   stage and the integrated chain. Keep cold compilation separate from steady state.
6. **Close physical geometry and release gates.** Validate metric depth by range,
   completeness, edge behavior, and surface reconstruction against independent
   ground truth; numerical positivity and rectification residuals are insufficient.
   Publish an AMD classical release only with its demonstrated operating envelope.
   Keep neural models, other vendors, ROS 2, and binary redistribution as separately
   qualified extensions.

The next bounded implementation is gate 1: its CPU oracle and behavioral fixtures
already exist, and it closes an earlier required capability. Starting full native
integration first would carry an incomplete RGB contract into that interface;
optional neural work does not address the outstanding classical acceptance gates.
