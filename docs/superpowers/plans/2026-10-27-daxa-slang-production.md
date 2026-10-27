# Daxa + Slang Production Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and qualify an AMD-first C++20/Vulkan stereo pipeline using Daxa TaskGraph and reusable Slang operations, preserving the approved seven-stage StereoForge output and validation contracts.

**Architecture:** Daxa is the proposed production owner of the Vulkan device, resources and TaskGraph; it is an alternative to the existing Slang RHI runtime, not a layer on that runtime. Retain the working RHI path and its byte-bound fixtures as the frozen migration/comparison reference until the Daxa foundation and matched proof of concept pass. SlangPy remains optional research; ncnn Vulkan is an optional later model component.

**Tech Stack:** Linux, C++20, ZED Open Capture, Vulkan, pinned Daxa + Linux Slang SDK, CMake/Ninja; optional ncnn after classical qualification. No CUDA, HIP, ROCm or TensorRT.

**Spec:** The user-approved seven-stage plan and its contracts in [requirements-audit.md](../../requirements-audit.md), [architecture.md](../../architecture.md), and the three 2026-10-27 attachments: “High-performance GPU kernel development strategy,” “Automatic kernel fusion and GPU execution graphs,” and, superseding their framework choices, “StereoForge: simplify the GPU stack.” This document records the resulting decisions so execution does not depend on local attachment paths.

## Global Constraints

- Production target: **C++20 + Daxa TaskGraph + Slang**, with Daxa as the one GPU owner for the classical/temporal/view pipeline. This is a migration decision and plan; Daxa is not installed, adopted or qualified by this document.
- No custom Vulkan abstraction, task graph/synchronization scheduler, arbitrary fusion compiler, general GPU auto-tuner or custom neural execution engine. No parallel production RHI/Daxa device owners. An optional ncnn component may own a separate Vulkan context only behind an explicit owned-copy/completion boundary; this does not block the classical pipeline.
- Daxa schedules declared work and barriers; it does not automatically fuse arbitrary shaders. Compose small Slang operations into manually authored fused/unfused entry points.
- Keep reusable math/sampling/photometry independent of Daxa bindings. Add only files needed by the next tested slice; no wholesale directory scaffold, mass rename or shader rewrite.
- Preserve acquisition/calibration, stereo, motion, temporal, adaptive, RGB synthesis and output boundaries. Preserve full unrectified post-ISP source RGB for wide synthesis even when the rectified stereo canvas crops it.
- Outputs remain `left_rgb`, `right_rgb`, `rectified_rgb`, `wide_rgb`, strict `wide_rgbd`, and timestamp-bound `stabilized_wide_rgb`. Strict/preview behavior, separate RGB/depth masks, source attribution and explicit approximation remain unchanged.
- Disparity stays in the source rectified camera. Perspective virtual depth is axial Z; cylindrical virtual depth is ray range, both metres. Missing calibration blocks metric depth and calibrated views.
- Preserve validity, confidence score, provenance, signed 64-bit original observation ID (−1 invalid), age in nanoseconds, device identity, ownership and completion. Historical reuse cannot become independent evidence; confidence is not calibrated uncertainty.
- Wide RGB-D precedes temporal/adaptive migration. Supplied validated motion is sufficient; replacing VIO/SLAM is outside scope. Neural inference, ROS 2 and other vendors do not block classical releases.
- Start with one main compute-capable queue. Query subgroup capabilities, memory types/coherency and limits; never assume wave32, zero-cost UMA mapping or automatic cross-frame synchronization.
- Target sustained **1280×720 per eye at 60 fresh frames/s**, budget **16.67 ms/frame**, subject to actual capture and full-pipeline qualification. Existing measurements are not that result.

## Review Focus

- Framework/compiler support or external TaskGraph resource registration may fail on this exact RADV/SDK combination: fail the foundation gate before migration (Tasks 1–2).
- Delayed consumers, wraparound or calibration/resolution changes may reuse live storage: test bounded rings, stale handles and history ownership across frames (Tasks 2, 4, 7).
- Nearby occluders, lens distortion and missing depth may turn color coverage into fabricated geometry: retain strict holes and approximation metadata (Task 6).
- Unknown pose clocks, motion loss or shared observation ancestry may corrupt temporal confidence: reject/reset and recover without double counting (Task 7).
- A faster variant may alter ties, numeric precision, quality or memory requirements: retain exact references, resource ceilings and frozen held-out gates (Tasks 3, 5, 8).

---

## Existing evidence and migration boundary

At `451040e`, the standalone RHI runtime already supports a shared device/queue, retained GPU geometry inputs and final GPU observation IDs/ages. Matched static 720p fusion measured **1.427 ms strict / 1.468 ms preview p50**; source upload was outside those calls. A synthetic real GPU stereo → bridge → compositor → consumer fixture verifies fresh outputs and lifetime/reuse using 16-byte final counters. See [native-resident-evidence.md](../../native-resident-evidence.md).

The subsequent optimized selection implementation is committed as `0c1b958` with **294 passing Python tests**: D128 select-only GPU median improved approximately **31.05 → 5.68 ms** with matching outputs. Carry that validated shared algorithm forward, preserving its exact shader/compiler/report identities. It is a causal kernel result, not full stereo. Neither result proves the complete 16.67 ms budget.

The current RHI shaders, fixture manifests and benchmark binaries are reference evidence. Daxa migration must not discard them or silently attribute their performance to Daxa. Optional Python research remains available; the production executable must build and run without Python, SlangPy or SGL.

## Foundation facts that require explicit gates

The upstream [3.6 release](https://github.com/Ipotrick/Daxa/releases/tag/3.6) resolves to commit `00761e80c1dd6d1972d5f4569ce49620a7b4d384`. Pin full source/dependency commits and archive hashes, not only a version string: upstream CMake's project version is stale. Daxa's [MIT license](https://github.com/Ipotrick/Daxa/blob/3.6/LICENSE) is permissive; transitive notices and binary redistribution still require an inventory.

**External-resource blocker:** the official [TaskGraph documentation, “External Task Resources”](https://docs.daxa.dev/wiki/taskgraph-bottom-up/#7-external-task-resources), warns about malformed views from external resource registration. The affected paths exist in the pinned [3.6 implementation](https://github.com/Ipotrick/Daxa/blob/3.6/src/utils/impl_task_graph.cpp#L792-L811), unlike its designated task-buffer creation path. Before registering application-owned resident buffers, use a verified fixed upstream revision or a minimal versioned patch plus a GPU regression. Do not work around this by introducing another graph or importing the RHI device into Daxa.

**Linux/compiler blocker:** [3.6 dependency bootstrap](https://github.com/Ipotrick/Daxa/blob/3.6/cmake/deps.cmake#L88-L169) defaults to a Slang 2025.11 Windows x86-64 archive. It is not a working Linux dependency strategy, and compatibility with StereoForge's current Linux Slang 2026.12 SDK is unproven. Select an explicit Linux SDK target, checksum and compiler compatibility test before claiming a supported build.

The [required feature implementation](https://github.com/Ipotrick/Daxa/blob/3.6/src/impl_features.cpp#L204-L238) and actual device creation are the authority for Vulkan 1.3, buffer device address, descriptor indexing, variable pointers, shaderInt64, synchronization2, timeline semaphores, subgroup-size control and scalar-layout requirements. A GPU-family marketing list is not qualification. [TaskGraph resource documentation](https://docs.daxa.dev/wiki/taskgraph-bottom-up/) distinguishes transient content from retained history; callbacks execute on graph execution, so graph reuse is not proof of zero CPU work. Only the main queue is the initial contract; cross-queue use needs explicit submission boundaries and separate qualification.

Target, preset and file names below are proposed StereoForge additions, not names
claimed to exist upstream. Confirm pinned Daxa API spellings in Task 1 before
writing bindings; do not infer compatibility from the current RHI API.

## Task 1: Qualify the pinned Daxa foundation on AMD

**Files:** Create `cmake/Daxa.cmake`, `apps/daxa_probe.cpp`, `tests/daxa_foundation.cpp`, `docs/daxa-foundation-evidence.md`; modify only the opt-in Daxa section of `CMakeLists.txt`, `CMakePresets.json`, and dependency rows/notices in `docs/license-evaluation.md` and `third_party/README.md`.

**Interfaces:** `STEREOFORGE_DAXA=OFF` by default; a `daxa` preset enables an independent Daxa executable without linking `stereoforge_runtime`. `daxa_probe --output FILE` emits exact versions, required/available features, subgroup sizes, resource limits, memory/coherency choices and actionable missing-feature errors.

- [x] Write failing `daxa_foundation_contracts` cases for the pinned SDK/build configuration and externally registered resident-buffer views. The GPU regression writes/reads a Daxa-created application buffer across graph executions, validates its view and generation, and checks tiny final counters under Khronos validation.
- [x] Pin Daxa full commit/source hash and an explicit Linux Slang SDK. First test the existing 2026.12 Linux SDK deliberately; if incompatible, record the failure and test a separately pinned Linux SDK supported by Daxa. Preserve compiler identity in every comparison; do not silently use its Windows bootstrap or RHI-linked compiler.
- [x] Resolve the documented external-resource blocker with a verified upstream fix or one minimal checked-in patch and its regression. Record the original failing revision; this is a dependency fix, not a new StereoForge resource framework.
- [x] Add the probe/build wiring, source/transitive license inventory and notices; query actual AMD capabilities and UMA coherency rather than hardcoding support.
- [x] Run `cmake --preset daxa`, `cmake --build --preset daxa`, then `ctest --test-dir build/daxa -R daxa_foundation --output-on-failure`. Require actual Daxa/Slang device execution and clean core validation; unavailable features/build support remain a recorded blocker. A skipped native test is not a pass.

**Gate:** No production migration or externally registered history/ring resources until Linux build, feature, license and external-view checks pass. If they fail, retain the working RHI reference and record the specific blocker; do not claim Daxa adoption.

Verified 2026-10-27: the bounded foundation gate passes; see [foundation evidence](../../daxa-foundation-evidence.md). This does not qualify the Task 2 compositor or the complete production chain.

## Task 2: Match the resident RHI compositor in a small Daxa POC

**Files:** Create `apps/daxa_view_benchmark.cpp`, `tests/daxa_view.cpp`, `benchmarks/compare_daxa_view.py`, `python/stereoforge/shaders/ops/projection.slang`, and `python/stereoforge/shaders/daxa/view.slang`; extend the opt-in Daxa CMake section only. Existing `shaders/` aliases this canonical shader directory; do not create a second shader tree.

**Interfaces:** `daxa_view_benchmark --fixture DIR --mode strict|preview --frames N --warmup N --output FILE --dump-prefix PREFIX` consumes the existing resident fixture format. Daxa owns its device/buffers; comparison with `native_view_benchmark --input resident --readback gpu` uses separate processes, identical input bytes and declared source/compiler identities. Reusable operation functions contain no Daxa descriptor declarations; the Daxa entry wrapper supplies bindings.

- [x] Freeze the currently passing RHI fixture/shader/compiler/report hashes and add failing Daxa tests for all eleven output planes, strict/preview behavior, 64-bit IDs/ages and unsupported holes. Keep the existing float32 half-pixel contract explicit; do not relax exact comparisons on a matching execution profile.
- [x] Port only the projection/compositor slice into a Daxa TaskGraph using reusable operations and thin wrappers. Retain Daxa-owned persistent inputs/outputs; declare each pass's resource reads/writes and required usage precisely. Use the documented `create_task_buffer` path where appropriate; application-owned resident registration first passes Task 1's fixed-view regression. No foreign RHI resources/device imports.
- [x] Exercise one, two and three ring generations, delayed consumers, graph reuse, external-buffer registration, attempted overwrite and resolution/calibration changes. Retain outputs until completion; recycled resources must never corrupt a pending consumer. Use documented persistent/double-buffer lifetime semantics for retained state, with explicit completed-frame bookkeeping across executions; within-graph barriers alone do not prove safe next-frame reuse. Transient aliasing must exclude persistent inputs/history/live outputs.
- [x] Run the matched 720p resident benchmark, setup cost separately, at least 3 warmups and 30 samples; report GPU timestamps, CPU callbacks/submission, waits, memory and CPU dumps outside the GPU-output measurement. Require exact equality of all eleven Daxa/RHI fixture planes, including RGB, depth and confidence, on the shared execution profile. The current 720p fixture already matches all eleven exactly; do not introduce a one-intensity or new floating-point allowance for migration. A compiler-change disagreement requires a diagnostic artifact and resolution before the gate passes. Separately retain the documented float32-versus-float64 camera half-pixel boundary; it is not a waiver for this matched fixture.
- [x] Run `ctest --test-dir build/daxa -R 'daxa_(foundation|view)' --output-on-failure` with core validation and a separately reported synchronization-validation run. Compare against RHI without assuming Daxa is faster. Preserve regressions and fail the POC if correctness, lifetime, support or agreed resource/latency ceilings fail.

**Gate:** Record an explicit POC decision and measured operating envelope. The production target remains Daxa, but cutover requires a passing foundation/POC; a slower or unsupported result is evidence to resolve, not permission to relabel a migration as complete.

Verified 2026-10-27: the immutable resident-view POC passes exact eleven-plane equality, bounded lease tests and separate core/synchronization validation. CPU enqueue cost is lower; wall latency has no consistent advantage across modes. See [view evidence](../../daxa-view-evidence.md). Incremental migration may proceed; full production cutover and live 60 Hz remain unqualified.

## Task 3: Establish shared operation contracts and curated variants

**Files:** Create only needed functions in `python/stereoforge/shaders/ops/photometry.slang` and `ops/stereo_output.slang`; add `configs/kernel-variants.json` and `tests/daxa_variant_contracts.cpp`; extend the existing bounded benchmark helpers instead of building a tuner service.

**Interfaces:** Each operation declares its grid, units, precision, border/sampling rule, input/output layout and alias permissions. Manually maintained variant names select fused/unfused entry wrappers. A manually approved config selects the validated variant; cache keys include GPU/device/driver, Daxa/Slang versions, shader hashes, specialization, dimensions, D/path count, precision and quality profile.

- [ ] Add failing cache-invalidation/config tests for changed driver/compiler/shader/shape and an unsupported subgroup size. Reject unknown variants and stale results rather than silently reusing a performance choice.
- [ ] Extract operations incrementally as their next native task needs them; keep the RHI baseline frozen as a comparison reference, not a second production backend receiving new features.
- [ ] Implement a short explicit variant list and offline comparison command that writes measurements. Select a passing winner through manual config review; no arbitrary graph partitioner, online policy learner or general auto-tuner.
- [ ] Keep an unfused baseline for every fusion. Use Slang modules, statically resolvable generics/link-time specialization and compiler/pipeline caches, without a custom compiler; benchmark compatible locality/workgroup variants 8×8, 16×8, 16×16 and 32×4, using GPU timestamps first and optional RenderDoc/Mesa RGP/RGA only where actually supported. Measure register allocation, LDS, spills and memory traffic where tooling exposes them; record unavailable counters honestly. Accept fusion for measured locality/traffic benefit within resource ceilings, not dispatch-count reduction. Re-run exactness and transfer-inclusive timing after each accepted change.

## Task 4: Native capture and fused preprocessing

**Files:** Add `include/stereoforge/capture.hpp`, `src/capture.cpp`, `include/stereoforge/daxa_frame.hpp`, `include/stereoforge/daxa_pipeline.hpp`, `src/daxa_pipeline.cpp`, `apps/daxa_camera.cpp`, `python/stereoforge/shaders/daxa/preprocess.slang`, and `tests/daxa_preprocess.cpp`; reuse the pinned Open Capture dependency and existing calibration/cache contracts.

**Interfaces:** `CaptureFrame` exposes owned packed UVC pixels, camera/control/sensor timestamps and calibration identity; `DaxaPipeline::enqueue(CaptureFrame)` returns a bounded-slot frame lease with completion and GPU handles. No Python or SlangPy dependency in this executable. One chosen Daxa device/queue owns all subsequent GPU resources. `daxa_frame.hpp` defines a Daxa-specific GPU view carrying the pinned SDK's concrete buffer handle, byte offset/size/stride, unique context identity, retained allocation/frame lease and Daxa completion primitive. It must not reinterpret the existing RHI `BufferView.native` pointer or pretend a host token is a cross-runtime semaphore; GPU consumers reject foreign contexts and stale generations.

- [ ] Add failing recorded-pair tests for YUYV conversion, distorted calibration, rectification borders, luma values, original source RGB coverage, missing calibration and clock changes. State conversion/color coefficients and rounding in the reference; do not silently change existing matching inputs.
- [ ] Implement asynchronous bounded capture with honest drop/backpressure counters and lossless deterministic replay. Preserve control/IMU metadata; late IMU samples remain a separate association issue, not a validated pose.
- [ ] Compare two-pass versus manually fused format conversion/rectification/luma. Preserve the full original left/right RGB canvases as separate view-synthesis inputs; stereo crops cannot become the only RGB storage. Precompute calibration-only maps and upload on configuration changes.
- [ ] Measure staging versus suitable host-visible UMA allocations, including flush/invalidate/coherency and ownership. Choose by measured complete cost; no assumed zero copy. TaskGraph declarations handle within-graph dependencies; ring/history reuse additionally follows completed-frame ownership.
- [ ] Run native recorded preprocessing tests, then a bounded physical capture screening. Report unique input frames, timestamps, drops and frame age separately from repeated processing of a retained fixture.

## Task 5: Classical stereo and local output fusion

**Files:** Add `python/stereoforge/shaders/daxa/stereo.slang`, extend `ops/stereo_output.slang` and `src/daxa_pipeline.cpp`, add `tests/daxa_stereo.cpp` and `benchmarks/compare_daxa_stereo.py`. Keep scalar references and archived RHI comparison sources unchanged; reusable optimized selection operations originate from `0c1b958`.

**Interfaces:** Both stereo directions produce rectified disparity/confidence/validity; LR consistency and explicit calibrated depth conversion produce source-grid metric observations with original ID, age 0 and provenance. Virtual-view depth never masquerades as disparity.

- [ ] Add failing exact Census/cost/SGM tests, including D32/64/128, four/eight paths, ties, disparity-zero/upper limits, subpixel curvature, invalid borders and calibration principal offset. Reuse existing scalar fixtures; integer stages remain exact.
- [ ] Port the classical tasks incrementally on the single Daxa graph. Compute both references and LR consistency before accepting metric observations. Compare separate selection/subpixel/confidence/depth/masks versus a manually fused output entry point; LR dependencies across reference views remain explicit tasks when needed.
- [ ] Measure a bounded set of SGM memory/barrier variants: current dense costs versus narrower exact integer storage after overflow proof, materialized versus on-demand costs, and subgroup/LDS reductions with queried capabilities. The existing invalid cost sentinel 10000 cannot fit raw uint8, and eight-path accumulated sums can exceed uint16; prove bounds and any separate mask/sentinel encoding before compacting, leaving the exact reference unchanged. Keep directional recurrence/occlusion semantics; do not fuse all paths by assumption or hardcode a wave width.
- [ ] Run exact/native/scalar parity and core/synchronization validation before timing. Record all GPU stages, both directions, CPU work, transfers, waits and resource peaks. Keep D128/four-path/subpixel/LR quality profile explicit; eight paths remain a separately measured quality mode. Changing D/paths/precision cannot silently satisfy a faster budget.

## Task 6: Production wide RGB-D before temporal/adaptive

**Files:** Promote the tested POC bindings into `src/daxa_pipeline.cpp`; extend `tests/daxa_view.cpp` and add output cases to `apps/daxa_camera.cpp`. Add no second compositor or generic image graph.

**Interfaces:** Preserve `VirtualCamera` pose relative to the rig, resolution, perspective/cylindrical projection, requested FOV, pose timestamp, masks and coverage. Derive default canvas resolution/FOV from calibration, expose both in configuration, and default to rig alignment centered between lenses. Original source RGB and supported source-aligned metric geometry feed view synthesis; right depth is explicit right-reference estimation or visibility-checked left projection.

- [ ] Test identity views, planes, near/far overlap, thin foregrounds, exposure changes, occlusion, absent depth, distorted lens coverage, virtual translation and cylindrical boundaries. Repeat exact GPU metadata checks through a downstream Daxa GPU consumer and exported CPU views.
- [ ] Wire strict/preview outputs using the established operation/visibility/seam/exposure contract. Preview may fill unsupported RGB using explicitly marked infinity directions; it must preserve invalid depth, ID −1, age 0 and zero confidence/provenance. Never blend incompatible surfaces or fabricate disocclusion depth.
- [ ] Report requested FOV, observed RGB coverage and measured depth coverage separately; no doubled-FOV promise. Record synthesis's incremental cost with RGB disabled in the primary stereo comparison. Close this task's geometry/mask/performance gate before starting Tasks 7's temporal/adaptive production work.

## Task 7: Known-motion temporal processing and adaptive ablations

**Files:** Add `python/stereoforge/shaders/ops/temporal.slang`, `python/stereoforge/shaders/daxa/temporal.slang`, `tests/daxa_temporal.cpp` and `benchmarks/compare_daxa_ablations.py`; extend only the needed history fields in `src/daxa_pipeline.cpp`.

**Interfaces:** Supplied motion includes pose convention, timestamp and clock domain. History has a bounded age/count and explicit retained ownership. A selected measurement keeps original observation identity and age; fusion conservatively tracks shared ancestry instead of treating reused estimates as new independent evidence.

- [ ] Write known-motion tests first: occlusion/disocclusion, stale history, motion loss/recovery, calibration/clock reset, late frames, duplicated ancestry and delayed consumers. Test persistent/double-buffer history across graph executions; transient data cannot carry history.
- [ ] Port reprojection, depth ordering, confidence gating and bounded fusion. Compare separate global visibility resolution plus gather fusion against a local fused variant only where dependencies permit it. Orientation stabilization follows validated motion; translated virtual views require valid translation and retain holes.
- [ ] Add adaptive search bounds only for reliable fresh priors, full search in uncertain regions, periodic full refresh and separate candidate/aggregation work counts. Do not claim speedup from reduced matching bounds while aggregation remains dense. Optional GPU-generated indirect dispatch requires measured benefit; `VK_AMDX_shader_enqueue` is not a required dependency.
- [ ] Freeze quality constraints on development sequences; run independent full SGM, full SGM+fusion, adaptive matching, and adaptive+fusion on the same held-out bytes/motion. Disable RGB synthesis for the main comparison and report its incremental cost separately. Report completeness, metric/surface error, temporal stability and recovery; validate estimated motion separately from supplied motion.

## Task 8: Sustained classical acceptance and release envelope

**Files:** Add `configs/amd-720p-qualification.json`, `benchmarks/qualify_daxa_pipeline.py`, `tests/daxa_pipeline.cpp`, and `docs/daxa-classical-evidence.md`; update the requirements/implementation ledgers only from new evidence.

**Interfaces:** The qualification config fixes resolution, FPS, D, paths, subpixel/LR settings, camera controls, strict/preview output, history/ring caps, memory budget, timestamp/frame-age/drop limits and quality thresholds before held-out execution. Freeze the configured wide-output dimensions as well as full 1280×720-per-eye input; downsampling or changing D/paths/quality is a different profile, not a way to pass the same target. Record input/calibration, dependency, shader and config hashes. Freeze frame-age caps on development data before held-out execution; do not invent them from the test results.

- [ ] Add failing lifetime/stress tests for ring exhaustion, consumer retention, slow output, aliasing, allocation growth and resource replacement. Keep persistent history/live outputs out of transient aliases. Freeze a bounded freshness/drop/age policy; out-of-order temporal inputs reset/reject rather than entering history. Query-specific resource ceilings are part of the fixed profile.
- [ ] Run recorded and physical 720p-per-eye/60-FPS input campaigns with unique fresh frames; start with a 10-minute sustained run and disclose warmup/compilation separately. Measure offered/delivered/accepted/completed/dropped IDs, throughput and p50/p95/p99 frame age/latency, not repetitions of one frame or a sum of unrelated stage percentiles. Report source delivery separately: the prior approximately 59.7 FPS capture does not become 60 fresh live frames/s because processing capacity exceeds 60 or replay is offered at 60. Record timestamp domains and validate camera/host alignment before camera-to-output latency or frame-age claims. If capture-clock alignment is unverified, report host-ingress-to-completion latency and camera age as unknown; never combine unrelated clocks.
- [ ] Measure GPU stages, CPU capture/control/submission, transfers, waits, output work and peak/steady memory independently. Require the declared 16.67 ms budget and predeclared latency/drop/memory/quality limits; unmet gates remain failures. Profile before adding queues, and qualify any later queue change independently.
- [ ] Run positive core and synchronization validation on the complete chain and lifetime stress. Obtain independent physical geometry/surface truth; freeze development thresholds before held-out assessment. Publish only the demonstrated AMD classical envelope; other vendors and binary redistribution remain separate gates.

## Task 9: Optional inference and downstream integrations

**Files, only after a demonstrated need:** `src/ncnn_refinement.cpp`, `tests/ncnn_refinement.cpp`, a single versioned model/config artifact and its license/conversion evidence. ROS 2/Python runtime bindings and vendor qualification are separate incremental tasks.

- [ ] Select one compact refinement/confidence model with acceptable license and supported Vulkan operators. Compare ncnn conversion/numerics against its original model and measure transfer-inclusive benefit under the frozen classical quality constraints.
- [ ] Start ncnn as an optional separate component with explicit owned copies and completion/fence boundaries, measuring the complete transfer/synchronization cost. It may own a separate Vulkan context if needed and documented; never alias Daxa buffers across contexts or claim zero copy. Prefer verified shared-device interoperability only after a measured boundary cost justifies it. Neural context support is not a classical release gate, and no RHI/Daxa stacking or custom inference engine is introduced.
- [ ] Evaluate IREE only if a named model/operator/performance need is demonstrated that the chosen ncnn path cannot meet. Do not build multiple inference runtimes, handwrite a general NN engine or expect an ML compiler to fuse arbitrary external Slang dispatches.

## Execution order and completion rule

Start with Task 1's dependency/API/feature/license regressions, then the small
matched Task 2 proof of concept. Only a passing POC permits the incremental
production migration. Keep the original sequence: capture/reference → classical
stereo → wide RGB-D → temporal → adaptive → full production qualification →
optional extensions. The POC and shared-operation preparation are prerequisites,
not a claim to have completed those stages. Profiling begins in the first POC and accompanies every native slice; it does not wait for ROS 2 or inference. Each task uses watched failing tests,
minimal implementation, an independent performance review before GPU validation,
validation/measurement, and a scoped reviewable commit. Do not stop at an API
scaffold or infer completion from numerical fixtures alone.

## Task 5 continuation: calibrated dual-reference observations

This continues the existing approved production task at `404b42a`; it does not
replace wide RGB-D, temporal/adaptive work or sustained 60 Hz qualification.

### 5a. Retain validated setup geometry

Files: new `include/stereoforge/daxa_calibration.hpp`,
`src/daxa_calibration.cpp`, `tests/daxa_calibration.cpp`; extend only setup parsing
in `src/daxa_preprocess.cpp`, its header and optional CMake file.

Interface: `DaxaCameraModel` retains dimensions, double fx/fy/cx/cy and distortion;
`DaxaStereoCalibration` retains two models and a double row-major right-to-left
pose. `DaxaRectifiedGeometry` retains original/rectified calibrations, R1/R2/P1/P2
and both identity strings. `DaxaRectification::geometry` is optional value data
inside the existing immutable owner. `metric_calibration()` validates the complete
binding and returns float MetricCalibration at setup; absence rejects metric
processing while existing manually supplied 2D maps remain supported.

- [x] Watch missing-payload and inconsistent-geometry CPU tests fail first.
- [x] Validate proper rotations, R1*R*R2^T=I, R1*t=(baseline,0,0), rectified
  horizontal positive-x pose, zero rectified distortion, compatible epipolar
  intrinsics, P pinholes, P1 last column zero and P2 x translation=-fx*baseline.
  Retain existing JSON hashes, dimensions and rigid-pose bounds; publish explicit
  comparison tolerances rather than silently accepting nominal parameters.
- [x] Load all geometry without changing calibration serialization/IDs/maps.
  No per-frame matrix validation or allocations. Test nonidentity original
  rotation, nonzero principal offset, malicious rebound matrices, float overflow,
  missing geometry and manually supplied 2D configurations.
- [x] CPU/offline build only; independent review and performance pass before GPU.

### 5b. Shared-context dual stereo, LR and metric materialization

Files: new `include/stereoforge/daxa_depth.hpp`, `src/daxa_depth.cpp`,
`python/stereoforge/shaders/ops/stereo_observation.slang`,
`python/stereoforge/shaders/daxa/depth.slang`, `cmake/DaxaDepth.cmake`,
`tests/daxa_depth.cpp`, `apps/daxa_depth_benchmark.cpp`. The preprocessing CMake
owner adds the optional include hook after this interface is ready.

Interface: `DaxaDepthConfig` contains StereoConfig, lr_tolerance=1.0, slots and
required subgroup size. `DaxaDepthPipeline` owns preprocessing, left/right
PackedSpecialized engines and one bounded observation materialization graph on
one shared DaxaFoundation. It consumes CaptureLease and returns a retained typed
frame with completion, calibration owner and original source metadata. Original
RGB remains GPU-resident. No RHI BufferView reinterpretation or second device.

- [x] Watch CPU LR/metric/metadata and lifetime tests fail before implementation.
- [x] Preserve positive disparity x_left-x_right. Left LR uses
  floor(x-dL+0.5), right LR uses floor(x+dR+0.5), target bounds before loads,
  target validity and absolute disparity difference <= tolerance. Right symmetric
  checking is an explicit additional source-layer policy. No confidence threshold.
- [x] Keep LR validity/disparity/confidence separate from metric validity. LR
  rejection writes canonical NaN disparity and zero stereo confidence. Metric Z
  is axial metres: fx*baseline/(d-(cxL-cxR)), finite and positive only.
- [x] Explicit strict metric-support policy additionally requires source remap
  support and nearest-correspondence remap support. It does not modify matching
  or the raw LR mask. Report the policy; no silent comparison with the older
  pipeline's less restrictive metric mask.
- [x] Final metric-invalid depth is canonical NaN, confidence/provenance zero,
  source ID -1 and age zero. Valid fresh observations retain exact signed-int64
  CaptureLease ID, provenance STEREO=1, age zero. Reject IDs>INT64_MAX rather than
  truncating; retain clocks/timestamps/control/IMU associations without pose claims.
- [x] Concatenate left then right observational planes for later DaxaView intake.
  Keep original and rectified RGB accessible through the retained preprocessing
  frame; do not combine rectified depth with raw rays.
- [x] Allocate slots/graphs/parameters at setup. Do not retain constructor
  prototypes. Whole-chain admission rejects a retained/full output ring before
  uploading/dispatching another source. Producer RGB/stereo leases remain owned
  through consumers; collect complete unleased output inputs, then stereo inputs,
  then preprocessing sources before next capture/admission. No intermediate CPU
  wait/readback or image processing, no per-frame application heap allocation.
- [x] Native GPU tests (root only) isolate LR using literal supplied float arrays,
  including half rounding, tolerance equality/nextafter, NaNs, missing support,
  occlusion and principal-offset planes. Exact integer/masks/64-bit metadata and
  canonical invalid defaults are mandatory. No blanket CPU compound-float
  tolerance. Use exactly representable planes and independent frozen GPU depth
  arithmetic; keep unresolved general scalar float qualification explicitly open.
- [x] Test dynamic different-content/same-ID inputs, wrong contexts/owners/reference
  labels, source pairing, full rings, delayed consumers and producer destruction.
- [x] Measure actual combined upload/preprocess/two references/LR/depth/metadata
  graph cost and host wall; include CPU collection/query/waits. RGB synthesis
  stays disabled. A sum of separate-reference medians is not the result.

### 5c. Qualification and subsequent integration

Independent functional and performance passes precede root GPU execution. Bind
all sources, compiler, binaries, SPIR-V, fixtures and reports. Preserve previously
qualified 25 stereo and two preprocessing SPIR-V files if unchanged; qualify any
necessary change explicitly. Run core/synchronization and bounded lifetime stress,
then 720p measurements. Commit/push only the demonstrated envelope. Next, extend
the existing DaxaView to accept this typed shared-context frame, then implement
rectified-to-raw calibrated depth/ray conversion for complete original RGB
coverage. Full live qualification and the original seven-stage scope stay open.

Verified 2026-10-27 for this bounded slice: independent functional/performance
reviews clear; native CPU contracts 4/4 and Python tests 812 pass. Native GPU
core/synchronization contracts, dynamic profile checks and 300+3 static stress
pass. Independent frozen-RHI comparison is exact for all nine observation and
six raw stereo planes on small core/sync, 720p core/sync and dynamic P8/wave64
cases. Actual 720p/D128/P4/wave32/two-slot queue span is 31.683 ms mean and host
wall is 32.116 ms mean (30+3, no validation/readback). Requested combined buffers
are 840,500,528 bytes. This is not live 60 Hz or general scalar compound-float
qualification. See [depth evidence](../../daxa-depth-evidence.md). Task 6 shared
context/compositor intake and correct association of rectified geometry with
original RGB remain next; the full goal stays active.


## Task 6 continuation: retained depth intake and calibrated wide RGB views

**Approved architecture (2026-10-27):** extend the tested `DaxaView`; keep
metric observations on their rectified geometry grid and retain full original
RGB in a separate preview namespace. This supersedes the proposed
rectified-to-raw depth conversion in 5c. There is no second compositor, SDK
change, rounded raw-depth canvas, or permission to stop after an API shim.
Task 6a is an integration prerequisite; Task 6b supplies the calibrated grid
semantics. End-to-end Task 6 acceptance requires both and the existing strict,
preview, peripheral-coverage and full-chain 60 Hz goals remain unchanged.

### Ownership and interfaces

One integration owner changes `include/stereoforge/daxa_view.hpp`,
`src/daxa_view.cpp`, `include/stereoforge/daxa_view_parameters.hpp`,
`tests/daxa_view.cpp`, new `tests/daxa_view_depth.cpp`, and the existing view
build block in `CMakeLists.txt`. That owner also implements the calibrated
setup and narrow shader plumbing in those files and new
`include/stereoforge/daxa_view_calibration.hpp`,
`src/daxa_view_calibration.cpp`, `shaders/daxa/view_depth.slang`, plus the
necessary reader specialization in
`python/stereoforge/shaders/ops/projection.slang`. Preserve the existing
`shaders/daxa/view.slang` fixture wrapper where possible; explicitly requalify
any changed legacy SPIR-V. Do not duplicate the visibility/resolve compositor.

An independent oracle owner creates `benchmarks/compare_daxa_view_depth.py`
and `tests/test_daxa_view_depth_comparison.py`; the integration owner creates
`apps/daxa_view_depth_benchmark.cpp`. An independent performance reviewer owns
only `tests/test_daxa_view_depth_hotpath.py` and an ignored review report,
reads production changes, and does not build or launch GPU workloads. Shared
files have one writer; owners coordinate signatures before body edits.

Add the following overloads to the existing class, using the existing depth
header and value types:

```cpp
DaxaView(GeometryConfig, VirtualView const&,
         std::shared_ptr<DaxaFoundation>,
         DaxaDepthPipeline::Frame const& prototype, unsigned slots=2);
Frame project(DaxaDepthPipeline::Frame const&);
size_t collect();
bool can_project() const;
```

The old `GeometrySources` constructor, `project()` and fixture controls remain
supported and independently tested. The supplied foundation determines the
actual device identity. The new constructor validates and caches immutable
setup from the prototype, then releases the prototype itself. `project(frame)`
consumes only matching typed depth frames and retains that exact frame until
its consumer slot retires. `collect()` reclaims completed, unleased slots;
`can_project()` is a nonblocking admission snapshot on the single submission
thread. The chain calls collection/admission before capture acquisition and
upstream upload, with no intervening competing submission. It must not discover
view exhaustion only after spending an upstream upload/stereo operation.

### 6a. Shared foundation and typed retained intake

- [ ] Write CPU contract tests named `shared_foundation_typed_intake`,
  `foreign_context_and_map_owner_reject`, `constructor_prototype_released`,
  `retained_source_metadata_per_slot`, and
  `view_ring_rejects_before_upstream_upload`. Assert actual device identity,
  exact immutable calibration-owner pairing, rectified dimensions/layers and
  plane byte sizes, different-content/same-ID generations, timestamp/calibration
  metadata per slot, producer destruction safety, full/delayed rings, and no
  prototype lease retained after setup. Watch these tests fail before changes.
- [ ] Convert the implementation's foundation storage to shared ownership.
  The legacy constructor creates its own foundation; supplied shared mode never
  closes another stage's context. View resources retire before their retained
  foundation. Preallocate slot state, external task resources, graph bindings,
  parameters and lease state at setup; avoid the old per-projection
  `make_shared<Lease>` in the resident path. Store source metadata per slot,
  not engine-wide. Update only bounded resource addresses and scalar frame
  metadata at submission, retaining the producer rather than importing RHI
  buffers or copying images through the host.
- [ ] Implement typed `project(frame)` using the existing graph/output contract
  and same-queue producer dependencies. Release retained producer inputs only
  after view completion and output release; no hidden wait/readback, context
  creation, JSON/hash scan, matrix validation or heap allocation per frame.
  Reject shape/owner/context mismatches before graph execution. Keep source
  calibration/support validation at setup and constant-time owner checks at
  submission.
- [ ] Run the new CPU tests and existing `daxa_view` fixture contracts, including
  `legacy_view_fixture_controls`; preserve old fixture output controls and
  explicit diagnostic readback. This establishes intake/lifetime correctness,
  not calibrated wide-view acceptance by itself.

### 6b. Direct calibrated geometry and separate full RGB preview

Setup must distinguish two source spaces: two rectified depth/color grids of
`rect_width * rect_height` each, and two full original RGB grids of
`eye_width * source_height` each. Validate sizes, both-eye offsets, checked
products/counts, preview-index bounds and the count actually used by every
shader dispatch. Never use original dimensions to index rectified metadata or
use a raw preview index as a metric observation owner.

For rectified metric pixels, cache unit-Z rays
`q = ((x-cx)/fx, (y-cy)/fy, 1)` from each rectified camera model. With
`V = virtual_from_original_left`, use the effective transforms exactly once:

```
left:  V * block_rotation(R1^T)
right: V * original.right_to_left * block_rotation(R2^T)
```

Multiply these rays by the measured rectified axial Z. Do not apply rectified
baseline translation a second time. If the caller supplies two original-eye
virtual transforms, validate their relation against the original right-to-left
pose at setup. Metric color comes from the existing qualified
`RectifiedLeft/RectifiedRight` RGB images, remapped from the same retained
original capture. Full `OriginalLeft/OriginalRight` RGB remains independently
available for peripheral preview; a rectified crop cannot replace it.

Hash/identity and exact immutable ownership bind payload bytes but **do not
establish per-pixel association between maps and camera geometry**. Introduce
an explicit setup association gate: use authoritatively geometry-generated
maps/fixtures and validate their association with the retained models,
rotations, dimensions and sampling convention. State the accepted calibration
producer/trust boundary in setup/report metadata. Arbitrary manual maps may
remain valid for the legacy 2D RGB/stereo fixture path, but cannot silently be
promoted to strict calibrated view input. Consistent synthetic fixtures and an
authoritative setup validation/trust boundary are mandatory before strict/live
acceptance; physical calibration truth remains a separate qualification.

- [ ] Add watched CPU RED cases `rectified_grid_rotation_once`,
  `shared_raw_cell_distinct_virtual_points`,
  `original_rectified_dimension_mismatch`,
  `map_geometry_association_required`, `preview_namespace_bounds_and_eye`,
  `strict_holes_and_original_rgb_preview`, and
  `perspective_axial_and_cylindrical_range`. Include nonidentity R1/R2,
  nonzero principal offset, unequal raw/rectified dimensions, right-eye
  extrinsics, distortion, unsupported remap samples and both eye boundaries.
  Two rectified observations that round to one raw pixel but project to
  distinct virtual pixels must both survive to visibility processing; never
  reconstruct their positions from the rounded raw pixel's ray.
- [ ] Implement the setup-only calibrated cache and association checks, then
  bind separate geometry rays, rectified metric colors and original preview
  colors in the retained view slot. Generate original preview directions from
  the original calibrated model/distortion and rotate into the virtual view;
  explicitly omit translation for infinity preview. Keep metric projection,
  depth ordering, deterministic owner ties, layer compatibility, exposure and
  resolution in the existing shared projection operations. Add narrow typed
  readers/wrappers for the separate RGB namespaces rather than a compositor.
- [ ] Preserve strict holes. Preview may fill RGB only: mark RGB valid and
  approximate, keep metric validity false, depth canonical NaN, metric
  confidence/provenance/age zero and source ID -1. A preview-only owner is
  `UINT_MAX`; metric metadata gathers require a valid metric owner within
  `2 * rect_width * rect_height`. Perspective output depth is virtual axial Z;
  cylindrical output depth is virtual ray range (`length(local.xyz)`), with
  vertical projection using the horizontal cylinder radius. Expose that depth
  convention alongside the virtual camera model in reports.
- [ ] Run CPU GREEN and the legacy fixture controls, then freeze the production
  sources, shader/compiler/dependency identities and old/new SPIR-V manifests.
  Independent functional and performance reviews must clear before root GPU
  execution. The performance gate `no_per_frame_alloc_wait_readback_matrix_validation`
  inspects actual setup/submission/retirement paths and probes bounded host
  work, retained buffers and graph reuse. Do not infer speed from source review.

### 6c. Whole-chain qualification; root alone launches GPU

- [ ] Build an independent geometry-generated held-out fixture oracle with
  original RGB, supported rectified depth, explicit poses and known 3D points.
  Bind fixtures, map/model association evidence, all sources, helpers, compiler,
  dependencies, binaries, SPIR-V and reports before/after children. Compare
  actual final strict/preview outputs and metadata against the independent
  geometric oracle and preserved frozen compositor controls; no blanket
  compound-float tolerance or CPU depth approximation substitutes for final
  output gates. Keep previously unresolved scalar float qualification open.
- [ ] Root runs isolated native GPU cases for the named 6a/6b contracts, then
  core and synchronization validation, dynamic-content/same-ID inputs,
  delayed consumers, full rings, destruction orders and bounded long stress.
  Exercise perspective and cylindrical outputs, rotated original/rectified
  models, mismatched grid dimensions, occlusion/ties, strict remap holes,
  peripheral original RGB and invalid preview indices. Check every output
  plane/default, exact int64 IDs/ages and correct depth convention. Re-run
  legacy fixture outputs and qualify every changed shader explicitly.
- [ ] Root measures actual capture/upload/preprocess/dual stereo/LR/metric/view
  queue span plus host wall, query/collection costs, retained memory, throughput,
  frame age, offered/completed/skipped inputs and backpressure at the required
  720p/wide configurations. Report diagnostic readback separately. Use one
  shared-context same-clock span including inter-graph gaps; sums of isolated
  medians are not full-chain latency or FPS. The current approximately 32 ms
  depth result does not establish 60 Hz. Full strict/preview/peripheral output
  correctness and transfer-inclusive live 60 Hz acceptance remain mandatory;
  document any unmet gate rather than concluding at 6a.


Implementation verification commands (the new native contract target registers
`daxa_view_depth_cpu_contracts` and hardware-labelled
`daxa_view_depth_contracts`; CPU mode must not initialize Vulkan):

```bash
cmake --build build/daxa --target daxa_view_contracts daxa_view_depth_contracts daxa_view_depth_benchmark
ctest --test-dir build/daxa -R '^daxa_view(_depth)?_cpu_contracts$' --output-on-failure
.venv/bin/python -m pytest -q tests/test_daxa_view_depth_comparison.py tests/test_daxa_view_depth_hotpath.py
```

Watch the relevant new CPU test fail before its implementation, then require
these commands to pass. After independent functional/performance clearance,
root additionally runs `ctest --test-dir build/daxa -R
'^daxa_view(_depth)?_contracts$' --output-on-failure` in the isolated core and
synchronization validation environments, then the bound comparison harness and
measurement app. These commands are implementation gates, not runs performed
by this planning task. Root reviews the demonstrated envelope before any
commit/push; planning or CPU success alone does not close Task 6.


## User steering: paired left/right compute scheduling

2026-10-27: test the useful analogue of VR multiview on the existing compute
pipeline before continuing Task 6 production bodies. The wide-view scope stays
active; its initial CPU RED test is preserved. The published `659f4fd` baseline
and its exact source/native/SPIR-V archive remain the control.

### First causal variant: pair eyes within each path

Keep `Sequential` as the default; add explicit `PairedPerPath` scheduling and
`--stereo-schedule sequential|paired-per-path`. One persistent paired graph
records the existing left/right specialized dispatches back-to-back within each
Census/clear/scan-direction/selection phase. Both references retain independent
penalty/output buffers. Keep barriers between same-eye scan directions because
current packed additions are non-atomic. No shader, scan recurrence, D128,
four-path, subpixel, LR, mask, confidence or metric policy change is authorized.
All 29 existing shader binaries must remain unchanged for this control.

Retain duplicate Census buffers/computation initially to isolate scheduling.
Measured historical duplicate Census work is only about 0.09 ms, so removing it
cannot explain a large speedup. Shared-Census storage can be evaluated separately
later with exact ownership and source-generation checks. Concurrent same-eye
paths require a separate atomic or private-accumulator design and qualification;
plain packed += must never be raced.

Implementation owner: `kernel_variants`, existing Stereo/Depth headers and
implementations, depth benchmark, native contracts and optional depth build
registration. Oracle owner: `right_reference`, existing depth comparison harness
and its Python tests only. Independent performance owner: `native_capture`, new
`tests/test_daxa_depth_pair_hotpath.py` and ignored report only. One writer per
shared surface. Root alone executes GPU workloads and makes promotion decisions.

- [ ] Watch new mode/admission/timing contract tests fail, then implement the
  private persistent pair coordinator and existing typed per-eye frames.
- [ ] Validate aligned pair capacity before upstream upload. Retain source,
  calibration, generation and producers through output consumers; release setup
  prototypes and avoid control-block cycles. No per-frame allocation, waits,
  readback, matrix validation or image loops. Use the existing admission
  completion snapshot; report SDK scans separately in measured host cost.
- [ ] Timestamp each paired phase around both dispatches; do not insert
  per-reference timestamps between eyes and then claim overlap. Individual eye
  timings are explicitly unavailable/null in paired reports. Pair total/phases
  and actual upload-start through metric-end queue span are required. Preserve
  real individual timings in sequential mode. Coordinate exact JSON keys before
  authoring strict report protocol tests.
- [ ] Preserve independent Census/integer/actual packed-P checks and all nine
  observation plus six raw stereo byte comparisons against frozen RHI. Extend
  source/binary/compiler/dependency/config/report binding to the schedule. Keep
  CPU compound-float qualification explicitly open; add no float tolerance.
- [ ] Complete independent functional and mandatory performance reviews before
  root GPU validation. Guard actual admission/submit/collect paths and probe
  bounded host work, allocations, staging bytes and completion snapshots.
- [ ] Root runs small and dynamic both-mode core/synchronization contracts,
  then full 720p exact-output checks, full rings/delayed consumers/producer
  destruction and short static stress. No GPU run by workers.
- [ ] Root measures sequential / paired / paired / sequential in separate,
  fresh, non-overlapping runs at identical 720p/D128/P4/W32/two-slot settings,
  100 measured plus 20 warmup iterations each. Keep readback/reference checks
  outside performance loops. Compare actual complete queue span, host wall,
  paired direction/selection intervals, enqueue/query/wait cost and requested
  buffers. Do not sum unrelated percentiles or infer GPU occupancy from VGPRs.
- [ ] Record benefit or regression against run-to-run variation. No automatic
  promotion or 2x claim. If batching does not provide useful acceleration, retain
  its causal result and continue with a separately reviewed layered dispatch or
  all-path atomic experiment. A negative scheduling experiment does not close
  the full stereo optimization, wide RGB-D or live 60 Hz goal.


### Next causal variant: one layered dispatch per path

The first paired graph is qualified as a negative control at `2094237`:
100/20 ABBA means 32.056/32.013 ms sequential host wall versus 40.255/41.298 ms
paired host wall, each value already covering BOTH references. These are
separate repeated-run measurements, never per-eye costs to sum. Sequential stays
default. All 29 shader binaries remained unchanged in that first experiment.

Append `LayeredPerPath`, CLI `layered-per-path`, retaining both controls. Use
one compute dispatch with z=2 per aggregation path and one z=2 selection dispatch.
Workgroup z chooses the eye; XY flattening must exclude z. Keep separate packed
penalty/output buffers and required inter-path barriers; Census/clear stay as
two independent dispatches in this initial layered experiment. This tests one
shader program for both references without changing SGM recurrences or quality.

- [x] B extracts one explicit-parameter implementation into canonical
  `python/stereoforge/shaders/ops/sgm_specialized.slang`, with shared guarded
  `daxa/stereo_parameters.slang` and new `daxa/stereo_layered.slang` wrapper.
  Existing specialized wrappers pass the same parameters into the same helpers.
  Preserve exact subpixel expression/control flow and legacy Push16. New Push24
  supplies both parameter addresses/pass/XY group stride; static layout guards
  and device push/z-dispatch checks are mandatory. No algorithm duplication,
  atomics, precision/search changes, new compositor or per-frame memory.
- [x] Uniformly reject z>=2 before address loads or collective/barrier work.
  One workgroup uses one eye and unchanged full-wave32/64 rules; all selection
  group threads participate consistently. Retain canonical invalid defaults,
  exact metadata, source/owner identity, pre-upload admission and producer life.
- [x] A extends the existing bound nine-observation/six-raw-plane harness for
  all three scheduling modes. Reports require the actual dispatch form and
  null isolated-eye timing for either paired mode, real paired phase/total
  statistics and the same actual upload-through-metric span. Add compiled
  runtime identities for shared Parameters/helper and layered wrapper; bind
  all three source files, four new SPIR-V files and actual compiler dependencies.
- [x] C independently guards/probes actual hot paths, uniform eye/bounds/XY
  mapping, one-dispatch phase structure, allocations/lifetimes/completion and
  truthful timing/compiled source identity. Preserve prior controls.
- [x] Build and hash all 29 existing shader controls. Explicitly list every
  changed binary; helper extraction alone is not evidence of byte identity.
  Before new-mode GPU qualification root requalifies any changed legacy shader
  through its appropriate exact integer/direct-P/frozen-GPU output controls.
  Keep unresolved general scalar compound-float gates open.
- [x] After independent functional/performance clearance, root runs new native
  CPU/GPU core+sync contracts, small and dynamic/P8/W64/packing16 exact controls,
  full 720p fixed-quality nine+six output comparison and lifetime stress.
  Root alone launches GPU work. No mode is promoted on build/CPU/source review.
- [x] Root repeats fixed-profile 100/20 sequential/layered/layered/sequential
  measurements with outputs resident, no concurrent GPU jobs, validation and
  readback excluded from performance samples. Actual complete queue span and
  host wall are the result; compare repeats and frame budget without adding
  per-eye costs or claiming occupancy/bandwidth mechanisms without evidence.
  Record acceleration or regression and continue the full goal either way.


### Exact data reuse: shared Census, sequential independent references

The layered experiment is qualified and published as `eaea904`; the
[measured evidence](../../daxa-depth-layered-evidence.md) records exact outputs
but a negative full-chain result. Sequential GPU means were 31.883/31.795 ms,
versus 39.971/40.312 ms layered. The goal remains sustained live 720p/60 Hz,
wide RGB-D and temporal/adaptive integration; this experiment does not finish it.

The next bounded exact optimization is `SharedCensusSequential`, CLI
`shared-census-sequential`. Capture already supplies one side-by-side YUYV
frame and preprocessing already serves both references. Compute the two Census
images once, share their descriptors and validity, and preserve complete
independent Left and Right SGM. Raw Hamming pair costs are symmetric, but the
independent aggregated path histories are not interchangeable. The later
single-aggregation/dual-disparity technique observed in libSGM must remain a
separate quality-changing experiment with explicit provenance and quality gates.

- [x] Native implementation: one physical four-buffer descriptor/validity
  quartet per aligned slot, with shared RAII storage retaining the Foundation.
  Both engine slots retain this storage; no Frame/Impl ownership cycle and no
  duplicate allocation. Either retained reference keeps descriptors alive;
  aligned admission prevents overwriting them before both consumers retire.
- [x] A shared graph registers each aliased physical buffer once, runs exactly
  two Census dispatches, then all Left clear/path/selection work, then all Right
  clear/path/selection work. Preserve same-eye penalty hazards and explicitly
  order the Right clear after Left selection to isolate data sharing from
  interleaved scheduling. No shader algorithm, precision or search changes.
- [x] Keep Sequential default and all prior controls. Add actual timing phase
  names for shared Census and each reference's clear/path/selection, fourteen
  for P4 and twenty-two for P8; isolated eye times remain null. Combined GPU
  span remains actual upload-start through metric-end, with host wall separate.
- [x] Report actual descriptor storage, two Census dispatches, independent
  reference penalties and savings. Requested application buffers decrease by
  16*width*height*slots, exactly 29,491,200 B at 720p/two slots, to
  811,009,328 B at D128/P4. Preserve the 2 GiB guard and distinguish requested
  buffer bytes from peak driver/allocator memory.
- [x] Extend the existing source-bound nine-observation/six-raw-plane harness
  with strict new-mode memory/phase/identity claims. Preserve existing oracle
  arithmetic, metadata, masks and invalid conventions without tolerances.
- [x] Independent functional and mandatory performance review must pass before
  root GPU execution. Guard allocation/admission, graph hazards, descriptor
  ownership, queries, aliasing, source pairing and unchanged shader binaries.
  Use an extracted actual host-path probe; report its excluded SDK/GPU scope.
- [x] Root builds and verifies all 33 shader controls byte-identical, runs CPU,
  native GPU core/sync, exact small/full720p/P8W64 packing16 gates, and retained
  producer/frame lifetime and repeated-fixture stress checks.
- [x] Root measures fixed-quality 100/20 Sequential/shared/shared/Sequential
  runs without readback or validation, retaining outputs on GPU. Promote only
  on measured useful benefit; record memory-only benefit honestly if latency
  improvement is small. Further descriptor-load/cost reuse and single-volume
  dual-disparity research remain open toward the full 60 Hz goal.


### Experimental single aggregation with derived right selection

Shared Census is qualified at `0afc996`: it saves 29,491,200 B but does not
establish a repeatable latency improvement. The next substantial work reduction
is a separate estimator mode, not another scheduling claim.

Append `DaxaRightReferenceMode::{IndependentSGM,LeftAggregatedSelection}` to
configuration, keeping IndependentSGM default. Derived mode requires the existing
SharedCensusSequential coordinator. Preserve full left SGM, then select right
pixel xR/disparity d using left scores at xL=xR+d:
`Sderived(xR,d)=paths*Cleft(xR+d,d)+unpack(Pleft(xR+d,d))`.
This is correlated left-aggregate evidence, not an independently regularized
right SGM. Both original camera images still contribute matching costs.

Root algorithm choices for this experimental mode: exclude unsupported
candidates before descriptor/P access; require at least two supported
candidates; choose lowest disparity on equal costs; retain the existing
selection-validity and confidence formulas. Refine subpixel only if both
neighboring candidates are supported, gathering each at its own shifted left
pixel; otherwise retain integer disparity. Do not add rectification-map support
vetoes to raw stereo selection. Metric support remains the existing source and
nearest-correspondence gate. Raw left outputs remain byte-exact; LR-filtered
left outputs may differ because the right check changes.

- [x] Add private selector-only right setup with no right penalty allocation,
  clear or aggregation. Share left packed storage through bounded noncyclic
  RAII retaining the Foundation; a right-only lease keeps its dependencies
  alive and blocks aligned slot reuse. Reject derived right PackedPenalties or
  reconstructed-intermediate access rather than labeling left state as right
  recurrence. Preserve all old constructors, modes and admission behavior.
- [x] Dispatch two shared Census tasks, left clear, all left paths, unchanged
  left selection, then derived right selection: P+5 stereo phases (9/13).
  Declare shared physical resources once, preserve left path hazards and
  bootstrap synchronization, and add no per-frame CPU image work, allocation,
  wait or readback. Report actual upload-through-metric span and host wall.
- [x] Add separate production selector wrapper and derived metric wrapper;
  preserve all 33 prior control shaders and all 46 existing build SPVs. Bind
  actual compiled new source identities and all new SPVs in diagnostics.
- [x] Expose typed right estimator and correlated LR dependency on retained
  frames/readbacks. Both metric layers use a new explicit provenance value 5
  (LeftAggregatedStereo), invalid remains 0, and legacy valid stereo remains 1.
  IDs, observation age, metric geometry and invalid NaN conventions are unchanged.
- [x] Add a derived-only schema and honest diagnostic flags; the legacy exact
  dual-SGM harness must reject that schema. Native diagnostics independently
  check left Census/integer/P and the derived right mask, without claiming
  independent right recurrence or general CPU compound-float equivalence.
- [x] Build a separate independent reference selector over dense left scores
  produced by frozen RHI SGM. Its per-pixel selection cannot call production
  gather/selection helpers. Validate raw left bytes, derived right bytes and all
  nine observation planes, with provenance 5 on both supported metric layers.
  Bind source/compiler/dependency/SPV/binary/fixture/report identities before
  and after execution. Keep the legacy nine-plus-six oracle unchanged.
- [x] Required independent performance and functional reviews precede root GPU
  runs. Guard support-before-load, neighbor coordinates, ties, provenance,
  shared ownership, alias registration, ring admission, startup and completion.
  Exercise packing8/16, P4/P8, W32/W64, one/two slots, missing candidates,
  boundaries, subpixel neighbors and retained-right lifetime.
- [x] Verify actual buffer accounting; retaining other allocations removes
  another packed volume (235,929,600 B at 720p/D128/two slots), yielding
  575,079,728 B requested buffers. Keep the 2 GiB guard and report buffer-only
  scope rather than peak driver memory.
- [x] Root measures matched 100/20 independent/derived/derived/independent
  resident runs after exact algorithm gates. Record latency and buffer accounting
  separately from accuracy and fresh-camera throughput.
- [ ] **Partial/open quality gate:** controlled reference development cases cover
  planes/slopes, thin foregrounds, occlusions, photometric changes and borders,
  with regional errors/completeness and false consistency. Repeated-texture
  coverage, production Daxa execution on those quality fixtures, frozen accuracy
  thresholds and held-out assessment remain open. No default promotion,
  physical accuracy or live 60 Hz claim follows from exact implementation bytes
  or timing.

Completed reviewed evidence is recorded in
[derived estimator evidence](../../daxa-depth-derived-evidence.md) and
`artifacts/daxa-depth-derived-20261027/`. Native freeze SHA256
`093223f8156b6df729f6c676f309e23b1e33070259a3321e73b8dec0acba5fd1`
was rebound after the quality eye-region review fix; independent functional and
performance reviews cleared that evidence. Fresh precommit CPU verification
passes 1263 Python tests (37 GPU/camera tests deselected) and all seven CPU CTest
contracts. Four derived nine-plus-six comparisons and the legacy sequential
control pass exact bytes. Native core/synchronization contracts and 300-sample
synchronization stress pass with empty stderr, 303 retirements and zero
warnings/errors; all 46 preceding build SPVs and 33 prior controls remain byte
identical.

Matched ABBA means are approximately 19.1 ms GPU / 19.5 ms host for derived
selection versus 31.4–31.6 / 31.9–32.1 ms for independent SGM, about 39% lower
latency for a different estimator. Requested buffer bytes fall from 811,009,328
to 575,079,728; this is buffer-only accounting, not peak driver memory. The
16.67 ms budget remains unmet. Controlled reference quality exposes regressions:
thin-occluder Right false LR correspondences rise 55→105 and thin-region MAE
1.611648→2.239916 px. Those fixtures did not execute production Daxa and are
neither physical nor held-out evidence; the mode is not promoted.

A subsequent CPU-only development precision probe finds that LR tolerance 0.75
reduces aggregate Right false correspondences 162→82 while retaining correct
correspondences 9621→9620. This is exploratory rejection analysis, not a
production precision fix. No production threshold changed; held-out validation
and an independently frozen quality gate remain pending.
