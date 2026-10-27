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

- [ ] Write failing `daxa_foundation_contracts` cases for the pinned SDK/build configuration and externally registered resident-buffer views. The GPU regression writes/reads a Daxa-created application buffer across graph executions, validates its view and generation, and checks tiny final counters under Khronos validation.
- [ ] Pin Daxa full commit/source hash and an explicit Linux Slang SDK. First test the existing 2026.12 Linux SDK deliberately; if incompatible, record the failure and test a separately pinned Linux SDK supported by Daxa. Preserve compiler identity in every comparison; do not silently use its Windows bootstrap or RHI-linked compiler.
- [ ] Resolve the documented external-resource blocker with a verified upstream fix or one minimal checked-in patch and its regression. Record the original failing revision; this is a dependency fix, not a new StereoForge resource framework.
- [ ] Add the probe/build wiring, source/transitive license inventory and notices; query actual AMD capabilities and UMA coherency rather than hardcoding support.
- [ ] Run `cmake --preset daxa`, `cmake --build --preset daxa`, then `ctest --test-dir build/daxa -R daxa_foundation --output-on-failure`. Require actual Daxa/Slang device execution and clean core validation; unavailable features/build support remain a recorded blocker. A skipped native test is not a pass.

**Gate:** No production migration or externally registered history/ring resources until Linux build, feature, license and external-view checks pass. If they fail, retain the working RHI reference and record the specific blocker; do not claim Daxa adoption.

## Task 2: Match the resident RHI compositor in a small Daxa POC

**Files:** Create `apps/daxa_view_benchmark.cpp`, `tests/daxa_view.cpp`, `benchmarks/compare_daxa_view.py`, `python/stereoforge/shaders/ops/projection.slang`, and `python/stereoforge/shaders/daxa/view.slang`; extend the opt-in Daxa CMake section only. Existing `shaders/` aliases this canonical shader directory; do not create a second shader tree.

**Interfaces:** `daxa_view_benchmark --fixture DIR --mode strict|preview --frames N --warmup N --output FILE --dump-prefix PREFIX` consumes the existing resident fixture format. Daxa owns its device/buffers; comparison with `native_view_benchmark --input resident --readback gpu` uses separate processes, identical input bytes and declared source/compiler identities. Reusable operation functions contain no Daxa descriptor declarations; the Daxa entry wrapper supplies bindings.

- [ ] Freeze the currently passing RHI fixture/shader/compiler/report hashes and add failing Daxa tests for all eleven output planes, strict/preview behavior, 64-bit IDs/ages and unsupported holes. Keep the existing float32 half-pixel contract explicit; do not relax exact comparisons on a matching execution profile.
- [ ] Port only the projection/compositor slice into a Daxa TaskGraph using reusable operations and thin wrappers. Retain Daxa-owned persistent inputs/outputs; declare each pass's resource reads/writes and required usage precisely. Use the documented `create_task_buffer` path where appropriate; application-owned resident registration first passes Task 1's fixed-view regression. No foreign RHI resources/device imports.
- [ ] Exercise one, two and three ring generations, delayed consumers, graph reuse, external-buffer registration, attempted overwrite and resolution/calibration changes. Retain outputs until completion; recycled resources must never corrupt a pending consumer. Use documented persistent/double-buffer lifetime semantics for retained state, with explicit completed-frame bookkeeping across executions; within-graph barriers alone do not prove safe next-frame reuse. Transient aliasing must exclude persistent inputs/history/live outputs.
- [ ] Run the matched 720p resident benchmark, setup cost separately, at least 3 warmups and 30 samples; report GPU timestamps, CPU callbacks/submission, waits, memory and CPU dumps outside the GPU-output measurement. Require exact equality of all eleven Daxa/RHI fixture planes, including RGB, depth and confidence, on the shared execution profile. The current 720p fixture already matches all eleven exactly; do not introduce a one-intensity or new floating-point allowance for migration. A compiler-change disagreement requires a diagnostic artifact and resolution before the gate passes. Separately retain the documented float32-versus-float64 camera half-pixel boundary; it is not a waiver for this matched fixture.
- [ ] Run `ctest --test-dir build/daxa -R 'daxa_(foundation|view)' --output-on-failure` with core validation and a separately reported synchronization-validation run. Compare against RHI without assuming Daxa is faster. Preserve regressions and fail the POC if correctness, lifetime, support or agreed resource/latency ceilings fail.

**Gate:** Record an explicit POC decision and measured operating envelope. The production target remains Daxa, but cutover requires a passing foundation/POC; a slower or unsupported result is evidence to resolve, not permission to relabel a migration as complete.

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
