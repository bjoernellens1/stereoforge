# Native Temporal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox syntax for tracking. Supervisor owns integration, commits and all GPU execution.

**Goal:** Execute known-motion temporal reconstruction on retained GPU observations, with complete bounded ancestry and direct virtual-view consumption.

**Architecture:** Persistent output slots double as bounded history. Shared Slang kernels perform projection/visibility/fusion; fixed-size C++ admission handles pose and sequence validation. Common observation leases carry current RGB and accumulated metric lineage without CPU transfers.

**Tech Stack:** C++20, Daxa pinned 00761e80c1dd6d1972d5f4569ce49620a7b4d384, Slang, Vulkan, independent Python scalar test oracle.

**Spec:** `../specs/2026-10-28-native-temporal-design.md`; parent `2026-10-27-daxa-slang-production.md` Task7b.

## Global Constraints

- Vulkan-only production; no CUDA/HIP/ROCm/TensorRT or SlangPy in runtime.
- Defaults: slots2, complete contributors4, reuses4 inclusive, oldest age200000000ns inclusive, relative depth tolerance0.03, confidence floor0.05.
- Original RGB retained; stereo disparity unchanged; perspective temporal depth is rectified axial Z metres.
- No per-frame allocation, implicit wait/readback, CPU image loops, graph rebuild or unbounded capture chains.
- Workers verify implementation worktree/branch first; not alone, preserve others' edits; no commit/push/GPU. Root alone integrates/publishes.

## Review Focus

- Two-slot history refcounts and read-after-write completion protect retained old consumers and pending history reads.
- Active history releases upstream capture when externally unused; Depth's ring must keep progressing.
- Failure during submission pins pending inputs but cannot replace accepted history/pose.
- Pose/calibration/clock mismatches reject transactionally, including recovery after missing pose and epoch resets.
- Complete lineage, selected-vs-oldest age and float boundary classification remain independent evidence; no tolerance hides discrete mismatches.

### Task 1: Calibrated native stage and retained interfaces

**Owner:** one native implementation worker; intertwined production changes must have one owner.
**Files:** create `include/stereoforge/daxa_temporal.hpp`, `src/daxa_temporal.cpp`, `python/stereoforge/shaders/ops/temporal.slang` and Daxa wrapper; modify `include/stereoforge/daxa_observation.hpp`, `src/daxa_observation.cpp`, `include/stereoforge/daxa_view.hpp`, `src/daxa_view.cpp`, `include/stereoforge/daxa_calibration.hpp`, `src/daxa_calibration.cpp`, `src/daxa_view_calibration.cpp`, `CMakeLists.txt`; add `tests/daxa_temporal.cpp` and targeted CPU source/contract tests.

**Interfaces:** exact signatures/resource semantics in spec. Produces `DaxaTemporal`, `DaxaTemporalConfig`, `DaxaTimedRigPose`, common ancestry descriptor, `DaxaView::Frame::source_observations()` and setup-only `validate_map_association`. Follow existing public Frame naming for timings; document final method names before other workers compile consumers.

- [ ] Write failing CPU admission/lifetime/source tests: invalid config/motion/clock, ring read/write tickets, reset, assignment/facade destruction and source retirement. Preserve existing View association rejection and operation order.
- [ ] Run targeted tests with invalid ICD and record RED before implementation.
- [ ] Extract generic map association and implement persistent native stage, four visibility/fusion passes, lineage adapter and View source getter. Keep old shaders unchanged; use current Depth support descriptors.
- [ ] Build CPU/native targets (exclusive build directory ownership); run targeted CPU/source tests and shader compilation. No GPU.
- [ ] Independent functional review against spec; fix and rerun affected checks. Supervisor records final interfaces for Task2/3.

### Task 2: Independent sequence fixture and comparator

**Owner:** independent oracle worker, no production changes.
**Files:** create `apps/daxa_temporal_fixture.cpp`, `benchmarks/compare_daxa_temporal.py`, `tests/test_compare_daxa_temporal.py`; coordinate CMake target addition through production owner. Read existing reviewed `benchmarks/temporal_known_motion_oracle.py`.

**Interfaces:** fixture constructs actual Depth and native Temporal, waits only at diagnostic boundaries, overrides controlled fresh test planes after Depth completion, exports six metric + complete sidecar planes and View owners. Comparator computes expectations independently from literal timed-world points and reviewed scalar oracle; source/build/input hash binding.

- [ ] Failing comparator tests for corrupted masks, selected/oldest IDs/ages, missing planes/hash mismatch and nonfinite results; run RED.
- [ ] Implement sequence fixture and independent comparator; no public arbitrary-buffer construction, no fixture flags in production benchmark.
- [ ] Cover all spec cases including genuine moving rotation, age/reuse limits, ties/occlusion/support, provenance5, lineage intersection/cap, duplicate/order/clock/reset and View lineage. Add explicit normal-domain canonicalization and float32 bounded-convex policy tests; do not alter frozen float64 oracle or claim its extreme-value whole-transaction rejection as native behavior.
- [ ] Run comparator CPU tests with warnings as errors; label exact rational vs general float domain and restricted frozen controls explicitly.
- [ ] Independent source review; no GPU dispatch by workers.

### Task 3: Performance refinement and actual chain benchmark

**Owner:** separate performance reviewer after production diff stabilizes; benchmark implementation coordinated with native owner.
**Files:** new `apps/daxa_temporal_benchmark.cpp`, `tests/test_daxa_temporal_hotpath.py`, actual-source mock hotpath probe and scoped report under ignored `artifacts/daxa-temporal-20261028/`.

**Interfaces:** actual unmodified SGM Depth->Temporal and optional View, preallocated capture metadata ring; `process`, `collect`, `can_process`, completion tickets and per-stage timestamps. Unique metadata over static pixels is not physical unique-frame throughput.

- [ ] Audit/refine allocations, graph recording, explicit waits/readbacks, deep capture retention, write/read ticket exceptions and barriers.
- [ ] Add static guards plus executing instrumented actual-source probe; assert zero allocation and zero implicit poll/wait in process apart from documented admission/collection snapshots, bounded nodes/slots, successful source retirement and retained-consumer backpressure.
- [ ] Build benchmark, report CPU mock cost and requested memory; no GPU before separate PERF_CLEAR and functional CLEAR.
- [ ] Run final CPU suite once stable: `STEREOFORGE_NATIVE_DIR="$PWD/build/native-capture-joint/native" PYTHONPATH=python VK_DRIVER_FILES=/nonexistent .venv/bin/python -m pytest -q tests -m 'not gpu and not camera'`; expected all applicable pass, no exclusions hiding new failures. Run CPU CTests.

### Task 4: Root GPU qualification and publication

**Files:** scoped evidence doc, README/status/parent plan; ignored bound manifests/results.

- [ ] Freeze complete actual source/compiler/binary/SPIR-V and comparator input hashes after independent functional/performance CLEAR.
- [ ] Root run exact known-motion sequence matrix, core/sync validation, legacy View regression and retained/facade/ring/reset tests. Confirm discrete six-plane/ancestry/owner metadata exactly; broader float results separately scoped. Iterate on failures, preserve failed evidence.
- [ ] Root run >=300-frame stress with synchronization validation through teardown; verify bounded retention/memory and capture source release. No live/physical claim from static content.
- [ ] Root benchmark Depth alone, Depth+Temporal and optional View with matched config/order; report stages/transfers/waits/host/frame-age limitations/requested bytes, no inferred throughput from percentiles.
- [ ] Independent final evidence review; supervisor commit, fast-forward main and push existing authorized origin, verify SHA locally/remotely. Keep adaptive/live60/physical/general-float/peak-memory gates open unless independently proven.
