# ZED quality benchmark implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Root owns integration, GPU admission and publication; preserve other workers' edits.

**Goal:** Build a reproducible common-input quality benchmark before further Fast60 optimization or promotion, then measure sustained delivery only for qualified profiles.

**Architecture:** Versioned portable records bind exported pixels, effective geometry, settings and truth independently of the proprietary SDK. An optional isolated C++ SDK exporter feeds a native already-rectified RGB adapter and offline evaluators. Full independent SGM precedes Fast60; SDK modes are measured references, never ground truth.

**Tech Stack:** C++20/Daxa/Slang/Vulkan runtime; optional ZED SDK C++ exporter; Python offline export indexing/evaluation/reporting; JSON policy; optional ROS 2 delivery benchmark.

**Spec:** [Quality-first benchmark design](../specs/2026-10-28-zed-quality-benchmark-design.md). Read with the [Fast60 plan](2026-10-28-fast60.md) and [refinement plan](2026-10-28-fast60-refinement.md).

## File boundaries

`schema.py` owns portable identities; `svo_to_frames.py` only indexes exports.
`compare_quality.py` and `evaluate_temporal.py` own offline metrics; `performance.py`
owns timing admission; `report.py` owns decisions and presentation. Native SDK
export and Daxa input remain independent optional/runtime boundaries.

## Global Constraints

- Production remains C++20/Daxa/Slang/Vulkan, AMD first, without Python, CUDA, HIP, ROCm or TensorRT dependencies.
- SDK runs only in an optional isolated runner on an inventoried NVIDIA host with vendor-supported Ubuntu; default CMake option OFF, no core/CI/deployed SDK dependency.
- Keep SDK binaries/models, camera serials and private recordings out of Git; inspect installer/model/sample/recording/export terms before redistribution.
- SDK absence permits portable evaluator/adapter work; external parity stays unqualified. No vendor accuracy table substitutes for execution.
- Preserve lossless original and rectified RGB/effective calibration; never round-trip RGB through YUYV. SDK export is post-ISP, not USB/Bayer identity.
- Primary instantaneous lane disables stabilization/tracking/filling where supported; record effective settings or explicit failure. Temporal-on is separate.
- Freeze required regions, uncertainty/statistical policy and thresholds on development data before untouched held-out evaluation; missing policy/data means unqualified.
- Provisional NEURAL_LIGHT margins: completeness loss ≤2 percentage points; relative MAE ≤1.10× reference; no excess false accepted boundary rate, with separate thin/boundary/near gates.
- Publish common supported-range and full requested 0.3–8 m results; D128 indices 0…127 cannot silently cover larger disparities.
- Scientific quality tolerances do not relax existing exact shader integers, masks, identities or canonical invalid encodings; arithmetic qualification remains its own gate.
- Sustained lane: at least 600 s nominal 60 Hz, bounded 2–3 slots, unique depth/RGB deliveries, age/drops/gaps/queues/resources; fixed View and ROS delivery are separate gates.

## Review Focus

- Missing SDK mode/settings/metadata must produce unavailable/unqualified records, not substituted modes or fabricated maps (Task 2).
- Changed calibration/pixels or clock domains must split strata or reject direct comparison, preserving original geometry (Tasks 1, 3).
- All-invalid, sparse truth and zero reference error must retain denominators and uncertainty rather than yield a false pass (Tasks 1, 4).
- Lost/duplicated frames and historical/predicted pixels must not inflate fresh independent coverage or temporal stability (Tasks 5, 7).
- Cold/host-output/resident and cross-platform timings must keep their actual completion boundaries, avoiding algorithmic speedup claims (Tasks 6, 7).

---

### Task 1: Portable records and frozen policy

**Files:** Create `benchmarks/zed_comparison/__init__.py`, `benchmarks/zed_comparison/schema.py`, `configs/benchmarks/zed_quality_policy.json`, `tests/test_zed_comparison_schema.py`, `tests/test_zed_schema_io_guard.py`, and `benchmarks/zed_comparison/README.md`.
**Interfaces:** Consumes relative record paths and policy JSON. Produces `load_record(path: Path) -> dict`, `validate_record(record: dict, root: Path) -> None`, `load_policy(path: Path) -> dict`, `validate_partitions(records: list[dict]) -> None`.

- [x] RED: add `test_v1_round_trip_hashes`, `test_reject_unsafe_paths_and_changed_hash`, `test_partition_scene_sequence_leakage`, `test_missing_policy_is_unqualified`, `test_calibration_and_clock_identity_required`.
- [x] Run `python -m pytest tests/test_zed_comparison_schema.py -q`; expect missing-module/interface failure.
- [x] Implement schema version 1: immutable sequence/frame IDs, source/SVO/pixel/calibration hashes, relative files, dimensions/stride/color, timestamp values/domains, settings and implementation identities; missing fields use explicit unavailable status.
- [x] Define lossless depth float32 metres (raw little-endian planes are the mandatory interchange; EXR is optional and round-trip tested), original invalid encodings plus metric-valid mask, left reference/frame and axial-Z/range convention; distinguish observed/predicted/historical/approximate pixels. Bind camera/distortion model, pixel centres, stereo transforms and rectified support masks; unknown geometry cannot become all-valid support.
- [x] Encode development/held-out scene AND sequence separation and geometry strata; reject mutating canonical export identities.
- [x] Store provisional margins above with `frozen=false`; required regions, correctness tolerances, zero-error/uncertainty/statistical procedure must be explicitly frozen before held-out execution.
- [x] GREEN: rerun the named command; require all tests pass without SDK installed.
- [x] Review schema/policy against spec and commit only Task 1 files: `git add benchmarks/zed_comparison/{__init__,schema}.py benchmarks/zed_comparison/README.md configs/benchmarks/zed_quality_policy.json tests/test_zed_comparison_schema.py tests/test_zed_schema_io_guard.py && git commit -m "feat: define portable ZED comparison records"`.

**Verified 2026-10-28:** 111 focused schema/I/O tests; 1,868 CPU tests passed, 37 GPU/camera tests deselected. Independent functional and offline I/O reviews approved. Source hashing is bounded and deduplicated within one record; Task 2 must batch shared-source verification across frames. Normalized calibration admission and `policy_ready` are structural gates, not SDK, physical accuracy, support-map or 60 FPS qualification.

### Task 2: Isolated optional SDK export

**Portable subset verified 2026-10-28:** SDK-free `svo_to_frames.py` indexing and batch `load_records(paths, expected=...)` are implemented. A sealed `export.json` enumerates record SHA256/size descriptors; deterministic `index.json` groups exact input identities and remains quality-unqualified. Shared artifacts are hashed once per batch with file-change checks and per-record mapping cleanup. Independent functional/performance reviews passed; 1,928 CPU tests passed, 37 GPU/camera tests deselected. User explicitly authorized isolated SDK benchmarking; distribution terms and source/model boundaries remain inventoried.

**Native diagnostic prerequisite verified 2026-10-28:** Default-OFF CMake isolation and the C++ exporter build against installed SDK 5.5.0. A bounded two-frame `NONE` replay captured exact raw/rectified BGRA, effective settings/calibration and timestamps from the official ZED 2 lake sample (2208×1242 per eye, nominal 15 Hz). Independent functional and CPU serialization reviews preceded capture. The SDK does not expose a documented per-eye rectification rotation or geometric support map in the inspected API. Therefore `--operation capture-diagnostic` produces unqualified diagnostic metadata only, and canonical export remains rejected. No depth-mode quality, physical camera or 60 FPS claim follows from this capture.

- [ ] Resolve raw pose direction and each eye's raw-to-rectified rotation against retained pixels; report distributed training/validation residuals and both pose hypotheses. Interior fit alone cannot establish SDK border/interpolation support.
- [ ] Establish geometric support semantics from a documented footprint or independently validated sampling model with conservative uncertainty margins; alpha and black pixels are not support evidence. Preserve unresolved support as unavailable rather than fabricate masks.
- [ ] Admit actual SDK canonical records only after these gates, including honest legacy-SVO versus SVO2 identification, axial-depth validation and executed-mode availability. Diagnostic export is an intermediate implementation, not Task 2 completion.

**Files:** Create `benchmarks/zed_comparison/export_sdk.cpp`, `benchmarks/zed_comparison/CMakeLists.txt`, `benchmarks/zed_comparison/svo_to_frames.py`, `tests/test_zed_svo_to_frames.py`; modify root `CMakeLists.txt` optional inclusion and `docs/license-evaluation.md` host/export boundary.
**Interfaces:** Consumes Task 1 records/policy and local SVO. Produces exporter CLI `export_sdk --svo PATH --output DIR --mode MODE --profile instantaneous`; Python `index_export(root: Path) -> Path` validates/indexes its record without invoking depth algorithms.

- [ ] RED: add `test_missing_sdk_portable_index_works`, `test_unsupported_mode_is_not_substituted`, `test_confidence_direction_and_unavailable_texture_map`, `test_effective_calibration_change_splits_stratum` using synthetic exporter output.
- [ ] Run `python -m pytest tests/test_zed_svo_to_frames.py -q`; expect missing wrapper/interface failure.
- [ ] Add `STEREOFORGE_BUILD_ZED_REFERENCE=OFF`; optional isolated target discovers SDK only when enabled. No SDK linkage, headers or CUDA flags propagate to core targets.
- [ ] Implement C++ export of lossless original/rectified RGB, float depth/invalid values/masks, raw confidence (low trustworthy), effective raw/rectified calibration plus resolved playback file, timestamps/domains and frame IDs. Record source codec/lossless attestation separately; lossless exported PNGs do not prove that an unknown SVO codec was lossless. Disable real-time replay pacing.
- [ ] Record SDK/model/driver/device/OS, self-calibration changes, effective depth range, confidence/texture/stabilization/tracking/filling settings, precision and graph flags with actual fallbacks; export all available named modes including NEURAL_LIGHT, NEURAL, NEURAL_PLUS and supported PERFORMANCE. Unavailable or unexposed values are explicit; do not infer proprietary model architecture.
- [ ] Inventory approved NVIDIA/Ubuntu host and installer/model/sample/export terms locally; sanitize serials/private paths. Keep artifacts ignored and document redistribution findings without blanket clearance.
- [ ] GREEN: rerun Python tests; `cmake -S . -B build/zed-portable -DSTEREOFORGE_BUILD_ZED_REFERENCE=OFF` must configure without SDK discovery. On approved SDK host only, export a bounded SVO and validate every hash.
- [ ] Review effective settings and licensing inventory; stage explicit Task 2 files and commit `feat: add isolated SDK reference exporter`.

### Task 3: Native already-rectified RGB input

**Files:** Create `include/stereoforge/daxa_rectified_input.hpp`, `src/daxa_rectified_input.cpp`, `python/stereoforge/shaders/daxa/rectified_input.slang`, `apps/daxa_depth_rectified_benchmark.cpp`, `tests/daxa_rectified_input.cpp`, `tests/test_daxa_rectified_input_hotpath.py`; narrowly modify `include/stereoforge/{capture,daxa_frame,daxa_depth,daxa_refinement}.hpp`, `src/{capture,daxa_depth,daxa_refinement}.cpp`, root `CMakeLists.txt`; a separate `apps/daxa_depth_rectified_benchmark.cpp` selector preserves the frozen raw-input application.
**Interfaces:** Consumes Task 1 validated `RectifiedRgbFrame` (original/rectified RGB spans, support, IDs, calibration, timestamps). Produces `DaxaRectifiedInput::enqueue(frame) -> DaxaRectifiedInput::Frame` on a setup-bound context; `Frame::full()` and `Frame::coarse()` expose retained `DaxaFrame` leases. Add a `DaxaDepthPipeline` constructor overload taking a prototype `DaxaFrame`, `enqueue_preprocessed(const DaxaFrame&) -> DaxaDepthPipeline::Frame`, and `DaxaRefinement::process(coarseDepth, fullRectifiedFrame) -> DaxaRefinement::Frame`. This explicitly connects the producer to both existing consumers.

- [x] RED: add native cases `rectified_skips_decode_remap`, `refinement_never_remaps_sdk_pixels`, `full_coarse_share_source_identity`, `original_rgb_preserved`, `reject_geometry_context_mismatch`, `lease_survives_submit_failure`, `bounded_slot_reuse`, `metadata_only_source_rejected_by_yuyv_capture`; register target `daxa_rectified_input_contracts` and test `daxa_rectified_input_cpu_contracts` before implementing types.
- [x] Run `cmake --build build/daxa --target daxa_rectified_input_contracts`; expect compilation failure on the missing input API. After implementation assert `ctest -N --test-dir build/daxa -R '^daxa_rectified_input_cpu_contracts$'` lists exactly one test before executing it; no-tests is never a passing RED/GREEN check.
- [x] Implement RGB-to-luma input path with explicit color/stride contract, bypassing capture decode and remap; do not re-encode RGB as YUYV or change frozen capture inputs.
- [x] Preserve original post-ISP RGB for wide synthesis; use the existing eight-plane `DaxaFrame` lease/layout. Add explicitly tagged canonical-RGB source metadata to `CaptureFrame`; raw YUYV capture/replay admission rejects this tag rather than fabricating packed bytes. Retain source metadata/calibration/host backing through the adapter and shared stereo/metric completion. Preserve the existing raw-capture constructor/enqueue behavior.
- [x] Configure optional 2× coarse generation at setup: downsample the canonical rectified input with a declared filter/support and half-pixel-scaled projection, retaining the same original RGB resources. The refinement overload consumes exact full rectified input and bypasses its current target-remap pass. Label this sampling profile separately from F1's direct original-camera remap; remeasure its quality and cost.
- [x] Reject wrong dimensions/calibration/foreign context; enforce bounded preallocated 2–3 slots, partial-submit cleanup and no per-frame allocation/readback/wait in the resident path.
- [x] GREEN: build new target and run named CPU contracts. A separate worker must complete the mandatory hot-path performance pass with an actual-source allocation/wait micro-probe and static guard before root GPU admission. Freeze source, actual compiler dependencies, binaries, SPIR-V and fixtures; qualify awkward strides, exact pixel/luma/support equality, core/sync and retained-consumer stress separately.
- [x] Review hot-path/resource ownership and exact unchanged decode/remap control lane; benchmark app must emit full independent SGM depth via `enqueue_preprocessed`, then exercise refinement from its coarse frame. Stage the verified native input checkpoint; full SGM and refinement run through the separate canonical benchmark selector.

**Core checkpoint 2026-10-28 (Task 3 remains open):** Actual root GPU
128×64 D32/P4 verification matched all eight producer planes, SGM integers,
masks and IDs exactly; metric floats met `2e-6`. The retained-consumer run used
300 unique pre-admitted records cycling three synthetic textures: accepted 300,
completed 300, unique outputs 300, gaps 0, maximum in flight 2, with strict
View-only retention. Core/synchronization validation reported zero errors and
warnings through teardown on RADV 8060S / Mesa 26.2.3. Five native CPU contracts
and raw GPU depth/refinement/View contracts (3/3) passed. Full CPU regression
passed 2,038 tests, with 37 GPU/camera tests deselected and no skips (119.70 s).

The mandatory independent actual-source mock performance/lifecycle probes
passed 76 tests; canonical View setup/control probes passed 25. Warm application
allocations, implicit waits and readbacks were zero. Six fixed CPU staging
copies of 12.9024 MB measured 0.254–0.272 ms per frame, excluding offline
admission. Mock device/graph calls exclude driver, upload, shader and GPU
completion costs. These are bounded CPU/control and native 128×64 correctness
results, without FPS, 720p, live-camera or scientific-quality qualification.
Evidence: `artifacts/zed-task3/{adapter-gpu-stream-v1.log,adapter-cpu-v6.log,raw-gpu-regression.log}`
and `artifacts/zed-task3/performance/{summary.md,view-setup/summary.md}`.

**Native analytic implementation checkpoint:** The portable bridge, cold
C++ loader and `daxa_depth_rectified_benchmark` now run full independent
D128/P4 SGM or coarse D64/P4 plus full-grid refinement without YUYV encoding
or raw remapping. The existing raw application remains frozen. Independent
functional and CPU performance reviews preceded both 128×64 analytic GPU
variants (three measured iterations plus one warmup, zero validation errors or
warnings through teardown). The connected 300-record stress and raw GPU
regressions remain green. Full CPU regression: 2,140 passed, 37 deselected,
no skips, 130.89 s. Both unused-coarse-ID admission and subpixel-reporting
regressions have targeted RED/GREEN guards.

The native implementation lane is verified for analytic known geometry;
SDK geometry/support admission, held-out quality evaluation and sustained
delivery remain separate open gates. The new loader/benchmark contract and
reproduction commands are in [canonical record input](../../canonical-record-input.md). See
[native input reproduction](../../native-rectified-input.md#verified-boundary-and-reproduction);
GPU reproduction unsets `VK_LAYER_ENABLES` with `env -u` to avoid inherited
obsolete-setting warnings.

### Task 4: Independent truth and quality-first comparisons

**Files:** Create `benchmarks/zed_comparison/{compare_quality,truth,quality_metrics,native_observation}.py` with corresponding evaluator, truth, metrics, adapter and materialization tests; modify `configs/benchmarks/zed_quality_policy.json` only through recorded development-policy freeze.
**Interfaces:** Consumes Task 1 records/policy, Task 3 outputs and independent truth masks. Produces `compare_quality(records: list[dict], truth: dict, policy: dict) -> dict` with metrics, denominators, uncertainty and qualified/unqualified decisions.

- [x] RED: add `test_off_axis_range_to_axial_z`, `test_invalid_sparse_truth_denominators`, `test_zero_reference_error_policy`, `test_confidence_direction_coverage_curve`, `test_d128_range_strata`, `test_unexecuted_sdk_never_parity_pass`.
- [x] Run `python -m pytest tests/test_zed_compare_quality.py -q`; expect missing evaluator failure.
- [x] Implement relative depth MAE by distance, absolute p95, intersection-valid error, fixed-truth completeness, false accepts, thin recall/boundary F1, support populations and error-versus-coverage/reliability curves; include surface reconstruction error only where independently supported geometry exists. Confidence remains an uncalibrated score.
- [ ] Sweep matching and texture-filter settings independently on development data; preserve raw scores/direction and effective settings. Do not equate SDK low-is-good thresholds with StereoForge high-is-good scores or invent a missing texture map.
- [x] Require independent truth registration/timestamp uncertainty; sparse LiDAR supports only actual samples. Left aligned outputs are primary; independent-right diagnostics stay auxiliary.
- [ ] Run full-resolution independent SGM D128 first on canonical exported pairs, then each executed SDK mode, then Fast60; changed pixel/calibration identity reruns a separate stratum, never silently warps SDK depth.
- [x] Report `d=fx*b/Z+cx_offset`, absolute search bounds and common-supported/full 0.3–8 m results; retain out-of-range failures. Any shifted/local near search beyond 128 is a separate profile.
- [ ] Freeze thin/boundary/near definitions, tolerances and paired scene/sequence statistical uncertainty on development data before held-out evaluation; preserve provisional 2 pp / 10% / no-excess-boundary limits without inventing vendor parity.
- [ ] GREEN: rerun named tests; root admits frozen analytic comparisons before real records. Review missing-data decision paths and commit `feat: evaluate stereo quality against independent truth` with explicit Task 4 files.

**Implementation checkpoint 2026-10-28:** The [detailed evaluator contract](2026-10-28-independent-quality-evaluation.md) now has strict independent truth admission, exact native observation export, fixed-support metrics, confidence curves, deterministic scene/sequence bootstrap and gated coordinator decisions. Full CPU regression: 2,290 passed, 37 deselected, no skips. Independent functional and performance reviews passed; the actual 128×64 analytic comparison remains unqualified because the policy is provisional and no SDK reference has executed. Software acceptance tests are not SDK/held-out quality qualification. Development sweeps, real SDK admission/comparison, frozen policy and held-out physical evaluation remain open. [Usage and measured scope](../../independent-quality-evaluation.md).

### Task 5: Physical and temporal controlled lanes

**Files:** Create `benchmarks/zed_comparison/evaluate_temporal.py`, `tests/test_zed_evaluate_temporal.py`; extend `compare_quality.py` and its tests for calibration residuals, and Task 1 schema tests for physical reference metadata.
**Interfaces:** Consumes Task 1 sequences and Task 4 per-frame metrics. Produces `evaluate_temporal(records: list[dict], truth: dict, policy: dict) -> dict` and `evaluate_epipolar(matches: np.ndarray, geometry: dict) -> dict` (matches are Nx4 left/right pixel coordinates) with median/p95 vertical residuals, static jitter, ghosting, recovery and sequence uncertainty.

- [x] RED: add `test_duplicate_frame_not_fresh`, `test_clock_mismatch_rejected`, `test_historical_depth_not_observed`, `test_motion_occlusion_recovery`, `test_sparse_reference_not_dense_edges`.
- [x] Run `python -m pytest tests/test_zed_evaluate_temporal.py -q`; expect missing temporal evaluator failure.
- [x] Implement separate temporal-off/on strata with effective stabilization/tracking/filling settings; never combine them into instantaneous parity.
- [x] Define controlled acquisition records: surveyed planes 0.3–8 m, 1–8-pixel thin geometry, edges, low/repeated texture, reflective/dark materials, 10–30 s static sequences and camera/object motion; use independent scanner/LiDAR support for complex geometry only with recorded registration uncertainty. Retain reference geometry/registration/timestamp uncertainty.
- [x] Add factory-versus-effective calibration ablation and vertical epipolar residual measurement; do not use calibration speculation to explain ideal analytic failures.
- [x] Add `test_epipolar_residual_identity_and_vertical_offset` to `tests/test_zed_compare_quality.py` and rerun that file with the temporal tests; retain correspondence/inlier-selection provenance separately from independent metric truth.
- [x] GREEN: rerun named tests using synthetic sequences; root schedules physical acquisition only with inventoried devices and frozen development/held-out partitions.
- [x] Review correlation-aware sequence aggregation and physical-truth limitations; stage explicit Task 5 files and commit `feat: separate physical and temporal quality lanes`.

**Software checkpoint 2026-10-28:** Full CPU regression: 2,423 passed, 37 GPU/camera tests deselected, no skips (133.46 s). Controlled temporal evaluation, strict physical-reference metadata and epipolar diagnostics are implemented with independent functional and CPU performance reviews. The software distinguishes historical output from fresh observations, rejects selected method subsets and contradictory stationary poses, preserves censored recoveries, and requires explicit temporal-off settings for instantaneous parity. Sparse physical references cannot qualify dense thin/boundary labels. The [usage and deterministic evidence](../../controlled-temporal-evaluation.md) retain all synthetic/offline limits. Physical collection, real SDK admission, frozen development/held-out datasets and native temporal qualification remain open; this checkpoint does not establish measured depth quality or 60 FPS.

### Task 6: Matched performance records after quality

**Files:** Create `benchmarks/zed_comparison/performance.py`, `tests/test_zed_performance.py`; extend isolated `export_sdk.cpp` with `--operation benchmark` and narrowly modify `apps/daxa_depth_benchmark.cpp` and `apps/daxa_refinement_benchmark.cpp` report metadata only where missing.
**Interfaces:** Consumes quality-qualified profile IDs, frozen records and native/SDK timing reports. Produces `validate_timing_lane(record: dict) -> None`, `summarize_timing(records: list[dict]) -> dict`.

- [x] RED: add `test_reject_mixed_completion_boundaries`, `test_cold_separate_from_warm`, `test_cross_platform_not_speedup`, `test_incomplete_interleaving_unqualified`.
- [x] Run `python -m pytest tests/test_zed_performance.py -q`; expect missing interface failure.
- [x] Record cold preparation, warmed GPU stages/upload/readback/synchronization/CPU/file-I/O/publication separately; retain raw samples and p50/p95/p99/max, allocation/cache mode and actual resident/host-output completion boundary.
- [x] Separate the image/depth export pass from timing: no disk writes, diagnostic image retrieval or unnecessary output requests in a measured compute lane. The SDK benchmark uses the same frozen session/settings as its quality export, with actual completion and replay pacing recorded; its internal preprocessing remains opaque.
- [x] Mark unavailable SDK stages unavailable; opaque grab cost is not isolated matching. Same-GPU NVIDIA Vulkan/SDK is a runtime comparison; AMD/NVIDIA are platform operating points.
- [ ] GREEN: portable tests pass; root still must admit complete controlled interleaved timing only after quality/policy gates, with source/compiler/SPIR-V/input freezes and before/after contention admission.
- [x] Review transfer-inclusive comparability; stage explicit Task 6 files and commit `feat: record matched stereo performance boundaries`.

**Software checkpoint 2026-10-28:** Strict raw timing admission, native additive metadata and an isolated SDK timing-only opaque API operation are implemented. 2,507 CPU tests passed, 37 GPU/camera tests deselected, no skips (140.43 s); independent functional/performance reviews passed. Three actual native targets and the optional SDK 5.5.0 target compiled; SDK version query ran without an exposed GPU. Native measured loops are unchanged. SDK sampling skips diagnostic buffers, retrieval and file I/O. Synthetic ABBA exercises and retained old-report rejection are software evidence, not executed matched GPU/SDK performance. Quality-qualified SDK/canonical inputs, frozen physical/held-out policy and actual complete interleaved measurements remain open. [Usage, limits and reproduction](../../matched-performance-records.md).

### Task 7: Sustained delivery, ROS 2 and final report

**Files:** Create `benchmarks/zed_comparison/report.py`, `benchmarks/zed_comparison/ros2_delivery.py`, `apps/ros2_depth_publisher.cpp`, `cmake/ROS2.cmake`, `tests/test_zed_report.py`, `tests/test_zed_ros2_delivery.py`, `docs/zed-quality-benchmark.md`; optional default-OFF root CMake inclusion only.
**Interfaces:** Consumes Tasks 4–6 reports and native observation leases. Produces optional C++ `stereoforge_ros2_depth_publisher` (`rclcpp`/`sensor_msgs`), `build_report(records: list[dict], policy: dict) -> dict` and `summarize_delivery(events: list[dict]) -> dict`; Python remains offline evaluation, not the live pipeline.

- [ ] RED: add `test_599_seconds_unqualified`, `test_repeated_depth_not_unique`, `test_queue_growth_rejects_sustained`, `test_missing_quality_blocks_promotion`, `test_ros_units_qos_and_frame_identity`.
- [ ] Run `python -m pytest tests/test_zed_report.py tests/test_zed_ros2_delivery.py -q`; expect missing interfaces failure.
- [ ] Implement ≥600 s nominal60 Hz delivery accounting for unique depth/RGB, drops/gaps/observation age/queue growth/memory, bounded 2–3 slots and separately fixed View; include RGB-only teleoperation as a distinct mode.
- [ ] Add optional native ROS 2 `32FC1` metre depth with matched QoS/resolution and explicit transfer-inclusive host publication cost. Image `header.frame_id` is the camera coordinate frame; mapped capture stamps remain capture stamps. Source observation IDs and capture/GPU-completion/publication timestamps live in a companion trace. Reject unknown clock mappings and do not republish historical depth as a fresh measurement.
- [ ] Run the unchanged standalone ZED Topic Benchmark separately against SDK and StereoForge depth/RGB topics for600s, typed mode and matched QoS depth1. Keep any capture-header age separate from publication-to-subscriber middleware delay derived from correlated traces. Finite composed benchmarks stop their container; use a standalone process. ROS success cannot satisfy missing GPU/physical gates.
- [ ] Generate reproducible commands, relative artifact hashes, hardware/software identity, settings, policy/partition freeze and explicit open gates; SDK absence yields external parity unqualified while portable results remain usable.
- [ ] GREEN: rerun Task 7 tests plus all portable `tests/test_zed_*.py`; root alone admits physical/live/ROS runs and reviews the complete report before any promotion.
- [ ] Review whole-branch contracts and license boundaries; stage explicit Task 7 files and commit `docs: report quality and sustained stereo delivery gates`.
