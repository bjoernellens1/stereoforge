# Controlled temporal and calibration evaluation implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development. The active end-to-end implementation goal already authorizes execution.

**Goal:** Implement Task 5's controlled temporal, physical-reference and epipolar diagnostics without inventing fresh measurements, correspondence truth or physical qualification.

**Architecture:** Offline NumPy evaluation consumes validated portable records and independent truth. A separately hash-bound temporal protocol supplies correspondence tracks and controlled motion/event declarations. Physical reference manifests add survey/registration evidence; epipolar diagnostics remain separate from metric ground truth. Native capture/GPU execution is unchanged.

**Tech Stack:** Existing Python/NumPy/portable file utilities. No new package or GPU backend.

**Spec:** [Task 5 of the quality benchmark plan](2026-10-28-zed-quality-benchmark.md#task-5-physical-and-temporal-controlled-lanes) and [independent quality contracts](2026-10-28-independent-quality-evaluation.md).

## Global constraints and review focus

- All evaluation is offline. No GPU/SDK imports, camera controls or full-host round trips are added to the native frame loop.
- Validate files, exact input/calibration/source/capture bindings and partitions; no implicit cwd. Preserve original bytes and source identities.
- Duplicate delivery is not fresh acquisition. Fused/reprojected or old-source depth is not a fresh independent measurement, even if numerically correct.
- Missing settings, clocks, uncertainty, motion/correspondence or frozen definitions yield explicit unavailable/unqualified output.
- Never treat sparse unknown samples as negatives or dense boundaries; never infer physical correctness from a zero epipolar residual.
- Per-frame arrays and per-sequence bounded track state only. Avoid Python loops over pixels; report offline cost separately from pipeline performance.
- Review repeat inputs with conflicting outputs, clock changes, capture gaps, tracks moving between pixels, old depth with reset age, and event targets not recovered before sequence end.

## Task 1: Pure epipolar diagnostics

**Owner/files:** Epipolar worker owns `benchmarks/zed_comparison/epipolar.py` and `tests/test_zed_epipolar.py`. Root re-exports `evaluate_epipolar` from `compare_quality.py` and adds the named compatibility test.

`evaluate_epipolar(matches: np.ndarray, geometry: dict) -> dict` accepts finite Nx4 `[uL,vL,uR,vR]` coordinates. Empty Nx4 is unavailable, not zero error. No outlier rejection or fitting occurs. All points must be inside declared image footprints `[-0.5,width-0.5) × [-0.5,height-0.5)`.

The geometry envelope has exactly `{schema, profile, space, geometry, calibration_sha256, original_pixel_identity, correspondence_provenance}`: schema `stereoforge.zed_comparison.epipolar.v1`, profile `factory|effective|analytic`, space `rectified`, and geometry is the existing portable perspective/axial-Z left geometry. Calibration/original identities are lowercase SHA256. Correspondence provenance is exactly `{method, selection, population_identity}` with nonempty labels and Availability[SHA256]. These are declared provenance, not independent truth; normalized little-endian float64 matches receive an actual SHA256 in the report.

Report signed/absolute vertical median (ordinary median), absolute p95 (nearest rank), max and horizontal disparity summaries/counts. Always `metric_truth=false`, `quality_qualified=false`. Reject raw-space vertical residual claims, malformed geometry, nonfinite/bool matches and unsupported conventions.

`compare_epipolar_profiles(results: list[dict]) -> dict` retains each profile diagnostic. Deltas are available only for two results with identical original-pixel identity, available identical population identity, selection/method and count. Otherwise record unmatched populations, with no inferred improvement. Neither case qualifies metric geometry.

- [x] RED/GREEN: identity, known vertical offset, signed shifts, empty/invalid arrays, coordinate bounds, raw rejection, p95 policy and different-selection populations.

## Task 2: Physical reference metadata and acquisition protocol

**Owner/files:** Physical worker owns `benchmarks/zed_comparison/physical_reference.py`, `tests/test_zed_physical_reference.py` and `configs/benchmarks/zed_physical_acquisition.json`.

`validate_physical_reference(document: dict, *, root=None, files=None) -> dict` validates strict metadata and consumed FileDescs. Explicit `root` is required for standalone file validation; optional internal `files` is an existing schema._Files shared cache with that root. Return `{status, reasons, metadata, checked_paths, sampling, dense_geometry_labels_allowed, quality_qualified:false}`. Status is metadata_ready only when required files/registration/timestamp evidence are available; otherwise unqualified. This never proves survey accuracy.

The strict document has exactly:

```
{schema: "stereoforge.zed_comparison.physical_reference.v1",
 kind: "surveyed_plane"|"scanner"|"lidar", independent:true,
 artifact: FileDesc, calibration_sha256: SHA256, units:"m",
 coordinate_axes:"x_right_y_down_z_forward", sampling:"dense"|"sparse",
 interpolated:false, dense_geometry_labels:bool, labels:Availability[FileDesc],
 registration:Availability[{method,transform_to_left:rigid4x4,max_error_m,justification}],
 timestamp_uncertainty:Availability[{max_error_ns,justification}],
 survey:Availability[{plane_normal_left:[3],plane_offset_m,axial_intercept_m,
                       absolute_error_m,method}]}
```

Survey normal is unit within 1e-6; `axial_intercept_m = -offset/normal.z > 0` within abs1e-12+rel1e-9. Bounds are finite nonnegative. A surveyed_plane requires survey available for metadata_ready. Scanner/LiDAR do not pretend to be surveyed planes. Dense labels require sampling=dense and an available independently measured labels artifact; sparse references cannot claim dense geometry labels, and interpolation is rejected. File hashing is bounded/streamed with observed fingerprints and no dense array invention.

The acquisition JSON remains `frozen:false` and a planning artifact: include 0.3–8m surveyed frontal planes, 1–8px thin geometry, edges, low/repeated texture, reflective/dark materials, static10–30s and camera/object motion, factory/effective calibration profiles and development/held-out partition placeholders. Device/reference availability stays explicitly unavailable until inventoried/measured. It must not schedule camera work or fabricate collected data.

- [x] RED/GREEN metadata, missing root/evidence, calibration hash, unsafe/symlink/mutated files, bad survey/pose/bounds and `test_sparse_reference_not_dense_edges`.

Root integration adds an OPTIONAL `physical_reference: Availability[FileDesc]` field to each truth.v1 frame, preserving old documents. Descriptors/reference payload paths are relative to explicit truth root. Measured truth without metadata_ready physical reference can be scored but cannot scientifically qualify. Analytic truth cannot masquerade as a physical reference. Reference calibration must match the frame, and registration/timestamp bounds must not understate the reference bounds. Integrate checked paths into frame-local and final fingerprint checks through the shared file cache. Root adds a lazy `schema.validate_physical_reference` convenience API and physical-source schema tests; raw physical capture without truth remains valid capture data.

## Task 3: Controlled temporal evaluator

**Owner/files:** Temporal worker owns `benchmarks/zed_comparison/evaluate_temporal.py` and `tests/test_zed_evaluate_temporal.py`.

`evaluate_temporal(records, truth, policy, *, root=None, truth_root=None, protocol=None, protocol_root=None) -> dict` follows Task4 explicit root/common-root-list rules. No implicit stationary-camera assumption. `temporal_settings(record) -> dict` standardizes effective settings from `implementation.settings.temporal` only; missing fields remain unavailable.

Effective temporal fields are exactly `{stabilization:Availability[number0..100], tracking:Availability[bool], filling:Availability[bool]}`. All available and stabilization=0/filling=false defines temporal_off; stabilization>0 or filling=true defines temporal_on; otherwise unknown. Preserve strengths/tracking values. Group full implementation/calibration/sequence/temporal settings separately; never pool on/off/unknown.

The strict protocol has exactly:

```
{schema:"stereoforge.zed_comparison.temporal_protocol.v1", frozen:bool,
 partition_digest:SHA256,
 provenance:{independent:true,kind:"analytic"|"measured",artifact:FileDesc,
             generator:Availability[FileDesc]},
 limits:{max_gap_ns:positiveInt,pose_timestamp_tolerance_ns:nonnegativeInt,
         track_position_tolerance_m:nonnegativeFloat,max_tracks:Int1..2097152,
         ghost_change_absolute_m:positiveFloat,ghost_change_relative:nonnegativeFloat},
 sequences:[{scene_id,sequence_id,partition,rig_motion:"stationary"|"known"|"estimated"|"unavailable",
             scene_motion:"static"|"dynamic",calibration_profile:"factory"|"effective"|"analytic",
             frames:[{input_identity,world_from_left:Availability[{matrix:rigid4x4,capture:Timestamp}],
                      track_ids:Availability[PlaneDesc_i64_le]}],
             recovery_events:[{event_id,kind:"disocclusion"|"motion_reset"|"capture_gap"|"appearance",
                               start_input_identity,target_track_ids:[nonnegativeUniqueInt64],justification}]}]}
```

Protocol files use explicit protocol_root, strict existing descriptor/path/hash rules, 16MiB JSON and ≤512MiB plane extent per frame. Track IDs -1 mean unknown, nonnegative IDs must be unique within one frame and supported by independent sampled truth. Analytic zero-motion/bounds need a bound generator artifact. Validate capture/pose clocks and pose timestamp tolerance, rigid units in metres. Protocol frames exactly bind the unique input set for each sequence/calibration stratum. For known rig motion require all poses; for static scenes verify repeated tracked world positions within track_position_tolerance_m when poses exist. Implicit pixel tracks are allowed ONLY for declared stationary rig + static scene, with same dimensions/calibration and constant sampled truth within tolerance. Otherwise explicit independent tracks are required for aligned jitter/recovery. Estimated/unavailable motion is separately reported and cannot count as known-motion qualification.

Validate individual portable records, then deduplicate exact repeated measurement/output content (ignoring descriptor paths) before partition validation; report repeat counts and count each input once. Conflicting repeated output for the same method/input is rejected. Unique captures in a sequence have strictly increasing timestamps and source IDs, one known capture clock; mismatches are rejected. Gaps beyond max_gap_ns reset jitter segments and are reported explicitly; do not silently bridge them.

Freshness: valid pixels with all metadata available, age0, source_id=current source and provenance in {1 STEREO,4 NEURAL,5 LEFT_AGGREGATED} are current-source observations; retain derived code5 as a separate class. REPROJECTED/FUSED and old-source/aged depth are historical, never independently fresh. Missing metadata is unknown, not fresh. Record inconsistent ages for seen historical sources and unsupported/future source identities; these block qualification. Fused ancestry is incomplete in portable single-ID records and must not become independent evidence. Track output-error jitter and fresh-only error jitter separately; historic output may be stable without new information.

Compute streaming per-track float64 Welford moments of `predicted axial Z - independent current truth Z`, using vectorized ID association and bounded ≤max_tracks state. Counts require ≥2 unique captures. Raw depth standard deviation is reportable only for explicitly static rig+scene. Report populations, track counts, bias and error std summaries with no pooled p95 fabrication. Do not retain all frame arrays. Recoveries use independent target track IDs fixed at event start; require targets supported at start, preserve nonrecovered targets as censored, and report first-correct-geometry time separately from first-fresh-correct-measurement time. Correctness uses existing frozen policy definitions; missing threshold leaves ghost/recovery decisions unavailable.

Ghosting is defined only on a declared stationary image grid where independent current sampled/known-negative truth differs from previous known truth by the protocol change threshold. A predicted point close to prior truth and incompatible with current truth is a ghost-compatible output. Unknown current truth is not negative. Known-rig-motion sequences still receive aligned track error/recovery metrics; unprovided dense visibility/reprojection geometry must leave image-grid ghosting unavailable rather than comparing unrelated pixel locations. This limitation is explicit in the report.

Temporal quality qualification requires frozen protocol+policy, matching partition digest, held-out data, known/stationary validated correspondence, complete effective settings and metadata/uncertainty. Otherwise metrics remain useful but unqualified. Do not call this SDK parity or independent VIO qualification.

- [x] RED/GREEN named `test_duplicate_frame_not_fresh`, `test_clock_mismatch_rejected`, `test_historical_depth_not_observed`, `test_motion_occlusion_recovery`; plus old-age inconsistency, gaps, track movement/duplicate IDs, missing settings/pose/correspondence/threshold, unobserved recovery targets, metadata/file mutations and bounded state.

## Task 4: Integration and measured boundary

**Root ownership:** `truth.py`, `compare_quality.py`, `schema.py` narrow integration; related existing tests, plans, docs/evidence; shared builds, hardware inventory and publication. Other workers must not edit these files.

- [x] Add the physical-reference integration and `evaluate_epipolar` re-export with named `test_epipolar_residual_identity_and_vertical_offset`.
- [x] Enforce instantaneous quality parity only with explicit complete temporal_off settings for BOTH methods; temporal_on and unknown remain separate lanes. Existing positive qualification fixtures must explicitly declare off settings, never infer missing SDK settings. Add tests for on/off mismatch, on/on not instantaneous and unknown metadata.
- [x] Independently review correctness/correlation/physical limitations and complete a separate CPU performance pass with actual-source guards/probes (no pixel Python loops, bounded frame/track lifetime, linear file fingerprints).
- [x] Root creates and evaluates deterministic synthetic sequence(s) with independent motion/track truth, history and fresh recovery, plus epipolar factory/effective diagnostics on known synthetic offsets. These demonstrate evaluator contracts, not native temporal or physical accuracy. No new GPU run is required.
- [x] Full CPU regression, clean diff, publish compact evidence and commands, commit/push and verify CI. Keep physical collection, real SDK/held-out qualification and the overall goal open.
