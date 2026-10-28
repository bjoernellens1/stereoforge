# Independent quality evaluation implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development to implement the tasks below. Execution is already authorized by the active end-to-end goal.

**Goal:** Evaluate identical stereo inputs against independently declared metric truth, reporting error, fixed-support completeness, confidence tradeoffs and honest qualification decisions.

**Architecture:** Cold Python adapters validate files and materialize one input frame at a time. Pure NumPy metrics retain explicit denominators. The coordinator groups exact inputs, implementation settings and calibration strata; paired statistical analysis treats scenes/sequences as units. No evaluation code enters the native GPU loop.

**Tech Stack:** Existing Python/NumPy, portable schema and SHA256 utilities; no new dependency or GPU backend.

**Spec:** Task 4 of [the quality benchmark plan](2026-10-28-zed-quality-benchmark.md), refined by the contracts below.

## Global constraints

- Preserve input/pixel/calibration identities, source IDs, camera clocks and original file bytes. Never use an SDK or full-SGM prediction as independent truth.
- Default policy remains provisional. Do not edit/freeze it or invent vendor execution, physical registration, held-out data or calibrated uncertainty.
- Never interpolate sparse truth. Unknown samples are neither valid truth nor known-negative geometry.
- Primary evaluation uses left rectified axial Z in metres. Ray-range truth requires explicit conversion on the exact same rectified grid. Other coordinate systems require a separately registered truth adapter.
- Requested 0.3–8 m and search-supported populations are separate. Out-of-range failures stay in the requested-range denominator.
- Offline copies/readback are explicit output/evaluation boundaries. Native runtime source/resource ownership and GPU residency remain unchanged.
- Root owns integration, shared builds, GPU jobs, commits and pushes. Workers own separate modules/tests and must preserve other edits.

## Review focus

- Sparse/empty truth: count misses over fixed sampled support; never award a score for empty denominators.
- Frame/calibration/pixel mismatch and observed file mutation: reject rather than resample or silently rebind.
- Confidence ties/direction: include whole tied groups; never make ordering of equal scores improve a method.
- Undefined regions, uncertainty, statistics or absent reference: report unavailable/unqualified, not a passing comparison.
- Two-eye native dumps: preserve exact left float bytes and reject malformed extents, masks, provenance or source age.

## Shared interfaces and truth format

Create `benchmarks/zed_comparison/truth.py`, `quality_metrics.py`, `native_observation.py`, and `compare_quality.py` with corresponding `tests/test_zed_truth.py`, `test_zed_quality_metrics.py`, `test_zed_native_observation.py`, `test_zed_compare_quality.py`.

`compare_quality(records: list[dict], truth: dict, policy: dict, *, root=None, truth_root=None) -> dict` retains the three planned positional arguments. `root` is an explicit common directory or a list of directories matching record order; `truth_root` is explicit. Missing roots produce an unqualified missing-binding result. Invalid documents, content or bindings raise `ValueError`. Never use implicit cwd. Validate portable records and all partitions before scoring. A method key hashes canonical implementation name/version/commit/device/settings; human labels retain name and variant/depth mode. Group comparisons separately by calibration content/geometry stratum and exact `input_identity`.

The strict truth document has exactly:

```
{schema: "stereoforge.zed_comparison.truth.v1", independent: true,
 provenance: {kind: "analytic"|"measured", artifact: FileDesc,
              generator: Availability[FileDesc]}, frames: [TruthFrame, ...]}
```

Each TruthFrame has exactly:

```
{input_identity, source_id, capture: {value_ns, clock_domain},
 partition: "development"|"held_out", calibration: FileDesc,
 rectified_pixel_identity, geometry: <exact portable record geometry>,
 depth_convention: "axial_z"|"ray_range", depth: TruthPlane,
 masks: {sample: MaskPlane, geometric_support: Availability[MaskPlane],
         known_negative: Availability[MaskPlane], visible: Availability[MaskPlane],
         occluded: Availability[MaskPlane], mixed: Availability[MaskPlane],
         boundary_edges: Availability[MaskPlane]},
 regions: {<name>: Availability[MaskPlane], ...},
 registration: Availability[{method, transform_to_left: rigid4x4,
                              max_error_m, justification}],
 timestamp_uncertainty: Availability[{max_error_ns, justification}]}
```

All FileDesc/Availability and strict JSON/path/hash rules match the existing portable schema. TruthPlane uses PlaneDesc fields and `f32_le_m` or truth-only `f64_le_m`; masks use `u8_mask`. Active masks are binary. Positive finite metric truth is mandatory on `sample`; invalid/unknown bytes remain unscored. `known_negative` must not intersect `sample`. Geometry and calibration match the portable record exactly; normalized calibration validates through existing schema helpers. `rectified_pixel_identity` is the path-independent digest documented by the canonical bridge (rectified RGB + both support descriptors). Capture/source/partition/input bindings must match. Registration metadata describes an already applied registration; the evaluator does not reproject arbitrary sensor samples. Analytic zero bounds need a nonempty justification and bound generator artifact; missing bounds block qualification.

`truth.load_truth(truth, records, *, root) -> TruthBundle` validates metadata/bindings/files with bounded hashing. `TruthBundle.frame(input_identity)` is a context manager yielding one materialized frame dictionary, then closes mappings. Expose `metadata`, `geometry`, `depth_m` (float64 axial), `sample`, optional boolean `geometric_support`, `known_negative`, `visible`, `occluded`, `mixed`, `boundary_edges`, `regions` (name -> mask or None), `registration`, `timestamp_uncertainty`. No whole-dataset array retention. Maximum 512 MiB of declared plane extents per truth frame; strict JSON remains bounded at 16 MiB. Check observed file fingerprints around materialization. Range conversion is `Z=range/hypot(1,(u-cx)/fx,(v-cy)/fy)` using exact rectified left calibration. Missing/unsupported or nonfinite numerical conversions must not fabricate valid geometry.

## Task 1: Truth admission and normalization

**Owner/files:** Truth worker owns `truth.py` and `test_zed_truth.py` only.

- [x] RED: missing module/API; tests for off-axis range conversion, f32/f64 awkward strides, sparse sample masks, unknown versus known-negative, empty samples, mismatched input/calibration/pixel/source/time/partition, duplicate truth frames, nonbinary masks, symlink/hash mutation, missing uncertainty and negative bounds.
- [x] Implement the strict truth contract and one-frame materialization without altering portable records.
- [x] GREEN: `.venv/bin/python -m pytest tests/test_zed_truth.py -q`. Do not edit other workers' fixtures/modules.

## Task 2: Pure metrics and paired statistical decisions

**Owner/files:** Metrics worker owns `quality_metrics.py` and `test_zed_quality_metrics.py` only.

`evaluate_frame(depth, validity, confidence, confidence_direction, truth_frame, policy, *, search_bounds=None) -> dict` consumes exact aligned arrays. `search_bounds` is an available `{min_disparity, max_disparity_exclusive}` or None, in full-grid pixels. `paired_decision(units, policy) -> dict` consumes a list of `{scene_id, sequence_id, region, candidate, reference}` summaries (same metrics schema); no per-frame independence assumption.

- [x] RED named tests: `test_invalid_sparse_truth_denominators`, `test_confidence_direction_coverage_curve`, `test_d128_range_strata`, `test_zero_reference_error_policy`, tied confidence, empty regions and boundary definitions.
- [x] Report counts/sums and MAE, relative MAE, absolute p95/max/RMSE, ordinary completeness and threshold-correct completeness. Primary fixed sampled truth in the requested range includes missing predictions and out-of-search truth. Also report full sampled domain, common geometric-supported and search-supported lanes, unknown accepts, unsupported accepts, explicit known-negative false accepts and distance bins [0.3,0.5,1,2,3,5,8] plus outside-range populations. `d=fx*b/Z+cx_left-cx_right` defines the search mask. Never substitute prediction validity for truth support.
- [x] Correctness comes only from supplied policy definitions: disparity-pixel tolerance or max(absolute_m, relative*Z). Missing definitions leave threshold-dependent fields unavailable while ordinary errors/coverage remain reportable. Absolute p95 uses nearest rank ceil(0.95*N)-1. Surface reconstruction remains explicitly unavailable unless independently supported surface geometry is supplied; a depth-map difference is not surface distance. Errors use valid predicted/sample intersections. Count accepted incorrect depth separately from known-negative accepts; unknown pixels never enter false-accept numerators.
- [x] Regions include implicit `all` plus declared masks. Thin pixel recall is correct completeness on the thin mask. Boundary false acceptance uses the declared boundary region; boundary F1 additionally requires independent `boundary_edges` and policy boundary definition fields `absolute_jump_m`, `relative_jump`, `tolerance_pixels` (integer 0..8). Detect adjacent valid depth jumps, compare only independently sampled neighborhoods, and use bounded pixel tolerance without treating unknown neighborhoods as negatives. Otherwise return unavailable with a reason.
- [x] Error-versus-coverage curves sort confidence in its declared direction, admit whole ties, and sample at most 101 tied thresholds; preserve denominators, unknown accepts and actual threshold. Report confidence as an uncalibrated score, never uncertainty.
- [x] Pair scene/sequence aggregates only. Implement deterministic `hierarchical_scene_sequence_bootstrap_v1`: 2,000 bootstrap draws, seed 0, scenes sampled with replacement, sequences sampled within each drawn scene. Within a scene average sequence-unit comparisons, then average scenes equally; report interval definition/seed/draws/unit counts. Upper paired one-sided interval bounds must meet completeness-loss pp, candidate-minus-ratio*reference MAE and boundary false-accept pp margins. At exactly zero reference MAE use the policy absolute-MAE margin. Reject missing/nonfinite summaries; unknown method labels, insufficient units or missing frozen parameters remain unqualified. No statistic alone establishes held-out data or registration.
- [x] GREEN: `.venv/bin/python -m pytest tests/test_zed_quality_metrics.py -q`.

## Task 3: Native output to portable observations

**Owner/files:** Adapter worker owns `native_observation.py` and `test_zed_native_observation.py` only.

`export_native_observation(report_path, source_record_path, dump_root, output_dir) -> pathlib.Path` writes a fresh portable record bundle. Complete input source bytes/dependencies are copied unchanged (existing cold native-input helper semantics), and the new observation/implementation replace only their respective record fields. Recompute and assert the unchanged input identity. Preserve original source artifact, calibration, images and timestamps.

- [x] RED: missing adapter; exact left-half bytes from two-eye float dumps (including NaN payloads), padded input preservation, U32 -> U8 validity/provenance, confidence direction, source-ID/age checks, malformed sizes/hash/path/source/calibration/profile, absent dump, unavailable commit, fresh output collisions/mutation.
- [x] Admit only executed `stereoforge.daxa_depth_rectified_benchmark.v1` full_sgm/coarse_refine reports. Match input/pixel/calibration/source/ordinal/dimensions with the source record and validate report frame count/settings. Verify every consumed dump SHA256/elements/byte extent (2*N); native report dump paths can be absolute but must resolve under explicit dump_root without symlink/traversal. Select first N left pixels; preserve f32 bytes. Require binary U32 masks; valid provenance STEREO/current source ID/age0 for these fresh profiles. Keep invalid metric bytes untouched and masks explicit.
- [x] Emit confidence high-is-good, portable provenance/source/age planes, and implementation name `StereoForge`, version `canonical_native_v1`, settings consisting of stable executed profile fields, explicit full-grid search bounds and available compiler/shader identities. Report absent commit as unavailable, never infer it from current checkout. Copy/hash the native report as an implementation artifact. Use fresh `native-observation/` output paths; reject collisions. Complete portable validation before successful publication.
- [x] GREEN: `.venv/bin/python -m pytest tests/test_zed_native_observation.py -q`.

## Task 4: Coordinator and actual analytic comparison

**Owner/files:** Coordinator ownership was explicitly delegated after truth admission froze. Root owns shared `schema.validate_policy`, integration, evidence, builds and publication.

- [x] Extract dict policy validation from `load_policy` without changing path-loader behavior; tests prove shared validation and no input mutation.
- [x] RED named `test_unexecuted_sdk_never_parity_pass`, missing roots, changed bytes, duplicate/cross-partition methods, different calibration strata and actual native left-depth integration.
- [x] Materialize one frame/method group at a time, report per-frame/per-method/per-stratum populations and same-input intersection errors across methods. Aggregate counts/sums (not averages of per-frame percentages) before scene/sequence comparison. Missing observations/truth produce explicit unavailable results.
- [x] Primary reference requires an available observation with implementation name `ZED SDK` and settings.depth_mode equal to the policy primary reference. Do not recognize method names or absent observations as executed references. Scientific qualification additionally requires frozen policy, matching partition digest, held-out data, complete required region definitions, uncertainty bounds, actual implemented paired statistics and adequate independent scene/sequence counts. Provisional policy, missing reference or any missing gate always yields parity_qualified=false with reason codes. Do not edit the default policy.
- [x] Independent review and CPU performance pass cover readback/cold-adapter boundaries, no pixel Python loops, one-frame array lifetime, bounded hashing and file rechecks; add actual-source probes/static or mutation guards.
- [x] Root verifies unit/integration tests, adapts the existing frozen analytic GPU dumps into both portable method records, runs the evaluator on independent plane truth, and records fixed-support errors/completeness without generalizing beyond that development fixture. A bounded one-iteration rerun of the unchanged frozen native binary was needed because older reports omitted corrected subpixel settings while v3 reports omitted dumps. Both variants were captured with settings and dumps in the same execution, with zero validation errors/warnings. No data from different executions was silently combined.
- [x] Full CPU regression and clean Git diff: 2,290 passed, 37 GPU/camera tests deselected, no skips, 132.74 s.
- [ ] Commit/push reviewed evidence and verify CI. Keep Task 4 scientific qualification open until its actual data/policy/reference requirements are satisfied.

## Verified implementation checkpoint

Independent functional review reproduced requested-domain counts, MAE and nearest-rank p95 directly from native bytes and confirmed exact left float preservation. Separate performance review passed 150 component tests including 19 CPU/materialization guards. Review fixed disjoint nonempty boundary F1, candidate easy-subset selection, incomplete boundary evidence and whole-run truth mutation detection. Actual 128×64 infinite-plane development results retain all 8,192 sampled truth pixels: full SGM 6,960 accepted (84.9609375%), refinement 5,837 (71.2524414%). Both remain unqualified with provisional policy and no executed SDK reference. See [evaluator evidence and usage](../../independent-quality-evaluation.md).

Frozen snapshot: `artifacts/zed-task4/plane-evaluation-freeze-v2.json`; report SHA256 `3917c040b1f3bb711c614fa3aaac6938a94352dfed0e890c1bdd04b6cd2106f1`. The 720p CPU metric timing of roughly 372–377 ms includes the complete 101-threshold confidence curve and is offline evaluator cost, with no effect on native GPU processing overhead.
