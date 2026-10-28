# Matched performance records implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development. End-to-end execution and publication are already authorized.

**Goal:** Implement Task 6's timing admission and actual native/SDK report boundaries without promoting unqualified quality or incomparable timings.

**Architecture:** Offline strict timing records reference hashed raw reports, quality reports, compiler/input freezes and an interleaving protocol. Native apps add metadata outside timed loops. The isolated SDK runner adds a timing-only opaque-grab operation; no undocumented GPU completion fence is invented.

**Tech Stack:** Existing Python/NumPy validators and C++20 JSON reports; optional pinned SDK 5.5.0 only in its existing default-OFF subtree.

**Spec:** [Task 6](2026-10-28-zed-quality-benchmark.md#task-6-matched-performance-records-after-quality) and [quality-first design](../specs/2026-10-28-zed-quality-benchmark-design.md).

## Constraints and review focus

- No SDK/GPU benchmark run is admitted before quality/policy gates. Compile/CPU contracts and offline retained-report evaluation are allowed now.
- No Python or full-host round trip enters the live runtime. Existing native schemas, summaries and raw values remain compatible.
- Never derive GPU completion from SDK grab return. Opaque measurements stay opaque and cannot produce a native-GPU speedup ratio.
- Cold setup, excluded warmup, warmed samples, cache temperature, allocation activity and export allocation properties are separate; unknown is unavailable.
- Raw samples preserve chronological order; compute nearest-rank p50/p95/p99/max from them, never from prior percentiles. A finite small p99 is descriptive, not a tail guarantee.
- A repeated fixture is not delivered fresh FPS. Cross-platform measurements are operating points; do not calculate algorithmic speedup ratios.
- Repeated runs/slots, incomplete/reordered ABBA, missing dependencies, mixed boundaries, changed settings and unavailable contention evidence must fail closed.
- Hash shared files once with bounded reads and final fingerprints; bound JSON/sample counts, avoid per-pixel loops and dataset-sized copies.

## Task 1: Portable timing admission and summary

**Owner:** Timing worker owns `benchmarks/zed_comparison/performance.py`, `tests/test_zed_performance.py` only.

Interfaces: `validate_timing_lane(record:dict)->None`; `summarize_timing(records:list[dict], *, root=None)->dict`; `profile_identity(record)->str`; `adapt_native_timing(report:dict, context:dict)->dict`. `root` accepts one explicit root or a list matching records. Missing root permits structural summaries but never qualified comparisons. Reuse schema validation/file helpers.

Strict record schema `stereoforge.zed_comparison.timing.v1` has exactly:

```
{schema, session_id, run_id, profile_id, method_key, implementation,
 platform:{host_id,device_uuid:Availability[str],vendor,driver,os},
 input_identities:[SHA256], calibration_identity:SHA256,
 phase:"cold"|"warm"|"unwarmed",warmup_count:Int>=0,
 lane:"gpu_resident"|"host_output"|"opaque_api",completion_boundary,scope,
 requested_outputs:[nonempty unique labels],
 isolation:{file_io_in_span:bool,diagnostic_retrieval_in_span:bool},
 cache:{temperature:Availability[str],allocation_activity:Availability[str],
        host_output_properties:Availability[str]},
 sources:{timing:FileDesc,freeze:FileDesc,quality:Availability[FileDesc],
          protocol:Availability[FileDesc]},
 order:{block:Int>=0,position:Int0..3},sample_count:Int1..100000,
 samples:{<fixed metric keys>:Availability[list[finite nonnegative ms]]},
 stages:{label:Availability[list[finite nonnegative ms]]}}
```

`implementation` is exact `{name,version,commit:Availability[SHA1],device:Availability[nonemptydict],settings:dict}` as compare_quality's method identity; method_key is SHA256 canonical implementation. host_id/session_id/run_id/scope/platform labels are nonempty strings. UUID is explicit stable identity, never inferred from a friendly device name. input IDs are unique nonempty SHA256 list sorted lexically; profile_id hashes canonical `{method_key,input_identities,calibration_identity,requested_outputs}` (outputs sorted unique). Bound canonical record JSON to16MiB.

Fixed metric keys: `cold_prepare_ms`, `gpu_total_ms`, `gpu_upload_ms`, `gpu_readback_ms`, `cpu_submit_ms`, `cpu_wait_ms`, `cpu_query_ms`, `cpu_control_ms`, `complete_wall_ms`, `opaque_api_ms`, `file_io_ms`, `publication_ms`. All available arrays match sample_count. Missing values are explicit Availability, not zero. Warm requires warmup_count>0, cold requires0 and available cold_prepare_ms; cold_prepare is unavailable outside cold phase. Completion mapping is exactly gpu_resident→`final_submit_wait_and_query`, host_output→`requested_cpu_output_ready`, opaque_api→`opaque_api_return`. Host-output without allocation properties remains scoreable but unqualified. Warm GPU-resident requires available GPU total and complete wall for qualification; host-output requires complete wall; opaque never establishes compute-complete comparison. File I/O/diagnostic retrieval inside span blocks matched-compute qualification.

`adapt_native_timing` accepts one of the three existing native report schemas and a context containing all record headers except schema/profile_id/sample_count/samples/stages. Map raw CPU/upload/GPU combined columns from the report using existing names; add remaining explicit unavailable fields. Respect per-column availability in new timing_metadata and null columns, never promote coarse_alone legacy zeros. Carry extra raw GPU columns into stages. Derive sample_count/phase/warmup from actual report; reject context/header contradictions, bad raw dimensions or missing timing_metadata. Actual profile/quality bindings remain supplied context and checked below. Never fabricate unknown cold samples or cache properties. Legacy reports without metadata can be structurally inspected elsewhere but cannot be adapted as qualified evidence.

Summaries separate phase/lane/completion/platform/profile populations. Retain raw samples in report; pooled stats use concatenated actual samples (bounded total1million), not averages/percentiles of summaries. No FPS inference. Reject repeated `(session_id,run_id)` and duplicate `(session,block,position)`; attempted ABBA block with mixed completion/lane/scope/outputs/phase is invalid ValueError. Cold and warm blocks remain distinct populations.

With explicit roots, hash/check timing, freeze, quality, protocol files and referenced dependencies once. Raw native samples must exactly match the adapted record's available samples/stages and timing metadata; SDK timing reports admitted only as opaque and never as GPU complete. Reject arbitrary unrecognized raw timing schema for qualified evidence. Return explicit reasons for missing quality/protocol/root/freeze rather than passing.

Freeze schema `stereoforge.zed_comparison.timing_freeze.v1` exact `{schema,frozen:bool,profiles:[{profile_id,method_key,input_identities,calibration_identity,requested_outputs,artifacts:[FileDesc]}]}`; profile bindings exactly match record and at least one hashed artifact required. Artifacts represent actual compiler/SPIR-V/source/input manifests retained by caller; do not claim their role or execution from a filename. SDK private assets stay private. Missing/unfrozen profile prevents qualification.

Protocol schema `stereoforge.zed_comparison.timing_protocol.v1` exact `{schema,frozen:bool,session_id,order:[A,B,B,A],min_blocks:Int>=2,min_samples_per_run:Int>=1,contention_before:Availability[FileDesc],contention_after:Availability[FileDesc]}`. A/B are distinct profile SHA256s. Contention documents are hash-bound strict `{schema:"stereoforge.zed_comparison.contention.v1",status:"clear"|"contended"|"unknown",platforms:[exact platform dicts],method:nonemptystr}`; clear is declared evidence, not proof of global exclusivity. Both must cover actual platforms; missing/contended/unknown blocks comparison. All methods in one comparison must reference identical protocol content/freeze content and have matching ordered input IDs/calibration. Require complete positions0..3 per block, exact ABBA profile order, min blocks and per-run samples. Scope/lane/completion/output sets must match. Cross-host/device or unavailable UUID produces operating-point comparison with ratio=None; same-device runtime ratio only with exact host/UUID/vendor/driver/OS, warm phase, complete data and quality qualification. Do not pool unlike profile implementation/settings.

Quality report verification: exact recognized quality.v1 schema, status qualified, parity_qualified true, no reasons; target method key and exact implementation present; partition digest a valid SHA; at least one qualified/no-reasons comparison involving method; available frame input set for method equals timing input IDs, all held_out and calibration stratum equals record calibration_identity. This checks declared supplied evidence, not independent SDK attestation. Missing/unqualified quality prevents promotion; do not require real SDK files in CPU tests.

- [x] RED/GREEN named `test_reject_mixed_completion_boundaries`, `test_cold_separate_from_warm`, `test_cross_platform_not_speedup`, `test_incomplete_interleaving_unqualified`; add file mutation/shared-cache, wrong raw arrays, malformed/duplicate records, unknown settings/cache, frozen quality binding, positive synthetic same-device ABBA and nearest-rank stats tests.

## Task 2: Native additive timing metadata

**Owner:** Native worker owns three apps and a new `tests/test_zed_native_timing_metadata.py`; shared CPU helper header allowed `apps/timing_metadata.hpp`.

Add post-loop `timing_metadata` to raw depth, refinement and canonical benchmark reports: exact object `{schema:"stereoforge.native_timing_metadata.v1",units:"ms",sample_population:"measured_iterations_only",sample_order:"chronological",measured_count:frames,excluded_warmup_count:warmup,phase:(warmup>0?"warm":"unwarmed"),source_reuse:"static_retained_record",completion_boundary:"final_submit_wait_and_query",readback_in_span:false,file_io_in_span:false,diagnostic_retrieval_in_span:false,legacy_summary_policy,columns:{rawkey:Availability[scopeLabel]},cold_prepare:Availability[numberMs],cold_prepare_scope,cache_temperature:Unavailable,allocation_activity:Unavailable,host_output_properties:Unavailable}`. Use existing native JSON Availability form `{status,value|reason}` here, not device.report's older shape.

Column availability represents actual used columns/nulls; refinement coarse_alone inactive target/refinement columns unavailable despite legacy zeros. Existing summary fields unchanged; label depth legacy upper-median/floor95, refinement/canonical nearest-rank. Depth's existing initialization_ms is available cold_prepare metadata scoped `foundation_and_pipeline_construction_only`; other apps unavailable. This is setup metadata, NOT a cold frame sample. All runtime allocation/cache properties remain unavailable unless actually measured.

Export depth paired total and phase raw arrays alongside existing summaries using existing phase-name mapping; no new timing queries, waits, imports or timed-loop modifications. Existing phase vectors are already collected. Add explicit direct/fake actual-source CPU probe validating helper emitted JSON, bounds/nulls, coarse-alone availability, no timed-loop calls; build the three actual targets CPU compilation only after root admits shared build. No GPU run.

- [x] Tests first; preserve old report checks and raw arrays; compile native actual targets after CPU performance review.

## Task 3: Isolated SDK timing-only operation

**Owner:** SDK worker owns `benchmarks/zed_comparison/export_sdk.cpp`, narrow new helper `sdk_benchmark.hpp` if needed, `tests/test_zed_exporter_timing.py`; existing tests only narrow compatibility update with root awareness.

Add explicit `--operation benchmark`, `--warmup 0..100`, `--frames1..100`, default warmup5 for benchmark, diagnostic unchanged defaults. `--timing-lane` supports only `opaque-api` initially; unsupported `gpu-resident`/`host-output` reject explicitly rather than substitute. Report schema `stereoforge.zed_comparison.sdk_timing.v1`, qualification unqualified, completion_boundary opaque_api_return, GPU completion unavailable and all internal GPU stages unavailable. `NONE` timing is no-depth capture, never depth mode.

Use existing settings/open/preserved SVO/private-root admission. After effective settings checks and persistent-buffer setup branch benchmark into separate function/loop. The timed loop may only call grab and read host steady_clock around it, store bounded preallocated samples; queries/settings/CPU image/depth retrieval/serialization/file writes cannot occur inside this loop. No periodic file writes; final success/error metadata outside. Collect successful warmup and measured counts separately; cursor/position may be queried outside timed grab interval ONLY before/after full loop, not per-iteration, and repeated/replay identities remain unavailable when not verified. No silent EOF wrap or substitute modes; partial/error report unqualified with measured successful samples.

Retain cold open/source hash/buffer prepare durations with exact existing scopes, requested/effective settings, source/build identity and non-real-time replay. Raw opaque_grab_api_ms and complete_wall_ms are the SAME host API-return span; never call complete_wall GPU completion. Record file/retrieval excluded, playback pacing false; exclude warmup samples explicitly; phase warm only if actual warmup completed>0 otherwise unwarmed. Mean/p50/p95/p99/max nearest rank. No SDK/CPU export or mode comparison executed now; compile on inventoried host/container only, and --help/--version are safe CPU commands.

SDK timing reports can later be adapted only to opaque_api timing records; unchanged diagnostic exporter source tests and serializer contracts must remain green. Do not add SDK/CUDA linkage to core. If factoring helper, CMake build/source identity hashes must cover helper bytes as well as export_sdk.cpp (root owns final optional CMake integration).

- [x] Test parse limits, requested operation/lane, no loop I/O/retrieval, bounded raw chronology and partial failure via actual-source fake-grab CPU helper; SDK-free configuration; optional pinned SDK build only, no GPU benchmarking.

## Task 4: Review, evidence and integration

**Root:** integration, shared builds/remote CPU compilation, docs/evidence, Git/CI; no GPU reservation/run needed. Extend source guards only for reviewed offline imports, preserve forbidden GPU/process mutants.

- [x] Independent functional review of quality/raw/ABBA/completion admission and actual native/SDK code; mandatory separate performance worker probes bounded/linear CPU admission and verifies native timed loops unchanged and SDK loop allocation/I/O-free except bounded retained samples.
- [x] Root adapt retained actual Task4 native report(s) using newly compiled metadata or explicit unqualified legacy reasons; no old timing retrofit pretending measured evidence. Synthetic positive fixtures test structural qualification; actual engineering report without frozen SDK quality remains unqualified. Retain source/report hashes and separate costs.
- [x] CPU tests, actual native compilation, SDK-free configure and optional actual SDK compile; full regression, diff/reviews, commit/push and CI. Task6 matched live GPU/SDK measurements remain open until quality and contention gates are satisfied. Overall end-to-end goal remains active.
