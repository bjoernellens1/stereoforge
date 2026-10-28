# Matched performance records

Task 6 provides offline timing admission and report metadata. It does not establish a native/SDK speedup or authorize live measurements before the quality and policy gates. The native GPU loop remains resident; Python validation, hashes and report processing are offline boundaries.

## API and record workflow

```python
import json
from pathlib import Path
from benchmarks.zed_comparison.performance import (
    adapt_native_timing, profile_identity, validate_timing_lane, summarize_timing,
)

root = Path('/absolute/path/to/retained-session')
raw = json.loads((root / 'native-report.json').read_text())
context = json.loads((root / 'timing-context.json').read_text())
record = adapt_native_timing(raw, context)
assert record['profile_id'] == profile_identity(record)
validate_timing_lane(record)                 # Structural validation only.
summary = summarize_timing([record], root=root)
```

The context supplies the strict record headers: session/run identities, implementation and method key, platform, sorted unique input identities, calibration identity, phase/warmup/lane/completion, scope, requested outputs, isolation, cache, source descriptors and ABBA block/position. The adapter derives schema, profile ID, sample count, samples and stages from the actual report. It derives phase/warmup and native boundary/cache fields too, rejecting contradictory context values. Do not fabricate context to fill evidence gaps. A single record can be described but cannot satisfy matched ABBA admission.

`profile_identity(record)` hashes the canonical method key, input identities, calibration identity and sorted requested outputs. Method identity includes exact implementation name/version/commit/device/settings. Platform and measured values are outside the profile hash but remain explicit comparison gates.

`validate_timing_lane(record)` checks the exact `stereoforge.zed_comparison.timing.v1` schema, finite nonnegative millisecond arrays, sample counts and Availability fields. Every available metric/stage array must match `sample_count` (1–100000). Record JSON is bounded to 16 MiB. Unknown cache temperature, allocation activity and host-output properties are unavailable with a reason, never zero. Available zero is a measured value.

`adapt_native_timing(report, context)` accepts the three supported native report schemas with actual `timing_metadata`. It maps chronological raw CPU/upload/combined-GPU columns and preserves additional stages. Metadata availability controls inactive columns, including legacy `coarse_alone` zeros. Missing metadata is rejected; old retained timing must not be retrofitted into measured evidence.

`summarize_timing(records, *, root=None)` accepts a common explicit root or a list of roots matching record order. Without roots it produces structural summaries with qualification reasons. With roots it verifies timing/freeze/quality/protocol descriptors, referenced dependencies and final file fingerprints using bounded reads and shared hash caches. It checks raw arrays and metadata against the adapted record. Total retained metric/stage values are capped at one million.

## Populations and evidence gates

Raw samples retain chronological order. Summaries concatenate actual compatible samples and compute nearest-rank p50/p95/p99/max; they do not average prior percentiles. These statistics describe the retained finite population and provide no tail guarantee or fresh-frame FPS claim. Populations remain separate by phase, lane, completion boundary, platform, profile and scope.

Cold construction/setup metadata is separate from cold frame samples. Native setup measurements retain their declared scope; they cannot be relabeled warmed execution or a cold frame. Warm samples exclude actual completed warmup; no warmup means `unwarmed`. Unknown cache state does not establish cold-cache or warm-cache behavior.

| Lane | Required completion boundary | Interpretation |
| --- | --- | --- |
| `gpu_resident` | `final_submit_wait_and_query` | Declared native completed submission/query scope |
| `host_output` | `requested_cpu_output_ready` | Requested CPU output ready, with output allocation properties required for promotion |
| `opaque_api` | `opaque_api_return` | Host API return only; GPU completion unproven |

Matched promotion requires all of the following supplied evidence:

- Qualified quality report with no reasons, explicitly available frames, exact method/implementation and held-out input/calibration bindings, valid partition digest and an involving qualified comparison.
- Frozen profile manifest matching method/input/calibration/output identities and hashing retained artifacts. For canonical native reports, content coverage checks admitted input extents, compiler identities and SPIR-V hashes. Generic artifact entries do not prove a filename's role; this verifies declared retained content, not execution attestation.
- Identical frozen protocol and freeze content, at least two complete ABBA blocks in exact `[A,B,B,A]` order, distinct profiles and enough samples per run. Repeated runs/slots, mixed boundaries/scopes/outputs/phases and contradictory raw values reject; missing/incomplete evidence remains unqualified.
- Hash-bound before/after contention documents declaring clear status and covering the actual platforms. These declarations do not prove global exclusivity.
- Exact same host, known stable device UUID, vendor, driver and OS for a warm same-device runtime ratio. Friendly device names cannot substitute for UUIDs. Cross-platform results are operating points with `ratio=None`.

File I/O or diagnostic retrieval inside the timed span blocks matched-compute qualification. Unbound raw input identity remains diagnostic only. Actual canonical native temporal settings must bind and establish `temporal_off`; absent settings block promotion with `raw_temporal_settings_not_bound`. Current SDK raw reports also lack verified per-grab input identity. An opaque SDK lane never becomes a GPU-complete native comparison, even when its API-return span is called `complete_wall_ms`.

## Optional SDK operation

The SDK exporter stays isolated behind `STEREOFORGE_BUILD_ZED_REFERENCE=OFF` by default. Existing optional CMake source/build identity hashing covers `export_sdk.cpp`, `export_io.hpp` and `sdk_benchmark.hpp`; SDK/CUDA linkage is not added to core targets.

The implemented CLI accepts this future operation (do not execute before quality/policy admission):

```bash
build/zed-reference/benchmarks/zed_comparison/export_sdk \
  --operation benchmark --svo /private/input.svo --output /private/fresh-result \
  --mode NEURAL_LIGHT --frames 20 --warmup 5 --timing-lane opaque-api
```

`--frames` is 1–100 and `--warmup` is 0–100; benchmark defaults to five warmups. Only `opaque-api` is supported; unsupported GPU-resident/host-output requests reject. `NONE` means no-depth capture. Private-source/settings admission and effective-setting checks precede the timing loop.

The separate sampler uses bounded preallocated storage and host steady-clock spans around `grab` only. It performs no output retrieval, settings queries, per-iteration cursor queries, serialization or file writes. `opaque_grab_api_ms` and `complete_wall_ms` contain the same API-return spans. Internal GPU stages/completion, cache/allocation state and per-grab replay identities remain unavailable. Successful warmup and measured counts are separate; errors/EOF retain partial unqualified results without silent wrap or mode substitution. Cold open and source hashing remain separately scoped metadata. The benchmark returns before diagnostic CPU buffers allocate; their preparation is explicitly unavailable rather than a measured zero.

## Current verification boundary

Verified locally on 2026-10-28: **2,507 CPU tests passed**, 37 GPU/camera tests deselected, no skips (140.43 s). Independent functional and CPU performance reviews passed. All three native targets compiled. The isolated runner compiled against SDK 5.5.0 and its version command passed with no GPU exposed. The benchmark operation has not been executed. Actual matched live GPU/SDK measurements remain open under the [Task 6 contract](superpowers/plans/2026-10-28-matched-performance-records.md).

The offline admission probe took 10.41 ms for eight runs and 59.31 ms for 128 runs; an 8 MiB shared artifact was hashed once in either case. These CPU/file costs are outside the native frame loop. SDK-free sampling tests use fake grab calls and cannot establish SDK performance. Reproduce deterministic structural cases with:

```bash
PYTHONPATH=python:. .venv/bin/python docs/evidence/matched-performance-2026-10-28.py
```

The [saved evidence](evidence/matched-performance-2026-10-28.json) additionally inspects a retained old native report and rejects its missing timing metadata. The script accepts that private/local file through `--legacy-report`; without it, only the synthetic cases run. Synthetic ABBA acceptance is a validator test, not a vendor or hardware result. Source hashes accompany the output.

For this host's full regression, use a disk-backed temporary directory. The unchanged serializer probe produced a partial copy on RAM-backed `/tmp` and passed unchanged on disk; the exact OS write error was not established.

```bash
mkdir -p artifacts/zed-task6/tmp
TMPDIR="$PWD/artifacts/zed-task6/tmp" PYTHONPATH=python:. \
  .venv/bin/python -m pytest tests -q -m 'not gpu and not camera'
```
