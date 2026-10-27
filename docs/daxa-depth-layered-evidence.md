# Layered per-path stereo scheduling

`LayeredPerPath`, selected by `--stereo-schedule layered-per-path`, is an explicit
experimental scheduling mode for the resident dual-reference depth pipeline.
It preserves exact outputs but is slower than Sequential on the measured AMD
target. `Sequential` remains the default; layered mode is not promoted. The earlier two-dispatch `PairedPerPath`
regression remains documented separately in the unchanged
[paired-control evidence](daxa-depth-paired-evidence.md).

This experiment is based on `20942372c016c649a2ed3ce6dd0640636b6a6552`.
Its API and observation semantics remain those of the
[depth contract](daxa-depth-contract.md). Local evidence is retained under
`artifacts/daxa-depth-layered-20261027/`; recordings, raw image planes and
compiled binaries are not included in the source publication.

## Actual dispatch and resource boundary

Each aggregation direction now uses one XY×2 dispatch: GroupID.z selects the
Left or Right reference. Selection also uses one XY×2 dispatch. Aggregation
retains 64-thread workgroups and full subgroup32/64 requirements; selection
retains its 256-thread group-shared reduction. A workgroup-uniform eye guard
rejects z>=2 before parameter loads or collectives. XY line/pixel flattening
excludes z.

Each eye keeps separate packed penalties and final output buffers. Same-eye
read/write hazards still order successive paths because accumulation uses
ordinary non-atomic addition. Both original Census pairs and the two clear
operations remain duplicated. The phase count remains eight for P4 and twelve
for P8; dispatch counts are eleven and fifteen, respectively. Dispatch count
alone is not evidence of a speedup.

The single shared aggregate/selection implementation receives explicit
Parameters, path and XY stride. Legacy entrypoints retain their 16-byte push
layout; the new wrapper uses 24 bytes containing both parameter addresses, path
and XY stride. Cost, recurrence, packing, lowest-index tie selection, confidence
and subpixel control flow are unchanged. No search range, path count, precision,
quality tolerance or algorithm was reduced.

The existing aligned-slot coordinator, typed StereoFrames and depth output
leases remain in use. Admission checks both producer slots before upstream
upload; original/rectified RGB and both stereo producers remain retained through
consumer lifetime. There is no new per-frame buffer, application heap allocation, wait or
readback. At 1280×720/D128/P4/P2=32/two slots, requested application buffers remain
**840,500,528 B**. This excludes query objects, allocator/driver state,
diagnostics and CPU-source storage; it is not a measured peak-memory bound.

Layered reports use pair total/phase timings and null individual-eye timings.
The explicit `stereo_dispatch_form` distinguishes the z2 aggregation/selection
from the two-dispatch paired control. Combined GPU latency remains the actual
same-queue upload-start through metric-end interval, including intergraph gaps.
Host wall includes staging, collection/submission, completion and query costs.

## Source identity and control requalification

The canonical Parameters schema and specialized body are shared includes;
`stereo_layered.slang` adds four entries: `aggregate_layered8/16` and
`select_layered8/16`. Runtime report fields and the CPU-only
`--shader-source-identities` getter bind the actual compiled legacy wrapper/ops,
shared Parameters/body and layered wrapper hashes. The comparison harness
checks these files, all SPVs, compiler dependencies, binaries and fixture/report
bytes before and after execution.

Extracting the shared body changed exactly **eight of 29 prior SPVs**: Left and
Right specialized aggregation and selection, each in 8/16-bit packing. The
other 21 prior modules remain byte-identical. Four new modules bring the total
to 33. Source-body comparison establishes the mechanical parameter/address
extraction, but does not replace GPU output qualification.

Before running layered dispatches, root requalified the changed legacy controls
using the new source-bound harness in Sequential mode:

| Control artifact | Profile | Samples / warmups | Result |
|---|---|---:|---|
| `control-720p-sequential-core` | 1280×720, D128/P4/P2=32, W32, packing8, slots2 | 30 / 3 | All nine observations and six raw stereo planes exact |
| `control-dynamic-p8-w64-sequential-sync` | 64×32 dynamic record, D32/P8/P2=32, W64, packing16, slots2 | 3 / 1 | Same exact gate |

Both controls have zero native validation warnings/errors through teardown and
empty child diagnostic streams. This covers the changed 8/16-bit Left/Right
wrapper families without treating their new binaries as unchanged controls.

## Correctness and CPU evidence

Root native GPU contracts pass with core and synchronization validation, with
empty stderr. The suite includes Sequential, PairedPerPath and LayeredPerPath,
W32/W64 and one/two slots, dynamic different-content input, exact integer/direct
packed-word checks, source metadata, retained/in-flight producers, full rings
and consumer lifetime.

The independent frozen-RHI harness passes all nine observation planes and six
raw stereo planes in the following layered runs:

| Artifact | Profile | Samples / warmups | Status |
|---|---|---:|---|
| `small-layered-sync` | 8×5, D128/P4/W32, slots2 | 3 / 1 | Exact, zero native teardown warnings/errors |
| `720p-layered-core` | 1280×720, D128/P4/W32, slots2 | 30 / 3 | Exact, zero native teardown warnings/errors |
| `720p-layered-sync` | Same 720p profile, synchronization validation | 30 / 3 | Exact, zero native teardown warnings/errors |
| `dynamic-p8-w64-layered-sync` | 64×32 dynamic record, D32/P8/W64, packing16, slots2 | 3 / 1 | Exact, zero native teardown warnings/errors |

The oracle derives OpenCV fixture luma/support independently, uses frozen RHI
Left matching and literal mirrored Right matching with final-row unmirroring,
and compares actual raw float bytes before the nine-plane composition. Native
diagnostics independently compare Census, masks, costs, sums and actual packed
words for both references. Diagnostic readback/oracle work occurs after timing.
General compound CPU-float flags remain false; no floating tolerance or physical
metric accuracy is inferred from these recorded GPU-byte results.

Root's full Python suite passes **1043 tests in 20.76 s** and four native CPU
contracts pass in **0.57 s**. Independent performance review passes 397 focused
CPU tests, including legacy mutation guards and the new source/ABI/z2 checks.
Its extracted actual admission/retirement/staging probe processes 256 warmed
720p frames per mode with zero application heap allocations and one explicit
admission snapshot/frame. The recorded layered mean is **0.0536 ms/frame**;
this is application bookkeeping with GPU/SDK stand-ins and excludes real Daxa,
driver, GPU execution and waits. It is not native pipeline latency.

## Matched performance result

`abba-summary.json` binds four serial native child processes in Sequential,
LayeredPerPath, LayeredPerPath, Sequential order. Each processes the same retained
1280×720-per-eye fixture, D128/P4/P1=8/P2=32, subpixel enabled, W32 and two slots,
with **100 measured samples after 20 warmups**. Validation, reference checks and
dumps are absent; diagnostic flags remain false and stderr is empty. Packed
upload, preprocessing, both references, LR and metric/metadata materialization
are included in the complete queue/host span. There is no full image readback.

| ABBA run | Schedule | GPU mean / p95 (ms) | Host wall mean / p95 (ms) |
|---|---|---:|---:|
| 1 | sequential | 31.882531 / 32.427883 | 32.331435 / 32.934179 |
| 2 | layered-per-path | 39.971237 / 41.083179 | 40.495722 / 41.622792 |
| 3 | layered-per-path | 40.311689 / 41.486495 | 40.851693 / 42.102625 |
| 4 | sequential | 31.794582 / 32.358001 | 32.239447 / 32.854200 |

Both layered runs are slower than both Sequential controls. This rejects this
layered scheduling variant as a latency improvement at the measured operating
point; fewer dispatches did not improve the complete chain. It does not identify
a cache, occupancy, bandwidth or synchronization cause. No runtime hardware
counters establish such a diagnosis.

The measured target is AMD Radeon 8060S Graphics / RADV Strix Halo, Mesa26.2.3,
with the pinned Slang2026.12 toolchain. The native benchmark SHA256 is
`9594595611b112e14e2782c8d172af60dda003292ca7698c19f85940121813fa`;
the harness SHA256 is
`e463eb64b6dc95f8331e1597eea7fde044f995866b65679739a5cc03b03b8599`.
The ABBA summary retains each exact command and report hash; child reports bind
fixture/calibration, source metadata, record ordinal and compiled source identity.
Serial child execution does not imply globally exclusive system use.

A separate `layered-300-sync.json` run passes 300 measured iterations after
three warmups at the same 720p profile with synchronization validation and native
reference checks. It reports 303 completed output retirements, all four native
diagnostic flags true, zero warnings/errors through teardown and empty stderr.
GPU mean/p95 are **39.850667/41.269663 ms**; host wall mean/p95 are
**40.682263/42.261916 ms**. Its **5448.266374 ms** final diagnostic work occurs
after timing. This short repeated-fixture stress run is not long sustained or
fresh-source qualification; its checked/validated timing scope differs from the
no-validation ABBA runs, which remain the primary latency comparison.

## Reproduction and limits

Build the optional targets and configure validation layers as described in
[depth evidence](daxa-depth-evidence.md#reproduction-and-remaining-qualification).
Run Sequential changed-control checks before layered validation after changing
the shared shader source. For a fresh output directory:

```bash
PYTHONPATH=python .venv/bin/python benchmarks/compare_daxa_depth.py \
  --manifest artifacts/daxa-preprocess-20261027/fixture-720p/manifest.json \
  --compiler build/standalone/_deps/sf_slang_sdk-src/bin/slangc \
  --output artifacts/new-layered-exact \
  --frames 30 --warmup 3 --slots 2 --wave-size 32 \
  --disparities 128 --paths 4 --p1 8 --p2 32 \
  --stereo-schedule layered-per-path --validation
```

Use `VK_LAYER_VALIDATE_SYNC=1` and another fresh directory for synchronization
validation. The harness requires the configured offline compiler and Python/
OpenCV4.14.0; the native executable has no Python runtime dependency. The native
GPU contract executable now includes layered dispatches, so run it only after
the changed Sequential controls pass.

Timing-only native runs omit validation, `--check-reference` and dump output.
Use matched settings, fresh report paths, and the same recorded fixture across
all schedule modes. Source/compiler/SPV/binary identities and host allocation
scope must remain bound to each result.

This is static repeated-fixture source-grid depth. Fresh camera throughput,
exposure-to-display age, physical calibration/depth accuracy, wide RGB-D/view
integration, general compound CPU float accuracy and sustained live 60 Hz remain
outside this experiment. No hardware-counter evidence identifies cache,
occupancy or bandwidth as the cause of a measured schedule difference.
