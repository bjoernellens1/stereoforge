# Daxa resident-view proof of concept — 2026-10-27

The independent Daxa/Slang resident compositor preserves the frozen Slang RHI
strict/preview output contract on a static 1280×720 fixture. This qualifies an
immutable retained-source proof of concept, not production pipeline adoption,
mutable capture ingest, stereo/temporal integration, physical depth accuracy,
or sustained 60 Hz operation.

## Implementation and source attribution

Daxa owns its Vulkan device, resources and persistent TaskGraphs independently
of RHI. Binding-free Slang projection operations are mechanically extracted
from the unchanged RHI shader. Offline Slang 2026.12 compilation supplies eight
compositor passes and a downstream GPU consumer; the native Daxa executable does
not link RHI, the Slang compiler runtime, Python, SlangPy or SGL.

Bounded one/two/three-slot contract fixtures exercise retained output leases,
completion-gated reuse, delayed consumers, generation changes, facade destruction,
foreign-context rejection and replacement calibration/resolution contexts.
The downstream 16-byte consumer checks noncancelling high-word ID/age metadata.
A submission still allocates one shared lease control block; no allocation-free
claim is made. Daxa internally allocates/reclaims command-recorder resources.

The frozen reference inventory is
`artifacts/daxa-view-20261027/reference-manifest.json`: fixture input identity
`6b1ae7dacab2e81554b56e504fb2fd235657069c8716485e666cf436caa46803`,
RHI binary SHA256
`341af88a59bb7ed5b5d90bd9eab89e7013503883e41a5ffce3ecf123d395a8a9`,
and measured reference source commit
`140fbfcef3398cf3ed04eb8f3e4495e55f2894c9`.
The reference was frozen at `87fb70541a952d64de00bb42dc45ed94b7ae2e72`.
Comparison artifacts bind binary, compiler, actual loaded SPIR-V, source,
all three dependency patches and applied dependency source hashes.

## Scheduler failure and bounded correction

The first native core run failed Daxa's compact-batch assertion before compositor
dispatch. Pinned Daxa initialized the first access group's minimum batch to
`max_batch_index + 1` despite having no preceding access group. Optimization
could then leave an empty first batch. The third versioned dependency fix,
`cmake/patches/daxa-3.6-first-access-group-batch.patch`, initializes that first
group from the submit's first batch and preserves the existing previous-group
dependency rule. Reordering, transient lifetime optimization and assertions
remain enabled.

The actual narrowed native graph-construction/completion reproducer exits **-6**
unpatched (`scheduler-unpatched-red.json`) and **0** patched
(`scheduler-patched-green.json`). This reproducer performs no GPU dispatch.
Both artifacts remain under `artifacts/daxa-view-20261027/`. This is a fix to
MIT-licensed dependency source and adds no source-license obligations; binary
redistribution inventory remains a separate gate.

## Numerical and validation results

- `core-comparison/comparison.json`: all eleven strict and preview output planes
  are byte exact against the frozen RHI, including floats and 64-bit identity/age;
  no tolerance or NaN-payload waiver is used.
- `core-second.json` and `sync-final.json`: filtered native Daxa suites **5/5** each (two foundation tests, view CPU/GPU
  contracts and the native scheduler graph regression),
  Khronos layer loaded, zero warning/error diagnostics through teardown.
  The final synchronization run explicitly uses `VK_LAYER_VALIDATE_SYNC=1`.
- `720p-validation-report.json`: four Daxa-only strict/preview core and
  synchronization-validation runs pass with loaded layer and zero diagnostics.
- The initial paired forced-layer `core-comparison-run.json` is **FAILED in
  validation scope**: the frozen RHI emitted **112 error reports and 6 warning
  reports** (230 matched diagnostic lines), despite exact numerical parity.
  These counts include duplicate-limit reports and are not unique conditions. Its schema-1 numerical `passed: true` and zero process
  exit must not be read as validation success. The original evidence is retained;
  the frozen RHI remains unchanged. The updated harness records child streams,
  return codes and diagnostics, and separates numeric parity, validation
  cleanliness and the overall verdict.

Physical-buffer-address shader accesses limit synchronization validation's
ability to infer hazards. Clean syncval therefore supplements the explicit
resource/lifetime contracts; it does not prove all physical-pointer accesses safe
or qualify a live producer chain.

## Timing and memory scope

The preliminary `performance-comparison/` run may overlap other GPU commands and
is not used for performance conclusions. The final exclusive sequential run is
`final-performance-comparison/comparison.json` (schema 2): **30 samples after
3 warmups** per case on AMD Radeon 8060S Graphics, RADV/Mesa 26.2.3. The root
launched no concurrent GPU tests during that paired run. All eleven planes are
again byte exact in both strict and preview modes. Its `validation: false` is
intentional; no diagnostics were captured, but validation qualification comes
from the separate runs above, not this performance invocation.

| Mode | RHI total p50 / p95 (ms) | Daxa total p50 / p95 (ms) | RHI enqueue p50 (µs) | Daxa enqueue p50 (µs) | Daxa GPU span p50 (ms) |
|---|---:|---:|---:|---:|---:|
| Strict | 1.5538925 / 1.9571184 | 1.4559835 / 1.55984245 | 366.3 | 17.3425 | 1.393868 |
| Preview | 1.5134615 / 2.0440827 | 1.54474 / 1.7294149 | 292.1205 | 21.0345 | 1.451598 |

The resident slice has materially lower CPU enqueue cost, but no consistent
wall-latency advantage across modes: preview median total latency is higher.
Exact compositor/lifetime contracts and bounded core/synchronization results
permit incremental migration; they do not justify full production cutover or
a sustained 60 Hz claim.

Measurements separate one-time setup/upload, CPU enqueue, completion wait,
GPU timestamp span, CPU timestamp-query cost, and explicit full-plane diagnostic
export. The Daxa GPU timestamp spans the actual first scheduled pre-callback to
last scheduled post-callback across eight dispatches and internal barriers;
initial/final graph-boundary barriers are excluded. RHI has CPU submission and
completion timing, so its totals are not comparable to a Daxa GPU timestamp.
Serial wait-per-sample results measure latency, not fresh-frame throughput.

Requested retained buffer bytes are **309,756,712** for two output slots, including
retained staging and parameter storage. Requested growth is **zero** after setup,
warmup and measured samples. Actual driver/VMA committed or peak memory was
**not measured**; requested bytes do not include padding/internal allocations or
the separate diagnostic export's transient allocations. No complete peak-memory
envelope is claimed.

Raw reports, command lines and streams are ignored local artifacts under
`artifacts/daxa-view-20261027/`; source/performance review handoffs are under
`.superpowers/sdd/2026-10-27-daxa-slang-production/task2-{implementation,perf,review}.md`.

## Regression verification

The final root Python run reports **363 passed in 5.28 s**
(`artifacts/daxa-view-20261027/pytest-final.stdout`, empty stderr).
The fresh existing standalone RHI CTest run reports **5/5 passed in 2.12 s**
(`artifacts/daxa-view-20261027/rhi-regression.stdout`, empty stderr).
The clean Daxa 5/5 core and 5/5 synchronization suites above refer specifically
to the filtered Daxa tests, not an expanded full CTest count.
