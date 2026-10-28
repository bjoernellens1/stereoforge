# Native temporal reconstruction evidence — 2026-10-28

The native Depth → Temporal → View chain runs with bounded two-slot resources and retains its intermediate observations on the GPU. On static synthetic 1280×720-per-eye inputs, D128, four independent SGM paths per reference, and wave32, its measured GPU chain mean is **36.67–37.73 ms**. Temporal contributes about **2.50 ms**, and strict View about **1.95–1.96 ms**. This remains above the 16.67 ms input interval and the Fast60 target of p99 ≤15 ms. The next performance experiment is the matched multiresolution matrix in [the Fast60 plan](superpowers/plans/2026-10-28-fast60.md).

These are serial retained-output experiments on AMD Radeon 8060S / RADV Strix Halo, Mesa 26.2.3, Slang 2026.12. They do not qualify unique physical-camera throughput, live 60 Hz, held-out stereo accuracy, general floating-point motion, or translated-view completeness. The temporal semantics and precision domain are specified in [the contract](daxa-temporal-contract.md).

## Completion and correctness

A real-driver retry exposed an inclusive completion-boundary error: submission 18 was already complete, but the shared strict comparison treated equality as pending after submission 22. The common completion check now accepts `submitted <= completed_boundary`, matching the pinned Daxa queue/global-min implementation. Future tickets, outstanding read tickets, and externally retained references still prevent reuse. Duplicate View and Stereo checks use the shared helper. The formerly failing Depth → Temporal case completed all 20 warmup plus three measured iterations after this fix (`root-boundary-after-short.json`).

The independently reviewed snapshot is `performance-freeze-inclusive-boundary.json`, SHA-256:

```text
0da994ca1bb4efbfd603f7110baf6b878ed0cd93522894e20fb7f22a9c9a22e1
```

It binds source, fixtures, native binaries, compiler/dependency closures, build configuration, and SPIR-V identities. The review verified 1,510 hash leaves; existing shader binaries were unchanged by the host completion fix.

- Qualified CPU suite: **1,565 passed, 37 deselected**; **11/11 CPU CTests passed**.
- Actual-source host probes: 10,000 Temporal frames for each of two identity sizes, with **zero application allocations and zero implicit waits**. These use mocked GPU/producer boundaries and establish bookkeeping behavior, not GPU speed or a wait-free benchmark.
- Native GPU qualification: **10 contract runs** across core and synchronization validation passed. **23 exact-rational cases × two validation modes = 46 exact runs** passed, including the eleven Temporal planes and associated metadata checks.
- Synchronization-validation stress: **300 measured + three warmup**, 303 offered/completed, zero admission/capture-pool skips, and 303 retired outputs/sources/inputs at each stage. No validation warnings or errors, including teardown.

The stress run measured GPU-chain mean/p99/max **36.63 / 37.92 / 38.07 ms** and host-wall **37.52 / 38.77 / 39.12 ms**. Validation was enabled, so these are qualification timings rather than the unvalidated performance comparison below.

## Matched stage timings

Nine runs used three interleaved orders (ABC, BCA, CAB), with 20 warmup and 100 measured samples per run. A is Depth alone, B adds Temporal, and C adds strict View. Validation and diagnostic output readback were outside the measured performance path. Each serial iteration explicitly waits for terminal completion; the host column includes that wait. Desktop activity was not disabled.

Values below are ranges across the three runs of each variant, in milliseconds. GPU chain spans are measured directly; percentiles are not sums of stage percentiles.

| Variant | GPU mean | GPU p99 | GPU max | Host mean | Host p99 | Host max |
|---|---:|---:|---:|---:|---:|---:|
| Depth | 31.87–32.24 | 32.83–33.35 | 32.96–33.84 | 32.38–32.84 | 33.46–34.33 | 33.75–35.20 |
| Depth → Temporal | 34.60–34.78 | 35.47–36.30 | 35.74–36.40 | 35.14–35.37 | 36.16–36.93 | 36.36–37.06 |
| Depth → Temporal → View | 36.67–37.73 | 37.84–40.18 | 38.06–40.62 | 37.14–38.31 | 38.36–40.87 | 38.82–41.30 |

Direct Temporal stage means are **2.490–2.499 ms** across B/C, with p99 **2.539–2.574 ms** and maximum **2.588 ms**. View means are **1.947–1.958 ms**, p99 **1.978–1.996 ms**, maximum **2.056 ms**. These do not establish pipelined FPS or exposure-to-consumer latency.

Requested disjoint stage buffers total **840,500,528 B** for A, **1,179,650,352 B** for B, and **1,371,442,328 B** for C. Temporal owns **339,149,824 B**. These exclude driver/process peak memory, query/command-pool overhead, diagnostics, and the separate capture pool; they are not measured peak VRAM.

## Open numerical gates

The preserved moving-yaw general-float comparison remains failed: two depth words differ from the frozen float64-reference export by one ULP. The observed GPU word is `0x3f9be76c`, versus reference `0x3f9be76d`, near 1.218 m. Independent analysis explains this case through the specified final float32 transform coefficients; masks, confidence, ancestry, and counters remain exact. This explanation does not promote the strict comparison or establish a general rounding/contraction model. No epsilon was added.

A separate standalone Stereo strict CPU-float confidence gate also remains open. Its D2/P4/index95 result matches the preserved pre-existing diagnostic: best=56, second=63, confidence 7/63, GPU `0x3de38e38` versus CPU `0x3de38e39`. Integer Census, costs, sums, and validity passed before confidence failed; later standalone cases, timing, and teardown remain unqualified. The shared completion check is reached only by later timing code in that executable. This classification is not a matched pre/post replay or a relaxation of the confidence gate.

## Evidence and reproduction

Evidence paths below are relative to `artifacts/daxa-temporal-20261028/`. Artifacts are ignored local outputs and are not included in Git; these references identify the measured run, rather than downloadable release assets.

- `performance-review-inclusive-boundary.json` and its named CPU/probe logs.
- `root-boundary-after-short.json`.
- `root-validation-inclusive-v2/summary.json`, `exact-matrix/matrix.json`, and `720p-stress.json` under that directory.
- `root-timings-inclusive/summary.json`; raw samples are in `root-timings-inclusive/root-timing-r{1,2,3}-{depth_alone,depth_temporal,depth_temporal_view}.json`, field `raw_samples_ms`.
- `yaw-precision-diagnostic/report.md` and its bound JSON; the failed original general-float comparison is preserved at `root-matrix-v4/core/unequal-models-moving-yaw/comparison.json`.
- The current standalone confidence failure is preserved in `root-validation-inclusive/core-daxa_stereo_contracts.log`; its pre-existing baseline is `artifacts/daxa-stereo-20261027/core-confidence-diagnostic.log` relative to the repository root.

The comparator can generate a portable synthetic sequence and execute an explicit validated comparison after the native targets are built. Substitute the configured standalone compiler path:

```bash
PYTHONPATH=python .venv/bin/python benchmarks/compare_daxa_temporal.py export \
  --case identity --output artifacts/temporal-repro/fixture
PYTHONPATH=python .venv/bin/python benchmarks/compare_daxa_temporal.py compare \
  --sequence artifacts/temporal-repro/fixture/sequence.json \
  --native build/daxa/daxa_temporal_fixture \
  --compiler /path/to/slangc --output artifacts/temporal-repro/comparison --validation
```

Use the configured Vulkan validation layer/ICD environment and add synchronization validation for the second mode. Reproduction must bind the actual compiler, binaries, fixtures, dependencies, and shader hashes; generating new reports does not recreate the frozen snapshot automatically. Physical geometry, quality/completeness, live frame age/drops, and the ten-minute Fast60 gate remain future qualification work.
