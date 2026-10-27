# Paired per-path scheduling: exact output, higher latency

The first paired scheduling control preserves the combined depth outputs but is
slower than the sequential baseline on the measured AMD Radeon 8060S/RADV target.
`Sequential` remains the default. `PairedPerPath`, selected by
`--stereo-schedule paired-per-path`, remains an explicit experimental control;
there is no automatic promotion.

This evidence supplements the [depth contract](daxa-depth-contract.md) and
[combined-depth evidence](daxa-depth-evidence.md). It covers the source based on
`659f4fd67605d0057f0cb9f46d5f90bb462d7d18`, not a new physical-camera or release
qualification. The ignored local artifact root is
`artifacts/daxa-depth-paired-20261027/`.

## What changed

A persistent paired graph records the existing Left and Right dispatches back
to back within each Census, clear, aggregation-direction and selection phase.
Every phase timestamp pair encloses both eye dispatches; there is no timestamp
or explicit barrier between the two dispatches inside that phase. Same-eye
read/write hazards and barriers between aggregation directions remain required
because packed penalty accumulation uses ordinary non-atomic addition.

This is two existing dispatches in each graph phase. It is not yet one combined
kernel with a layered eye dimension. Census is deliberately still duplicated;
matching, disparity direction, penalties, selection and floating arithmetic are
unchanged. All **29 prior stereo/preprocess/depth SPIR-V modules are identical**
to the controls in `before.json` and `freeze.json`.

Existing image and parameter buffers are reused. Requested application buffers
remain **840,500,528 B** for the measured two-slot 720p profile. Added setup-time
graph, tracking and query-pool objects are outside that buffer count, as are
allocator/driver memory and diagnostic allocations. The number is not a GPU
peak-memory measurement.

Paired reports explicitly set `per_reference_stereo_timings_available=false`
and both `gpu_left_stereo_ms` and `gpu_right_stereo_ms` to null. The
`gpu_paired_stereo_ms` and named phase intervals cover both references. The
actual upload-start through metric-end queue interval remains the authoritative
combined GPU latency; no sums of isolated timings or percentiles substitute for
it.

## Matched ABBA result

`abba-summary.json` binds four separate child processes in sequential, paired,
paired, sequential order. Each processes the same retained recorded 1280×720
per-eye fixture, D128, four paths, P1=8/P2=32, subpixel enabled, wave32 and two
slots, with **100 measured samples after 20 warmups**. Validation,
`--check-reference` and dump output are absent from these timing runs. Packed
YUYV upload, preprocessing, both references, LR, metric depth and metadata are
inside the complete chain. No full image readback occurs in its timed path.

| ABBA run | Schedule | GPU mean / p95 (ms) | Host wall mean / p95 (ms) |
|---|---|---:|---:|
| 1 | sequential | 31.639122 / 31.966176 | 32.056245 / 32.392676 |
| 2 | paired-per-path | 39.772079 / 41.311372 | 40.254520 / 41.919675 |
| 3 | paired-per-path | 40.793352 / 42.333143 | 41.298220 / 42.733819 |
| 4 | sequential | 31.587630 / 31.837602 | 32.013248 / 32.313217 |

Both paired runs regress relative to both sequential runs. These measurements
reject this scheduling control as a latency improvement at this operating point.
They do not identify a cache, bandwidth, occupancy or synchronization bottleneck:
no runtime hardware-counter evidence establishes such a cause. The next proposed
causal experiment is a true layered per-path dispatch; it has not been
implemented or measured in this evidence.

The device is **AMD Radeon 8060S Graphics (RADV STRIX_HALO), Mesa 26.2.3**;
Slang is 2026.12 with compiler SHA256
`82e29ecd96714c64a31021160eaff4bf8604f44934e9f7b98448e962a5d12dad`.
The native benchmark SHA256 is
`969ce7da2ba323ee86d23edd26d0373b67a4393bc64363304a05aec2c8049340`;
the comparison harness SHA256 is
`605cd0e1a40e9541a4fee58905c91e6f9e0aa9e87351e3e10b35a14f18818c35`.
Individual reports bind fixture/calibration identities, record ordinal, source
metadata, shader/compiler identity and actual timing scope; the ABBA summary
binds each report hash and command.

## Correctness and CPU evidence

Root native GPU contracts pass in core and synchronization validation, covering
sequential/paired schedules, wave32/64 and one/two slots. Tests include changing
pixel bytes with retained source IDs, integer/Census/direct packed-penalty
checks, full-ring rejection, producer/output lifetime and metadata. The direct
core/sync stdout logs report success; stderr is empty.

The independent frozen-RHI harness passes all **nine observation planes and six
raw stereo planes byte for byte** in:

| Artifact directory | Profile | Counts |
|---|---|---|
| `small-paired-sync` | 8×5, D16/P4/W32, slots2, synchronization validation | 3 measured / 1 warmup |
| `720p-paired-core` | 1280×720, D128/P4/W32, slots2, core validation | 30 / 3 |
| `720p-paired-sync` | same 720p profile, synchronization validation | 30 / 3 |
| `small-sequential-core` | 8×5, D16/P4/W32, slots2, sequential core control | 3 / 1 |
| `dynamic-p8-w64-paired-sync` | 64×32 dynamic record0, D32/P8/W64, slots2, synchronization validation | 3 / 1 |

The comparison reports retain independently derived expected planes, both RHI
oracle commands and input/output identities. Native diagnostic checks also
compare both references' actual Census, validity, costs, sums and packed words
against independent integer scalar results. Full readbacks and oracle work run
after timing. Exact GPU/frozen-RHI bytes establish this compiler/device profile;
they do not close the general compound CPU float contract or introduce an
arbitrary scalar floating tolerance.

The root full Python suite passes **928 tests in 20.44 s**
(`root-full-python.log`); the four native CPU contract executables pass in
**0.60 s** (`root-native-cpu.log`). The mandatory independent performance pass
covers **198 focused CPU tests**, including 16 new source/mutation/probe tests.
Its extracted actual paired/Depth/preprocessing bookkeeping probe runs 256
warmed 720p frames with zero application heap allocations, one explicit
admission snapshot per frame, three mocked graph submissions per frame, and
3,686,400 packed bytes copied per frame. One recorded mean is **0.05569 ms/frame**.
This excludes Daxa internals, driver, actual GPU, garbage-collection work and
waits, and is not a measured native end-to-end speedup. SDK GC has additional
internal completion scans; native host-wall timing includes them and timestamp
query/readiness costs.

A separate `paired-300-sync.json` run passes 300 measured frames after three
warmups at 720p D128/P4/W32/slots2 with synchronization validation and native
checks enabled. It reports 303 completed output retirements, all four native
reference flags true, zero warnings/errors through teardown and empty stderr.
Actual queue mean/p95 are **41.245279/42.263060 ms**, host wall mean/p95
**41.920099/43.012350 ms**. Its **5408.448258 ms** final diagnostic cost is after
timing. This short static stress run is roughly 12 seconds of repeated GPU
work, not long sustained operation, fresh-source throughput or a bounded GPU
peak-memory proof. Its validation/checking scope differs from the no-validation,
no-readback ABBA runs, which remain the primary latency comparison.

## Reproduction

Build and validation-layer setup follow the existing
[depth instructions](daxa-depth-evidence.md#reproduction-and-remaining-qualification)
and [validation evidence](vulkan-validation-evidence.md). Use fresh output paths.
The direct native GPU contracts exercise both schedules:

```bash
ctest --test-dir build/daxa -R '^daxa_depth_gpu_contracts$' --output-on-failure
VK_LAYER_VALIDATE_SYNC=1 \
  ctest --test-dir build/daxa -R '^daxa_depth_gpu_contracts$' --output-on-failure
```

An independent paired comparison, including after-timing dumps/oracles, is:

```bash
PYTHONPATH=python .venv/bin/python benchmarks/compare_daxa_depth.py \
  --manifest artifacts/daxa-preprocess-20261027/fixture-720p/manifest.json \
  --compiler build/standalone/_deps/sf_slang_sdk-src/bin/slangc \
  --output artifacts/new-paired-exact \
  --frames 30 --warmup 3 --slots 2 --wave-size 32 \
  --disparities 128 --paths 4 --p1 8 --p2 32 \
  --stereo-schedule paired-per-path --validation
```

Use `VK_LAYER_VALIDATE_SYNC=1` and another fresh directory for synchronization
validation. To reproduce the timing control, leave validation and diagnostic
flags unset and run the native command four times with fresh output names,
selecting `sequential`, `paired-per-path`, `paired-per-path`, `sequential`:

```bash
build/daxa/daxa_depth_benchmark \
  --manifest artifacts/daxa-preprocess-20261027/fixture-720p/manifest.json \
  --output artifacts/new-abba-1.json \
  --frames 100 --warmup 20 --slots 2 --wave-size 32 \
  --disparities 128 --paths 4 --stereo-schedule sequential
```

These are static repeated-fixture latency samples. They do not measure fresh
camera throughput, exposure-to-display age, long sustained operation, physical
geometry accuracy or achieved 60 Hz. The complete chain remains above the
16.67 ms target, and view integration remains outside this experiment.
