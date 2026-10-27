# Shared Census with sequential stereo references

`--stereo-schedule shared-census-sequential` selects the explicit
`SharedCensusSequential` experiment. It reuses physical Census descriptors across
true Left and Right `PackedSpecialized` matching while processing the two
references sequentially. `Sequential` remains the default. Existing
[paired](daxa-depth-paired-evidence.md) and
[layered](daxa-depth-layered-evidence.md) experiments retain their controls and
recorded negative latency results.

## Storage, scheduling and ownership

Each aligned slot physically allocates one LeftDesc/RightDesc/LeftValid/RightValid
quartet. The right engine receives those IDs before its allocation loop and
skips allocating another quartet. Separate reference penalties, disparity,
confidence and validity buffers remain. The packed recurrence, selection,
subpixel helpers, symmetric LR checks and metric materialization are unchanged.

A bounded `CensusStorage` object owns the quartet and retains its
`DaxaFoundation`. Both producer slots retain that storage; it has no owning
engine or Frame back-reference. A right producer Frame can therefore outlive the
left engine and outer pipeline without dangling descriptor IDs or a reference
cycle. The descriptor-sharing constructor is private to the depth coordinator;
ordinary public stereo engines cannot independently submit against shared
storage.

`DaxaDepthPipeline::Frame::stereo_frame(reference)` returns a typed GPU producer
lease only in `SharedCensusSequential`. It validates the output frame and
reference enum and performs no wait, allocation or readback. Other schedules
reject this getter. Both aligned producer slots must be available before the
upstream packed upload; a retained right Frame alone blocks shared-slot reuse.
Existing generation, exact calibration-owner and context checks remain active.

The combined graph registers each shared descriptor resource once and aliases
both reference views to that registration. It dispatches:

1. Shared Census left, then shared Census right: two Census dispatches total.
2. Left clear, all left paths, then left selection.
3. Right clear, all right paths, then right selection.

Explicit resource dependencies order the preparatory phases and left selection
before right clear. Each reference retains its own packed-penalty read/write
hazards between paths. There are **14 phases for P4** and **22 for P8**. Shared
preparation cannot be independently charged to either eye: reports set isolated
per-eye timings to null and expose actual joint total/phase intervals under
`gpu_paired_stereo_ms` and `gpu_paired_stereo_phases_ms`. The dispatch form is
`shared_census_once_then_sequential_independent_references`; phases use
`shared_census_left/right`, followed by `left_` and `right_` clear/path/select
names.

The complete GPU measure remains actual same-queue upload-start through
metric-end, including intergraph gaps. Host wall includes staging, collection,
submission, completion and timestamp queries. No new per-frame wait, full
readback, hashing, CPU image work or application heap allocation was introduced.

At 1280×720 per eye, D128/P4/P2=32 and two slots, actual requested buffers decrease
from **840,500,528 B to 811,009,328 B**, saving **29,491,200 B**
(`16 * width * height * slots`). The new mode's 2 GiB preflight uses that actual
saving. Reports expose shared storage, two Census dispatches, independent
reference penalty buffers, descriptor bytes and saved bytes. This accounting
covers requested buffers only; it excludes queries, allocator/driver storage,
diagnostics and CPU-source storage and does not measure peak memory.

## Review and correctness evidence

Independent functional and mandatory performance-refinement reviews cleared the
frozen implementation before root GPU execution. Review caught a setup hazard:
ordinary bootstrap graphs now wrote the same descriptors. Shared mode now waits
for left bootstrap completion before submitting right bootstrap, with one wait
per eye during setup. Legacy bootstrap ordering remains unchanged. This adds no
wait to the frame path.

The final Python CPU suite passes **1193 tests in 28.40 s**; all four native CPU
contracts pass. Performance review passes 87 focused CPU tests, with static
guards for private-constructor restrictions and mutations that remove shared
ownership, bootstrap ordering, graph hazards or aligned admission. Extracted warmed host
probes report zero application allocations and one admission snapshot/frame;
they use GPU/SDK stand-ins and do not measure native GPU latency.

No shader or algorithm source changed. All **33 prior depth/control SPVs** and
all **46 SPVs** in the configured build remain byte-identical. `freeze.json`
binds seven native/API/CLI/test sources, the native benchmark and contracts
binaries, and those 46 modules. Root native GPU contracts pass under both core
and synchronization validation with empty stderr and zero teardown warnings or
errors. Coverage includes both references, one/two slots, W32/W64, P4/P8 and
8/16-bit packing, descriptor identity, independent penalties, retained-right
lifetime and backpressure.

The existing frozen-RHI comparison harness passes all nine observation planes
and six raw stereo planes byte for byte in each recorded gate:

| Artifact | Profile | Samples / warmups | Validation |
|---|---|---:|---|
| `small-shared-sync` | 8×5, D128/P4/W32, slots2 | 3 / 1 | Synchronization |
| `small-sequential-core` | Same small control | 3 / 1 | Core |
| `720p-shared-core` | 1280×720, D128/P4/W32, slots2 | 30 / 3 | Core |
| `720p-shared-sync` | Same 720p profile | 30 / 3 | Synchronization |
| `dynamic-p8-w64-shared-sync` | 64×32 dynamic record, D32/P8/W64, slots2 | 3 / 1 | Synchronization |

Each comparison preserves the independent OpenCV preprocessing and frozen-RHI
Left/mirrored-Right oracle gates. Native diagnostic checks independently compare
Census/masks, integer matching and actual packed penalty words. Diagnostic work
occurs after timing. General compound CPU-float reference flags remain false;
exact recorded GPU bytes do not establish physical depth accuracy.

## Matched performance

`abba-summary.json` records serial Sequential, SharedCensusSequential,
SharedCensusSequential, Sequential child runs on AMD Radeon 8060S Graphics /
RADV Strix Halo, Mesa 26.2.3 and Slang 2026.12. Each uses the same retained
1280×720-per-eye fixture, D128/P4/P1=8/P2=32, subpixel enabled, W32, two slots,
**100 measured samples after 20 warmups**. Validation, reference checks and dumps
are absent; diagnostic flags remain false and stderr is empty. Packed upload,
preprocessing, both references, LR and metric metadata are included.

| ABBA run | Schedule | GPU mean / p95 (ms) | Host wall mean / p95 (ms) |
|---|---|---:|---:|
| 1 | sequential | 31.670481 / 32.177638 | 32.084721 / 32.617313 |
| 2 | shared-census-sequential | 31.567512 / 32.117434 | 31.997049 / 32.526210 |
| 3 | shared-census-sequential | 31.693526 / 32.334085 | 32.188806 / 32.805407 |
| 4 | sequential | 31.622721 / 32.106322 | 32.036024 / 32.591614 |

The shared runs straddle the Sequential controls. These runs demonstrate the
buffer saving without establishing a repeatable complete-chain latency
improvement. Sequential remains the default. They do not identify a GPU cache,
occupancy, bandwidth or utilization cause, and serial child execution does not
imply globally exclusive system use.

The native benchmark SHA256 is
`1261c18317a2533b7b52afa4e49511d7430cfc92384c12ff05731532bf2118a4`;
the comparison harness SHA256 is
`d4b0c3a093ecbb81745728d4b6dff940f15d581e3cc060d8d21aebe682c81665`.
The summary binds child commands and report hashes; reports bind fixture,
calibration, source metadata and compiled identities. Artifacts are under
`artifacts/daxa-depth-shared-census-20261027/`.

A separate `shared-300-sync.json` run passes 300 measured iterations after three
warmups at the same 720p profile with synchronization validation and native
reference checks. It records 303 completed output retirements, all four native
diagnostic flags true, zero teardown warnings/errors, empty stderr and
811,009,328 B requested buffers. GPU mean/p95 are **31.630039/32.269492 ms**;
host wall mean/p95 are **32.457449/33.197065 ms**. Its final **5365.299982 ms**
diagnostic work occurs after timing. This short repeated-fixture stress run is
separate from the no-validation ABBA latency comparison and does not qualify
fresh-source or sustained live operation.

## Reproduction and limits

Use the optional build and validation setup in
[depth evidence](daxa-depth-evidence.md#reproduction-and-remaining-qualification).
For a fresh comparison destination:

```bash
PYTHONPATH=python .venv/bin/python benchmarks/compare_daxa_depth.py \
  --manifest artifacts/daxa-preprocess-20261027/fixture-720p/manifest.json \
  --compiler build/standalone/_deps/sf_slang_sdk-src/bin/slangc \
  --output artifacts/new-shared-census-exact \
  --frames 30 --warmup 3 --slots 2 --wave-size 32 \
  --disparities 128 --paths 4 --p1 8 --p2 32 \
  --stereo-schedule shared-census-sequential --validation
```

Set `VK_LAYER_VALIDATE_SYNC=1` for synchronization validation. Timing-only native
runs omit validation, `--check-reference` and dumps. Keep matched fixtures,
settings, source/compiler/binary/SPV bindings and host allocation scope.

This evidence concerns static repeated-fixture source-grid depth. Fresh camera
throughput, camera age, physical calibration/depth accuracy, wide RGB-D/view
integration, temporal/adaptive processing and sustained live 60 Hz remain open.
