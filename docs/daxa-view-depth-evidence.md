# Retained depth-to-view integration: bounded evidence

Task 6 full acceptance remains **OPEN**. The v2 lifetime repair has cleared
source review and bounded GPU contracts, and all 24 fixture comparison runs pass.
Refreshed long-run, ABBA and exported-coverage evidence is recorded below.
This document does not declare full live or physical Task 6 acceptance complete.

## Interface and geometry boundary

The typed `DaxaView` constructor accepts a shared `DaxaFoundation` and a retained
`DaxaDepthPipeline::Frame` prototype. `project(depth_frame)` consumes a compatible
retained frame in the same device context. View output leases retain producer
storage through downstream use. Admission is bounded by the configured slots;
full rings reject rather than wait or overwrite a live output. Readback remains
an explicit diagnostic operation. The v2 lifetime regression results below verify this ownership contract on the
bounded tested paths; they do not establish live-producer operation.

Original RGB and rectified depth have separate dimensions and namespaces.
Rectified axial Z is reconstructed with its rectified intrinsics, inverse
rectification rotation, original Right-to-Left transform where applicable, and
then the virtual-camera transform. Metric RGB uses the rectified Left/Right RGB buffers; preview RGB uses the
original images through their calibrated sampling association. The helper checks the supplied models
against effective OpenCV 4.14 quantized remap sampling. This establishes model/map
association, not physical calibration truth. Raw-grid depth resampling is not an
intermediate geometry representation.

Perspective output depth is virtual-camera axial Z; cylindrical output is
virtual ray range. Strict mode exposes supported metric geometry. Preview can
fill RGB-only pixels: these remain approximate, have no metric validity or
metric identity, and use the invalid owner sentinel. Preview RGB is not evidence
of wider measured depth coverage. Metric metadata retains selected observation
identity, age and provenance. Unsupported metric output uses canonical invalid
NaN/default metadata. Output RGB follows the retained RGB888 convention.

## Independent fixture qualification: v2 bounded passes

The separate `daxa_view_depth_fixture` executable creates a real typed producer,
waits for completion, and uploads nine known observation planes once before View
construction and consumption. It reports
`observations_overridden_for_fixture=true`, `measured_stereo_output=false`,
`metric_source=synthetic_known_geometry`, and `performance_qualified=false`.
Capture/device identities explicitly identify synthetic input. This path tests
calibration, input readers and compositing; it cannot qualify stereo accuracy.
The actual benchmark consumes untouched retained stereo/depth output and accepts
no observation-override option.

Twelve prepared profiles are under
`artifacts/daxa-view-depth-20261027/final-fixtures/`. They cover unequal raw and
rectified dimensions, nonidentity rectification, principal-point offset, Right
translation, nearest-surface ties and conflicts, missing and unsupported metric
samples, preview periphery, two rectified samples sharing a raw RGB cell, virtual
rotation, and perspective/cylindrical conventions. The required comparison covers
all eleven planes: depth, confidence, RGB, metric validity, RGB validity,
provenance, source, approximate, owner, source IDs and age.
`root-v2-oracle-matrix.json` records **24 passed runs**, comprising twelve profiles
each under core and synchronization validation, with strict and preview outputs
exact on all eleven planes. Seven perspective profiles additionally match the
frozen RHI control byte-for-byte after owner normalization only. The five
cylindrical profiles use the independent known-point oracle only.

The independent oracle uses known original-space 3D witnesses and actual
capture/OpenCV preprocessing. Its byte-exact CPU claims are restricted to the
specified rational geometry, copied metadata, exact confidence and provable
exposure clamp/unity cases. General compound floating-point agreement is not
claimed. Perspective also retains the unchanged frozen RHI compositor control;
only owner namespace labels are normalized. Cylindrical profiles explicitly mark
that control unavailable, since the old binary cannot represent cylindrical
projection. They do not convert ray range to Z to manufacture agreement.

Inputs, geometry associations, source/compiler/binary dependencies, compiled
shader identities, SPIR-V payloads and output bytes are bound before and after
comparison. The frozen RHI runtime does not expose its generated SPIR-V; its
scope is source/compiler/binary/dependency/configuration binding and actual
output comparison, not a claimed runtime SPIR-V hash.

## Validation split and preserved failure

The original failed run is preserved at
`artifacts/daxa-view-depth-20261027/root-oracle-core-00/`. Its typed Daxa child
returned zero, with empty stderr and zero reported validation warnings/errors
through teardown. The frozen RHI child also returned zero, but emitted validation
errors in **stdout**, including unknown device feature structures
and ray-tracing-related resource/stage validation. Therefore the comparator
failed. Return code zero does not override its diagnostic gate.

The frozen benchmark has no required-layer guard on this path. Updating or
suppressing its diagnostics would change the control boundary. To request native
validation while explicitly omitting control validation activation, first export
a new fixture, then compare it in the implementation worktree. The example
fixture and comparison directories must both be fresh. Existing local frozen
fixtures are ignored artifacts, not checked-in input files:

```bash
.venv/bin/python benchmarks/compare_daxa_view_depth.py export \
  --output artifacts/new-knownpoint-fixture --case tie --projection perspective

VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
VK_LAYER_PATH="$PWD/build/validation-layer/usr/share/vulkan/explicit_layer.d" \
LD_LIBRARY_PATH="$PWD/build/validation-layer/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
VK_LAYER_VALIDATE_SYNC=1 PYTHONPATH=python .venv/bin/python \
  benchmarks/compare_daxa_view_depth.py compare \
  --fixture artifacts/new-knownpoint-fixture \
  --compiler build/standalone/_deps/sf_slang_sdk-src/bin/slangc \
  --native build/daxa/daxa_view_depth_fixture \
  --rhi build/standalone/native_view_benchmark \
  --output artifacts/daxa-view-depth-20261027/new-postrepair-explicit-split-00 \
  --validation --native-only-validation
```

`--validation` alone retains both-child activation. The additional split flag
requires `--validation`; it removes and records inherited activation variables
only for the control child. There is no automatic retry or fallback. Both stdout
and stderr remain fatal diagnostic sources for both children, regardless of
validation activation. Per-child environments, diagnostics and log hashes remain
recorded on failure. Evidence distinguishes clean required native validation
from unavailable control validation qualification. It never claims overall
validation qualified, or proves the absence of an internally activated RHI layer.
All 24 v2 matrix runs use this explicit split. Native validation is clean,
including synchronization validation in the sync runs. Frozen-control numerical
parity is qualified separately; control validation remains unqualified.

## CPU evidence and lifetime repair boundary

`native-freeze-v2.json` binds the repaired payload with SHA256
`ae449be484a067f19b23099b8a6ed7ee3784ee4406a3523fac56e0c320ec9b4e`.
Independent functional and performance follow-up reviews are CLEAR for their
stated source/CPU scopes. All 58 SPIR-V payloads are unchanged in v2.
`performance-followup-review.json` and its referenced logs record:

- 1,436 Python tests passed, 37 deselected, in 43.08 seconds.
- Nine CPU CTest contracts passed in 0.56 seconds.
- 53 owned guard/probe/lifetime tests passed.
- 88 oracle CPU tests passed with warnings treated as errors.

These overlapping scopes must not be added together. The warmed extracted host
probe now includes the actual repaired Frame assignment: 10,000 frames,
zero application heap allocations, mean 0.000346474 ms per frame. It exercises
actual View admission, binding, validation, retirement and shared ownership;
GPU operations, graph execution/callback recording and producer getter/storage
boundaries are mocked. It excludes SDK/driver allocation and execution, real
GPU work and image transfers. It is not full-chain CPU latency or throughput.

The original legacy View synchronization run aborted with a Daxa child-lifetime
assertion. The preserved backtrace passes through `DaxaFoundation` destruction
and Frame assignment. See `root-legacy-view-sync.log` and
`root-legacy-view-sync-backtrace.log`. The repair keeps lifetime ownership valid
during assignment. All four v2 reruns returned zero with pass messages:
`root-v2-legacy-view-core.log`, `root-v2-legacy-view-sync.log`,
`root-v2-typed-view-core.log`, and `root-v2-typed-view-sync.log`.
The original failure remains preserved; the reruns establish only their tested
legacy and typed core/synchronization contract paths.

The actual small-input and 720p preview synchronization-validated chains both
passed 300 measured frames after three warmup frames, with zero reported
validation errors/warnings. See `root-v2-actual-small-sync.json` and
`root-v2-actual-720p-sync.json`. These are static retained-input validation runs;
they do not establish unique-input throughput or a live capture path.

## Historical and repaired full-chain measurements

`root-actual-abba-summary.json` records strict/preview/preview/strict runs of a
static repeated retained synthetic 1280x720-per-eye input, independent stereo,
D128, P4, W32 and two slots. Each run used 100 measured frames after 20 warmup
frames, without validation or output readback in the measured loop. These
measurements bind the **pre-repair** freeze. They are preserved separately from
the completed v2 reruns below.

| Historical measurement | Range of run means |
|---|---:|
| GPU chain queue span | 33.68–34.14 ms |
| Host wall time | 34.04–34.61 ms |
| Strict View GPU stage | 1.95–1.96 ms |
| Preview View GPU stage | 2.03–2.06 ms |
| Requested chain buffers | 1,032,292,504 bytes |

These are ranges of means, not percentiles. Requested bytes are reported buffer
requests, not peak driver/process memory. No FPS is derived by combining stage
measurements. The historical summary binds native freeze hash
`d790d42e2c8cbd4eaa5271dc129c752910838e55f42afcde1fea4989b6342641`.
`root-v2-actual-abba-summary.json` records the completed repaired strict/preview/
preview/strict sequence, again 100 measured frames after 20 warmup frames per
run with D128/P4/W32/two slots, sequential independent SGM, no validation or
readback in the measured loop. Payload hashes were verified before and after.
All four run means are included in these ranges:

| Repaired v2 measurement | Range of run means |
|---|---:|
| GPU chain queue span | 33.979471–35.854207 ms |
| Host wall time | 34.435118–36.315621 ms |
| Strict View GPU stage | 1.977694–2.064487 ms |
| Preview View GPU stage | 2.045264–2.062332 ms |
| Requested chain buffers | 1,032,292,504 bytes |

No improvement over the historical run is claimed. The first v2 run is higher;
shared-GPU variation is possible, but its cause was not established by these
measurements. The v2 summary binds the repaired native freeze above.

## Actual stereo output and exported coverage

`root-v2-export-summary.json` binds actual unmodified independent-SGM output
from the synthetic 720p fixture, with each raw dump's SHA256. This is separate
from the known-observation override fixture. Coverage fractions are:

| Projection/mode | Requested horizontal FOV | RGB valid | Metric valid | Approximate |
|---|---:|---:|---:|---:|
| Perspective strict | 0.6435011088 rad (36.87°) | 0.2406184895833 | 0.2406184895833 | 0 |
| Perspective preview | 0.6435011088 rad (36.87°) | 1.0 | 0.2406184895833 | 0.7593815104167 |
| Cylindrical preview | 1.57079637 rad (90°) | 0.40237847 | 0.16748155 | 0.23489692 |

The perspective FOV follows the synthetic calibration. Low metric coverage is
reported synthetic-fixture data, not physical coverage or accuracy qualification.
Preview RGB coverage does not increase metric support. All seven metric planes
(depth, confidence, validity, provenance, owner, source IDs and age) are byte-exact
between perspective strict and preview exports. Metric-invalid pixels were
checked for canonical NaN/default metadata in all three exports.

Post-timing diagnostic readback **plus disk export** took 207.34 ms for
perspective strict, 214.20 ms for perspective preview and 220.63 ms for cylindrical
preview. These are not transfer-only latency measurements and are excluded from
the retained no-readback ABBA loop.

Local summary SHA256 bindings:

- `root-v2-oracle-matrix.json`:
  `f329e144aa63413e7efbd6d14dc3a747b0bd7f5a197e8c81d40a7785ed92361e`
- `root-v2-actual-abba-summary.json`:
  `f990ca25f58e95c57276830031958d794e65192fe07453985a6061f30fb66eb3`
- `root-v2-export-summary.json`:
  `d3948c70d5dd5ac6c3a417e0b5bcb803f00d0262eedbf445d880cbc767ceceec`

Full live 60 Hz, unique-input throughput, frame age, offered/completed/dropped
inputs, backpressure under a live producer, physical depth/calibration truth,
general floating-point profiles and real-scene peripheral coverage remain open.
No fixture parity or static repeated-frame measurement closes those gates.
