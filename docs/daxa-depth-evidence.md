# Dual-reference depth: current evidence

The experimental native pipeline now executes staged preprocessing, true left
and right packed-specialized stereo, symmetric LR, rectified metric depth and
fresh source metadata on one Daxa foundation. The API, layouts, calibration and
lifetime rules are specified in the [depth contract](daxa-depth-contract.md).
Native validation and independent frozen-RHI comparisons establish the tested
source-grid boundary below. General compound CPU-float, physical geometry, wide
RGB-D and sustained live qualification remain open.

## Verified source and CPU boundary

The setup extension retains validated original/rectified models, poses,
R1/R2/P1/P2 and identity bindings. Missing geometry blocks metric processing;
existing manual 2D configurations remain supported. Matrix/hash/JSON validation
is setup-only. Root review corrected scale-dependent baseline comparisons and
projection structural zeros before source freeze.

Independent performance review checked persistent slots, whole-chain
pre-upload backpressure, producer/consumer leases, completion ordering and the
actual same-queue timestamp span. Its extracted CPU probe uses real depth and
preprocessing admission/retirement/staging code with GPU/Stereo submission
stand-ins: 256 warmed 720p frames, 3,686,400 copied bytes/frame, zero application
allocations and one explicit admission snapshot/frame. The final sample averaged
0.0512 ms/frame; it excludes real SDK/driver/GPU work and is not GPU throughput.

Naive float32 pixel-plus-disparity rounding had a watched failing regression.
The corrected integer/half-threshold helper passed 46,350 independent
half/nextafter index cases against a double-coordinate oracle, including large
pixel coordinates and negative ties. No float tolerance was introduced.

Four native CPU contracts pass. The current full Python test run passed
**812 tests in 18.34 s**, with native capture bindings selected by
`STEREOFORGE_NATIVE_DIR=build/native-capture-joint/native` and `PYTHONPATH=python`.
The retained log is `root-validation/full-python-tests-final.log`. The independent
focused performance/ownership checks passed 139 tests. These CPU checks do not
replace native GPU or output-accuracy gates.

The frozen manifest `artifacts/daxa-depth-20261027/freeze.json` binds sources,
compiled native binaries and shader identities. All 25 prior stereo and two
preprocessing SPIR-V files remain unchanged; two observation entries are added:
`lr_consistency` and `metric_observation`. The approved preprocessing raw timing
payload reuses already queried timestamp values and adds no GPU pass/query.

## Root GPU validation

Recorded on AMD Radeon 8060S / RADV Strix Halo, Mesa 26.2.3, pinned Daxa and
official Linux Slang 2026.12:

- Native depth GPU contracts pass under core and synchronization validation.
  Tests exercise literal LR/metric/default/64-bit planes, half/nextafter index
  boundaries, strict support, pairing/context/owner rejection, retained/in-flight
  slots, independent RGB pressure and producer lifetime.
- A direct synchronization run reports the integer/metadata/lifetime contracts
  passed, with zero bytes on stderr. Compound float qualification remains
  explicitly false.
- The 8×5/D16/four-path small fixture passes three measured iterations after one
  warmup, with two slots and wave32. All four diagnostic flags are true: expected
  preprocessing, both-reference integer, actual packed penalties and
  LR/mask/metadata checks. Teardown warnings/errors are zero.
- Six additional checked synchronization cases pass across the small fixture
  and both different-content dynamic records, D16/D32/D128, four/eight paths,
  wave32/wave64 and one/two slots. Every recorded stderr stream is empty;
  `native-matrix.json` retains the exact commands.
- A 300-iteration checked synchronization run at full 720p completes with 303
  output retirements including warmup. All four diagnostic flags are true,
  warnings/errors through teardown are zero and stderr is empty.

The first core invocation failed before pipeline execution because the Khronos
layer path was absent. That environmental failure is retained in `gpu-core.log`;
`gpu-core-layerpath.log` records the subsequent pass using the qualified local
layer (Fedora validation layers 1.4.341.0-2.fc44). No validation diagnostic was filtered into a passing result.

Checks use independent OpenCV fixture luma for both scalar stereo integer
oracles, including actual GPU packed-word comparison. LR/mask/metadata checks
consume the actual GPU stereo float arrays. General metric depth checks are
finite-positive/canonical-invalid domain checks; isolated exactly representable
planes have exact native GPU tests. No blanket compound CPU-float tolerance or
physical depth-accuracy claim is made. Independent GPU-byte comparisons close
the separate recorded-output gate for the profiles below; they do not establish
general scalar CPU-float accuracy.

## Independent frozen-GPU output comparison

The reviewed `compare_daxa_depth.py` harness compares all **nine observation
planes** and all **six raw left/right stereo planes** exactly, including float
bits, canonical invalid defaults, support/LR masks and signed 64-bit metadata.
Recorded passes are:

| Fixture | Profile | Measured / warmup | Validation |
|---|---|---:|---|
| Small 8×5 | D16, four paths, wave32, two slots | 3 / 1 | Core and synchronization |
| Full 1280×720 | D128, four paths, wave32, two slots | 30 / 3 | Core and synchronization |
| Dynamic record 0, 64×32 | D32, eight paths, wave64, two slots | 3 / 1 | Synchronization |

The oracle uses independent OpenCV fixture RGB/luma/support, frozen RHI left
matching and literal mirrored right/left matching followed by output-row
unmirroring. Declared LR/support/default/metadata rules combine those outputs.
Metric operands use the reported cached float bits, and metric values come from
frozen GPU depth arithmetic. This is an exact observed GPU execution profile;
it does not replace the open compound CPU-float contract with a tolerance.

All children exit zero, native teardown warnings/errors are zero, and the
harness binds source, fixture/calibration bytes, compiler, dependencies, native
binary, all relevant SPIR-V, native reports and dumps. The reviewed depth harness
source SHA256 is `3530c2847350e4603b621e152c85595eb999e8a22fa22c7f3116d298c0986721`.
RHI compiles frozen source at runtime and exposes adapter name rather than an
independently checked device UUID; that identity boundary remains explicit.
Its two transfer-inclusive oracle runs have no separate warmup and provide no
comparable end-to-end performance ratio.

The exact comparison directories are `harness-small-exact-core`,
`harness-small-exact-sync`, `harness-720p-exact-core`,
`harness-720p-exact-sync` and `harness-dynamic-p8-w64-exact-sync` under
`artifacts/daxa-depth-20261027/`, each with `comparison.json`. The following
no-validation resident measurement remains the primary performance result;
comparison runs include final diagnostic exports outside their sample clocks.

## Actual combined 720p cost

Profile: full 1280×720 per eye, D128, four paths, P1=8/P2=32, subpixel=true,
LR tolerance 1, two slots, required full wave32, 8-bit packed penalties. RGB
synthesis is disabled. Inputs repeat one retained packed record; every iteration
includes its staging upload and preprocessing. The benchmark waits for each
output, so this is serial per-iteration cost, not saturated asynchronous or fresh
camera throughput.

| Run | GPU queue span mean / p95 | Host wall mean / p95 |
|---|---:|---:|
| 30 measured + 3 warmup, validation disabled, no readback | 31.683376 / 32.087256 ms | 32.115600 / 32.574330 ms |
| 300 measured + 3 warmup, synchronization validation, final checks | 31.674728 / 31.990703 ms | 32.417868 / 32.857137 ms |

The no-validation run's queue p50 is 31.624337 ms and host p50 32.056566 ms.
Its mean GPU stages are upload 0.038700, decode 0.037530, rectify 0.237235,
left stereo 15.283821, right stereo 15.189094, LR 0.185090 and metric/metadata
0.440818 ms. Combined time comes from the actual first/last GPU timestamps,
including inter-graph gaps; the individual stage statistics are not summed to
invent a frame percentile. Host wall includes collection/admission, staging,
submissions, wait, timestamp/readiness work and SDK garbage-collection scans.

Requested combined buffer bytes are **840,500,528**, including **162,201,936**
observation buffer bytes. These are requested setup buffers, not measured GPU
peak usage, allocator/driver memory or a long-run leak bound.

The no-validation run performs no full readback/reference checks: all four
reference flags are false. Its initialization is 75.543211 ms and retirement
count 33. The checked synchronization run reports final diagnostics separately:
5,306.147078 ms **after** timing, initialization 82.624149 ms and retirement
count 303. Its roughly ten-second repeated-record workload is short static
stress, not sustained live qualification. Neither run establishes 16.67 ms or
60 fresh frames/s; measured combined latency exceeds that budget.

Capture delivery/drops, camera-clock age, physical calibration truth, original
FOV RGB-D synthesis and translated-view geometry are outside these results.
Original RGB remains retained, but the current metric layers use rectified rays.
No serial sums of earlier separate-reference medians are used here.

## Reproduction and remaining qualification

```bash
cmake --preset daxa -DSTEREOFORGE_DAXA_STEREO=ON \
  -DSTEREOFORGE_DAXA_PREPROCESS=ON -DSTEREOFORGE_DAXA_DEPTH=ON
cmake --build build/daxa -j4
ctest --test-dir build/daxa \
  -R '^daxa_(calibration|preprocess|stereo|depth)_cpu_contracts$' \
  --output-on-failure
```

GPU contracts require the validation-layer environment in
[validation evidence](vulkan-validation-evidence.md):

```bash
ctest --test-dir build/daxa -R '^daxa_depth_gpu_contracts$' --output-on-failure
VK_LAYER_VALIDATE_SYNC=1 \
  ctest --test-dir build/daxa -R '^daxa_depth_gpu_contracts$' --output-on-failure
```

Use modern synchronization settings alone; do not combine deprecated settings
that produce configuration warnings. For a new pre-exported fixture and new
output destinations:

```bash
build/daxa/daxa_depth_benchmark \
  --manifest /path/to/fixture/manifest.json \
  --output artifacts/new-depth-check.json --dump-prefix artifacts/new-depth-check \
  --frames 3 --warmup 1 --slots 2 --wave-size 32 \
  --disparities 128 --paths 4 --p1 8 --p2 32 --validation --check-reference
```

A timing-only run uses frames 30/warmup 3 and
omits validation, dump prefix and check-reference. Fixture generation is offline;
native execution has no Python runtime requirement. Calibration identities,
record ordinal, compiler/shader/binary hashes and output report bytes must remain
bound when making a measurement claim.

The independent comparison can be reproduced with the fixture above, the
reviewed harness and the configured offline compiler (a fresh output directory
is required):

```bash
PYTHONPATH=python .venv/bin/python benchmarks/compare_daxa_depth.py \
  --manifest artifacts/daxa-preprocess-20261027/fixture-720p/manifest.json \
  --compiler build/standalone/_deps/sf_slang_sdk-src/bin/slangc \
  --output artifacts/new-depth-exact \
  --frames 30 --warmup 3 --slots 2 --wave-size 32 \
  --disparities 128 --paths 4 --p1 8 --p2 32 --validation
```

Repeat in another new directory with `VK_LAYER_VALIDATE_SYNC=1` for
synchronization validation. Use the actual configured compiler path if it
differs. The recorded compiler SHA256 is
`82e29ecd96714c64a31021160eaff4bf8604f44934e9f7b98448e962a5d12dad`.
The harness needs Python/OpenCV 4.14.0 offline; the native processing executable
retains its C++-only runtime boundary.

General compound CPU float qualification remains open. DaxaView intake,
rectified-to-raw depth/rays, temporal/adaptive work, held-out physical geometry
and sustained live acceptance remain later gates.

Local ignored artifacts are under
`artifacts/daxa-depth-20261027/root-validation/`: core/sync logs, direct sync
stdout/stderr, `small-sync-native.json` and dumps, `720p-performance.json`,
`720p-stress-sync.json`, `native-matrix.json`, `native-cpu-final.log`, `full-python-tests-final.log` and their
streams. These are local evidence, not bundled
camera data or a redistributable binary release.
