# Native preprocessing and GPU stereo integration evidence

Measured 2026-10-27 on AMD Radeon 8060S / RADV Strix Halo, Mesa 26.2.3,
with the pinned Daxa 3.6 revision and official Linux Slang 2026.12 compiler.
This is an experimental native C++20 path with no Python or SlangPy in frame
processing. The Python comparison harness is offline orchestration and evidence
checking. See the [numerical and ownership contract](daxa-preprocess-contract.md).

## Measured boundary

The input is one retained synthetic packed YUYV record, 1280×720 per eye,
2560×720 side by side, 5120 bytes per row. Calibration is synthetic, with
identity rectification maps and a 0.12 m baseline. Repeating this record does
not measure fresh camera throughput or validate physical metric geometry.

Every timed iteration copies the packed source to persistent staging, submits
YUYV decode and RGB rectification/grayscale, and optionally submits one-reference
D128/four-path/subpixel stereo on the same Daxa context. The serial host wall
interval includes retirement, CPU upload, submission, completion wait and GPU
timestamp retrieval. Stereo consumes GPU luma directly, without intermediate
image readback or a CPU completion wait between the two graphs.

No camera acquisition, right-reference estimation, LR consistency, metric-depth
conversion, wide RGB-D, temporal work or downstream consumer is timed here.
Final diagnostics run after all samples. Resident-performance mode still checks
one final set of eight preprocessing planes on the CPU; its reported diagnostic
cost is about 117 ms and is excluded from timed iterations. Stereo intermediate
readback and scalar checks are disabled in that mode and explicitly flagged false.

## Performance

Thirty measured iterations after three warmups, two resource slots, validation
disabled. Values are milliseconds. These are separate runs, not stage values
assembled into a synthetic end-to-end latency.

| Measurement | Preprocessing only p50 / p95 | Preprocessing + stereo p50 / p95 |
|---|---:|---:|
| Complete measured host wall | 0.448 / 0.556 | 16.077 / 16.360 |
| CPU enqueue, including CPU upload | 0.083 / 0.166 | 0.203 / 0.332 |
| CPU upload copy, included above | 0.065 / 0.124 | 0.120 / 0.173 |
| GPU preprocessing total | 0.289 / 0.294 | 0.276 / 0.290 |
| GPU staging copy | 0.038 / 0.040 | 0.020 / 0.021 |
| GPU original RGB decode | 0.037 / 0.038 | 0.038 / 0.038 |
| GPU RGB remap + grayscale | 0.210 / 0.213 | 0.215 / 0.229 |
| GPU one-reference stereo | — | 15.444 / 15.614 |
| CPU completion wait, included in wall | 0.363 / 0.454 | 15.846 / 16.053 |
| CPU timestamp queries, included in wall | 0.002 / 0.006 | 0.009 / 0.020 |

The preprocessor requests 103,219,456 bytes for its two-slot staging/maps/output
configuration. This excludes stereo, driver allocations, temporary diagnostics
and CPU source storage. It is not a measured memory peak.

The 16.077 ms median leaves little room within 16.67 ms, and the missing stages
remain substantial. It does not qualify a complete 60 Hz RGB-D pipeline.

## Correctness and lifetime gates

The final full Python suite passed 698 tests; both native CPU contract tests
passed. These CPU results do not replace the GPU gates below.

- Fourteen bounded core/synchronization cases passed: small and different-content
  64×32 fixtures, one/two slots, D16/D32/D128, four/eight paths and required
  wave32/wave64 configurations. Each checks eight OpenCV output planes, actual
  stereo Census/validity/cost/sums/final masks and independently packed scalar
  penalties against actual GPU words.
- Full 1280×720/D128/four-path/wave32 checks passed separately under core and
  synchronization validation, including exact eight-plane and integer/packed
  comparisons. Compound floating-point scalar qualification remains open.
- Native GPU contract tests passed under core and synchronization validation.
  They cover padded rows, retained outputs, different-image GPU inputs, foreign
  map owners, prototype release, input retirement and capture-pool recovery.
- A standalone 720p preprocessor completed 300 measured iterations plus three
  warmups under synchronization validation, retired all 303 sources and matched
  final output bytes. This checks command-pool recovery; it is not a long-term
  leak or sustained-camera qualification.
- All these validation processes captured zero warnings/errors through teardown.
  Harnesses bind source, headers, compiler, binary, dependencies, SPIR-V, fixture
  and report bytes before/after execution.

The app diagnostic extension preserved the measured loop byte-for-byte. Relevant
21 existing stereo SPIR-V files remained unchanged; the two preprocessing shaders
are additional. Optional fixture parsing uses pinned nlohmann/json and separately
obtained OpenSSL Crypto 3.x, with retained and build-copied notices.

## Reproduction

```bash
cmake --preset daxa -DSTEREOFORGE_DAXA_STEREO=ON \
  -DSTEREOFORGE_DAXA_PREPROCESS=ON
cmake --build --preset daxa
ctest --test-dir build/daxa \
  -R '^daxa_(stereo|preprocess)_cpu_contracts$' --output-on-failure
# Requires the Khronos validation layer; this executes native GPU contracts.
build/daxa/daxa_preprocess_contracts
```

Configure the local validation-layer paths as described in
[validation evidence](vulkan-validation-evidence.md). For synchronization
validation set `VK_LAYER_VALIDATE_SYNC=1`; do not combine deprecated layer
settings that generate configuration warnings.

Export a lossless packed recording and matching calibration into a new offline
fixture directory. Python/OpenCV 4.14.0 are needed for fixture creation only.
Keep recordings, calibration and generated dumps outside version control.

```bash
PYTHONPATH=python .venv/bin/python benchmarks/compare_daxa_preprocess.py \
  --export /path/to/capture.sfp --record-ordinal 0 \
  --calibration /path/to/calibration.json --output artifacts/new-fixture
PYTHONPATH=python .venv/bin/python benchmarks/compare_daxa_preprocess.py \
  --manifest artifacts/new-fixture/manifest.json \
  --compiler build/standalone/_deps/sf_slang_sdk-src/bin/slangc \
  --output artifacts/new-preprocess-check --validation --validation-only \
  --frames 3 --warmup 1 --slots 2 --stereo \
  --disparities 128 --paths 4 --wave-size 32
```

Use the actual configured `slangc` path if it differs. After parity and validation
pass, repeat with a fresh output directory, `--frames 30 --warmup 3
--resident-performance`, omitting `--validation` and `--validation-only`. For
preprocessing-only measurements also omit `--stereo` and its profile options.
Every output directory must be new. Full exact diagnostics, including scalar
stereo, can take seconds and are reported separately from GPU timing.

Local ignored evidence is under `artifacts/daxa-preprocess-20261027/`: the
`root-final-matrix-summary.json`, `root-final-720p-core`, `root-final-720p-sync`,
`root-final-720p-standalone-stress-sync`, `root-final-720p-preprocess-performance`
and `root-final-720p-integrated-performance` reports. These artifacts are local
evidence, not bundled camera data or a redistributable binary release.
