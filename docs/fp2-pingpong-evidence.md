# FP2 shared-state ping-pong experiment

The isolated ping-pong shader passed scoped parity but did not demonstrate a meaningful performance improvement over the existing parallel shader. In a matched old/new/old test, GPU p50 was 91.948 / 91.394 / 91.659 ms. The distributions overlap substantially; the small median difference does not establish a robust win. Both implementations remain well above the 16.67 ms frame interval in this synthetic diagnostic.

The candidate alternates two shared predecessor/current arrays and removes a shared-state copy and one workgroup barrier per pixel. It retains the seven-step shared minimum reduction and all 128 lanes participate even when a compact band has fewer candidates. It is not work-proportional compact recurrence. This result does not establish live-camera throughput, transfer-inclusive latency, physical depth quality, unrestricted full-resolution SGM or native/subpixel equivalence.

## Parity and measurement

The GPU parity fixture compared all four path arrays, full sums and padding with the independent CPU dense oracle for both eyes on synthetic 7×5 and 320×180 inputs. On the pinned analytic 1280×720 input it compared final raw selections and observations to frozen CPU selections and integrated GPU hashes; it did not compare 720p per-path arrays. The GPU raw valid counts were 897,952 / 898,067 and LR/metric-valid counts were 810,398 / 813,291, left/right. Metric conversion uses the scoped integer-disparity LUT and source ID 7. The 720p host requested 656,178,404 GPU buffer bytes, not measured peak memory or total driver allocation.

The matched test used the same preuploaded frame, resources, one in-flight slot, 10 warmups and 100 samples per block. Its independently bound prior parallel and ping-pong parity proofs were checked before timing, and the nonblocking shared GPU lock covered all three blocks. No timed full-frame readback occurred. Durations below are milliseconds; percentiles use NumPy linear interpolation. GPU timestamps had 64 valid bits and a 10.019 ns period, with availability and wrap checks. Raw per-frame durations are retained; raw ticks are not. CPU wait overlaps GPU execution and must not be added to GPU time.

| Block | GPU p50 | GPU p95 | GPU p99 | GPU max | CPU record p50 | CPU submit p50 | CPU wait p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Existing parallel 1 | 91.948 | 95.468 | 96.469 | 97.316 | 0.647 | 0.134 | 92.388 |
| Ping-pong | 91.394 | 95.173 | 95.920 | 96.158 | 0.647 | 0.125 | 91.673 |
| Existing parallel 2 | 91.659 | 94.664 | 94.991 | 95.032 | 0.669 | 0.143 | 91.994 |

Parity and all timing blocks exited 0 on AMD Radeon 8060S / RADV with Khronos instance/device validation; fixture counters were checked through teardown. Synchronization validation was requested but not independently attested. Loader notices are distinct from fixture diagnostics. Before/after source, binary, SPIR-V, manifest, baseline and prior-proof hashes matched. The 307 CPU/guard tests passed before GPU execution, using `/dev/shm` for temporary compiler files.

## Identities

| Artifact | SHA-256 |
|---|---|
| Ping-pong shader source | `399ddc656c99b4d52c86956a1c6ebfd90bbfc365cc468d3708666ddce10b3709` |
| 720p host source / binary | `5b8662cc216d63a9c9c657f4d1c152c8efabe8c02c204c90b561068992969619` / `9c1ee9d1ab57597c1f4e20158e10eef16ea3433ed001a83cdd5c9126d61d9d31` |
| Tiny host source / binary | `9711470f337543214372218508f2874b5eb994d204964b9b6f3c9802edb98bf1` / `fd42f7e2a6f1d58cf12c41d2beff84d35779f7cf55a9725c2127b6440f989523` |
| Parity / matched harness | `fde77a2d13f85038adc5688251579c49f67b8ce5283e21322cbaebcc124dda26` / `474aa923ed71cc89dd7794fee48fe35c64e36a8c521af73727be26a384a2dfa8` |
| Existing parallel / ping-pong proof | `ce5610937f578f7e7a15c302374bb5efa69687f041739f282f0b63cfe5f07c18` / `9abe70ad431f1f8971be37276df4b11746e793715157a8faddad6d80c7473dac` |
| Matched summary | `bc0305e3ddd791f69851539e18fbd2873b7e2e473b2ffcdff93668fe2f4b52fd` |
| Old 1 / new / old 2 sample files | `baea0139ca2999c314f0a90b98fa5c6e4853f812f0c4959a331b82756fc848f1` / `02e592a9a94aa1728d39eb8d28a9aadcc19bc02f8b775f0bcf9fbdef84b420f7` / `c9264d7ae3e86a574cecbdeb3b8c2d1d41024684512733d08eeda78a1b38a27f` |

The ignored local directories `artifacts/fp2-pingpong-20261029-v1/` and `artifacts/fp2-pingpong-matched-20261029-v1/` retain parity and per-block logs, exits, identities, sample durations and result files, and may be absent in a fresh checkout. The pinned manifest SHA-256 is `7f70c357ec371d5c434c435e4a2500d87a149e43a5859e1695ada38cf021ce17`; the baseline result SHA-256 is `e690b1895b63cf2c9c158419bf0662e7e316426d1d3c164f8c078d3700090041`.

## Reproduction

Use the pinned local artifacts, configured Daxa build and validation layer described in [parallel evidence](fp2-parallel-evidence.md). Run the complete guards before GPU work:

```bash
TMPDIR=/dev/shm cmake --build build/daxa --target fp2_parallel_fixture fp2_parallel_tiny fp2_pingpong_fixture fp2_pingpong_tiny -j 4
TMPDIR=/dev/shm PYTHONPATH=python:. .venv/bin/python -m pytest -q \
  tests/test_fp2_pingpong_cpu.py tests/test_fp2_pingpong_hotpath.py tests/test_fp2_pingpong_matched.py \
  tests/test_fp2_parallel_cpu.py tests/test_fp2_parallel_hotpath.py \
  tests/test_fp2_resident_cpu.py tests/test_fp2_resident_hotpath.py \
  tests/test_fp2_integrated_cpu.py tests/test_fp2_integrated_hotpath.py \
  tests/test_fp2_metric_lut_cpu.py tests/test_fp2_selection_cpu.py tests/test_fp2_selection_hotpath.py \
  tests/test_fp2_invalid_index_hotpath.py tests/test_fp2_720p_prepare.py \
  tests/test_fp2_720p_diagnostic.py tests/test_fp2_720p_hotpath.py \
  tests/test_fp2_census_cpu.py tests/test_fp2_census_hotpath.py \
  tests/test_fp2_image_cpu.py tests/test_fp2_image_hotpath.py \
  tests/test_fp2_slots_cpu.py tests/test_fp2_slots_hotpath.py \
  tests/test_fp2_prototype_cpu.py tests/test_fp2_dispatch_hotpath.py
```

For fresh ping-pong parity, use the locked command from [resident evidence](fp2-resident-evidence.md) with `benchmarks/fp2_pingpong_diagnostic.py`, `build/daxa/fp2_pingpong_fixture`, and `build/daxa/fp2_pingpong_tiny`; choose a fresh output directory and retain exit, logs and before/after hashes. Do not run timing after any parity failure. Then use the resulting proof and its computed SHA in a fresh comparison output:

```bash
proof=/path/to/fresh/pingpong-parity/result.json
proof_sha=$(sha256sum "$proof" | cut -d ' ' -f 1)
TMPDIR=/dev/shm PYTHONPATH=python:. .venv/bin/python benchmarks/fp2_pingpong_matched.py \
  --output-root artifacts/fp2-pingpong-matched-new \
  --new-proof "$proof" --new-proof-sha256 "$proof_sha"
```

The comparison harness binds the existing parallel proof, both current binaries, candidate proof, pinned inputs and baseline; it sets the AMD ICD and Khronos paths, requests sync validation, holds `/tmp/stereoforge-gpu.lock` nonblocking, and applies a 300-second process-group watchdog per block. Inspect the device and validation logs and retain actual exit statuses. Sync activation must be independently established before claiming synchronization validation.
