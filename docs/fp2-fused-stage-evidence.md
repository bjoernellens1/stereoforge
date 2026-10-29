# FP2 fused-cost stage profiling

The isolated fused-cost stage profiler passed synthetic GPU parity and completed a balanced timing run on AMD Radeon 8060S / RADV. GPU whole-chain p50 was 64.272 ms instrumented and 64.341 ms for its matched control. This remains well above the 16.67 ms frame interval. The measurement does not establish live-camera throughput, transfer-inclusive latency, physical depth quality, unrestricted full-resolution SGM, or native/subpixel equivalence.

The profiler uses the existing [fused-cost shader](fp2-fused-evidence.md) unchanged and binds its earlier parity proof as well as a profiler-specific parity proof. The 7×5 and 320×180 GPU fixtures checked both eyes' four path arrays, sums and padding against an independent dense CPU oracle. The pinned analytic 1280×720 run checked final raw selections and observations against frozen CPU selections and integrated GPU hashes; it did not compare 720p path arrays. Metric conversion uses the scoped integer-disparity LUT and source ID 7.

Both control and instrumented modes add the same stage-boundary compute barriers; the instrumented mode also writes timestamp annotations. Thus control is not the original uninstrumented fused timing path, and stage intervals are ordered, barrier-inclusive queue intervals, not bare kernel durations. The experiment still executes all 128 lanes at each path pixel, including inactive candidates; it is not work-proportional compact recurrence.

## Balanced timing

The run used 10 balanced warmups, then 25 ABBA blocks (50 samples per group), one in-flight slot and one repeated preuploaded analytic frame. No full-frame readback occurred during timing. GPU timestamps had 64 valid bits and 10.019 ns period; availability and wrap checks passed. Values are milliseconds; p50/p95/p99 use NumPy linear interpolation. Do not add percentiles from different rows. Per-frame duration samples are retained; raw timestamp ticks are not.

| Interval | p50 | p95 | p99 | Max |
|---|---:|---:|---:|---:|
| Whole chain, control | 64.341 | 65.158 | 65.231 | 65.240 |
| Whole chain, instrumented | 64.272 | 65.200 | 65.253 | 65.255 |
| Census left | 0.112 | 0.116 | 0.117 | 0.117 |
| Census right | 0.114 | 0.119 | 0.121 | 0.121 |
| Left reset | 1.155 | 1.221 | 1.245 | 1.255 |
| Left path 0 | 5.016 | 5.093 | 5.122 | 5.131 |
| Left path 1 | 8.399 | 8.940 | 9.002 | 9.025 |
| Left path 2 | 6.320 | 6.392 | 6.452 | 6.456 |
| Left path 3 | 6.402 | 6.555 | 6.605 | 6.625 |
| Left selection | 3.952 | 4.001 | 4.018 | 4.022 |
| Right reset | 1.069 | 1.171 | 1.212 | 1.227 |
| Right path 0 | 4.599 | 4.685 | 4.728 | 4.736 |
| Right path 1 | 8.165 | 8.930 | 9.052 | 9.100 |
| Right path 2 | 6.706 | 6.779 | 6.809 | 6.810 |
| Right path 3 | 6.351 | 6.437 | 6.491 | 6.494 |
| Right selection | 4.110 | 4.148 | 4.187 | 4.194 |
| LR, both eyes | 0.748 | 0.778 | 0.783 | 0.783 |
| Metric, both eyes | 0.905 | 0.947 | 0.968 | 0.976 |
| CPU record, control | 0.430 | 0.729 | 0.951 | 1.107 |
| CPU submit, control | 0.112 | 0.157 | 0.172 | 0.175 |
| CPU wait, control | 64.580 | 65.334 | 65.394 | 65.419 |
| CPU record, instrumented | 0.458 | 0.744 | 0.888 | 1.007 |
| CPU submit, instrumented | 0.109 | 0.137 | 0.172 | 0.203 |
| CPU wait, instrumented | 64.444 | 65.357 | 65.418 | 65.427 |

CPU wait overlaps GPU work and is not additive. The median group difference does not prove zero timestamp overhead. Parity and timing took the shared GPU lock separately and exited 0. Khronos instance/device validation was enabled and fixture counters checked through teardown; synchronization validation was requested but not independently attested. Loader notices are distinct from fixture diagnostics. The profiler requested 422,338,900 GPU buffer bytes, not measured peak memory.

Before both runs, 335 CPU/identity and independent guard tests passed. Before/after hashes of sources, binaries, SPIR-V, manifest, baseline and prior proof matched. The ignored local `artifacts/fp2-fused-stage-20261029-v1/` directory retains identities, exits, logs and duration samples; the prepared payload remains separately in `artifacts/fp2-720p-cpu-20261028-v1/prepared`. These artifacts may not exist in a fresh checkout.

| Artifact | SHA-256 |
|---|---|
| Profiler host source / binary | `abdfbd8609cd5d61dd636b82d38eeaa28f9f3b0dda37d94b3d4259efbfc02bad` / `4daafb72655a584a7ed1c0496f623e626f12a76bcb72d28720e320433ddfe36b` |
| Profiler harness | `63bb5a09eeb4bba05a929f8bf73e0275fe5ed5d82ef0b8d70c1ae77c306c2408` |
| Fused shader source | `343f35c04a647007eeaa53d00726c66e34aff9e5a669ba13b7ef67828a57014b` |
| Prior fused / profiler parity proof | `513b83b4bffc8fd7df3992c5ea322237417da1a79b0b3d0bc9a7f3d32fbb685d` / `6629d798772f7fe1b81a2e13b9d846a8d4c1e4f387a4b31481df1f90b0c7a154` |
| Timing result / samples | `98948a7bb7e5fd292fa5be4923ce5e9b8fa6507d2c2a9668bac1a6e2c02afe53` / `55660ec40408ad3cc21e0671ed18317fba3e7ef9df3a746ac26c393b9282440e` |
| Parity / timing log | `f6d9ba9fecd5a996299027dc263b243cedeab156645030dc673424fba2ff9bc4` / `5bd0fb00af1040486ddda674f7f15a0b48136fbdcb787cbe4425bb3b2ff888ea` |

## Reproduction

Use the configured Daxa build, validation layer and pinned local artifacts in [fused evidence](fp2-fused-evidence.md). Before GPU work run the complete guards:

```bash
TMPDIR=/dev/shm cmake --build build/daxa --target fp2_fused_stage fp2_fused_tiny -j 4
TMPDIR=/dev/shm PYTHONPATH=python:. .venv/bin/python -m pytest -q \
  tests/test_fp2_fused_stage_cpu.py tests/test_fp2_fused_stage_hotpath.py tests/test_fp2_fused_stage_independent.py \
  tests/test_fp2_stage_timestamps.py tests/test_fp2_fused_cpu.py tests/test_fp2_fused_hotpath.py tests/test_fp2_fused_matched.py \
  tests/test_fp2_parallel_cpu.py tests/test_fp2_parallel_hotpath.py tests/test_fp2_resident_cpu.py tests/test_fp2_resident_hotpath.py \
  tests/test_fp2_integrated_cpu.py tests/test_fp2_integrated_hotpath.py tests/test_fp2_metric_lut_cpu.py \
  tests/test_fp2_selection_cpu.py tests/test_fp2_selection_hotpath.py tests/test_fp2_invalid_index_hotpath.py \
  tests/test_fp2_720p_prepare.py tests/test_fp2_720p_diagnostic.py tests/test_fp2_720p_hotpath.py \
  tests/test_fp2_census_cpu.py tests/test_fp2_census_hotpath.py tests/test_fp2_image_cpu.py tests/test_fp2_image_hotpath.py \
  tests/test_fp2_slots_cpu.py tests/test_fp2_slots_hotpath.py tests/test_fp2_prototype_cpu.py tests/test_fp2_dispatch_hotpath.py
```

In a shell that stops on failure, define the following; use fresh output paths, preserve before/after hashes and logs, and stop if parity or lock acquisition fails:

```bash
common=(--prepared artifacts/fp2-720p-cpu-20261028-v1/prepared
  --manifest-sha256 7f70c357ec371d5c434c435e4a2500d87a149e43a5859e1695ada38cf021ce17
  --host build/daxa/fp2_fused_stage --tiny build/daxa/fp2_fused_tiny
  --artifact-root artifacts/compact-sgm-step-20261028-v2
  --family-root artifacts/fast60-refinement-20261028/analytic-family-720p
  --baseline-result artifacts/fp2-integrated-20261029-v1/outputs/result.json
  --fused-proof artifacts/fp2-fused-20261029-v1/parity/result.json)
run_gpu() {
  timeout 300s flock -n -E 75 /tmp/stereoforge-gpu.lock env -u VK_LAYER_ENABLES TMPDIR=/dev/shm \
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
    VK_LAYER_PATH="$PWD/build/validation-layer/usr/share/vulkan/explicit_layer.d" \
    LD_LIBRARY_PATH="$PWD/build/validation-layer/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    VK_LAYER_VALIDATE_SYNC=1 VK_LOADER_DEBUG=layer PYTHONPATH=python:. \
    .venv/bin/python benchmarks/fp2_fused_stage.py "$@"
}
run_gpu "${common[@]}" --output-root /tmp/fp2-fused-stage-parity-new --parity
proof=/tmp/fp2-fused-stage-parity-new/result.json
proof_sha=$(sha256sum "$proof" | cut -d ' ' -f 1)
run_gpu "${common[@]}" --output-root /tmp/fp2-fused-stage-timing-new \
  --time --proof "$proof" --proof-sha256 "$proof_sha"
```

Timing verifies both proofs and current identities. Inspect the device and validation logs and retain actual exit statuses. Synchronization validation activation must be independently established before claiming it.
