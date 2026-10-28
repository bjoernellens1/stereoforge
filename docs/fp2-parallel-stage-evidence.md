# FP2 parallel stage profiling

The isolated 128-lane parallel shader's proof-bound stage profiler passed synthetic parity and completed a separate timing run on AMD Radeon 8060S / RADV. The 720p raw and observation outputs matched the pinned CPU selections and integrated GPU hashes; the tiny and 320×180 both-eye fixtures compared all four path buffers, sums and padding against the dense CPU oracle. The profiler also checked the prior [parallel parity proof](fp2-parallel-evidence.md). This does not establish live-camera performance, physical depth quality, or equivalence to unrestricted full-resolution SGM.

Timing used 10 balanced warmups (five control, five instrumented), then 25 ABBA blocks with 50 samples per group, on one repeated preuploaded analytic frame with one in-flight slot. Both groups have the same added compute barriers at stage boundaries; the instrumented group adds timestamp annotations. The control therefore is not the original parallel resident timing baseline. No full-frame readback occurred in timed iterations. The 128-lane reduction executes for each path pixel even when fewer candidates are active; it is not work-proportional compact recurrence.

Values are milliseconds. Each stage is an ordered, barrier-inclusive queue interval, including timestamp effects, not a bare kernel duration. p50, p95 and p99 use NumPy linear interpolation; do not add percentiles from separate rows. GPU timestamps used 64 valid bits at 10.019 ns per tick, with availability and wrap checks. Raw per-frame durations are retained; raw ticks are not.

| Interval | p50 | p95 | p99 | Maximum |
|---|---:|---:|---:|---:|
| Whole chain, control | 91.711 | 94.209 | 94.881 | 94.984 |
| Whole chain, instrumented | 91.587 | 94.601 | 95.863 | 96.265 |
| Census left | 0.115 | 0.118 | 0.120 | 0.120 |
| Census right | 0.115 | 0.119 | 0.121 | 0.121 |
| Left reset | 2.171 | 2.278 | 2.372 | 2.391 |
| Left cost | 9.270 | 9.405 | 9.427 | 9.436 |
| Left path 0 | 6.265 | 6.351 | 6.398 | 6.435 |
| Left path 1 | 8.861 | 9.307 | 9.429 | 9.434 |
| Left path 2 | 7.197 | 7.250 | 7.318 | 7.359 |
| Left path 3 | 7.214 | 8.012 | 9.060 | 9.444 |
| Left selection | 4.266 | 4.957 | 6.117 | 6.426 |
| Right reset | 2.018 | 2.115 | 2.774 | 3.386 |
| Right cost | 9.080 | 9.309 | 9.659 | 9.973 |
| Right path 0 | 6.262 | 6.373 | 6.460 | 6.522 |
| Right path 1 | 8.636 | 9.177 | 9.233 | 9.250 |
| Right path 2 | 7.383 | 7.511 | 7.562 | 7.566 |
| Right path 3 | 7.437 | 7.510 | 7.837 | 8.133 |
| Right selection | 4.466 | 4.562 | 4.808 | 5.005 |
| LR, both eyes | 0.728 | 0.749 | 0.900 | 0.901 |
| Metric, both eyes | 0.885 | 0.902 | 1.085 | 1.087 |
| CPU record, control | 0.709 | 0.937 | 0.959 | 0.961 |
| CPU submit, control | 0.157 | 0.180 | 0.184 | 0.187 |
| CPU wait, control | 91.968 | 94.601 | 95.523 | 95.942 |
| CPU record, instrumented | 0.749 | 0.907 | 0.995 | 1.047 |
| CPU submit, instrumented | 0.161 | 0.188 | 0.218 | 0.229 |
| CPU wait, instrumented | 91.922 | 95.077 | 96.451 | 97.143 |

CPU wait overlaps GPU work and is not additive. The median control/instrumented difference does not establish zero annotation overhead. Parity and timing each acquired the nonblocking shared GPU lock and exited 0. Khronos instance/device validation was enabled and fixture counters checked through teardown; synchronization validation was requested but not independently attested. Loader notices are distinct from fixture diagnostics. The 720p host requested 656,178,404 buffer bytes, not measured peak memory. Source ID was 7.

Before both runs, 302 CPU/identity and guard tests passed. Source, binary, SPIR-V, prepared manifest, baseline and prior proof hashes matched before and after. The ignored local `artifacts/fp2-parallel-stage-20261029-v1/` directory preserves logs, exit statuses, identities and samples; it may be absent in a fresh checkout.

| Artifact | SHA-256 |
|---|---|
| Profiler host source / binary | `69eebf98d4c56b0f98a5cdab7f8c2106883178b502c179a01b7e7ee18a7a5519` / `404bda630932e24896f9129a1a7d4a58db8f5347b8e17f45264aea531d363fd2` |
| Profiler harness | `19358cd9cfd2888e48babf46d6ddd9fc012fdac5c7a9e5d3adacb6a4d847a29d` |
| Parallel shader source | `d510bab2ad4f3590f905bd88e1575efd100788e43ba658a6f7aed2379b52b641` |
| Prior parallel / profiler parity proofs | `ce5610937f578f7e7a15c302374bb5efa69687f041739f282f0b63cfe5f07c18` / `35c9ea18c326b9807cbb0656d2676d79f73947717315867113c1b594f638aa1b` |
| Timing result / raw samples | `31a5fe3fb1816d1b7430b638c7adb5b8ed3a8f37ba6ccd3503ee1f724e084138` / `a3ffbcf99a2881d58fd3f27c71ad14d46c769379b14957d78d5c6931ed6b86f2` |
| Parity / timing log | `b08262ac50545f2aa6a3e9c4cfc498683ac2a38cc48f82caf719f07e938d728e` / `09d347fb42b140a1fd3c0c245ef5505f06b6ac461f594be82f78c0de925bf481` |

## Reproduction

Use the configured build, local pinned artifacts and validation layer from [parallel evidence](fp2-parallel-evidence.md). Build and run all guard suites before GPU execution. If `/tmp` quota is exhausted, use an existing workspace directory for `TMPDIR` and a fresh workspace pytest base directory.

```bash
cmake --build build/daxa --target fp2_parallel_stage fp2_parallel_tiny -j 4
PYTHONPATH=python:. .venv/bin/python -m pytest -q \
  tests/test_fp2_parallel_stage_cpu.py tests/test_fp2_parallel_stage_hotpath.py \
  tests/test_fp2_stage_timestamps.py tests/test_fp2_parallel_cpu.py tests/test_fp2_parallel_hotpath.py \
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

Define the following in a shell that stops on failure; both outputs must be fresh. Record pre/post identities, logs and exits and inspect device/validation diagnostics. The lock is nonblocking and exit 75 means it was unavailable.

```bash
common=(--prepared artifacts/fp2-720p-cpu-20261028-v1/prepared
  --manifest-sha256 7f70c357ec371d5c434c435e4a2500d87a149e43a5859e1695ada38cf021ce17
  --host build/daxa/fp2_parallel_stage --tiny build/daxa/fp2_parallel_tiny
  --artifact-root artifacts/compact-sgm-step-20261028-v2
  --family-root artifacts/fast60-refinement-20261028/analytic-family-720p
  --baseline-result artifacts/fp2-integrated-20261029-v1/outputs/result.json
  --parallel-proof artifacts/fp2-parallel-20261029-v1/parity/result.json)
run_gpu() {
  timeout 300s flock -n -E 75 /tmp/stereoforge-gpu.lock env -u VK_LAYER_ENABLES \
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
    VK_LAYER_PATH="$PWD/build/validation-layer/usr/share/vulkan/explicit_layer.d" \
    LD_LIBRARY_PATH="$PWD/build/validation-layer/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    VK_LAYER_VALIDATE_SYNC=1 VK_LOADER_DEBUG=layer PYTHONPATH=python:. \
    .venv/bin/python benchmarks/fp2_parallel_stage.py "$@"
}
run_gpu "${common[@]}" --output-root /tmp/fp2-parallel-stage-parity-new --parity
proof=/tmp/fp2-parallel-stage-parity-new/result.json
proof_sha=$(sha256sum "$proof" | cut -d ' ' -f 1)
run_gpu "${common[@]}" --output-root /tmp/fp2-parallel-stage-timing-new \
  --time --proof "$proof" --proof-sha256 "$proof_sha"
```

Timing verifies the profiler parity proof and upstream parallel proof against current identities. Do not run it after failed parity. Sync activation must be independently established before claiming synchronization validation.
