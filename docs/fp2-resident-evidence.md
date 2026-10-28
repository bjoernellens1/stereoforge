# FP2 no-path resident parity and timing

On 2026-10-28, the isolated no-path FP2 lane passed its parity gate on AMD Radeon 8060S / RADV Strix Halo. The 7×5 and 320×180 synthetic fixtures compared each eye's complete packed sums, including INF padding, with an independent scalar recurrence. The 1280×720 analytic compact-band run compared actual GPU raw and observation outputs with pinned CPU selections and the previous integrated diagnostic output hashes. The 720p GPU counts were 897,952 / 898,067 raw valid and 810,398 / 813,291 LR and metric valid, left / right. The parity proof is `artifacts/fp2-resident-20261029-v1/parity/result.json`, SHA-256 `91a53952841c9eeab9ba03a8ec75b5b2c73b582ba26ca1d1a569e35565796bc0`.

After parity, a separate locked run measured 10 warmups and 100 samples of the same preuploaded analytic frame with one in-flight slot. Pipelines and GPU resources remained allocated; a fresh Daxa command list was recorded for each submission because Daxa uses one-time-submit command buffers. The fixture waited for completion before reusing resources and did not perform a full per-frame host readback. Values below are milliseconds; percentiles use NumPy linear interpolation.

| Measure | p50 | p95 | p99 | Maximum |
|---|---:|---:|---:|---:|
| GPU whole-chain timestamps | 839.96 | 854.86 | 858.55 | 859.91 |
| CPU command recording | 0.732 | 0.916 | 1.005 | 1.543 |
| CPU submit | 0.151 | 0.192 | 0.216 | 0.227 |
| CPU completion wait | 840.45 | 855.48 | 859.40 | 860.81 |

**Performance decision: reject this prototype as a 60 Hz candidate.** The whole-chain GPU timing includes the isolated serial-line recurrence and resets; it is far above the 16.67 ms target. It does not identify which stage dominates. This is not a matched old/new performance comparison, live-camera throughput, transfer-inclusive latency, production graph, independent depth quality or 60 Hz result; no FPS claim is made. CPU wait overlaps the GPU execution interval and must not be added to it. Host input staging is outside the measured loop. No stage-specific breakdown was collected.

The no-path 720p host requested 656,178,404 GPU buffer bytes; this excludes driver overhead and CPU reference storage and is not measured peak memory. Diagnostic path storage was removed; no claim about speedup follows without matched baseline measurements. The test keeps the per-invocation integer-disparity metric LUT and the same identity/provenance limits as the [integrated evidence](fp2-integrated-evidence.md).

Both parity and timing acquired the nonblocking shared `/tmp/stereoforge-gpu.lock` in separate runs and exited 0. Before/after source, binary, SPIR-V and input hashes matched, and 230 CPU and guard tests passed beforehand. Khronos instance/device validation was enabled during both runs; fixture counters were zero through teardown. Loader environment-layer notices are separate from fixture diagnostics. Synchronization validation was requested but not independently attested. The main-queue timestamp valid width was 64 bits, timestamp period 10.019 ns; availability and wrap ambiguity were checked. The raw samples and linear-percentile summary are preserved in the ignored local directory `artifacts/fp2-resident-20261029-v1/`, which may not exist in a fresh checkout.

| Artifact | SHA-256 |
|---|---|
| Resident host binary | `0c09abee070e6eebbbe898b8672c466cb02738acbc8895af3e14198eacb88573` |
| Tiny fixture binary | `d9330cf2c23a961f93ae454e5bdb7cbb0dca52f9e38d5e76e6cae44868152007` |
| Resident host source | `90b9cb78482d296ccf13d5f8fb075a7a0b50f9b11509e4220a74974451672e4d` |
| Tiny fixture source | `41e943c8efd96d667af9881a94e44dfea37b00f353508439172c3f0cd0d21f27` |
| Harness | `313c5e72686cfc4ea84c6561c760b0ca73ecf44b67aa9562135f8d9691283c6e` |
| No-path shader | `90a5ba43ae1895ed4fd02e2e0ee7f31e5c3f3d1d4c6358303a9baec1a941ad2d` |
| Sum reset shader | `c307743ac1103b0b4a461c1ea8078f42519cccc94787b31aced76e7542442fc9` |
| Parity log / proof | `b08262ac50545f2aa6a3e9c4cfc498683ac2a38cc48f82caf719f07e938d728e` / `91a53952841c9eeab9ba03a8ec75b5b2c73b582ba26ca1d1a569e35565796bc0` |
| Timing log / result | `77ca3ba78a7021d84ff89013b61a0f945cf62c6ccb36b08e4de0bab84d7e57d1` / `b749da82b0b48805a0eb3308a37ecd782d9f0bb4d61ff96e9b75e6ce4a0d6bd6` |
| Raw samples | `fc65ab2b4118d6895c93852360f3d86a683cda1028baccf31b076492e7e6aa64` |

The complete before/after lists and exact timing inputs are retained beside these artifacts. The timing harness requires a matching parity proof and rejects changed source, binaries, prepared manifest or output identities.

## Reproduction

The following requires the retained local artifacts and configured Daxa and validation-layer builds. Use a fresh output root; an existing root is rejected. The test suite and both executable identities must pass before either GPU run. Each invocation takes the shared lock independently; stop if parity or lock admission fails. This reproduces the procedure, not the exact timing samples.

```bash
cmake --build build/daxa --target fp2_resident_fixture fp2_resident_tiny -j 4
PYTHONPATH=python:. .venv/bin/python -m pytest -q \
  tests/test_fp2_resident_cpu.py tests/test_fp2_resident_hotpath.py \
  tests/test_fp2_integrated_cpu.py tests/test_fp2_integrated_hotpath.py \
  tests/test_fp2_metric_lut_cpu.py tests/test_fp2_selection_cpu.py tests/test_fp2_selection_hotpath.py \
  tests/test_fp2_invalid_index_hotpath.py \
  tests/test_fp2_720p_prepare.py tests/test_fp2_720p_diagnostic.py tests/test_fp2_720p_hotpath.py \
  tests/test_fp2_census_cpu.py tests/test_fp2_census_hotpath.py \
  tests/test_fp2_image_cpu.py tests/test_fp2_image_hotpath.py \
  tests/test_fp2_slots_cpu.py tests/test_fp2_slots_hotpath.py \
  tests/test_fp2_prototype_cpu.py tests/test_fp2_dispatch_hotpath.py

baseline=artifacts/fp2-integrated-20261029-v1/outputs/result.json
printf '%s  %s\n' e690b1895b63cf2c9c158419bf0662e7e316426d1d3c164f8c078d3700090041 "$baseline" | sha256sum -c
common=(--prepared artifacts/fp2-720p-cpu-20261028-v1/prepared
  --manifest-sha256 7f70c357ec371d5c434c435e4a2500d87a149e43a5859e1695ada38cf021ce17
  --host build/daxa/fp2_resident_fixture --tiny build/daxa/fp2_resident_tiny
  --artifact-root artifacts/compact-sgm-step-20261028-v2
  --family-root artifacts/fast60-refinement-20261028/analytic-family-720p
  --baseline-result "$baseline")
run_gpu() {
  timeout 900s flock -n -E 75 /tmp/stereoforge-gpu.lock \
    env -u VK_LAYER_ENABLES \
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
    VK_LAYER_PATH="$PWD/build/validation-layer/usr/share/vulkan/explicit_layer.d" \
    LD_LIBRARY_PATH="$PWD/build/validation-layer/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    VK_LAYER_VALIDATE_SYNC=1 VK_LOADER_DEBUG=layer PYTHONPATH=python:. \
    .venv/bin/python benchmarks/fp2_resident_diagnostic.py "$@"
}
run_gpu "${common[@]}" --output-root /tmp/fp2-resident-parity-new --parity
proof=/tmp/fp2-resident-parity-new/result.json
proof_sha=$(sha256sum "$proof" | cut -d ' ' -f 1)
run_gpu "${common[@]}" --output-root /tmp/fp2-resident-timing-new \
  --time --proof "$proof" --proof-sha256 "$proof_sha"
sha256sum "$proof" /tmp/fp2-resident-timing-new/result.json \
  /tmp/fp2-resident-timing-new/samples.csv
```

Run in a shell that stops on failure (for example, `set -e`), preserve logs and exit statuses, and record source, binary, SPIR-V and input hashes before and after. Check the selected AMD device and validation diagnostics; requesting synchronization validation alone does not establish that it was active. A freshly generated proof hash will differ if its bound identities differ; the harness checks the proof against current inputs and binaries.
