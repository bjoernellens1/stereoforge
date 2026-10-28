# FP2 selection, consistency and metric LUT evidence

On 2026-10-28, the isolated FP2 experiment passed its synthetic-cost tests on AMD Radeon 8060S / RADV Strix Halo. At 7×5 and 320×180, two independently processed eye inputs share one packed recurrence workspace, retain separate selection results, then execute LR consistency and metric observation. Explicit compute barriers separate stages; no host wait or read occurs until the chain completes. The fixture then performs full debug readback and compares raw selection and observation words exactly with CPU references. Both sizes passed. This is not a 720p, image-derived Census, live-camera, subpixel, production pipeline, quality or performance result.

| Synthetic fixture | CPU selected | CPU LR-valid | Requested GPU buffer bytes |
|---|---:|---:|---:|
| 7×5 | 52 | 25 | 52,908 |
| 320×180 | 78,313 | 1,325 | 96,737,268 |

The selected and LR-valid counts are preliminary CPU-reference counts across both eyes using an all-one support mask; they are not per-eye GPU counts. Requested bytes exclude driver allocation overhead and CPU reference storage. The explicit GPU witness includes both lower and upper restricted endpoints, tied winners, empty/single-candidate and zero-disparity rejection, LR differences of one and two, and unsupported metric pixels; it asserts nonzero valid metric observations in each eye. It checks confidence, canonical invalid NaN, validity, provenance, 64-bit source ID and zero observation age. Contributor ancestry is not emitted by this fixture.

## Failures and correction

- **v1:** The first normal GPU run exited 1 on a raw witness mismatch. It did not retain the GPU witness buffer, so its specific fields cannot be established from that run.
- **v2:** An instrumented capture exited 1 and retained both raw buffers. Twelve differences occurred only in confidence: the 2/3 witness was `0x3f2aaaab` on CPU and `0x3f2aaaaa` on GPU, one ULP lower. Winner, best, second, count, disparity and validity matched. Vulkan permits bounded `OpFDiv` error; the captured result does not establish the driver's internal lowering.
- **v3:** After the bounded integer nearest-even confidence calculation was introduced, aggregate 225 was rejected with the expected error and exit 1. The normal run passed the raw comparison but exited 1 on observation mismatch; the differing field was not captured in that run.
- **v4:** Instrumented observation capture exited 1. Two depth values differed by one ULP: CPU `0x42104925` versus GPU `0x42104924` for pre-rounded numerator 63.125 and denominator 1.75. Every other observation field matched.
- **v5:** The out-of-range aggregate and out-of-range LUT index each exited 1 with their distinct expected errors. The normal witness and both synthetic recurrence fixtures then passed with exit 0. The metric LUT removes shader division from this scoped path; the generated metric SPIR-V was checked for the absence of `OpFDiv`.

Confidence uses exact integer nearest-even conversion under the explicitly enforced finite aggregate bound of 224: this fixture's four paths each have cost at most 24 plus P2 at most 32. An unexpected finite aggregate above the bound sets a GPU error flag and fails the host check. All 25,200 valid bounded `(best, second)` pairs were tested against NumPy and independent rational nearest-even checks; out-of-range inputs fail closed.

The depth lookup is deliberately narrower: its inputs are an **already rounded binary32 numerator** and binary32 principal offset, and it covers integer disparities 0–127 only. It does not define how arbitrary original-precision focal length and baseline are multiplied; a CPU test demonstrates that the original-precision and separately rounded-input routes can differ. The LUT builder checks round-to-nearest mode, rejects x86 DAZ/FTZ, probes subnormal arithmetic and fails closed on unsupported architectures or input modes. Tests toggle and restore the x86 flags in a subprocess. The fixture creates and uploads the LUT for each invocation, verifies its contents and diagnostic device/configuration identity, and retains it through that invocation. This is not a demonstrated production calibration cache or cross-frame ownership implementation.

## Validation and identities

The v5 CPU and guard suite passed 177 tests; the expanded reproduction suite below passed 189 tests when rechecked. Khronos instance and device validation layers were present in the loader logs; the successful normal fixture reported no validation diagnostics through teardown. The score-error path throws before the fixture's explicit context close and counter assertion; its log does not establish that post-teardown counter check. The loader printed environment-layer notices separately. Synchronization validation was requested with `VK_LAYER_VALIDATE_SYNC=1`, but activation was not independently attested. Each GPU invocation acquired the nonblocking shared lock and had a 60-second watchdog.

The v5 source, binaries and SPIR-V before/after hashes matched:

| Artifact | SHA-256 |
|---|---|
| Fixture binary | `7874dc16886406f87bb4777d6c2e581b791f0efe0abdb123c290b5bdc0a216f2` |
| Fixture source | `73511f0e825797dc1d5e1182cc52f83123581e0fcbf458556b98c3187d6b18f7` |
| LUT host helper | `570aac9fba5f28052d707c09726e0910515f289206dc3a4a781b90f0a86967ed` |
| Included FP2 host | `e3040970f8216eeaed21a75ceef3ba8ee38317562b83ffc91cb675dd9c2b65cf` |
| Selection shader | `3a7dc960ee5b90a2884cd38993c1606c5744125123b1b18a949a01d331762c72` |
| Recurrence shader | `169df335ccb52dd6de711f8c1980ec88d8ee0fe2684aed8af1321f8775506ea9` |
| Reset SPIR-V | `6f9d062a402e013b62836989daff0fce68945323896248c00b1d0e83642a07aa` |
| Selection SPIR-V | `4c52146bf76df170466958f5745c086972296d85c0f6af0ff19405dc0d07a94f` |
| Consistency SPIR-V | `d32fa9cb3143a9d32c6b27fbd50e414cd08b70f660497ece00954c7306c1c751` |
| Metric SPIR-V | `bf420ad2ca69d19d6f8cd603fc5f7e760bf5452226b50cbb3c9240f540acb8fb` |
| Recurrence SPIR-V | `df44163e38d1fac96e4738f482bdffe70fa72f3cc089df5e792a2ef099af45ca` |
| Slang compiler | `82e29ecd96714c64a31021160eaff4bf8604f44934e9f7b98448e962a5d12dad` |

The ignored local `artifacts/fp2-selection-20261028-v1/` through `v5/` directories preserve logs and hashes; v2 and v4 also retain the captured raw and observation witness buffers. These artifacts may not exist in a fresh checkout. The v5 score, index and normal log hashes are respectively `544c0565fe670dbe6c0cf28d50fcda1a9a372c793c6e8ba8c1005d46d31f9984`, `ecf30568cf64e0044a78f6d57a51d6bcae5024ae9744fe3e2d8d0f99c5894c96`, and `8e4fa6ba15495b812acce128575a7a388d5f277ca646a68f3a45ebd0f27756ec`.

## Reproduction

Use the configured Daxa build and local validation layer. Capture new outputs, timestamps, exit codes and pre/post identities in a fresh directory. Verify the two negative tests each exit 1 with the exact expected message before running the normal test; a lock failure exits 75. The CPU identity test verifies the current sources, helper, compiler and SPIR-V against the binary.

```bash
cmake --build build/daxa --target fp2_selection_fixture -j 4
PYTHONPATH=python:. .venv/bin/python -m pytest -q \
  tests/test_fp2_metric_lut_cpu.py tests/test_fp2_selection_cpu.py tests/test_fp2_selection_hotpath.py \
  tests/test_fp2_invalid_index_hotpath.py \
  tests/test_fp2_720p_prepare.py tests/test_fp2_720p_diagnostic.py tests/test_fp2_720p_hotpath.py \
  tests/test_fp2_census_cpu.py tests/test_fp2_census_hotpath.py \
  tests/test_fp2_image_cpu.py tests/test_fp2_image_hotpath.py \
  tests/test_fp2_slots_cpu.py tests/test_fp2_slots_hotpath.py \
  tests/test_fp2_prototype_cpu.py tests/test_fp2_dispatch_hotpath.py
./build/daxa/fp2_selection_fixture --identity
```

For each of `--gpu-overflow`, `--gpu-invalid-index`, then `--gpu`, invoke:

```bash
timeout 60s flock -n -E 75 /tmp/stereoforge-gpu.lock \
  env -u VK_LAYER_ENABLES \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
  VK_LAYER_PATH="$PWD/build/validation-layer/usr/share/vulkan/explicit_layer.d" \
  LD_LIBRARY_PATH="$PWD/build/validation-layer/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  VK_LAYER_VALIDATE_SYNC=1 VK_LOADER_DEBUG=layer \
  ./build/daxa/fp2_selection_fixture MODE
```

Replace `MODE` with one option at a time. Expected negative messages are `witness aggregate outside score contract` and `metric LUT index outside contract`, respectively; the normal mode should exit 0. Use a fresh private `SF_FP2_WITNESS_DIR` when capturing mismatch buffers. Do not proceed to the normal mode if either negative case fails for a different reason or validation reports an error.
