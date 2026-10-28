# Integrated FP2 720p analytic parity

On 2026-10-28 at 23:06:28–23:06:33 UTC, the isolated 1280×720 compact-band diagnostic exited 0 on AMD Radeon 8060S / RADV Strix Halo. It processed a pinned analytic rectified-luma stereo pair using one GPU command chain: support-aware Census, independent left and right cost/SGM, packed selection, LR consistency and metric lookup. It reset the entire reusable packed cost, sum and path workspace before each eye, with explicit stage barriers, and retained both eyes' selected outputs until final completion. There was no host wait or read between stages; full raw and observation readback occurred afterward for this debug test.

The actual GPU outputs matched the pinned compact CPU runner's winner, best, second and count for both eyes. Raw disparity, validity and confidence, LR validity, metric depth and validity, provenance, source ID and age were compared with CPU-derived references and the pinned five planes. Finite depths and raw outputs were compared by binary32 bits; invalid metric depth had to use canonical NaN while the pinned invalid reference had to be NaN. The run used source ID **7**; an ID above 32 bits was separately covered by a small CPU comparator test, not by this 720p GPU run.

| GPU output count | Left | Right |
|---|---:|---:|
| Raw valid | 897,952 | 898,067 |
| LR valid | 810,398 | 813,291 |
| Metric valid | 810,398 | 813,291 |

The fixture requested 1,591,535,844 bytes of GPU buffers; this is not measured peak memory and excludes driver allocation overhead and CPU-side data. Per-eye path and sum buffers are overwritten on reuse and were not independently read back by this integrated run. Earlier [per-eye 720p evidence](fp2-720p-evidence.md) compared those intermediate buffers against the packed CPU oracle under its own scoped fixture.

The LUT is a per-invocation diagnostic resource for integer disparities, keyed by a pre-rounded binary32 numerator and offset. For this input, the comparator constructs `refinement.Geometry`, which normalizes its geometry values to binary32 before the numerator product; the fixture consumes that pre-rounded product. The test does not establish a general original-precision calibration conversion or production LUT lifetime. See [selection evidence](fp2-selection-evidence.md) for the LUT and confidence precision limits.

These are analytic inputs and pinned CPU references. The result does not establish live USB camera operation, native production-graph or subpixel equivalence, full-D behavior, independent physical depth quality, latency, throughput or sustained 60 Hz. The wall time includes validation, reference checks and debug readback and is not a performance measurement.

The pre-run CPU and guard suite passed 206 tests; the expanded reproduction suite below passed 218 tests when rechecked. The loader confirmed Khronos instance and device validation layers; fixture validation counters were zero through teardown. Loader notices about environment-selected layers are separate. Synchronization validation was requested with `VK_LAYER_VALIDATE_SYNC=1`, but activation was not independently attested. The run used a nonblocking shared GPU lock and 300-second watchdog.

## Artifacts and identities

Before/after hashes matched. The ignored local directory `artifacts/fp2-integrated-20261029-v1/` contains the source/SPIR-V/compiler/binary identity lists, timestamps, exit status, log and output files; the pinned CPU preparation is under `artifacts/fp2-720p-cpu-20261028-v1/prepared/`. These artifacts may not exist on another checkout.

| Artifact | SHA-256 |
|---|---|
| Integrated host binary | `4532b7a759026ab0366891456b3194a684b93a0eaa4b8753263290768198e6b3` |
| Integrated host source | `bb911a87e0b0b34a31f22a5854c3967abbce10d9868963a8edb128a3d84f1ffd` |
| Harness | `2bcc65f52f07ba47a973ab561bfbba2581046617a684494d4948e8a196227716` |
| LUT helper | `570aac9fba5f28052d707c09726e0910515f289206dc3a4a781b90f0a86967ed` |
| Cost-reset shader / SPIR-V | `e8c0f23a7bc4df847cf7edfdee3037a2b02a7d097a792f280e2b19cd36dfb709` / `1cb971df981dc973d9d5c7c070125b00b0161688c7d729d5d5608e2b8e0e4886` |
| Census shader / helper / SPIR-V | `7026a1f3eb3b4a446cf2b8c00cb83ac77641c2e951bbbad0c7d50737991a2944` / `dc4e735bfd0300d54bec71471403e3a0fb2742e9375f2e39053811d4db878c2b` / `682490d0483993e17c676e173fec7a649201f5a0f3ee45d92a81dd2ee388d14a` |
| Cost shader / SPIR-V | `537a9517d26ec2e9b3f86c1866fc05a28e9b33c7a8f08e9f2468a5f2510c7bcd` / `fd2ae15ecdf9e0dba425654a88212c428593e382627b733e66a083edb63d0bb6` |
| Recurrence shader / SPIR-V | `169df335ccb52dd6de711f8c1980ec88d8ee0fe2684aed8af1321f8775506ea9` / `df44163e38d1fac96e4738f482bdffe70fa72f3cc089df5e792a2ef099af45ca` |
| Selection shader | `3a7dc960ee5b90a2884cd38993c1606c5744125123b1b18a949a01d331762c72` |
| Selection reset / select / LR / metric SPIR-V | `6f9d062a402e013b62836989daff0fce68945323896248c00b1d0e83642a07aa` / `4c52146bf76df170466958f5745c086972296d85c0f6af0ff19405dc0d07a94f` / `d32fa9cb3143a9d32c6b27fbd50e414cd08b70f660497ece00954c7306c1c751` / `bf420ad2ca69d19d6f8cd603fc5f7e760bf5452226b50cbb3c9240f540acb8fb` |
| Slang compiler | `82e29ecd96714c64a31021160eaff4bf8604f44934e9f7b98448e962a5d12dad` |
| Pinned prepared manifest | `7f70c357ec371d5c434c435e4a2500d87a149e43a5859e1695ada38cf021ce17` |
| Raw / observation output | `56cadc581a831da2f46853e741d1633d868fc3ed8070693deae08baf40d15d54` / `7032c8c610e205349caedaf7e65f27fcd391b7595cb97aa8e7aadc3438e7f4f4` |
| Result / log | `e690b1895b63cf2c9c158419bf0662e7e316426d1d3c164f8c078d3700090041` / `8f13a3d325aa255cc23f14630d4140825baa3af51138479499edfcb90e29a683` |

## Reproduction

Use the configured Daxa build and local validation layer. The CPU identity test verifies current sources, helper, SPIR-V and compiler against the host. The harness verifies the pinned manifest and prepared files before and after the run. Use a fresh output directory and separately capture exit, log, timestamps and before/after hashes; lock failure exits 75.

```bash
cmake --build build/daxa --target fp2_integrated_fixture -j 4
PYTHONPATH=python:. .venv/bin/python -m pytest -q \
  tests/test_fp2_integrated_cpu.py tests/test_fp2_integrated_hotpath.py \
  tests/test_fp2_metric_lut_cpu.py tests/test_fp2_selection_cpu.py tests/test_fp2_selection_hotpath.py \
  tests/test_fp2_invalid_index_hotpath.py \
  tests/test_fp2_720p_prepare.py tests/test_fp2_720p_diagnostic.py tests/test_fp2_720p_hotpath.py \
  tests/test_fp2_census_cpu.py tests/test_fp2_census_hotpath.py \
  tests/test_fp2_image_cpu.py tests/test_fp2_image_hotpath.py \
  tests/test_fp2_slots_cpu.py tests/test_fp2_slots_hotpath.py \
  tests/test_fp2_prototype_cpu.py tests/test_fp2_dispatch_hotpath.py
timeout 300s flock -n -E 75 /tmp/stereoforge-gpu.lock \
  env -u VK_LAYER_ENABLES \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
  VK_LAYER_PATH="$PWD/build/validation-layer/usr/share/vulkan/explicit_layer.d" \
  LD_LIBRARY_PATH="$PWD/build/validation-layer/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  VK_LAYER_VALIDATE_SYNC=1 VK_LOADER_DEBUG=layer PYTHONPATH=python:. \
  .venv/bin/python benchmarks/fp2_integrated_diagnostic.py \
  --prepared artifacts/fp2-720p-cpu-20261028-v1/prepared \
  --manifest-sha256 7f70c357ec371d5c434c435e4a2500d87a149e43a5859e1695ada38cf021ce17 \
  --host build/daxa/fp2_integrated_fixture \
  --artifact-root artifacts/compact-sgm-step-20261028-v2 \
  --family-root artifacts/fast60-refinement-20261028/analytic-family-720p \
  --output-root /tmp/fp2-integrated-new-output --gpu
```
