# FP2 720p analytic compact parity

On 2026-10-28, the isolated compact-band Census → cost → four-path FP2 diagnostic exited 0 on AMD Radeon 8060S / RADV Strix Halo. The input is the retained 1280×720 analytic rectified-luma fixture and its pinned `compact-fixed` disparity intervals. Eye 0 and eye 1 ran sequentially. This is an output-parity test, not a live-camera, full-D, production-native pipeline, independent ground-truth quality, sustained throughput or performance result.

For each eye, the GPU computes support-aware 5×5 Census for both images, packed Census costs and four directional path accumulations in a single chain with explicit compute barriers. No host wait or read occurs between these stages. Once the chain finishes, debug readback compares the two descriptor and validity planes, every active cost and INF/padding slot, and all four packed paths and sums against CPU references. CPU winner, best, second and count are also compared with a reduction of the GPU sums. All comparisons passed. The 720p recurrence reference is the pinned compact CPU runner, not an independently structured dense 720p oracle; separate small fixtures test the independent dense oracle.

The five pinned output planes—raw disparity and validity, LR validity, metric validity and depth—were reconstructed and checked on the CPU from the pinned CPU winner planes. Their connection to this GPU run is the exact GPU-sum-derived winner/best/second/count comparison above; LR consistency and metric depth were not executed on the GPU in this test. Float comparison accepts matching NaN positions.

| Per-eye diagnostic | Eye 0 | Eye 1 |
|---|---:|---:|
| Supported Census pixels per input | 913,616 | 913,616 |
| Valid packed matching costs | 47,579,821 | 47,399,933 |
| Packed bucket words | 58,459,840 | 58,150,496 |
| Requested GPU buffer bytes | 1,451,410,208 | 1,443,985,952 |

Requested buffer figures exclude driver allocation overhead and CPU oracle storage; peak device and process memory were not measured. The GPU allocations are admitted only when the device heap budget, maximum allocation, host free memory and fixture caps pass; the two eyes do not coexist on the GPU. Packed CPU oracle path and sum files are compared in chunks of at most 8 MiB, and full readback is solely a debug-verification boundary. Each CPU oracle subprocess used a 2 GB address-space cap. The CPU preparation completed 22:34:10–22:34:17 UTC and the locked GPU diagnostic completed 22:38:01–22:38:53 UTC; these wall-clock intervals include preparation, validation, comparisons and readback and are not performance measurements.

Before the run, 107 CPU identity and hotpath tests passed. The loader confirmed Khronos validation at instance and device levels; fixture validation counters were zero through both teardowns. Loader notices about environment-selected layers are distinct from fixture warnings. Synchronization validation was requested with `VK_LAYER_VALIDATE_SYNC=1`, but activation was not independently attested. The GPU lock was nonblocking, with a 300-second watchdog.

## Recorded identities

Before/after identity lists matched. The ignored local directories `artifacts/fp2-720p-cpu-20261028-v1/` and `artifacts/fp2-720p-gpu-20261028-v1/` retain CPU oracle files, manifest, hashes, timestamps, exit status, validation log and result. They may be absent on a fresh checkout. The prepared manifest identifies its source, frozen intervals and all oracle files; SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| Prepared manifest | `7f70c357ec371d5c434c435e4a2500d87a149e43a5859e1695ada38cf021ce17` |
| Diagnostic result | `564aea3a9b60532e013f11074cc78ddc197d848755be5fc429e5c04121a5871c` |
| GPU log | `bbc1a1be14ff512757f21e7ffdfa4c83071e43a209524962ef3c55e0a0c0189e` |
| Native host binary | `2a080cd5cb1118a7bc322bc343c502c406ea1732d13165279c0fc123a1f6a4db` |
| Native host source | `a1d2b82b15abd9e95496453ab01a7ddc6503132580214e756b87a8daf47390c1` |
| Included recurrence host | `e3040970f8216eeaed21a75ceef3ba8ee38317562b83ffc91cb675dd9c2b65cf` |
| Diagnostic harness | `694615abfd3fbd6bf841f6e9c1b36dfd9b1e48740361b78f470f4cafb1c7e475` |
| Preparation harness | `5f9a228f8580a107122694cacbd57f8745a167b621332aa102cc651d2085211a` |
| Census shader | `7026a1f3eb3b4a446cf2b8c00cb83ac77641c2e951bbbad0c7d50737991a2944` |
| Census included helper | `dc4e735bfd0300d54bec71471403e3a0fb2742e9375f2e39053811d4db878c2b` |
| Cost shader | `537a9517d26ec2e9b3f86c1866fc05a28e9b33c7a8f08e9f2468a5f2510c7bcd` |
| Recurrence shader | `169df335ccb52dd6de711f8c1980ec88d8ee0fe2684aed8af1321f8775506ea9` |
| Census SPIR-V | `682490d0483993e17c676e173fec7a649201f5a0f3ee45d92a81dd2ee388d14a` |
| Cost SPIR-V | `fd2ae15ecdf9e0dba425654a88212c428593e382627b733e66a083edb63d0bb6` |
| Recurrence SPIR-V | `df44163e38d1fac96e4738f482bdffe70fa72f3cc089df5e792a2ef099af45ca` |
| Slang compiler | `82e29ecd96714c64a31021160eaff4bf8604f44934e9f7b98448e962a5d12dad` |

## Reproduction

The commands below use the retained prepared manifest and local validation-layer paths. If inputs are regenerated, use the preparation harness and pin the new manifest hash; do not overwrite the original artifacts. A busy GPU lock exits 75. Capture a fresh run's output, exit, timestamps and before/after identities separately.

```bash
cmake --build build/daxa --target fp2_720p_fixture -j 4
PYTHONPATH=python:. .venv/bin/python -m pytest -q \
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
  .venv/bin/python benchmarks/fp2_720p_diagnostic.py \
  --prepared artifacts/fp2-720p-cpu-20261028-v1/prepared \
  --manifest-sha256 7f70c357ec371d5c434c435e4a2500d87a149e43a5859e1695ada38cf021ce17 \
  --host build/daxa/fp2_720p_fixture \
  --artifact-root artifacts/compact-sgm-step-20261028-v2 \
  --family-root artifacts/fast60-refinement-20261028/analytic-family-720p \
  --result /tmp/fp2-720p-new-result.json --gpu
```
