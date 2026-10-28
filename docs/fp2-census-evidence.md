# FP2 Census, packed cost and recurrence parity

On 2026-10-28 at 22:30:42–22:30:43 UTC, a single isolated test exited 0 on AMD Radeon 8060S (RADV Strix Halo). Both eye directions were tested sequentially at 7×5 and 320×180 with deterministic synthetic rectified luma and support masks. The shared support-aware 5×5 Census helper feeds the packed cost producer, which feeds four directional recurrence passes. Explicit compute barriers separate these stages; no host wait or read occurs between them.

After GPU completion, debug readback compared all descriptors, Census validity, packed costs, paths, sums and padding exactly against scalar CPU references. All matched. The 7×5 case covers a partial 64-thread workgroup. This complete readback is a debug boundary, not a proposed real-time host round trip.

| Dimensions | Eye | Valid descriptors per input | Valid costs | Requested GPU buffer bytes |
|---|---:|---:|---:|---:|
| 7×5 | 0 | 3 | 5 | 42,448 |
| 7×5 | 1 | 3 | 4 | 42,448 |
| 320×180 | 0 | 55,566 | 1,761,667 | 77,526,176 |
| 320×180 | 1 | 55,566 | 1,759,091 | 77,525,024 |

Requested sizes exclude CPU oracle storage and driver allocation overhead. The nonempty counts prevent an all-invalid parity result from being mistaken for a useful test. The input is synthetic luma, not a camera stream; this is not a 720p, measured-performance, depth-quality or end-to-end qualification.

The CPU identity and hotpath test set passed 79 tests. The Vulkan loader confirmed Khronos instance and device validation layers; fixture validation counters were zero through teardown. The loader printed environment-layer notices, separate from fixture validation warnings. Synchronization validation was requested with `VK_LAYER_VALIDATE_SYNC=1`, but activation was not independently attested.

The following SHA-256 values matched before and after the run:

| Artifact | SHA-256 |
|---|---|
| Census fixture binary | `d1528f63d89a2b569f1e479ffafdfeb95ffb94e06f2db3a29fac8847005f1151` |
| Census fixture source | `3135ad367c63a75f5f0c09d738b45cd4157cf51c3df6f0b378ccbbf1fa43d7a1` |
| Recurrence host source | `e3040970f8216eeaed21a75ceef3ba8ee38317562b83ffc91cb675dd9c2b65cf` |
| Census shader | `7026a1f3eb3b4a446cf2b8c00cb83ac77641c2e951bbbad0c7d50737991a2944` |
| Shared refinement helper | `dc4e735bfd0300d54bec71471403e3a0fb2742e9375f2e39053811d4db878c2b` |
| Cost shader | `537a9517d26ec2e9b3f86c1866fc05a28e9b33c7a8f08e9f2468a5f2510c7bcd` |
| Recurrence shader | `169df335ccb52dd6de711f8c1980ec88d8ee0fe2684aed8af1321f8775506ea9` |
| Census SPIR-V | `682490d0483993e17c676e173fec7a649201f5a0f3ee45d92a81dd2ee388d14a` |
| Cost SPIR-V | `fd2ae15ecdf9e0dba425654a88212c428593e382627b733e66a083edb63d0bb6` |
| Recurrence SPIR-V | `df44163e38d1fac96e4738f482bdffe70fa72f3cc089df5e792a2ef099af45ca` |
| Slang compiler | `82e29ecd96714c64a31021160eaff4bf8604f44934e9f7b98448e962a5d12dad` |

The ignored local `artifacts/fp2-census-20261028-v1/` directory contains identity, before/after hashes, timestamps, exit code and loader/validation log. Its log SHA-256 is `a5e43fb4b34b40231247c2ebf95330d85cc95250d5f178330c870490bcf0e714`; local artifacts may not exist in a fresh checkout.

## Reproduction

Use the configured Daxa build and local validation-layer paths. The CPU identity test verifies sources, included helper, compiler and SPIR-V against the binary. A new run should retain its own output, exit, timestamps and before/after hashes instead of overwriting this evidence. A busy lock exits 75; the watchdog is 60 seconds.

```bash
cmake --build build/daxa --target fp2_census_fixture -j 4
PYTHONPATH=python:. .venv/bin/python -m pytest -q \
  tests/test_fp2_census_cpu.py tests/test_fp2_census_hotpath.py \
  tests/test_fp2_image_cpu.py tests/test_fp2_image_hotpath.py \
  tests/test_fp2_slots_cpu.py tests/test_fp2_slots_hotpath.py \
  tests/test_fp2_prototype_cpu.py tests/test_fp2_dispatch_hotpath.py
./build/daxa/fp2_census_fixture --identity
./build/daxa/fp2_census_fixture --cpu-only
timeout 60s flock -n -E 75 /tmp/stereoforge-gpu.lock \
  env -u VK_LAYER_ENABLES \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
  VK_LAYER_PATH="$PWD/build/validation-layer/usr/share/vulkan/explicit_layer.d" \
  LD_LIBRARY_PATH="$PWD/build/validation-layer/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  VK_LAYER_VALIDATE_SYNC=1 VK_LOADER_DEBUG=layer \
  ./build/daxa/fp2_census_fixture --gpu
```
