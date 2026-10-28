# FP2 synthetic descriptor cost producer parity

On 2026-10-28 at 22:26:32–22:26:33 UTC, one isolated run exited 0 on AMD Radeon 8060S, RADV Strix Halo. For each eye direction, the fixture tested 7×5 and 320×180 synthetic descriptor and support planes. The 7×5 dispatch also exercises a partial 64-thread workgroup. The producer uses `x-d` for eye 0 and `x+d` for eye 1, checks both support masks and image bounds, and writes INF for unsupported active candidates while leaving padding at its initialized INF sentinel. Each eye and size is processed sequentially.

The producer and four recurrence passes share one GPU command list with an explicit compute-write to compute-read/write barrier after the producer, and barriers between recurrence passes. There is no host read or wait at the producer/recurrence boundary. After completion, the debug fixture reads the whole cost, all four packed paths and sum buffers, including padding, and compares each word with the scalar CPU cost reference and independent dense recurrence oracle. All four cases matched. Full readback is a debug verification boundary, not a real-time design.

These are synthetic descriptor values, not Census descriptors computed from images. The result does not establish live-camera behavior, real-image stereo quality, 720p operation, GPU performance or end-to-end throughput. Requested buffers were 41,808 bytes in each 7×5 run and 76,604,496 / 76,603,344 bytes for 320×180 eye 0 / eye 1 respectively; these are requested buffer sizes, excluding driver allocation overhead and host reference storage.

The CPU identity and hotpath checks passed 56 tests. The loader confirmed Khronos instance and device validation layers, and fixture validation counters remained zero through each teardown. Loader notices about the environment variable `VK_INSTANCE_LAYERS` are distinct from fixture validation warnings. Synchronization validation was requested with `VK_LAYER_VALIDATE_SYNC=1`, but activation was not independently attested.

Before and after the run, the following SHA-256 values matched. The source and binary hashes below are identities of this narrow experiment:

| Artifact | SHA-256 |
|---|---|
| Image fixture binary | `0ae15d47ecabfef74cdadb6d2aeae21502127e7cce6f6511e9c840e51e625b13` |
| Image fixture source | `e26f94f049c558d068f619b8b49dd263caaf4af5266c308f9121af22e23db259` |
| Tiny recurrence fixture source | `e3040970f8216eeaed21a75ceef3ba8ee38317562b83ffc91cb675dd9c2b65cf` |
| Two-slot fixture source | `26854c80ecfed883bd4f7e5672f1a63bbedbd0b9c6355738cd01c108e83262c2` |
| Cost shader | `537a9517d26ec2e9b3f86c1866fc05a28e9b33c7a8f08e9f2468a5f2510c7bcd` |
| Recurrence shader | `169df335ccb52dd6de711f8c1980ec88d8ee0fe2684aed8af1321f8775506ea9` |
| Cost SPIR-V | `fd2ae15ecdf9e0dba425654a88212c428593e382627b733e66a083edb63d0bb6` |
| Recurrence SPIR-V | `df44163e38d1fac96e4738f482bdffe70fa72f3cc089df5e792a2ef099af45ca` |
| Slang compiler | `82e29ecd96714c64a31021160eaff4bf8604f44934e9f7b98448e962a5d12dad` |

The ignored local directory `artifacts/fp2-image-20261028-v1/` retains identity JSON, before/after hashes, timestamps, exit code and loader/validation log. The log SHA-256 is `1dae2121b4c32834fb3b2ee9cb31222d1f781e5b5e8f19d501b867b14a9e717d`. These local artifacts need not exist in another checkout.

## Reproduction

Use the configured Daxa build and local validation-layer paths. The CPU identity test verifies current source, compiler and SPIR-V against the binary; do not overwrite the retained evidence. Lock failure exits 75; timeout is 60 seconds. Capture a new run's timestamps, actual exit, output and pre/post hashes separately.

```bash
cmake --build build/daxa --target fp2_image_fixture -j 4
python -m pytest -q tests/test_fp2_image_cpu.py tests/test_fp2_image_hotpath.py \
  tests/test_fp2_slots_cpu.py tests/test_fp2_slots_hotpath.py \
  tests/test_fp2_prototype_cpu.py tests/test_fp2_dispatch_hotpath.py
./build/daxa/fp2_image_fixture --identity
./build/daxa/fp2_image_fixture --cpu-only
timeout 60s flock -n -E 75 /tmp/stereoforge-gpu.lock \
  env -u VK_LAYER_ENABLES \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
  VK_LAYER_PATH="$PWD/build/validation-layer/usr/share/vulkan/explicit_layer.d" \
  LD_LIBRARY_PATH="$PWD/build/validation-layer/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  VK_LAYER_VALIDATE_SYNC=1 VK_LOADER_DEBUG=layer \
  ./build/daxa/fp2_image_fixture --gpu
```
