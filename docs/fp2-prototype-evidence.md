# FP2 Slang scanline prototype: scoped correctness evidence

An isolated, opt-in Daxa/Vulkan fixture compiles and tests the packed-band recurrence. On three changing tiny fixtures, its four packed directional-path arrays and aggregate sums matched an independent dense CPU oracle exactly on AMD Radeon 8060S / RADV. This is a correctness test, not a 720p, live-camera, throughput or production result.

The standalone shader has one invocation per scanline, operates on actual contiguous band widths with explicit global-disparity predecessor indexing, uses line-owned bounded scratch and `UINT32_MAX` invalid values, and runs four ordered dispatches with compute read/write barriers. The fixture checks bucket and scratch layout, address non-overlap, and teardown validation counters. Its debug buffers retain all four packed paths for comparison; the fixture explicitly reads them back. It is not connected to the production stereo graph, does not produce depth, and is not an optimized kernel. Native finite-sentinel and subpixel behavior remain separate qualification gates.

The first GPU attempt **failed with exit 139** after the Vulkan validation layer reported invalid SPIR-V. The host passed a byte count to Daxa's `byte_code_size`, which expects a count of uint32 words; Daxa multiplied that value by four. The host was corrected to pass `words.size()`, and a CPU source/mutation test protects that API unit. The failed log remains at `artifacts/fp2-prototype-20261028-v1/gpu.log` (SHA-256 `82f1c4d2f86a3b29139a4a4b09ce25aaa4534dc94f12d1a00ab73670628e68bd`). A standalone `spirv-val` executable was not available on this host.

The reviewed retry exited 0 with zero recorded validation errors or warnings after each of the three context teardowns. Its loader log confirms Khronos instance and device validation layers and the AMD RADV device. Synchronization validation was requested with `VK_LAYER_VALIDATE_SYNC=1`, but activation was not independently attested. Loader notices about environment-selected layers are separate from the fixture's validation counters. The retry log is `artifacts/fp2-prototype-20261028-v2/gpu.log` (SHA-256 `4b3dd6464cbcde768fc6c7c20fdd3f77c0cb3e135df6b9d0fbde61cc4a129fe2`). Local logs may not exist on a fresh checkout.

Before the retry, `tests/test_fp2_dispatch_hotpath.py` and `tests/test_fp2_prototype_cpu.py` passed **20 tests**, including dispatch/barrier mutations, shader-size units, source/SPIR-V/compiler identities, and CPU-only layout/oracle checks. The separately reviewed actual-source fake-device microbenchmark exercises command-recording bookkeeping with fake calls; it excludes allocations, cleanup, the actual device, shaders and GPU work and is not performance evidence. The tiny fixtures include empty and all-invalid predecessors, disjoint bands, q=0/127, bucket boundaries, padding and changing costs. They do not exercise camera images or both-eye capture.

The executable present when verified after the retry has SHA-256 `a396cf92f1ce13940bc7baf0e13efa0bf539975d35c9e3f623a299042fb7ff49`; compiled SPIR-V `df44163e38d1fac96e4738f482bdffe70fa72f3cc089df5e792a2ef099af45ca`; shader source `169df335ccb52dd6de711f8c1980ec88d8ee0fe2684aed8af1321f8775506ea9`; host source `e3040970f8216eeaed21a75ceef3ba8ee38317562b83ffc91cb675dd9c2b65cf`; and pinned Slang compiler `82e29ecd96714c64a31021160eaff4bf8604f44934e9f7b98448e962a5d12dad`. The retry log does not itself bind the invoked binary hash. The executable's `--identity` embeds these source, compiler and generated SPIR-V identities. The CPU identity test compares them with the current files before GPU use; it is a separate check, not an atomic snapshot of those files during dispatch.

To reproduce on the qualified local build, first build the explicit opt-in target and run the guards; the GPU command uses a nonblocking shared lock and fails rather than running concurrently:

```bash
cmake -S . -B build/daxa -DSTEREOFORGE_FP2_PROTOTYPE=ON
cmake --build build/daxa --target fp2_prototype_fixture -j 4
PYTHONPATH=python:. .venv/bin/python -m pytest -q \
  tests/test_fp2_dispatch_hotpath.py tests/test_fp2_prototype_cpu.py
./build/daxa/fp2_prototype_fixture --identity
./build/daxa/fp2_prototype_fixture --cpu-only
env -u VK_LAYER_ENABLES \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
  VK_LAYER_PATH="$PWD/build/validation-layer/usr/share/vulkan/explicit_layer.d" \
  LD_LIBRARY_PATH="$PWD/build/validation-layer/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  VK_LAYER_VALIDATE_SYNC=1 VK_LOADER_DEBUG=layer \
  flock -n /tmp/stereoforge-gpu.lock ./build/daxa/fp2_prototype_fixture --gpu
```
