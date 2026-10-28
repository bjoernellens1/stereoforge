# FP2 320×180 two-slot parity experiment

On 2026-10-28, the isolated explicit-cost fixture completed once on AMD Radeon 8060S (RADV Strix Halo), exit 0, from 22:21:50 to 22:21:51 UTC. Both independently allocated slots have different deterministic 320×180 inputs. Each has empty, shifted and disjoint intervals, 16/32/64/128 buckets, padding, invalid cost holes and an all-invalid predecessor. Four serial-line directional dispatches use the existing FP2 shader per slot. Both command lists are **queued before the first host wait**; this does not establish simultaneous GPU execution. Each submission is waited on separately, and all packed path and sum words, including padding, are compared exactly with the independent dense CPU oracle. Both matched.

The fixture reads its complete results back for this debug parity check. This is a deliberate test boundary, not a proposed real-time host round trip. The synthetic inputs are explicit costs, not camera images or stereo eyes. This is neither a 720p result nor a live-camera, depth-quality, throughput or GPU-performance qualification.

Khronos validation was inserted at both instance and device levels, according to the loader log. The fixture's validation counters were zero through teardown. The loader emitted notices that `VK_INSTANCE_LAYERS` was set; those are loader notices, separate from fixture validation counters. Synchronization validation was requested via `VK_LAYER_VALIDATE_SYNC=1`; its activation was not independently attested.

## Requested buffer sizes

The following are requested GPU buffer bytes, calculated from this fixture's actual deterministic layouts; driver allocation overhead, CPU input/oracle storage and implementation-internal memory are excluded. Each slot owns separate buffers. Parameter structures are 88 bytes; offsets and line offsets are 64-bit, the remaining planes are 32-bit.

| Buffer | Slot 0 | Slot 1 |
|---|---:|---:|
| Low bounds | 230,400 | 230,400 |
| High bounds | 230,400 | 230,400 |
| Packed offsets | 460,808 | 460,808 |
| Costs | 12,287,808 | 12,287,616 |
| Sums | 12,287,808 | 12,287,616 |
| Four path planes | 49,151,232 | 49,150,464 |
| Line scratch | 1,024,000 | 1,024,000 |
| Line offsets | 10,272 | 10,272 |
| Parameters | 88 | 88 |
| **Total per slot** | **75,682,816** | **75,681,664** |

The combined requested buffers total 151,364,480 bytes. These are capacity figures, not measurements of bandwidth, latency or occupancy.

## Identity and checks

The pre- and post-run SHA-256 lists matched byte for byte. The exact identity was:

| Artifact | SHA-256 |
|---|---|
| `build/daxa/fp2_slots_fixture` | `52215b1d404a4c221b8ac717e6b5e7e71e859948b047ebac89cce7d5d9ebc585` |
| `apps/fp2_slots_fixture.cpp` | `26854c80ecfed883bd4f7e5672f1a63bbedbd0b9c6355738cd01c108e83262c2` |
| `apps/fp2_prototype_fixture.cpp` | `e3040970f8216eeaed21a75ceef3ba8ee38317562b83ffc91cb675dd9c2b65cf` |
| `python/stereoforge/shaders/daxa/fp2_banded.slang` | `169df335ccb52dd6de711f8c1980ec88d8ee0fe2684aed8af1321f8775506ea9` |
| Compiled SPIR-V | `df44163e38d1fac96e4738f482bdffe70fa72f3cc089df5e792a2ef099af45ca` |
| Slang compiler | `82e29ecd96714c64a31021160eaff4bf8604f44934e9f7b98448e962a5d12dad` |

The CPU identity and hotpath test set passed 36 tests. The log, start/end timestamps, exit code, identity JSON and before/after hashes are retained locally under `artifacts/fp2-slots-20261028-v1/`; these ignored artifacts may not exist on another checkout. The GPU log SHA-256 is `ee261b19045756df96585d8779897cd6625112a8356284c41e0d3c3454ac9bc2`.

## Reproduction

Use the configured Daxa build and local validation layer paths. Verify the source, compiler and SPIR-V identity against the executable before GPU execution; do not overwrite the retained original evidence. Lock failure exits 75 and the watchdog limits the attempt to 60 seconds.

```bash
cmake --build build/daxa --target fp2_slots_fixture -j 4
python -m pytest -q tests/test_fp2_slots_cpu.py tests/test_fp2_slots_hotpath.py \
  tests/test_fp2_dispatch_hotpath.py tests/test_fp2_prototype_cpu.py
./build/daxa/fp2_slots_fixture --identity
./build/daxa/fp2_slots_fixture --cpu-only
timeout 60s flock -n -E 75 /tmp/stereoforge-gpu.lock \
  env -u VK_LAYER_ENABLES \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
  VK_LAYER_PATH="$PWD/build/validation-layer/usr/share/vulkan/explicit_layer.d" \
  LD_LIBRARY_PATH="$PWD/build/validation-layer/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  VK_LAYER_VALIDATE_SYNC=1 VK_LOADER_DEBUG=layer \
  ./build/daxa/fp2_slots_fixture --gpu
```

The CPU identity test compares the executable's embedded hashes with the current sources, compiler and SPIR-V and rejects mutations. Capture the run output, actual exit code, timestamps and pre/post binary and input hashes when reproducing; a new run does not replace the evidence above.
