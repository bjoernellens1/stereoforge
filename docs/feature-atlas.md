# Shared stereo feature atlas — experimental GPU kernel

The same rectified luma and support that StereoForge already computes can
produce a compact atlas reusable by stereo matching, local refinement,
feature-based geometry and uncertainty estimation.

The initial implementation is **a real shared Slang compute entry point plus
an exact CPU oracle**, but is not yet an integrated default depth-graph stage.

## Contract

Each rectified camera eye owns one uint32 luma plane, one binary uint32 support
plane and an output of 16-byte aligned records:

| Lane | Data type | Meaning |
|---|---|---|
| 0 | uint32 | 24-bit Census (5×5 excluding center, dy-major then dx) |
| 1 | uint32 | Sobel L1 gradient magnitude |
| 2 | uint32 | Sum of eight absolute neighbour-to-center differences |
| 3 | uint32 | 5×5 support-valid mask, exactly 0 or 1 |

Out-of-bounds or unsupported 5×5 windows generate four zero words, never
unknown/stale descriptors. Inputs outside 0..255 likewise invalidate the
center. The GPU entry runs once for both eyes (dispatch z=2) using direct
resident GPU buffers; no intermediate CPU readback is required by the shader.

The record is read-only and can be shared by later SGM cost, multi-hypothesis
matching, adaptive recovery and confidence operators without recomputing the
same Census or texture features. Current consumers still use their old buffers;
**no efficiency or quality gain is claimed until real graph integration and
ABBA benchmarking pass**.

## Build

CPU oracle and tests:

```sh
cmake --preset cpu
cmake --build --preset cpu --target feature_atlas_cpu_contracts
ctest --preset cpu -R feature_atlas_cpu_contracts --output-on-failure
```

Optional offline SPIR-V compilation using the project-pinned Slang compiler:

```sh
cmake --preset daxa -DSTEREOFORGE_DAXA_FEATURE_ATLAS=ON
cmake --build --preset daxa --target daxa_feature_atlas_shader
```

No CUDA/ROCm dependency is introduced. This compile gate requires the pinned
Daxa and Slang toolchain, and does not execute a GPU dispatch.

## Integration gate

1. Give the atlas a typed producer lease on the same Daxa foundation and
   identical calibration/source ownership as the rectified frame.
2. Retain luma/support owners until the atlas writer and all readers complete;
   prevent aliasing a still-live feature buffer.
3. Compare GPU shader records byte-for-byte against the C++ CPU oracle on
   awkward sizes, bounds, holes and arbitrary textures under Vulkan core and
   synchronization validation.
4. Add opt-in SGM/refinement consumption without recomputing a second Census.
   Preserve exact 24-bit Census ordering and existing texture and support
   semantics; do not alter algorithmic quality gates implicitly.
5. Measure whole-chain GPU traffic and p50/p99 under a fixed quality/profile,
   compare to the existing shared-Census variant, then decide whether it is a win.

The atlas costs 16 bytes per rectified pixel per eye (about 29.5 MiB for two
1280×720 eyes), so it is useful only if enough duplicated work or intermediate
storage is eliminated. All numbers are requested payload bytes, not VRAM peaks.
