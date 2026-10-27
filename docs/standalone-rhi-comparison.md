# Standalone C++/Slang RHI comparison — 2026-10-27

The production build now compiles pinned Slang RHI source against the matching
official Slang SDK. It does not discover Python, import SlangPy, or link SGL.
Only Vulkan is enabled. Python remains available for research/reference tooling;
it is not needed to configure, build or run the standalone native applications.
The native live camera-to-depth chain is still a separate incomplete requirement.

```bash
cmake --preset standalone
cmake --build --preset standalone -j 12
ctest --preset standalone
```

Linux x86_64, C++20, CMake/Ninja and network access for pinned dependencies are
required. Source/SDK archives can also be supplied through the cache variables
in cmake/StandaloneRHI.cmake. The legacy `vulkan` preset explicitly selects the
SlangPy research provider; `standalone` uses source-built RHI and the official SDK.

## Independent dependency proof

- RHI revision: `1a9768741246b1f356116d5dfa305933034ce049`.
- RHI archive SHA256:
  `ab0daed27b68bf3878358d059694a35e36bf99ed38ac7b9b1926cefc84481396`.
- Slang SDK: official Linux x86_64 release 2026.12, archive SHA256
  `2409ecc7b6710783e05fd59792558c275c637e3e3c37fb5054fcfc858e570d82`.
- Both actual license files match the retained Apache-2.0 WITH LLVM-exception
  notice byte-for-byte. No dependency binaries are published with the source.
- `ldd` lists the official SDK's Slang compiler plus standard system libraries;
  no Python, SGL or SlangPy library/path appears.
- Configure/build and actual GPU runs worked with `PYTHONHOME=/nonexistent` and
  empty `PYTHONPATH`. The standalone CMake cache contains no Python discovery.
- RHI cache enables Vulkan and disables CPU, CUDA, OptiX, Metal, D3D and WebGPU.

GCC 16 emits an upstream COM delete-thunk warning in the pinned dependency.
The build narrowly removes `-Werror` for that warning on the dependency target;
the warning stays visible and StereoForge code is unchanged.

## Matched 720p fixture

The new native benchmark consumes an offline exported fixture, not a Python
process. Its 1280×720 source/target data exactly match the earlier fusion benchmark:
two RGB images, explicit depth/confidence/provenance for both lenses, calibrated
rays, poses and infinity lookup maps. Native additionally receives source
identities and ages. Input identity:
`6b1ae7dacab2e81554b56e504fb2fd235657069c8716485e666cf436caa46803`.

The exporter uses Python for test preparation only. Once exported, the fixture
and native benchmark run independently of Python/SlangPy:

```bash
# Optional offline test-data preparation in the research environment:
PYTHONPATH=python .venv/bin/python benchmarks/export_view_fixture.py \
  --width 1280 --height 720 --output artifacts/fixture-new

# Actual standalone C++ benchmark:
PYTHONHOME=/nonexistent PYTHONPATH='' \
  build/standalone/native_view_benchmark --fixture artifacts/fixture-new \
  --mode strict --readback gpu --warmup 3 --frames 30 \
  --output artifacts/native-strict-gpu.json
```

Use `--mode preview` for approximate RGB and `--readback cpu` for all native
output planes. `--dump-prefix` optionally reads a GPU-mode result after all
measured iterations; dump work is excluded and separately reported.

## Measured results

AMD Radeon 8060S, RADV STRIX_HALO. Three warmups and 30 samples per native case,
with initialization/input-file loading excluded. Each measured GPU-output call
still validates and uploads host source arrays, dispatches fusion and waits for
completion. It is not yet a persistent GPU-input path.

| Native standalone 720p fusion | p50 | p95 |
|---|---:|---:|
| Strict, outputs retained on GPU | 25.023 ms | 26.026 ms |
| Preview, outputs retained on GPU | 22.458 ms | 23.552 ms |
| Strict, complete native CPU readback | 150.126 ms | 163.581 ms |
| Preview, complete native CPU readback | 153.334 ms | 159.460 ms |

Strict GPU-output host enqueue p50 was 23.36 ms; completion wait p50 was 1.61 ms.
Preview enqueue/wait were 20.79/1.69 ms. These component medians do not sum to a
measured total percentile. Enqueue includes validation, metadata copies, uploads,
encoding and submission; wait includes queued GPU work/transfers and is not a
GPU dispatch timestamp. Complete native CPU readback p50 was 130.33/134.00 ms.

The prior Python adapter measured 90.455/90.318 ms strict/preview, using two
warmups, five interleaved samples, a packed 16-byte/pixel output, and fewer output
planes. Native full readback includes owners and identity/age gathering, with
separate buffer reads. Thus neither 25-vs-90 nor 150-vs-90 is a matched measure
of language overhead. Native CPU-output execution is not demonstrated faster.
Python GPU dispatch stage sums measured approximately 1.2–1.4 ms; native
completion waits are a different quantity, so equal GPU speed is not established.

No case includes stereo, rectification, camera delivery, temporal processing,
display or a downstream GPU consumer. Even native GPU-output p95 exceeds
16.67 ms. These runs do **not** demonstrate 60 Hz end to end. The next causal
step is persistent calibration/input resources and GPU-to-GPU stage handoff,
followed by an otherwise matched CPU-readback comparison if host outputs are
required. Merely changing the host language has not met the target.

Both native modes matched Python Vulkan and the CPU oracle exactly on this
fixture: RGB, masks, depth, confidence, source attribution and provenance.
Metric identities/ages matched selected owners; invalid metric pixels retained
ID -1 and age zero. This fixture does not remove the float32 half-pixel boundary
limitation documented in fusion-evidence.md.

Local ignored evidence: `artifacts/fixture-720p`,
`artifacts/native-vs-python-fusion-20261027/` (four raw timing reports, dumps,
comparison and hashed run manifest), and `artifacts/standalone-rhi-*` for build
provenance and validation. The native binary/source and fixture are bound by
the run manifest; raw arrays and dependency binaries are not committed.

## Separate stereo budget constraint

An instrumented shared-shader stereo run at 1280×720, four paths, two warmups
and three samples measured the following **single-reference** GPU dispatch sums:

| Disparity candidates | GPU stage-sum median |
|---|---:|
| 32 | 18.60 ms |
| 64 | 25.07 ms |
| 128 | 66.90 ms |

This profile uses the SlangPy research adapter solely for timestamps; it is not
the standalone fusion benchmark above. At D128, disparity selection alone takes
approximately 33 ms and the four aggregation stages approximately 29 ms. The
current Python pipeline runs another reference estimate for consistency. These
measurements exclude rectification, fusion, CPU readback and capture. Eliminating
host round trips therefore cannot by itself deliver 60 Hz with the current full
stereo kernels. Source/input and raw stage timings are retained in
`artifacts/720p-stereo-stage-profile-20261027.json`.

Required next sequence: persistent same-device GPU inputs/outputs, instrumented
complete native chain, coalesced/subgroup-parallel disparity selection, improved
SGM storage/aggregation, and finally bounded/adaptive work only under frozen
quality constraints. Reducing depth resolution, candidates, refresh rate or
consistency quality is not silently counted as meeting the original target.

## Verification results

Standalone CTest passes 4/4, including honest missing-layer rejection. Standalone
stereo self-tests and full geometry/temporal fixtures pass positive Khronos core
validation with zero diagnostics across both stdout and stderr. No Python runtime
was used in those GPU runs. Synchronization/GPU-assisted validation remains open.
