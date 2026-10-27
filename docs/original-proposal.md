# Project proposal: StereoForge

Portable, high-performance stereo vision and temporal depth reconstruction

Proposed repository: `stereoforge`\
Tagline: Open stereo depth. Any GPU. Real-time geometry.

StereoForge would be a modular C++20 and Slang project for real-time depth estimation, initially supporting the Stereolabs ZED 2 but designed to work with arbitrary stereo cameras.

The objective is to build a genuine alternative to proprietary GPU stereo pipelines, rather than a simple ZED SDK wrapper.

Its core design principles:

- Vulkan-first, Slang-based: No CUDA or ROCm dependency, including for neural inference.
- Modular stereo algorithms: Classical SGM, neural stereo, and hybrid pipelines behind common interfaces.
- Temporal reconstruction: Motion compensation, depth-history reuse, adaptive computation, and uncertainty propagation.
- GPU-resident processing: Minimize allocations, memory transfers, synchronization overhead, and CPU intervention.
- Camera-independent core: ZED 2 is the first adapter, not a permanent constraint.
- Measurable quality: Benchmark performance, metric depth accuracy, completeness, and geometric reconstruction against ZED SDK and other baselines.

I would use an Apache-2.0 license for the new project, subject to a dependency/license review.

## 1. System architecture

ZED 2

Recorded data

Other stereo

Capture & calibration

Stereo frames · Timestamps · Intrinsics · Extrinsics · IMU

GPU foundation — Slang + Vulkan

Compute graph · Resource allocation · Shader variants · Synchronization

Classical depth

Census · SGM · Subpixel disparity

Neural depth

Vulkan inference · Learned features · Refinement

Temporal geometry engine

Motion estimation · Reprojection · Occlusion handling · Confidence fusion

Adaptive scheduler

Per-region compute budgets · Disparity search · Neural refinement

Depth + confidence

Point clouds

SLAM / mapping

An important design decision: classical, neural, and hybrid stereo are alternative processing strategies, not three independent applications.

They should share capture, calibration, GPU resources, depth output formats, temporal history, and benchmarks.

This allows meaningful comparisons without maintaining three unrelated systems.

## 2. Technology stack

| Layer                  | Technology                        | Decision                       |
| ---------------------- | --------------------------------- | ------------------------------ |
| Main implementation    | C++20                             | Core runtime and public API    |
| Compute language       | Slang                             | All custom GPU algorithms      |
| GPU backend            | Vulkan 1.3                        | Primary execution target       |
| GPU abstraction        | Slang RHI                         | Resource and shader management |
| Capture                | ZED Open Capture                  | Original ZED 2 integration     |
| Classical stereo       | Census + SGM                      | Primary real-time baseline     |
| Neural inference       | ncnn Vulkan                       | First neural runtime           |
| Alternative ML runtime | IREE Vulkan                       | Research and advanced models   |
| CPU reference          | OpenCV + scalar implementations   | Validation and fallback        |
| Research interface     | Python + optional SlangPy         | Experiments and analysis       |
| Build                  | CMake + Ninja + CMake presets     | Reproducible builds            |
| Testing                | CTest + GoogleTest                | Unit and integration tests     |
| Benchmarking           | Google Benchmark + GPU timestamps | Performance analysis           |

Use Slang RHI as a pinned dependency. Its API is still changing and compatibility is not guaranteed between revisions.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



ncnn provides a Vulkan GPU inference backend, CPU support, and model conversion tooling without requiring a CUDA or ROCm runtime. IREE is a second possible Vulkan-based neural runtime, but there is no reason to integrate both before validating a first network.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+1



For maximum performance, the backend should support native Vulkan resource handles where necessary. This allows sharing images and buffers with rendering, mapping, or neural inference without mandatory host readbacks.

Such sharing is an explicit integration task: ncnn or IREE interoperability cannot be assumed to be zero-copy just because both use Vulkan.

## 3. Repository structure

I would keep the project in one monorepo, with strict interfaces between components.

```
stereoforge/
├── CMakeLists.txt
├── CMakePresets.json
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── pyproject.toml
│
├── cmake/
│   ├── Dependencies.cmake
│   ├── SlangShaders.cmake
│   └── CompilerOptions.cmake
│
├── include/stereoforge/
│   ├── camera.hpp
│   ├── calibration.hpp
│   ├── frame.hpp
│   ├── stereo.hpp
│   ├── depth.hpp
│   ├── inference.hpp
│   ├── pipeline.hpp
│   └── gpu.hpp
│
├── src/
│   ├── core/
│   │   ├── frame.cpp
│   │   ├── calibration.cpp
│   │   └── pipeline.cpp
│   │
│   ├── capture/
│   │   ├── zed2/
│   │   │   ├── zed_capture.cpp
│   │   │   ├── zed_sensors.cpp
│   │   │   └── zed_calibration.cpp
│   │   ├── video/
│   │   └── replay/
│   │
│   ├── gpu/
│   │   ├── vulkan/
│   │   ├── shader_cache/
│   │   ├── resource_pool/
│   │   └── frame_graph/
│   │
│   ├── stereo/
│   │   ├── classical/
│   │   │   ├── block_matching.cpp
│   │   │   ├── census_sgm.cpp
│   │   │   └── hierarchical_sgm.cpp
│   │   ├── neural/
│   │   │   ├── ncnn_backend.cpp
│   │   │   └── model_registry.cpp
│   │   └── hybrid/
│   │       └── adaptive_stereo.cpp
│   │
│   ├── temporal/
│   │   ├── motion.cpp
│   │   ├── reprojection.cpp
│   │   ├── fusion.cpp
│   │   └── history.cpp
│   │
│   ├── scheduling/
│   │   ├── confidence.cpp
│   │   ├── disparity_prior.cpp
│   │   └── adaptive_budget.cpp
│   │
│   └── output/
│       ├── pointcloud.cpp
│       ├── recorder.cpp
│       └── ros2/                # Optional package
│
├── shaders/
│   ├── common/
│   ├── preprocessing/
│   │   ├── yuv_to_rgb.slang
│   │   ├── grayscale.slang
│   │   └── rectify.slang
│   ├── stereo/
│   │   ├── census.slang
│   │   ├── matching_cost.slang
│   │   ├── sgm_aggregate.slang
│   │   ├── disparity_select.slang
│   │   ├── lr_consistency.slang
│   │   └── subpixel.slang
│   ├── temporal/
│   │   ├── optical_flow.slang
│   │   ├── depth_warp.slang
│   │   ├── occlusion.slang
│   │   └── temporal_fusion.slang
│   ├── hybrid/
│   │   ├── disparity_bounds.slang
│   │   └── confidence_mask.slang
│   └── geometry/
│       ├── depth_to_points.slang
│       └── normal_estimation.slang
│
├── models/
│   ├── manifests/
│   └── converters/
│
├── python/stereoforge/
│   ├── __init__.py
│   ├── capture.py
│   ├── pipeline.py
│   └── evaluation.py
│
├── apps/
│   ├── stereoforge-viewer/
│   ├── stereoforge-record/
│   ├── stereoforge-bench/
│   └── stereoforge-calibrate/
│
├── configs/
│   ├── cpu-reference.yaml
│   ├── classical-fast.yaml
│   ├── classical-quality.yaml
│   ├── neural-quality.yaml
│   └── hybrid-realtime.yaml
│
├── tests/
│   ├── unit/
│   ├── shader/
│   ├── integration/
│   ├── regression/
│   └── datasets/
│
├── benchmarks/
│   ├── performance/
│   ├── accuracy/
│   └── comparisons/
│
├── examples/
│   ├── zed2-live/
│   ├── stereo-images/
│   ├── temporal-depth/
│   └── python/
│
├── docs/
│   ├── architecture.md
│   ├── camera-calibration.md
│   ├── shader-performance.md
│   ├── neural-models.md
│   └── benchmark-protocol.md
│
├── scripts/
├── third_party/
│   └── README.md
└── .github/
    └── workflows/
```

The organization deliberately separates the GPU algorithms in `shaders/` from the scheduling and execution code in `src/`.

It should be possible to replace SGM, the neural model, or the optical-flow implementation without modifying the camera adapter or downstream applications.

## 4. Reusing existing work instead of rewriting everything

This is where I would invest substantial care.

| Component                                                                                              | Reuse                                          | Strategy                           |
| ------------------------------------------------------------------------------------------------------ | ---------------------------------------------- | ---------------------------------- |
| [ZED Open Capture](https://github.com/stereolabs/zed-open-capture?utm_source=chatgpt.com)              | Camera acquisition, sensor synchronization     | Direct dependency                  |
| [Slang RHI](https://github.com/shader-slang/slang-rhi?utm_source=chatgpt.com)                          | GPU resources and dispatch                     | Pinned dependency                  |
| [Fixstars libSGM](https://github.com/fixstars/libSGM?utm_source=chatgpt.com)                           | SGM implementation and optimization techniques | Study and port algorithms to Slang |
| [Vulkan Stereo Depth](https://github.com/tvaranka/vulkan_android_example_depth?utm_source=chatgpt.com) | Matching examples                              | Reference                          |
| [ncnn](https://github.com/Tencent/ncnn?utm_source=chatgpt.com)                                         | Neural inference                               | Optional dependency                |
| [IREE](https://github.com/iree-org/iree?utm_source=chatgpt.com)                                        | Additional neural compiler/runtime             | Later integration                  |
| [CODD](https://github.com/facebookresearch/CODD?utm_source=chatgpt.com)                                | Modular motion and temporal fusion             | Research reference only            |
| [TemporalStereo](https://github.com/youmi-zym/TemporalStereo?utm_source=chatgpt.com)                   | Sparse temporal stereo matching                | Research reference                 |

The licensing details materially affect what we can copy.

- ZED Open Capture: MIT.
- Fixstars libSGM: Apache-2.0.
- Slang RHI: Apache-2.0 with LLVM exception.
- ncnn: BSD-3-Clause.
- CODD: Predominantly CC-BY-NC; do not incorporate its non-commercial code into the Apache-licensed core without separate permission.

These license descriptions are verified against the respective upstream repositories.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+4



I would record each dependency's pinned revision, license, attribution obligations, and whether it is linked, vendored, or only consulted in `third_party/README.md`.

A port of libSGM is particularly promising because its CUDA implementation provides concrete ideas for memory-efficient path aggregation. We can translate applicable kernel designs to Slang without retaining any CUDA runtime dependency.

## 5. Define stable interfaces first

The central API should not expose ZED-specific types or Vulkan implementation details to users of the stereo algorithms.

A minimal C++ design could look like this:

```

namespace stereoforge {

struct StereoFrame {
    ImageHandle left;
    ImageHandle right;
    uint64_t timestamp_ns;
    CalibrationHandle calibration;
};

struct DepthFrame {
    ImageHandle disparity;
    ImageHandle depth_m;
    ImageHandle confidence;
    ImageHandle validity;
    uint64_t timestamp_ns;
    bool temporally_predicted;
};

struct StereoContext {
    GpuDevice& device;
    FrameHistory* history;
    MotionEstimate* motion;
};

class IStereoEstimator {
public:
    virtual ~IStereoEstimator() = default;

    virtual void enqueue(
        const StereoFrame& input,
        StereoContext& context,
        DepthFrame& output
    ) = 0;
};

} // namespace stereoforge

```

This is a conceptual API sketch, not complete compilable code. `ImageHandle` would reference a typed resource with documented ownership, lifetime, device identity, format, and synchronization state.

I would define four implementation families using this interface:

| Estimator               | Behavior                                       |
| ----------------------- | ---------------------------------------------- |
| `CpuSgmEstimator`       | Deterministic reference implementation         |
| `VulkanSgmEstimator`    | High-performance classical GPU stereo          |
| `NeuralStereoEstimator` | Full neural depth estimation                   |
| `HybridStereoEstimator` | Classical stereo + selective neural refinement |

The temporal processor should be a separate module that consumes and produces depth estimates. It should not be hardwired into a particular stereo estimator.

Likewise, the scheduler should select processing strategies through explicit capabilities instead of knowing which concrete implementation is active.

### One particularly important detail: predicted versus measured geometry

A temporal prediction must never silently masquerade as a fresh independent stereo measurement.

I would require the depth output to include a provenance mask or metadata for:

- Fresh stereo measurements.
- Motion-reprojected measurements.
- Temporally fused estimates.
- Neural completions or refinements.
- Invalid or occluded pixels.

This becomes especially valuable when integrating depth into SLAM or Gaussian mapping.

## 6. GPU processing design

The most important performance decisions should be made before writing the shaders.

Persistent resources: Allocate frame buffers and intermediate working storage once wherever possible. Reuse a bounded ring of resources rather than allocating every frame.

Asynchronous execution: Use Vulkan command buffers and timeline synchronization to overlap capture, GPU processing, and consumption when dependencies permit.

Minimal intermediate data: Avoid materializing an entire dense FP32 disparity cost volume if a compact integer representation, streaming aggregation, or hierarchical search produces equivalent results.

Shader specialization: Compile variants for disparity count, workgroup size, SGM path count, and device capabilities. Maintain correctness fallbacks on devices lacking optional features.

Profiling: Record GPU timestamp queries for individual stages, CPU enqueue cost, buffer transfers, and synchronization waits. Avoid judging optimization success from end-to-end FPS alone.

Neural memory interoperability: Start with a functional ncnn Vulkan integration even if it requires a copy. Optimize external resource sharing only once profiling shows it matters.

### Runtime pipeline selection

A configuration should look approximately like:

```

camera:
  driver: zed_open_capture
  model: zed2
  resolution: 1280x720
  fps: 60

gpu:
  backend: vulkan
  shader_language: slang
  precision: mixed
  async_execution: true

stereo:
  backend: hybrid
  classical:
    algorithm: census_sgm
    paths: 4
    max_disparity: 128
    subpixel: true

  neural:
    runtime: ncnn_vulkan
    model: null
    strategy: uncertain_regions

temporal:
  enabled: true
  motion_source: optical_flow
  history_frames: 4
  occlusion_check: true
  use_disparity_priors: true

scheduler:
  mode: latency_budgeted
  target_fps: 60
  max_compute_ms: 12.0

output:
  depth: true
  confidence: true
  pointcloud: false
  gpu_resident: true

```

The neural model name remains intentionally unspecified. We need to verify that an appropriate network can actually be converted and run correctly through the portable backend before selecting it as the default.

The 12 ms value is an initial processing target, not a measured configuration.

## 7. Development milestones

M0 — Camera and reliable reference

Working ZED 2 capture, calibration, IMU timestamps, deterministic recording/replay, CPU disparity and depth.

Exit criterion: Recorded data produces geometrically valid depth independently of the physical camera.

M1 — Portable Vulkan stereo

Slang preprocessing, Census descriptors, disparity search, SGM aggregation, subpixel refinement, and consistency checks.

Exit criterion: Passes numerical and geometric regression tests against the CPU reference; GPU timings are independently reproducible.

M2 — Portable neural inference

Integrate ncnn Vulkan and a lightweight stereo/refinement model, including conversion and operator validation.

Exit criterion: A complete neural processing path runs on AMD without CUDA or ROCm and has measured accuracy and latency.

M3 — Temporal and hybrid processing

Motion estimation, history reprojection, occlusion detection, adaptive disparity search, and neural refinement scheduling.

Exit criterion: Demonstrable improvement in compute or reconstruction quality without introducing unacceptable temporal drift.

M4 — Performance optimization

GPU resource reuse, shader specialization, reduced memory traffic, multi-device profiling, and sustained operation.

Exit criterion: Reproducible benchmarks on AMD, NVIDIA, and preferably Intel GPUs, with both latency and depth-quality metrics.

M5 — Robotics integration

Stable C++ and Python APIs, ROS 2 adapter, depth-to-point-cloud conversion, GPU buffer interoperability, and downstream mapping evaluation.

Exit criterion: A full live ZED 2-to-mapping demonstration with no compulsory CPU readback in the depth-processing path.

## 8. Initial GitHub issues

I would open the project with this small, actionable first backlog:

| Issue       | Priority | Scope                                        |
| ----------- | -------- | -------------------------------------------- |
| `CORE-001`  | P0       | CMake, tests, dependency lock, CI            |
| `GPU-001`   | P0       | Slang RHI Vulkan initialization and dispatch |
| `CAM-001`   | P0       | ZED 2 Open Capture adapter                   |
| `CAM-002`   | P0       | Calibration and rectification validation     |
| `DATA-001`  | P0       | Deterministic stereo recording/replay        |
| `REF-001`   | P0       | CPU SGBM and depth correctness oracle        |
| `GPU-002`   | P1       | Slang image preprocessing                    |
| `GPU-003`   | P1       | Census matching and disparity selection      |
| `GPU-004`   | P1       | SGM aggregation with four directions         |
| `BENCH-001` | P0       | GPU timing and depth accuracy harness        |
| `NN-001`    | P2       | ncnn Vulkan runtime proof of concept         |
| `TEMP-001`  | P2       | Temporal reprojection and validity tests     |

This is intentionally more conservative than starting with a complete hybrid pipeline. The biggest early risks are rectification correctness, SGM performance, and portable neural operator support.

## 9. How we should measure success

### StereoForge v1.0 targets

Classical stereo throughput

# 60 FPS

720p on Strix Halo — stretch target

CUDA / ROCm dependencies

# 0

Runtime and deployment

GPU implementations

# 3

Classical, neural, hybrid

Supported GPU vendors

# 3

AMD, NVIDIA, Intel — validation target

Quality target: Match or exceed a well-defined official ZED SDK operating point on measured depth accuracy and surface reconstruction, with latency and completeness reported separately.

All of these are project goals, not achieved measurements. Neural and hybrid throughput need their own measured targets once a model has been selected.

## Final architectural recommendation

I would establish StereoForge as an independent open-source project, not tie it to an existing Gaussian SLAM or simulation repository.

It should be usable as:

1. A standalone stereo-depth camera application.
2. A C++ library for real-time robotics.
3. A Python-accessible research platform.
4. A GPU-resident depth producer for Vulkan rendering and mapping.

For the initial implementation, prioritize Slang/Vulkan SGM, ZED 2 capture, reliable calibration, and benchmarking. Then integrate portable neural inference and temporal geometry.

That ordering provides a useful product before the more speculative research components are complete, while preserving the architecture needed for a genuinely competitive open stereo-depth engine.