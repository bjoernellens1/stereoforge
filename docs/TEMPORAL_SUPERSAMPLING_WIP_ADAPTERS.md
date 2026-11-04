# Temporal Supersampling (`v12`) — WIP Dummy Adapters

> **Status:** `WORK IN PROGRESS (DUMMY ADAPTERS)`
>
> While `P0` (continuous subpixel refinement), `P1` (`DaxaRefinement::Frame` integration into `DaxaTemporal`), `P2` (calibrated inverse-depth uncertainty fusion, foreground disocclusion guard, and 50% `left_eye_only` memory reduction), and `P3` (current-frame full-res Census re-search guided by pose-warped historical prior) are implemented natively in Vulkan/Daxa and Slang, the following three integration points are currently **WIP dummy adapters** and must **not** be treated as production hardware or shader implementations.

---

## 1. Live Visual Odometry / IMU Rig Pose Source (`P1`–`P3` on Real Camera Sequences)

- **C++ Adapter:** [`stereoforge::WipDummyRigPoseAdapter`](../include/stereoforge/daxa_temporal_wip_adapters.hpp)
- **Python Adapter:** [`stereoforge.temporal_wip_adapters.WipDummyLivePoseAdapter`](../python/stereoforge/temporal_wip_adapters.py)
- **Status String:** `"WIP_DUMMY_POSE_ADAPTER_NO_LIVE_VIO_IMU"` (`is_wip_dummy() == true`)
- **Current Dummy Behavior:**
  - Synthesizes a clock-aligned `DaxaTimedRigPose` (`timestamp_ns == capture.host_ingress_timestamp_ns`, `clock_domain == capture.host_clock_domain`) using either an identity transform or a fixed synthetic per-frame translation step `step_translation_m`.
  - Allows end-to-end wiring and contract testing of `DaxaTemporal::process(...)` on live or recorded `CaptureFrame` streams before an external visual-inertial odometry (VIO) or IMU-preintegration pose source is connected.
- **Required for Production Promotion:**
  - Replace the dummy pose generator with a hardware-synchronized VIO / IMU pose provider in the same monotonic clock domain (`capture.host_clock_domain`) and propagate pose covariance $\sigma_{\text{pose}}^2$ into the inverse-depth fusion weights.

---

## 2. Cross-Frame Historical Rectified Image-Pair Ring Buffer (`P3` Multi-Pair Cost Aggregation)

- **C++ Adapter:** [`stereoforge::WipDummyMultiPairStereoRingAdapter`](../include/stereoforge/daxa_temporal_wip_adapters.hpp)
- **Python Adapter:** [`stereoforge.temporal_wip_adapters.WipDummyMultiPairStereoRingAdapter`](../python/stereoforge/temporal_wip_adapters.py)
- **Status String:** `"WIP_DUMMY_CURRENT_PAIR_CENSUS_ONLY_NO_HISTORICAL_IMAGE_RING"` (`is_wip_dummy() == true`)
- **Current Dummy Behavior:**
  - Tracks a bounded ring buffer (`capacity = 4`) of historical frame metadata/pairs and delegates execution to `DaxaTemporal::process(input, pose, use_completed)` with `temporal_stereo_research = true`.
  - Native GPU `P3` currently re-searches the **current frame's** full-resolution Census volume around the pose-warped historical prior $d_{\text{pred}} = f_x B / z_{\text{warp}} + \Delta c_x$ in $[d_{\text{pred}}-2, d_{\text{pred}}+2]$ with left-right consistency verification. It does **not** yet retain previous rectified image pairs in GPU memory to aggregate matching costs across multiple historical stereo exposures.
- **Required for Production Promotion:**
  - Allocate a bounded GPU ring buffer of historical rectified luma/Census pairs in `DaxaTemporal::Slot`, warp disparity hypotheses into historical left/right views using `transforms`, and combine multi-frame photo/Census evidence before L-R verification.

---

## 3. Native GPU 2× Depth Super-Resolution Grid (`P4`)

- **C++ Adapter:** [`stereoforge::WipDummyGpuSuperRes2xAdapter`](../include/stereoforge/daxa_temporal_wip_adapters.hpp)
- **Python Adapter:** [`stereoforge.temporal_wip_adapters.WipDummyGpuSuperRes2xAdapter`](../python/stereoforge/temporal_wip_adapters.py)
- **Status String:** `"WIP_DUMMY_HOST_PROTOTYPE_ONLY_NO_GPU_2X_SHADER"` (`is_wip_dummy() == true`)
- **Current Dummy Behavior:**
  - In C++, `WipDummyGpuSuperRes2xAdapter::wrap(frame)` returns a `WipDummyGpuSuperRes2xFrame` carrying the 1× `DaxaTemporal::Frame` alongside the target 2× grid dimensions (`2 * width`, `2 * height`).
  - In Python, `WipDummyGpuSuperRes2xAdapter::reconstruct_2x(...)` wraps the CPU/Python reference implementation (`reconstruct_p4_superres_2x` in [`benchmarks/temporal_supersampling_v12.py`](../benchmarks/temporal_supersampling_v12.py)) and tags the returned dictionary with `is_wip_dummy = True` and `status = "WIP_DUMMY_HOST_PROTOTYPE_ONLY_NO_GPU_2X_SHADER"`.
- **Required for Production Promotion:**
  - Port the pose-aligned subpixel 2× grid splat/gather and joint-luma bilateral solver into a dedicated Daxa compute shader stage with explicit uncertainty and provenance sidecars at $2W \times 2H$ resolution.
