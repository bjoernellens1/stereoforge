# ROS 2 GPU-resident handoff and latency investigation

## Goal and boundary

Provide an opt-in same-process ROS 2 composed-node path for StereoForge depth and RGB GPU observations on the existing AMD/Vulkan/Daxa device. The ROS in-process message owns a Daxa observation lease; the pixels and sidecars remain in GPU resources. A metadata-only ROS wire representation never serializes pointers, GPU handles or claims that a remote process received pixels. Standard `sensor_msgs/Image` remains an explicit CPU-export alternative. Do not claim cross-process GPU transfer until independently implemented and validated.

## Lifetime and ordering

Each GPU frame contains source identity, capture stamp and clock provenance, geometry/calibration and the owning observation lease. A compatible consumer must share the Daxa device/context and submission owner thread, establish the appropriate GPU memory dependency, and retain its own frame lease until its consumer submission completes. Bound consumer in-flight work and report dropped or backpressured frames. Shutdown and failed/uncertain submission must not release a lease whose GPU use may still be pending. Metadata conversion from a remote wire message must never fabricate a GPU frame.

## Stages and evidence

1. Implement and test the adapter and bounded consumer owner without host pixel copies, including mismatched device, held leases, slot exhaustion and shutdown. Compile against real ROS Jazzy; keep readback strictly diagnostic.
2. Integrate an opt-in composed ROS producer/consumer app using the existing native camera/replay pipeline. Check actual intra-process handoff, metadata-only interprocess behavior and requested-output selection. Preserve provenance and mark unavailable timestamps honestly.
3. Obtain the separate performance review required by the repository policy before GPU validation. On target AMD, run GPU producer and consumer with synchronization validation, verify payloads against a known reference, test retained leases and consumer completion under pressure, and measure CPU enqueue, GPU work, callback, frame age, drops and memory separately.
4. Investigate first-publication outliers with repeated controlled CPU-only and GPU-present runs, distinguishing discovery, scheduling and first-use work. Do not infer a cause from one timing or hide the first frame.
5. Study an eventual interprocess path with capability-checked Vulkan external memory and synchronization plus OS file-descriptor transport, device identity, ownership, consumer acknowledgments and crash cleanup. Ordinary ROS/DDS messages containing integer file descriptors do not transfer GPU handles. Require separate end-to-end validation.

This plan does not qualify 720p/60 FPS, a live camera clock, ROS interprocess zero-copy, or quality versus the ZED SDK.
