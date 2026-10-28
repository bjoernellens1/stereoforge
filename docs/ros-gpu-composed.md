# Experimental ROS 2 same-process GPU handoff

Configure with `STEREOFORGE_ROS2=ON`, `STEREOFORGE_ROS2_PUBLISHER=ON`, and `STEREOFORGE_ROS2_GPU_COMPOSED=ON`; the ordinary build has no ROS dependency. Run `stereoforge_ros2_gpu_composed --help` for the bounded camera/replay arguments:

```text
--source camera|replay --manifest FILE --clock-map FILE --frame-id ID --trace FILE
[--replay DIR --topic TOPIC --duration-seconds N --max-frames N
 --disparities N --device-id N --validation]
```

For replay, `--replay` names a packed capture directory; the trace path must not already exist. The defaults are 30 seconds, at most 1,000 admitted/completed frames, D128, and `/stereoforge/gpu_frame`. The clock map and calibration manifest use the same contracts as `stereoforge_ros2_depth_publisher`.

The executable composes a publisher and subscriber in one process and one single-threaded ROS executor, sharing the same Daxa/Vulkan context and submission thread. Its ROS adapter sends an owned observation lease through the intra-process path. The demonstration subscriber makes a same-device GPU-to-GPU copy of axial depth and rectified left RGBA into a bounded two-slot ring; it retains the output until the copy completes. It does not export image pixels to the CPU. The producer's validity, confidence, age, and provenance remain available only on the source observation and are not copied to these two destination buffers; the copied depth alone must not be treated as a fully qualified robotics output.

The ROS wire representation is a `std_msgs/msg/Header` containing only the timestamp and frame ID. A remote subscriber does not receive pixels, Vulkan buffers, or handles. Converting a remote header into a GPU frame is rejected. This is not cross-process zero-copy; that needs external-memory and synchronization support and a real handle transport protocol.

The bounded JSON trace includes capture and enqueue times, publication and callback times, observed GPU copy completion, queue drops and the supplied clock-mapping error. The camera clock mapping has not been independently verified: the trace marks it unverified and observation age unavailable, so mapped timestamps do not establish frame age. GPU completion is observed by polling, not a hardware timestamp. The reported producer and consumer bytes are partial requested GPU buffer sizes; capture, IMU, ROS and driver memory are unavailable and excluded. Replay backpressures while both consumer slots are occupied; a live source can drop frames. Investigate the ROS publication-to-callback interval, observed GPU completion delay and capture queue drops separately. Existing CPU `sensor_msgs/Image` publishing remains a separate host-export path.

The embedded subscriber demonstrates a local GPU copy only. It is not a validated downstream robotics consumer, and it does not copy the validity, confidence, age or provenance sidecars.

This path has not been qualified for GPU correctness, remote transfers, live camera operation, or sustained 60 Hz. Validate data, queue ordering, lifetime, and Vulkan synchronization on the target device before relying on it.

## Integrating a same-process subscriber

With the options above, link your in-tree CMake target against `stereoforge_daxa_gpu_consumer`; its public dependency supplies the Daxa depth interfaces and headers. Also use `ament_target_dependencies(your_target rclcpp std_msgs)`. Include `<stereoforge/ros_gpu_frame.hpp>` and `<stereoforge/daxa_gpu_consumer.hpp>`. The producer and subscriber must be built into the **same process** and use the same `DaxaFoundation`; this target does not provide ROS dynamic components or interprocess GPU transport.

The following illustrates the callback and retirement portions of an application. Create `consumer` once from the shared foundation and a real producer observation, create the node with `NodeOptions().use_intra_process_comms(true)`, and run the producer, subscriber executor, consumer, and all frame destruction on the same owner thread. `slots` must outlive the subscription; call `retire()` regularly on that thread:

```cpp
using Adapted = rclcpp::TypeAdapter<stereoforge::RosGpuFrame, std_msgs::msg::Header>;
std::array<stereoforge::DaxaGpuConsumer::Frame, 2> slots;

// consumer was constructed with (foundation, prototype_observation, 2).
auto sub = node->create_subscription<Adapted>(
    "/stereoforge/gpu_frame", rclcpp::QoS(rclcpp::KeepLast(1)).durability_volatile().best_effort(),
    [&](std::shared_ptr<const stereoforge::RosGpuFrame> frame) {
        auto slot = std::find_if(slots.begin(), slots.end(),
                                 [](auto const& s) { return !s; });
        if (slot == slots.end() || !consumer.has_capacity()) {
            ++dropped; // Or backpressure replay before accepting another frame.
            return;
        }
        *slot = consumer.enqueue(frame->observation());
    });
auto retire = [&] {
    for (auto& slot : slots) {
        if (slot && slot.ready()) slot = {};
    }
    consumer.collect();
};
```

The callback does not catch enqueue failures as ordinary drops. The consumer preserves the producer observation until its copy completes. If an application submits additional work using a destination buffer, the work must be submitted on the shared Daxa main queue, then registered with `slot.retain_until(submission_ticket)` before releasing that slot. Retain the slot until all downstream GPU uses complete; if submission outcome is uncertain, keep the lease and fail closed. A copy-complete `ready()` check alone does not prove additional downstream work has completed. On shutdown stop publication, release queued ROS messages/subscriptions, drain consumer and downstream work, release all slots on the owner thread, then tear down the foundation. The example's copy does not include validity, confidence, age, or provenance sidecars; geometric consumers need those contracts separately.
