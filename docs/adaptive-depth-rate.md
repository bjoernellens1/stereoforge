# Adaptive depth update rate (experimental)

The native StereoForge ROS 2 publisher now separates **capture FPS** from **GPU stereo-depth admission**. The camera can continue at 60 Hz while the Daxa depth graph processes only selected fresh frames. The current implementation is CPU-only at the admission boundary and does **not** change matching quality, disparity search, or camera exposure settings.

## CLI

`--depth-rate 5|10|20|30|60|adaptive` selects depth processing rate. The default is **adaptive**. Use `--depth-rate 60` to reproduce the previous process-every-frame policy.

`--capture-fps 15|30|60` (legacy alias `--fps`) selects acquisition rate, **not** processing rate. A 720p ZED 2 supports the nominal 60 Hz setup. Capture-rate capabilities depend on the selected camera model and resolution; actual delivered timestamps, drops and hardware support must be checked independently. For capture below 60 Hz, set `--max-depth-fps` to a supported rate not exceeding acquisition.

Adaptive bounds:

- `--min-depth-fps` (default 5) and `--max-depth-fps` (default 60) bound the adaptive requested rate.
- `--consumer-min-depth-fps` (default 5) specifies a **static** minimum requested by the downstream application. Dynamic ROS subscription-rate negotiation is **not** implemented.
- `--max-depth-age-ms` (default 250) bounds elapsed *source capture time between accepted depth frames* where frames and processing capacity are available. It does not bound camera exposure-to-ROS arrival time and does not make old observations fresh. Policies whose requested interval exceeds the age limit are rejected.
- `--downshift-ms` (default 3000) requires sustained quiet before each one-step rate decrease.

All FPS bounds must be one of 5, 10, 20, 30, 60. Impossible floor/ceiling/freshness combinations fail at startup. If the camera disconnects, lacks usable frames, or all GPU slots are busy, the configured freshness limit cannot be guaranteed; these conditions must be handled by downstream consumers.

## Example

Add the following options to the existing ROS publisher command (which also requires `--source camera`, `--manifest`, `--clock-map`, `--frame-id`, `--run-id`, and `--trace`):

```sh
--capture-fps 60 --depth-rate adaptive \
--min-depth-fps 5 --max-depth-fps 60 \
--consumer-min-depth-fps 10 --max-depth-age-ms 250 \
--downshift-ms 3000 --rate-report depth-rate.json
```

For predictable 10 FPS stereo while keeping the camera at 60 FPS:

```sh
--capture-fps 60 --depth-rate 10 --rate-report depth-rate.json
```

The `--mode depth-rgb` output currently emits **both** rectified RGB and depth only on depth-admitted frames; it is a synchronized depth/RGB mode, not an independent 60 Hz RGB stream. `--mode rgb` is not throttled by the depth-rate controller. An independent RGB stream alongside adaptive depth requires a separate producer/export path and remains future work.

## Adaptive policy

`YuyvSceneMonitor` samples a fixed 48×27 luminance thumbnail from **each** camera eye directly from the packed YUYV capture lease. The CPU calculates mean-offset-compensated change against the preceding captured frame and half-weighted change against the last **depth-admitted** frame. It requires no GPU upload, heap allocation, neural model, full-frame optical flow, or host readback.

`DepthRateController` quantizes this activity to 5, 10, 20, 30 or 60 Hz:

1. Start conservatively at 60 Hz until a valid pair of camera samples is available.
2. **Increase** the target on the next camera frame when motion/scene change warrants it, including a full-rate fail-open when evidence is invalid or missing.
3. **Decrease** by at most one rate tier after each continuous `downshift-ms` quiet period.
4. Apply configured FPS floors and ceilings and the maximum accepted-observation interval.
5. Select the earliest fresh frame meeting the time-based due threshold, using a half-capture-period tolerance for 60 Hz timestamp jitter.
6. On a full GPU ring, **reject the due frame without waiting**; retry with the next fresh capture. Never count intentional skips as capture drops.

The thresholds are *initial empirical-free heuristics*, not calibrated motion, visual odometry, or task-aware perception. A textureless surface can move without triggering an image-change detector; auto-exposure and dynamic lighting can produce false triggers. **For safety-critical navigation/manipulation, explicitly set a conservative consumer minimum or choose fixed 30/60 FPS.** IMU/VIO is intentionally not a downshift condition because camera-to-IMU timing/extrinsics in StereoForge are not yet calibrated.

The camera and IMU acquisition run at their configured sensor rates, independently of depth admission. The active stereo estimator, validity/confidence/provenance rules, and ROS timestamp semantics are unchanged. Skipped captures are not republished or timestamped as new depth.

## Telemetry and proof boundaries

Standard ROS `stereoforge.ros_delivery_event.v1` traces remain structurally unchanged. The implementation settings in the trace start event include the requested rate policy. The optional `--rate-report FILE` writes a **separate JSON file after processing** (`stereoforge.depth_rate_report.v1`) with:

- `observed`: fresh source frames evaluated by the depth scheduler during the measured window.
- `admitted`: depth frames accepted into the Daxa producer.
- `intentional_skips`: source frames deliberately not sent to the GPU because the rate deadline had not elapsed.
- `slot_rejections`: due source frames rejected because the bounded producer ring was full.
- `upshifts`, `downshifts`, `unavailable_motion`, `final_target_fps`.
- The existing published depth/RGB counts, capture FPS and policy configuration.

The rate report is **not** a GPU-performance measurement or proof of exposure-to-consumer age. Source timestamp intervals are used for reproducible selection, but absolute camera-to-host clock mapping remains unqualified.

## Tests and remaining validation

Portable CPU contracts: `cmake --preset cpu && cmake --build --preset cpu --target depth_rate_controller_contracts && ctest --preset cpu -R depth_rate_controller_contracts --output-on-failure`. The tests cover all five rates, adaptive hysteresis and immediate upshift, missing-evidence fallback, invalid and non-monotonic timestamps, ring rejections, camera exposure shift, and image changes.

Hardware qualification remains open. Run matched 60 Hz vs fixed 30/20/10/5 Hz vs adaptive camera trials on identical source/replay input where possible. Collect unique-frame counts, intentional skips, capture queue drops, ring rejects, depth arrival p50/p99/max, output age with a **validated** camera/host clock map, background GPU workload throughput and power, and depth quality on admitted frames. Fixed-rate lower delivery must not be mistaken for delivery failure against a 60 Hz delivery-policy target; the legacy 60 Hz policy is not automatically applicable.
