# StereoForge: adaptive ROS 2 capture modes, light-aware policy (design + CPU prototype)

**Status:** only the pure CPU scene-light sampler and mode-decision policy are implemented in
'include/stereoforge/adaptive_camera_policy.hpp'. The native ROS 2 publisher has NOT been made
hot-swappable, and it does NOT yet publish CameraInfo. The capture benchmark on this branch
accepts --eye-width, --height and --fps to prepare separate physically recorded HD/FHD fixtures.

## Why HDR is not an HD720 device control

The ZED 2/ZED 2i camera exposes HD1080 at 15/30 Hz (cropped readout) and HD720 at
15/30/60 Hz (2x2 binned sensor readout). Exposure, gain, gamma and white balance are
supported in ZED Open Capture; a ZED 2 native hardware-HDR switch is NOT documented.
The 'enable_hdr' parameter in Stereolabs' ZED X One 4K ROS documentation is not
evidence for HDR on ZED 2. Do not silently add a hardware 'hdr=true' option.

- **Low light**: use binned HD720 and auto exposure/gain; consider 720/30 for
  stationary scenes, allowing a potentially longer shutter, but require depth
  quality measurements and a motion-blur limit. For navigation under motion,
  keep 720/60 and bound shutter time; accept higher gain/noise or use visible light.
- **High contrast**: highlight-aware exposure, synchronized controls across both
  cameras, and optional local tone mapping for **RGB preview**. Tone mapping
  cannot restore sensor-clipped information and must not silently modify the
  stereo matching input. Exposure bracketing/multi-frame HDR is an experimental
  static-scene-only option requiring motion registration, left/right exposure
  matching, scene-validity masks and separate image/depth provenance.
- **No usable texture / darkness**: do not manufacture metric depth; publish
  confidence loss and rely on alternate sensing/visible illumination.
- Gamma, denoising and gain need A/B assessment: cosmetics can reduce local
  Census contrast or introduce false correspondences.

Sources:
- https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/camera-controls
- https://docs.stereolabs.com/docs/products/cameras/zed/specifications
- https://stereolabs.github.io/zed-open-capture/d6/d2a/classsl__oc_1_1video_1_1_video_capture.html
- https://docs.stereolabs.com/docs/integrations/ros-2/zed-monocular-node

## Pure CPU policy: current experimental behavior

'AdaptiveCameraModePolicy' accepts a 48x27 luminance thumbnail from **both** eyes,
plus task priority, optional motion score and host-read gain percent:

| Scene/task | Desired camera | Image intent |
| --- | --- | --- |
| Normal, precision requested, little motion | HD1080 30 Hz | Standard |
| Normal, balanced | HD720 60 Hz | Standard |
| Low light, stationary | HD720 30 Hz | Low-light denoise (intent only) |
| Low light, moving/navigation | HD720 60 Hz | Low-light denoise (intent only) |
| High contrast | Keep current hardware mode | Highlight-aware preview (intent only) |
| Insufficient evidence | Keep current hardware mode | Standard |

Thresholds are **provisional**. The sampler uses per-eye Y, dark/saturated fraction,
gradient texture and a possible gain indication. Auto exposure can hide physical
low light, and a dark object can be misclassified. Add illumination and readback
ground truth before enabling automatic hardware transitions.

Mode selection has a preliminary 4-second stable-demand requirement and a
30-second hardware residence interval. Both are tunable **engineering defaults**,
not validated cutoffs. 'safe_to_interrupt_capture' must be affirmatively
granted by the robot/safety supervisor; software changes may be faster,
but no camera restart is automatic. 'observe' produces a RECOMMENDATION, not a
side effect. 'commit' happens only after a complete successful mode handoff;
'abort' leaves the former profile active in policy state and applies backoff.

## Existing ROS driver blockers (reviewed at main c246a83)

- 'apps/ros2_depth_publisher.cpp' constructs NativeCapture, DaxaFoundation,
  DaxaDepthPipeline, optional refinement, exporter, output image buffers and
  rectification **once** at startup.
- The rclcpp node is created with 'start_parameter_services(false)' and
  'start_parameter_event_publisher(false)'. It has no executor processing
  requests. No CameraInfo publishers currently exist.
- The existing manifest is bound to one capture mode, one original/rectified
  calibration identity, a source-width/height check and a recorded prototype.
  This prevents accidentally mixing geometry across modes. Never bypass it.
- The serial fallback override currently allowlists only the HD calibration
  identity for the connected ZED2; explicitly validate/rework this before FHD.
- 'benchmarks/multiresolution_fixture.py' creates **downscaled** calibration
  from a donor. It cannot create real FHD photons from an HD720 recording.
- A ZED 2 camera resolution/FPS change requires close/reopen. The Daxa GPU
  pipeline must drain old leases before replacing its shape-bound buffers.

## Runtime ROS driver: target architecture

A single stable ROS 2 node manages 'CameraSession' objects, each immutable
with regard to sensor mode, calibration, GPU buffers and source identities.
The control service/parameter callback validates a request and places it in
a separate control queue; it must **not** synchronously close the camera or
wait for the GPU inside a ROS executor callback.

    ACTIVE(epoch N)
      -> QUIESCING (stop accepting frames; notify downstream)
      -> DRAINING (wait/retire all GPU output leases and in-flight publishers)
      -> CLOSING_OLD_CAPTURE
      -> OPENING_NEW_CAPTURE
      -> BINDING_MODE_CALIBRATION (factory HD vs FHD + validated rectification)
      -> ALLOCATING_AND_WARMING_GPU (fresh geometry and metadata)
      -> VERIFYING_FIRST_CAPTURE (source, serial, timestamps, clock, stereo pair)
      -> ACTIVE(epoch N+1) (atomic publish of image and its CameraInfo)

On failure: attempt rollback to the previous mode with a **new** generation
and fresh calibration; if that also fails, publish FAULT and no stale frame.
Record reason, downtime, lost frames, retries, GPU retire waits and calibration
SHA. Do not claim old-camera continuity across a physical restart.

**ROS 2 API proposal** (not yet implemented):
- '/stereoforge/set_profile': an explicit request such as 'auto',
  'navigation', 'inspection', 'low_light' or 'manual' with expiry and priority.
  A service/action is preferable to a mutable low-level collection of
  independent width/FPS parameters.
- '/stereoforge/camera_mode/status': current/requested profile,
  epoch, light class, calibration SHA, actual sensor FPS, exposure/gain
  readback confidence, transition and fault state.
- '/stereoforge/left/image_rect' and '/stereoforge/left/camera_info'
  publish synchronized image and sensor_msgs/msg/CameraInfo on every frame.
  Add the right pair when right-eye images are published. The existing
  '/stereoforge/depth' is left-rectified axial Z and uses the same left
  projection geometry. Preserve existing depth and RGB topics as aliases
  during migration, without manufacturing a right-eye RGB output.

For CameraInfo use the **active mode's** raw per-eye intrinsics K, distortion
D, rectification R and rectified projection P (right P[0,3] = -fx*baseline).
For a rectified image, image_geometry consumers use P for projection. Use
the actual calibration for HD vs FHD, NOT a guessed 1.5x scaling of the HD
factory matrix. Width, height, Header.frame_id and acquisition timestamp
must match their image. Keep the physical TF relation fixed and use optical
left/right frame IDs. Do not duplicate physical sensor binning in CameraInfo
ROI/binning if the mode-specific calibrated image shape already accounts for
it. A separate depth processing grid requires its OWN calibrated projection.

**Atomic pairing/generation gate:** no image from epoch N may be published
with CameraInfo of epoch N+1. Stamp the corresponding mode and calibration hash
on the completed frame lease; publish pair from the SAME lease. Do not rely
only on a latched CameraInfo published at startup.

A reconfiguration pause may interrupt depth consumers, so external navigation
can veto it. Software-only refinement/rate switching should remain the normal
high-frequency adaptation path, while hardware mode changes happen rarely.

## Evaluation before live automatic switching

1. Acquire matched real HD720/60, HD720/30 and HD1080/30 recordings from
   the same ZED 2 with explicit mode-specific calibration, exposure/gain
   readback, manual lighting labels, target distance and motion annotations.
2. Evaluate independent truth **within overlapping FOV**, absolute 3D error
   in millimetres, correct-2px/full-resolution-scaled angular error,
   false matches, completeness, thin edges, low-texture, clipped highlights,
   motion blur, frame age and stereo consistency. Do not compare equal
   disparity-pixel thresholds blindly: focal length differs between modes.
3. 50+ automatic/manual mode cycles, dark-bright flicker and motion adversarial
   tests; confirm no mixed calibration, no stale epoch, monotonic timestamps
   per clock segment, no frame ID alias, no GPU lifetime leaks or deadlock.
   Measure switch pause p50/p95/p99; no assumed 'instant' transition.
4. ABBA fixed-scene comparisons with camera auto exposure ON/OFF and equal
   allowable shutter blur. Include visual-only tone mapping and (optional)
   static exposure-bracket fusion as separate experimental arms.
5. Run CMake 'adaptive_camera_policy_cpu_contracts' before any GPU work.
   These contracts only qualify mode-selection logic, not actual ROS switching.
