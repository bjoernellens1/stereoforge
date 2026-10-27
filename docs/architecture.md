# Execution and geometry contracts

Optical coordinates use +X right, +Y down, +Z forward. Translations/depth use metres.
`Calibration.right_to_left` is the raw right camera's rigid pose in the raw left
camera/rig frame. Calibration identity hashes the canonical serialized parameters.
Rectification creates a separate calibration identity and stores its rotations and
projection matrices. `depth_to_raw` reprojects measured geometry with visibility
checks before any raw-RGB view synthesis; it never treats rectified depth as raw.

Host `StereoFrame` owns uint8 RGB arrays and records timestamp, clock domain,
observation/frame identity, and raw sensor samples. `DepthFrame` owns disparity,
axial depth, confidence score in [0,1], validity, provenance, source observation ID,
and observation age in nanoseconds. Valid depths are finite and positive. Invalid
outputs are masked and generated depth holes use NaN.

Temporal fusion is conservative: confidence is not an independent-observation
probability. Same-source estimates are not re-averaged; FUSED ancestry survives
reprojection and is excluded from further averaging because a single source ID
cannot represent an arbitrary evidence set. Fresh measurements supersede conflicting
predictions. Missing motion, stale prediction, clock/calibration discontinuity and
backwards time prevent temporal reuse. History is bounded and snapshots are owned
independently of consumer-mutable outputs.

The adaptive scheduler only narrows reliable, fresh, finite disparity priors;
uncertain regions and periodic refresh use full search. Candidate work and dense
aggregation work are counted separately. It does not currently compact disparity
storage or guarantee proportional GPU savings.

Custom shaders use explicit entry points and uint32 scalar buffers, row-major
pixel order, disparity fastest. Census uses 24 neighbours in a 5x5 window. Borders
and out-of-support hypotheses carry invalid cost10000. Four paths are horizontal
and vertical; eight add diagonals. CPU scalar recurrence and shared Python/C++
shaders have exact integer comparisons and floating-point output tolerances.

VulkanStereo's Python adapter has one in-flight slot and rejects overwrite or
foreign/consumed submission IDs. Native APIs use retained leases and bounded slots;
resources stay alive through completion even if the facade is destroyed. Readback
is explicit. A host completion token is not an imported cross-queue semaphore.

View synthesis separates strict geometry from preview approximation. Both use
original lens canvases. Virtual-camera projection, pose, requested canvas FOV,
depth convention, visibility, and color/depth masks are explicit. Cylindrical views
can include rays behind the virtual +Z axis within their canvas; perspective views
cannot. Translation never fabricates disoccluded surfaces. Raw sensor data and a
sync flag do not imply calibrated VIO or valid translational motion.
