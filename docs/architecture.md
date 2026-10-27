# Execution and geometry contracts

Production architecture decision (2026-10-27): target C++20 + Daxa TaskGraph +
Slang, with Daxa as the single Vulkan device/resource/graph owner for the core
classical/temporal/view pipeline. This is a planned migration gated by Linux dependency/capability/license checks, the
upstream external-resource-view regression, and a matched resident proof of
concept; Daxa is not yet adopted or qualified. Keep the working Slang RHI path
frozen as comparison/migration evidence, not underneath Daxa. Reusable Slang
operations stay independent of Daxa bindings; curated manually fused/unfused
variants use bounded offline benchmarks and reviewed configuration. There is no
custom task scheduler, arbitrary fusion compiler or general auto-tuner. SlangPy
is optional research; ncnn Vulkan is optional for one validated model, with IREE
reserved for demonstrated later need. Optional ncnn may use a separate context
behind measured owned-copy/completion boundaries; cross-context buffer aliases
and zero-copy claims require proven interoperability. Geometry, metadata, completion and
output contracts below remain binding. See the [production implementation plan](superpowers/plans/2026-10-27-daxa-slang-production.md).

Optical coordinates use +X right, +Y down, +Z forward. Translations/depth use metres.
`Calibration.right_to_left` is the unrectified right camera's rigid pose in the
unrectified left camera/rig frame. Legacy image-geometry names such as
`depth_to_raw` mean the unrectified post-ISP source geometry, not Bayer sensor
raw data. Calibration identity hashes the canonical serialized parameters.
Rectification creates a separate calibration identity and stores its rotations and
projection matrices. `depth_to_raw` reprojects measured geometry with visibility
checks before source-RGB view synthesis; rectified and source geometry remain
distinct.

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
unrectified post-ISP lens canvases. Virtual-camera projection, pose, requested canvas FOV,
depth convention, visibility, and color/depth masks are explicit. Cylindrical views
can include rays behind the virtual +Z axis within their canvas; perspective views
cannot. Translation never fabricates disoccluded surfaces. Raw sensor data and a
sync flag do not imply calibrated VIO or valid translational motion.
