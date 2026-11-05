# Native stereo VO and motion-aware reconstruction

The opt-in native libraries run on the resident observation's shared Daxa device.
ROS profiles, physical accuracy, dynamic-scene robustness and live throughput
remain unqualified. Current-frame stereo is the production default.

## Stereo tracking and world poses

`DaxaStereoVo` detects tiled corners (768 by default), tracks patches
coarse-to-fine with mean-subtracted Lucas–Kanade and forward/backward validation,
and exports only bounded track records to the native pose solver. It owns the
previous left image/features and retains the producer through their GPU fence.
Metric landmarks require measured stereo provenance, validity, zero age and the
exact current source ID at every depth interpolation tap. A depth jump above
1% rejects sampling. Rectified axial depth is unprojected with its intrinsics;
`R1` transpose converts it into original left-camera rig coordinates.

`estimate_stereo_motion` uses bounded deterministic RANSAC, weighted rigid
alignment and optional current-bearing refinement. Metric scale comes from
previous stereo landmarks; current bearing directions avoid fitting axial noise
as camera rotation. A plane can constrain this pose; a line cannot. Default
admission requires 24 inliers and 65% consensus.

`DaxaStereoVo::process` returns `StereoVoFrame` with owned current and optional
previous `StereoFrameIdentity`. Frame zero is valid. Identities preserve device,
original/rectified calibrations, an administrative rig-session label, separate
acquisition/ingress timestamps and clocks, and optional clock-mapping uncertainty.
Empty session labels remain unverified; unknown uncertainty remains absent.

`StereoTrajectory::process` accumulates
`world_from_rig = previous_world_from_rig * inverse(current_from_previous)`.
Each noncopyable trajectory has a process-local gauge identity and separate
epochs. Loss, broken source chains, excessive ingress gaps, changed
calibration/session/clock, reversed acquisition times and reset break the world
chain. Anchors define an arbitrary origin; lost/mismatched frames publish no
pose. Recovery cannot reconnect to an old epoch. Admission checks bounded,
unique inliers, residuals, rigid transforms, positive local covariance and motion
limits.

Covariance is the latest **conditional local fit**, not accumulated world
uncertainty. The tracker currently assumes 5 mm isotropic landmark-difference
noise. Correlated stereo steps are not assumed independent. Native covariance
and temporal qualification flags remain false. Stability and synthetic controls
do not qualify accuracy, timing or moving-object rejection. A dominant moving
object can be indistinguishable from camera motion.

## Motion-model and image-support guard

The default `DaxaStereoVoConfig::guard` checks primary inlier support in both
images: at least six cells of a 4x3 grid and 45% span along each image axis. Local
support returns `InsufficientImageSupport`. This is a conservative image-support
criterion; it is distinct from the metric solver's geometric rank test.

A bounded secondary RANSAC search examines primary residual outliers. The default
requires at least 12 coherent tracks and 10% of all tracks, retaining the same
metric noise and degeneracy gates. Secondary support is absolute, so unrelated
outliers cannot hide a qualifying minority behind a majority requirement. The
primary camera estimator retains its 65% majority rule. A supported second model
separated by 20 mm or 0.01 rad returns `AmbiguousMotion`: the observations alone
do not identify which rigid body is static.

Rejected numeric pose/covariance/inliers are cleared. Diagnostics retain primary
and secondary counts, model separation, image span and occupied cells. Trajectory
admission also checks the guard report, resets the epoch and withholds a pose;
temporal processing uses current stereo. Recovery starts a new world chain and
excludes pre-failure contributors. `guard.enabled=false` is an explicit ablation
and reports `NotEvaluated`.

The rendered controls include stationary/moving cameras with a dominant moving
panel, camera-only/static scenes, exact fresh-stereo fallback and recovery.
A coherent minority mixed with unrelated residual noise is checked separately.
An infinite moving plane and the opposite camera translation generate identical
stereo images in the identifiability control. A consistent single model cannot
certify static-world camera motion in that case. Native temporal/world covariance
qualification remains false, and default fusion remains disabled. Broader dynamic
scenes and reliable static-world evidence still require qualification.

A longer moving-panel development sequence exposed a remaining failure after an
initial two-model rejection: later moving frames can lack enough secondary
support and resume research history. Five sampled fills put the old 1.28 m
foreground onto newly visible 1.6 m wall support. The earlier current-image ZNCC
check and disocclusion option both admitted these false fills.

Multi-pair confirmation now additionally requires current **other-eye measured
metric evidence**. Every interpolation tap must have support, measured stereo
provenance `IndependentStereo` (1), the exact current source ID, zero age, a valid depth consistent with
the candidate, and at most 1% depth spread. It uses the same metric consistency
and continuity thresholds as cached evidence. Correlated left-aggregated current
references (provenance 5) cannot satisfy independence. Historical cached evidence
retains its existing 1-or-5 policy and cannot bypass the independent current gate.
Current/cached validation shares
one GPU helper; existing buffers and the packed source token preserve the 512-byte
parameter layout. No full-frame transfer or new GPU allocation is required.

These checks reject the five retained witnesses and preserve fresh measured
output when history is rejected. Fault controls keep camera tracking and cache
healthy while corrupting only current other-eye evidence. Large source IDs test
both ID words; a healthy control still admits history. Accuracy/coverage costs
require measurement before promotion. The guard does not establish subsequent
static-world observability: dynamic scenes, uncertainty, clocks and physical/live
operation remain unqualified.

## Temporal admission and four-pair history

`DaxaTemporal::process_odometry` verifies pose/source frame IDs, acquisition and
ingress times/clocks, device and both calibration identities. Unqualified native
poses use fresh stereo by default. `odometry_research=true` explicitly enables
research fusion. Trajectory/epoch changes, anchors, pose loss and source mismatch
suppress historical contributions. Rejected duplicate/foreign inputs and
backpressured attempts preserve accepted history. Transitions commit with a
successful submission. Legacy known-pose submissions cannot reuse a VO world's
history. Canonical RGB uses its admitted calibrated grid association; raw inputs
retain complete remap/model validation.

Qualified external poses additionally enforce acquisition-age upper bounds,
including clock uncertainties. Accumulated history retains the minimum possible
acquisition time across contributors, using an overflow-safe lower-bound
envelope. It does not substitute ingress age for acquisition age.

`multi_pair_research=true` requires `odometry_research=true`. The
`DaxaStereoHistory` component owns up to four complete rectified RGB pairs and
both-eye depth, validity, provenance, source IDs and ages. Seven planes are
packed into one GPU allocation per pair; views expose explicit byte offsets.
The history uses 64 bytes/pixel/pair: four 720p pairs require 225 MiB; four 1080p
pairs require 506.25 MiB. Temporal working/output buffers are additional. Capacity,
a maximum 200 ms age, and a configurable memory budget are enforced.

Producer leases survive each GPU copy fence, then nonblocking collection releases
them. Four cached pairs can therefore coexist with the two-slot canonical
producer. Exported cache leases prevent overwrite; consumers record submitted
read tickets before dropping leases or retain them through read completion.
Capacity pressure skips a copy. Logical reset and ring destruction cannot free
exported buffers. Partially submitted copy failures poison the history and retain
resources until their recorded fences complete.

An application may inject a shared history in the temporal constructor. Temporal
is its writer; other consumers may retain/select resident evidence. Context and
calibration ownership must match. Selection also obeys Temporal's age bound when
the shared ring allows a wider window. Research history reports ingress age and
leaves acquisition-age qualification absent when its clock uncertainty is unknown.

The fuse shader projects historical candidates into each cached pair. It requires
current L/R photometric agreement, measured zero-age/current-source cached metric
taps in both eyes, depth continuity/occlusion agreement, and historical L/R
photometric agreement. Two cached votes are required when at least two pairs are
available; one is required for the first pair. `pair_checks` and `pair_rejections`
count candidate verification. This gates reconstructed contributions and does
not increase measured completeness. Raw measured stereo remains separate.

The initial research path waits its specific temporal output fence before
reusing the four cache slots, retires internal cache-reader leases, then queues
the current pair copy. It adds no device-wide wait or image readback. This
serialization and its GPU/host latency require measurement before live promotion.

## Verification and remaining work

Build with the existing native Daxa depth configuration:

```sh
ctest --test-dir build/ros-daxa-native \
  -R 'stereo_motion|stereo_trajectory|stereo_vo|stereo_history' --output-on-failure
```

CPU contracts cover metric transforms, coherent minority outliers, noise,
degeneracy, bounded inputs, world composition and epoch/source failures. Vulkan
contracts ray-render stereo images, compute native SGM depth, then run real GPU
VO, trajectory and research fusion. Ground-truth poses generate images and serve
as assertions only. Clock-admission tests explicitly inject external qualification
flags; native VO does not emit those flags.

Witnesses cover mapless/rotated geometry, texture loss/recovery, reset, expiry,
clock changes, duplicate/foreign frames, corrupted right images, producer reuse,
cache byte identity/lifetime/backpressure, cached-only ancestry/depth/continuity
and image corruption, and acceptance that requires pair indices two and three.
Acquisition controls include excessive separation, large/summed/varying
uncertainties and accumulated age. A compiled mutation restoring premature
history clearing fails the retry regression. Existing hotpath allocation and
retirement mutation checks retain their thresholds.

Remaining requirements include moving-foreground/observability qualification,
disocclusion and false-fill cohorts, calibrated uncertainty and clocks, separate
spatial completion, the production profiles/ROS/GPU cloud path, fixed independent
scene groups and physical/live acceptance. No research setting is promoted by
these controls.
