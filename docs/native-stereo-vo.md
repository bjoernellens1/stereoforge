# Native stereo VO development stage

`DaxaStereoVo` consumes resident `DaxaObservationFrame` buffers on their shared
Daxa device. It detects one corner per image tile (768 by default), tracks patches
coarse-to-fine with mean-subtracted Lucas–Kanade refinement and a forward/backward
check, and reads back only bounded landmark/track records. It keeps its previous
image and features in its own GPU buffers, releasing the producer after the copy
and tracking fence. The current implementation waits synchronously for this fence.

Landmarks require current stereo validity and measured provenance (`Stereo` or
`StereoLeftAggregated`) at all depth interpolation taps. A depth jump above 1%
rejects sampling. Rectified axial depth is unprojected with its own intrinsics;
`R1` transpose converts it to original left-camera rig coordinates. Temporal,
spatial and neural estimates cannot seed these landmarks.

The native `estimate_stereo_motion` solver uses bounded deterministic RANSAC and
weighted rigid alignment. Its optional bearing refinement uses previous metric
stereo landmarks and current image directions, avoiding fitting current axial
noise as camera rotation. `current_from_previous` maps previous rig points into
the current rig; it is not `world_from_rig`. A plane can constrain this metric
pose; a line cannot. The default requires at least 24 inliers and 65% consensus.

The 6x6 covariance is a **conditional, uncalibrated local fit** in translation
then rotation order, using left perturbations. The GPU adapter currently assumes
5 mm isotropic landmark-difference noise. This is an explicit development
assumption, not a range-calibrated camera uncertainty model. The result always
labels covariance and acquisition-clock qualification false. Pose stability
alone does not establish accuracy. A coherently moving dominant object may be
indistinguishable from camera motion; this stage does not qualify dynamic fusion.

Results associate the relative estimate with both exact source frame IDs and
preserve current acquisition/ingress times and clock identities separately.
Duplicate/late inputs are rejected. Device/clock changes, explicit reset and
ingress gaps exceeding 200 ms establish a new tracking anchor. No acquisition
age claim is derived from an ingress gap. The producer observation must belong
to the original shared context, calibration owner and registered image grid.

Build with the existing native Daxa depth configuration. Run:

```sh
ctest --test-dir build/ros-daxa-native -R 'stereo_motion|stereo_vo' --output-on-failure
```

The CPU contracts cover metric rigid transforms, coherent minority outliers,
noise, degeneracy, bounded inputs and axial-error bearing refinement. The Vulkan
contract ray-renders a textured world plane for separate left/right camera
origins and moving/rotating poses, computes actual native stereo depth, and
feeds those outputs to VO. Ground-truth poses are used only for rendering and
assertions. It also checks texture loss, recovery, reset, history expiry,
clock changes, duplicate frames, corrupted right images, producer reuse, and
nonidentity rectification for both translation and rotation.

This opt-in library is not wired into ROS or temporal fusion. Remaining work
includes accumulated world poses with quality gating, conservative foreground
and observability qualification, verified timing/uncertainty, the four-pair GPU
history, fusion integration, real scene sequences and live latency/throughput
qualification. The existing current-frame stereo path remains the default.
