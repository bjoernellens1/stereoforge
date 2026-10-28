# Quality-first StereoForge / ZED SDK benchmark design

Revised 2026-10-28 from the user's SDK comparison proposal. This changes the
next-work order, not the Vulkan runtime or the existing output contracts.

## Decision and observed boundary

Build the common-input external quality benchmark before further Fast60
optimization or promotion. First establish full-resolution independent SGM on
the same exported frames as the SDK, then evaluate Fast60. Primary competitive
target: measured `NEURAL_LIGHT` quality at sustained 720p/60 on AMD; this remains
unproven. `NEURAL` and `NEURAL_PLUS` are additional reference profiles, not truth.
`PERFORMANCE` is a deprecated classical control when actually supported; record
unsupported modes explicitly, never substitute another mode under its name.

The [current evidence](../../fast60-refinement-evidence.md) establishes conditional
guided-kernel agreement, not quality parity: synthetic correct metric completeness
is about 54.4% guided versus 95.3% independent full-resolution SGM. Exploratory
coarse-plus-refinement timing is approximately 8 ms without View; repetitions are
incomplete. CPU full-D recovery reaches about 75.5% with more false accepts.
Development-only prior ablations raise guided completeness to about 69.1% at a
0.01 confidence floor, but thin completeness remains about 0.52%. No thresholds
or faster default are promoted by these experiments.

## Runtime and licensing boundary

Production remains C++20/Daxa/Slang/Vulkan, AMD first, without Python, CUDA, HIP,
ROCm or TensorRT dependencies. The proprietary SDK runs only in an optional,
isolated reference runner on an inventoried NVIDIA host with vendor-supported
Ubuntu. Its CUDA/TensorRT dependencies never enter core builds, CI or deployed
StereoForge. Check installer, model, sample and recording/export terms before
redistributing anything; keep SDK binaries, models, camera serials and private
recordings out of Git. SDK absence does not block portable evaluator/adapter
work or explicitly unqualified classical engineering releases; it blocks SDK
parity claims. Literature and vendor accuracy tables do not replace execution.

## Shared records and geometry

Use versioned `benchmarks/zed_comparison/` records with relative, hashed files.
Each immutable sequence binds SVO/source hashes, lossless original and rectified
stereo RGB, frame ordinal/source ID, capture and sensor timestamps with clock
domains, effective calibration and source file identity, dimensions/stride/color
conversion, settings, SDK/model/driver/device identities and StereoForge commit,
compiler/SPIR-V identities. Preserve post-ISP original geometry for wide synthesis.
SDK exports are not claimed byte-identical to the USB YUYV transport or Bayer data.
Never round-trip RGB through YUYV to fit the existing capture format.
Bind pixel-centre convention, camera/distortion model and stereo transforms to
rectified support masks. Derive geometric support from validated models/maps;
black RGB is not an invalidity test, and absent support is not an all-valid mask.

SVO calibration is resolved on the playback machine. Snapshot the resolved file
and both effective raw/rectified API calibration, self-calibration setting and
changes over playback. A canonical export defines frame/pixel/calibration hashes.
Direct pixel comparisons require these identities to agree across runs; changed
geometry creates a separate stratum and reruns StereoForge on that stratum's
exported pair. Do not silently warp SDK depth to manufacture identity. Run a
separate factory-versus-effective calibration ablation and measure vertical
epipolar residuals; this cannot explain failures on ideal analytic calibration.

Depth is lossless float32 metres with original invalid encodings retained and
explicit metric-valid masks. Declare the left rectified reference, coordinate
frame and axial-Z/range convention; test off-axis planes/XYZ conversion before
comparison. Store raw confidence and its direction; SDK low means trustworthy,
StereoForge high means trustworthy. Texture filtering is a separate setting,
not an invented retrievable per-pixel map. Missing fields remain unavailable.
Observed, predicted, historical and approximate pixels remain distinguishable.

## Three tracks

1. **A: same-frame algorithms.** Lossless SVO replay without real-time pacing,
   effective calibration export, then a native already-rectified RGB adapter
   bypassing capture decode and remap. Run full SGM D128 before Fast60 and all
   available SDK modes. Primary outputs are left-aligned depth/confidence/masks;
   independent-right results remain an auxiliary internal evaluation. SDK internal
   pixels/preprocessing are opaque; exported-pixel equality is not proof of them.
   Disable temporal stabilization/tracking/filling for the primary instantaneous
   lane where supported, recording actual effective settings or explicit failure.
2. **B: independent truth.** Analytic scenes plus independently surveyed planes
   at 0.3–8 m, 1–8-pixel thin geometry, edges, low/repeated textures, reflective/
   dark materials, 10–30 s static sequences and camera/object motion. Record reference
   geometry, registration and timestamp uncertainty. Sparse LiDAR only supplies
   support at observed samples; never invent dense edge truth. Fixed development
   and untouched held-out scene/sequence partitions prevent tuning on test data.
3. **C: sustained output.** Actual capture→GPU→consumer, at least 600 s nominal
   60 Hz, bounded 2–3 slots, original full-resolution RGB retained, fixed View
   measured separately. Report unique depth/RGB deliveries, drops, gaps, queue
   growth, memory, p50/p95/p99/max and observation age. Add ROS 2 `32FC1` metre
   depth and matching QoS/topic benchmarking after quality. ROS delivery is an
   additional gate; it cannot qualify missing GPU or physical accuracy gates.

## Evaluation and acceptance policy

Report relative depth MAE by distance, absolute error p95, coverage, correct
metric completeness, false accepted geometry, thin recall/boundary F1, static
jitter, ghosting and recovery. Report both intersection-of-valid error and
completeness over fixed independent truth support, including missing estimates.
Separate clean/mixed/occluded/out-of-range populations and per-region false
accepts with denominators; do not improve apparent error by hiding difficult pixels.
Compare error-versus-coverage and reliability curves, not equal confidence numbers.
Confidence is not called calibrated uncertainty before separate validation.

Provisional non-inferiority policy against executed `NEURAL_LIGHT`: completeness
loss at most 2 percentage points; relative MAE at most 1.10 times the reference;
no excess false accepted boundary rate. Apply separate thin, boundary and
near-field gates, not just an average dominated by planes. These are proposed
engineering margins, not Stereolabs tolerances. Freeze exact region
definitions, correctness tolerances, uncertainty-aware zero-error handling,
statistical decision procedure and margins on development data before held-out
evaluation; missing policy/data means unqualified, never pass by default. Record
paired scene/sequence uncertainty without treating correlated frames as independent.

For every actual calibration report `d=fx*b/Z+cx_offset` and supported absolute
disparity limits. D128 supports indices 0…127: representative fx528/b0.119 needs
about 209 px at 0.3 m, outside that profile. Publish both common supported-range
and full requested 0.3–8 m results; out-of-range failures remain visible. Evaluate
shifted/local near-field search beyond 128 separately rather than silently forcing
global D256 or reducing the benchmark's range.

Time cold preparation separately from warmed execution. Keep GPU stages, upload,
readback, synchronization, CPU work, file I/O and publication distinct. Resident
and matched host-output lanes must name their actual completion boundaries and
allocation/cache mode. SDK-internal stage timing may be unavailable: do not label
opaque `grab()` cost as isolated matching. Same-GPU NVIDIA Vulkan/SDK runs isolate
runtime comparisons; AMD versus NVIDIA results are platform operating points,
not algorithmic speedup ratios. Never compare cold neural/full readback against
warmed resident classical work. Temporal-off and temporal-on studies stay separate.
ROS image headers retain the camera coordinate frame and mapped capture stamp;
source observation IDs and publication timestamps travel in a companion trace.
Join publication/subscriber events with declared clock mappings to isolate
middleware delay. Header-to-arrival latency alone may include camera processing.

## Verified primary references (checked 2026-10-28)

- [Recording and playback](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/recording): raw unrectified post-ISP frames; derived outputs recomputed; calibration resolved from playback settings.
- [Calibration](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/camera-calibration): API self-calibration can differ from factory file, including raw parameters.
- [Depth modes](https://docs.stereolabs.com/docs/development/zed-sdk/modules/depth-sensing/depth-modes): neural profiles, deprecated legacy controls; published ZED X figures are context, not ZED 2 ground truth.
- [Depth settings](https://docs.stereolabs.com/docs/development/zed-sdk/modules/depth-sensing/depth-settings): distinct confidence/texture filters and temporal stabilization; set explicit profiles rather than assuming defaults.
- [SDK release](https://www.stereolabs.com/developers/release): 5.5 precision/graph options and float OpenEXR support; record installed version and any fallback.
- [ROS topic benchmark](https://docs.stereolabs.com/docs/integrations/ros-2/topic-benchmark-tool): transport/delivery measurement, not depth accuracy. Finite composed runs shut down their shared container; use a standalone benchmark for finite runs.
- [Middlebury](https://vision.middlebury.edu/stereo/submit3/): an additional portable-stereo baseline; not direct SDK replay support.
