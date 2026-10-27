# Resident dual-reference depth observations

`DaxaDepthPipeline` combines native packed capture preprocessing, true left/right
stereo, LR consistency and calibrated metric observations on one shared
`DaxaFoundation`. It is an experimental source-grid stage. The current measured
boundary and remaining qualification gates are in
[depth evidence](daxa-depth-evidence.md).

## Setup and calibration

The optional target requires `STEREOFORGE_DAXA_STEREO`,
`STEREOFORGE_DAXA_PREPROCESS` and `STEREOFORGE_DAXA_DEPTH`. The depth option defaults
to OFF. It adds no runtime Python, SlangPy, SGL, CUDA or separate Vulkan device.
Offline fixture creation can use Python/OpenCV; native processing uses C++ and
offline Slang SPIR-V.

`DaxaDepthConfig` contains `StereoConfig`, `lr_tolerance` (default 1), one or two
slots, required full subgroup size 32 or 64, an explicit stereo schedule, and
`right_reference_mode` (default `IndependentSGM`). Both independent references use
`PackedSpecialized`; existing disparity, path, penalty and packing limits apply.
The benchmark fixes subpixel=true and LR tolerance=1. Its default slot count is
2, while the C++ configuration defaults to 1.

`DaxaRectification::geometry` retains optional typed setup data:

- `DaxaCameraModel`: dimensions, double-precision pinhole values and distortion.
- `DaxaStereoCalibration`: two camera models and a row-major right-to-left pose.
- `DaxaRectifiedGeometry`: original/rectified calibrations, R1/R2/P1/P2 and both
  calibration identity strings.

Missing geometry remains supported by existing 2D preprocessing, but blocks
metric pipeline construction. The loader binds the typed data to existing
calibration JSON byte identities, map hashes/extents and original/rectified
image dimensions. It does not normalize or reserialize calibration JSON. A
changed byte serialization has a different identity and needs explicit rebinding.

Metric admission checks proper rotations/affine poses, positive baseline,
`R1 * R_original * transpose(R2) = I`, and `R1 * t_original = (baseline,0,0)`.
The rectified right-to-left pose must have identity rotation and positive
horizontal X translation. Rectified fx/fy/cy must be compatible, distortion is
exactly zero, P1 has zero translation, and P2 has X translation `-fx*baseline`
with the other fourth-column entries zero. Principal X coordinates may differ.

Rotation/pinhole comparisons use `abs <= 1e-6 + 1e-5*abs(expected)`.
Translation/length comparisons apply that policy after baseline normalization;
translation is normalized before rotation. Affine bottom rows preserve the
legacy atol=1e-8 / rtol=1e-5 policy. Projection fourth-column structural zeros are exact,
and P2 X translation must be negative with a compatible normalized ratio.
Metric operands must remain finite and representable when converted to float.

Geometry, matrices, maps and hashes are checked during setup. The shared immutable
configuration owner supplies cached float `MetricCalibration`; enqueue does not
scan matrices or parse JSON. Frames from different owners require new setup even
when their identity strings match. Binding a supplied map/configuration object
is not proof that its maps were generated from those matrices or that a physical
camera calibration is accurate.

## Stereo, LR and metric rules

Coordinates are optical X right, Y down, Z forward; baseline/depth are metres.
Positive disparity is always `d = x_left - x_right`:

- Left reference compares `L(x)` with `R(x-d)`.
- Right reference compares `R(x)` with `L(x+d)`.

Swapping ordinary images does not implement the second rule. Census descriptors
keep physical left/right order. The unchanged matching shaders preserve their
integer recurrence, packed penalties, tie selection and float helper profile.
Preprocessing preserves the exact RGB conversion/remap/gray contract described
in [preprocessing evidence](daxa-preprocess-evidence.md).

Left LR samples `floor(x-d_left+0.5)`; right LR samples
`floor(x+d_right+0.5)`. Both require an in-bounds target, own/target validity,
finite disparities and absolute disparity difference <= tolerance. Symmetric
right checking is an explicit additional source-layer policy relative to the
older Python pipeline's published left-only LR result. There is no confidence
threshold.

GPU indexing preserves the integer-grid/double-expression result without GPU
float64: after finite/range admission, let `b=floor(d)` and `h=float(b)+0.5`.
Left delta is `b+(d>h)`; right delta is `b+(d>=h)`. Integer translation and bounds
checks precede target loads. Adding fractional disparity to a large float pixel
coordinate before rounding can lose nextafter distinctions and is not permitted.

LR rejection writes canonical NaN bits `0x7fc00000`, zero stereo confidence and
LR validity 0. Metric support is separate: both the source-center remap support
and nearest correspondence-center remap support must be present. This strict
policy does not change matching or the LR mask and does not certify all Census
taps. The remap support definition ignores out-of-bounds taps with zero
quantized interpolation weight.

Metric axial Z is:

```text
Z = fx * baseline_m / (d - (cx_left - cx_right))
```

The cached rectified calibration supplies these operands. Only finite positive
Z is accepted. Metric-invalid observations have canonical NaN depth, zero metric
confidence/validity/provenance, source ID -1 and age 0. Valid fresh observations
retain stereo confidence, provenance `STEREO=1` in independent-SGM mode, exact
source ID and age 0. The optional derived mode uses provenance
`LeftAggregatedStereo=5` for valid observations in both layers because both LR
checks are correlated; invalid provenance remains 0.
Age 0 describes fresh observation ancestry; it is not measured camera latency.
IDs 0 through INT64_MAX are preserved. Larger capture IDs reject before device
access or upload.

## Output layout and ownership

Every observation buffer contains a row-major `width*height` left layer followed
by the same-sized right layer. The nine `DaxaDepthPlane` values are:

| Plane | Element | Meaning |
|---|---|---|
| LRDisparity | float32 | Positive disparity in pixels after LR |
| LRStereoConfidence | float32 | Stereo confidence after LR |
| LRValidity | uint32 | LR acceptance |
| Depth | float32 | Rectified-camera axial Z, metres |
| MetricConfidence | float32 | Confidence gated by metric validity |
| MetricValidity | uint32 | Accepted supported metric observation |
| Provenance | uint32 | Fresh stereo 1, invalid 0 |
| SourceIds | int64 | Actual capture ID, invalid -1 |
| AgeNs | int64 | Fresh/invalid 0 |

`Frame` supplies dimensions, layer count, generation, context identity, concrete
buffer IDs/byte extents, cached metric calibration, immutable calibration owner,
source lease, readiness/completion and timings. Buffer IDs are borrowed: retain
the frame until downstream GPU work completes. These are Daxa resources; they
cannot be reinterpreted as the RHI `BufferView.native` pointer.

`Frame::image()` returns retained `DaxaImageView`s for original/rectified RGB,
luma and support. RGB packs R/G/B/A into successive low-to-high uint32 bytes,
with alpha 255. Both full original canvases remain resident even if rectified
geometry uses a different canvas. Original RGB carries original calibration
identity; rectified RGB/luma and metric geometry carry rectified identity.
Pairing rectified depth with raw rays is invalid.

An output retains preprocessing and both hidden stereo frames through consumer
lifetime. One submission thread serializes the shared foundation. Slots, staging,
parameters, graphs and query pools are persistent setup allocations; constructor
prototypes are released after graph establishment. Graph callbacks do not retain
prototype frames. Releasing the pipeline facade does not invalidate retained
output ownership.

Call `collect()` before acquiring the next capture lease. Collection releases
completed, unleased depth producers first, then stereo inputs/unleased metadata,
then preprocessing source metadata. Held or in-flight outputs remain protected.
A full output ring rejects before upload. Independently retained image views
consume preprocessing capacity and can also cause pre-upload backpressure; no
stereo producer lease can escape separately in the legacy schedules. Only
`SharedCensusSequential` permits the typed `Frame::stereo_frame(reference)`
getter; either retained producer blocks aligned slot reuse before upload. `drain()` waits during
shutdown; ordinary enqueue performs no intermediate wait or full readback.

Source metadata survives through `Frame::source()`: original ID/timestamps and
clock domains, controls, raw IMU values and association status. Recorded duplicate
IDs/timestamps remain valid inputs; typed GPU generations identify slot reuse.
Raw IMU/sync metadata does not certify pose, and unrelated clock domains cannot
be combined into a camera-age measurement.

## Resident processing and diagnostics

The frame path performs one packed-source staging copy, GPU decode/remap/gray,
stereo scheduling, LR and metric/metadata materialization. Coherent parameter
updates and producer compute writes have explicit same-queue visibility before
consumption. It performs no full luma/output CPU transfer, per-frame geometry
validation or application heap allocation on successful warmed admissions.

One explicit completion snapshot is shared across admission/retirement. SDK
garbage collection adds internal completion scans; timestamp/readiness queries
are additional work. Host timing includes these costs. Combined GPU time spans
actual upload-start through metric-end on the same queue, including inter-graph
gaps. Counter properties and host elapsed time guard ambiguous timestamp wraps.
It is not a sum of stage percentiles.

`readback`, image/stereo readbacks and `check_stereo_integer` are explicit
blocking diagnostic APIs returning owned CPU data. Benchmark `--check-reference`
checks all eight preprocessing planes, both stereo integer/Census/mask sets and
actual packed words against independent fixture/scalar data after timing. LR and
metadata checks then use the actual GPU stereo float arrays. Both compound CPU
float flags remain false; no arbitrary CPU float tolerance is a qualification
rule. Isolated representable depth planes and independent frozen GPU arithmetic
have separate gates; the recorded exact GPU-byte results and their profile
boundaries are listed in [depth evidence](daxa-depth-evidence.md).

Requested-buffer preflight includes preprocessing/map staging, both GPU-intake
stereo engines and observation buffers/parameters, with a 2 GiB ceiling and
actual storage/dispatch limit checks. Reported requested bytes exclude query,
allocator/driver, diagnostic and CPU-source storage; they are not a measured
peak-memory bound. Wide RGB-D synthesis, rectified-to-raw depth/ray conversion,
temporal/adaptive processing and sustained live qualification remain subsequent
work.


## Optional scheduling experiment

`DaxaDepthConfig::stereo_schedule` defaults to `Sequential`; `PairedPerPath`
reuses the same buffers, algorithms and shader binaries in a persistent graph
that records both reference dispatches within each phase. Required same-eye
packed-penalty barriers remain between paths. Source/frame ownership, nine
observation planes and the requested-buffer ceiling remain unchanged.

Paired timings bracket both references together. In paired mode,
`DaxaDepthTiming::per_reference_stereo_timings_available` is false and
`paired_stereo` carries the phase intervals/total; `left` and `right` must not be
interpreted as measurements. The native report writes null individual-eye
statistics. Direct isolated `DaxaStereo::Frame::timings()` rejects paired frames.
The same-queue upload-through-metric span remains the complete depth measure.

This explicit experimental mode measured a regression and is not promoted.
See [paired evidence](daxa-depth-paired-evidence.md). It records two dispatches
per phase. The separate optional `LayeredPerPath` mode uses one XY×2 dispatch
per aggregation direction and selection, with the same buffers, path hazards,
leases and paired timing interpretation; Census/clear work remains duplicated.
Its shared-body extraction and required legacy-control requalification are
documented in [layered evidence](daxa-depth-layered-evidence.md). Sequential
remains the default; qualification and timing are explicit per experiment.


`SharedCensusSequential` is a separate opt-in mode that physically shares one
Census descriptor/validity quartet per slot, dispatches Census twice total, then
runs complete Left and Right clear/path/select sequences with independent
penalties and outputs. It uses 14 phases for P4 or 22 for P8, joint timings and
null per-eye statistics. Its private sharing constructor, retained producer
ownership, actual buffer savings and measured limits are documented in
[shared Census evidence](daxa-depth-shared-census-evidence.md). The default
remains Sequential.


`right_reference_mode=LeftAggregatedSelection` is an additional estimator choice
admitted only with `SharedCensusSequential`. It keeps Left SGM but selects Right
from shifted Left costs and packed penalties, with no Right recurrence buffer.
The graph has `paths+5` phases (9/13), and the output layout stays unchanged.
Typed frame/readback metadata identifies the estimator and correlated LR
dependency. Right penalty access and intermediate readback reject; raw Right
outputs remain available. Left raw outputs remain unchanged, while Left LR can
change. The native derived report uses its own schema and preserves false flags
for independent Right recurrence/winner and compound CPU float qualification.
This option is not promoted: development geometry shows quality regressions.
See the [derived estimator contract and evidence](daxa-depth-derived-evidence.md)
for support rules, lifetime guarantees, exact-byte gates and measured limits.

The diagnostic native benchmark and derived comparison expose the existing
`DaxaDepthConfig::lr_tolerance` as `--lr-tolerance`, in disparity pixels.
The default remains 1.0; requests must be finite and nonnegative, and the
derived harness binds the normalized float32 value to native execution,
reported metadata and its observation oracle. Acceptance includes equality
at the threshold. Tightening this rejection gate changes output validity;
it does not refine raw disparity. See
[consistency threshold evidence](daxa-depth-consistency-evidence.md).
