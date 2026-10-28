# Native canonical RGB input

This adapter accepts offline canonical RGB records without decoding YUYV or
remapping already rectified pixels. It implements a bounded portion of
[Task 3](superpowers/plans/2026-10-28-zed-quality-benchmark.md#task-3-native-already-rectified-rgb-input).
Task 3 is not complete. SDK geometry qualification is tracked separately in the
[ZED diagnostic evidence](zed-reference-diagnostic-evidence.md).

## Record admission and layout

Construct `RectifiedRgbFrame(CaptureLease)` once per immutable input record,
outside the resident loop. Its constructor validates metadata and scans support
pixels for binary values. Cache the resulting admitted capability; copies retain
admission. Replacing its public source does not admit the replacement.

`CaptureFrame::source_kind` is `CanonicalRgbMetadata`. Its `canonical_rgb`
sidecar retains four pitched, interleaved RGB8 planes: original left/right and
canonical rectified left/right. Two pitched U8 planes carry rectified support.
The immutable backing owner keeps every span alive. Plane extents include row
pitch; row padding is not image content.

The sidecar records input, original/rectified pixel, calibration and sampling
identities. Metadata keeps the actual original dimensions:
`source_width = 2 * original_eye_width`, `height = original_height`,
`stride_bytes = 0`, and an empty `packed_yuyv` vector. No fabricated packed
payload or raw remap arrays are needed. Frame IDs, camera time and clock domains
remain associated with the retained source.

Unknown host ingress is allowed for offline input. It does not qualify capture
latency, frame age or a live timing result.

## Producer and consumer ownership

`DaxaRectifiedInput` has two fixed slots. Full and optional coarse `DaxaFrame`
leases share the original GPU RGB buffers and `CaptureLease`. Consumers retain
these resources until their work and leases retire. Full and coarse calibration
records have empty raw maps; their calibrated grid models bind the input source.

The sampling profile is
`canonical_rgb8_box2x2_halfup_gray15_support_and4_halfpixel_v1`.
Coarse RGB uses a 2×2 box with `(sum + 2) / 4` rounding. Luma uses gray15;
coarse support is the AND of all four input support values. K/P scaling follows
the half-pixel convention. This profile differs from F1's direct original-camera
remap and requires its own quality and cost measurements.

External depth accepts the producer's exact calibration and foundation through
its prototype `DaxaFrame`. Refinement accepts coarse depth plus the full
rectified frame and bypasses target remapping. Generic observations can feed
View through source-aware canonical calibration setup. The legacy raw setup
continues to require strict raw-map/source association.

The following is setup and submission pseudocode; lifetime and backpressure
handling belong to the caller:

```cpp
// Offline, once per immutable record:
RectifiedRgbFrame admitted(source_lease);

// Persistent setup on one shared foundation:
DaxaRectifiedInput producer(full_calibration, optional_coarse_calibration,
                           input_config, context, admitted);
auto prototype = producer.enqueue(admitted);
auto full = prototype.full();
auto coarse = prototype.coarse();
DaxaDepthPipeline depth(depth_config, context, coarse);
auto coarse_depth = depth.enqueue_preprocessed(coarse);
DaxaRefinement refine(context, coarse_depth, full, refinement_config);

// Resident submissions use cached admitted records and available slots:
auto input = producer.enqueue(next_admitted_record);
auto low = depth.enqueue_preprocessed(input.coarse());
auto result = refine.process(low, input.full());
auto observations = result.observations();
// Feed observations to the typed View path using canonical source setup.
```

Use `collect()` and release leases to retire work before slot reuse. Explicit
waits and readback are debug/output boundaries, separate from the GPU-resident
submission loop.

## Timing and memory accounting

External depth marks preprocessing and combined GPU timing unavailable; raw
preprocessing stages and the combined duration are NaN. Report the actual
`stereo_to_metric_gpu_ms` span, plus producer copy/upload and canonical packing
costs separately. Do not treat unavailable stages as zero-cost work.

External depth memory counters cover depth-owned buffers. Add producer bytes
when reporting the complete chain. Producer allocation has a 2 GiB preflight
bound and RAII cleanup for failed setup; this is not a total process-memory cap.

## Verified boundary and reproduction

Root verification covered a 128×64 D32/P4 GPU case: exact eight-plane producer pixels,
SGM integers, masks and IDs, with metric float error bounded by `2e-6`. A separate
300-frame test used 300 unique pre-admitted records cycling three synthetic
immutable textures:
300 accepted, completed and unique outputs, zero gaps, maximum two in flight,
and strict View-only retention. Validation reported zero errors and warnings
through teardown, including core and synchronization validation, on RADV 8060S /
Mesa 26.2.3.

Five native CPU tests and all three raw GPU depth/refinement/View regression
contracts passed. The full CPU regression passed 2,038 tests with 37 GPU/camera
tests deselected (119.70 s); no tests were skipped.
The independent actual-source performance probes passed 76 tests; canonical
View setup/control probes passed 25. Warm application allocations, implicit
waits and readbacks were zero. Six fixed CPU staging copies of 12.9024 MB took
0.254–0.272 ms per frame. Device/graph calls were mocked and offline record
admission was excluded: these are CPU copy/control measurements, not GPU upload
or execution timings.

Evidence is retained in `artifacts/zed-task3/adapter-gpu-stream-v1.log`,
`adapter-cpu-v6.log`, `raw-gpu-regression.log`, `full-cpu-regression-green.log`,
`performance/summary.md` and
`performance/view-setup/summary.md`. This 128×64 synthetic stress establishes
bounded runtime correctness only; it does not establish FPS, 720p throughput,
live-camera performance or scientific quality.

```bash
cmake --build build/daxa --target daxa_rectified_input_contracts
ctest -N --test-dir build/daxa -R '^daxa_rectified_input_cpu_contracts$'
ctest --test-dir build/daxa -R '^daxa_rectified_input_cpu_contracts$' --output-on-failure

env -u VK_LAYER_ENABLES \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
VK_LAYER_PATH="$PWD/build/validation-layer/usr/share/vulkan/explicit_layer.d" \
LD_LIBRARY_PATH="$PWD/build/validation-layer/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
VK_LAYER_VALIDATE_SYNC=1 build/daxa/daxa_rectified_input_contracts --gpu
```

The listing must contain exactly one CPU test before execution. Extracted-layer
paths follow the [View depth evidence](daxa-view-depth-evidence.md); use the local
layer installation corresponding to the recorded run.

The benchmark application, canonical record loader, refinement application input
selection, SDK geometry qualification,
quality evaluation and sustained 60 Hz remain open. This adapter and its bounded
correctness evidence do not close those Task 3 or later qualification gates.
