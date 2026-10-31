# StereoForge: same-input temporal consensus and stage audit, v2

## What this package does

Adds four Python benchmark modules and a pytest suite. It does **not** modify
ROS publishing, camera controls, SGM shaders, launch files, or Git state.
It uses the existing NumPy/OpenCV environment. Run the offline experiments
with `uv run`, not the ROS/Pixi Python interpreter.

1. `replay_temporal_v2.py audit` reinterprets the existing report using explicit
   conditional denominators and examines available raw payloads.
2. `replay_temporal_v2.py replay` compares unmodified input, current-frame
   temporal confirmation, and bounded 100/200 ms historical filling on the
   **same saved frames**.
3. `temporal_synthetic_v2.py` exercises noise, dropout, moving objects, hidden
   depth changes, and stable-but-wrong depth against synthetic truth.
4. `stereo_stage_audit_v2.py` reads the binary planes already exported by
   `daxa_depth_benchmark --dump-prefix`. No new native readback is introduced.

This is a diagnostic CPU reference, not a real-time implementation, a validated
robotics depth estimator, or a measured Vulkan speedup.

## Evidence from the uploaded unfiltered sequence

The recording is 90 frames at 1280x720. The source report and manifest hashes
and the exact calculations are in `docs/uploaded_sequence_audit.json`.

- Mean valid coverage: 21.0058%.
- Ever valid over the sequence: 85.2336%; never valid: 14.7664%.
- Valid in at least 80% of frames: 8.4038% of image locations.
- Of pixels valid in one frame, 43.0258% become invalid in the next.
- Mean common-valid support across adjacent frames: 11.9683% of the image.
- Median of framewise median disparity changes: 0.235264 px.
- Median of framewise p90 disparity changes: 2.048346 px.
- Mean high-disparity-boundary fraction among accepted depth: 0.8554%.
- First-to-last receipt cadence: 19.4412 Hz. This is not GPU throughput.
- The old MP4 is encoded at 30 fps, so playback is about 1.543x receipt speed.

The 79% invalid fraction in any one frame must not be confused with 79%
permanently unobservable scene area. Conversely, ever-valid pixels are not
necessarily *correct*: random or persistently biased correspondences count.

The uploaded RGB image shows a difficult scene: repetitive slats, a bright
window, shadowed surfaces, and a fairly textureless cabinet. That is a visual
observation, not a diagnosis of which native rejection stage is responsible.
Far depths through a window are not automatically outliers.

Only frame 89's raw float depth and RGBA payload were supplied, along with the
90-frame report, manifest and a lossy preview. The full real-sequence temporal
ablation therefore has **not** been run here. MP4 colors must not be decoded
back into purported float ground truth. Run replay against the original
90-frame directory on the workstation.

## Algorithm and limitations

Each fixed pixel retains up to nine **direct input** corrected disparities
`q = fB/Z = d - (cx_left-cx_right)`. A hypothesis needs at least three
observations within 1 px and at least 60% of available direct observations
in the recent window. Evidence older than 500 ms is expired.

Current valid measurements anchor the hypothesis. The output is the mean of
agreeing direct disparities. A valid contradictory measurement clears the old
hypothesis; it is not averaged across a different surface. Small/thin objects
are not removed by a component-size rule because there is no spatial borrowing.

For an invalid current sample, the last confirmed hypothesis can be reused
for at most 100 or 200 ms, provided sufficient compatible direct evidence
remains. Filled output is never fed back into the history or counted as a
new vote. Ages are since the last corroborated **host receipt**, not exposure
ages; the publisher clock remains unvalidated.

RGB change masks and reference-frame phase correlation reset history where
visible changes occur. On detected motion or an unobservable global RGB
alignment, the filter falls back to current raw measurements. These guards
are intentionally conservative, but cannot certify 3D stationarity. Slow,
textureless, axial or photometrically invisible motion can escape detection.
Use `--assume-static` only for stationary-camera experiments. A moving robot
needs pose/flow consistency and depth-aware reprojection/occlusion handling.

Output provenance:

| Code | Meaning |
|---|---|
| 0 | Invalid |
| 1 | Direct current raw measurement, including motion fallback |
| 2 | Current measurement corroborated by recent direct observations |
| 3 | Historical fill: current measurement is absent |

The metrics report raw/current/held coverage separately. Jitter is evaluated
both on all available output pairs and on identical raw/output **fresh-current**
support. Held data is excluded from the latter; otherwise holding a value
would misleadingly appear to eliminate all measurement noise.

Temporal consistency is not accuracy. The synthetic `stable_wrong` case
intentionally demonstrates this: the temporal output becomes nearly dense
and perfectly stable while retaining the full 8 px bias. Likewise the hidden
same-RGB depth-change case demonstrates the bounded ghosting risk of holding.
The package deliberately includes these failures rather than only favorable
examples.

## Install and test (fish)

From the existing checkout; no pull/merge/cherry-pick is needed:

```fish
cd ~/git/stereoforge
unzip -n ~/Downloads/stereoforge-temporal-consensus-v2.zip -d .
uv run pytest -q tests/test_temporal_consensus_v2.py
```

The files have new names. `unzip -n` does not overwrite an existing file.
Do not make another Git merge while the checkout reports `(MERGING)`.

## Replay the recorded sequence

This does not need a running camera, ROS, RViz or Vulkan:

```fish
cd ~/git/stereoforge
set input artifacts/depth-video/unfiltered-20261030-162036
set tag (date +%Y%m%d-%H%M%S-%N)
set output artifacts/depth-video/temporal-v2-$tag

uv run python benchmarks/replay_temporal_v2.py replay "$input" \
    --calibration-manifest configs/cameras/zed2/fixtures/SN26041639-fixture/manifest.json \
    --output-dir "$output" \
    --assume-static \
    --preview
```

`summary.json` contains four profile results. The first eight frames are
excluded from the profile comparison as equal warm-up for all profiles;
`raw_full_sequence` contains the full raw-sequence statistics.
`comparison.mp4` shows RGB | raw depth | consensus + 100 ms hold with a fixed
0.5-8 m display range. Its constant FPS equals measured average receipt
cadence; it is not exact variable-frame-rate playback.

Fresh versus held coverage, conditional dropout, paired-support jitter, raw
retention, contradictions, motion resets and worst historical age are all
reported. `returning_after_hold_disagree_gt1px_fraction` compares the last
held estimate to a newly available raw observation: useful self-consistency,
not ground truth. CPU reference p50/p95 timing is reported but is not a GPU
performance target.

Use a fresh output directory for every run. Existing output is never deleted.
To save one full derived recording, add `--save-sequence hold100`. It writes
float depth, RGBA, provenance, age and a compatible sequence manifest; about
1.1 GB for 90 720p frames. The original recording is not changed. The default
only writes summary and optional video, avoiding that storage cost.

A small parameter ablation, with no new capture:

```fish
for tolerance in 0.5 1.0 2.0
    set tag (date +%Y%m%d-%H%M%S-%N)
    uv run python benchmarks/replay_temporal_v2.py replay \
        artifacts/depth-video/unfiltered-20261030-162036 \
        --calibration-manifest configs/cameras/zed2/fixtures/SN26041639-fixture/manifest.json \
        --output-dir artifacts/depth-video/temporal-tol-$tolerance-$tag \
        --assume-static --agreement-px $tolerance
    or break
end
```

Do not choose a larger tolerance solely because it produces more pixels.

## Synthetic checks

```fish
set tag (date +%Y%m%d-%H%M%S-%N)
uv run python benchmarks/temporal_synthetic_v2.py \
    --output-dir artifacts/depth-video/temporal-synthetic-$tag
```

Committed sample results are in `docs/temporal-v2-synthetic-results.json`.
They use generated per-pixel disparity observations, not simulated stereo
images or real ZED 2 data. Five cases are included, including adversarial
stable-wrong and hidden-depth-change cases.

## Native validity-stage and repeatability audit

Stop the live publisher and other GPU experiments for this diagnostic. The
benchmark reads the retained camera fixture; it does not access live USB or
use the provisional ROS clock map. This fixture is not the same sequence as
the saved 90-frame ROS recording. It diagnoses the native computation path,
not temporal behavior on that entire recording.

The existing native application and its accepted options were inspected at
`apps/daxa_depth_benchmark.cpp`, blob
`7f1332ed461e3ad46131c393c547dfcdb3d5dcd9` on the integration branch.

```fish
cd ~/git/stereoforge
pixi run -m ros_ws cmake --build build/ros-daxa-native \
    --target daxa_depth_benchmark -j 8

set pixi_lib (realpath ros_ws/.pixi/envs/default/lib)
set fixture (realpath configs/cameras/zed2/fixtures/SN26041639-fixture/manifest.json)
set tag (date +%Y%m%d-%H%M%S-%N)
set diag artifacts/depth-video/native-stages-$tag
mkdir -p "$diag"

for repeat in 1 2 3
    env LD_LIBRARY_PATH="$pixi_lib" \
        LD_PRELOAD=/usr/lib64/libdrm.so.2:/usr/lib64/libdrm_amdgpu.so.1 \
        VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
        build/ros-daxa-native/daxa_depth_benchmark \
        --manifest "$fixture" \
        --output "$diag/run-$repeat.json" \
        --dump-prefix "$diag/run-$repeat" \
        --frames 3 --warmup 1 --slots 2 --wave-size 32 \
        --disparities 128 --paths 4 --p1 8 --p2 32 --lr-tolerance 1
    or break
end

uv run python benchmarks/stereo_stage_audit_v2.py \
    --reports "$diag/run-1.json" "$diag/run-2.json" "$diag/run-3.json" \
    --output "$diag/stage-summary.json"
```

This uses the known Fedora host-DRM workaround only for the GPU process.
It creates roughly 310 MB of diagnostic dumps at 720p for three runs.
The auditor checks:

- validity of left/right selection before LR;
- out-of-bounds corresponding coordinates;
- absent or invalid opposite-view correspondences;
- rejection by actual LR disparity difference;
- subsequent metric-depth rejection;
- CPU recomputation versus native LR masks;
- hypothetical LR tolerances 0.5/1/2/4 px (acceptance, not accuracy);
- bitwise repeatability of exported final snapshots across the three runs.

If identical input/config produces different exported stage planes, investigate
synchronization, state reuse and dispatch behavior before adding temporal
filtering. Matching final snapshots across three runs is evidence for those
samples only, not proof of all-frame determinism.

If selection validity is already low, investigate descriptors, rectification,
photometric mismatch and ambiguity. If selection is dense but LR collapses,
inspect the opposite-reference mapping and disparity agreement. If metric
conversion removes most remaining pixels, inspect remap support and geometric
conversion. Raw Census/support masks are not exported by this particular dump,
so the auditor does not pretend to distinguish all causes.

## Missing evidence and next integration boundary

The ROS recording contains one rectified RGB view and final depth, not a
synchronized left/right raw pair, cost volume, pre-LR disparity or pose. It is
sufficient for postfilter replay, but not to identify calibration or stereo
matching root causes. Store a synchronized native stereo recording for future
same-input SGM experiments. Do not infer a speedup from receipt FPS.

Keep the 50-pixel metric speckle filter disabled for the baseline. Do not promote
this temporal experiment into `/stereoforge/depth` until its coverage/age/
accuracy trade-offs are acceptable on motion and ground-truth tests. A later
Vulkan implementation can replace the Python reference only after parity and
same-input quality checks; no GPU throughput claim is made here.

## Related primary documentation

RealSense's post-processing documentation independently recommends disparity-
domain processing for stereo, describes temporal persistence, and warns about
motion smearing. These motivate the ablations, not the chosen unvalidated
thresholds. OpenCV StereoSGBM documents disparity-domain speckle range and LR
consistency controls. No implementation from those projects is copied here.

- `https://dev.realsenseai.com/docs/post-processing-filters/`
- `https://dev.realsenseai.com/docs/tuning-depth-cameras-for-best-performance/`
- `https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html`
