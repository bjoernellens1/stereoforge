# v14 physical depth corpus and audits

Correctness is measured against physical references, not against the estimator itself. Corpus data lives outside git
(`~/stereoforge_corpus/<scene>/`, ~0.8 GB per 150-frame scene with RealSense).

## Record scenes

See the session-labelled capture command in the frozen quality campaign below.

| Scene | What to show | Why |
|---|---|---|
| `S1_target_<d>m` (d = 0.5, 1.0, 1.5, 2.0, 3.0) | ChArUco/AprilTag board on a flat stand, tape-measured camera-to-board distance | absolute bias/precision vs range |
| `S2_white_wall` | empty white wall | textureless behaviour, boxes, prior locking |
| `S3_person_white_wall` | head and shoulders in front of a white wall | low-contrast silhouette / halo |
| `S4_textured_desk` | desk with objects, static | plane RMSE, temporal stability |
| `S5_person_moving` | slow arm/head motion | ghosting / wake |
| `S6_dim_room` | lights dimmed | exposure/noise behaviour |
| `S7_calib_board_poses` | board at ~12 varied poses/rotations visible to both sensors | ZED-RealSense extrinsics, rectification |

**Mount the ZED 2 and the RealSense rigidly side by side, pointing the same way, so both see the board and the scene** (check both views before recording). Targets must be fully visible to the ZED left camera. RealSense runs at 640x480 (USB 2.1 link). The tool starts the ZED only once the RealSense streams and prints the overlap; it should equal the ZED capture length.

## Audits (no ground truth needed)

```bash
PYTHONPATH=python:. .venv/bin/python benchmarks/physical_depth_v14/audit_rectification.py FRAMES.npy      # vertical disparity
PYTHONPATH=python:. .venv/bin/python benchmarks/physical_depth_v14/refine_calibration.py FRAMES.npy --out overlay.json
PYTHONPATH=python:benchmarks:. .venv/bin/python benchmarks/physical_depth_v14/baseline_cpu.py FRAMES.npy # coverage, pixel locking, temporal std
```

`FRAMES.npy` is `zed_raw.npy` (or a subsample) of a scene. Results of the first audit are in
`benchmarks/results/v14_phase0_baseline.json`.

## Frozen quality campaign (2026-11-05)

The engineering targets live in `quality_contract.py` and follow the campaign
plan: correct measured completeness on fixed independently known support, with
`tau(Z)=max(10 mm, 1% Z)`. A finite prediction is not necessarily correct. The
numeric accuracy gate is separate from qualification of survey, registration,
reference uncertainty and timing. Current recordings are development evidence:
they have no established mounting-session identity, native RealSense geometry,
verified clock mapping or independent dense surveyed support. Do not reuse the
long-S7 transform for them or report their sensor agreement as absolute accuracy.

New captures require a fresh scene directory, a rigid mounting-session identity,
a room and a scene-configuration group. Keep a whole group in one split. Select
at least ten independent groups across two rooms, with separate development,
validation and holdout groups. Moving either camera relative to the other starts
a new session. Use the native factory calibration for the requested profile;
720p calibration cannot be scaled and called native 1080p calibration.

```bash
PYTHONPATH=python:benchmarks:. .venv/bin/python -m benchmarks.physical_depth_v14.capture_corpus \
  --scene roomA_target_1m --rig-session mount-A --room roomA --scene-group target-A \
  --zed-calibration configs/cameras/zed2/fixtures/SN26041639-fixture/original-calibration.json \
  --zed-height 720 --frames 150 --realsense
```

Native RealSense depth is `rs_native_depth.npy` (uint16, apply
`rs_factory.json:depth_scale_m`). That file contains native intrinsics with
explicit distortion models and depth-to-stream transforms. `rs_timing.json`
retains device timestamp/domain/frame number per stream plus host monotonic
arrival. ZED OpenCV acquisition time remains unavailable; its arrival time is
labelled separately. Stream startup fails explicitly if the USB link cannot
carry depth/RGB/both IR. A capture is incomplete until `scene.json:status` says
`captured_unqualified`. Capturing files does not qualify clocks or geometry.

Verify installed SDK alignment semantics without opening a camera:

```bash
/home/bjoern/miniforge3/bin/python3 -m benchmarks.physical_depth_v14.verify_realsense_alignment \
  --out /tmp/alignment-semantics.json
```

The software-device probe applies a known optical-Z translation. The installed
2.58.1.10581 SDK preserves native depth Z in aligned pixels, matching
[librealsense align.cpp](https://raw.githubusercontent.com/IntelRealSense/librealsense/master/src/proc/align.cpp).
Deproject native depth with native intrinsics (including its distortion model),
then transform 3D points; do not deproject aligned values as colour-camera Z.
This probe qualifies those SDK semantics, not physical accuracy or calibration.

Run a bound, development-only baseline (CPU timings are not GPU timings):

```bash
PYTHONPATH=python:benchmarks:. .venv/bin/python -m benchmarks.physical_depth_v14.quality_baseline \
  --corpus ~/stereoforge_corpus --scenes S1_target_0.5m S2_white_wall S3_person_white_wall \
  --indices 0 74 149 --cpu --out /tmp/quality-baseline
```

The report binds source bytes, CPU library, recordings, frame indices, calibration
files/maps, configuration and retained NPZ depth/reason masks. It stays incomplete
until all requested work and identity rechecks finish. It compares independent
full-resolution reference SGM D128/D256 and the live/full CPU shader paths on
identical recorded inputs, without temporal reuse or RealSense input. The
reference computes both eyes independently and rejects out-of-image, ambiguous
and range-boundary right evidence. No interpolated resolution is scored.

`first_failure` partitions observed stage rejection in declared order;
`all_reasons` counts overlapping stage masks. Nonfast `shader_reason_counts`
separately counts flags that may include recovered priors or warnings on accepted
pixels; these are not causal first failures. Default fast shaders skip detailed
writes. Diagnostic readback now uses a separate instance with identical fast,
ultra-fast and adaptive decisions plus explicit reason writes. It compares all
exported outputs byte for byte before admitting reason planes, and preserves the
original producer buffers and timing queries. Reports expose this contract under
`diagnostic_readback`. Ultra-fast right metric outputs remain unavailable and
are initialized to invalid values on the GPU; host export does not manufacture
their validity. CPU APIs omit skipped reason/counter fields and reject
borrowed snapshots for equivalence checking. `require_equivalent` checks shape,
dtype and bytes before accepting diagnostic evidence. On physical recordings,
correct-disparity exclusion and absolute correctness remain unavailable without
qualified known support. The analytic controls below provide synthetic evidence. Measured and
completed outputs remain separate; planar inference cannot raise measured
coverage. No live estimator or new production profile is promoted by this run.

### Retained development result

`benchmarks/results/v14_quality_diagnosis.json` retains the 2026-11-05 decision
and artifact identities. Nine native 720p input frames (indices 0/74/149 from
three scenes) produced 36 bound outputs. Mean **measured coverage**, not correct
completeness:

| Development scene | Live CPU | Full CPU | Independent D128 | Independent D256 |
|---|---:|---:|---:|---:|
| Target clip labelled 0.5 m (distance unqualified) | 10.65% | 33.42% | 52.04% | 63.58% |
| White wall | 13.09% | 35.61% | 74.86% | 75.00% |
| Person against wall | 13.97% | 38.37% | 78.87% | 78.54% |

On these inputs, raw matching rejects 59.06–66.73% of frame pixels in the live
CPU path, followed by 21.75–26.43% rejected at LR. These are observed rejection
stages, not proof that relaxing them recovers correct depth. D256 reduces the
near-clip reference range-boundary rejection from 13.79% to 6.40%, while the
silhouette clip loses coverage. Absolute correctness of those changes is
unmeasured. The retained decision is **no promotion**.

Earlier `v14_phase0_baseline.json` and `v14_phase2_reference_prototype.json` are
historical witnesses with different inputs/semantics. In particular, the old
reference derived right costs from left aggregation and clipped LR coordinates;
its numbers do not qualify the corrected independent reference. Its sigma fit
measures temporal scatter on one scene, not absolute uncertainty.

The initial ABBA replay failed the unchanged 4.167 ms gate in both fresh and
existing native binaries with matching embedded shader identities: GPU p50 was
17.99–18.99 ms, dominated by a 12.69–14.70 ms coarse interval. All four 30-plane
parity checks pass, with zero validation errors or warnings. Existing user GPU
clients were left running; this was not isolated-hardware qualification, and
concurrency alone is not proof of the cause. Both pytest suites retained that
failure, so commits were withheld at that stage. The failure witnesses remain
immutable; the later green verification is recorded below.


### Qualified fast-path diagnostic replay

The follow-up `v14_fast_diagnostics.json` records byte-equivalent fast counts on
all nine of the same recorded inputs. Ambiguity flags cover 41.30–49.15% of a
frame, too-few-candidate flags 17.30–19.18%, and LR rejection 21.75–26.43%.
Flags overlap and can occur on recovered pixels; the separate first-stage
partition remains authoritative for total rejection. These counts identify
where matching work fails, without asserting correct recovery or absolute error.

Re-run just the CPU diagnostic comparison without recomputing reference SGM:

```bash
PYTHONPATH=python:benchmarks:. .venv/bin/python -m benchmarks.physical_depth_v14.quality_baseline \
  --corpus ~/stereoforge_corpus --scenes S1_target_0.5m S2_white_wall S3_person_white_wall \
  --indices 0 74 149 --disparities --cpu --out /tmp/fast-diagnostics
```

The user confirmed the clips were recorded on 2026-11-05 and that surveyed dimensions
and datum bounds are unavailable. The corpus sidecar
`campaign_tags_2026-11-05.json` tags all thirteen clips by date as development
recordings, with mounting-session status explicitly unverified. It changes no
historical scene metadata and does not authorize cross-scene transform reuse.

After authorized cleanup of stale task camera/visualization processes, the
unchanged existing microbenchmark binary records 3.206 ms GPU p50 and 3.975 ms
host p50 under the same 4.167 ms threshold. No context-ablation job was signalled.
Their earlier PIDs were already absent at cleanup, so this is a changed-workload
comparison, not causal attribution solely to ROS/RViz. The earlier failure
witnesses remain retained. Native fast diagnostic preservation passes the
720p adaptive on/off and planar on/off tests. Unavailable ultra-fast right-eye
metric buffers are now initialized on the GPU during setup, including invalid
source IDs and completed outputs. The regression downloads resident depth,
validity and IDs before host readback; host canonicalization has been removed.
Subsequent full-source verification is recorded below; the earlier incomplete
and failed verification runs remain retained as historical witnesses.

### Fresh native reference capture and geometry

The explicit `--realsense-fps 15` profile delivered native depth, RGB and both
IR streams on the connected USB2.1 link. The four-stream 30 fps attempt timed
out and remains an incomplete failure witness. The fresh 15 fps probe contains
10 native 720p ZED pairs and 17 RealSense sets. All 14 file hashes match;
per-stream frame IDs and timestamps increase without missing frame numbers.
Factory metadata records the actual profile FPS/format, scale, distortion,
extrinsics, serial and SDK version. This short probe qualifies capture content,
with room, surveyed support, rigid mounting and clock mapping still unqualified.

`native_reference.py` deprojects integer native-depth pixels through the
installed SDK and then applies the factory depth-to-color rigid transform.
It rejects aligned float previews, calibration/shape mismatches, unsupported
distortion, fractional sampling, and invalid transforms. Zero-depth and
out-of-image samples stay invalid. Fourteen tests pass with the installed SDK;
the regular test environment skips its five SDK-dependent cases.

The retained geometry probe contains 66,781 valid samples from all 17 frames.
Within 0.5–3 m, SDK versus zero-distortion algebra differs by at most 0.25 µm.
This is a numerical implementation check, with no absolute accuracy claim or
cross-camera rig refit. Artifacts are retained under
`~/stereoforge_quality_evidence/2026-11-05-native-reference-geometry-v2/`.

### Single-pair reference comparisons

The runner accepts explicit cost/path/fit variants and retains each candidate's
parameters and masks. Census requires its complete 5x5 support in both eyes;
both references compute independent right-eye evidence. Example:

```bash
PYTHONPATH=python:benchmarks:. .venv/bin/python -m benchmarks.physical_depth_v14.quality_baseline \
  --corpus ~/stereoforge_corpus --scenes S1_target_0.5m --indices 0 --disparities 256 \
  --reference-costs normalized census --reference-paths 4 8 \
  --reference-subpixel equiangular parabolic none --out /tmp/reference-cost-path-fit
```

`benchmarks/results/v14_reference_cost_path_fit.json` retains the first complete
12-candidate native 720p D256 comparison (one development frame): normalized
costs yielded 63.28–66.05% coverage and Census 32.70–40.82%. Every output hash and
all 606 source hashes were checked. Coverage does not establish correct recovery;
subpixel fits and parameters remain unselected pending qualified support.

The fresh native 1080p probe contains 10 ZED pairs and 24 RealSense sets, with
all capture hashes verified. Its FHD factory calibration produces rectified
fx=1062.311 px and baseline=0.119444 m: expected disparity at 0.5 m is 253.773 px.
A 16 px margin, rounded to a multiple of 32, gives D288. The first full-resolution
independent reference reports 47.28% coverage and is retained in
`benchmarks/results/v14_native1080_baseline.json`. This distinct capture cannot
support a matched 720p-versus-1080p quality claim. The runner accepts explicit
larger disparity counts; no output is resized into a higher-resolution measure.

Pinned RAFT-Stereo and IGEV++ research checkouts/checkpoints also ran outside
production on this native 720p frame, on CPU with independent eye passes and
no resizing. Their LR-filtered prediction coverages were 77.39% and 74.11%.
These are prediction coverage diagnostics, without calibrated uncertainty,
absolute accuracy or Vulkan parity. FoundationStereo and Fast-FoundationStereo
have pinned source identities but have not run. Full learned-reference and
single-pair matrices, physical qualification and motion reconstruction remain
open; no candidate is promoted from these counts.

### Adaptive right-eye memory error found during ablation

The expanded CPU shader sweep crashed after three frames. Its retained core
points into adaptive matching: the luma pointer already selects one eye, while
six reads indexed it with stereo-global `t`. Right-eye reads therefore exceeded
their image. They now use eye-local `i`. A bounded regression changes only memory
after the stereo input; before the fix it changes disparity and validity, and
after the fix all tested outputs remain identical. The native build and 53
focused tests passed for that increment; subsequent full verification is below. The corrected sweep
completed 108 outputs across nine frames, with 99 byte-exact fast diagnostic
checks; the failed run remains a failure witness. Earlier coverage artifacts retain
their original source identities and do not qualify this corrected matcher.

`benchmarks/results/v14_live_family_ablations.json` retains support gains/losses
and independent D256 SGM agreement for all twelve variant bundles. Mean measured
coverage is 12.57% for the baseline, 17.76% with independent-right work, 36.85%
with adaptive work disabled and 13.06% at D256. Among gained pixels with an SGM
reference, respectively 33.2%, 42.9% and 31.3% disagree by more than two pixels.
SGM is not ground truth. Increasing radius to 4/8 reduces total coverage on
these inputs. The wide-prior variant does not search pixels lacking a usable
prior, and the descriptor/ultra flags bundle several operations; these results
do not isolate every individual operation or establish physical correctness.

### Known-geometry analytic control

The existing ray renderer generated a native 720p control using the factory
camera geometry. Truth is independently intersected at each rectified centre
ray, with no resized truth. Fixed support includes all 519,557 known left rays
at axial Z in [0.5,3] m, irrespective of estimator output or right-eye visibility.
`benchmarks/results/v14_analytic_live_ablations.json` retains twelve variants,
region scores and search-exclusion masks. This is synthetic accuracy evidence;
it does not qualify physical calibration, radiometry or sensor accuracy.

The baseline measures 28.17% of fixed support but only 15.66% is correct under
the frozen tolerance. Thin-object correct completeness is 4.92%. There are
113,904 pixels without a usable prior and 174,855 supported pixels whose nearest
correct integer disparity lies outside the actual search interval. Disabling
adaptive work reaches 38.94% coverage and 20.54% correct completeness on this
single configuration; its increased incorrect output prevents a promotion claim.
The next search experiments must distinguish missing priors, excluded correct
candidates and ambiguity, and preserve the physical qualification boundary.

### Opt-in search and descriptor experiments

Two independent research controls are available in the CPU shader backend and
native refinement benchmarks: `search_fallback` / `--search-fallback 1`, and
`ordinal_census` / `--ordinal-census 1`. Both default to false. The first expands
missing or contradicted priors within the configured disparity range, retaining
the previous local proposal when an extra search cannot confirm a replacement.
It evaluates at most 2D Census candidates per pixel and uses an additional 3x3
photometric confirmation for full-search proposals. The second selects true
ordinal 5x5 Census for fine matching without changing coarse matching or hybrid
aggregation. These are research options, with no ROS profile promotion.

Diagnostic method 2 denotes a full-range candidate pool; method 3 denotes a
local pool restored after an unconfirmed extra search. Check validity separately.
Lower/upper describe the explored range, while winner/cost/candidate count
describe the selected pool. Evaluated work includes both sweeps. Reason bits
4096/8192/16384 record full search, unconfirmed search, and photometric rejection;
they can be warnings on an accepted retained local observation.

| Synthetic control variant | Correct completeness | Measured coverage |
|---|---:|---:|
| Baseline | 15.66% | 28.17% |
| Search recovery | 19.22% | 31.67% |
| Ordinal Census | 20.64% | 35.01% |
| Both | 23.00% | 37.22% |

`benchmarks/results/v14_selective_search.json` binds the synthetic and nine-frame
recorded comparisons, including 36 byte-exact diagnostic checks and per-cohort
bad-1/2-pixel rates with both fixed-support and measured-only denominators.
Every experimental variant exceeds the nominal 0.25-point bad-2-pixel regression
limit in at least one cohort. The decision is **no promotion**. The first search
prototype discarded useful local proposals and reached only about 8% correct
completeness; its source snapshot and failure witnesses remain retained.

Twenty focused CPU controls and four CPU/Vulkan combinations pass. The original
baseline's numerical, total-coverage and planar-path checks remain. Experimental
variants producing fewer planar fills must replace those fills with measured
output. All 29 retained default CPU exports remain byte-identical. That stage
reported 57/57 CTests, 3597 pytest passes and 57 GPU-marked passes, with the
unchanged performance-gate failure retained under concurrent GPU work.

### Strong reference accuracy on the same analytic input

`benchmarks/results/v14_sgm_analytic_reference.json` retains independent-eye D256
references on exactly the same native input and 519,557 known support pixels.

| Cost / aggregation | Correct completeness | Measured coverage |
|---|---:|---:|
| Normalized photometric/gradient, 4 paths | 54.55% | 74.35% |
| Normalized photometric/gradient, 8 paths | 55.56% | 75.46% |
| Census, 4 paths | 32.36% | 53.93% |
| Census, 8 paths | 34.11% | 56.78% |

This establishes synthetic headroom in cost and aggregation beyond search alone.
It does not select a live estimator, qualify physical accuracy, or establish
scene-level confidence intervals. Subpixel fits and thresholds remain development
hypotheses; physical validation and the rest of the full campaign remain open.

### Discontinuity-aware subpixel LR reference

The reference runner exposes `--reference-lr-sampling linear`,
`--reference-lr-tolerance` and `--reference-lr-max-jump`, all in native disparity
pixels. Nearest sampling remains the default. Linear verification requires every
contributing right tap to be reliable and finite and rejects disparity jumps
above the configured limit. An exact integer correspondence needs one tap.
Coordinates outside either image are rejected; no border clipping grants support.

The retained native 720p analytic study evaluates twelve LR settings using the
same independently estimated left/right maps. At tolerance 1 and maximum jump 1,
linear LR yields 54.08% correct completeness versus 55.56% for nearest, while
bad-2-pixel output in a fixed two-pixel edge band drops from 10.55% to 9.03% of
that band's known support. At tolerance 0.25, correct completeness is 48.91%
versus 48.83%, with edge-band bad-2 rates 5.75% versus 6.92%. This is a measured
trade-off on one synthetic scene, with no physical or live promotion.

The default's twelve retained fields remain byte-exact on the saved small
input, and depth/disparity/validity match the earlier full native-resolution
control exactly. The reference now exports independent right validity and
separate LR discontinuity/support-failure masks.

### Current software verification and remaining qualification

The completed `2026-11-05-subpixel-lr-verification` run passes 57/57 CTests,
3609 pytest tests (170 skipped), and 58 GPU-marked tests (2 skipped).
The unchanged 4.167 ms microbenchmark gate passes in both pytest runs.
Protected training PIDs were subsequently observed absent; none was signalled.
This changed-workload observation does not establish isolated live performance.
Independent review found no Critical/Important code defects.

These results verify this software stage. Surveyed physical geometry, qualified
timing and rig validation, scene-level promotion evidence, native stereo VO,
motion-aware history, production profiles and full live acceptance remain open.
The experimental matcher controls stay disabled by default.
