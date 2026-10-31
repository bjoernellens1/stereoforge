# LR disagreement diagnostic — StereoForge

## Why

On a retained 1280×720 D128 fixture, both independent SGM estimators select disparities for approximately 85% of pixels, yet LR consistency with a 1 px tolerance accepts only about 19.6%. Independent CPU LR verification produced zero mask mismatches and repeated native GPU runs produced bitwise-identical results. The diagnostic target is **correspondence quality and geometry**, not nondeterministic GPU execution or the LR mask implementation.

This script reads already-dumped per-eye disparity and selection-validity arrays and evaluates the **physical** left-to-right relation:

`d_left(x,y) ≈ d_right(round(x - d_left(x,y)), y)`

The matching rule is nearest-integer with `floor(...+0.5)`, as in the CPU pipeline. It also computes **nonphysical controls** (`same x`, `x+d`), horizontal and vertical sampling shifts, disparity-band breakdowns and the distribution of *signed* errors. These are **diagnostics only**. A control looking better does not authorize changing stereo sign or calibration conventions.

## Run (fish)

The `--dump-prefix` must match the existing benchmark run. No camera or GPU work is needed for the analyzer.

```fish
cd ~/git/stereoforge
set diag artifacts/depth-video/native-stages-20261030-173024-298892311
uv run pytest -q tests/test_lr_disagreement_probe.py
uv run python benchmarks/lr_disagreement_probe.py \
  --report "$diag/run-1.json" \
  --output "$diag/lr-disagreement.json" \
  --status-image "$diag/lr-status.png" \
  --stride 2 --horizontal-radius 4 --vertical-radius 2
```

Legend for status image (BGR writing, viewed as RGB): black=left invalid; orange=left valid but right invalid/out of bounds; red=both selected but disagree; green=LR accepted.

## Interpret in this order

1. If the physical relation `x-d` has the highest agreement but low absolute acceptance, investigate **matching ambiguity**: repetitive patterns, poor local texture, raw left-right photometric differences, excessive SGM path sensitivity, and occlusions. Disparity-banded failure counts help determine near/far impact.
2. If `x+d` or `same x` has substantially higher agreement, **investigate** the right-disparity convention and right input/cost construction, but do not invert the production coordinate model based on this result alone.
3. If a constant shift in `x` or `y` improves agreement meaningfully, examine rectification, row alignment and indexing first; small shifts can also spuriously help on repeating structures. Follow up with independent RGB stereo epipolar checks.
4. If residual median is strongly signed rather than near zero, test whether the right view has a systematic disparity bias or subpixel correction difference.
5. Analyze actual left/right *candidate costs* before relaxing LR tolerance. Accepting more mismatched disparities often increases false geometry.

## Limitations

- The native fixture is **one recorded stereo pair**, not the full 90-frame ROS sequence.
- The script uses only final per-eye disparities and selection masks. It cannot measure ground-truth accuracy or decide whether the left or right match is correct.
- Horizontal and vertical controls can overfit ambiguous repetitive image structures.
- The `--status-image` computation is full resolution; `--stride` applies to numerical scans only.
- Continue sequence-based temporal analysis on the original unfiltered recording. Do not enable temporal holding by default without motion compensation and provenance.

## Independent RGB epipolar probe

When native Left/Right disparity agreement is low, first test whether the
*rectified source images* are epipolar-aligned independently of SGM. ORB
features are matched with mutual Hamming nearest neighbors. This is a
**smoke test only**: repetitive image features and outliers can dominate
uncalibrated ORB correspondences. Report both unfiltered epipolar residual
and the conditional positive-disparity subset; do not treat either as ground
truth.

```fish
set fixture_dir configs/cameras/zed2/fixtures/SN26041639-fixture
uv run python benchmarks/stereo_epipolar_probe.py \
  --fixture "$fixture_dir" --max-disparity 128 \
  --output "$diag/epipolar-orb.json" \
  --image "$diag/epipolar-orb.jpg"
```

If *independent, high-quality* left/right feature matches have large vertical
residuals, inspect rectification/calibration and row indexing. If feature
matches follow epipolar rows but native LR agreement remains poor, inspect
stereo cost/aggregation and right-view selection first.
