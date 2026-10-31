# StereoForge v5 — independent stereo-estimator cross-check

This is a **diagnostic-only** script. It reads the fixed, SHA-256 verified 1280×720 ZED2 rectified RGB fixture and the native left/right SGM exports from `daxa_depth_benchmark --dump-prefix`, then computes independent OpenCV StereoSGBM results from the *same* images.

The most recent reciprocal-NCC experiment retained only 119 features. Its 57.1% `left_right_agree_at_independent_patch` compares native left and right disparities **at NCC-selected x/y coordinates**, which can have nonzero vertical offsets. It is **not** identical to StereoForge's production same-row LR acceptance; do not compare 57.1% directly with 25.7% native LR agreement. Also, reciprocal NCC and ORB matches can be misleading on repetitive texture.

## Why this diagnostic

The native fixture produced 85.9% left selection, 85.2% right selection, but ~19.6% final LR-valid output. Bitwise repeatability and zero CPU-mask disagreements suggest a *systematic, deterministic* correspondence problem. We need a second algorithmic viewpoint before changing matching code or calibration.

OpenCV SGBM differs from the native implementation in block matching, penalty scale and aggregation. A higher LR consistency score from OpenCV **is not proof** of correct depth. This is a diagnostic for likely origin and where to focus a controlled benchmark.

## Install without changing Git state

From `~/git/stereoforge` (fish shell):

```fish
unzip -n ~/Downloads/stereoforge-stereo-reference-crosscheck-v5.zip -d .
uv run pytest -q tests/test_stereo_reference_crosscheck_v5.py
```

No CMake, ROS, camera or GPU is required. The scripts are standalone; you can keep the existing `(MERGING)` state untouched.

## Run on the *same frozen* ZED2 fixture + native dumps

```fish
cd ~/git/stereoforge
set native artifacts/depth-video/native-stages-20261030-173024-298892311/run-1
set out artifacts/depth-video/stereo-reference-v5-(date +%Y%m%d-%H%M%S-%N)
mkdir -p $out

uv run python benchmarks/stereo_reference_crosscheck_v5.py \
    --fixture configs/cameras/zed2/fixtures/SN26041639-fixture \
    --native-prefix $native \
    --disparities 128 \
    --blocks 3 5 \
    --uniqueness 5 \
    --output $out/summary.json \
    --image $out/comparison.png

xdg-open $out/comparison.png
```

Expected inputs (verified byte lengths): `run-1.{left,right}.{disparity.f32,validity.u32}`; manifest references `expected/rect_left.rgb8` and `expected/rect_right.rgb8`.

**Optional ablation**: rerun with `--blocks 7` or `--uniqueness 10` and a **fresh output directory**. OpenCV `numDisparities=128`, penalties `8 × block²` and `32 × block²`, no built-in speckle or LR filter, and a separate positive-disparity right map computed by swapping cameras and mirroring both images. LR acceptance is recomputed identically with nearest rounded `x_right=x_left-d_left`, no vertical offset, tolerance 1 px.

## Read the JSON

The summary contains:

- `native_native`: production-style LR consistency of the exported native maps.
- `reference_profiles.*.cv_cv`: independent OpenCV LR agreement.
- `nativeL_cvR`, `cvL_nativeR`: *cross-algorithm* LR agreement; not production settings.
- `native_cv_left`, `native_cv_right`: per-view disparity agreement on pixels valid for both estimators.
- `texture_bins`: coverage and LR agreement by local 9×9 luminance standard deviation (0–4, 4–8, 8–16, 16–32, 32+).

**Interpretation gates:**

1. If OpenCV/OpenCV achieves much higher LR consistency and native/OpenCV per-view agreement is poor, investigate native Census cost volume, right reference, path aggregation and selection. Do **not** assume OpenCV's result is accurate.
2. If OpenCV/OpenCV also fails, inspect calibration/rectification, weak/repetitive texture and camera matching range using a known-depth target.
3. If one cross pairing (native-left/OpenCV-right vs OpenCV-left/native-right) performs much better, investigate a **view-specific** estimator error.
4. If failures correlate with low local texture, investigate photometric ambiguity/uniqueness and confidence, not merely a uniform pixel coordinate correction.

Use a controlled textured planar target or independent depth ground truth to establish actual depth correctness. No change to the production LR threshold, no temporal output expansion and no GPU performance claim are made here.
