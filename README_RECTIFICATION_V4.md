# StereoForge — reciprocal rectification / SGM diagnostic v4

Non-production CPU evaluation using the already pinned rectified left/right fixture and optional dumped left/right disparities from `daxa_depth_benchmark`. No model calibration, shader, LR tolerance, or temporal filtering changes.

## Install on fish shell

Download the ZIP and run:

```fish
cd ~/git/stereoforge
unzip -n ~/Downloads/stereoforge-rectification-reciprocal-v4.zip -d .
uv run pytest -q tests/test_stereo_rectification_v4.py tests/test_stereo_rectification_v3.py
```

The new script imports the previously installed `benchmarks/stereo_rectification_v3.py` and `benchmarks/stereo_epipolar_probe.py`. The ZIP includes these dependencies for fresh setups. `unzip -n` does not overwrite existing files.

## Run fixture plus independent SGM comparisons

```fish
cd ~/git/stereoforge
set diag artifacts/depth-video/native-stages-20261030-173024-298892311
set out artifacts/depth-video/rectification-v4-(date +%Y%m%d-%H%M%S-%N)
mkdir -p "$out"
uv run python benchmarks/stereo_rectification_v4.py \
  --fixture configs/cameras/zed2/fixtures/SN26041639-fixture \
  --sgm-prefix "$diag/run-1" \
  --output "$out/summary.json" \
  --image "$out/reciprocal-overlay.jpg" \
  --max-disparity 128 --vertical-radius 8 --samples 350
xdg-open "$out/reciprocal-overlay.jpg"
```

This requires previously generated `run-1.left.disparity.f32`, `run-1.right.disparity.f32`, `run-1.left.validity.u32`, and `run-1.right.validity.u32`. If the dump files are unavailable, omit `--sgm-prefix` to run only the image-based probe.

## Interpreting results

- `forward.high_confidence`: NCC match candidates according to v3's heuristic score/uniqueness margin.
- `reciprocal.strict_reciprocal`: both match directions return to the same x/y pixel (±1 px) and pass NCC thresholds in both directions.
- `reciprocal.abs_dy`: absolute row misalignment **only on reciprocally matched samples**.
- `fraction_abs_dy_gt2_and_gain_gt08`: fraction of the reciprocal subset whose vertical match differs by >2px and improves NCC score by >0.08 relative to same-row matching.
- `tiles`: 4×3 spatial groups. Consistent signed offsets covering multiple tiles would motivate independent calibration-target verification; a few concentrated outliers in repetitive texture would not.
- `sgm_comparison`: compares the native left and right disparity estimates (on their own matched pixel locations) to image-based NCC disparity. Evidence is deliberately sparse/texture-biased. It is **not** a ground-truth stereo quality score.

If both independently matched disparities have low agreement with NCC even when `abs(dy)<=2`, inspect Census, independent Right cost and SGM selection. If many reciprocal patches reliably show a consistent nonzero vertical shift, physically revalidate rectification/calibration and actual capture mode first. Avoid increasing LR tolerance or editing camera calibration based on this sparse diagnostic alone.

## Provenance and limits

This reads one **retained fixture frame**, not live raw captures; it does not certify that the active camera's geometry matches its calibration. A left/right wrong pair can still pass NCC cycle consistency on repeated patterns; ORB RANSAC itself can fit false correspondences. The output has no per-pixel depth ground truth. Treat it as triage only.

No Git action is performed. Resolve your outstanding `(MERGING)` state separately before committing anything.
