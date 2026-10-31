# StereoForge LR / confidence / Census audit v6

Offline diagnostic tools for understanding why most independently selected left and right SGM disparities do not survive left-right (LR) consistency. These tools **do not modify** shaders, the ROS publisher, calibration, or live depth output.

## Evidence motivating this experiment

On the pinned ZED 2 fixture at 1280x720 and 128 disparities, the native pipeline selected ~85.9% of left and ~85.2% of right disparities, but accepted only ~19.5% of all pixels after LR consistency. OpenCV StereoSGBM with block size 5 and 5% uniqueness produced ~33.4% LR acceptance overall and ~78.4% among mutually selected pairs, using a different matching cost and validity policy. **Neither LR agreement nor OpenCV output is depth ground truth.**

The native Slang selector accepts a minimum when `best > 0 && best_cost < 10000 && second > best_cost`; its confidence is `(second-best_cost)/max(second,1)`. Test whether this confidence discriminates stable and inconsistent matches, and whether raw 5x5 Census is already ambiguous.

## 1. Unit tests

From the repository root:

    uv run pytest -q tests/test_lr_confidence_audit_v6.py

Requires Python with NumPy and pytest, plus the pinned fixture for the actual audit. Unit tests do not require a GPU.

## 2. Audit an existing native diagnostic dump

In fish:

    set native artifacts/depth-video/native-stages-20261030-173024-298892311/run-1
    set output artifacts/depth-video/lr-confidence-v6-(date +%Y%m%d-%H%M%S-%N).json

    uv run python benchmarks/lr_confidence_audit_v6.py \
        --fixture configs/cameras/zed2/fixtures/SN26041639-fixture \
        --native-prefix $native \
        --disparities 128 \
        --stride 8 \
        --output $output

The input prefix must have per-eye `.disparity.f32`, `.confidence.f32`, and `.validity.u32` data from `daxa_depth_benchmark --dump-prefix`. The fixture's expected 32-bit luma planes are hash-verified against its manifest.

Compare confidence-based LR acceptance **and** the fraction of the entire image retained. A high-confidence subset may look consistent only because it drops difficult pixels. A raw single-pixel Census winner-takes-all result is **not** a complete SGM test.

## 3. Sweep P1/P2 on the same fixture

Build the existing benchmark first and run the script from the repository root:

    pixi run -m ros_ws cmake --build build/ros-daxa-native \
        --target daxa_depth_benchmark -j 8

    bash scripts/run-stereo-penalty-sweep-v6.sh

This runs P1/P2 = (4,16), (8,32), (12,48), (16,56) with the existing 4-path 128-disparity configuration. All stay inside the packed penalty representation's bound. The script writes separate native dumps and confidence/Census reports. It uses the host Fedora RADV and libdrm paths; run it only on the intended AMD workstation. Pause competing GPU workloads.

## 4. Optional OpenCV uniqueness control

If `benchmarks/stereo_reference_crosscheck_v5.py` is installed in the checkout, repeat the previous exact-fixture OpenCV comparison using `--uniqueness 0` versus `5`. Compare both image-wide coverage and LR agreement on mutually selected pairs. The optional v5 script is not required for v6.

## Interpretation and limits

- Higher native confidence strongly predicts LR agreement: investigate calibrated confidence/uniqueness gates, measuring the lost support and thin-structure impact.
- High-confidence native disparities still disagree: investigate Census cost construction, right and left aggregation, geometry, and false minima.
- Raw Census best and second-best costs are often tied: local descriptors are ambiguous; consider additional photometric support or discriminative cost before altering SGM.
- Raw Census looks distinctive but native output does not: inspect path aggregation and selection.

The fixture is a **single retained image**, and internal LR consistency is **not metric depth accuracy**. Do not deploy penalty or confidence changes based only on these measurements. Validate later with ground truth, dynamic scenes, thin objects, and a latency budget.

Files: `benchmarks/lr_confidence_audit_v6.py`, `tests/test_lr_confidence_audit_v6.py`, and `scripts/run-stereo-penalty-sweep-v6.sh`.
