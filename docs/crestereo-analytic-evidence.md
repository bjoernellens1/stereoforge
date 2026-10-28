# CREStereo: analytic quality and input-filter sensitivity

This is an **offline CPU development experiment**, not a neural Vulkan benchmark,
physical-camera validation, a held-out evaluation, or a candidate for promotion.
The high-frequency analytic scene, single model and nonuniform resize are a narrow
stress case. The optional prefilter was selected **after inspecting** the baseline
failure; no threshold or configuration has been frozen for held-out evaluation.

## Input, model and geometry

The input is the rectified 1280×720 left/right image pair from
`artifacts/fast60-refinement-20261028/analytic-family-720p/full`, the same analytic
scene and input manifest used by the [Fast60 quality evidence](fast60-refinement-evidence.md).
The manifest SHA-256 is `2799d18f1bfa13bf862817aabaf05c917055af371472e162467437bc837c7ea0`;
the independent truth manifest SHA-256 is
`adee08e0cd5998f6864416d1a441206f9eb79aa6a363c68a462a59ecc29cae5c`.
These are analytic centre-ray truth, not recorded camera images or physical truth.

The model is the pinned Qualcomm ONNX and external weights described in the
[feasibility report](crestereo-vulkan-feasibility.md); the ONNX and weights
SHA-256 are `0f0c59b353c09b571508791b0ecf5dcf657a389d1934a0912db5d2c52e897de9`
and `8e09346b8300891f72440b436eba8b1106f3eafd49fb7d0bad490ae6857dfd82`.
The grayscale scene has identical RGB channels. The adapter resizes RGB to
320×240, converts to BGR float32 [0,1], and constructs the model's 160×120
inputs using aligned-corner bilinear sampling. OpenCV's pixel-centre resize
maps x as `(x+0.5)*320/1280-0.5`; the y scale is 240/720. The output is
bilinearly resized onto the original grid and its assumed horizontal pixel
units multiplied by 4; it is interpolated, including across surface boundaries.
Truth stays on its original grid. This assumption is supported only by the
limited behavioral witnesses below, not a full calibration of model outputs.

The left clean domain matches the native comparison denominator: visible,
in-range centre-ray truth with rectification support, full 5×5 Census support,
and no mixed raw-image footprint. It has 816,257 pixels. A broader visible
truth domain and regions are also measured. A finite effective disparity
`d-(cx_left-cx_right)>0` is merely **numerically valid**, not verified stereo
correspondence or confidence. False accepted counts in this report mean
numerically valid predictions more than one disparity pixel from truth; they
are not independent left/right consistency accepts. The test calibration has
`fx=528`, baseline `0.12 m` and principal-point offset zero.

## Signed-shift behavioral witness

With fixed random 320×240 texture and central support, model medians were
+7.9934 for +8 and +15.9916 for +16 pixel shifts. Identical inputs yielded
+4.7394; −8 and −16 yielded +91.5854 and +172.1960. Swapping the +8 pair
produced +91.6465. These examples support the positive-shift scale in those
cases and show that zero/negative disparity can return large positive values;
they do not prove signed range or general accuracy.

## Frozen predictions and quality

Predictions were frozen before reading analytic truth. The sigma-0 output SHA-256
is `09dbfe1e377f2fbe3c27cb8df5fbb5a752e4c99b7a727ce24ac683cf5524d457`.
Its model-grid minimum/median/maximum was 326.316 / 340.932 / 354.705 pixels;
the median exceeds the 320-pixel model width. Independent input inspection
confirmed the model's declared NCHW input/output shapes, BGR [0,1] inputs,
and matching grayscale channel data. The result should not be attributed to
a specific model defect: direct 4× subsampling of the high-frequency scene is
a substantial aliasing confound.

The post-hoc sigma-2 run Gaussian-filters each full-resolution eye before the
same model resize; its frozen output SHA-256 is
`8f012994635f40d2c91800933f2fbef6499d6a91246de102f87110f4e1e4a851`.
Correct completeness below means the fraction of fixed-support truth pixels
whose numerically valid prediction is within one original-grid disparity pixel.

| Domain | Pixels | Sigma 0 | Sigma 2, post-hoc | Full SGM reference |
|---|---:|---:|---:|---:|
| Clean visible in-range | 816,257 | 0% | 88.20% | 95.26% |
| Thin | 3,164 | 0% | 0% | 60.46% |
| Edges | 10,119 | 0% | 29.16% | 65.02% |
| Near | 33,768 | 0% | 5.58% | 99.04% |

The historical SGM numbers are from the linked report's independently checked
native binary and the same support; they are not a matched CPU/GPU latency
comparison. Sigma 2 produces 96,345 incorrect numerically valid predictions
in the clean domain and finite positive effective disparity for all 61,105
truth-occluded pixels. It does not provide a qualified occlusion mask. In the
4-pixel edge neighborhood, correct completeness is 39.36%, compared with
90.83% elsewhere; this measures boundary impact without claiming that every
error was caused by interpolation. One sigma-2 CPU prefilter invocation took
5.87 ms; this is one sample and does not establish performance.

## Reproduction

The model and weights are external and their redistribution terms remain
unverified. No model weights are committed. Use the pinned model directory,
the existing analytic fixture, and an isolated Python environment with
ONNX Runtime CPU support. The recorded environment used Python 3.13.12,
ONNX Runtime 1.31.1, NumPy 2.5.4, OpenCV 5.0.0, 4 ONNX intra-op threads,
1 inter-op thread and 1 OpenCV thread. The final freeze records script and
input hashes and the exact settings. Its quality script SHA-256 was
`05b20aa671da5786061ab5b07bcc54f856426feea7e225ef1485948f7e95ff2f`, and
its adapter SHA-256 was
`0b09adc2c850154c1a1555c5c7d9118e6495420705270684ab638a0767534296`.
Before/after hashes are not an atomic loading guarantee.

```bash
MODEL=/path/to/crestereo-onnx-float
FIXTURE=artifacts/fast60-refinement-20261028/analytic-family-720p/full
PYTHON=/path/to/onnx-runtime-environment/bin/python
"$PYTHON" benchmarks/crestereo_quality.py witness --model "$MODEL" --output /tmp/crestereo-witness.json
"$PYTHON" benchmarks/crestereo_quality.py freeze --model "$MODEL" --fixture "$FIXTURE" --sigma 0 --output /tmp/crestereo-sigma0
"$PYTHON" benchmarks/crestereo_quality.py freeze --model "$MODEL" --fixture "$FIXTURE" --sigma 2 --output /tmp/crestereo-sigma2
"$PYTHON" benchmarks/crestereo_quality.py evaluate --fixture "$FIXTURE" --frozen /tmp/crestereo-sigma0 --output /tmp/crestereo-sigma0-quality.json
"$PYTHON" benchmarks/crestereo_quality.py evaluate --fixture "$FIXTURE" --frozen /tmp/crestereo-sigma2 --output /tmp/crestereo-sigma2-quality.json
```

Both freezes precede evaluation. Outputs must not already exist. The artifact
selection for sigma 2 was nevertheless post-hoc; freeze order does not make it
held-out. Run the small geometry, masking, input and hash tests with
`TMPDIR=/dev/shm .venv/bin/python -m pytest -q tests/test_crestereo_quality.py tests/test_crestereo_cpu_smoke.py`.

An exploratory private eye-split ONNX rewrite matched CPU outputs on two inputs,
but its pnnx 20261020 attempt reported rank warnings and hit disk quota with empty
conversion output. The rewritten graph SHA-256 is
`4e8efbcd2f0b7720d3b0bb8201b67562675c3741a2163ef71b4d72d119c6a4e5`; the transient log SHA-256 is
`2a16e57f3237815c86acef89f4c569871d500f31bd688eaab96752c6867e56af`. No rewritten model is distributed and Vulkan inference remains
unmeasured. The earlier unmodified conversion attempt and license limits are in
the [feasibility report](crestereo-vulkan-feasibility.md).
