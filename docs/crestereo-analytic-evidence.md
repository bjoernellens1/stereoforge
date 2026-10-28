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

## Additional texture-seed sensitivity

Two additional 720p analytic families were generated with texture seeds `1729`
and `8675309` using the existing default calibration, scene geometry and
photometric settings. These are new texture realizations of the **same scene
geometry**, not independent scene layouts, physical images, or a held-out
qualification set. The two settings, seeds and original 1-pixel threshold were
specified before these runs; sigma 2 had already been chosen post-hoc from the
previous scene. All four raw predictions were frozen before either truth was
read for scoring. Inputs are the original full-grid rectified image pairs;
truth remains on that grid and the fixed-support definitions above apply.

| Seed | Sigma | Clean correct / eligible | Thin correct / eligible | Edge correct / eligible | Near correct / eligible |
|---|---:|---:|---:|---:|---:|
| 1729 | 0 | 0 / 816,257 | 0 / 3,164 | 0 / 10,119 | 0 / 33,768 |
| 1729 | 2 | 722,310 / 816,257 | 0 / 3,164 | 2,890 / 10,119 | 8,582 / 33,768 |
| 8675309 | 0 | 0 / 816,257 | 0 / 3,164 | 0 / 10,119 | 0 / 33,768 |
| 8675309 | 2 | 695,191 / 816,257 | 0 / 3,164 | 2,695 / 10,119 | 2,267 / 33,768 |

All four outputs produced finite positive effective disparity for all 61,105
truth-occluded pixels. This is numerical validity only: the model output has
no demonstrated independent correspondence check or calibrated confidence.
The clean-region error counts are likewise numerical predictions differing by
more than one pixel, not validated LR false-accept counts. These results show
sensitivity on this one analytic geometry; they do not establish physical
accuracy, inference throughput or a GPU performance comparison.

The generator and evaluator were run from commit
`c2288d79375bc2857e41d94f66c848f36cbdd1fd`, with generator SHA-256
`dbad448214e2309a399ee37f01bf2f760e828917df89eb1818b6a3b74aa922b6` and
the evaluator/adapter hashes and ONNX CPU runtime settings stated above. The
full input manifest and truth hashes are:

| Seed | Full manifest SHA-256 | Truth manifest SHA-256 |
|---|---|---|
| 1729 | `7efcff15c8a2cbb7237158f3677f46e98f8cf39d3e02fc475d092bb25c6e3698` | `10ab6fe292f090ec0eb5dbc8dd3a4de7b9844ebfc8e06046fcb2dd46edfc848d` |
| 8675309 | `10657c54a25f3f8f9cae07fe7515885a3e1372e6139c6ad9fee2bb4f3e6a4f64` | `2e672ff6e66c55110bb33d7c555e124c6d6df1cfdff475b79b8ae5054fefda2a` |

Frozen output SHA-256 values, in table order, are
`cc79d9c9970f0d1d5659bf04ef8bd398590b9a7675703bea5aca5cd80bc6e2eb`,
`ecf6a713c953fb473a8f0390d33eab716c4f95d6b71fa3c0d52e3e956fb94765`,
`305bb330af8ccf4898b1bfb712cce5347779f01345fa73e66b6f17bbf0f08257`, and
`9fa39e721648512aca8bae1cff26d2d4611160c56f8abf1367823c517259d78e`.

Reproduce from the project root with the same pinned model and Python CPU
runtime specified earlier. Generate both inputs, then freeze *all* predictions,
then evaluate:

```bash
for SEED in 1729 8675309; do
  PYTHONPATH=python:benchmarks .venv/bin/python benchmarks/refinement_quality_fixtures.py \
    --output "/dev/shm/stereoforge-neural-seed${SEED}" --seed "$SEED"
done
for SEED in 1729 8675309; do
  for SIGMA in 0 2; do
    "$PYTHON" benchmarks/crestereo_quality.py freeze --model "$MODEL" \
      --fixture "/dev/shm/stereoforge-neural-seed${SEED}/full" \
      --output "/dev/shm/stereoforge-neural-seed${SEED}-sigma${SIGMA}" --sigma "$SIGMA"
  done
done
for SEED in 1729 8675309; do
  for SIGMA in 0 2; do
    "$PYTHON" benchmarks/crestereo_quality.py evaluate \
      --fixture "/dev/shm/stereoforge-neural-seed${SEED}/full" \
      --frozen "/dev/shm/stereoforge-neural-seed${SEED}-sigma${SIGMA}" \
      --output "/dev/shm/stereoforge-neural-seed${SEED}-sigma${SIGMA}-report.json"
  done
done
```

The generator uses its 1280×720 defaults. Output paths must be fresh, and the
model and its data remain external. Runtime and platform differences can change
floating-point outputs; reproduce with the recorded environment before comparing
hashes. These tests did not alter the production pipeline.
