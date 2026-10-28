# Census-anchor affine exposure correction: CPU development experiment

On the inspected synthetic scene, an affine fit from Census LR-consistent correspondences recovers much of the joint matcher's loss under synthetic brightness changes. It still fails the predeclared surface-level false-accept gates in both eyes for every full-D exposure; it is not qualified for production or live-camera use.

The fit uses only the pinned `2*Census` baseline's left-reference, LR-valid correspondences and current rectified images. It projects each left integer disparity to the right image and verifies both Census support masks. It excludes 0 and 255 center samples. Reference-image 64×64 tiles use `(tile_x + 3*tile_y) % 4`: residues 0–2 train, 3 validate, taking at most the first 64 eligible points per tile in raster order. At least 1,024 / 256 samples, 16 / 4 tiles and right-luma 10–90 span ≥32 are required for train / validation. Ordinary least squares with intercept is refit three times after rejecting residuals >12; ≥512 inliers, gain [0.75,1.33], absolute offset ≤40, validation median residual ≤8 and p90 ≤20 are required. Failed fits select the exact frozen Census baseline. The matcher reuses the original perturbed-image Census descriptors and frozen D128 intervals. The corrected joint cost falls back per candidate to Census if either original left sample or original/corrected right sample for its center or horizontal gradient is 0 or 255.

All six fits accepted. They used 11,520 training and 3,840 validation anchors, from 180 and 60 tiles respectively. For full-D, identity estimated gain 0.99986 and offset 0.0339; synthetic +25 estimated 0.99999 and −24.982; synthetic `0.8*x+20` estimated 1.24962 and −24.920. Validation p90 residuals were 0.030, 0.018 and 0.550 intensity units. Training and validation occupy distinct tiles but remain correlated samples from the same frame and same matcher. Raster sampling can bias the fit; endpoint filtering detects only observed 0/255 values and cannot recover already clipped texture.

| Correct metric thin completeness %, left / right | Census baseline | Uncorrected joint | Corrected joint |
|---|---:|---:|---:|
| Compact identity | 45.101 / 46.239 | 57.427 / 55.973 | 57.427 / 55.973 |
| Compact +25 | 44.943 / 46.081 | 44.848 / 45.480 | 57.554 / 55.657 |
| Compact 0.8×+20 | 45.070 / 46.176 | 52.465 / 51.296 | 57.427 / 55.910 |
| Full-D identity | 60.714 / 62.263 | 71.871 / 71.113 | 71.871 / 71.113 |
| Full-D +25 | 60.619 / 62.168 | 55.341 / 56.100 | 71.555 / 70.417 |
| Full-D 0.8×+20 | 60.872 / 62.453 | 65.013 / 64.475 | 71.713 / 70.986 |

Each comparison uses the same input and per-eye intervals; completeness is against the fixed clean, visible, in-range, unmixed truth domain (3,164 eligible thin pixels per eye). +25 saturates 45,725 right-image pixels; 875,875 of 921,600 right pixels are individually marked safe; a candidate additionally requires all six stencil samples across both eyes to be safe. Correction does not restore clipped detail. The perturbations are deterministic synthetic transformations, not measured ZED sensor exposure or ISP response. Fixed prior intervals are not recomputed for those exposures.

The same exploratory gates as the preceding joint-SGM experiment require ≥5 percentage points gain in thin completeness, ≤1 point loss in all/near/edge completeness, no increase in whole-thin-region false LR accepts, and no increase in whole-surface false accepts for any eligible thin surface. Exact integer arithmetic enforces the boundaries. For full-D +25 surface 8, baseline / uncorrected / corrected false accepts are 99 / 121 / 116 left and 76 / 87 / 154 right; both corrected eyes fail. Compact identity and 0.8×+20 pass left but fail right; compact +25 fails both eyes. These criteria are exploratory, not user-frozen release gates.

The prior baseline and uncorrected joint reports and all 12 selection hashes were verified against the pinned earlier experiment. All six fit metadata files and corrected/fallback selections were written and checked before truth was opened. All fits were accepted on this fixture; exact rejected-fit fallback is verified by a focused test, not a full-scene refusal run. The run exited 0 after independent review; `PYTHONPATH=python:. .venv/bin/python -m pytest -q tests/test_compact_sgm_affine.py` passed 11 tests. CPU workload is comparable to the prior full SGM and has not been benchmarked as a GPU or live pipeline.

The ignored local report is `artifacts/compact-sgm-affine-20261028-v1/results/result.json` (SHA-256 `f90ba3161b8d07e9cd3cba5613cf3a31fadaa104d11c7d77a22bf217c8a5ac89`); adjacent fit and selection hashes are in that report. Runner SHA-256 `b6490158ad6a770b98d65bc4e17274abb5cd4a1a348d17bdcb93b60f8b20ec47`; build manifest `7834a6da547a33b01304d80c3bede2ab9dce836082a0024e390946e4b7602014`. Source hashes: `compact_sgm_affine.py` `6316e6e14cebcccb162044540b53a057e115dd89df2b7401691be2b6ae8317d9`; `compact_sgm_affine_experiment.py` `cb12b69d1c2fa4fd2a2ee64e3acab3f6b4d98c9f44cd86c08fa26bb057f6d089`; `compact_sgm_affine_runner.cpp` `caa2a11846ade8eb966f3dee830a835d990770602beb2fee042f003907b2f3b7`. Local artifacts may be absent on a fresh checkout.

To reproduce with the existing pinned inputs and a fresh output directory:

```bash
F=artifacts/fast60-refinement-20261028/analytic-family-720p
N=artifacts/daxa-refinement-20261028/root-frozen-numeric-matrix-v1/r2-subpixel0
A=artifacts/compact-sgm-step-20261028-v2
P=artifacts/compact-sgm-joint-20261028-v2/results
R=artifacts/compact-sgm-affine-20261028-v1
O=/tmp/stereoforge-affine-repro
PYTHONPATH=python:. .venv/bin/python benchmarks/compact_sgm_affine_experiment.py \
  --runner "$R/runner" --build-manifest "$R/build.json" --artifact-root "$A" \
  --previous-root "$P" --family-root "$F" --native-report "$N/native.json" \
  --dump-root "$N/planes" --output-root "$O"
```
