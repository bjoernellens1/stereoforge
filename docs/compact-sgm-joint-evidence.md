# Joint Census/luma SGM: CPU development experiment

On the inspected synthetic scene, the joint local cost improves thin-region completeness with unchanged exposure, but fails the predeclared per-surface false-accept gate. Under a synthetic +25 right-image brightness shift, its thin completeness falls below the matched Census baseline. Keep it out of production pending better photometric robustness and independent quality validation.

This experiment runs independent four-path SGM in both eyes, integer winners, D128, LR tolerance 1 and the same metric validity contract. The baseline cost is `2*Hamming`; joint adds `floor(min(abs(centerL-centerR),32)/4)` and `floor(min(abs(horizontal_gradientL-horizontal_gradientR),64)/8)`, where gradient is `I(x+1)-I(x-1)`. Both use P1=16/P2=64 and identical supported Census footprints and frozen intervals. The scale preserves the original Census/P1=8/P2=32 recurrence: independent scalar tests verify paths scale by two, and unmodified-image baseline selections (raw disparity/validity, LR/metric masks and metric depth) match pinned v2 for both compact and full-D. This selection check does not attest full 720p path-score parity.

| Correct metric completeness %, left / right | Compact baseline | Compact joint | Full-D baseline | Full-D joint |
|---|---:|---:|---:|---:|
| Identity, all | 95.279 / 95.317 | 95.509 / 95.524 | 95.339 / 95.371 | 95.579 / 95.601 |
| Identity, thin | 45.101 / 46.239 | 57.427 / 55.973 | 60.714 / 62.263 | 71.871 / 71.113 |
| Identity, edges | 61.073 / 61.451 | 64.394 / 63.671 | 65.273 / 64.894 | 69.029 / 68.360 |
| +25 clipped, thin | 44.943 / 46.081 | 44.848 / 45.480 | 60.619 / 62.168 | 55.341 / 56.100 |
| 0.8× + 20, thin | 45.070 / 46.176 | 52.465 / 51.296 | 60.872 / 62.453 | 65.013 / 64.475 |

Each exposure uses its own matched baseline; the original per-eye intervals are retained across perturbations. The perturbations are deterministic synthetic uint8 transforms on the right image, with Census recomputed, not measurements of the camera ISP or a re-estimated live prior. The +25 transform saturates 45,725 of 921,600 right-image pixels at 255. Correct completeness uses a fixed valid, visible, in-range, unmixed truth domain with Census/remap support; each eye has 3,164 eligible thin pixels. D128 excludes the portion of near field requiring higher disparities.

The predeclared exploratory gate requires, for every eye/exposure/interval, ≥5 percentage points thin completeness gain, ≤1 point loss in all/near/edge completeness, no increase in whole-thin-region false LR accepts and no increase in whole-surface false LR accepts for any surface having eligible thin support; empty or unknown support fails. Percentage boundaries use exact integer count arithmetic. Compact identity passes left but fails right (surface IDs 6 and 8); full-D identity fails both eyes (surface 8). Full-D identity surface-8 false accepts rise from 99 to 121 left and from 76 to 154 right. Full-D +25 fails both eyes, including thin, edge and false-accept checks. Compact +25 fails both eyes. The output records every gate and per-surface count; these are exploratory development criteria, not user-frozen release thresholds.

All 12 configuration selections were written and hash-checked before any truth was opened. Each mode/eye uses the same candidate work for the two costs: compact 206,615,684 / 205,233,948 four-path interval slots and 190,319,284 / 189,599,732 supported updates; full-D 471,859,200 / 471,859,200 slots and 444,492,800 / 444,492,800 supported updates. This CPU runner materializes dense aggregate scores; these figures are not GPU timings, transfer-inclusive cost or live throughput.

The reviewed fixture run exited 0; focused tests `PYTHONPATH=python:. .venv/bin/python -m pytest -q tests/test_compact_sgm_joint.py` passed 8. Its ignored local result is `artifacts/compact-sgm-joint-20261028-v2/results/result.json` (SHA-256 `3cd89a0e5236e41702316a88458465deb3cf3bd69b3f586259444a8ec8f5e981`), with 12 adjacent frozen selection files and their hashes recorded in the report. Runner SHA-256 `34e54f1c5fea6d8deae1179c981b1c80ad8f72ddfb2c4a793ce32ac80a1cc0e3`; build manifest SHA-256 `a7bacae454a6f70df19b42721a6989d84fcf4b067fc3a54c3b29aff340eefe8e`; source `compact_sgm_joint.py` SHA-256 `b1951172453418a905cbdd03f1f6c12331fba3c5cc04a3d5cdcf179252d4c91b` and `compact_sgm_joint_runner.cpp` SHA-256 `385d73cb23d6247800626a3d70fcf03087f8db3c09ac701eeed45a2ca13020ba`. Local artifacts may be absent on a fresh checkout.

To reproduce with the recorded fixture and an empty output directory:

```bash
F=artifacts/fast60-refinement-20261028/analytic-family-720p
N=artifacts/daxa-refinement-20261028/root-frozen-numeric-matrix-v1/r2-subpixel0
A=artifacts/compact-sgm-step-20261028-v2
R=artifacts/compact-sgm-joint-20261028-v2
O=/tmp/stereoforge-joint-sgm-repro
PYTHONPATH=python:. .venv/bin/python benchmarks/compact_sgm_joint.py \
  --runner "$R/runner" --build-manifest "$R/build.json" --artifact-root "$A" \
  --family-root "$F" --native-report "$N/native.json" --dump-root "$N/planes" --output-root "$O"
```
