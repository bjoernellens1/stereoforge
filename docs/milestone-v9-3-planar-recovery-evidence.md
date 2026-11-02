# StereoForge Milestone v9.3: Provenance-Tagged Edge-Aware Planar Recovery & Color Tie-Break Causal Qualification (`240+ FPS`)

- **Tracked Qualification Artifact:** [`benchmarks/results/v9_3_planar_qualification.json`](../benchmarks/results/v9_3_planar_qualification.json)
- **Observation Contract Reference:** [`docs/daxa-observation-contract.md`](daxa-observation-contract.md)
- **v9.2 Live 60 Hz & Selective Repair Baseline:** [`docs/milestone-v9-live-and-selective-refinement-evidence.md`](milestone-v9-live-and-selective-refinement-evidence.md)

---

## 1. `[P0]` Observation Contract & Provenance Separation (`Provenance.PLANAR_INFERRED = 6`)
- **Distinct Provenance Code:** Planar-inferred observations use `Provenance.PLANAR_INFERRED = 6` (`DaxaProvenance::PlanarInferred = 6`, `SF_PROVENANCE_PLANAR_INFERRED = 6u`) and **never reuse `Provenance.REPROJECTED = 2`** (which is reserved strictly for temporal reprojection).
- **Measured vs. Completed Stream Separation:**
  - Standard measured planes (`output[0..8]` in `DaxaRefinement`, `DaxaRefinement::Frame::observations()`, and default `/stereoforge/depth` in `stereoforge_ros2_depth_publisher`) remain **100% bit-exact** with `--planar-recovery 0` (`Provenance` strictly in `{0, 1}`).
  - Completed observations (`Provenance` in `{0, 1, 6}`) are written exclusively by the `planar` pass (`pass == 5`) into dedicated left-eye buffers (`CompletedDepth`, `CompletedConfidence`, `CompletedValidity`, `CompletedProvenance`, `CompletedSourceIds`, `CompletedAgeNs`, `CompletedDisparity`, exposed via `DaxaRefinement::Frame::completed_observations()`) and published on `--completed-depth-topic` (default `/stereoforge/depth_completed`) when `--planar-recovery 1` is active.

---

## 2. `[P0/P1]` Guarded GPU `planar` Shader Architecture (`python/stereoforge/shaders/daxa/refinement.slang`)
1. **8-Tap Diagonal Quadrant Pre-Screen (`qpre`):** Before executing coarse neighborhood fitting or the 36-tap full-resolution stencil, invalid pixels sample 2 diagonal taps (`radius 3` and `7`) in each of the 4 quadrants. Pixels with `< 2` quadrants represented or spanning a strong luma edge exit immediately.
2. **Local $3\times 3$ Coarse Affine Plane Fit (`--planar-coarse-affine 1`) & Reciprocal Right-Eye Coarse Guard (`--planar-coarse-lr-guard 1`):**
   - Fits a local affine plane $d_c(u, v) = a_c \Delta u + b_c \Delta v + c_c$ across the $3\times 3$ coarse neighborhood (`>= 5` valid coarse cells, coarse fit `RMSE <= 0.55` coarse px, scaled to full-res `d_prior`), or falls back to 4-tap bilinear coarse prediction when `--planar-coarse-affine 0`.
   - Verifies reciprocal right-eye coarse SGM support (`p.coarse[0..2]` at layer `1`, offset `cn = p.coarse_width * p.coarse_height` at `cx_r = cx - d_coarse * step`) within `planar_coarse_lr_tolerance = 2.0` full-res px, plus full-resolution right-eye raw disparity (`fplane(21)`, `uplane(23)`) non-contradiction.
3. **36-Tap Quadrant Stencil (`{-9, -5, -2, 2, 5, 9}`) with Midpoint Ray-Crossing Guard (`--planar-ray-crossing-guard 1`):**
   - Requires valid measured anchors (`provenance == 1`, `confidence >= 0.10`) in `>= 3` quadrants and `cnt >= planar_min_anchors` (default `10`, ablated across `{5, 10, 16}`), with a top-half empty early-out (`iy_i == 2 && (qmask == 0 || n_anc * 3 < min_anc)`).
   - Rejects any anchor whose midpoint `(x + nx)/2, (y + ny)/2` crosses a luma edge (`> 18 DN`).
4. **Weighted Full-Resolution Affine Fit + Post-Fit 5-Tap `ortho_thermo32` & Cross-Eye Patch Luma Verification:**
   - Fits $d(x, y) = a \cdot dx + b \cdot dy + c$ (`|a| <= 0.22`, `|b| <= 0.55`, `RMSE <= min(planar_max_rmse, clamp(0.32 + 0.0045 * d_coarse, 0.36, 0.75))`).
   - Verifies 5-tap cross-eye patch mean luma agreement (`|mean_L - mean_R| <= 12.0 DN`) and 5-tap `ortho_thermo32` descriptor Hamming cost (`<= 28`).

---

## 3. Hardware GPU `240 FPS` (`<= 4.167 ms`) Qualification (`daxa_dual_preprocess_benchmark`)

| Dual-Preprocess + 4× Refinement Pipeline (`1280×720`, `120` Frames, `20` Warmup) | Single-Interval GPU Span `p50 / p95` (`FPS`) | Stage-Sum GPU `p50 / p95` | Host Wall `p50 / p95` | `dual_preprocess` GPU | `coarse` GPU | `refinement_metric` GPU | `refinement_planar` GPU | `<= 4.167 ms` (`p50 & p95`) | 30-Plane Bit-Exact | Vulkan Validation Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: | ---: |
| `dual4x_selective_repair_only` (`--planar-recovery 0`) | **`3.100 / 3.215 ms` (`322.5 FPS`)** | `3.054 / 3.158 ms` | `3.345 / 3.588 ms` | `0.344 ms` | `1.167 ms` | `0.447 ms` | `0.000 ms` | **`true`** | `true` | `0` |
| **`dual4x_selective_repair_plus_planar` (`--planar-recovery 1`, default `a=10, 3×3 affine`)** | **`3.880 / 4.158 ms` (`257.7 FPS`)** | **`3.822 / 4.104 ms`** | **`4.124 / 4.371 ms`** | **`0.334 ms`** | **`1.412 ms`** | **`0.443 ms`** | **`0.861 ms`** | **`true`** | **`true`** | **`0`** |

---

## 4. Canonical 5-Scene `1280×720` Synthetic Ground-Truth Suite (Measured vs. Planar Completed & `Provenance == 6` Inferred Cohort)

| Pipeline / Cohort (`1280×720` Native Vulkan GPU across 5 Canonical GT Scenes) | Mean Non-Occluded Coverage | Mean Disparity MAE (`px`) | Mean Bad `>1 px` (`%`) | Mean Bad `>2 px` (`%`) | Unconditional Good `≤1 px` Completeness | Unconditional Good `≤2 px` Completeness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `4x Refinement (Independent)` — Measured (`prov == 1`) | `93.83%` | `0.187 px` | `0.28%` | `0.23%` | `93.55%` | `93.61%` |
| `4x Selective Repair (Independent)` — Measured (`prov == 1`) | `97.81%` | `0.188 px` | `0.29%` | `0.23%` | `97.53%` | `97.59%` |
| **`4x Planar Recovery Completed (Independent)` (`prov in {1, 6}`)** | **`98.29%`** | **`0.187 px`** | **`0.29%`** | **`0.23%`** | **`98.00%`** | **`98.06%`** |
| `4x Refinement (Derived Right)` — Measured (`prov == 1`) | `93.77%` | `0.194 px` | `0.34%` | `0.28%` | `93.44%` | `93.50%` |
| `4x Selective Repair (Derived Right)` — Measured (`prov == 1`) | `97.75%` | `0.194 px` | `0.34%` | `0.28%` | `97.42%` | `97.48%` |
| **`4x Planar Recovery Completed (Derived Right)` (`prov in {1, 6}`)** | **`98.23%`** | **`0.194 px`** | **`0.34%`** | **`0.28%`** | **`97.89%`** | **`97.95%`** |
| — **`low_texture_surface` Inferred Cohort (`prov == 6`, Independent / Derived)** | `+16,409 / +16,249 px` | **`0.115 / 0.116 px`** | **`0.00% / 0.00%`** | **`0.00% / 0.00%`** | — | — |
| — **`occlusion_boundaries` Inferred Cohort (`prov == 6`, Step-Edge Non-Bridging)** | **`0 / 0 px`** | **`0.000 px`** | **`0.00%`** | **`0.00%`** | — | — |

---

## 5. Physical ZED 2 Desk-Scene GPU Ablation (`v9.3` Measured vs. `dea13a1` Unguarded vs. Guarded `v9.3` Across `5 / 10 / 16` Anchors & `Bilinear` vs. `3×3 Coarse Affine`)

| Desk-Scene GPU Configuration (`1280×720` ZED 2 Capture) | Measured Planes Bit-Exact | Full-Frame Measured (`prov=1`) | Full-Frame Inferred (`prov=6`) | Broad Desk ROI Measured / Inferred / Completed | Inferred Cohort Reproj MAE / `ZNCC5` / `3×3 Planar Inlier` | Upright AprilTag Target (`prov=1` vs. `prov=6` MAE / Orth RMSE) |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| `v93_measured_no_repair` (`repair=0, planar=0`) | — | `228,004` | `0` | `33.37% / +0.00% / 33.37%` | `null` | `67.27% cov` (`2.13 mm MAE / 3.54 mm RMSE`) |
| `v93_measured_selective_repair` (`repair=1, planar=0`) | `true` | `244,070` | `0` | `34.80% / +0.00% / 34.80%` | `null` | `70.17% cov` (`2.18 mm MAE / 3.62 mm RMSE`) |
| `dea13a1_unguarded_planar` (`a=5, bilinear, ray_guard=0, lr_guard=0`) | `true` | `244,070` | `20,293` | `34.80% / +1.99% / 36.79%` | `3.613 DN / 0.1660 / 79.76%` | `70.17% + 11.41% = 81.58%` (`prov=6`: `1.11 mm MAE / 1.38 mm RMSE`) |
| `v93_planar_a5_bilinear` (`a=5, bilinear, ray_guard=1, lr_guard=1`) | `true` | `244,070` | `13,508` | `34.80% / +1.29% / 36.09%` | `3.557 DN / 0.1743 / 82.06%` | `70.17% + 9.01% = 79.18%` (`prov=6`: `1.12 mm MAE / 1.41 mm RMSE`) |
| `v93_planar_a5_affine3x3` (`a=5, 3×3 affine, ray_guard=1, lr_guard=1`) | `true` | `244,070` | `13,165` | `34.80% / +1.31% / 36.11%` | `3.411 DN / 0.1731 / 82.09%` | `70.17% + 9.14% = 79.31%` (`prov=6`: `1.12 mm MAE / 1.42 mm RMSE`) |
| `v93_planar_a10_bilinear` (`a=10, bilinear, ray_guard=1, lr_guard=1`) | `true` | `244,070` | `8,797` | `34.80% / +0.79% / 35.59%` | `3.052 DN / 0.1730 / 83.27%` | `70.17% + 6.93% = 77.09%` (`prov=6`: `0.97 mm MAE / 1.21 mm RMSE`) |
| **`v93_planar_a10_affine3x3_default` (`a=10, 3×3 affine, ray_guard=1, lr_guard=1`)** | **`true`** | **`244,070`** | **`8,844`** | **`34.80% / +0.82% / 35.62%`** | **`2.863 DN / 0.1681 / 83.08%`** | **`70.17% + 7.02% = 77.18%` (`prov=6`: `0.98 mm MAE / 1.23 mm RMSE`)** |
| `v93_planar_a16_affine3x3` (`a=16, 3×3 affine, ray_guard=1, lr_guard=1`) | `true` | `244,070` | `3,614` | `34.80% / +0.28% / 35.08%` | `2.111 DN / 0.1811 / 83.54%` | `70.17% + 2.85% = 73.02%` (`prov=6`: `0.81 mm MAE / 1.04 mm RMSE`) |

---

## 6. `[P2]` Causal Color Tie-Break Ablation (`D-Y` vs. `D-UV` vs. `D-YUV` vs. `D-UV-Shuffled`)

To test whether selective color tie-breaking (`Variant D`) derives its coverage gain from independent chromatic evidence (`U/V`) or from sub-quanta luminance differences (`Y`) resolving `ortho_thermo32` ties, we evaluated four controlled variants across all `1,264` coarse-supported ambiguous pixels (`reason & (REF_AMBIGUOUS | REF_PLATEAU | REF_LOW_MARGIN) != 0`) on the ZED 2 desk scene:

| Tie-Break Variant (`1,264` Coarse-Supported Ambiguous Candidates) | Admitted Pixels (`%` Gain) | `Q1` Low-Luma Low-Chroma | `Q2` Low-Luma High-Chroma | `Q3` High-Luma Low-Chroma | `Q4` High-Luma High-Chroma | Low-Chroma Share (`Q1 + Q3`) | High-Chroma Share (`Q2 + Q4`) | Admitted Epipolar Luma MAE (`DN`) | Admitted Epipolar Chroma MAE (`DN`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `D_Y_luma_only` (`|ΔY|/4` tie-breaker only) | `1,160` (`+0.126%`) | `570` | `170` | `246` | `174` | **`70.34%`** | `29.66%` | **`10.516 DN`** | `1.317 DN` |
| `D_UV_chroma_only` (`(|ΔCr|+|ΔCb|)/4` tie-breaker only) | `1,174` (`+0.127%`) | `577` | `169` | `251` | `177` | **`70.53%`** | `29.47%` | `11.054 DN` | `1.426 DN` |
| `D_YUV_combined` (`|ΔY|/4 + (|ΔCr|+|ΔCb|)/4`) | `1,137` (`+0.123%`) | `559` | `166` | `242` | `170` | **`70.45%`** | `29.55%` | `10.609 DN` | **`1.313 DN`** |
| **`D_UV_shuffled_control` (spatially permuted right-eye `Cr, Cb`)** | **`1,088` (`+0.118%`)** | **`526`** | **`150`** | **`238`** | **`174`** | **`70.22%`** | **`29.78%`** | **`11.690 DN`** | **`1.476 DN`** |

- **Causal Conclusion:** Because `ortho_thermo32` integer costs (`6 * popcount + |offset|`) frequently produce exact integer ties (`second_cost - best_cost == 0`), **any continuous perturbation breaks the integer tie**: even spatially shuffled right-eye chroma (`D_UV_shuffled_control`) admits `1,088` of the `1,174` pixels (`92.7%` of `D_UV_chroma_only`'s count) with worse epipolar luma error (`11.690 DN` vs. `10.516 DN` for `D_Y_luma_only`), and over `70%` of admitted pixels lie in low-chroma regions (`Q1 + Q3`). Consequently, chroma tie-breaking is **not** folded into production `--selective-repair` or `--planar-recovery`.
