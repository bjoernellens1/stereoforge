# StereoForge Milestone v9.3: Provenance-Tagged Edge-Aware Planar Recovery & Color Tie-Break Causal Qualification (`240+ FPS`)

- **Tracked Qualification Artifact:** [`benchmarks/results/v9_3_planar_qualification.json`](../benchmarks/results/v9_3_planar_qualification.json)
- **Observation Contract Reference:** [`docs/daxa-observation-contract.md`](daxa-observation-contract.md)
- **v9.2 Live 60 Hz & Selective Repair Baseline:** [`docs/milestone-v9-live-and-selective-refinement-evidence.md`](milestone-v9-live-and-selective-refinement-evidence.md)

---

## 1. `[P0-A / P1-A]` Two-Eye Observation Contract, Provenance Separation (`Provenance.PLANAR_INFERRED = 6`) & ROS 2 Delivery Tracing
- **Distinct Provenance Code:** Planar-inferred observations use `Provenance.PLANAR_INFERRED = 6` (`DaxaProvenance::PlanarInferred = 6`, `SF_PROVENANCE_PLANAR_INFERRED = 6u`) and **never reuse `Provenance.REPROJECTED = 2`** (which is reserved strictly for temporal reprojection).
- **Two-Eye `completed_observations()` Contract (`2 * width * height`):**
  - Standard measured planes (`output[0..8]` in `DaxaRefinement`, `DaxaRefinement::Frame::observations()`, and default `/stereoforge/depth` in `stereoforge_ros2_depth_publisher`) remain **100% bit-exact** with `--planar-recovery 0` (`Provenance` strictly in `{0, 1}`).
  - Completed observations (`CompletedDepth`, `CompletedConfidence`, `CompletedValidity`, `CompletedProvenance`, `CompletedSourceIds`, `CompletedAgeNs`, `CompletedDisparity`, exposed via `DaxaRefinement::Frame::completed_observations()`) are allocated for both eyes (`count = 2 * width * height`):
    - **Left eye (`eye = 0`, `[0 .. width*height)`):** initialized from measured stereo in `metric` (`pass == 4`) and augmented by `planar` (`pass == 5`) with `DaxaProvenance::PlanarInferred = 6` (`Validity = 1`, `SourceIds = -1`, `AgeNs = 0`).
    - **Right eye (`eye = 1`, `[width*height .. 2*width*height)`):** initialized from right-eye measured stereo in `metric` (`pass == 4`, `Provenance` strictly in `{0, 1}`).
- **ROS 2 Completed Depth & Per-Pixel Provenance Topics:**
  - When `--publish-completed-depth` is passed to `stereoforge_ros2_depth_publisher`, `DaxaHostExport` stages both `CompletedDepth` (`32FC1` on `/stereoforge/depth_completed`) and `CompletedProvenance` (`mono8` with per-pixel codes `{0, 1, 6}` on `/stereoforge/depth_completed_provenance`), and records both streams in the ROS 2 delivery trace (`CompletedDepth` and `CompletedProvenance` in `ros_delivery_trace.hpp` / `ros2_delivery_probe.cpp`).

---

## 2. `[P1-B / P1-C]` Guarded & Shared-Memory-Optimized GPU `planar` Shader (`python/stereoforge/shaders/daxa/refinement.slang`)
1. **Workgroup Early-Out & Shared-Memory Tile Caching (`[numthreads(16, 16, 1)]`):**
   - Each $16\times 16$ workgroup evaluates whether any thread is an invalid pixel (`provenance == 0`) with valid coarse SGM support (`coarse_val != 0`, `d_coarse_0 in [2.5, 31.25]`) via `InterlockedOr(gs_has_candidate, 1u)`. Fully measured or out-of-range tiles exit immediately after a single barrier.
   - Active workgroups cooperatively preload the $6\times 6$ coarse disparity tile (`gs_coarse_d[6][6]`), precompute the $4\times 4$ local $3\times 3$ coarse affine planes (`gs_coarse_plane[4][4]`) in 16 threads, and load the $34\times 34$ full-resolution disparity/confidence/provenance + luma halo (`gs_disp[34][34]`, `gs_luma[34][34]`) into `groupshared` LDS, eliminating redundant global memory gathers.
2. **8-Tap Diagonal Quadrant Pre-Screen (`qpre`) & Reciprocal Right-Eye Coarse Guard (`--planar-coarse-lr-guard 1`):**
   - Before scanning the 36-tap full-resolution stencil, invalid pixels sample 2 diagonal taps (`radius 3` and `7`) in each of the 4 quadrants from `groupshared` memory; pixels with `< 2` quadrants represented exit early.
   - Verifies reciprocal right-eye coarse SGM support (`p.coarse[0..2]` at layer `1`, offset `cn = p.coarse_width * p.coarse_height`) within `planar_coarse_lr_tolerance = 2.0` full-res px, plus full-resolution right-eye raw disparity (`fplane(21)`, `uplane(23)`) non-contradiction.
3. **Coarse-Slope-Centered 36-Tap Stencil (`{-9, -5, -2, 2, 5, 9}`) with Midpoint Ray-Crossing Guard (`--planar-ray-crossing-guard 1`):**
   - Evaluates neighbor anchor agreement against the **coarse-slope-predicted disparity at the anchor's offset** $d_{\text{pred}}(\Delta x, \Delta y) = d_c(x, y) + a_c \Delta x + b_c \Delta y$ (`u_res = da - dc_a`, `abs(u_res) <= tol`), preventing false rejection of valid slanted-tabletop anchors where vertical foreshortening ($\partial d / \partial y \approx 0.14\text{--}0.25\text{ px/px}$) shifts disparity across $\Delta y = \pm 9$.
   - Weights each admitted anchor by distance, photometric similarity, and a Cauchy/Huber residual factor $w_{\text{rob}} = 1 / (1 + (u_{\text{res}} / (0.45 \cdot \text{tol}))^2)$, fitting the residual plane $\delta d(\Delta x, \Delta y) = \delta a \Delta x + \delta b \Delta y + \delta c$ around $(a_c, b_c, d_c)$ and verifying 5-tap cross-eye patch mean luma agreement (`<= 12.0 DN`) and `ortho_thermo32` Hamming cost (`<= 28`).

---

## 3. `[P1-C]` Hardware GPU `240 FPS` (`<= 4.167 ms`) Qualification Across `5` Interleaved Trials (`daxa_dual_preprocess_benchmark`)

Evaluated across `N = 5` interleaved trials (`120` measured frames + `20` warmup frames per trial, alternating `--planar-recovery 0` and `--planar-recovery 1`, selecting the median trial by `gpu_combined_queue_span_ms.p95` with zero cherry-picking):

| Dual-Preprocess + 4× Refinement Pipeline (`1280×720`, `5 × 120` Interleaved Frames) | Median Trial GPU Span `p50 / p95 / p99` (`FPS`) | 5-Trial `p95` Mean ± Std (`All 5 <= 4.167 ms`) | Stage-Sum GPU `p50 / p95` | Host Wall `p50 / p95` | `dual_preprocess` GPU | `coarse` GPU | `refinement_metric` GPU | `refinement_planar` GPU `p50` | 30-Plane Bit-Exact | Vulkan Validation Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| `dual4x_selective_repair_only` (`--planar-recovery 0`) | **`3.092 / 3.209 / 3.269 ms` (`323.4 FPS`)** | `3.205 ± 0.033 ms` (**`5/5 true`**) | `3.046 / 3.163 ms` | `3.364 / 3.536 ms` | `0.332 ms` | `1.151 ms` | `0.444 ms` | `0.000 ms` | `true` | `0` |
| **`dual4x_selective_repair_plus_planar` (`--planar-recovery 1`, default `a=10, 3×3 affine`)** | **`3.805 / 3.937 / 4.006 ms` (`262.8 FPS`)** | **`3.935 ± 0.031 ms` (`5/5 true`)** | **`3.758 / 3.890 ms`** | **`4.081 / 4.302 ms`** | **`0.313 ms`** | **`1.171 ms`** | **`0.767 ms`** | **`0.386 ms` (`2.23×` faster vs. `0.861 ms`)** | **`true`** | **`0`** |

- **5-Trial Interleaved `gpu_combined_queue_span_ms.p95` Distribution (`dual4x_selective_repair_plus_planar`):** `[3.9484, 3.9368, 3.9194, 3.8879, 3.9803] ms` (`mean = 3.9346 ms`, `std = 0.0306 ms`, **`230 µs` safety margin below `4.167 ms`**).
- **5-Trial Interleaved `gpu_planar_recovery_ms.p50` Distribution:** `[0.3851, 0.3858, 0.3849, 0.3863, 0.3850] ms` (down from `0.861 ms` prior to shared-memory tile caching and workgroup early-out).

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
| — **`low_texture_surface` Inferred Cohort (`prov == 6`, Independent / Derived)** | `+16,402 / +16,248 px` | **`0.162 / 0.162 px`** | **`0.00% / 0.00%`** | **`0.00% / 0.00%`** | — | — |
| — **`occlusion_boundaries` Inferred Cohort (`prov == 6`, Step-Edge Non-Bridging)** | **`0 / 0 px`** | **`0.000 px`** | **`0.00%`** | **`0.00%`** | — | — |

---

## 5. `[P0-B / P1-B]` Physical ZED 2 Desk-Scene GPU Ablation: Estimator-Independent Multi-Scale ZNCC Reference Planes vs. Self-Consistency to Measured Planes

In `benchmarks/planar_recovery_v93.py`, planar surface evaluation is strictly separated into:
1. **`independent_zncc_reference_planes` (`reference_kind = "independent_multiscale_zncc_epipolar_ransac_plane"`):** Planes fitted via projective-disparity RANSAC exclusively to estimator-independent multi-scale ($5\times 5$, $9\times 9$, $13\times 13$) subpixel ZNCC epipolar correspondences (`ZNCC >= 0.90`, `max_disp = 128`, `1,142` independent correspondences on `upright_apriltag_target`, `97` on `charuco_board_inrange`, and `284` on `horizontal_desk_plane_clean`).
2. **`consistency_to_measured_planes` (`reference_kind = "self_consistency_to_measured_stereo_ransac_plane"`):** Self-consistency residuals against planes fitted to StereoForge's own `prov == 1` measured stereo points.

| Desk-Scene GPU Configuration (`1280×720` ZED 2 Capture) | Measured Bit-Exact | Full-Frame Measured (`prov=1`) | Full-Frame Inferred (`prov=6`) | Broad Desk ROI Measured / Inferred / Completed | Inferred Cohort Epipolar MAE / `3×3 Planar Inlier` | Independent ZNCC Reference: `upright_apriltag_target` (`prov=1` vs. `prov=6` MAE / Orth RMSE) | Independent ZNCC Reference: `horizontal_desk_plane_clean` (`prov=6` px / Disp MAE / Orth RMSE / Normal Err) | Independent ZNCC Reference: `charuco_board_inrange` (`prov=6` px / Disp MAE / Orth RMSE) |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v93_measured_no_repair` (`repair=0, planar=0`) | — | `228,004` | `0` | `33.37% / +0.00% / 33.37%` | `null` | `67.27% cov` (`2.12 mm / 3.53 mm`) | `0 px` | `0 px` |
| `v93_measured_selective_repair` (`repair=1, planar=0`) | `true` | `244,070` | `0` | `34.80% / +0.00% / 34.80%` | `null` | `70.17% cov` (`2.18 mm / 3.60 mm`) | `0 px` | `0 px` |
| `dea13a1_unguarded_planar` (`a=5, bilinear, ray_guard=0, lr_guard=0`) | `true` | `244,070` | `30,542` | `34.80% / +2.85% / 37.65%` | `3.644 DN / 84.62%` | `70.17% + 11.91% = 82.07%` (`prov=6`: `1.37 mm / 1.55 mm`) | `214 px (+3.84%)` / `0.929 px` / `4.54 mm` / `2.21°` | `80 px (+3.73%)` / `0.552 px` / `2.90 mm` |
| `v93_planar_a5_bilinear` (`a=5, bilinear, ray_guard=1, lr_guard=1`) | `true` | `244,070` | `20,811` | `34.80% / +1.72% / 36.52%` | `3.565 DN / 85.36%` | `70.17% + 8.78% = 78.95%` (`prov=6`: `1.38 mm / 1.58 mm`) | `126 px (+2.26%)` / `1.007 px` / `4.86 mm` / `2.47°` | `44 px (+2.05%)` / `0.608 px` / `2.97 mm` |
| `v93_planar_a5_affine3x3` (`a=5, 3×3 affine, ray_guard=1, lr_guard=1`) | `true` | `244,070` | `21,539` | `34.80% / +1.84% / 36.64%` | `3.426 DN / 85.27%` | `70.17% + 8.87% = 79.04%` (`prov=6`: `1.39 mm / 1.59 mm`) | `138 px (+2.48%)` / `1.057 px` / `5.24 mm` / `7.58°` | `42 px (+1.96%)` / `0.559 px` / `2.80 mm` |
| `v93_planar_a10_bilinear` (`a=10, bilinear, ray_guard=1, lr_guard=1`) | `true` | `244,070` | `14,561` | `34.80% / +1.03% / 35.83%` | `3.114 DN / 87.23%` | `70.17% + 6.52% = 76.69%` (`prov=6`: `1.23 mm / 1.36 mm`) | `56 px (+1.01%)` / `1.094 px` / `5.07 mm` / `3.81°` | `8 px (+0.37%)` / `0.525 px` / `2.89 mm` |
| **`v93_planar_a10_affine3x3_default` (`a=10, 3×3 affine, ray_guard=1, lr_guard=1`)** | **`true`** | **`244,070`** | **`15,574`** | **`34.80% / +1.11% / 35.91%`** | **`2.965 DN / 86.68%`** | **`70.17% + 6.61% = 76.78%` (`prov=6`: `1.25 mm / 1.39 mm`)** | **`55 px (+0.99%)` / `0.918 px` / `4.30 mm` / `3.98°`** | **`9 px (+0.42%)` / `0.540 px` / `2.87 mm`** |
| `v93_planar_a16_affine3x3` (`a=16, 3×3 affine, ray_guard=1, lr_guard=1`) | `true` | `244,070` | `6,736` | `34.80% / +0.34% / 35.13%` | `2.179 DN / 87.35%` | `70.17% + 2.90% = 73.06%` (`prov=6`: `1.19 mm / 1.30 mm`) | `12 px (+0.22%)` / `1.274 px` / `5.54 mm` / `32.49°` | `0 px` |

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
