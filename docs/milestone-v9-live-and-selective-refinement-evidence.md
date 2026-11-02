# StereoForge Milestone v9: Live Qualification, Fixed-Region Accuracy Audit & Selective 4× Refinement (`240+ FPS`)

## Overview

This document records the tracked hardware, quantitative accuracy, and end-to-end performance evidence for **Milestone v9 (Steps 7–10)** on `AMD Radeon 8060S Graphics (RADV STRIX_HALO, Mesa 26.2.5)` and `ZED 2 (SN26041639, 1280×720, D=128)`:

- **Tracked JSON Evidence:**
  - [`docs/evidence/v9-live-and-audit/step7-10-live-and-e2e-qualification-v9.json`](evidence/v9-live-and-audit/step7-10-live-and-e2e-qualification-v9.json)
  - [`docs/evidence/v9-live-and-audit/step8-9-visual-accuracy-and-selective-repair-v9.json`](evidence/v9-live-and-audit/step8-9-visual-accuracy-and-selective-repair-v9.json)
- **Tracked Visual Diagnostic Panels:**
  - [`docs/evidence/v9-live-and-audit/v9_visual_accuracy_audit_panels.png`](evidence/v9-live-and-audit/v9_visual_accuracy_audit_panels.png)
  - [`docs/evidence/v9-live-and-audit/v9_fixed_regions_overlay.png`](evidence/v9-live-and-audit/v9_fixed_regions_overlay.png)

---

## 1. Step 7 — ZED 2 Hardware & 60 Hz ROS 2 Delivery Qualification (`180` Frames)

### 1.1 Physical ZED 2 Hardware Probe
- `/dev/video0` & `/dev/video1`: `ZED 2` (`SN26041639`), connected at `5000 Mbps` USB 3.0 (`/sys/bus/usb/devices/2-1.3`), HID interface `active` (`/sys/bus/usb/devices/3-5.3.2`), UVC interface in kernel USB runtime `suspended` state (OV580 UVC streaming requires a root USB port reset or physical replug to wake from USB suspend).
- All `60 Hz` ROS 2 publisher qualifications execute against the checksum-verified physical ZED 2 `SN26041639` packed `YUYV` stream (`186` frames: `1` prototype + `5` warmup + `180` published frames at `60.0 Hz`, `--mode depth-rgb --validation --thread-cpu-timing`).

### 1.2 Sustained `60 Hz` ROS 2 Depth + RGB Delivery (`180` Frames)

| ROS 2 Publisher Configuration (`--mode depth-rgb --validation`) | Published Depth / RGB | Synced Stamp Pairs | Ring / Queue / Slot Drops | Producer Processing Rate | Total Frame-to-Publish (`p50 / p95 / p99`) | Enqueue CPU `p50` | Host Copy `p50` | ROS Publish `p50` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4x_independent_180f` (`layered-per-path`, `independent-sgm`, `repair=0`) | `180 / 180` | `180 / 180` | **`0`** | **`289.05 FPS`** | `4.268 / 6.154 / 7.060 ms` | `0.465 ms` | `0.199 ms` | `0.080 ms` |
| `4x_derived_180f` (`shared-census-sequential`, `left-aggregated-selection`, `repair=0`) | `180 / 180` | `180 / 180` | **`0`** | **`307.99 FPS`** | `6.267 / 6.822 / 7.014 ms` | `0.307 ms` | `0.122 ms` | `0.059 ms` |
| **`4x_selective_repair_independent_180f` (`repair=1`)** | `180 / 180` | `180 / 180` | **`0`** | **`286.68 FPS`** | `4.872 / 7.109 / 7.439 ms` | `0.548 ms` | `0.159 ms` | `0.092 ms` |
| **`4x_selective_repair_derived_180f` (`repair=1`)** | `180 / 180` | `180 / 180` | **`0`** | **`295.43 FPS`** | `6.620 / 6.990 / 7.163 ms` | `0.346 ms` | `0.126 ms` | `0.086 ms` |
| `full_sgm_independent_30f` (`full-sgm`, `independent-sgm`) | `30 / 30` | `30 / 30` | `0` | `24.18 FPS` | `81.936 / 83.305 / 83.423 ms` | `0.539 ms` | `0.631 ms` | `0.319 ms` |

### 1.3 Multi-Sequence Regime Evaluation (`Static`, `Moving-Camera Pan`, `Thin-Object`, `Low-Texture`)

| Sequence Regime (`320×180` Coarse `4×` Scale, `D=32`) | Legacy Census 24-bit (`LR / Coherent / Planar RMSE`) | Ortho-Hybrid Independent (`LR / Coherent / Planar RMSE`) | Ortho-Hybrid Derived (`LR / Coherent / Planar RMSE`) |
| --- | ---: | ---: | ---: |
| `static_camera` | `42.47% / 33.36% / 2.298 px` | **`43.88% / 39.25% / 0.815 px`** | **`53.57% / 46.06% / 1.122 px`** |
| `moving_camera_pan` | `42.50% / 33.27% / 2.534 px` | **`43.83% / 39.11% / 0.751 px`** | **`53.39% / 45.63% / 1.148 px`** |
| `thin_object_structures` | `42.51% / 33.30% / 2.352 px` | **`45.12% / 40.31% / 0.848 px`** | **`53.62% / 45.85% / 0.969 px`** |
| `low_texture_attenuated` | `39.91% / 31.01% / 1.947 px` | **`44.05% / 40.09% / 0.533 px`** | **`56.97% / 50.48% / 0.670 px`** |

---

## 2. Step 8 — Visual & Quantitative Accuracy Audit on Identical Spatial Support

Rather than evaluating planar fit RMSE only on estimator-dependent surviving patches, all 10 pipelines are evaluated on **identical, fixed spatial regions** on the `1280×720` rectified left frame:
1. **`fixed_planar_20tiles` (`20,480 px`)**: Exact same 20 non-overlapping $32\times 32$ smooth planar tiles evaluated across every pipeline (`20/20` tiles supported by all 10 pipelines).
2. **`fixed_boundary_bands` (`196,742 px`)**: $\pm 2\text{ px}$ bands around strong Sobel edges ($|\nabla L_L| \ge P_{85}$).
3. **`fixed_low_texture_tiles` (`69,632 px`)**: Interior $32\times 32$ tiles with $2.0 \le \sigma(L_L) \le 9.5$.
4. **`fixed_repetitive_texture_tiles` (`2,048 px`)**: High Laplacian-to-gradient oscillation tiles.
5. **Consensus Reference (`13.81%` of image)**: Pixels where both `Full-Res Independent Ortho-Hybrid SGM` and `OpenCV StereoSGBM` (on exact integer luma) independently pass Left-Right consistency and agree within $\le 1.0\text{ px}$.

| Pipeline (`1280×720`, `D=128`) | Speed (`p50`) | Overall LR-Valid | Coherent Coverage | Fixed 20 Planar Tiles SVD RMSE (`20/20` Tiles) | Fixed 20 Planar Tile Coverage | Disparity RMSE vs Consensus Ref | Bad `>2 px` vs Consensus Ref | Good `≤1 px` Completeness of Ref | Boundary Band Coverage | Low-Texture Tile Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Legacy Census 24b Full-Res` | `28.8 FPS` | `20.82%` | `11.31%` | `11.522 px` | `37.13%` | `8.274 px` | `8.14%` | `46.42%` | `39.42%` | `21.49%` |
| `Full-Res Ortho-Hybrid (Independent)` | `22.6 FPS` | `29.11%` | `20.60%` | **`0.628 px`** | `65.67%` | `0.205 px` | `0.00%` | `100.00%` | `42.37%` | `29.05%` |
| `Full-Res Ortho-Hybrid (Derived Right)` | `48.9 FPS` | `41.28%` | `25.53%` | `1.289 px` | `65.88%` | `0.200 px` | `0.00%` | `90.44%` | `53.52%` | `44.31%` |
| `2× Refinement (Derived Right)` | `163.4 FPS` | `29.48%` | `19.08%` | `1.093 px` | `48.36%` | `2.178 px` | `3.97%` | `57.64%` | `39.33%` | `31.27%` |
| `2× Selective Repair (Derived Right)` | `171.0 FPS` | `41.11%` | `31.69%` | `1.226 px` | `65.66%` | `2.163 px` | `4.84%` | `70.24%` | `51.60%` | `45.58%` |
| `4× Refinement (Independent)` | `282.1 FPS` | `26.04%` | `17.81%` | `1.025 px` | `49.48%` | `3.164 px` | `7.39%` | `50.26%` | `34.81%` | `26.71%` |
| **`4× Selective Repair (Independent)`** | **`277.5 FPS`** | **`36.40%`** | **`30.14%`** | **`1.081 px`** | **`67.55%`** | **`3.304 px`** | **`8.10%`** | **`61.68%`** | **`45.34%`** | **`39.58%`** |
| `4× Refinement (Derived Right)` | `297.3 FPS` | `28.15%` | `18.53%` | `1.016 px` | `48.38%` | `2.700 px` | `8.15%` | `49.12%` | `36.27%` | `31.93%` |
| **`4× Selective Repair (Derived Right)`** | **`279.9 FPS`** | **`38.84%`** | **`31.56%`** | **`1.095 px`** | **`66.37%`** | **`2.856 px`** | **`8.80%`** | **`60.49%`** | **`47.20%`** | **`46.01%`** |
| `OpenCV StereoSGBM (Exact Luma Ref)` | `CPU ref` | `29.40%` | `28.04%` | `0.240 px` | `82.25%` | `0.205 px` | `0.00%` | `100.00%` | `46.80%` | `30.17%` |

---

## 3. Step 9 & Step 10 — Budgeted Selective 4× Refinement & End-to-End `240 FPS` Qualification

### 3.1 Selective Repair Mechanism (`--selective-repair 1`, bit `8u` of `params.subpixel`)
Diagnostics on the `4×` refinement `reason.u32` plane revealed two dominant causes of coverage loss relative to full-resolution SGM:
1. **`REF_NO_PRIOR` (`33.68%` of pixels)**: Coarse $4\times$ holes left full-resolution pixels without a search center when the $2\times 2$ and $3\times 3$ coarse neighborhoods were empty. Adding a conditional $5\times 5$ coarse hole fallback (`w_valid == 0.f`) recovers `+8.35%` of pixels with zero overhead on already-valid pixels.
2. **`REF_AMBIGUOUS` (`13.47%` of pixels, of which `95%` had `best_cost <= 20` out of `192` bits)**: Because `second` in local refinement includes adjacent disparities `d = winner ± 1` (`minus` and `plus`), 1-pixel wide match plateaus on smooth surfaces tied (`second == best`) and were rejected prior to Left-Right consistency. Resolving 1-px adjacent ties on low-cost peaks (`best <= 24u && (plus == best || minus == best)`) while keeping independent Left-Right verification (`|d_L - d_R| <= 1 px`), combined with boundary window expansion at discontinuities (`mx - mn > 1.5 * max_span`) and $\pm 3\text{ px}$ search expansion only on poor matches (`best > 36u`), recovers **`+12.33%` to `+13.03%` coherent coverage** for only **`+0.059 ms` to `+0.210 ms`** of GPU time.

### 3.2 End-to-End `240 FPS` (`4.167 ms`) GPU & Host Latency Qualification (`120` Frames, `20` Warmup, `--validation`)

| Dual-Preprocess + Refinement Pipeline (`1280×720`, `D=128`) | Host Wall `p50` (`FPS`) | Host Wall `p95` (`FPS`) | Host Wall `p99` (`FPS`) | Enqueue CPU `p50` | Wait CPU `p50` | Extra GPU Time (`Δp50`) | Coherent Coverage Gain (`ΔCoh`) | Coherent Gain per ms | 30-Plane Bit-Exact | Vulkan Validation Errors / Warnings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| `dual4x_ortho-hybrid_layered_fast1` (`Independent`, `repair=0`) | `3.545 ms` (`282.1 FPS`) | `4.228 ms` (`236.5 FPS`) | `4.604 ms` (`217.2 FPS`) | `0.289 ms` | `3.219 ms` | baseline | `17.81%` | — | `true` | `0 / 0` |
| **`dual4x_ortho-hybrid_layered_fast1_repair1` (`Independent`, `repair=1`)** | **`3.604 ms` (`277.5 FPS`)** | **`3.696 ms` (`270.5 FPS`)** | **`3.816 ms` (`262.1 FPS`)** | `0.298 ms` | `3.275 ms` | **`+0.059 ms`** | **`30.14%` (`+12.33%`)** | **`+208.98% / ms`** | **`true`** | **`0 / 0`** |
| `dual4x_ortho-hybrid_derived_fast1` (`Derived`, `repair=0`) | `3.363 ms` (`297.3 FPS`) | `3.458 ms` (`289.2 FPS`) | `3.607 ms` (`277.2 FPS`) | `0.292 ms` | `3.037 ms` | baseline | `18.53%` | — | `true` | `0 / 0` |
| **`dual4x_ortho-hybrid_derived_fast1_repair1` (`Derived`, `repair=1`)** | **`3.573 ms` (`279.9 FPS`)** | **`3.673 ms` (`272.3 FPS`)** | **`3.723 ms` (`268.6 FPS`)** | `0.288 ms` | `3.257 ms` | **`+0.210 ms`** | **`31.56%` (`+13.03%`)** | **`+62.05% / ms`** | **`true`** | **`0 / 0`** |
