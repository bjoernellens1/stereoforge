# StereoForge Milestone v9.1: Live 60 Hz Physical Camera Qualification, Non-Circular Accuracy Audit & Selective 4× Refinement (`240+ FPS`)

## Overview

This document records the tracked hardware, non-circular accuracy, newly-admitted pixel cohort, and end-to-end performance evidence for **Milestone v9.1 (Steps 7–10)** on `AMD Radeon 8060S Graphics (RADV STRIX_HALO, Mesa 26.2.5)` and physical `ZED 2 (SN26041639, 1280×720, D=128)`:

- **Tracked JSON Evidence:**
  - [`docs/evidence/v9-live-and-audit/step7-10-live-and-e2e-qualification-v9.json`](evidence/v9-live-and-audit/step7-10-live-and-e2e-qualification-v9.json)
  - [`docs/evidence/v9-live-and-audit/step8-9-visual-accuracy-and-selective-repair-v9.json`](evidence/v9-live-and-audit/step8-9-visual-accuracy-and-selective-repair-v9.json)
- **Tracked Visual Diagnostic Panels:**
  - [`docs/evidence/v9-live-and-audit/v9_visual_accuracy_audit_panels.png`](evidence/v9-live-and-audit/v9_visual_accuracy_audit_panels.png)
  - [`docs/evidence/v9-live-and-audit/v9_fixed_regions_overlay.png`](evidence/v9-live-and-audit/v9_fixed_regions_overlay.png)

---

## 1. Step 7 (Gate 1) — Real Physical `60 Hz` ZED 2 Camera (`--source camera`) + Concurrent ROS 2 Subscriber Delivery (`180` Frames)

### 1.1 Physical ZED 2 Hardware & Automatic OV580 Wake/Reset Recovery
- `/dev/video0` & `/dev/video1`: `ZED 2` (`SN26041639`), connected at `5000 Mbps` USB 3.0 (`/sys/bus/usb/devices/2-1.3`, `idle_runtime_power_status: "active"`).
- Both `src/capture.cpp` (`NativeCapture`) and `src/zed_capture.cpp` (`stereoforge._zed_capture`) automatically invoke `sl_oc::sensors::SensorCapture::resetSensorModule()` (`HIDIOCSFEATURE(2)` `[0x21, 0xE1]` on `/dev/hidraw2`) if V4L2 video initialization fails or OV580 serial readback returns `<= 0` after USB U3 suspend.
- All 4 primary `60 Hz` runs execute live physical acquisition (`--source camera`) paired with a concurrent [`stereoforge_ros2_delivery_probe`](../apps/ros2_delivery_probe.cpp) subscriber process verifying payload sizes, non-zero content, timestamps, and end-to-end DDS arrival latency. Unpaced offline replay runs (`--source replay`) are reported separately under `unpaced_offline_replay_ros2_e2e_runs`.

### 1.2 Live Physical `60 Hz` Camera (`--source camera`) + Concurrent Subscriber E2E Results (`180` Frames)

| Live Physical Camera Run (`--source camera --mode depth-rgb --validation`) | Capture / Publish / Subscriber Cadence | Published Depth / RGB | Subscriber Received (`KeepLast(1)`) | Measured-Window Queue / Ring / Slot Drops | Camera Inter-Frame Interval `p50` (`std`) | Host Ingress → Enqueue `p50` | Frame Processing → Publish `p50 / p95` | ROS 2 DDS Publish → Subscriber `p50` | Host Ingress → Subscriber E2E `p50` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4x_independent_180f` (`layered-per-path`, `independent-sgm`, `repair=0`) | `59.01 / 58.34 / 57.91 FPS` | `180 / 180` | `175 / 175` (`97.2%`) | **`0 / 0 / 0`** | `16.666 ms` (`0.098 ms`) | `0.009 ms` | `5.741 / 6.255 ms` | `0.310 ms` | **`6.125 ms`** |
| `4x_derived_180f` (`shared-census-sequential`, `left-aggregated-selection`, `repair=0`) | `60.00 / 59.32 / 60.26 FPS` | `180 / 180` | `179 / 179` (`99.4%`) | **`0 / 0 / 0`** | `16.666 ms` (`0.099 ms`) | `0.010 ms` | `5.520 / 5.938 ms` | `0.340 ms` | **`5.876 ms`** |
| **`4x_selective_repair_independent_180f` (`repair=1`)** | **`60.00 / 59.31 / 58.38 FPS`** | **`180 / 180`** | **`174 / 174` (`96.7%`)** | **`0 / 0 / 0`** | **`16.666 ms` (`0.099 ms`)** | **`0.012 ms`** | **`5.859 / 6.945 ms`** | **`0.336 ms`** | **`6.368 ms`** |
| **`4x_selective_repair_derived_180f` (`repair=1`)** | **`60.00 / 59.31 / 59.26 FPS`** | **`180 / 180`** | **`170 / 170` (`94.4%`)** | **`0 / 0 / 0`** | **`16.666 ms` (`0.100 ms`)** | **`0.009 ms`** | **`5.643 / 6.322 ms`** | **`0.177 ms`** | **`5.856 ms`** |

---

## 2. Step 8 (Gate 2) — Non-Circular Quantitative Accuracy Audit

All circular consensus references (`v_full_ind & v_ocv`) have been removed and replaced by two strictly independent references:

### 2.1 Reference A: 5-Scene `1280×720` Known-Ground-Truth Suite on Native Vulkan GPU (`daxa_refinement_benchmark --synthetic-yuyv`)

Evaluated across `slanted_planes`, `occlusion_boundaries`, `thin_structures`, `repetitive_and_subpixel`, and `low_texture_surface` (`1280×720`, `D=128`):

| Pipeline (`1280×720` Native Vulkan GPU Execution) | Mean Valid Coverage | Mean Disparity MAE (`px`) | Mean Disparity RMSE (`px`) | Mean Depth MAE (`mm`) | Mean Bad `>1 px` (`%` of Valid) | Mean Bad `>2 px` (`%` of Valid) | Unconditional Good `≤1 px` Completeness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4x Refinement (Independent)` | `93.83%` | `0.187 px` | `0.268 px` | `14.1 mm` | `0.28%` | `0.07%` | `93.55%` |
| **`4x Selective Repair (Independent)`** | **`97.74%` (`+3.91%`)** | **`0.173 px`** | **`0.246 px`** | **`13.0 mm`** | **`0.17%` (`-0.11%`)** | **`0.05%` (`-0.02%`)** | **`97.57%` (`+4.02%`)** |
| `4x Refinement (Derived Right)` | `93.77%` | `0.194 px` | `0.278 px` | `14.6 mm` | `0.34%` | `0.08%` | `93.44%` |
| **`4x Selective Repair (Derived Right)`** | **`97.67%` (`+3.90%`)** | **`0.177 px`** | **`0.252 px`** | **`13.4 mm`** | **`0.20%` (`-0.14%`)** | **`0.06%` (`-0.02%`)** | **`97.47%` (`+4.03%`)** |

### 2.2 Reference B: 5-Frame ZED 2 Independent Epipolar ZNCC Subpixel Reference (`record_ordinal 0..4`)

Evaluated against an estimator-independent 1D $9\times 9$ epipolar ZNCC subpixel template matcher (`peak ZNCC >= 0.90`, second-peak ratio `<= 0.78`, bidirectional LR symmetry `<= 0.65 px`) across all 5 distinct captured ZED 2 frames (`frame_id 5..9`), plus 20 fixed $32\times 32$ planar tiles selected purely from ZNCC planarity:

| Pipeline (`5` Distinct ZED 2 Frames, `1280×720`) | 5-Frame Mean LR-Valid Coverage | 5-Frame Mean ZNCC MAE (`px`) | 5-Frame Mean ZNCC Bad `>1 px` | 5-Frame Mean ZNCC Bad `>2 px` | Frame 0 Coherent Coverage | Fixed 20 Planar Tiles SVD RMSE (`20/20` Tiles) | Fixed 20 Planar Tile Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Legacy Census 24b Full-Res` | `20.82%` (F0) | `3.151 px` (F0) | `13.43%` (F0) | `10.67%` (F0) | `11.31%` | `11.675 px` | `46.67%` |
| `Full-Res Ortho-Hybrid (Independent)` | `29.11%` (F0) | `0.782 px` (F0) | `3.21%` (F0) | `1.39%` (F0) | `20.60%` | `0.415 px` | `78.59%` |
| `4x Refinement (Independent)` | `26.75%` | `0.898 px` | `6.00%` | `2.67%` | `17.81%` | `0.642 px` | `59.43%` |
| **`4x Selective Repair (Independent)`** | **`28.28%` (`+1.53%`)** | **`0.835 px` (`-0.063 px`)** | **`5.24%` (`-0.76%`)** | **`1.88%` (`-0.79%`)** | **`19.99%` (`+2.18%`)** | **`0.594 px`** | **`63.77%`** |
| `4x Refinement (Derived Right)` | `29.33%` | `0.952 px` | `6.70%` | `3.13%` | `18.53%` | `0.583 px` | `60.27%` |
| **`4x Selective Repair (Derived Right)`** | **`30.78%` (`+1.45%`)** | **`0.843 px` (`-0.109 px`)** | **`5.68%` (`-1.02%`)** | **`2.03%` (`-1.10%`)** | **`20.80%` (`+2.27%`)** | **`0.542 px`** | **`64.86%`** |

---

## 3. Step 9 (Gate 5) — Newly-Admitted Pixel Cohort Decomposition & Hardened 2-Pixel Peak Rule

In [`python/stereoforge/shaders/daxa/refinement.slang`](../python/stereoforge/shaders/daxa/refinement.slang), conditional per-pixel `selective_repair` (`bit 8u` of `params.subpixel`) tracks `best`, `second`, and `third` (`third_best`) Hamming costs so 1-px tie resolution only admits **isolated 2-pixel subpixel peaks** (`best <= 14u`, `plus == best || minus == best`, `third >= best + max(p.min_margin + 2u, 3u)`), rejecting $\ge 3$-pixel flat/repetitive plateaus:

| Disjoint Pixel Cohort (`1280×720` ZED 2 Frame 0) | `Independent 4×` Pixels (`%` Image) | `Independent 4×` ZNCC MAE / Bad `>1 px` / Bad `>2 px` | `Derived 4×` Pixels (`%` Image) | `Derived 4×` ZNCC MAE / Bad `>1 px` / Bad `>2 px` |
| --- | ---: | ---: | ---: | ---: |
| `baseline_valid` ($V_{\text{base}}$) | `239,379` (`25.97%`) | `0.909 px / 5.47% / 2.46%` | `259,465` (`28.15%`) | `1.044 px / 6.08% / 2.98%` |
| `selective_repair_valid` ($V_{\text{rep}}$) | `252,665` (`27.42%`) | **`0.857 px / 4.77% / 1.77%`** | `272,729` (`29.59%`) | **`0.916 px / 5.19% / 1.98%`** |
| **`newly_admitted_total` ($\Delta V_+ = V_{\text{rep}} \setminus V_{\text{base}}$)** | **`17,070` (`1.85%`)** | **`0.936 px / 5.74% / 1.80%`** | **`17,770` (`1.93%`)** | **`1.046 px / 6.71% / 1.76%`** |
| — `cohort_a_recovered_coarse_hole_or_span` | `484` (`0.05%`) | `0.000 px / 0.00% / 0.00%` | `258` (`0.03%`) | `0.000 px / 0.00% / 0.00%` |
| — `cohort_b_accepted_2px_subpixel_peak` | `16,449` (`1.79%`) | `0.941 px / 5.78% / 1.81%` | `17,361` (`1.88%`) | `1.053 px / 6.75% / 1.77%` |
| — `cohort_c_expanded_search_window` | `137` (`0.02%`) | `0.233 px / 0.00% / 0.00%` | `151` (`0.02%`) | `0.159 px / 0.00% / 0.00%` |

---

## 4. Step 10 (Gate 4) — Hardware GPU Timestamps & Multi-Trial Interleaved `240 FPS` (`4.167 ms`) Qualification

Evaluated across **3 interleaved trials × 5 distinct ZED 2 frames (`360` measured frames per configuration)** plus a canonical `120`-frame run (`20` warmup) with hardware `TimelineQueryPool` GPU timestamps:

| Dual-Preprocess + Refinement Pipeline (`1280×720`, `D=128`) | 3-Trial × 5-Frame Host Wall `p50 / p95` (`std`) | 3-Trial × 5-Frame GPU Queue Span `p50 / p95` (`std`) | Canonical 120f Host Wall `p50` (`FPS`) | Canonical 120f GPU Span `p50` (`FPS`) | GPU Coarse / Guided Match / Total (`ms`) | 30-Plane Bit-Exact | Vulkan Validation Errors / Warnings |
| --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| `dual4x_ortho-hybrid_layered_fast1` (`Independent`, `repair=0`) | `3.310 / 3.428 ms` (`0.076 ms`) | `2.703 / 2.734 ms` (`0.030 ms`) | `3.269 ms` (`305.9 FPS`) | `2.690 ms` (`371.7 FPS`) | `1.285 / 0.563 / 2.702 ms` | `true` | `0 / 0` |
| **`dual4x_ortho-hybrid_layered_fast1_repair1` (`Independent`, `repair=1`)** | **`3.406 / 3.665 ms` (`0.140 ms`)** | **`2.814 / 2.893 ms` (`0.072 ms`)** | **`3.355 ms` (`298.1 FPS`)** | **`2.788 ms` (`358.7 FPS`)** | **`1.272 / 0.678 / 2.792 ms`** | **`true`** | **`0 / 0`** |
| `dual4x_ortho-hybrid_derived_fast1` (`Derived`, `repair=0`) | `3.203 / 3.311 ms` (`0.081 ms`) | `2.626 / 2.668 ms` (`0.041 ms`) | `3.186 ms` (`313.8 FPS`) | `2.626 ms` (`380.7 FPS`) | `1.166 / 0.590 / 2.626 ms` | `true` | `0 / 0` |
| **`dual4x_ortho-hybrid_derived_fast1_repair1` (`Derived`, `repair=1`)** | **`3.334 / 3.432 ms` (`0.085 ms`)** | **`2.753 / 2.789 ms` (`0.042 ms`)** | **`3.347 ms` (`298.7 FPS`)** | **`2.747 ms` (`364.0 FPS`)** | **`1.164 / 0.724 / 2.747 ms`** | **`true`** | **`0 / 0`** |
