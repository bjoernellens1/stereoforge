# StereoForge Milestone v9.2: Live 60 Hz Physical Camera Qualification, Non-Circular Accuracy Audit & Selective 4× Refinement (`240+ FPS`)

## Overview

This document is **programmatically generated** from the two canonical provenance-attested JSON files (`step7-10-live-and-e2e-qualification-v9.json` and `step8-9-visual-accuracy-and-selective-repair-v9.json`) on `AMD Radeon 8060S Graphics (RADV STRIX_HALO, Mesa 26.2.5)` and physical `ZED 2 (SN26041639, 1280×720, D=128)`:

- **Tracked JSON Evidence:**
  - [`docs/evidence/v9-live-and-audit/step7-10-live-and-e2e-qualification-v9.json`](evidence/v9-live-and-audit/step7-10-live-and-e2e-qualification-v9.json)
  - [`docs/evidence/v9-live-and-audit/step8-9-visual-accuracy-and-selective-repair-v9.json`](evidence/v9-live-and-audit/step8-9-visual-accuracy-and-selective-repair-v9.json)
- **Tracked Visual Diagnostic Panels:**
  - [`docs/evidence/v9-live-and-audit/v9_visual_accuracy_audit_panels.png`](evidence/v9-live-and-audit/v9_visual_accuracy_audit_panels.png)
  - [`docs/evidence/v9-live-and-audit/v9_fixed_regions_overlay.png`](evidence/v9-live-and-audit/v9_fixed_regions_overlay.png)

---

## 1. Step 7 (Gate 1) — Real Physical `60 Hz` ZED 2 Camera (`--source camera`) + Concurrent ROS 2 Subscriber Delivery

### 1.1 Physical ZED 2 Hardware, Verified VID:PID Recovery, & 64-Bit Payload Checksums
- `/dev/video0` & `/dev/video1`: `ZED 2` (`SN26041639`), connected at `5000 Mbps` USB 3.0 (`/sys/bus/usb/devices/2-1.3`, `idle_runtime_power_status: "suspended"`), `live_frame_read_verified: true`.
- Both `src/capture.cpp` (`NativeCapture`) and `src/zed_capture.cpp` (`stereoforge._zed_capture`) verify a single Stereolabs ZED 2 MCU (`2B03:F780` in `/sys/class/hidraw/*/device/uevent`) before invoking `sl_oc::sensors::SensorCapture::resetSensorModule()` (`HIDIOCSFEATURE(2)` `[0x21, 0xE1]`), and trigger bounded active-stream timeout recovery if `snapshotFrame` times out for `>= 2000 ms`.
- Both [`stereoforge_ros2_depth_publisher`](../apps/ros2_depth_publisher.cpp) (`--verify-payload-checksum`) and [`stereoforge_ros2_delivery_probe`](../apps/ros2_delivery_probe.cpp) compute full-frame 64-bit FNV-1a checksums (`payload_fnv1a64`) and `valid_pixels` across all `3,686,400` payload bytes (`1280×720` `32FC1` depth and `rgba8` RGB) on every frame.

### 1.2 Primary Lossless (`KeepLast(16)`, `--qos-depth 16`) Live Physical `60 Hz` Camera Runs (`180` Frames)

| Live Physical Camera Run (`--source camera --qos-depth 16 --verify-payload-checksum`) | Capture / Publish / Subscriber Cadence | Published Depth / RGB | Subscriber Received | Full-Frame FNV-1a64 Checksum Verified | Queue / Ring / Slot Drops | Camera Inter-Frame `p50` (`std`) | Host Ingress → Enqueue `p50` | Frame Processing → Publish `p50 / p95` (`max`) | ROS 2 DDS Publish → Sub `p50` | Host Ingress → Subscriber E2E `p50 / p95` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4x_independent_180f` (`layered-per-path`, `independent-sgm`, `repair=0`) | `59.66 / 59.02 / 59.25 FPS` | `180 / 180` | `180 / 180 (`100.0%`, QoS=16)` | `180/180 (`0` err)` | **0 / 0 / 0** | `16.625 ms (`1.247 ms`)` | `0.010 ms` | `5.581 / 6.158 ms (`max 39.535 ms`)` | `0.203 ms` | **5.881 / 15.801 ms** |
| `4x_derived_180f` (`shared-census-sequential`, `left-aggregated-selection`, `repair=0`) | `60.00 / 59.33 / 59.38 FPS` | `180 / 180` | `180 / 180 (`100.0%`, QoS=16)` | `180/180 (`0` err)` | **0 / 0 / 0** | `16.626 ms (`0.059 ms`)` | `0.013 ms` | `5.719 / 6.369 ms (`max 39.890 ms`)` | `0.225 ms` | **6.060 / 16.432 ms** |
| **`4x_selective_repair_independent_180f` (`repair=1`)** | **`59.66 / 58.98 / 59.23 FPS`** | **`180 / 180`** | **`180 / 179 (`100.0%`, QoS=16)`** | **`180/180 (`0` err)`** | **0 / 0 / 0** | **`16.626 ms (`1.248 ms`)`** | **`0.010 ms`** | **`5.710 / 6.692 ms (`max 41.335 ms`)`** | **`0.198 ms`** | **6.009 / 16.027 ms** |
| **`4x_selective_repair_derived_180f` (`repair=1`)** | **`60.00 / 59.33 / 59.56 FPS`** | **`180 / 180`** | **`180 / 180 (`100.0%`, QoS=16)`** | **`180/180 (`0` err)`** | **0 / 0 / 0** | **`16.626 ms (`0.059 ms`)`** | **`0.009 ms`** | **`5.430 / 6.555 ms (`max 39.678 ms`)`** | **`0.273 ms`** | **5.859 / 16.850 ms** |

### 1.3 Latest-Frame (`KeepLast(1)`), Extended `600`-Frame (`10.0 s`) Endurance, & Shared-GPU Contention Live Camera Runs

| Live Physical Camera Regime | Capture / Publish / Subscriber Cadence | Published Depth / RGB | Subscriber Received | Full-Frame FNV-1a64 Checksum Verified | Queue / Ring / Slot Drops | Camera Inter-Frame `p50` (`std`) | Host Ingress → Enqueue `p50` | Frame Processing → Publish `p50 / p95` (`max`) | ROS 2 DDS Publish → Sub `p50` | Host Ingress → Subscriber E2E `p50 / p95` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `keep_last_1_latest_frame_180f` (`--qos-depth 1`, latest-frame mode) | `60.00 / 59.33 / 58.25 FPS` | `180 / 180` | `176 / 173 (`97.8%`, QoS=1)` | `176/176 (`0` err)` | **0 / 0 / 0** | `16.626 ms (`0.059 ms`)` | `0.020 ms` | `5.796 / 6.822 ms (`max 40.290 ms`)` | `0.182 ms` | **6.049 / 16.550 ms** |
| **`extended_600f_10s_endurance` (`600` frames = `10.0 s` continuous live 60 Hz)** | **`60.00 / 59.79 / 59.80 FPS`** | **`600 / 600`** | **`600 / 600 (`100.0%`, QoS=16)`** | **`600/600 (`0` err)`** | **0 / 0 / 0** | **`16.626 ms (`0.059 ms`)`** | **`0.011 ms`** | **`5.835 / 6.896 ms (`max 40.357 ms`)`** | **`10.074 ms`** | **15.715 / 26.100 ms** |
| **`shared_gpu_contention_180f` (concurrent background Vulkan compute load)** | **`60.00 / 59.32 / 59.56 FPS`** | **`180 / 180`** | **`180 / 179 (`100.0%`, QoS=16)`** | **`180/180 (`0` err)`** | **0 / 0 / 0** | **`16.626 ms (`0.059 ms`)`** | **`0.018 ms`** | **`5.674 / 6.650 ms (`max 40.049 ms`)`** | **`0.188 ms`** | **5.879 / 15.793 ms** |

---

## 2. Step 8 (Gate 2) — Non-Circular Quantitative Accuracy Audit

### 2.1 Reference A: 5-Scene `1280×720` Known-Ground-Truth Suite on Native Vulkan GPU (`daxa_refinement_benchmark --synthetic-yuyv`)

| Pipeline (`1280×720` Native Vulkan GPU Execution) | Mean Valid Coverage | Mean Disparity MAE (`px`) | Mean Bad `>1 px` (`%` of Valid) | Mean Bad `>2 px` (`%` of Valid) | Unconditional Good `≤1 px` Completeness | Unconditional Good `≤2 px` Completeness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `4x Refinement (Independent)` | `93.83%` | `0.187 px` | `0.28%` | `0.23%` | `93.55%` | `93.61%` |
| **`4x Selective Repair (Independent)`** | **`97.74%`** | **`0.173 px`** | **`0.17%`** | **`0.12%`** | **`97.57%`** | **`97.62%`** |
| `4x Refinement (Derived Right)` | `93.77%` | `0.194 px` | `0.34%` | `0.28%` | `93.44%` | `93.50%` |
| **`4x Selective Repair (Derived Right)`** | **`97.67%`** | **`0.177 px`** | **`0.20%`** | **`0.15%`** | **`97.47%`** | **`97.52%`** |
| `OpenCV StereoSGBM` | `92.63%` | `0.101 px` | `0.06%` | `0.06%` | `92.58%` | `92.58%` |

### 2.2 Reference B: 5-Frame ZED 2 Multi-Scale (`7×7 + 9×9`) Epipolar ZNCC Subpixel Reference (`3.44%` Anchor Coverage on Frame 0)

| Pipeline (`5` Distinct ZED 2 Frames, `1280×720`) | 5-Frame Mean LR-Valid Coverage | 5-Frame Mean ZNCC MAE (`px`) | 5-Frame Mean ZNCC Bad `>1 px` | 5-Frame Mean ZNCC Bad `>2 px` | Frame 0 Coherent Coverage | Fixed 20 Planar Tiles SVD RMSE (`20/20` Tiles) | Fixed 20 Planar Tile Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Legacy Census 24b Full-Res` | `20.82%` (F0) | `1.274 px` (F0) | `12.34%` (F0) | `4.84%` (F0) | `11.31%` | `12.466 px` | `28.80%` |
| `Full-Res Ortho-Hybrid (Independent)` | `29.11%` (F0) | `1.158 px` (F0) | `6.72%` (F0) | `3.16%` (F0) | `20.60%` | `1.349 px` | `52.95%` |
| `4x Refinement (Independent)` | `26.75%` | `1.248 px` | `11.92%` | `5.22%` | `18.35%` | `1.019 px` | `44.20%` |
| **`4x Selective Repair (Independent)`** | **`28.28%`** | **`1.172 px`** | **`10.92%`** | **`4.13%`** | **`20.33%`** | **`1.023 px`** | **`47.47%`** |
| `4x Refinement (Derived Right)` | `29.33%` | `1.265 px` | `12.44%` | `5.52%` | `19.15%` | `1.076 px` | `43.53%` |
| **`4x Selective Repair (Derived Right)`** | **`30.78%`** | **`1.160 px`** | **`11.34%`** | **`4.26%`** | **`21.19%`** | **`1.084 px`** | **`46.76%`** |

---

## 3. Step 9 (Gate 5) — Newly-Admitted Pixel Cohort Decomposition & 100%-Coverage Epipolar Reprojection Audit

Every cohort reports both **multi-scale ZNCC reference error** (returning explicit `null` if `joint_independent_ref_pixels == 0`) AND **dense 100%-coverage epipolar photometric reprojection error** ($|I_L(x,y) - I_R(x - \hat{d}, y)|$), $5\times 5$ patch ZNCC (`ZNCC5`), and $3\times 3$ local planar inlier percentage across all in-bounds pixels:

| Disjoint Pixel Cohort (`1280×720` ZED 2 Frame 0) | `Independent 4×` Pixels (`%` Image) | `Independent 4×` ZNCC MAE / Bad `>1 px` / Bad `>2 px` (`Ref Support`) \| Dense Reproj / ZNCC5 / Planar | `Derived 4×` Pixels (`%` Image) | `Derived 4×` ZNCC MAE / Bad `>1 px` / Bad `>2 px` (`Ref Support`) \| Dense Reproj / ZNCC5 / Planar |
| --- | ---: | ---: | ---: | ---: |
| `baseline_valid` ($V_{\text{base}}$) | `239,379` (`25.97%`) | `1.253 px / 11.62% / 5.08%` (`21721/239379` = `9.1%` ZNCC) | `6.00 luma / 0.200 ZNCC5 / 80.7% planar` | `260,782` (`28.30%`) | `1.188 px / 12.51% / 5.56%` (`20874/260782` = `8.0%` ZNCC) | `7.39 luma / 0.177 ZNCC5 / 79.5% planar` |
| `selective_repair_valid` ($V_{\text{rep}}$) | `252,665` (`27.42%`) | `1.183 px / 10.82% / 4.17%` (`23306/252665` = `9.2%` ZNCC) | `5.09 luma / 0.204 ZNCC5 / 82.7% planar` | `273,660` (`29.69%`) | `1.094 px / 11.41% / 4.30%` (`22328/273660` = `8.2%` ZNCC) | `6.32 luma / 0.182 ZNCC5 / 81.6% planar` |
| **`newly_admitted_total` ($\Delta V_+ = V_{\text{rep}} \setminus V_{\text{base}}$)** | **`17,070` (`1.85%`)** | `1.145 px / 11.74% / 3.73%` (`1823/17070` = `10.7%` ZNCC) | `3.10 luma / 0.243 ZNCC5 / 83.9% planar` | **`17,770` (`1.93%`)** | `1.097 px / 11.90% / 3.24%` (`1757/17770` = `9.9%` ZNCC) | `3.65 luma / 0.225 ZNCC5 / 82.9% planar` |
| — `cohort_a_recovered_coarse_hole_or_span` | `484` (`0.05%`) | `45.856 px / 100.00% / 100.00%` (`1/484` = `0.2%` ZNCC) | `8.01 luma / 0.038 ZNCC5 / 46.0% planar` | `275` (`0.03%`) | `null` (`0/275` ZNCC ref) | `6.81 luma / 0.009 ZNCC5 / 52.38% planar` |
| — `cohort_b_accepted_2px_subpixel_peak` | `16,449` (`1.78%`) | `1.126 px / 11.77% / 3.70%` (`1810/16449` = `11.0%` ZNCC) | `2.97 luma / 0.245 ZNCC5 / 84.8% planar` | `17,251` (`1.87%`) | `1.106 px / 12.00% / 3.27%` (`1741/17251` = `10.1%` ZNCC) | `3.57 luma / 0.225 ZNCC5 / 83.7% planar` |
| — `cohort_c_expanded_search_window` | `137` (`0.01%`) | `0.237 px / 0.00% / 0.00%` (`12/137` = `8.8%` ZNCC) | `10.13 luma / 0.431 ZNCC5 / 51.8% planar` | `244` (`0.03%`) | `0.207 px / 0.00% / 0.00%` (`16/244` = `6.6%` ZNCC) | `8.04 luma / 0.268 ZNCC5 / 42.6% planar` |
| — Region: `planar_20tiles` | `673` (`0.07%`) | `0.547 px / 9.57% / 4.26%` (`94/673` = `14.0%` ZNCC) | `2.61 luma / 0.216 ZNCC5 / 86.8% planar` | `667` (`0.07%`) | `0.537 px / 8.24% / 3.53%` (`85/667` = `12.7%` ZNCC) | `2.59 luma / 0.209 ZNCC5 / 84.9% planar` |
| — Region: `boundary_bands` | `5,242` (`0.57%`) | `1.116 px / 10.11% / 3.33%` (`1652/5242` = `31.5%` ZNCC) | `4.36 luma / 0.626 ZNCC5 / 91.2% planar` | `5,311` (`0.58%`) | `1.068 px / 10.15% / 2.76%` (`1596/5311` = `30.1%` ZNCC) | `4.72 luma / 0.605 ZNCC5 / 90.3% planar` |
| — Region: `low_texture_tiles` | `1,243` (`0.14%`) | `0.854 px / 30.43% / 13.04%` (`46/1243` = `3.7%` ZNCC) | `3.66 luma / 0.081 ZNCC5 / 83.5% planar` | `1,430` (`0.15%`) | `0.754 px / 25.00% / 6.25%` (`48/1430` = `3.4%` ZNCC) | `4.78 luma / 0.072 ZNCC5 / 82.2% planar` |
| — Region: `repetitive_tiles` | `50` (`0.01%`) | `0.556 px / 10.00% / 10.00%` (`10/50` = `20.0%` ZNCC) | `3.67 luma / 0.186 ZNCC5 / 74.0% planar` | `69` (`0.01%`) | `0.588 px / 9.09% / 9.09%` (`11/69` = `15.9%` ZNCC) | `3.58 luma / 0.171 ZNCC5 / 79.7% planar` |

---

## 4. Step 10 (Gate 4) — Hardware GPU Single-Interval & Stage-Sum Timestamps + Multi-Trial Interleaved `240 FPS` (`4.167 ms`) Qualification

Both the **single hardware `TimelineQueryPool` timestamp interval** (`gpu_single_interval_upload_to_metric_ms` = `refinement.end_tick - dual_preprocess.start_tick` on `daxa::QUEUE_MAIN`) and the **per-stage GPU sum** (`gpu_stage_sum_ms` = `preprocess + coarse + refinement`) are reported alongside host-wall timings across **3 interleaved trials × 5 distinct ZED 2 frames (`360` measured frames per configuration)** and a canonical `120`-frame run:

| Dual-Preprocess + Refinement Pipeline (`1280×720`, `D=128`) | 3-Trial × 5-Frame Host Wall `p50 / p95` (`std`) | 3-Trial × 5-Frame Single-Interval GPU Span `p50 / p95` (`std`) | 3-Trial × 5-Frame Stage-Sum GPU `p50 / p95` | Canonical 120f Host Wall `p50` (`FPS`) | Canonical 120f Single-Interval GPU `p50` (`FPS`) | Preprocess / Coarse / Refinement GPU (`ms`) | 30-Plane Bit-Exact | Vulkan Validation Errors / Warnings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| `dual4x_ortho-hybrid_layered_fast1` (`Independent`, `repair=0`) | `3.290 / 3.916 ms` (`0.180 ms`) | `3.104 / 3.422 ms` (`0.120 ms`) | `3.051 / 3.249 ms` | `3.288 ms` (`304.1 FPS`) | `3.090 ms` (`323.6 FPS`) | `0.351 / 1.287 / 1.419 ms` | `true` | `0 / 0` |
| **`dual4x_ortho-hybrid_layered_fast1_repair1` (`Independent`, `repair=1`)** | **`3.400 / 3.508 ms` (`0.085 ms`)** | **`3.214 / 3.278 ms` (`0.055 ms`)** | **`3.163 / 3.209 ms`** | **`3.388 ms` (`295.1 FPS`)** | **`3.201 ms` (`312.4 FPS`)** | **`0.339 / 1.244 / 1.480 ms`** | **`true`** | **`0 / 0`** |
| `dual4x_ortho-hybrid_derived_fast1` (`Derived`, `repair=0`) | `3.212 / 3.916 ms` (`0.234 ms`) | `3.027 / 3.415 ms` (`0.120 ms`) | `2.974 / 3.256 ms` | `3.184 ms` (`314.1 FPS`) | `3.014 ms` (`331.7 FPS`) | `0.350 / 1.160 / 1.471 ms` | `true` | `0 / 0` |
| **`dual4x_ortho-hybrid_derived_fast1_repair1` (`Derived`, `repair=1`)** | **`3.338 / 3.780 ms` (`0.167 ms`)** | **`3.160 / 3.431 ms` (`0.103 ms`)** | **`3.103 / 3.231 ms`** | **`3.329 ms` (`300.4 FPS`)** | **`3.142 ms` (`318.3 FPS`)** | **`0.352 / 1.161 / 1.585 ms`** | **`true`** | **`0 / 0`** |
