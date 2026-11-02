# Correspondence-First `ortho_thermo32` & 240+ FPS Multiresolution Refinement Evidence (`v8`)

## Executive Summary

Following the **StereoForge Code and Results Review (`main @ 114f800`)**, we executed the 6-step correspondence-first research and engineering plan on the pinned ZED 2 hardware fixture ([`configs/cameras/zed2/fixtures/SN26041639-fixture/manifest.json`](../configs/cameras/zed2/fixtures/SN26041639-fixture/manifest.json)) and `AMD Radeon 8060S Graphics (RADV STRIX_HALO)`:

1. **Step 1 — Controlled Exact-Luma OpenCV Cross-Check ([`benchmarks/stereo_reference_crosscheck_v5.py`](../benchmarks/stereo_reference_crosscheck_v5.py), [`docs/evidence/v8-review-milestone/step1-exact-luma-crosscheck-v5.json`](evidence/v8-review-milestone/step1-exact-luma-crosscheck-v5.json)):**
   Eliminated the RGB-to-grayscale conversion confounder by feeding the exact native `uint32` `gray15` luma buffers (`expected/left_luma.u32`, `expected/right_luma.u32`) into OpenCV `StereoSGBM` via `--input-mode exact-fixture-luma`. The grayscale rounding difference was at most `±1` intensity level on `26.86%` of pixels (`RMSE = 0.518` levels) and changed OpenCV's LR-valid coverage by `< 0.08%` (`33.36%` vs `33.44%`), proving the gap between baseline 24-bit Census (`19.56%` LR-valid) and OpenCV SGBM (`33.36%` LR-valid) is strictly caused by the **correspondence cost model**.
2. **Step 2 — Pre-SGM vs. Post-SGM Cost-Curve Inspector ([`benchmarks/matching_cost_inspector_v8.py`](../benchmarks/matching_cost_inspector_v8.py), [`docs/evidence/v8-review-milestone/step2-3-4-matching-cost-inspector-v8.json`](evidence/v8-review-milestone/step2-3-4-matching-cost-inspector-v8.json)):**
   Inspected raw matching cost $C(x, y, d)$ and 4-path aggregated cost $S(x, y, d)$ across all $d \in [0, 127]$. Under 24-bit $5\times 5$ Census, pixels that fail LR consistency have **3.10 local minima within $+2$ bits of `best_cost`** even after 4-path SGM, and on `53.26%` of pixels where Native and OpenCV disagree (`|d_native - d_cv| > 2 px`), the true disparity already has a **higher raw pre-SGM Census cost** ($\text{median } \Delta C_{\text{raw}} = +1.0\text{ bit}$) before SGM runs.
3. **Steps 3 & 4 — Single-Variable Matching-Cost Ablation & Independent Geometric Qualification:**
   Evaluated 4 matching-cost models (`census_24bit`, `census_32bit_multiscale`, `census_box3x3`, and `ortho_hybrid_u32` / `ortho_thermo32`) with and without non-adjacent uniqueness gating (`|d - d^*| > 1`) against independent geometric reference metrics (planar surface SVD fit RMSE on smooth patches, Sobel edge completeness, and 8-connected disparity speckle survival). **`ortho_thermo32`** (which packs 4 thermometer-coded 8-bit 5×5 sub-window horizontal-gradient and vertical-luma features into a single `uint32` so standard `countbits(a ^ b)` computes the exact $L_1$ feature distance `[0..32]` with zero SGM shader or buffer changes) increased LR-valid coverage from `19.56%` to **`29.11%`** (`41.28%` with derived right reference), cut median LR disagreement by **3.1×** (`14.48 px` $\to$ `4.93 px`), and reduced planar disparity RMSE by **4.6×–9.9×** (`13.30 px` $\to$ `2.90 px` raw / `1.35 px` gated).
4. **Step 5 — Native Slang/Daxa Port & Bit-Exact Reference Qualification ([`shaders/ops/sgm.slang`](../shaders/ops/sgm.slang), [`src/daxa_stereo.cpp`](../src/daxa_stereo.cpp), [`docs/evidence/v8-review-milestone/step5-native-ortho-hybrid-depth-benchmark.json`](evidence/v8-review-milestone/step5-native-ortho-hybrid-depth-benchmark.json)):**
   Ported `sgm_ortho_thermo_census` to the native Daxa stereo pipeline (`--descriptor-mode ortho-hybrid`) with **100% bit-exact parity** against the C++ scalar reference (`both_stereo_integer_reference_checked: true`, `lr_mask_metadata_reference_checked: true`), zero additional buffer memory (`840,500,528` bytes independent / `575,079,728` bytes derived), and `0` Vulkan validation errors/warnings.
5. **Step 6 — Multiresolution Refinement (`240+ FPS`) & 5-Frame Temporal Stability ([`shaders/daxa/refinement.slang`](../shaders/daxa/refinement.slang), [`src/daxa_refinement.cpp`](../src/daxa_refinement.cpp), [`docs/evidence/v8-review-milestone/step6-multiresolution-refinement-240fps-v8.json`](evidence/v8-review-milestone/step6-multiresolution-refinement-240fps-v8.json)):**
   Diagnosed why the legacy refinement shader rejected ~95% of real ZED 2 pixels (requiring all 4 bilinear coarse taps simultaneously valid with `confidence >= 0.05` caused $p^4$ dropout, and single-pixel Hamming distance without spatial support caused ties and boundary rejections). Implemented validity-weighted coarse prior interpolation + 5-tap cross `ortho_thermo32` matching + coarse-SGM prior tether in `shaders/daxa/refinement.slang`, achieving:
   - **`297.3 FPS` (`3.363 ms` host wall p50, `3.081 ms` GPU wait p50)** on $4\times$ coarse (`320×180` $D=32 \to 1280×720$ $D=128$, `dual4x_ortho-hybrid_derived_fast1`) with **`26.75%` LR-valid coverage** and **`1.27 px` planar RMSE** (`428` patches).
   - **`282.1 FPS` (`3.545 ms` host wall p50, `3.269 ms` GPU wait p50)** on $4\times$ coarse with independent coarse SGM (`dual4x_ortho-hybrid_layered_fast1`) with **`24.77%` LR-valid coverage** and **`1.21 px` planar RMSE** (`372` patches).
   - **2.06× higher corroborated current-frame coverage (`19.32%` vs `9.39%`)** and **1.94× higher 100 ms hold coverage (`22.71%` vs `11.72%`)** across the 5-frame ZED 2 capture sequence under `TemporalConsensus`.

---

## Step 1: Controlled Exact-Luma OpenCV Cross-Check

| Metric | `rect-rgb-cvtcolor` (Legacy) | `exact-fixture-luma` (`left_luma.u32`) | Finding |
| :--- | :---: | :---: | :--- |
| Max / RMSE Luma Diff vs Native `gray15` | `1` level (`26.86%` pixels, RMSE `0.518`) | **`0` levels (`0.00%` pixels, exact)** | Eliminates grayscale conversion confounder |
| Native 24-bit Census Left / Right Selected | `85.94%` / `85.17%` | `85.94%` / `85.17%` | Identical native input |
| Native 24-bit Census LR-Valid Coverage | `19.56%` (`26.25%` of both-selected) | `19.56%` (`26.25%` of both-selected) | Median LR error `15.25 px` |
| OpenCV SGBM (`blockSize=5, P1=200, P2=800`) LR-Valid | `33.44%` (`75.86%` of both-selected) | **`33.36%` (`75.77%` of both-selected)** | `< 0.08%` shift from luma rounding |
| OpenCV SGBM (`blockSize=3, P1=72, P2=288, u=0`) LR-Valid | — | **`36.35%` (`46.97%` of both-selected)** | Even with $u=0$, Birchfield-Tomasi + Sobel is $1.86\times$ more LR-consistent |

---

## Steps 2, 3 & 4: Pre/Post-SGM Cost-Curve Inspection, Ablation & Geometric Qualification

### Step 2: Pre-SGM vs. Post-SGM Cost-Curve Diagnostics (`SN26041639-fixture`)

| Cost-Curve Diagnostic on `SN26041639-fixture` | LR-Consistent Pixels (`19.56%`) | LR-Inconsistent Selected Pixels (`66.38%`) |
| :--- | :---: | :---: |
| Pre-SGM Raw Winner Margin (`second - best`, bits) | `1.14` | `0.15` |
| Post-SGM Aggregated Margin (`second - best`) | `4.41` | `3.31` |
| Post-SGM Non-Adjacent Margin ($\min_{|d-d^*|>1} S(d) - S(d^*)$) | `12.66` | **`6.16`** |
| Post-SGM Near-Tie Basins Within `+2` of `best_cost` | `2.37` | **`3.10`** |
| Disagreement Rate Where Adjacent $|d_2 - d^*| = 1$ Masks Multi-Modal Basin | `72.42%` | **`79.88%`** |
| Pre-SGM Failure Rate on Native-vs-OpenCV Disagreements ($C_{\text{raw}}(d_{\text{cv}}) \ge C_{\text{raw}}(d_{\text{native}})$) | — | **`53.26%`** (median $\Delta C_{\text{raw}} = +1.0$ bit) |

### Steps 3 & 4: Single-Variable Matching-Cost & Selector Ablation Table

| Cost Model ($D=128$, 4-Path SGM $P_1=8, P_2=32$) | Selector Policy | Left Valid % | LR-Valid % | LR Agreement (Both-Selected) | Median LR Error | Speckle Survival % | Coherent Surface % | Planar RMSE (Median px) | Edge LR-Valid % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`census_24bit`** (Baseline) | Raw (`min_margin=1`) | `85.94%` | `19.56%` | `26.25%` | `15.25 px` | `54.32%` | `11.31%` | `13.30 px` | `39.83%` |
| `census_24bit` | Non-adjacent `u >= 0.08, m >= 3` | `53.27%` | `16.43%` | `50.25%` | `0.98 px` | `57.21%` | `9.40%` | `5.34 px` | `38.28%` |
| `census_32bit_multiscale` | Raw (`min_margin=1`) | `86.48%` | `20.66%` | `27.64%` | `14.12 px` | `53.68%` | `11.09%` | `11.88 px` | `40.42%` |
| `census_box3x3` ($3\times 3$ sum) | Raw (`min_margin=1`) | `88.91%` | `25.88%` | `33.24%` | `10.12 px` | `64.54%` | `16.70%` | `8.29 px` | `43.25%` |
| **`ortho_hybrid_u32` (`ortho_thermo32`)** | Raw (`min_margin=1`) | `85.00%` | **`29.11%`** | **`39.39%`** | **`4.93 px`** | **`70.74%`** | **`20.60%`** | **`2.90 px`** | **`42.13%`** |
| **`ortho_hybrid_u32` (`ortho_thermo32`)** | Non-adjacent `u >= 0.08, m >= 3` | `47.55%` | **`21.72%`** | **`69.03%`** | **`0.46 px`** | **`73.92%`** | **`16.05%`** | **`1.35 px`** | **`37.49%`** |

---

## Step 5: Native Slang/Daxa Full-Resolution GPU Qualification (`1280×720`, $D=128$)

| Native GPU Run (`daxa_depth_benchmark --check-reference --validation`) | Descriptor | Right Reference Mode | LR-Valid % | LR Agree % | Median LR Err | Planar RMSE | Speckle Survival | Buffer Memory | GPU Queue p50 / p95 | Scalar Ref Parity & Validation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `depth_census_sequential` | `census` | `independent-sgm` | `20.82%` | `26.92%` | `14.48 px` | `13.30 px` | `54.32%` | `840.5 MB` | `33.09 / 34.47 ms` | Exact (`0` errs/warns) |
| **`depth_ortho-hybrid_sequential`** | **`ortho-hybrid`** | `independent-sgm` | **`29.11%`** | **`39.39%`** | **`4.93 px`** | **`2.90 px`** | **`70.74%`** | `840.5 MB` | **`32.40 / 33.29 ms`** | **Exact (`0` errs/warns)** |
| `depth_census_layered` | `census` | `independent-sgm` | `20.82%` | `26.92%` | `14.48 px` | `13.30 px` | `54.32%` | `840.5 MB` | `42.32 / 43.60 ms` | Exact (`0` errs/warns) |
| **`depth_ortho-hybrid_layered`** | **`ortho-hybrid`** | `independent-sgm` | **`29.11%`** | **`39.39%`** | **`4.93 px`** | **`2.90 px`** | **`70.74%`** | `840.5 MB` | `42.72 / 44.89 ms` | **Exact (`0` errs/warns)** |
| `depth_census_derived` | `census` | `left-aggregated-selection` | `32.36%` | `40.73%` | `3.01 px` | `21.02 px` | `41.63%` | `575.1 MB` | `19.28 / 19.90 ms` | Exact (`0` errs/warns) |
| **`depth_ortho-hybrid_derived`** | **`ortho-hybrid`** | `left-aggregated-selection` | **`41.28%`** | **`55.21%`** | **`0.69 px`** | **`5.38 px`** | **`61.85%`** | `575.1 MB` | **`19.41 / 19.85 ms`** | **Exact (`0` errs/warns)** |

---

## Step 6: Multiresolution Refinement (`60 Hz` $\to$ `297 FPS`) & Temporal Stability

### Multiresolution GPU Performance & Geometric Quality (`1280×720`, $D=128$)

| Benchmark Configuration (`--subpixel 1 --fast-mode 1 --validation`) | Coarse Grid | Descriptor | Coarse Right Mode | Full-Res LR-Valid % | LR Agree % | Planar RMSE (Patches) | Speckle Survival | Host Wall p50 / p95 | GPU Wait p50 | Throughput (Host p50 / GPU Wait) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ref2x_census_layered_fast1` | `640×360` D64 | `census` | `independent-sgm` | `2.04%` | `81.75%` | `0.90 px` (22) | `29.97%` | `6.44 / 6.63 ms` | `6.01 ms` | `155.3 / 166.3 FPS` |
| **`ref2x_ortho-hybrid_layered_fast1`** | `640×360` D64 | **`ortho-hybrid`** | `independent-sgm` | **`25.28%`** | **`73.21%`** | **`1.49 px` (372)** | **`66.41%`** | `7.03 / 7.12 ms` | `6.73 ms` | **`142.3 / 148.6 FPS`** |
| **`dual2x_ortho-hybrid_derived_fast1`** | `640×360` D64 | **`ortho-hybrid`** | `left-aggregated` | **`28.16%`** | **`69.31%`** | **`1.72 px` (445)** | **`63.41%`** | **`6.12 / 6.54 ms`** | **`5.85 ms`** | **`163.4 / 170.8 FPS`** |
| `ref4x_census_layered_fast1` | `320×180` D32 | `census` | `independent-sgm` | `2.86%` | `74.04%` | `1.19 px` (42) | `23.97%` | `3.05 / 3.19 ms` | `2.77 ms` | `327.6 / 361.7 FPS` |
| **`dual4x_ortho-hybrid_layered_fast1`** | `320×180` D32 | **`ortho-hybrid`** | `independent-sgm` | **`24.77%`** | **`62.79%`** | **`1.21 px` (372)** | **`66.39%`** | **`3.54 / 4.23 ms`** | **`3.27 ms`** | **`282.1 / 305.9 FPS`** |
| **`dual4x_ortho-hybrid_derived_fast1`** | `320×180` D32 | **`ortho-hybrid`** | `left-aggregated` | **`26.75%`** | **`60.78%`** | **`1.27 px` (428)** | **`63.61%`** | **`3.36 / 3.46 ms`** | **`3.08 ms`** | **`297.3 / 324.5 FPS`** |

### 5-Frame ZED 2 Capture Sequence Temporal Stability ([`benchmarks/temporal_consensus_v2.py`](../benchmarks/temporal_consensus_v2.py))

| Correspondence Baseline (`SN26041639-fixture`, 5 Frames) | Mean Raw LR-Valid % | Frame-to-Frame Raw Dropout % | Confirmed Current Coverage % | Confirmed + 100 ms Hold Coverage % |
| :--- | :---: | :---: | :---: | :---: |
| `census_24bit` (Legacy Baseline) | `19.78%` | `48.47%` | `9.39%` | `11.72%` |
| **`ortho_hybrid_u32` (`ortho_thermo32`)** | **`29.79%`** (`+10.01%`) | **`31.08%`** (`-17.39%`) | **`19.32%`** (**`2.06×`**) | **`22.71%`** (**`1.94×`**) |
