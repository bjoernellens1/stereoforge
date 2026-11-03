# StereoForge Milestone v10 — Ultra-Fast GPU Processing, Portable Slang-to-CPU Execution & Cooperative GPU Duty-Cycle Governance

## 1. Executive Summary

Milestone v10 addresses two independent engineering objectives defined in the v10 architecture plan:

1. **Ultra-Fast GPU Processing (`--profile ultra-fast` vs. `--profile quality`):**
   - Reduces `1280×720` end-to-end stereo GPU processing time from `3.937 ms` (`quality`, planar enabled) to **`3.130 ms` (`ultra-fast` with planar recovery, `p95 = 3.149 ms`, `319.5 FPS` equivalent)** and **`2.554 ms` (`ultra-fast` measured-only, `p95 = 2.571 ms`, `391.5 FPS` equivalent)**.
   - Eliminates `66.36 MB/frame` of redundant right-eye `metric` and `completed` buffer writes (`refinement_metric_ms`: `0.607 ms` $\rightarrow$ `0.304 ms` with planar recovery, and `0.154 ms` measured-only), reduces `refinement_guided_match_ms` from `0.860 ms` to `0.507 ms` via asymmetric right-eye single-tap Census LR verification, and uses `shared-census-sequential` + `left-aggregated-selection` coarse SGM.
   - Reduces per-frame net dynamic GPU energy from `121.47 mJ/frame` (`180.25 mJ` gross) to **`85.09 mJ/frame` (`131.82 mJ` gross)** with planar recovery and **`65.55 mJ/frame` (`103.68 mJ` gross)** measured-only.
2. **Portable Universal Execution (`slangc -target cpp -DSF_CPU_TARGET=1`):**
   - Compiles all 6 production Slang compute entrypoints (`rectify`, `census`, `match`, `lr`, `metric`, `planar`) in `python/stereoforge/shaders/daxa/refinement.slang` directly into native C++ (`libstereoforge_slang_cpu.so`) via `slangc -target cpp -DSF_CPU_TARGET=1` + `g++ -O3 -fopenmp` without CUDA, ROCm, or a mandatory Vulkan GPU.
   - Achieves **`100.000%` bit-exact parity** against the Vulkan SPIR-V GPU backend on `census`, `census_validity`, `lr` validity, `metric_validity`, and `provenance` across all `1,843,200` (`2 × 1280 × 720`) pixels (`112.95 FPS` at `640×360` mobile profile, `22.57 FPS` at `1280×720` full resolution on CPU).

---

## 2. Roofline Model & Joint 3-Metric Baseline (`v10.0`)

From `benchmarks/results/v10_0_roofline_and_energy_baseline.json`:

| Stage | Read (MB) | Write (MB) | Total Traffic (MB) | Measured GPU (ms) | Roofline Floor at 288 GB/s (ms) | Achieved Effective BW (GB/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `dual_preprocess` | 19.35 | 45.62 | 64.97 | 0.311 | 0.238 | 209.08 |
| `coarse_sgm` | 48.20 | 42.10 | 90.30 | 1.182 | 0.329 | 76.44 |
| `refinement_census` | 14.75 | 14.75 | 29.49 | 0.228 | 0.107 | 129.35 |
| `refinement_guided_match` | 125.34 | 81.10 | 206.44 | 0.858 | 0.752 | 240.60 |
| `refinement_lr` | 22.12 | 22.12 | 44.24 | 0.168 | 0.161 | 263.31 |
| `refinement_metric` (with 2-eye `completed`) | 14.75 | 103.22 | 117.96 | 0.607 | 0.430 | 194.34 |
| `refinement_planar_recovery` | 1.80 | 0.60 | 2.40 | 0.369 | 0.009 | 6.50 |
| **Total Pipeline (`v9.4` baseline)** | **246.30** | **309.50** | **555.80** | **3.723** | **2.014** | **149.29** |

- **Joint 3-Metric Baseline (`v10.0`):**
  - $T_{\text{frame}}$: `gpu_combined_p50_ms = 3.697 ms` (`p95 = 4.085 ms`), consuming `11.09%` serialized GPU time at `30 Hz` and `22.18%` at `60 Hz`.
  - $E_{\text{frame}}$: Idle APU power `14.93 W` (`43.0°C`), active mean power `43.31 W` (`63.02 W` peak), gross energy **`160.10 mJ/frame`**, net dynamic energy **`104.90 mJ/frame`**.
  - $\Delta T_{\text{competing workload}}$: Reference competing GPU compute probe (`40` frames) slows from `3.069 ms` solo to `4.564 ms` (`+48.75%`) under continuous back-to-back contention, corresponding to **`+10.81%` duty-scaled slowdown at `60 Hz`** and **`+5.41%` at `30 Hz`**.

---

## 3. Ultra-Fast GPU Pipeline (`v10.2`) & Stage Breakdown

From `benchmarks/results/v10_ultra_fast_and_universal_qualification.json` (`180` measured frames, `20` warmup):

| Metric | `--profile quality` (Planar=1) | `--profile ultra-fast` (Planar=1) | `--profile ultra-fast` (Planar=0) |
| :--- | :---: | :---: | :---: |
| **Combined GPU `p50` (ms)** | `3.937 ms` | **`3.130 ms` (`1.26×` speedup)** | **`2.554 ms` (`1.54×` speedup)** |
| **Combined GPU `p95` (ms)** | `4.301 ms` | **`3.149 ms`** | **`2.571 ms`** |
| **Host Wall `p50` (ms)** | `4.152 ms` | **`3.315 ms`** | **`2.734 ms`** |
| **Equivalent GPU Throughput (`p50`)** | `254.0 FPS` | **`319.5 FPS`** | **`391.5 FPS`** |
| **Serialized GPU Fraction (`30 Hz` / `60 Hz`)** | `11.81%` / `23.62%` | **`9.39%` / `18.78%`** | **`7.66%` / `15.33%`** |
| `dual_preprocess_total_ms` | `0.314 ms` | `0.312 ms` | `0.356 ms` |
| `coarse_total_ms` | `1.283 ms` | **`1.174 ms`** | **`1.167 ms`** |
| `refinement_census_ms` | `0.233 ms` | `0.230 ms` | `0.228 ms` |
| `refinement_guided_match_ms` | `0.860 ms` | **`0.507 ms` (`-0.353 ms`)** | **`0.487 ms` (`-0.373 ms`)** |
| `refinement_lr_ms` | `0.171 ms` | `0.160 ms` | **`0.112 ms` (`-0.059 ms`)** |
| `refinement_metric_ms` | `0.607 ms` | **`0.304 ms` (`-0.303 ms`)** | **`0.154 ms` (`-0.453 ms`)** |
| `refinement_planar_recovery_ms` | `0.370 ms` | `0.369 ms` | `0.000 ms` |
| **Gross / Net Dynamic Energy per Frame** | `180.25 mJ` / `121.47 mJ` | **`131.82 mJ` / `85.09 mJ`** | **`103.68 mJ` / `65.55 mJ`** |

---

## 4. Extended Near-Field Disparity (`D_coarse = 64`, `256 px` Full-Res Range) on `y >= 463`

In Milestone v9.4, the near-field tabletop cohort (`y in [463, 700)`, `x in [220, 1180)`, where true disparity exceeds `127 px`, $Z < 0.496\text{ m}$) was isolated as out-of-range under `D_coarse = 32` (`128 px` full-res limit) rather than extrapolated. Evaluating `--coarse-disparities 64` (`256 px` full-res disparity range at $4\times$ scale) in `v10.2`:

| Near-Field Tabletop Metric (`y in [463, 700)`) | `D_coarse = 32` (`128 px` limit) | `D_coarse = 64` (`256 px` limit) |
| :--- | :---: | :---: |
| **GPU Combined `p50` (ms)** | `3.174 ms` | **`3.138 ms`** |
| **Measured Valid Pixels with `d > 127.5 px`** | `0 px` (saturates at `126.3 px`) | **`23,237 px`** |
| **Completed Valid Pixels with `d > 127.5 px`** | `0 px` | **`25,064 px`** |
| **Maximum Recovered Disparity (px)** | `126.33 px` | **`254.00 px`** |
| **Minimum Recovered Metric Depth (m)** | `0.4990 m` | **`0.2482 m`** |

---

## 5. Portable Slang-to-CPU Backend (`v10.1`) & Mobile Profile (`v10.3`)

- **Single-Source Slang Portability (`python/stereoforge/slang_cpu_backend.py`):**
  - `python/stereoforge/shaders/daxa/refinement.slang` compiles unmodified to Vulkan SPIR-V (`slangc -target spirv`) and to native C++ (`slangc -target cpp -DSF_CPU_TARGET=1`).
  - On CPU (`SF_CPU_TARGET=1`), `planar` bypasses GPU workgroup LDS barriers (`GroupMemoryBarrierWithGroupSync`) and evaluates the exact same $3\times 3$ coarse affine plane and bilateral stencil directly on host memory via OpenMP (`#pragma omp parallel for`).
- **Bit-Exact Parity & CPU Performance:**
  - `census`, `census_validity`, `lr` validity, `metric_validity`, and `provenance`: **`100.000%` bit-exact agreement** with Vulkan GPU across `1,843,200` pixels.
  - **`640×360` Mobile/Low-Power Profile (`v10.3`):** **`8.85 ms` CPU `p50` (`112.95 FPS` equivalent)** with zero GPU usage.
  - **`1280×720` Full Resolution on CPU:** **`44.31 ms` CPU `p50` (`22.57 FPS` equivalent)** with zero GPU usage.

---

## 6. Cooperative GPU Duty-Cycle Budget Governor (`v10.4`)

Implemented in `DutyCycleBudgetGovernor` (`include/stereoforge/depth_rate_controller.hpp`):

| Pipeline Profile | `3%` GPU Budget | `5%` GPU Budget | `10%` GPU Budget | `20%` GPU Budget |
| :--- | :---: | :---: | :---: | :---: |
| **`v9.3` / `v10 quality` (`3.55 ms`)** | `5 FPS` (`1.77%`) | `10 FPS` (`3.55%`) | `20 FPS` (`7.10%`) | `30 FPS` (`10.65%`) |
| **`v10 ultra-fast` Planar (`2.95–3.13 ms`)** | **`10 FPS` (`2.95%`)** | `10 FPS` (`2.95%`) | **`30 FPS` (`8.85%`)** | **`60 FPS` (`17.70%`)** |
| **`v10 ultra-fast` Measured (`2.55–2.62 ms`)** | **`10 FPS` (`2.62%`)** | `10 FPS` (`2.62%`) | **`30 FPS` (`7.86%`)** | **`60 FPS` (`15.72%`)** |

---

## 7. Reproducibility Commands

```bash
# 1. Build C++ targets and SPIR-V shaders
AMENT_PREFIX_PATH=/home/bjoern/git/stereoforge/ros_ws/.pixi/envs/default \
  cmake --build build/ros-daxa-native --target daxa_refinement_contracts daxa_refinement_benchmark daxa_dual_preprocess_benchmark -j 8

# 2. Run CTest refinement contract suite
VK_LAYER_PATH=/home/bjoern/git/stereoforge/ros_ws/.pixi/envs/default/share/vulkan/explicit_layer.d \
  ctest --test-dir build/ros-daxa-native --output-on-failure -R "refinement"

# 3. Run Milestone v10 pytest suites (Slang CPU backend + Ultra-Fast GPU + DutyCycleBudgetGovernor)
VK_LAYER_PATH=/home/bjoern/git/stereoforge/ros_ws/.pixi/envs/default/share/vulkan/explicit_layer.d \
  PYTHONPATH=.:python:benchmarks /home/bjoern/git/stereoforge/ros_ws/.pixi/envs/default/bin/pytest \
  tests/test_slang_cpu_backend_v10.py tests/test_ultra_fast_gpu_v10.py -o addopts="" -v
```
