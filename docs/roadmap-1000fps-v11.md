# StereoForge v11: a qualified 1,000 FPS research target

**Status: NO-GO on v10 evidence.** The initial target is **1280×720 stereo processing on a GPU-resident replay**, not 1,000 fresh ZED 2 captures per second or 1,000 ROS messages per second.

## Current evidence and limitations

Reference: benchmarks/results/v10_ultra_fast_and_universal_qualification.json, committed at c246a83. Report provenance records f3e7a59 (the preceding commit).

| v10 profile | GPU p50 | GPU p95 | Host p50 | Valid left-eye pixels |
| --- | ---: | ---: | ---: | ---: |
| Ultra-fast measured | 2.554 ms | 2.571 ms | 2.734 ms | 294,942 (32.0%) |
| Ultra-fast with planar completion | 3.130 ms | 3.149 ms | 3.315 ms | 334,828 completed (36.3%) |

These are repeated-frame, single-GPU-queue timings (180 timed frames, 20 warmups) from one retained input. GPU FPS equivalents from reciprocal p50 **are not demonstrated sustained frame delivery**. Benchmarks exclude camera acquisition, GPU-to-RGB-D view synthesis, and ROS transport. Each stage timing in the v10 profile is a **last-frame snapshot**, not a per-stage p95. The quality comparison uses layered-per-path coarse scheduling; ultra-fast instead uses shared Census plus derived right reference. Its overall speedup is therefore not an isolated, same-algorithm kernel acceleration.

The v10 near-field D_coarse=64 experiment reports 25,064 high-disparity completed pixels but only ONE overlapping high-disparity pixel against the independent ZNCC reference. That single sample has 107.901 px disparity error. This is **unqualified**, not evidence all added pixels are wrong. Acceptance based on pixel count alone is invalid.

The v10 competing-GPU-workload subreport is empty because of incorrect JSON field names. This branch fixes the generator's lookups; it does not silently rewrite historical evidence.

## Target contracts

**T0, microbench:** isolated kernel >1,000 FPS, explicitly labeled kernel-only.

**T1, measured stereo pipeline:** 1280×720 per eye; YUYV upload, rectification, stereo matching, independent verification, metric output and provenance; GPU single-interval p95 ≤1.000 ms, host p95 ≤1.000 ms, p99 reported. In addition, prove ≥1,000 **completed** frames/second over ≥1,000 processed frames using ≥128 distinct recorded stereo pairs with unique frame identities/timestamps. Measure both single-slot latency and 2+-slot pipeline throughput, and distinguish queue execution from elapsed wall time.

**T2, completed depth:** T1 plus planar completion, preserving measured vs inferred source tags.

**T3, contention:** T1/T2 with a real concurrent workload. Report its slowdown, instantaneous clocks, energy/temperature, GPU duty and frame drops. No false “success” from cache reuse or skipped work.

**T4, live:** separately qualify 60 Hz HD720 or 100 Hz VGA on the ZED 2, and ROS 2 fresh-frame age, delivery, losses, and GPU-sharing. The physical camera cannot acquire 1,000 distinct HD720 frames/s.

## Engineering stage allocations (ms, sum 1.000)

| Stage | v10 snapshot | v11 allocation | Speedup required |
| --- | ---: | ---: | ---: |
| Dual preprocessing | 0.356 | 0.180 | 1.98× |
| Coarse correspondence/SGM | 1.167 | 0.250 | 4.67× |
| Full-res Census | 0.228 | 0.100 | 2.28× |
| Guided match | 0.487 | 0.230 | 2.12× |
| LR verification | 0.112 | 0.030 | 3.73× |
| Metric/metadata | 0.154 | 0.050 | 3.08× |
| Planar completion | 0.369 in planar profile | 0.130 | 2.84× |
| Queue/sync overhead reserve | — | 0.030 | — |

These are proposed engineering targets, not performance predictions or empirical roofline bounds. Coarse SGM alone currently exceeds the complete 1 ms budget. Dispatch fusion is not enough; an algorithmic replacement is necessary.

## Experimental sequence

1. **v11.0: qualification.** Matched ABBA comparisons, dynamic replays, ≥128 input pairs, ≥1,000 completed frames, warm clocks, GPU p50/p95/p99, host throughput, per-stage distributions, correct/valid/completed coverage, energy, contention. Compare independent SGM with derived coarse, and quality with ultra-fast, changing ONE parameter family per ablation.
2. **v11.1: coarse replacement (first priority).** Explore 1/8-resolution Census plus local cost filtering, or sparse/work-proportional hypotheses, with a selective 1/4 fallback for ambiguous/edge/thin regions. A 1/8 grid with 32 disparity bins spans up to 256 full-resolution px, but wider range must not be confused with correct geometry. Freeze the fallback threshold on training/development data, never held-out truth. Target coarse ≤0.25 ms.
3. **v11.2: full-resolution matching.** Test tiled rectified-luma/Census fusion, optional diagnostic buffers, fixed 3/5/9 guided candidates, and subgroup cooperation. Keep independently verifiable right-eye evidence; do not silently exchange quality for latency.
4. **v11.3: LR + metric fusion.** Reduce intermediate memory writes and dispatches, target ≤0.08 ms together. Exact validity, provenance and source identity are mandatory; float differences need explicit tolerances.
5. **v11.4: completion.** Tile-level gate before planar inference, skip tiles without useful holes; measure occlusion/low-texture/near-field separately. Preserve STEREO, PLANAR_INFERRED, and temporal provenance separation.
6. **v11.5: other GPUs.** Qualify independently on Intel and NVIDIA Vulkan after AMD; retain Slang/Daxa/Vulkan, without CUDA or ROCm dependencies.

## Frozen quality gates before optimization

Use the v9.4 five-scene synthetic suite, slanted tabletop and narrow-gap counterexamples, ZED 2 desk multi-frame, AprilTag/ChArUco, independent ZNCC, and available RealSense cross-check. Report common-valid MAE *and* unconditional correct completeness, bad-1px/bad-2px, false fills, thin objects, occlusion, near-field, per-provenance confidence, and dropped support. Do not hide risky small cohorts in global means.

Initial proposed development gate: ≤0.5 percentage-point decrease from v9.4 completed unconditional good-2px of 98.81%; no new false obstacles/occlusion regressions; ≤0.25 percentage-point increase in bad-2px on each cohort. These are *proposed*, not demonstrated passes. Freeze on development scenes before held-out evaluation.

For D_coarse=64, require substantive high-disparity reference overlap (provisionally ≥100 independent pixels), plus validated false-positive rate, independent coverage, precision, and geometric checks. Count alone never qualifies new matches.

## Reproducible, read-only audit and v11 qualification suite

    PYTHONPATH=. python benchmarks/fps1000_gate_v11.py \
      --input benchmarks/results/v10_ultra_fast_and_universal_qualification.json \
      --output /tmp/stereoforge-v11-fps1000-audit.json

    PYTHONPATH=python:benchmarks:. python benchmarks/qualify_v11_1000fps_and_coarse_replacement.py

    PYTHONPATH=python:benchmarks:. pytest -q tests/test_fps1000_gate_v11.py

`fps1000_gate_v11.py` is a **CPU-only evidence audit**; it rejects 1,000 FPS claims (`global_verdict: "NO_GO"`) until `< 1.000 ms` GPU `p95` across `>= 1,000` distinct stereo frames is demonstrated.

## Implemented v11.0, v11.1, and v11.3 Results

Reference: `benchmarks/results/v11_0_qualification_and_coarse_replacement.json` and `benchmarks/results/v10_ultra_fast_and_universal_qualification.json`.

### 1. v11.0 Qualification Resolution (`A0..A7` One-Parameter-Family Ablation + Near-Field `D_coarse=64`)

- **Bit-exact write-skip verification (`A1` vs `A2`):** `exact-ultra` (skipping the 14 diagnostic planes while preserving 5-tap right-eye matching, `r=4`, and `independent-sgm`) is **100% bit-exact** on left-eye measured and completed disparity/validity against `A1`.
- **8-step ablation (`A0..A7`) across the 5-scene 720p synthetic GT suite, `narrow_gap_bars` counterexample, and ZED 2 capture:**
  - `A0` (`quality`, `layered-per-path` SGM, `r=4`): `98.90%` unconditional `good_2px`, `0.24%` `bad_2px`, `3.852 ms` GPU `p50` (`1.236 ms` coarse stage).
  - `A5` (`v10 ultra-fast`): `98.97%` unconditional `good_2px`, `0.06%` `bad_2px`, `3.102 ms` GPU `p50` (`1.148 ms` coarse stage).
  - `A6` (`v11 hierarchical-1to8-fallback` with exact 5-tap right-eye matching): `99.07%` unconditional `good_2px`, `0.15%` `bad_2px`, `2.238 ms` GPU `p50`.
  - `A7` (`v11-fast`, `hierarchical-1to8-fallback`): **`99.12%` unconditional `good_2px` (`+0.22 pp` vs `A0`), `0.13%` `bad_2px` (`-0.11 pp` vs `A0`), `1` false-bridged narrow-gap pixel (vs `3` on `A0`)**.
- **Near-field `D_coarse=64` geometric qualification (`assess_near_field`):** Evaluated against an independent 256-px epipolar ZNCC reference (`11×11` window, forward-backward LR `<= 1.0 px`), yielding **`143` high-disparity (`d > 127.5 px`) overlap pixels with `3.50%` `bad_2px_pct` (`PASS_PRELIMINARY`)**.

### 2. v11.1 Coarse SGM Replacement (`hierarchical-1to8-fallback`, `<= 0.250 ms` Gate Passed)

Implemented in `python/stereoforge/shaders/daxa/stereo_hierarchical.slang` (`coarse_prep_1to8_1to4`, `coarse_match_1to8`, `coarse_lr_risk_1to8`, `coarse_selective_fallback_1to4`) and wired into `DaxaStereoPipeline`, `DaxaDepthPipeline`, and `stereoforge_ros2_depth_publisher --profile v11-fast`:

| Metric (1280×720, AMD Radeon 8060S) | v9.4 `A0` Independent SGM | v10 `A5` Shared-Census SGM | v11 `A7` Hierarchical `1/8`+`1/4` | v11 Budget |
| --- | ---: | ---: | ---: | ---: |
| **Coarse stage total GPU `p50`** | `1.2362 ms` | `1.1475 ms` | **`0.2035 ms` (`6.07×` / `5.64×` speedup)** | `<= 0.2500 ms` (**PASS**) |
| **Coarse stage total GPU `p95`** | `1.2429 ms` | `1.1539 ms` | **`0.2240 ms`** | `<= 0.2500 ms` (**PASS**) |
| **Full pipeline + planar GPU `p50` (`p95`)** | `3.8523 ms` (`3.880 ms`) | `3.1016 ms` (`3.116 ms`) | **`2.1524 ms` (`2.186 ms`, `464.6 FPS` equiv)** | `1.0000 ms` (T2 target) |
| **Full pipeline measured-only GPU `p50` (`p95`)** | `3.5067 ms` (`3.524 ms`) | `2.5734 ms` (`2.589 ms`) | **`1.6444 ms` (`1.687 ms`, `608.1 FPS` equiv)** | `1.0000 ms` (T1 target) |

### 3. v11.3 Complete Camera-to-Depth Slang-to-C++ CPU Pipeline

Extended `python/stereoforge/slang_cpu_backend.py` (`SlangCpuRefinementBackend.run_full_camera_to_depth_pipeline`) to compile and execute all **14 production shader entrypoints** across `preprocess.slang`, `stereo_hierarchical.slang`, `depth.slang`, and `refinement.slang` via `slangc -target cpp -DSF_CPU_TARGET=1` and OpenMP:
- **`640×360` (`mobile-balanced`) full YUYV-to-completed-depth CPU latency:** `30.35 ms` `p50` (`32.95 FPS`).
- **`1280×720` (`full`) full YUYV-to-completed-depth CPU latency:** `118.85 ms` `p50` (`8.41 FPS`).
