# StereoForge v11.2: next steps from 608 to 1,000 FPS

**Research status:** v11.1 hierarchical coarse stage qualifies its 0.250 ms GPU stage budget (0.2035 ms p50 / 0.2240 ms p95), but neither the 1.0 ms full pipeline nor real-camera quality promotion is qualified. This is a plan, not a performance prediction.

## Stage accounting at 1280x720 on Radeon 8060S (A7 measured-only)

| Stage | v11 p50 (ms) | Proposed p50 budget (ms) | Experiment |
| --- | ---: | ---: | --- |
| Full-frame preprocess | 0.353 | 0.200 | Direct YUYV to rectified luma/support, avoid unconsumed RGB planes |
| Hierarchical coarse | 0.204 | 0.200 | Preserve and qualify, no speculative rewrite |
| Full-resolution Census | 0.232 | 0.110 | Cooperative tiled descriptor computation and read reuse |
| Guided two-eye match | 0.504 | 0.220 | Risk-adaptive 1/3/5 taps and/or subgroup matching, independent right evidence |
| LR + metric | 0.260 combined | 0.100 | Fuse without changing supported-pixel and provenance semantics |
| Queue/graph interval gaps | about 0.091 | 0.050 | Remove unnecessary transitions and evaluate queue graph |
| **Total** | **1.644** | **0.880** | **0.120 ms margin to 1 ms p95, not guaranteed** |

Stage numbers are approximate p50 values from stage-wise distributions, not a sum of simultaneous p50 samples. The queue/graph gap is estimated by subtracting stage p50s from single-interval GPU p50. Variability, resource dependencies, and p95 must be measured directly.

**Full T1 target:** measured stereo, YUYV upload through calibrated metric output, 1280x720 per eye, single-interval p95 < 1.000 ms AND demonstrated >=1,000 completed unique-timestamp stereo pairs/s over >=1,000 pairs and >=128 distinct frames. The former does not imply the latter. **T2** adds completed depth; current A7 planar path is 2.152 ms p50 / 2.280 ms p95 and must be qualified separately.

## Priority 0: repair comparability and quality screening (before replacing more kernels)

- A7 synthetic 5-scene good-2px exceeds A0 by 0.22 percentage points, but on one recorded ZED2 frame its measured coverage falls by about 2.71 percentage points (of full 720p image) and completed coverage by 3.19 points. The ZNCC bad-2px conditional rate grows from 3.23% to 5.22%. Its newly admitted pixels show 20.42% bad-2px on only 3,261 ZNCC-overlap pixels. Treat these as development signals, not physical truth.
- Record and evaluate >=8 ZED2 frames at varying textures/distances/light, and ideally multiple scenes, plus physically measured targets and false-obstacle counterexamples. Run actual interleaved/randomized ABBA GPU comparisons, not 180 repeated frames per configuration in fixed order. Freeze the held-out dataset before tuning any quality parameters.
- The currently reported independent desk plane is fitted to A0 disparity selected by ZNCC; label it A0-anchored. The near-field D64 evaluator is imported from v10 and has not qualified A7.
- Candidate feature mismatch: stereo_hierarchical.slang reads the ortho-hybrid flag, yet hardcodes ordinary Census at 1/4 and 1/8. Run descriptor-only H0/H1/H2/H3 comparisons (Census/Census, Ortho/Census, Census/Ortho, Ortho/Ortho), changing no other configuration. The feature switch must be visible in metadata. Compare stage latency and unconditional-good2 plus real supported cohorts.
- The CPU-derived 14 shader entrypoints do not by themselves prove bit exact parity for the entire 1080p or 720p GPU pipeline. Verify comparable masks and bounded floating-point outputs against the frozen Vulkan result.

## Priority 1: measured-only fast path (first hardware experiments)

**E1: Fused LR + metric (one dispatch)**

- Compute each eye's LR consistency from independently retained raw left/right local matches; immediately produce calibrated axial Z, confidence, validity, provenance, source ID, and completed measured outputs where needed. The full matcher must complete both references first; no synchronization shortcut is allowed.
- Replace separate ~0.109 ms LR and ~0.151 ms metric with a target combined <=0.100 ms. Preserve rejection as canonical NaNs and exact source/provenance/validity, including values used by planar completion.
- Benchmark isolated E1 off/on under identical parameters. CPU oracle: exact masks, source IDs, provenance, and bounded float errors. Inspect SPIR-V register counts and occupancy; fusion can increase register pressure.

**E2: Work-proportional full-resolution guided matching**

- Introduce a tile-level risk route computed without ground truth from 1/8/1/4 confidence, left-right disagreement, gradient/occlusion boundaries and support. Easy pixels use a smaller comparison stencil; hard pixels preserve 5-tap symmetric evidence. Consider 3/5/9 candidate counts and independent LR confirmations.
- Current match p50 is ~0.504 ms. Target <=0.220 ms after E2 without loss of frozen thin-structure correctness. Do not skip the independent right-eye verification merely to meet budget.
- Measure actual candidate evaluations, subgroup divergence, memory transactions, and conditional reference error *separately*. A sparse candidate selector that misses all thin-structure truth candidates is NO-GO regardless of speed.

**E3: Full-resolution Census + preprocess memory traffic**

- Evaluate 2x2 and 16x16/32x8 tile strategies: reuse quantized rectification values and neighborhood luma reads; avoid materializing RGB data when the consumer requests depth-only, but keep existing RGB products unchanged for depth-rgb and wide-view consumers.
- Test U8/U16 luma descriptors, packed support and buffered/prepacked reciprocal matching with one changed factor at a time. Verify no extra rectification blur or left/right remap asymmetry.
- Combined preprocess+Census currently ~0.585 ms in the measured-only A7 run; proposed combined target <=0.310 ms. Watch register spilling and LDS occupancy, particularly on wave32 vs wave64. Compare compilers/binary hashes and GPU counters.

**E4: Graph/synchronization minimization**

- Reuse the existing retained-buffer Daxa context. Report stage timestamps and one exact queue timestamp interval per processed stereo pair. Identify the ~0.09 ms interval-minus-stage-sum difference, but do not assume a guaranteed savings; queue barriers may enforce required dependencies.
- Benchmark zero-copy or transfer-free input paths only if equal input ownership, image geometry, and host-to-GPU boundary are preserved. Keep camera, ROS host export, and wide RGB view excluded explicitly when claiming GPU-only processing.

## Priority 2: completed planar fast path

- A7 measured-only p50 is 1.644 ms; A7 with planar completion 2.152 ms: +0.508 ms, of which planar compute is ~0.321 ms and additional completed-buffer writes/initialization consume the remainder. Do not optimize planar alone while ignoring its completed-output materialization.
- Try tile early-out plus one 1/8/1/4 coarse affine plane per stable tile and sparse measured anchor confirmation; then one inexpensive per-pixel completion pass. Preserve explicit PLANAR_INFERRED tags and full-res reciprocal occlusion/ray-crossing guards.
- Target actual T2 p95 <=1.000 ms only after T1 p95 is achieved; avoid using wrong-but-dense completion to hit an FPS target.

## Required trial procedure

1. Pin a commit, compiler build, SPIR-V digests, camera input hashes, test scene hashes, temperature and clock range. Warm GPU clocks, report idle and active telemetry.
2. Profile each E1/E2/E3/E4 in ABBA or randomized blocks (>=4 repeated blocks, >=100 samples/block) on distinct captured input pairs, with a separate 1,000+ frame sustained throughput run.
3. Record quality on fixed-support synthetic truths, A0 retained/new/lost real cohorts, independent ZNCC plus physical targets, thin bars, slanted planes, high-disparity near range, and 720p/1080p scene variations. Use exact per-mode camera calibration.
4. Only merge optimizations that pass explicit one-factor quality and resource-lifetime gates. Record negative results and do not loosen limits after seeing sealed data.

### Run the non-mutating quality screen

    PYTHONPATH=. python benchmarks/audit_v11_real_quality.py \
      --input benchmarks/results/v11_0_qualification_and_coarse_replacement.json \
      --output /tmp/stereoforge-v11.2-real-quality-screen.json

    PYTHONPATH=. pytest -q tests/test_audit_v11_real_quality.py

**Scope:** this document and script perform no GPU or hardware benchmark themselves.
