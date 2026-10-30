# Experimental CSR work-proportional SGM

This separate research lane tackles the largest algorithmic cost: dense
D128 four-path SGM still performs work for disparities that are already known
to be irrelevant. CSR records make valid per-pixel candidate intervals explicit,
and the recurrence accesses only admitted hypotheses.

## Layout and algorithms

\`benchmarks/csr_sgm.py\` provides:

1. **\`candidate_plan\`**: union of coarse disparity windows with *independent*
   full-resolution sparse anchor windows. Where neither exists, it deliberately
   falls back to full disparity search. A contradictory anchor is retained as an
   alternative instead of silently overwritten.
2. **\`pack_csr\`**: per-pixel strictly increasing disparities, offsets of length
   \`W*H+1\`, and packed integer costs. Invalid/out-of-range costs cannot become
   valid hypotheses.
3. **\`csr_sgm\`**: four actual work-proportional directional recurrences using
   only the packed candidates and globally aligned disparity indices. Empty
   predecessor states reset the recurrence. Exact integer reference penalties
   are preserved.
4. **\`normalized_work\`**: candidate state-update ratio relative to dense
   full-D work, **not** a GPU runtime or memory-traffic measurement.

\`python/stereoforge/shaders/daxa/csr_sgm.slang\` implements the matching GPU
algorithm experimentally. One 64-thread workgroup follows each directional
scanline, visiting only active CSR slots. Sorted predecessor hypotheses are
found by binary search, predecessor minima are reduced within the workgroup,
and each direction writes its independent sparse path state. A separate kernel
sums all four path buffers.

This is a **correctness-oriented GPU prototype**, not a measured optimized
kernel. Long scanlines still require per-pixel synchronization; inactive scanline
workgroup lanes and binary-search overhead can dominate GPU time. Do not
claim an FPS improvement from the candidate-update count.

## Verification

\`\`\`sh
uv run pytest -q tests/test_csr_sgm.py tests/test_csr_sgm_shader.py
\`\`\`

The 12 randomized tests check exact parity against the existing independent
\`compact_sgm_reference.dense_masked_sgm\` (including empty candidate sets and
fully dense local hypotheses). A thin foreground fixture exercises independent
high-resolution anchor recovery; missing priors fall back to full search.
This confirms an **algorithmic possibility**, not real-world thin-object
precision/recall.

The GPU shader can be compiled separately from default production kernels:

\`\`\`sh
cmake --preset daxa -DSTEREOFORGE_DAXA_CSR_SGM=ON
cmake --build --preset daxa --target daxa_csr_sgm_shaders
\`\`\`

Shader compilation and actual CPU↔GPU parity have **not** yet been verified on
the local AMD device. The shader is intentionally not wired into the production
depth graph. GPU dispatch must verify the CSR capacity, offset monotonicity,
sorted disparities and p1/p2 bounds at a trusted setup boundary and include
the required barriers between the four path passes and final sum.

## Scientific quality and speed gates

Before promoting this to the native pipeline:

- Compare fixed-input full-D, coarse-refined, and multi-hypothesis CSR output
  against independent truth: completeness, false accepted matches, near range,
  occlusions, individual thin surfaces and textureless geometry.
- Require a practical GPU candidate producer that is cheaper than avoided dense
  work. Sparse ORB/Census anchors alone have previously failed thin-surface
  gates; no blanket quality claim follows from adding them.
- Measure per-stage GPU p50/p99/max and memory traffic on AMD, including CSR
  construction, barriers, LR consistency and final metric output.
- Optimize wave-level aggregation and short dispatches only after exact GPU
  parity and acceptance thresholds pass.
- Preserve the standard full-SGM estimator as a controlled accuracy fallback;
  never count interpolated depth as independently measured.
