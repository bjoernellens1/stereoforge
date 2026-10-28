# Fast60 multiresolution measurements — 2026-10-28

Independent two-eye SGM at 640×360/D64 reduced measured native GPU queue time to **5.88–5.99 ms**, compared with **31.42–31.79 ms** at 1280×720/D128. This supports proceeding to full-resolution selective refinement. It does not establish 720p depth quality or sustained live 60 Hz.

## Matched measurement boundary

Hardware: AMD Radeon 8060S, RADV STRIX_HALO, Mesa 26.2.3. Native Daxa commit `00761e80c1dd6d1972d5f4569ce49620a7b4d384`, offline Slang 2026.12 SPIR-V. Three interleaved repetitions used orders full/medium/half, medium/half/full, half/full/medium. Each run had 20 warm-up and 100 measured iterations of the same original synthetic capture.

All profiles used independent left and right SGM, four paths per reference, sequential scheduling, wave32, two slots, P1=8/P2=32, subpixel enabled and LR tolerance=1 pixel. RGB synthesis and Temporal were disabled. Original full-resolution capture, camera calibration and both original RGB images remained byte-identical; rectification maps were recomputed directly into each target grid using the pixel-centre-scaled projection. Original peripheral RGB was retained.

The foreign renderer had finished before dispatch. Process monitoring at dispatch and during execution observed desktop clients and the owned native benchmark. This is not a claim of a globally idle GPU.

## Results

Each cell is the minimum–maximum of the **three per-run statistics**, in milliseconds; percentiles are nearest rank (`ceil(p*N)-1`), recomputed from retained raw samples. These are not pooled percentiles.

| Rectified profile | GPU mean | GPU p99 | GPU maximum | Host mean | Host p99 | Host maximum |
|---|---:|---:|---:|---:|---:|---:|
| 1280×720/D128 | 31.416–31.792 | 31.796–32.529 | 31.865–33.183 | 31.810–32.180 | 32.269–33.029 | 32.300–33.630 |
| 960×540/D96 | 15.860–15.984 | 16.298–16.624 | 16.437–18.341 | 16.164–16.334 | 16.794–17.174 | 16.841–18.886 |
| 640×360/D64 | 5.877–5.988 | 6.209–6.287 | 6.251–6.290 | 6.084–6.212 | 6.473–6.503 | 6.480–6.616 |

GPU queue span includes upload, decode, rectification, both stereo references, LR and metric metadata. Host wall includes enqueue, completion wait and timestamp retrieval. Initialization and diagnostic full readback/reference comparisons occurred outside timing. Per-stage samples, upload CPU/GPU time and waits remain in the native reports. Timing runs did not enable validation.

| Profile | Candidate hypotheses `2*W*H*D` | Requested buffer bytes | Synthetic-calibration nominal near limit |
|---|---:|---:|---:|
| 1280×720/D128 | 235,929,600 | 840,500,528 | 1.814 m |
| 960×540/D96 | 99,532,800 | 419,329,328 | 1.819 m |
| 640×360/D64 | 29,491,200 | 173,262,128 | 1.829 m |

Requested bytes exclude allocator/driver overhead, queries, diagnostics and CPU source storage; they are not peak device/process memory. Half resolution reduces candidate hypotheses by 8×, while the observed complete GPU queue span improves approximately **5.3×**. Candidate counts are not a measured aggregation instruction count.

The near limits use this **synthetic fixture's** calibration and `fx*b/((D-1)-(cxL-cxR))`. They are neither measured ZED limits nor calculations using a representative paper focal length. The slight profile differences follow the integer maximum disparity.

## Correctness and remaining gates

All nine runs passed post-timing checks for both references' scalar integer outputs, packed penalties, OpenCV preprocessing planes, LR masks and metadata. Input hashes and runtime alias targets remained unchanged. A separate awkward-size **126×19/D96** qualification passed native core and synchronization validation, plus a separately executed frozen-RHI numerical comparison without validation. This does **not** qualify frozen-RHI validation.

These checks establish implementation agreement on the bound fixture. There is no independent metric truth, held-out evaluation or evidence yet for thin-object recovery, near-field completeness or false accepted correspondences. Quality remains explicitly unknown. Separate Stereo compound-float and Temporal general-float gates remain open; their thresholds were not waived by this matrix.

The 360p result leaves room to investigate full-resolution refinement and fixed-720p View cost. Neither is included here. The 540p p99 and maximum already exceed the desired ≤15 ms processing gate. No throughput is inferred from these static timings: unique live frames, camera-to-consumer age, drops, queue growth, sharing pressure and the ≥10-minute live60 gate remain unmeasured.

## Bound evidence and reproduction

The ignored local artifact directory is `artifacts/fast60-f0-20261028/`; it is not a downloadable repository dataset. `root-f1-matrix/comparison.json` indexes all nine native reports and their hashes. Its SHA-256 is `fb9f711f91c0545aeff79068ab48bd5dc199331eca4c18c77b80ed45dd7f778c`.

The F1 dispatch freeze is `f1-dispatch-freeze-candidate-v1.json`, SHA-256 `43662090b606bbc3f97067ab338d56cde4c523c72445d01ba9479181274d4d8d`. Its 683 bound file hashes and all runtime alias targets were checked unchanged after measurement. Native binary SHA-256: `4cbdd56548301517d6fe3332be5ebe4fb7e3ca77cc18dc3aaf20832ee6e3a0ac`; compiler SHA-256: `82e29ecd96714c64a31021160eaff4bf8604f44934e9f7b98448e962a5d12dad`. The freeze binds source, inputs, dependencies, compiler and SPIR-V identities; the D96 gate is indexed separately by `root-d96-summary.json` and `d96-qualification-freeze.json`.

Portable offline export syntax (fresh output directories, your own bound source manifest):

```bash
PYTHONPATH=python .venv/bin/python benchmarks/multiresolution_fixture.py \
  --source-manifest /path/to/source/manifest.json \
  --output /path/to/profiles/profile-640x360 --width 640 --height 360
```

Repeat for 1280×720 and 960×540. Create a profiles JSON array with `name`, relative `manifest` and `disparities` entries for `full`/128, `medium`/96 and `half`/64. Prepare and independently review a freeze binding those exact generated inputs and the exact native/compiler binaries before GPU dispatch. A newly exported dataset cannot reuse the historical freeze.

On the original local artifact set, the comparison command is:

```bash
PYTHONPATH=python .venv/bin/python benchmarks/compare_multiresolution.py compare \
  --profiles artifacts/fast60-f0-20261028/profiles.json \
  --native build/daxa/daxa_depth_benchmark \
  --freeze artifacts/fast60-f0-20261028/f1-dispatch-freeze-candidate-v1.json \
  --freeze-sha256 43662090b606bbc3f97067ab338d56cde4c523c72445d01ba9479181274d4d8d \
  --output artifacts/fast60-rerun --frames 100 --warmup 20 --repeats 3
```

Run GPU comparisons only in a coordinated hardware window. Hash mismatches require a new reviewed freeze, not bypassing identity checks. See the [Fast60 plan](superpowers/plans/2026-10-28-fast60.md) for the refinement and live qualification sequence.
