# Local performance evidence

Measured 2026-10-27 on **AMD Radeon 8060S Graphics (RADV STRIX_HALO)** using the explicit Vulkan backend. This is a synthetic research-adapter microbenchmark, not camera throughput, metric-depth qualification, a quality comparison, or production latency qualification.

Reproduce with:

```sh
PYTHONPATH=python .venv/bin/python benchmarks/gpu_hotpaths.py
```

The script uses deterministic 320×180 RGB inputs with a four-pixel stereo shift, 32 disparity candidates, four SGM paths, two warm-up frames and three measured frames per case. Every measured pipeline call performs both stereo directions and left/right consistency checking. Temporal/adaptive processing receives explicit identity motion; no motion is inferred. The strict wide view uses the default centered virtual camera and GPU geometric renderer; exposure matching/blending and preview are absent from that renderer.

| Case | Local wall samples (ms) | Wall p50 (ms) | Last stereo pass GPU stage-sum p50 (ms) |
| --- | --- | --- | --- |
| Independent stereo | 12.392096001349273, 9.979062999263988, 8.56984800157079 | 9.979062999263988 | 0.9891594236250967 |
| Temporal + adaptive stereo | 22.676833999867085, 27.158834998772363, 23.19437900041521 | 23.19437900041521 | 0.9931670374498935 |
| Independent stereo + strict GPU wide RGB-D | 49.8041690007085, 49.85979400044016, 45.21903500062763 | 49.8041690007085 | 1.0439435154694365 |

Wall time includes rectification, host processing, transfers, waits, readbacks and Python adapter work. The GPU timestamp column sums dispatch durations **only for the last (right-reference) stereo pass** because the current pipeline exposes only that pass's timestamps. It excludes the left pass, transfer work, temporal kernels, renderer kernels and CPU work; it cannot be interpreted as total GPU pipeline time. Device/pipeline construction and initial shader compilation are outside the measured calls. Three samples characterize only this short run.

Each independent frame evaluates 1,843,200 left and 1,843,200 right disparity candidates. Adaptive left evaluations were 608,635, 608,631 and 608,635, while right evaluations remain 1,843,200. Both configurations still aggregate the full 14,745,600 directional candidates per frame. Their reported raw stereo valid fraction was 0.9533333333333334. This synthetic result does not demonstrate an application speedup or justify changing quality thresholds.

For the wide case, the measured view section (including CPU depth-to-raw projection) took 42.68298900024092, 40.82435000054829 and 36.12177300055919 ms. Renderer adapter total including readback took 28.162553999209194, 27.220260999456514 and 22.367582001606934 ms. Its `upload_dispatch_wait_ms` field also includes CPU source-cloud/right-visibility preparation; it is not a pure upload or GPU duration.

## Data-path review and remaining limitations

- Stereo buffers persist at a fixed shape, and shape changes alone trigger an idle wait. The single pending submission is enforced. The pipeline immediately consumes each direction's result, performing two waits and six output-buffer readbacks per frame, then performs consistency and metric conversion on the CPU.
- Cost and sum volumes remain full `H×W×D` allocations even with adaptive bounds. Shape changes rebuild these buffers. Narrowed candidate counts do not imply proportional aggregation or memory reductions.
- GPU temporal reprojection and fusion share the stereo device but remain synchronous research adapters. Each uploads host planes and reads output planes back; owner/selection readback drives CPU gathering of 64-bit identities and ages. Bounded history snapshots add host copies.
- The wide renderer uses a separate Vulkan device and host uploads. It performs CPU raw-depth projection, source-cloud/right visibility work, packing and concatenation before dispatch, then reads six output buffers. It copies right source rays each call to retain projected geometry. These operations remain production optimization candidates.
- Undistorted source rays are cached by calibration with a capacity of two; repeated left/right calls preserve both entries. Stereo, temporal and renderer resources are reused while their dimensions remain unchanged. Tests guard against Python pixel loops, enqueue readback, unconditional idle waits and unbounded ray-cache growth.
- Grayscale conversion creates uint32 host images even for CPU SGBM; that backend then casts back to uint8. Per-frame consistency indices and intermediate masks are allocated on the host. These are review findings, not changes made by this benchmark pass.

Production work should preserve the correctness gates while measuring complete transfer-inclusive paths, retaining shared-device resources and metadata on the GPU, and exposing timestamps for both stereo passes and temporal/view kernels. No threshold or architecture adoption decision follows from this tiny fixture.
