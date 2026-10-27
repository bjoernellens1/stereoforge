# Shared Vulkan RGB fusion — 2026-10-27

The Python adapter and C++ `NativeGeometry::project` now dispatch the same Slang
compositor. It keeps separate left/right nearest-visible layers, orders
incompatible surfaces by depth, selects compatible seams by confidence, and
blends only compatible surfaces with similar confidence. Color blending retains
the winning depth and provenance; it does not create an independent depth
observation. Unsupported strict pixels remain holes.

Preview fills only unsupported RGB using calibrated infinity directions. Its
pixels have RGB validity and source attribution, but retain invalid metric depth,
zero depth confidence/provenance and an explicit approximation flag. Translation
is intentionally ignored for these approximate directions. Strict metric
projection retains translation and disocclusion holes.

Vulkan stereo defaults to Vulkan fusion and reuses its device. Explicit CPU
stereo defaults to the CPU oracle. Ordinary stereo outputs create no view
renderer, keeping RGB synthesis outside the primary stereo ablation.

## Numerical and resource contracts

- Source geometry is unrectified RGB-aligned axial Z in metres, with undistorted
  unit-Z rays. Disparity remains in rectified stereo geometry. Perspective output
  depth is axial Z; cylindrical output is ray range.
- Projection matrices, rays and depths use float32 on the GPU. Positive float
  bits support deterministic integer depth minima; flat source index breaks
  exact-depth ties. The CPU oracle uses float64 geometric intermediates.
- Two layers are compatible when their depth difference is at most
  `0.01 + 0.01 * min(depths)` metres. Nearest depth wins otherwise. On compatible
  surfaces, higher confidence selects the metric owner; color blends when
  confidence difference is at most 0.1. These are matching CPU-oracle engineering
  thresholds, not experimentally calibrated uncertainty.
- Exposure matching excludes dark/saturated channel values outside 9–246.
  Three fixed 4096-bin logarithmic histograms represent ratios from `9/246` to
  `246/9`. Both central order statistics are reconstructed and averaged before
  the median gain is clamped to `[0.5, 2]`. Quantization can change an output
  channel by one intensity value relative to the CPU median. An adversarial
  even-median regression caught and corrected the earlier clamp-before-median
  error; the color tolerance was not increased.
- RGB packs R/G/B into low bytes of uint32. Masks, owners and provenance use
  uint32 resources. Layer z/owner arrays each contain twice the target pixel
  count. Histograms occupy 49,152 bytes plus 12 gain bytes.
- Preview lookup maps contain two target-sized uint32 layers of concatenated
  source indices, with `0xffffffff` marking unsupported directions. The Python
  calibration-only OpenCV map cache holds at most two configurations; translation
  does not invalidate an infinity map. Native callers supply and own these maps
  until upload; size and per-eye bounds are validated before dispatch.
- Python submits all stages once, waits once and reads one packed uint4 per
  target pixel. The 16-byte packed ABI holds RGB, depth bits, confidence bits,
  and flags: metric/RGB/approximation at bits 0/1/2, source at bits 8–9,
  provenance at bits 16–23. Individual GPU output resources remain available
  in native code; packing does not replace their contracts.
- Native output views retain their device, completion token and slot lease.
  Two slots are persistent and bounded. Preview owners identify RGB samples;
  metric validity must gate identity/age gathering. Native readback applies
  that gate. Python virtual-view identity/age planes remain an open contract
  item in requirements-audit.md.

CPU rectification, raw-depth remapping, derived-right visibility, uploads and
host arrays remain in the research adapter. Cached preview calibration costs
approximately 442 ms on its first 720p configuration; changing orientation may
require another map. This is not a GPU-resident production pipeline.

## Measured incremental adapter cost

AMD Radeon 8060S, RADV STRIX_HALO, Vulkan. Two warmups and five measured frames
per case, with rotating interleaved case order. Synthetic inputs have a thin near
surface, holes, exposure differences and explicit right depth. No stereo or
physical capture runs inside this benchmark. Legacy nearest-only behavior is
loaded from adapter revision `88a9e37`, using preserved legacy shader entry points.

| Resolution | Legacy nearest wall median | Full strict | Full preview |
|---|---:|---:|---:|
| 320×180 | 7.995 ms | 5.885 ms | 5.943 ms |
| 1280×720 | 119.929 ms | 90.455 ms | 90.318 ms |

At 720p the median of per-frame summed GPU dispatch times is 1.196 ms strict
and 1.414 ms preview. Readback/unpack medians are 73.20 and 72.77 ms; the packing
dispatch costs approximately 0.228 ms. Host preparation includes transfers and
is not reported as CPU-only time. End-to-end timings include upload, submission,
wait, readback and unpack. These few variable samples establish neither a
general speedup nor sustained real-time throughput.

The unpacked full compositor previously measured 155.929/153.043 ms strict/preview.
Packing was introduced in response to its measured readback bottleneck. Original
reports remain alongside the packed results. Current Python buffer payload is
125,386,764 bytes at 720p, versus 81,100,800 for the legacy adapter; this excludes
host arrays, staging, driver allocations and programs. Native project-only buffer
payload for two leased 720p slots is 295,010,328 bytes, excluding staging,
programs and temporal scratch. Neither number is measured process peak memory.

```bash
PYTHONPATH=python .venv/bin/python benchmarks/benchmark_view_fusion.py \
  --width 1280 --height 720 --warmup 2 --frames 5 \
  --output artifacts/fusion-cost-new.json
```

Reports retain raw timing samples, sampling order, input identities, byte counts,
device, configuration, per-stage timestamps, allocations and source hashes.
Current adapter SHA256 is
`26bb89007792047279bbcc805c70b0cb2fe2a817ab91842456a5a22c38527350`;
shader SHA256 is
`d17346b667d1b4bce0b816351eaed16aea107ded910249732622f682259b70ec`.

| Local artifact | SHA256 |
|---|---|
| `artifacts/fusion-cost-320x180.json` | `34517eb4cd280c8d6da0d34c329624bffe1439d7d2693c71dd7ab1048a723493` |
| `artifacts/fusion-cost-1280x720.json` | `1bcc545b28e55b6a456ae51c0a59368b3b8b236d262bf8e9ae13720b31d32d45` |

## Real-camera replay and float precision

Three recorded VGA ZED frames, IDs 16–18, ran through complete Python Vulkan
stereo (D64, four paths) and strict synthesis. Two warm calls took 117.9–118.2 ms;
the initial call took 298.7 ms. This is a numerical smoke on a saved scene, not
physical distance validation or sustained live throughput. Strict depth coverage
was 24.50–24.80%; preview RGB coverage was 100%, with approximately 75% explicitly
approximate pixels. Approximate pixels never became metric-valid and retained
zero confidence/provenance. Wide color coverage is not measured RGB-D coverage.

The float64 CPU oracle and float32 GPU differ at a few pixel-rounding boundaries.
For frame 18, three validity pixels and four source pixels differ. A diagnostic
shader using the production projection proved that all affected pixels arise
from source projections within 0.000008 pixel of a half-pixel boundary: float32
arithmetic rounds to an adjacent output cell. This can produce a large local
color difference when one cell is a strict hole or has another owner; away from
those four cells the channel difference is at most one. No algorithm or acceptance
threshold was changed to hide these differences. Exact float64/GPU mask parity
at quantization boundaries is not claimed.

The failed exact comparison and diagnostic are retained separately in
`artifacts/fusion-camera-replay-20261027.json` and
`artifacts/projection-precision-frame18.json`. The latter SHA256 is
`114ccb36b2bff35f8095efb2cdcd9ebacfb2acdebe4802c0d2935e2571db2bc8`.

## Native core validation

Full native compositor/temporal contract fixtures and strict/preview CLI runs
completed with Khronos core validation enabled, exit zero and empty stderr.
The fixtures exercised actual approximate-only pixels, metric identity masking,
exposure, blending, foreground ordering, map validation, temporal behavior and
retained leases. A loader trace confirms layer insertion for both instance and
device. Evidence: `artifacts/native-compositor-core-validation-1793103711691456662/report.json`,
SHA256 `07f62a13dbc4d3b273514b677e6523d2d19400672f259cc2b34e7b128bef7e6a`.
Synchronization/GPU-assisted validation and the integrated native live chain
remain unqualified. Layer setup is documented in vulkan-validation-evidence.md.
