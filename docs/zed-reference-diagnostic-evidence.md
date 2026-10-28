# SDK reference diagnostic evidence

Measured 2026-10-28. This implements an intermediate step toward the
[quality benchmark](superpowers/plans/2026-10-28-zed-quality-benchmark.md).
Canonical SDK records, depth-quality comparison and sustained delivery remain
unqualified.

## Executed boundary

The optional C++20 exporter built with GCC 13.3, `-Wall -Wextra -Werror`, against
installed SDK **5.5.0** and CUDA Toolkit 12.8.93 on Ubuntu 24.04 / RTX 4070 Ti.
The vendor image was pinned by manifest digest:

```
stereolabs/zed@sha256:0297481b344437ad2874edaef4c32da821491c15cc761187c1d6e0c902148cde
```

Its development image lacked OpenSSL headers. An isolated derived image added
`libssl-dev=3.0.13-0ubuntu3.16`; host SDK/CUDA packages were not installed.
Production Vulkan targets do not discover these optional reference dependencies.
The default-OFF build configured, built and passed `native_contracts` without SDK
discovery. Build instructions are in the
[comparison README](../benchmarks/zed_comparison/README.md).

The official [ZED 2 lake sample](https://download.stereolabs.com/assets/svo_samples/ZED2_Lake_H264)
was preserved and hashed: 95,926,145 bytes, SHA256
`8c590a68c25e299a43eab9e116faa435be4e5553c62fb8f8ab6413abb700fe8d`.
This legacy `.svo` sample is 2208×1242 per eye at nominal 15 Hz. It is not the
attached camera or a 720p/60 test. The publisher labels it H.264; source lossless
encoding remains unattested, distinct from byte-exact diagnostic serialization.

After independent functional and CPU serialization reviews, a named container
replayed two frames with a host timeout of 120 seconds and forced cleanup.
Pre/post checks found no compute processes or remaining owned containers.
Effective settings after each grab retained `NONE`, depth disabled, tracking
disabled, stabilization zero, filling disabled, non-real-time playback, metres
and IMAGE coordinates. Cursors were 0 and 1. All eight exported BGRA plane hashes
and byte extents were independently verified.

Initial runs failed explicitly: a settings directory without its SDK-required
trailing slash could not resolve calibration; with that fixed, SDK `NONE` mode
reported a minimum-depth sentinel instead of the requested metric range. The
exporter now normalizes the separator and enforces metric range equality only
when depth is enabled, retaining both requested and effective settings.

## Diagnostic timings

| Scope | Frame 0 | Frame 1 |
| --- | ---: | ---: |
| Opaque SDK grab API | 41.83 ms | 22.91 ms |
| Four-image CPU retrieval APIs | 31.53 ms | 8.28 ms |
| Serialization, hashing and file I/O | 81.27 ms | 81.28 ms |

These are two diagnostic samples without warmup or isolated GPU timestamps.
They include explicit full-image host output and disk writes, and establish no
kernel speed, real-time throughput or camera-to-output latency claim. The live
Vulkan pipeline continues to retain outputs on the GPU.

Separately, the actual SDK-free serialization helper was compiled and probed on
the local CPU. A 3,686,400-byte packed BGRA plane took 2.834/1.724/1.842 ms for
write plus SHA; F32 took 1.777/1.737/1.749 ms. Packed planes use one write, padded
planes write logical rows only. Independent fixed hashes verify padding exclusion
and preservation of NaN payloads and infinities. A 32 MiB source copied in 32
chunks of at most 1 MiB took 16.66 ms. These are CPU diagnostic costs only.

## Geometry diagnostics

The inspected SDK API does not provide a documented per-eye rectification
rotation or geometric sampling-support map. Alpha, black pixels and finite depth
do not establish RGB support. The exporter therefore writes
`captured_unqualified` diagnostics, never canonical `export.json` records.

An offline fit used SIFT descriptors, reciprocal L2 matches and a fixed 0.7 ratio
test in both directions, with alternating sorted matches split into training and
validation. No geometric outlier filter or RANSAC removed difficult pairs.

| Frame / eye | Selected pairs | Validation median / p95 / maximum |
| --- | ---: | ---: |
| 0 / left | 2,598 | 0.063 / 0.281 / 1.158 px |
| 0 / right | 2,456 | 0.067 / 0.317 / 4.735 px |
| 1 / left | 2,612 | 0.717 / 1.154 / 2.700 px |
| 1 / right | 2,504 | 0.070 / 0.287 / 715.790 px |

Frame 1 left also retained an 850 px training residual. Descriptor matches are
not ground truth, and contaminated least-squares fits cannot qualify geometry.
The two tested rigid-transform interpretations disagree: inversion improves
rotation consistency but reverses the baseline, while the direct interpretation
preserves baseline direction with higher rotation residual. Neither is admitted.
The vendor's [OpenCV example](https://github.com/stereolabs/zed-opencv-native/blob/master/python/zed_opencv_native.py)
uses a negative calibration-file baseline for OpenCV rectification; this does
not by itself establish the installed SDK matrix convention or identical maps.

The reusable math module keeps supplied rotations strict at `1e-10`. SDK matrix
representation errors may be at most `1e-6` before an explicitly reported polar
projection; input bytes and translations remain unchanged. This accommodates
observed float32 roundoff, not physical calibration error. Independent scalar
synthetic tests verify fitting, validation isolation and rejection behavior.

Next gates are verified raw-pose normalization, distributed raw-to-rectified
correspondences with localization uncertainty, SDK border/interpolation support,
and axial-depth validation. Canonical records and the native rectified adapter
must preserve these distinctions. No SDK neural depth mode or competitive quality
result has been qualified by this diagnostic.

Ignored local evidence is under `artifacts/zed-task2/` (build logs, functional
review, CPU performance report and captures) and `artifacts/zed-task3/fit/`
(math tests, complete matched pairs/residuals, source hashes and scripts).
Proprietary headers, libraries, models, settings and sample payloads are excluded
from Git.
