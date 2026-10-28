# Direct borrowing: matched 720p refinement experiment

On AMD Radeon 8060S with RADV Mesa 26.2.3, borrowing full-resolution preprocessing buffers produced lower median combined GPU queue time than six-copy reuse in both a synthetic A-B-C-C-B-A run and a recorded-camera B-C-C-B run. The magnitude is small relative to run-to-run variation, and the tail results do not establish a consistent p99 improvement. This is a repeated-record service-time experiment, not sustained live throughput or a 60 FPS qualification.

Every arm computes independent 640×360/D64 coarse stereo and radius-2 refinement to 1280×720, with two slots, 20 warmups and 100 measured iterations. Each also uploads, decodes and rectifies the same source separately at full resolution. A (`coarse_refine_full`) performs the full producer and separately rectifies refinement input; B (`raw_reuse`) copies six full-frame planes; C (`raw_borrow`) borrows those buffers. GPU combined time is the same-queue interval from coarse upload start to final refinement metric completion, including intervening gaps. Host wall includes submission, wait and timing query; each frame is waited before the next is submitted. Readback and parity checks occur after timing. The three arms do not include acquisition, wide view, ROS, downstream consumers or frame age.

## Synthetic fixture

The order was A1, B1, C1, C2, B2, A2. The capture is a single repeated synthetic packed-YUYV record. Percentiles use nearest rank; max and individual chronological samples are in the local JSON. At 100 measurements p99 is the second-highest sample, not a qualified sustained tail estimate.

| Run | Variant | GPU combined p50 / p99 / max (ms) | Host wall p50 / p99 / max (ms) |
|---|---|---:|---:|
| A1 | direct + full producer | 8.462 / 8.903 / 8.916 | 8.727 / 9.179 / 9.218 |
| B1 | copy | 8.236 / 8.476 / 8.489 | 8.498 / 8.776 / 8.785 |
| C1 | borrow | 8.072 / 8.426 / 8.522 | 8.272 / 8.695 / 8.752 |
| C2 | borrow | 7.972 / 8.385 / 8.391 | 8.198 / 8.660 / 8.685 |
| B2 | copy | 8.202 / 8.441 / 8.452 | 8.447 / 8.707 / 8.794 |
| A2 | direct + full producer | 8.222 / 8.441 / 8.593 | 8.469 / 8.769 / 8.894 |

The copy packing stage p50/p99 was 0.167/0.184 ms and 0.162/0.185 ms in B1 and B2. Borrow has no packing dispatch; its unavailable first-stage value is represented as null, not zero. Each B run matched all 30 output planes byte-for-byte against direct on its final repeated frame; each C run matched both direct and copy. Timing had validation disabled; zero recorded counters there do not prove validation was active. Independent synthetic borrow validation ran 300 measured repetitions plus one warmup with core and synchronization validation, zero errors or warnings through teardown, and final-frame exact parity against both references. That run serializes the same frame; it does not establish concurrent multi-frame behavior.

## Recorded connected-camera fixture

An existing private 720p capture from the connected ZED 2 was replayed as a single repeated frame; it was not live acquisition during these measurements. The matching full and half-resolution fixtures retain the same capture record and calibration. The B1, C1, C2, B2 order used the same 100/20 settings as above.

| Run | Variant | GPU combined p50 / p99 / max (ms) | Host wall p50 / p99 / max (ms) |
|---|---|---:|---:|
| B1 | copy | 7.796 / 7.947 / 8.636 | 8.061 / 8.991 / 9.515 |
| C1 | borrow | 7.649 / 7.921 / 7.927 | 7.929 / 8.204 / 8.249 |
| C2 | borrow | 7.684 / 7.924 / 7.924 | 7.919 / 8.165 / 8.225 |
| B2 | copy | 7.788 / 10.683 / 10.765 | 8.079 / 11.213 / 14.045 |

B2 has a pronounced tail outlier; its cause is not established. Each B run passed final-frame 30-plane parity against direct and each C run against direct and copy. A separate camera-record borrow run used three measured repetitions and one warmup with core and synchronization validation: zero errors/warnings through teardown and final-frame parity against both references. No metric ground truth was available, so this is neither a physical depth-accuracy evaluation nor a live-camera throughput test. Private imagery and identifying calibration and capture metadata remain in local temporary files and are not included here.

All arms requested 733,596,016 bytes for copy and borrow and 748,341,616 bytes for direct with full producer. These are requested buffer totals, including the formula for full preprocessing, not actual driver allocations or peak memory. Borrow retains the legacy output allocations; removal of six copies does not itself save those allocations. Earlier lifecycle and two-distinct-frame synthetic tests are described in [initial borrow evidence](daxa-refinement-direct-borrow-evidence.md). Guided refinement's thin-object quality regression remains unresolved; exact agreement between these variants does not establish accuracy.

## Synthetic reproduction and identities

The fixture and result files are ignored local data and may not exist in a fresh checkout. Build the benchmark, then run the following sequence with fresh output paths; do not run concurrent GPU work. Set up the Vulkan validation layer separately as in [raw reuse evidence](daxa-raw-frame-reuse-evidence.md). Validation runs must not be used as timing comparisons.

```bash
cmake --build build/daxa --target daxa_refinement_benchmark
F=artifacts/fast60-refinement-20261028/analytic-family-720p
O=/tmp/stereoforge-borrow-720p-repro
mkdir -p "$O"
for spec in a1:coarse_refine_full b1:raw_reuse c1:raw_borrow c2:raw_borrow b2:raw_reuse a2:coarse_refine_full; do
  name=${spec%%:*}; variant=${spec#*:}
  env -u VK_LAYER_ENABLES -u VK_LAYER_VALIDATE_SYNC -u VK_INSTANCE_LAYERS \
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
    flock /tmp/stereoforge-gpu.lock build/daxa/daxa_refinement_benchmark \
    --coarse-manifest "$F/half/manifest.json" --target-manifest "$F/full/manifest.json" \
    --output "$O/$name.json" --radius 2 --subpixel 0 --slots 2 --wave-size 32 \
    --frames 100 --warmup 20 --variant "$variant"
done
# With core and synchronization validation configured, separately:
VK_LAYER_VALIDATE_SYNC=1 flock /tmp/stereoforge-gpu.lock build/daxa/daxa_refinement_benchmark \
  --coarse-manifest "$F/half/manifest.json" --target-manifest "$F/full/manifest.json" \
  --output "$O/borrow-validation-300.json" --radius 2 --subpixel 0 --slots 2 \
  --wave-size 32 --frames 300 --warmup 1 --variant raw_borrow --validation
```

Measured executable SHA-256: `38b920f8442be1dee3b11dc5147fe70d69ee631977c908bc4dde31829157e502`; benchmark source: `ff9d3f5998e7d18f47cf8573b52e558a1c7651b52c04011ae23a64af66ab9ada`; refinement source: `297de3dd34c614ba5823330cf0ff0c362c2a85c526e3989c4f5a666218bb943e`. The coarse/target manifest hashes were `98e76b180aaad43a38cc8bdfafa8f5e73d045e568279242c118719270877d5ad` / `2799d18f1bfa13bf862817aabaf05c917055af371472e162467437bc837c7ea0`; the synthetic capture hash was `0b7e45a21752c6a0fcf867b327f58b2d0d7b5c8d8f1949044e873b9935c42078`. Each result records compiler and shader identities.

| Local synthetic artifact | SHA-256 |
|---|---|
| `a1.json` | `4568bd8f0cd1807bde84208501198a8f4dc75a899610d1354b63ab44021d7a20` |
| `b1.json` | `ef752d7ec1be80614741e6198e4b887f8efe020ad4fa4bda2d91773cc89f191c` |
| `c1.json` | `6860dad91564425b66c4a1a48721528fc6704c72b9080d768f490d608681dc98` |
| `c2.json` | `f65566c77cc397e472e14a367fc096fe4753d13b5749c50c32d43c6ae652e8ab` |
| `b2.json` | `79db829e3215bc58cae240b67098b1fea93225d8fade2c32ba88b4fd5a1ce67f` |
| `a2.json` | `c7eadca26483e00d8f7d19626b4d501cbc99e2c7eb36c8aa8ea0232b1c991ebe` |
| `borrow-validation-300.json` | `611a8448172d03d983469cdf9f109b08adfc24df59d98cf21531b86fb908b393` |

These files reside under `artifacts/borrow-720p-20261028-v2/`. The separate recorded-camera JSON files reside in private `/dev/shm/stereoforge-borrow-test/` and include camera metadata. A live, current-source borrow pipeline, multi-frame overlapping validation, calibrated accuracy and sustained capture-to-consumer measurements remain required before adopting this path for real-time output.
