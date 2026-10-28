# Raw full-frame reuse: matched 720p experiment

A matched A-B-B-A experiment on AMD Radeon 8060S (RADV, Mesa 26.2.3) found **no repeatable reduction in complete GPU or host latency** from reusing a full preprocessing frame. The six-copy packing stage itself was faster than the direct target rectification stage. This is an opt-in experiment, not a production or 60 FPS qualification.

Both arms run independent 640×360/D64 coarse stereo and radius-2 refinement to 1280×720 on the same repeated synthetic packed-YUYV record (source ID 7); two slots, 100 measured iterations after 20 warmups, and validation disabled for timing. A (`coarse_refine_full`) also produces the full frame but runs direct refinement; B (`raw_reuse`) produces the same full frame and copies six planes into refinement. Both include a second upload/decode and full target rectification. The source, queue, calibration and configuration are shared; this is not zero-copy. GPU combined timing spans the coarse upload start through final refinement metric completion on the same queue, including gaps. Host wall includes submission, waiting and timing queries. Full-frame readback and the B-versus-direct 30-plane byte comparison happen only after timing, on the final repeated frame. Timing excludes live acquisition, wide view, ROS and a downstream consumer.

| Run | Variant | GPU combined p50 / p99 ms | Host wall p50 / p99 ms | Direct rectify or reuse pack p50 / p99 ms |
|---|---|---:|---:|---:|
| A1 | direct + full producer | 8.276 / 8.532 | 8.575 / 8.872 | 0.251 / 0.261 |
| B1 | reuse + full producer | 8.366 / 8.617 | 8.611 / 8.888 | 0.163 / 0.180 |
| B2 | reuse + full producer | 8.310 / 8.528 | 8.527 / 8.769 | 0.164 / 0.181 |
| A2 | direct + full producer | 8.260 / 8.559 | 8.507 / 8.787 | 0.250 / 0.263 |

Percentiles use nearest rank and are calculated independently per run; they must not be added or subtracted to infer frame-level latency. B1 and B2 each passed the post-timing exact comparison against the direct output for all 30 planes. Separate three-frame, one-warmup runs with Vulkan synchronization validation enabled reported zero errors and warnings through teardown in both arms, with B also passing the exact comparison. Timing runs had validation disabled; their zero counters alone do not establish validation coverage. Prior small-image two-distinct-frame lifecycle coverage is described in [raw-frame reuse evidence](daxa-raw-frame-reuse-evidence.md).

The two sequential C controls omit the full producer entirely and are contextual, not part of the paired ABBA comparison: C1 GPU p50/p99 7.857/8.027 ms and host 8.163/8.446 ms; C2 GPU 7.959/8.267 ms and host 8.212/8.556 ms. Requested buffer totals for two slots were 748,341,552 bytes for A, 733,595,952 for B and 645,122,096 for C. The A/B totals include the full producer's 103,219,456-byte `DaxaRectification::requested_bytes(2)` formula. These are requested-buffer accounting figures, not measured driver allocations; allocator and driver overhead, CPU source and diagnostics are excluded.

This experiment does not establish 60 FPS, physical accuracy, live throughput, a complete RGB-D pipeline, or full-resolution SGM quality. The existing guided refinement has severe thin-geometry completeness regressions relative to independent full-resolution SGM; see [refinement quality evidence](fast60-refinement-evidence.md). The shared atlas, true direct borrowing, dynamic validity and sustained multi-slot behavior remain to be qualified.

## Reproduction and identities

The local fixture and artifacts are ignored build data, not guaranteed present in a fresh checkout. These commands use fresh output filenames and the existing local fixture. The timing results use the order A1, B1, B2, A2, followed by C1 and C2; validation was separate. Use the local Vulkan validation-layer setup described in [initial evidence](daxa-raw-frame-reuse-evidence.md) for the two validation runs.

```bash
cmake --build build/daxa --target daxa_refinement_benchmark
F=artifacts/fast60-refinement-20261028/analytic-family-720p
O=/tmp/stereoforge-raw-reuse-repro
mkdir -p "$O"
for spec in a1:coarse_refine_full b1:raw_reuse b2:raw_reuse a2:coarse_refine_full c1:coarse_refine c2:coarse_refine; do
  name=${spec%%:*}; variant=${spec#*:}
  env -u VK_LAYER_ENABLES -u VK_LAYER_VALIDATE_SYNC -u VK_INSTANCE_LAYERS \
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
    flock /tmp/stereoforge-gpu.lock build/daxa/daxa_refinement_benchmark \
    --coarse-manifest "$F/half/manifest.json" --target-manifest "$F/full/manifest.json" \
    --output "$O/$name.json" --radius 2 --subpixel 0 --slots 2 --wave-size 32 \
    --frames 100 --warmup 20 --variant "$variant"
done
# With the validation-layer environment configured, repeat separately:
for variant in coarse_refine_full raw_reuse; do
  VK_LAYER_VALIDATE_SYNC=1 flock /tmp/stereoforge-gpu.lock build/daxa/daxa_refinement_benchmark \
    --coarse-manifest "$F/half/manifest.json" --target-manifest "$F/full/manifest.json" \
    --output "$O/$variant-validation.json" --radius 2 --subpixel 0 --slots 2 --wave-size 32 \
    --frames 3 --warmup 1 --variant "$variant" --validation
done
```

SHA-256 for this measured local executable: `b17194bc229573dc80f0d2f39f17b00f4c4c4f624d3b1c9449ecc200362b8da0`; benchmark source: `c47c73ff7e4a2f0d2f81b333d8bd68bce83eb5f3d83d6f3501195c71b5ee0030`; refinement source: `addba038f273a9c9ff56959ae326f1598738777bcc5ae9c016caa587a8cec8bc`. The coarse and target manifest hashes were `98e76b180aaad43a38cc8bdfafa8f5e73d045e568279242c118719270877d5ad` and `2799d18f1bfa13bf862817aabaf05c917055af371472e162467437bc837c7ea0`; the capture hash was `0b7e45a21752c6a0fcf867b327f58b2d0d7b5c8d8f1949044e873b9935c42078`. Shader and compiler identities are in each result.

| Local artifact | SHA-256 |
|---|---|
| `a1.json` | `e87028fb0a28fd6e6f4f9d2bc52012fe1eb63e1031279c8ee796d4be80487059` |
| `b1.json` | `9469721cc7b66bc7aa4903ca991b1d6d728a9e2997da3cd70bc0d0c02e533dcb` |
| `b2.json` | `9df465a929c03bd36eef14b6e4844ad29c0afb88c48e827cab8033379f29f5c1` |
| `a2.json` | `06d09b2ed3cf80743f7194a68a989ad5c3f3dd29a612f1c6af8b761d7ca663c1` |
| `c1.json` | `df6a8196993b716631ce6929905328c07b156893e49d82fc4086e343c0e3f982` |
| `c2.json` | `8772c77dbb829afcd1f902f798ae54cb3f63c9782034d3cf782131d8ababe2c0` |
| `/tmp/stereoforge-raw-direct-validation-v1.json` | `e8943a249d77b9d3ca6a96bd1a1819be497f3c1b3de6232d3ebb27c17f9ab12b` |
| `/tmp/stereoforge-raw-reuse-validation-v1.json` | `1ed2b1f8ef33e363b57856b248f5c450e895045e2def0216ce7f86259b886bcf` |

The six timing JSON files are under `artifacts/raw-reuse-720p-20261028-v1/` and contain chronological samples and complete setup metadata.
