# Dual-resolution preprocessing: development evidence

An opt-in Daxa producer uploads and decodes one packed capture, then remaps it independently onto the full and coarse calibrated grids. It shares the original RGB buffers while keeping grid-specific rectified RGB, luma and support. The borrowed refinement path consumes the two products from the same source lease. This is an experiment; it has not been integrated into the live ROS publisher or shown to improve GPU time.

## Matched 720p measurements

On AMD Radeon 8060S / RADV Mesa 26.2.3, A = separate coarse and full producers and B = dual producer, both followed by the same independent 360p/D64 four-path SGM and 720p/D128 local refinement, radius 4, no subpixel, two slots, and borrowed full-frame inputs. The ABBA runs each used 20 warmup and 100 measured iterations, one retained source repeated serially. Times are **host wall milliseconds** from collection through the final refinement wait, including CPU submission and waiting; they are not GPU timestamps and exclude capture, wide RGB-D and transport. Enqueue includes resource collection and admission through submission. Both configurations were allocated in each benchmark process, so this is not a memory comparison.

| Input | Run | Host wall p50 | p99 | max | Enqueue p50 | p99 | max |
|---|---|---:|---:|---:|---:|---:|---:|
| Synthetic | A1 | 8.447 | 9.129 | 9.167 | 0.435 | 0.874 | 0.877 |
| Synthetic | B1 | 8.276 | 8.530 | 8.596 | 0.179 | 0.420 | 0.495 |
| Synthetic | B2 | 8.323 | 8.566 | 8.675 | 0.207 | 0.400 | 0.417 |
| Synthetic | A2 | 8.456 | 8.779 | 8.801 | 0.288 | 0.604 | 0.669 |
| Recorded ZED 2 frame | A1 | 7.859 | 8.150 | 8.194 | 0.278 | 0.678 | 0.692 |
| Recorded ZED 2 frame | B1 | 7.886 | 8.129 | 8.241 | 0.187 | 0.295 | 0.342 |
| Recorded ZED 2 frame | B2 | 7.891 | 8.123 | 8.133 | 0.177 | 0.391 | 0.405 |
| Recorded ZED 2 frame | A2 | 7.885 | 8.112 | 8.195 | 0.279 | 0.552 | 0.595 |

Percentiles use nearest rank `ceil(p*N)-1`. Dual consistently reduced median host enqueue time here, while overall host-wall medians improved for the synthetic frame but not the recorded frame. Tail results are mixed; do not infer a consistent end-to-end or GPU speedup. The recorded input is a repeated previously captured camera frame, not a live stream. Both branches were prepared within each process, and after timing their 30 logical refinement planes matched byte-for-byte for the tested input. This parity checks equivalence to the existing path, not metric accuracy.

Separate short synthetic 720p runs requested Vulkan validation: three measured frames after one warmup per variant, all 30 output planes equal, with zero reported errors or warnings through teardown. A separate dual-only run requested validation for 300 serialized repeated frames after one warmup, reported zero errors and warnings through teardown, and checked all 30 output planes after timing; it did not test overlapping frames. These are repeated-frame checks; synchronization-validation activation was not independently attested. Small GPU contracts exercised both grids on two distinct input frames against all eight scalar preprocessing planes and against independent GPU producers, retained leases, backpressure and recycling. A downstream two-frame test compares all 30 refinement output planes, tests a retained View consumer and checks zero validation warnings/errors through teardown. These do not establish long-running live operation or physical depth quality.

An initial dual-producer graph used duplicate resource names and triggered a Daxa assertion before measurement; unique per-slot/resource names fixed it. The retained-source fake-device microprobe used the actual enqueue implementation and a 3,686,400-byte copy: approximately **49.2 µs/frame** with copy enabled, one copy and zero tracked application allocations. The fake graph/device and host memory do not measure Vulkan execution, GPU synchronization, or camera input.

## Reproduction and identities

Local ignored reports and fixtures may not exist in a fresh checkout. The following reproduces the synthetic runs with a fresh output directory and existing analytic fixture; it does not reproduce private camera frames. Use a coordinated GPU window. Verify shader/compiler and input hashes in each generated report and independently preserve the command environment. `--validation` requests validation but does not by itself attest synchronization-validation activation.

```bash
cmake --build build/daxa --target daxa_dual_preprocess_benchmark -j 4
F=artifacts/fast60-refinement-20261028/analytic-family-720p
O=/tmp/stereoforge-dual-repro
mkdir -p "$O"
for entry in a1:separate b1:dual b2:dual a2:separate; do
  name=${entry%%:*}; variant=${entry##*:}
  flock /tmp/stereoforge-gpu.lock ./build/daxa/daxa_dual_preprocess_benchmark \
    --coarse-manifest "$F/half/manifest.json" --target-manifest "$F/full/manifest.json" \
    --frames 100 --warmup 20 --slots 2 --radius 4 --subpixel 0 \
    --variant "$variant" --output "$O/$name.json"
done
```

The synthetic report SHA-256 values are A1 `1e436fc895a6c22b5880ae7ca5b56dbba28f38919fd88d621c78177458b23696`, B1 `b8fd13146e98b5744863682dc7c076165723b130f6e49a58e3fe00edfd50e673`, B2 `048c0d1a399abe49e66e5c6840c44bfcbb36037365b171686e98e6e435bdbf76`, A2 `595a6fe112b4ec5c82ed5b1e0d1e4d329fc5cbfc0af7fca9d1287c3aeb3e8839`; validation A `be0bf2991d6f0303135768f9026857016bf2a68274951ee55a86c9262987947d` and B `d502e59e5a3a618ec284384ba9118ebc9d3aff7e7cadf24462cf13033666944c`; the 300-frame dual validation report is `d758974af565a8210c3cc9b8da96697ffa4a02030f8b690ed4dd2f3c1e140ab8`. The measured benchmark source and binary hashes were `5e9c22ec8af5e9d0bedef61782a33d16c5000b3a01238ce42ca255b0f5beff08` and `de9f0d1c266b9f266c6da9e3b6c8087621e0d763541a9766fa403bcd1aa57e25`; the dual producer source hash was `d3c91d59a42d2a442ee6aaf8c2cd9ddfdb0cf4764a9d5ca2f5d504a412ddec51`. Binary identity is local-build specific. Private camera identifiers and images are not published.

The generic producer validates each input lease against the bound source geometry, stride and original calibration identity; it does not prove the two maps share valid stereo geometry merely from their identity. The refinement consumer independently validates the stricter calibrated 2× relationship. Dynamic frame support and downstream validity are not replaced by static calibration identity.

Next, instrument same-queue upload-through-metric GPU timestamps and exact requested allocation accounting, test variable in-flight camera frames, and integrate an opt-in live path before making a live throughput or memory claim. Continue thin-object quality and independent ground-truth qualification separately.
