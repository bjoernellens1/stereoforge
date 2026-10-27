# Native retained GPU inputs and downstream handoff

Measured 2026-10-27 on AMD Radeon 8060S (RADV STRIX_HALO), Vulkan only.
The standalone C++/Slang RHI executable links the official Slang compiler and
system C++ libraries; it has no SGL, SlangPy or Python dependency. Each measured
child ran with `PYTHONHOME=/nonexistent` and `PYTHONPATH=''`. Python only supervises
commands, checks file hashes and compares output dumps outside the measurements.

## Matched 1280×720 fusion measurement

The source arrays, target view and seven compositor passes are identical between
host and resident input modes. Host mode validates and uploads source planes each
iteration. Resident mode validates/uploads one static snapshot before warmup;
each timed call consumes its retained GPU buffers. Both wait for completion and
leave output on the GPU. A full CPU dump happens once after each timing loop.
Device/pipeline construction and initial resident upload are excluded.

Thirty measured calls follow three warmup calls per case. Cases ran sequentially,
with no other coordinated GPU workload. Times include native submission and
completion waiting; they are not isolated shader timestamps.

| Mode | Host inputs p50 / p95 | Resident inputs p50 / p95 | One-time resident upload + wait |
|---|---:|---:|---:|
| Strict | 22.362 / 23.310 ms | 1.427 / 1.500 ms | 22.708 ms |
| Preview | 23.492 / 26.085 ms | 1.468 / 2.134 ms | 18.670 ms |

Resident submission p50 is 0.255 ms strict and 0.251 ms preview. These are static
fixture measurements: retaining an unchanged source avoids repeated source work.
They establish the benefit of a GPU-input interface, not live camera freshness,
full stereo throughput, a sustained 60 Hz pipeline or complete CPU-output latency.
The earlier Python adapter's approximately 90 ms includes CPU output readback and
host preparation; that scope is different from retained GPU output here.

All eleven final planes match host mode exactly in both views: depth, confidence,
packed RGB, metric/RGB validity, source, provenance, approximation, owner, original
64-bit observation identity and age. Invalid pixels retain identity −1 and age 0.
Preview-only pixels retain invalid metric depth and zero confidence/provenance.
There are 366,753 metric pixels; preview adds 731 approximate color pixels without
adding metric observations. This fixture does not exercise every calibration or
floating-point raster boundary; the documented float32 projection limit remains.

## Actual GPU producer and consumer

`native_gpu_handoff` uses one `NativeDevice` and its queue for calibrated
`NativeStereo`, a bridge kernel, `NativeGeometry`, and a downstream compute
consumer. Three 64×24 shifted-pair frames each produce GPU depth. The bridge
copies only measured left depth/confidence, updates RGB and a fresh 64-bit
observation identity, and keeps right depth invalid. Undistorted parallel source
geometry makes the source left grid identical to the rectified grid in this test.
The compositor materializes selected metric identities/ages directly on the GPU.

There is no intermediate CPU wait or full-plane download in each chain. The GPU
consumer checks depth, masks, confidence, source, provenance, RGB, identity and age;
only four uint counters (16 bytes) return after consumer completion. Every frame
reports 1,154 measured pixels, 382 invalid pixels, zero errors and 1,120 pixels at
the known 1.8 m synthetic depth. Fresh IDs/colors are checked across all three.

After consumer submission, the fixture drops high-level producer/result owners
and submits two further stereo jobs before waiting. All three rounds successfully
reuse producer slots without backpressure and preserve the consumer results.
Same-queue ordering and RHI command resource retention are essential; this does
not establish cross-queue or external-device interoperability. Standalone CTest
passes 5/5. Full Python regression verification passes 265 tests; CPU CTest
passes 1/1. Positive Khronos core validation loaded the actual layer and emitted
zero diagnostics across both stdout and stderr for native stereo, geometry/temporal
contracts and this handoff fixture. The handoff repeated the same three successful
freshness/reuse rounds under validation. This covers core validation, not
synchronization validation, GPU-assisted validation or the complete live chain.

This closes a synthetic shared-device producer/consumer correctness slice. A
native ZED preprocessing/rectification, both-direction stereo consistency, temporal
processing and live output chain remains open. Stereo compute and the complete
frame budget still require measurement and optimization; the fusion number alone
cannot qualify 60 Hz.

## Reproduction and provenance

```bash
cmake --preset standalone
cmake --build --preset standalone
.venv/bin/python benchmarks/compare_native_resident.py \
  --fixture artifacts/fixture-720p \
  --output artifacts/native-resident-20261027
ctest --preset standalone --output-on-failure
export VK_LAYER_PATH="$PWD/build/validation-layer/usr/share/vulkan/explicit_layer.d"
export LD_LIBRARY_PATH="$PWD/build/validation-layer/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
build/standalone/native_gpu_handoff --validation
build/standalone/native_geometry_contracts --validation
build/standalone/native_stereo --validation --self-test
```

The supervisory script's `--verify-only` option checks fixture bytes/linkage and
prints commands without executing GPU work. A fresh/empty output directory is
required. The fixture is exported by `benchmarks/export_view_fixture.py`; see
`standalone-rhi-comparison.md` for its generation and the previous host-only scope.

Local ignored evidence is in `artifacts/native-resident-20261027/`: raw timing
samples, post-measurement binary planes and `comparison.json`. The latter binds
commands, environment, linked dependencies, source/binary hashes and every output.

Core-validation logs and loader proof are retained in
`artifacts/native-resident-core-validation-20261027/report.json`; its three cases
record successful exits, actual layer loading and empty diagnostic lists across
both streams. See vulkan-validation-evidence.md for the signed local layer setup.

- Input identity: `6b1ae7dacab2e81554b56e504fb2fd235657069c8716485e666cf436caa46803`.
- Benchmark binary: `341af88a59bb7ed5b5d90bd9eab89e7013503883e41a5ffce3ecf123d395a8a9`.
- Comparison JSON: `e0d151dd908b9e5a8635020f1a7c8e2905ce85f4651c5fb63ec4154e7a2ccc41`.
