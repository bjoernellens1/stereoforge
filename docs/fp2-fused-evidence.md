# FP2 fused Census cost experiment

The isolated fused recurrence passed scoped GPU parity and reduced matched synthetic GPU p50 from 90.471–93.061 ms in the two existing parallel control blocks to 66.171 ms, approximately 27–29%. It remains well above the 16.67 ms frame interval. This is repeated-input diagnostic timing, not live-camera throughput, transfer-inclusive latency, physical depth quality, unrestricted full-resolution SGM or native/subpixel equivalence.

The candidate retains GPU Census descriptors and support masks, removes the packed cost buffer and its reset/producer dispatch, and recomputes supported descriptor XOR/popcount in each of four path passes. Eye 0 searches `x-d`; eye 1 searches `x+d`. It retains 128-lane scanline workgroups and the shared tree reduction; inactive lanes still participate, so this is not work-proportional compact recurrence.

## Parity and work

GPU parity compared both eyes' four path arrays, sums and padding with an independent dense CPU recurrence on deterministic descriptor/support 7×5 and 320×180 fixtures. The pinned analytic 1280×720 fixture compared final raw selections and observations against frozen CPU selections and prior integrated GPU output hashes; 720p per-path arrays were not read back. The 720p raw valid counts were 897,952 / 898,067 and LR/metric-valid counts were 810,398 / 813,291, left/right. The `cpu_fixture` selection counts in the tiny log are preliminary legacy controls, not counts from the fused GPU fixture. Metric conversion uses the scoped integer-disparity LUT and source ID 7.

The pinned supports and intervals yield the following CPU-derived counts over all four paths. These count logical predicate-qualified visits; they are not measured GPU instructions, memory transactions, bandwidth or cache traffic.

| Work estimate | Left | Right |
|---|---:|---:|
| Active band candidate visits | 206,615,684 | 205,233,948 |
| In-bounds candidate visits | 193,859,988 | 193,129,668 |
| In-bounds with own support | 190,626,820 | 189,896,500 |
| Both supports valid: descriptor-pair XOR/popcount | 190,319,284 | 189,599,732 |

The 720p candidate requested 422,338,900 GPU buffer bytes versus 656,178,404 for each control. These are requested buffer sizes, not measured peak usage or total driver allocation.

## Matched timing

One nonblocking shared GPU lock covered parallel/fused/parallel blocks. Each used the same pinned preuploaded frame, one in-flight slot, 10 warmups and 100 samples; no full-frame readback occurred during timing. Timing was bound to independently checked parity proofs. GPU timestamps had 64 valid bits and a 10.019 ns period, with availability and wrap checks. Values are milliseconds; percentiles use NumPy linear interpolation. CPU wait overlaps GPU execution and must not be added to GPU time. Per-frame duration samples are retained; raw timestamp ticks are not.

| Block | GPU p50 | GPU p95 | GPU p99 | GPU max | CPU record p50 | CPU submit p50 | CPU wait p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Parallel 1 | 93.061 | 96.040 | 96.397 | 96.469 | 0.593 | 0.120 | 93.236 |
| Fused | 66.171 | 67.947 | 68.045 | 68.113 | 0.594 | 0.126 | 66.355 |
| Parallel 2 | 90.471 | 92.865 | 93.572 | 93.911 | 0.533 | 0.118 | 90.660 |

Parity and all three timing blocks exited 0 on AMD Radeon 8060S / RADV with Khronos instance/device validation and fixture counters checked through teardown. Synchronization validation was requested but not independently attested; loader notices are distinct from fixture diagnostics. Before and after hashes of source, binaries, SPIR-V, input manifest, baseline and prior proof matched. The 301 CPU/identity and guard tests passed before execution.

## Identities

| Artifact | SHA-256 |
|---|---|
| Fused shader source | `343f35c04a647007eeaa53d00726c66e34aff9e5a669ba13b7ef67828a57014b` |
| 720p host source / binary | `4c8f0a28083cf6ec26d01886389f26d28dddf718a4cce0ed0d5c49a2fb4773bf` / `c3f291df3fe680429d8d0b577b35f1e270d2d55fa42515f481139b43eeaf6a5e` |
| Tiny host source / binary | `39d314924cca25acd233ce7d79f394d428fe972cfe1092a61265e2b780796c27` / `7235814fc5fa23a0f7f145a83a881a1d63084a0c86036131bc28c56a77589378` |
| Diagnostic / matched harness | `02f8bc2bdbbb56a5965280a3e5c398c703574b0b68c48e1960cfbd030d39750e` / `5cd9f57cd6e5dc7720b63462660d4229b47ff8f9151dcfd414a5222a9bc6e607` |
| Parallel / fused parity proof | `ce5610937f578f7e7a15c302374bb5efa69687f041739f282f0b63cfe5f07c18` / `513b83b4bffc8fd7df3992c5ea322237417da1a79b0b3d0bc9a7f3d32fbb685d` |
| Matched summary | `186ea7202f276c85e3ef36b0b034e1732eacdd8e16029ed0094f19256963c6ba` |
| Parallel 1 / fused / parallel 2 samples | `1a287c3b92db123219bb44bb8e5e76ba9638a88702df4fd949f2061c925802ba` / `d320112caf8268d59252a88dd3d83dfa651b678faec56b5022de485b48805e85` / `ec12c532e3471dc0bbab2502e4d328b77d1c64a286df1497f9d609c8225ffbb1` |

The ignored local directories `artifacts/fp2-fused-20261029-v1/` and `artifacts/fp2-fused-matched-20261029-v1/` retain logs, exits, proofs, identities, results and duration samples; the pinned input payload is separately at `artifacts/fp2-720p-cpu-20261028-v1/prepared`. They may be absent in a fresh checkout. The prepared manifest SHA is `7f70c357ec371d5c434c435e4a2500d87a149e43a5859e1695ada38cf021ce17`; the baseline result SHA is `e690b1895b63cf2c9c158419bf0662e7e316426d1d3c164f8c078d3700090041`.

## Reproduction

Use the configured Daxa build, validation layer and pinned local artifacts described in [parallel evidence](fp2-parallel-evidence.md). Run all CPU and hotpath guards before GPU work:

```bash
TMPDIR=/dev/shm cmake --build build/daxa --target fp2_parallel_fixture fp2_parallel_tiny fp2_fused_fixture fp2_fused_tiny -j 4
TMPDIR=/dev/shm PYTHONPATH=python:. .venv/bin/python -m pytest -q \
  tests/test_fp2_fused_cpu.py tests/test_fp2_fused_hotpath.py tests/test_fp2_fused_matched.py \
  tests/test_fp2_parallel_cpu.py tests/test_fp2_parallel_hotpath.py \
  tests/test_fp2_resident_cpu.py tests/test_fp2_resident_hotpath.py \
  tests/test_fp2_integrated_cpu.py tests/test_fp2_integrated_hotpath.py \
  tests/test_fp2_metric_lut_cpu.py tests/test_fp2_selection_cpu.py tests/test_fp2_selection_hotpath.py \
  tests/test_fp2_invalid_index_hotpath.py tests/test_fp2_720p_prepare.py \
  tests/test_fp2_720p_diagnostic.py tests/test_fp2_720p_hotpath.py \
  tests/test_fp2_census_cpu.py tests/test_fp2_census_hotpath.py \
  tests/test_fp2_image_cpu.py tests/test_fp2_image_hotpath.py \
  tests/test_fp2_slots_cpu.py tests/test_fp2_slots_hotpath.py \
  tests/test_fp2_prototype_cpu.py tests/test_fp2_dispatch_hotpath.py
```

For fresh fused parity use the locked command from [resident evidence](fp2-resident-evidence.md), substituting `benchmarks/fp2_fused_diagnostic.py`, `build/daxa/fp2_fused_fixture` and `build/daxa/fp2_fused_tiny`; use a fresh output directory and retain logs, exit and before/after identities. Stop if parity fails. Then run the proof-bound matched comparison:

```bash
proof=/path/to/fresh/fused-parity/result.json
proof_sha=$(sha256sum "$proof" | cut -d ' ' -f 1)
TMPDIR=/dev/shm PYTHONPATH=python:. .venv/bin/python benchmarks/fp2_fused_matched.py \
  --output-root artifacts/fp2-fused-matched-new \
  --new-proof "$proof" --new-proof-sha256 "$proof_sha"
```

The harness binds the existing parallel proof, candidate proof, binaries, pinned inputs and baseline; sets the AMD ICD and Khronos layer; requests synchronization validation; takes `/tmp/stereoforge-gpu.lock` nonblocking; and uses a 300-second process-group watchdog per block. Retain actual exits and inspect device and validation logs. Sync activation must be independently established before claiming synchronization validation.
