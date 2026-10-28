# FP2 disparity-parallel experiment

The isolated 128-lane scanline shader passed scoped parity and was faster than the serial no-path diagnostic on the same pinned synthetic 720p input. In a locked old/new/old test, GPU queue timestamp p50 was 835.287 / 90.246 / 834.285 ms respectively, approximately 9.26× and 9.24× relative to the two serial controls. The parallel result still exceeds the 16.67 ms frame interval; this is not live camera throughput or production pipeline qualification.

Each scanline workgroup uses 128 lanes, shared predecessor/current states, and a tree reduction at every pixel, including lanes outside the compact band. This experiment is **not work-proportional compact recurrence**. It preserves the tested integer candidate semantics, not a claim of full-resolution unrestricted SGM, subpixel/native baseline equivalence, physical depth accuracy, or quality preservation on real scenes.

## Parity

Before timing, the tiny GPU fixture tested 7×5 and 320×180 synthetic inputs for both eyes. It compared all four packed path arrays, full sums and padding against an independent dense CPU recurrence. The 720p pinned analytic test compared final GPU raw selections and observations against frozen CPU selections and prior integrated GPU hashes; it did **not** read back and compare 720p per-path arrays. The 720p raw valid counts were 897,952 / 898,067 and LR/metric-valid counts were 810,398 / 813,291, left/right. Calibration-derived metric values use the scoped integer-disparity LUT described in [resident evidence](fp2-resident-evidence.md). Source ID was 7.

Both shaders and fixtures ran with the Khronos layer on AMD Radeon 8060S / RADV. Parity acquired the nonblocking shared GPU lock, exited 0, and checked fixture validation counters through teardown. Synchronization validation was requested but not independently attested; loader notices are separate from fixture diagnostics. Before/after source, binary, SPIR-V, manifest and baseline hashes matched. The 720p host requested 656,178,404 GPU buffer bytes; the 320×180 diagnostic with path readback requested 170,463,156 bytes. These are requested buffer sizes, not measured peak usage or total driver allocation.

## Matched timing

The comparison used the same pinned input and output chain, one persistent resource set and one in-flight slot per block. Each block used 10 warmups and 100 samples of the same preuploaded frame. No full image was read back during timed iterations. One lock covered all three blocks; each child had a 300-second process-group watchdog and all exited 0. NumPy linear interpolation was used for p50/p95/p99. GPU timestamps have a 10.019 ns period and 64 valid bits, with availability and wrap checks. Raw per-frame durations are retained; raw timestamp ticks are not. Each timing result was bound to its independently validated parity proof. The 255 CPU/guard tests passed before timing after redirecting compiler temporary files from `/tmp` due to user quota.

| Block | GPU p50 | GPU p95 | GPU p99 | GPU max | CPU record p50 | CPU submit p50 | CPU wait p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Old serial 1 | 835.287 | 850.541 | 855.211 | 862.033 | 0.735 | 0.128 | 835.732 |
| New parallel | 90.246 | 94.031 | 95.000 | 95.016 | 0.620 | 0.142 | 90.470 |
| Old serial 2 | 834.285 | 850.535 | 855.237 | 862.507 | 0.715 | 0.141 | 834.601 |

All values are milliseconds. CPU wait overlaps GPU execution and must not be added to GPU time. This measures the isolated repeated-input workload, not capture, transfers, frame age, delivered FPS, or a match against the established full SGM quality baseline.

## Identities and artifacts

| Item | SHA-256 |
|---|---|
| Prepared input manifest | `7f70c357ec371d5c434c435e4a2500d87a149e43a5859e1695ada38cf021ce17` |
| Integrated baseline result | `e690b1895b63cf2c9c158419bf0662e7e316426d1d3c164f8c078d3700090041` |
| Parallel shader source | `d510bab2ad4f3590f905bd88e1575efd100788e43ba658a6f7aed2379b52b641` |
| Parallel 720p host source / binary | `08e965b4bda98d8a0e010a5848fd3071007b6d016c5c653302ebdd29869fcb37` / `cc0ebe3740bbb5a83a03f642a86d517d8004e8f6af9fd0f40646598214bd7ab0` |
| Parallel tiny host source / binary | `6288160a7eaebf55ca7a91f9b421d4ff03ed3293bc68a1cf015023713d63da7f` / `bc142703ee64535fb489ea9473741bb2213b9aa20209258a0c1110392caadcf8` |
| Serial / parallel parity proof | `91a53952841c9eeab9ba03a8ec75b5b2c73b582ba26ca1d1a569e35565796bc0` / `ce5610937f578f7e7a15c302374bb5efa69687f041739f282f0b63cfe5f07c18` |
| Matched harness | `cab51cf2737a4c0eb978b65263b1936387d23b60183120cbd27af79557de3e01` |
| Matched summary | `330fd045905174544ceddba9436f01754d54e4f0282b83b14ed72a0e8698217c` |

The ignored local directories `artifacts/fp2-parallel-20261029-v1/` and `artifacts/fp2-matched-20261029-v1/` retain pre/post hashes, proof, per-block logs, exits, identities, raw sample durations and results; they may be absent in a fresh checkout. The proof identities include the SPIR-V and transitive sources.

## Reproduction

The reproduction requires the pinned local artifacts and the configured Daxa build and validation layer. Build all four targets, then run the CPU suite in [resident evidence](fp2-resident-evidence.md), adding `tests/test_fp2_parallel_cpu.py`, `tests/test_fp2_parallel_hotpath.py`, `tests/test_fp2_matched_timing.py`, and `tests/test_fp2_matched_timing_hotpath.py`. The six focused parallel and resident suites passed 70 tests when rechecked. Configure `TMPDIR` and pytest's `--basetemp` on a filesystem with sufficient quota if needed.

```bash
cmake --build build/daxa --target fp2_resident_fixture fp2_resident_tiny fp2_parallel_fixture fp2_parallel_tiny -j 4
PYTHONPATH=python:. .venv/bin/python -m pytest -q tests/test_fp2_parallel_cpu.py tests/test_fp2_parallel_hotpath.py tests/test_fp2_matched_timing.py tests/test_fp2_matched_timing_hotpath.py
PYTHONPATH=python:. .venv/bin/python benchmarks/fp2_matched_timing.py \
  --output-root artifacts/fp2-matched-new-output
```

The harness requires the exact retained serial and parallel parity proof hashes above and validates their source, binary, prepared input and baseline bindings before each timed block. It sets the Radeon ICD and validation-layer paths, requests synchronization validation, takes `/tmp/stereoforge-gpu.lock` nonblocking, and rejects an existing output directory. For a new parity run, follow the locked command in [resident evidence](fp2-resident-evidence.md), substituting `benchmarks/fp2_parallel_diagnostic.py`, `build/daxa/fp2_parallel_fixture`, and `build/daxa/fp2_parallel_tiny`; use a fresh parity output directory. A newly generated proof has to be independently reviewed and bound in the comparison harness before it can replace the pinned proof. Preserve logs, exit statuses and before/after identities; do not run timing after a parity failure.
