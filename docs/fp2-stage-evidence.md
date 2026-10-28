# FP2 resident stage profiling

The isolated, no-path profiler passed its GPU parity gate before timing on AMD Radeon 8060S / RADV. The tiny fixture checked both eyes' complete sums and padding against the CPU oracle; the pinned analytic 1280×720 fixture checked GPU raw and observation outputs against the pinned CPU selections and earlier GPU output hashes. This is synthetic data, not camera or production qualification. Source ID was 7; the host requested 656,178,404 bytes of GPU buffers, excluding driver overhead and CPU storage, not a measured peak.

Both parity and timing acquired `/tmp/stereoforge-gpu.lock` independently, completed with exit 0, and checked validation counters through teardown. Khronos validation was enabled; synchronization validation was requested but not independently attested. Loader layer notices are separate from fixture diagnostics. Source, binary, SPIR-V and pinned manifest/baseline hashes matched before and after. The 254-test CPU and guard suite passed before the runs.

The timed run used one persistent resource set and one in-flight slot, with one repeated preuploaded analytic frame. There were 10 balanced warmups, then 25 ABBA blocks (50 control and 50 instrumented samples). Both paths use the same dispatches and added boundary barriers; only the instrumented path adds stage timestamp annotations. Consequently, the control is **not** the original uninstrumented resident baseline. Each timed frame waited for its own completion and read timestamp-query results; no full frame was read back in the timed loop. Validation-enabled shared-GPU results are not non-validation release timing. Timestamp query availability, valid bits and wrap bounds were checked; 64 valid bits and 10.019 ns period were reported. Per-sample durations are retained, but raw timestamp ticks are not.

Values are milliseconds, with NumPy linear-interpolated percentiles and 50 samples per group. Stage values are ordered, barrier-inclusive queue intervals, including timestamp effects; they are not isolated kernel durations. Percentiles from different rows must not be summed.

| Interval | p50 | p95 | p99 | Max |
|---|---:|---:|---:|---:|
| Whole chain, control | 839.118 | 853.836 | 854.642 | 855.228 |
| Whole chain, instrumented | 838.731 | 847.435 | 849.849 | 851.049 |
| Census left | 0.108 | 0.113 | 0.115 | 0.115 |
| Census right | 0.103 | 0.107 | 0.108 | 0.109 |
| Left reset | 2.257 | 2.379 | 2.643 | 2.687 |
| Left cost | 8.583 | 9.968 | 9.988 | 9.992 |
| Left path 0 | 88.990 | 90.517 | 91.228 | 91.424 |
| Left path 1 | 132.675 | 135.766 | 138.252 | 139.311 |
| Left path 2 | 90.836 | 93.727 | 94.120 | 94.343 |
| Left path 3 | 90.761 | 93.129 | 94.219 | 94.746 |
| Left selection | 3.995 | 4.318 | 4.356 | 4.377 |
| Right reset | 2.048 | 2.216 | 2.272 | 2.283 |
| Right cost | 8.600 | 10.080 | 10.919 | 11.170 |
| Right path 0 | 89.140 | 90.417 | 90.682 | 90.764 |
| Right path 1 | 132.313 | 135.381 | 136.404 | 136.887 |
| Right path 2 | 89.119 | 91.893 | 93.044 | 93.672 |
| Right path 3 | 89.312 | 92.488 | 94.447 | 94.652 |
| Right selection | 4.178 | 4.533 | 4.587 | 4.615 |
| LR, both eyes | 0.731 | 0.747 | 0.763 | 0.769 |
| Metric, both eyes | 0.886 | 0.903 | 0.925 | 0.938 |

The matched groups show no positive timestamp overhead estimate at the median in this run; that is not proof of zero overhead. Recurrence intervals dominate this diagnostic execution. These results do not establish live throughput, transfer-inclusive latency, stereo quality, or a speedup against another implementation.

Artifacts are retained locally in the ignored `artifacts/fp2-stage-20261029-v1/` directory:

| Artifact | SHA-256 |
|---|---|
| Parity proof | `dd58b130ae938c1ef76b139921c7be2c14a4f6540b15d0c41802f349e03b8289` |
| Parity log | `b08262ac50545f2aa6a3e9c4cfc498683ac2a38cc48f82caf719f07e938d728e` |
| Timing result | `aec02628158b1268b95d466f70bb7fb3119dfb210d8f1bab873aaebe718b0b38` |
| Timing log | `09d347fb42b140a1fd3c0c245ef5505f06b6ac461f594be82f78c0de925bf481` |
| Per-sample durations | `2084d89a67c9e7b2b1c39ab53c3f2eaa2c336327a34e70e4e85859e96a696985` |

The directory also retains exact pre/post identity lists and exit statuses. It may be absent from a fresh checkout.

## Reproduction

Use the same configured build, validation installation and pinned artifacts as in [resident evidence](fp2-resident-evidence.md). Run the CPU suite listed there plus `tests/test_fp2_stage_profile_cpu.py`, `tests/test_fp2_stage_timestamps.py`, and `tests/test_fp2_stage_hotpath.py`. Build `fp2_stage_profile` and `fp2_resident_tiny`; retain hashes and logs before and after. Set `common` and define `run_gpu` as in the resident evidence, substituting `benchmarks/fp2_stage_profile.py` in the function and `build/daxa/fp2_stage_profile` for `--host`. The common arguments bind the pinned manifest and baseline. Execute each command in a shell that stops on failure, with fresh output paths:

```bash
cmake --build build/daxa --target fp2_stage_profile fp2_resident_tiny -j 4
run_gpu "${common[@]}" --output-root /tmp/fp2-stage-parity-new --parity
proof=/tmp/fp2-stage-parity-new/result.json
proof_sha=$(sha256sum "$proof" | cut -d ' ' -f 1)
run_gpu "${common[@]}" --output-root /tmp/fp2-stage-timing-new \
  --time --proof "$proof" --proof-sha256 "$proof_sha"
sha256sum "$proof" /tmp/fp2-stage-timing-new/result.json \
  /tmp/fp2-stage-timing-new/samples.csv
```

The separate nonblocking lock and 900-second watchdog in the referenced function are conservative reproduction bounds; the recorded runs used a 300-second bound. Check device and validation logs and retain actual exit codes. Parity proof binds the profiler source and binary as well as input and baseline identities; timing fails if they differ.
