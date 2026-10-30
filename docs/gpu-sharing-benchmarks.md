# Shared-GPU performance measurement

This is an **offline analyzer**, not a benchmark runner or claim of available GPU compute. A separate workload producer must measure every frame and background-work interval and write JSONL samples.

Use `benchmarks/gpu_sharing_summary.py`:

```bash
python benchmarks/gpu_sharing_summary.py samples.jsonl --output new-summary.json
pytest -q tests/test_gpu_sharing_summary.py
```

Each JSONL record contains `mode` (e.g. `background_only`, `stereo_30`, `stereo_60`), a distinct `frame_identity` within that mode, and measured `latency_ms`. Optional `background_work_units` and `background_elapsed_ms` must be supplied together; they represent observations of the competing workload. Reported background throughput is total measured work divided by total measured observation time, **not** a mean of instantaneous rates. The `background_only` mode is the retention denominator.

The report computes latency p50/p95/p99/max and background-throughput retention. It does **not** infer GPU occupancy, GPU bandwidth, stereo image quality, power, sustained camera FPS or a shared reference clock. Do not mix different input resolutions, GPU clocks, hardware, scenes or background workloads into a comparison group. Sampling methodology and calibration must be recorded externally.

Recommended experimental matrix: 5/10/20/30/60 Hz stereo, unthrottled single-frame stereo, and background-only control. Run ABBA interleaved trials with a Vulkan graphics workload and a separate ML inference workload. Record quality independently, plus capture frame identities, arrival gaps and original camera clock domain. Targets such as 80% retained background throughput remain **targets**, not proof of performance.
