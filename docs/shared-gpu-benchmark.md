# Concurrent-GPU ABBA experiment runner

`benchmarks/shared_gpu_benchmark.py` executes real benchmark programs to
measure how much a concurrent GPU workload degrades StereoForge service and
how much capacity StereoForge leaves to a renderer, SLAM mapper or model.

Both programs must be finite, explicitly chosen commands that write one JSON
report to the path supplied in their argv. **No shell commands are evaluated.**
Run on a controlled GPU machine, freeze input and model/profile versions, and
avoid mixing source frames when comparing quality.

## Example configuration

Save as `shared-gpu.json`, replacing paths and key names with those **actually
emitted** by your built workload binaries:

```json
{
  "schema": "stereoforge.shared_gpu_benchmark.v1",
  "foreground": {
    "argv": [
      "/path/to/daxa_depth_benchmark",
      "--manifest", "/path/to/frozen/native-input.json",
      "--frames", "200", "--warmup", "20",
      "--output", "{report}"
    ],
    "metric_path": "raw_samples_ms.gpu_combined_queue_span_ms",
    "identity_path": "input_sha256"
  },
  "background": {
    "argv": [
      "/path/to/your_renderer_benchmark",
      "--duration", "15", "--output", "{report}"
    ],
    "metric_path": "throughput_fps"
  },
  "rounds": 3,
  "timeout_seconds": 120,
  "minimum_overlap_fraction": 0.80,
  "minimum_samples": 100
}
```

The example's `identity_path` and background report fields are **illustrative**.
Check the actual JSON field names before execution; no configuration value
fabricates an identity or metric.

```sh
uv run python -m benchmarks.shared_gpu_benchmark \
  --config ./shared-gpu.json --output ./artifacts/shared-gpu-abba-new
```

The destination must not exist. Each cycle executes foreground solo,
background solo and both processes concurrently. Order alternates by cycle
to expose thermal/performance drift. The background process launches first
in each concurrent pair. The runner checks process lifetime overlap, not
actual GPU-command overlap, and includes timestamps and SHA256 for reports.

The scalar background throughput must be measured over a comparable period in
each run. Foreground `metric_path` must resolve to a list of **individual GPU
frame times in milliseconds**, and background `metric_path` to a positive
throughput scalar. The runner makes no assumptions about GPU vendor or driver.

## Output and decision boundaries

`results.json` contains per-cycle timings and report identities plus:
- worst individual-run stereo p99 (solo and paired);
- background isolated and paired throughput medians and worst paired retention;
- process overlap fraction;
- fixed provisional gates: stereo p99 <= 15 ms, background throughput >= 80%;
- identity consistency if a report identity was actually supplied.

`performance_qualified` is true only when overlap, identity consistency and
the two provisional performance gates pass. This remains **performance-only**
qualification, not a guarantee of independent physical stereo accuracy or
camera exposure-to-ROS delivery age. The latter are explicitly unverified.
The runner does not infer GPU occupancy from wall time or from theoretical
idle-duty fractions. For meaningful research comparisons, also report GPU
clock, power, residency, memory bandwidth, distinct source IDs, p99 frame age,
driver versions and scene quality with independent truth.

## Tests

```sh
uv run pytest -q tests/test_shared_gpu_benchmark.py
```

The tests launch two **real subprocesses** using a synthetic CPU fixture, check
run overlap, report parsing, conservative qualification, identity requirements,
configuration errors and the no-overwrite policy. They do not claim GPU speed.
