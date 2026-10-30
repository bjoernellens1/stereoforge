# Optional GPU duty budget (experimental)

The native ROS publisher offers an optional **best-effort self-budget** so stereo
processing can leave execution time for SLAM, neural inference, or rendering.
This is not an OS GPU scheduler, hardware reservation, hard real-time guarantee,
or proof that memory bandwidth/compute units are actually available to other apps.

## CLI

- \`--gpu-duty-percent 100\` (default): no GPU work admission cap.
- \`--gpu-duty-percent 25\`: aim to consume at most 25% of source-time in
  **measured/estimated GPU stereo work**. Normal depth rate and quality are not changed.
- \`--gpu-estimate-us 8000\`: initial estimated GPU service cost per admitted frame.
- \`--gpu-burst-ms 33\`: maximum token bucket credit. Allows short response bursts.

\`\`\`sh
# Add to existing publisher arguments:
--capture-fps 60 --depth-rate adaptive \
--consumer-min-depth-fps 5 --gpu-duty-percent 25 \
--gpu-estimate-us 8000 --gpu-burst-ms 33 \
--rate-report depth-rate.json
\`\`\`

A configuration fails at startup if the requested static consumer minimum cannot
fit within the initial GPU-time estimate and configured duty percent. It cannot
guarantee that floor in the presence of GPU contention, camera issues, or estimator
cost increases. All three budget CLI values have finite integer bounds.

## Execution

The budget uses a **bounded three-frame allocation-free token bucket**, replenished
with source timestamp increments. Work is charged *when admitted*, so overlapping
frames do not spend the same credit twice. When completed, the reserved cost is
reconciled with measured GPU service time (if a complete timing span is available).
The next-frame cost estimate is an eighth-step exponential smoothing of measured
cost, while actual accounting uses the unsmoothed measurement. Unexpectedly expensive
frames create debt that future source-time replenishment repays.

Only frames which are already due under \`DepthRateController\` consult the budget.
Rejected frames do not reach Daxa/Slang GPU processing, and the next fresh source
frame is reconsidered. A budget skip is not a capture-frame loss, a GPU slot
rejection, or a successful depth observation.

### GPU measurement boundary

- The independent full-resolution SGM path can report **source upload through
  final metric metadata GPU queue span**. The budget reconciles that GPU span.
- The experimental coarse/refinement path currently has **separately measured
  producers**, so a whole-queue source-to-final-refinement GPU span is not
  available to the ROS path. For that path the budget retains the configured
  estimate, increments \`missing_timings\`, and does **not** invent a measurement.
- Neither path accounts for host export, display, camera USB transport,
  ROS publication, GPU commands from unrelated processes, or physical bandwidth
  contention in this controller.

These limitations mean the percent is an **admission target**, not a certified
physical GPU utilization cap. Under external GPU pressure, monitor actual
latency and competing workload throughput to validate whether headroom is real.

## Report and verification

\`--rate-report\` adds \`budget_rejections\` and a \`gpu_budget\` object containing
percent, estimates, burst capacity, reservations, denials, completions,
missing timing count, charged GPU nanoseconds, final estimated frame cost,
maximum debt and explicit timing scope. Report written only after the loop.

Portable tests: \`cmake --preset cpu && cmake --build --preset cpu &&
ctest --preset cpu -R '(depth_rate|gpu_duty)_controller|gpu_duty_budget' --output-on-failure\`.

Before considering production use, test source/replay rate accuracy, two frames
in flight, renderer+VLM interference, p99/max end-to-end depth age, memory pressure
and sustained 600 s delivery on the actual AMD workstation. The proposed
co-running performance criterion from design discussions is **>=80% background
throughput retention**, not a measured fact or guarantee.
