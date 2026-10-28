# Connected ZED 2: live ROS/Zenoh smoke test

On 2026-10-28, a short physical-camera run delivered **518 unique depth frames and 518 unique rectified RGB frames**, each at **34.53 Hz** over the common 15.0001-second window. The ZED 2 was attached on a 5 Gb/s USB link and configured for 1280×720 per eye at nominal 60 Hz. This is an operational smoke test, not a 60 Hz qualification or a depth-accuracy evaluation.

The run used the experimental `packed-specialized-overwrite` full-resolution D128, four-path independent stereo configuration on Radeon 8060S with Mesa RADV 26.2.4. The depth and rectified RGB planes were exported to host memory, then sent through ROS Jazzy `rmw_zenoh_cpp` to a separate subscriber. QoS was best-effort, volatile, keep-last one. Both processes reported the Zenoh RMW and exited successfully. The previous binary's build commit was not attested, so this run cannot establish which current source revision it represents or validate the newer refinement borrowing path. Vulkan validation was not requested for this run.

| Measurement | Depth | Rectified RGB |
| --- | ---: | ---: |
| Published and matched unique deliveries | 518 | 518 |
| Delivery rate | 34.53 Hz | 34.53 Hz |
| Host ingress to arrival p50 | 51.57 ms | 52.50 ms |
| Host ingress to arrival p99 | 67.58 ms | 68.41 ms |
| Arrival gap p99 | 33.01 ms | 30.83 ms |

The capture counters recorded 896 source frames and 895 delivered to the publisher's capture reader. The publisher read 895, admitted 520, recorded 375 ring drops and published 518 of each output within the common window. The capture queue reported three drops. The subscriber's total counters were 520 per output, including deliveries outside the common window; the table uses the 518 matched in-window deliveries. Capture, publisher and subscriber counters have different endpoints and must not be treated as a single closed loss ledger.

Host ingress is after capture and host copying. The camera-to-host clock mapping was unverified, so exposure-to-consumer age is unavailable; the host-ingress timings are not camera-to-consumer latency. The summary also flags a missing frozen delivery policy and incomplete requested-storage accounting. A 15-second run does not establish sustained operation, frame-age bounds, metric depth quality, or the planned ten-minute 60 Hz gate.

Separately, a raw V4L2 capture-only check returned 600 buffers in approximately 10.32 seconds. This is not a unique-frame analysis or a formally timed end-to-end throughput measurement, and does not measure stereo processing or ROS delivery.

An initial container attempt exited with code 139 before capture when host `/usr/lib64` libraries were mixed into the Ubuntu container. The successful retry used curated Ubuntu DRM libraries. The raw traces, logs and generated summary are local ignored artifacts under `artifacts/live-connected-20261028-v2/` and include private camera metadata; they are not part of this report.
