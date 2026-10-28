# Live ZED 2 ROS Zenoh diagnostic, 2026-10-28

This is one bounded real-camera smoke test, not a 60 FPS qualification or an
accuracy test. It used a ZED 2 at 1280 × 720 per eye and 60 Hz nominal capture,
full-resolution D128 independent left/right four-path SGM, and rectified RGB
and axial float32 depth delivered to separate ROS 2 image subscriptions. The
publisher performs host export before publishing; this test is not the
same-process GPU handoff.

The fixture was generated from eight physical packed frames and the associated
factory calibration. The prior capture report identifies the ZED serial and
records eight unique frames, while the live publisher verifies the same device
identity and calibration association. The fixture and camera calibration are
private temporary data under `/dev/shm/stereoforge-live-20261028/fixture` and
may disappear on reboot. They are not included in this repository.

The ROS Jazzy disposable container installed `rmw_zenoh_cpp` 0.2.11. Publisher
and subscriber each reported `ROS middleware: rmw_zenoh_cpp`; a local router
was running. The session used the default non-SHM settings; no Zenoh shared
memory transfer is claimed. QoS was best-effort, volatile, keep-last one.
The GPU was Radeon 8060S, Mesa RADV 26.2.4, exposed through
`/dev/dri/renderD128`. The publisher requested Vulkan validation and the
container set the synchronization-validation feature; publisher and subscriber
logs contain no validation messages. The logs do not separately attest layer
loading. Both executables exited 0.

The publisher was configured for 25 seconds after five warmup frames, two
pipeline slots, and an eight-second bounded discovery wait. The subscriber was
configured for 45 seconds. The common measurement window was 25.0007 seconds;
statistics below use matching fresh source identities within that window.

| Measure | Depth | Rectified RGB |
| --- | ---: | ---: |
| Published / matched received, common window | 780 / 780 | 780 / 780 |
| Delivered rate | 31.20 Hz | 31.20 Hz |
| Host ingress to arrival p50 | 57.55 ms | 58.37 ms |
| Host ingress to arrival p95 | 64.95 ms | 65.75 ms |
| Host ingress to arrival p99 | 66.25 ms | 66.98 ms |
| Publish call wall time p50 | 1.00 ms | 0.64 ms |
| Arrival gap p99 | 33.60 ms | 33.50 ms |

The publisher reported 1,491 source frames over its counter interval and 711
ring drops; the summary's matched completeness is 780 / 1,491, a 47.69% drop
fraction for each output. Subscriber totals were 782 per output, including
messages received outside the common window during drain; these do not increase
the 25-second matched count. Queue growth, warmup and endpoint differences
mean source, read, admitted and published totals should not be treated as
interchangeable.

Host ingress is recorded after the camera frame has been copied into host
memory; it is not an exposure timestamp. The supplied camera-to-ROS clock map
was an **unverified diagnostic identity mapping**. Exposure-to-output age is
unavailable, and the reported host-ingress latency must not be described as
camera-to-consumer latency. The summary is unqualified because camera clock
mapping was not verified, no frozen policy was applied, and total memory
accounting was incomplete. Requested tracked storage was 892,744,160 bytes,
not total process, driver or system memory.

Raw traces, logs, the summary and the container-side command script are saved
locally under `artifacts/zed-task7/live-zenoh-20261028/`; these artifacts are
ignored by Git. The script refers to the ephemeral fixture and configuration
paths used for this run. Camera identity and calibration are not included in
this document. Longer paced trials, exposure-clock validation, complete memory
accounting, depth quality and sustained consumer delivery remain open.
