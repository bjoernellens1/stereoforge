# ROS transport diagnostic, 2026-10-28

These are exploratory synthetic results, not a sustained live-camera or 60 Hz
qualification. Both runs used the same 1280x720, D128 replay, ten unpaced
frames, separate publisher and subscriber, best-effort QoS with depth one,
and Vulkan validation. Each depth or RGBA image carries 3,686,400 bytes. The
publisher exported images to host memory before ROS publication. ROS Jazzy
used Fast DDS in one run and `rmw_zenoh_cpp` 0.2.11 in the other; Zenoh was
installed in a disposable container, not on the host.

| Transport | Depth received | RGB received | Mean depth publish wall time | Mean RGB publish wall time |
| --- | ---: | ---: | ---: | ---: |
| Fast DDS | 1/10 | 7/10 | 24.06 ms | 0.52 ms |
| Zenoh, default configuration | 10/10 | 10/10 | 1.29 ms | 0.92 ms |

The first Fast DDS depth publish took 235.17 ms wall time and 1.38 ms of
publishing-thread CPU time; the mean is dominated by that call. A separate
64x32 Fast DDS run recorded 313.83 ms wall time and 0.152 ms of thread CPU
time on its first depth publish. The discrepancy indicates waiting or
descheduling during the call, but does not identify its cause. Ten unpaced
frames and keep-last-one QoS are insufficient to rank transports generally;
the synthetic clock also cannot establish frame age.

Zenoh shared-memory transport was not established in the initial runs. In a separate diagnostic,
increasing `/dev/shm` to 512 MiB did not fix allocation: Zenoh attempted to
lock 48 MiB while the container's locked-memory limit was 8 MiB. Do not
interpret the default Zenoh result as a verified shared-memory result.
Even verified Zenoh shared memory would not by itself transfer a Vulkan GPU
resource. The experimental same-process GPU handoff is described in
[ros-gpu-composed.md](ros-gpu-composed.md).

Raw publisher and subscriber traces are under
`artifacts/zed-task7/synthetic-720/compare-{fast,zenoh}-20261028-v2-{pub,sub}.json`;
the separate thread-CPU trace is
`artifacts/zed-task7/synthetic-ros-replay/publisher-smoke11.json`. These local
artifacts are not committed. Next, verify shared-memory activation and compare
both transports with paced, longer runs, matched configuration, delivery,
frame age, CPU cost, and validated real capture timestamps.

A follow-up `rmw_zenoh_cpp` 0.2.11 diagnostic explicitly enabled shared
memory with a 6 MiB pool. Logs show a 3,686,464-byte publisher payload allocation and its
6,291,456-byte segment opened by the subscriber; it received 10/10 depth and
10/10 RGB messages. The short, unpaced run does not establish that every
message used shared memory, nor sustained throughput. A 7 MiB pool failed to
allocate under the container's 8 MiB locked-memory limit; the 6 MiB pool is
also too small for two concurrent 720p four-byte images and logged retries and
garbage collection. These settings are a diagnostic, not a production pool
recommendation. Raw local ignored artifacts have the prefix
`artifacts/zed-task7/synthetic-720/ourcmp5-zenoh-shm-20261028-v5` (with
router log `ourcmp5-zenoh-shm-router.log`); see
[experimental setup](ros-zenoh.md#experimental-shared-memory-configuration-jazzy-0211).
