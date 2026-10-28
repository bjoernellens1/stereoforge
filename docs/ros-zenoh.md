# ROS middleware and Zenoh

The native `stereoforge_ros2_depth_publisher`, `stereoforge_ros2_gpu_composed`,
 and `stereoforge_ros2_delivery_probe` executables select `rmw_zenoh_cpp`
when `RMW_IMPLEMENTATION` is **unset**. Selection happens before ROS
initialization, inside each process; it does not change other ROS applications.
Install the ROS-distribution-matched Zenoh RMW package and source your ROS
setup. Missing middleware is an initialization error, not an automatic switch
to another transport. If `RMW_IMPLEMENTATION` is set, its value is respected;
an invalid value is not silently replaced, and an explicitly empty value is rejected before ROS initialization.

For ROS 2 Jazzy's default Zenoh configuration, start one shared router and
leave it running independently of the applications:

```bash
source /opt/ros/jazzy/setup.bash
./scripts/stereoforge-ros router
```

In separate terminals, start StereoForge normally, or use the launcher, which
checks the chosen RMW package before executing a command:

```bash
source /opt/ros/jazzy/setup.bash
./scripts/stereoforge-ros run -- /path/to/stereoforge_ros2_depth_publisher [arguments]
./scripts/stereoforge-ros run -- /path/to/stereoforge_ros2_delivery_probe [arguments]
```

The upstream Jazzy router configuration listens on all network interfaces by
default. To restrict it to this host, set an override **only for the router**
before starting it, without changing the session settings:

```bash
ZENOH_CONFIG_OVERRIDE='listen/endpoints=["tcp/127.0.0.1:7447"]' ./scripts/stereoforge-ros router
```

If you already provide router configuration or overrides, incorporate that
endpoint in your configuration instead of overwriting your settings. Check
the installed release and your network/security requirements.

The router is a foreground process, so its lifecycle can be managed explicitly
by a service manager or terminal. The launcher neither starts duplicate routers
nor kills a router used by other applications. The router/session settings and
network visibility remain under the user's Zenoh configuration. Default Jazzy
node discovery needs the router; alternate topologies require matching node
configuration. Successful ROS initialization alone does not establish that a
router or subscriber was discovered; Jazzy may continue initialization when
its router check fails. The publisher offers `--subscriber-wait-seconds` to
wait a bounded period for subscribers, but this is optional and is not a
continuous discovery or delivery guarantee. Other ROS tools need the same RMW, for example
`RMW_IMPLEMENTATION=rmw_zenoh_cpp ros2 topic list`.

Select another installed middleware explicitly when required:

```bash
./scripts/stereoforge-ros run --rmw rmw_fastrtps_cpp -- /path/to/program [arguments]
# Or set RMW_IMPLEMENTATION=rmw_fastrtps_cpp when starting the executable directly.
```

All communicating ROS processes need compatible RMW and transport settings;
this launcher is not a DDS-to-Zenoh bridge. Zenoh is the default based on the
initial short transport experiment, not a proven sustained-performance win.
See [ROS transport evidence](ros-transport-evidence.md) for the measurements
and limitations. The existing sensor publishers use best-effort keep-last-one
QoS to avoid stale frame queues; changing reliability or history changes
delivery semantics and should be benchmarked with the intended consumers.

Zenoh shared memory is **not forced on**: availability, configuration, pool
size, container IPC namespace, `/dev/shm` capacity, and locked-memory limits
must be verified for the deployed package. It may fall back to other transport.
See the [Jazzy rmw_zenoh configuration and shared-memory guide](https://github.com/ros2/rmw_zenoh/blob/jazzy/README.md#zenoh-shared-memory).
Shared host memory is not a Vulkan GPU buffer handoff. For the experimental
same-process GPU path, see [ROS GPU handoff](ros-gpu-composed.md). Compare
latency, delivered unique frames, CPU use, memory, queue growth and frame age
on representative paced, sustained workloads before claiming an improvement.

### Experimental shared-memory configuration (Jazzy 0.2.11)

The short diagnostic used `ros-jazzy-rmw-zenoh-cpp` 0.2.11 with shared memory
explicitly enabled in both router and session configuration and a 6 MiB pool.
The following reproduces those settings from that package's default files;
the assertions intentionally stop if the installed configuration differs. The
router remains loopback-only. Run this in a disposable directory, then export
both paths in every relevant terminal before starting its router or nodes:

```bash
mkdir -p "$HOME/stereoforge-zenoh-test"
python3 - <<'PY'
from pathlib import Path
root = Path('/opt/ros/jazzy/share/rmw_zenoh_cpp/config')
out = Path.home() / 'stereoforge-zenoh-test'
for kind in ('ROUTER', 'SESSION'):
    s = (root / f'DEFAULT_RMW_ZENOH_{kind}_CONFIG.json5').read_text()
    old = '/// ROS setting: disabled by default until fully tested\n      enabled: false,'
    assert s.count(old) == 1 and s.count('pool_size: 50331648') == 1
    s = s.replace(old, '/// Explicit diagnostic setting\n      enabled: true,')
    s = s.replace('pool_size: 50331648', 'pool_size: 6291456')
    if kind == 'ROUTER':
        assert s.count('"tcp/[::]:7447"') == 1
        s = s.replace('"tcp/[::]:7447"', '"tcp/127.0.0.1:7447"')
    (out / f'{kind.lower()}.json5').write_text(s)
PY
export ZENOH_ROUTER_CONFIG_URI="$HOME/stereoforge-zenoh-test/router.json5"
export ZENOH_SESSION_CONFIG_URI="$HOME/stereoforge-zenoh-test/session.json5"
```

The diagnostic mapped a publisher-owned 6,291,456-byte segment in the
subscriber, with a 3,686,464-byte publisher payload allocation, and delivered
10/10 depth and 10/10 RGB messages. A 6 MiB pool
cannot hold two 720p four-byte images simultaneously; retries and garbage
collection appeared in logs. This proves a shared segment was mapped in that
short test, not that every message used shared memory or that this pool is
suitable for sustained delivery. Size each process's pool for actual payloads
and in-flight messages; confirm sufficient `/dev/shm` and `ulimit -l` for all
processes, plus a shared IPC namespace for containers. Set
`RUST_LOG=info,zenoh_transport::shm=trace,zenoh_shm=trace` for diagnostics and
inspect both ends for segment allocation and mapping, errors and fallback.
Detailed trace logging itself changes timing and should be disabled for the
performance run. Memory transport can silently fall back, so configuration
alone does not prove activation or a speedup.
