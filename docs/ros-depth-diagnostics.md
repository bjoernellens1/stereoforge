# Measured depth diagnostics over ROS

The native publisher keeps the existing depth and rectified RGB topics. Enable
optional diagnostics with `--diagnostics 1`; the launch file accepts
`diagnostics:=1`. `--diagnostic-prefix` defaults to `/stereoforge/diagnostics`.
`--rig-session-id` declares a mounting label, never verifies it. An omitted label
is reported as unavailable. Diagnostics require a depth-producing mode.

Every diagnostic image has the measured depth image's grid, optical frame and
mapped capture stamp. Join metadata to images using `mapped_header_stamp_ns`;
do not associate independently arriving topics by reception order.

| Suffix | Encoding | Meaning |
| --- | --- | --- |
| `disparity` | `32FC1` | Actual rectified GPU match disparity, in pixels; NaN where published depth is invalid |
| `validity` | `32SC1` | 1 for finite positive published measured depth, otherwise 0 |
| `confidence` | `32FC1` | Raw matching confidence; zero when invalid; not metric uncertainty |
| `uncertainty_m` | `32FC1` | NaN throughout: no calibrated metric error model is available |
| `provenance` | `32SC1` | Native measured provenance 1 or 5; zero when invalid |
| `source_ids` | `32SC2` | Exact signed 64-bit source ID; -1 when invalid |
| `source_age_ns` | `32SC2` | Exact signed 64-bit relative producer age, zero for current-frame measurements |
| `metadata` | `std_msgs/String` | JSON frame identity, settings, calibration, clocks and qualification status |

`32SC2` is two little-endian signed 32-bit words containing the low/high halves
of a two's-complement signed 64-bit integer. Reinterpret the eight bytes per
pixel as little-endian int64; never convert IDs through floating point.
Metadata IDs and timestamps are quoted, zero-padded 20-digit decimal strings.
Decode them as integers. This preserves IDs above 2^53.

The metadata identifies the executable hash, actual estimator/settings/shader
identities, requested legacy profile, calibration identities and declared rig
session. It explicitly reports metric uncertainty, acquisition-clock
qualification and rig verification as unavailable/unqualified. Source age zero
is relative to the current producer, not sensor acquisition age. The mapped
header follows the supplied clock map; this alone does not qualify that map.
The publication host clock ID includes boot and time-namespace identity;
ingress clock instance identity is not attested by the current capture format.
Image publication timestamps describe completion of publisher calls, not
subscriber delivery. Metadata is sent after diagnostic image publish calls;
RGB arrives independently.

## Measured and completed geometry

The serializer checks all inputs before writing outputs. Measured pixels must
retain the producer depth bits, actual disparity/confidence, current source ID,
zero relative age and measured provenance. Filter rejection clears all measured
masks/identity; fills and modified finite depths are rejected from this layer.

Speckle filtering on measured depth is now rejection-only, including when
diagnostics are disabled. Previously the disparity-domain speckle setup also
smoothed finite geometry. That smoothing remains available only on the separate
completed layer. Completed rejection gets provenance 0; smoothing or guided
filling that changes geometry gets spatial provenance 8. Unchanged completed
pixels retain their original provenance. These CPU output filters remain an
explicit export-boundary operation; they are not a resident GPU completion path
or a qualified quality improvement.

## Cost and bounded ownership

Diagnostics request the existing GPU observation planes plus optional resident
disparity. They do not rerun matching or reconstruct disparity on the CPU.
Producer leases remain retained through bounded export slots. Temporal fused
observations currently expose no disparity; requesting it fails before exporter
allocation rather than synthesizing a measurement.

At 1280x720, diagnostics add 32 bytes/pixel (29,491,200 bytes/frame) of GPU export
beyond depth/RGB and 36 bytes/pixel (33,177,600 bytes/frame) of ROS image payload,
plus JSON metadata. These are separate buffers and exclude middleware copies,
serialization and queue storage. Message storage is preallocated; normal packing
and metadata stamping allocate no memory. Enabling all planes is substantial
bandwidth, not a live-rate qualification. The separate `<trace>.diagnostics.json`
report accounts diagnostic bytes, packing time and publisher call time without
changing the legacy delivery trace schema. Receiver delivery requires separate
evidence.

## Verification and limits

`pytest -q tests/test_ros_depth_diagnostics.py` compiles the actual serialization
contracts and publisher options/filter setup without a ROS SDK. It exercises
exact IDs, NaN uncertainty, atomic rejection, aliasing, zero-allocation packing,
measured geometry preservation and completed spatial provenance. Native GPU
contracts additionally check exact selective export bytes and producer leases.

Retained development evidence at
`/home/bjoern/stereoforge_quality_evidence/2026-11-06-ros-diagnostics-replay-e2e`
contains `prepare.py`, `run.py`, `protocol.json`, reference dumps, private Zenoh
configs, commands, binary hashes and receiver results. It replays original YUYV
bytes with explicitly synthetic IDs above 2^53 and synthetic clock metadata.
Baseline, speckle and experimental overwrite-D64 cases each delivered all four
expected frames on ten topics;
retained measured depth/disparity/confidence and registered RGB match references
bit-for-bit. Execution settings and compiled shader hashes remain available even when a
variant lacks a qualified canonical estimator identity. Native valid zero disparity
is preserved. The failed pre-fix smoothing witness is preserved. This proves
software transport semantics for those frames, not physical accuracy, calibrated
uncertainty, native acquisition timing, sustained ROS/RViz throughput or a
quality/live profile promotion. Named `quality-reference` and `live-quality`
profiles remain pending measured frontier selection and qualification.
