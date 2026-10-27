# Raw IMU recording and replay evidence — 2026-10-27

Raw IMU delivery and lossless recording/replay passed on ZED 2 serial
`26041639`. **Strict association of each frame batch with the immediately
preceding image interval failed.** These are separate results; the aggregate
qualification report remains `fail`.

The user corrected HID access before this run. Inspection identified
`/dev/hidraw2` as `2b03:f781`, serial `26041639`, owned by `bjoern` with read/write
access. The agent made no permissions changes. Video `/dev/video7` identifies
`2b03:f780` at 5000 Mbps. Open Capture reported `sensors_available=true` and
`hardware_sync_enabled`; this reports SDK configuration, not measured physical
synchronization accuracy.

## Fresh recording

A fresh 672×376-per-eye capture requested 60 FPS, drained 12 warm-up frames, then
retained 48 frames. Capture used 0.800606 host seconds; opening, warming, and
capturing used 1.379167 seconds. Recording and replay, performed after closing
the camera, used 1.873602 seconds. No GPU processing or camera-control tuning
was performed.

| Observation | Result |
| --- | --- |
| Retained SDK frame IDs | 16–63, contiguous |
| Delivered image rate from 47 camera timestamp intervals | 59.9054 frames/s |
| Image interval p50/p95/max | 16.738 / 16.7624 / 16.823 ms |
| Decoded IMU samples | 319, all timestamp-unique and strictly increasing |
| IMU rate from 318 sample timestamp intervals | 399.4435 Hz |
| IMU interval p50/p95/max | 2.490368 / 2.538856 / 2.538856 ms |
| Samples per retained frame | 6–7; no frame without samples |
| SDK sync-marked samples | 48 of 319 |
| Native sensor queue drops | 0 |
| Samples newer than their containing frame | 0 |
| Samples timestamped at/before the previous delivered image | 38, across 33 frames |
| Late-sample lag behind previous image, min/p50/p95/max | 0.010166 / 1.112684 / 2.957161 / 3.614846 ms |
| Acceleration norm min/p50/max | 9.78478 / 9.80546 / 9.83359 m/s² |
| Angular velocity norm min/p50/max | 0.07475 / 0.22837 / 0.48349 deg/s |
| Temperature range | 30.70–30.79 °C |

The lag metric subtracts a late sample timestamp from the previous delivered
image timestamp. It measures observed batching behavior, not calibrated
sensor-to-exposure latency. Samples remained ordered and were delivered once,
but some arrived too late to be included in the earlier frame's batch. A
consumer must use individual timestamps and accommodate late arrivals; current
per-frame grouping is not a certified exposure-synchronous interval.

## Integrity and provenance

Live captured data and replay agreed exactly for RGB byte arrays, frame IDs,
frame timestamps, clock domains, calibration identities, complete sensor JSON
fields, and cached camera-control metadata on every frame. Decoded IMU fields
also matched an explicit little-endian binary representation of timestamp,
seven binary64 values (acceleration, angular velocity, temperature), and sync
flag, bound to each frame ID and sample index. The fixture regression includes
positive versus negative zero to detect bit changes that numeric equality misses.
Raw HID report bytes and hardware IMU sequence IDs are not exposed or recorded.

- Calibration identity: `b63291872f090935f3a14a9057439ea4dcdf6137e6e73e64670dcd86dcce6195`.
- Recording manifest SHA256: `ead903814ad7b3da1d181dde1c001be85f25c36c8e939b4f5d83e2714326d4e3`.
- Captured and replayed ordered decoded IMU SHA256: `63f80cf53aeb78a377c44d5998b6b1a09d9c6fa6bf8d037e6719bc2a56b3b67e`.
- Original failed report SHA256: `0e63a831e59f2c352ad0be272ff2513ff1411831f197ca63e67ce78186432be9`.

Local ignored evidence:

- `artifacts/imu-recording-qualification-20261027/recording/`: 48 checksum-bound frame payloads and metadata.
- `artifacts/imu-recording-qualification-20261027/imu-qualification.json`: original aggregate failure, preserved unchanged.
- `artifacts/imu-recording-qualification-20261027/imu-qualification-reanalysis.json`: same saved recording, explicit delivery/replay PASS and interval-association FAIL gates, plus lag statistics. No recapture or relaxed gate.
- `artifacts/imu-access-recheck-1793102478048966698.json`: earlier access recheck, separate from this recording test.

## Reproduction and remaining limits

```bash
STEREOFORGE_NATIVE_DIR="$PWD/build/capture-build/native" \
.venv/bin/python benchmarks/qualify_imu.py \
  --device-id 7 --expected-serial 26041639 --frames 48 --warmup 12 \
  --artifacts artifacts/imu-qualification-new-run

.venv/bin/python -m pytest -q tests/test_imu_qualification.py
```

Use a new artifact destination and verify current device identity. The script
returns nonzero when strict interval association fails even if raw delivery and
recording/replay pass. Eleven offline fixture tests cover exact recording/replay,
malformed fields, duplicate/future samples, signed-zero preservation, controls,
and retention of the failed association gate for late unique samples.

No IMU extrinsics, sensor bias/noise calibration, camera-IMU clock calibration,
physical sync accuracy, calibrated motion, or VIO has been qualified. SDK frame
IDs and zero application queue drops do not prove absence of hardware/USB loss.
Camera controls remain cached host readbacks, not exposure-time truth. A
consumer-side timestamp association policy that handles late samples remains
open before integrated calibrated motion can be claimed.
