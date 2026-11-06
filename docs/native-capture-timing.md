# Native driver timing evidence

`CaptureFrame::driver_timing` optionally retains observations from the exact
successful V4L dequeue that supplied its pixels. The generated SDK adapter writes
the observations under the pixel producer mutex; its snapshot copies them with
the same frame. Pool reuse clears the optional value before invoking a source.

Existing `camera_timestamp_ns`, camera clock domain and host-ingress semantics
remain unchanged. Raw driver observations do not qualify physical exposure
timing, cross-sensor synchronization, acquisition age or uncertainty. In
particular, a driver SOE flag is a reported event and requires independent
validation before physical qualification.

| Field | Type | Meaning |
|---|---|---|
| `timestamp_ns` | u64 | Returned V4L timeval converted to nanoseconds |
| `sequence` | u32 | Returned kernel sequence; distinct from the SDK source ID |
| `flags` | u32 | Complete returned V4L buffer flags, including unknown bits |
| `dequeue_begin_monotonic_ns` | u64 | Host monotonic observation before dequeue |
| `dequeue_end_monotonic_ns` | u64 | Host monotonic observation after dequeue |
| `sdk_wall_anchor_ns` | u64 | SDK's wall-clock anchor for this stream |
| `sdk_first_timestamp_ns` | u64 | SDK's first raw driver timestamp |

The SDK association is checked with overflow-safe integer arithmetic:

```text
camera_timestamp_ns = sdk_wall_anchor_ns + timestamp_ns - sdk_first_timestamp_ns
```

Device identity and both SDK anchors distinguish SDK stream instances. A clock
association must not be reused after reinitialization. Host clock-instance
attestation, namespace association and physical timing-error bounds remain
separate qualification requirements; this metadata alone supplies none of them.

## Recording and replay

New `PackedCaptureWriter` recordings use SFP version 2. The container magic remains
`SFYUYV1\0`; the following little-endian u32 version is 2. The finalized u64 record
count and legacy record fields/pixels keep their existing layout. After each
pixel payload and before its FNV-1a-64 checksum, v2 appends:

- One u8 presence flag, strictly 0 or 1.
- If present: `timestamp_ns` u64, `sequence` u32, `flags` u32, followed by the four
  dequeue/SDK-anchor u64 fields in table order.

The extension is covered by the record checksum. FNV detects accidental
corruption and is not authentication. Invalid brackets, wrapped SDK associations
and mismatched SDK timestamps reject. Readers accept v1 and clear absent timing
instead of inventing observations. Explicit `CaptureRecordingFormat::legacy_v1`
writers reject frames carrying timing, preventing silent metadata loss.

## Requested diagnostic output

`native_capture_benchmark --timing-report FILE` writes one JSON line per received
frame. Source IDs and all u64 timestamps are decimal strings; sequence/flags are
u32 integers. Absent timing is `null`, and physical timing qualification remains
false. The report can be requested for capture or replay. Its destination must be
fresh and distinct from the main report and packed capture file. No timing report
is generated unless requested.

Native preprocessing/depth fixture reports also retain optional driver timing in
their source metadata. Those benchmark JSON fields use exact integer values and
require an exact-integer decoder. The shared formatter preserves opaque IMU float
bits. No estimator, production clock mapping or ROS profile is selected by these
observations.
