# Independent native capture slice — 2026-10-27

The C++ capture/replay slice builds without Python or GPU runtimes and passed a
bounded physical acquisition screening. It retains full packed source pixels and
raw metadata with bounded immutable leases. Daxa preprocessing, calibrated depth
and sustained production-pipeline 60 Hz remain unqualified; Task 4 is incomplete.

## Physical screening

Evidence: local `artifacts/native-capture-20261027/physical-600.json` and adjacent
run/identity/stdout/stderr records. The run identifies the physical device before
opening it, uses full 2560×720 side-by-side YUYV (1280×720 per eye), and requests
60 FPS. No source images, calibration contents or device serial are reproduced
here. Reported results:

| Observation | Result |
|---|---:|
| Delivered unique frames / records | 600 / 600 |
| Source records / fresh observations | 601 / 601 |
| Queue drops / no-slot drops / frame-ID gaps | 0 / 0 / 0 |
| Repeats / timeouts / invalid frames | 0 / 0 / 0 |
| Raw IMU samples / reported sensor drops | 3978 / 0 |
| Camera / host-ingress interval cadence | 59.8965 / 59.9047 frames/s |
| Initialization | 0.395327 s |
| Post-snapshot host-ingress-to-delivery p50 / p95 | 0.017843 / 0.035617 ms |

The finite 600-record/wall rate is 59.9999 frames/s; it does not replace the
approximately 59.9 interval cadence. Ingress is stamped after the producer-locked
snapshot/copy, so these small delivery ages exclude camera exposure, USB delivery
and snapshot-copy cost. Camera-to-host age is null: video, IMU and steady clocks
lack an independently validated mapping. Raw hardware-sync metadata is not a pose,
validated IMU association or camera/IMU calibration. The physical calibration
identity was empty; calibrated processing was not run.

The original binary predates `queued_frames_at_stop`. Its 601 source/600 delivered
counts do **not** directly observe one queued frame. Current code exposes queue
occupancy after stop/join; CPU conservation tests cover a callback completing
during stop, held leases, drops and recorded duplicates. The original physical
report is preserved; a fresh physical run of that telemetry revision is pending.

## Ownership, preservation and CPU evidence

A preallocated bounded pool/ring publishes immutable frame leases. Held leases
survive the facade and prevent storage reuse. Live capture drops oldest queued
frames or counts no-slot discards; replay waits for consumers instead. Source
records, fresh observations, delivered records, unique delivered identities,
gaps, repeats and drop reasons stay separate. Missing calibration permits capture
but `require_calibration()` blocks processing; supplied identity is not proof that
arbitrary calibration bytes are correct.

Packed replay streams one record at a time and preserves duplicate records,
unknown/zero ingress timestamps, pixels, control metadata and raw IMU float bits.
The finalized little-endian format bounds metadata/sample/payload sizes and rejects
checksum errors, truncation, trailing bytes and unfinalized counts. FNV detects
accidental corruption and provides no authentication. Recording requires a new or
empty directory; the benchmark rejects existing/symlink report destinations,
including replay-input aliases, before writes. Functional review reproduced and
then verified fixes for input clobber and replay deduplication/restamping.

A warmed CPU-only probe (32 warmups, 256 full-size fixture frames per policy)
observed zero heap allocations and zero drops for both queue policies. Mean
copy+delivery was 0.06463 ms (drop-oldest) / 0.06644 ms (wait-for-consumer). This
resident CPU fixture is not physical USB timing or proof of allocation-free
camera/control/IMU execution. Independent CPU CTest passes 2/2, including lease,
shutdown, replay/corruption, conservation and destination protection cases. The
shared Open Capture snapshot extraction builds with the existing Python binding;
65 capture/control/metadata tests pass in that joint lane without opening a
camera. Final root regression passes **433 Python tests** (7.70 s, no skips),
with fresh joint-capture and Daxa CPU CTests each passing **2/2**. The log is
`artifacts/task3-task4-final-pytest.log`; the Python command was:

```sh
PYTHONPATH=python \
STEREOFORGE_NATIVE_DIR="$PWD/build/native-capture-joint/native" \
SF_NATIVE_VIEW_BENCHMARK="$PWD/build/standalone/native_view_benchmark" \
.venv/bin/python -m pytest -q
```

## Reproduction and remaining boundary

```sh
cmake -S . -B build/native-capture-cpu -G Ninja \
  -DSTEREOFORGE_NATIVE_CAPTURE=ON -DSTEREOFORGE_NATIVE_CAPTURE_ZED=OFF
cmake --build build/native-capture-cpu
ctest --test-dir build/native-capture-cpu --output-on-failure
build/native-capture-cpu/native_capture_benchmark \
  --source synthetic --frames 60 --output NEW-synthetic.json
```

The physical build enables `STEREOFORGE_NATIVE_CAPTURE_ZED=ON` with the pinned
Open Capture/hidapi sources. Re-identify the current device/USB/HID access before
running `native_capture_benchmark --device-id DEVICE_ID --frames 600 --output
NEW-physical.json`; do not reuse an old numeric device mapping. No permissions
were changed for this screening. `ldd` confirms the independent executable has
standard C/C++/hidapi dependencies, without Python, SlangPy/SGL or GPU runtimes.

Capture adds no image conversion, crop, rectification or calibrated geometry.
Physical depth/surface truth, strict IMU association, clock mapping, motion/pose,
Daxa preprocessing, stereo integration and full latency/drop/quality qualification
remain open. Source additions follow repository Apache-2.0 licensing and preserve
pinned dependency notices; this does not qualify binary redistribution.
