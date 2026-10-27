# Hardware evidence, 2026-10-27

These are bounded local observations, not release, metric-distance, or real-time
pipeline qualification. The attached ZED 2 serial is `26041639`; USB sysfs reports
5000 Mbps. Factory calibration reports a 0.119444 m nominal baseline.

A 600-frame 720p-per-eye RGB capture delivered 59.6973 frames/s over 10.033951
camera timestamp seconds. SDK frame IDs were contiguous, but these IDs count SDK
received frames and cannot prove absence of USB/hardware frame loss. Read wall
latency p50/p95/p99 was 16.662/17.455/17.963 ms. Evidence:
`artifacts/capture-smoke-720p60.json`.

The first calibrated capture screening retained three warmed 720p RGB stereo
observations with SHA256-bound metadata and exact byte-for-byte replay:
`artifacts/qualification-initial/recording-720p`. The initial parser's rectification
failed the predeclared 2 px vertical p95 screen. SIFT ratio/mutual matching and an
independent fundamental-matrix RANSAC gave 87–88 likely correspondences per frame,
vertical residual p50 1.39–1.55 px and p95 2.31–3.47 px. Matches were not selected
by their vertical residual.

A separate diagnostic used identical raw correspondence IDs and RANSAC masks for
all extrinsic candidates. Current right-to-left rotation `R` yielded vertical
p95 3.214/3.365/3.562 px; `R.T` yielded 1.692/1.636/1.697 px and 100% positive
horizontal disparities. Flipping translation sign leaves rectified image
coordinates unchanged and cannot be decided from this image screen; physical
camera-center conventions determine the baseline sign. The diagnostic supports
correcting the rotation direction. Evidence:
`artifacts/qualification-initial/fixed-correspondence-extrinsics.json`.

CPU SGBM produced finite positive depth at 20.3–20.8% of pixels on the initial
recording, with median reported depth about 2.32 m. Scene distances are unknown:
these are numerical output checks, not depth accuracy measurements. Processing
wall times were approximately 95–108 ms/frame, including rectification and both
stereo directions, without wide RGB synthesis. Evidence:
`artifacts/qualification-initial/camera-qualification.json` retains the failed
rectification screen and the successful capture/replay/CPU smoke separately.

ZED IMU acquisition is blocked: `/dev/hidraw2` identifies `2b03:f781` with serial
`26041639` and is root-owned mode 0600. The adapter reports sensor unavailability
explicitly. The sensor thread and synchronization path build, but live IMU and
calibrated camera motion remain unverified. No device permissions were changed.

Reproduce the bounded screen with:

```bash
PYTHONPATH=python \
STEREOFORGE_NATIVE_DIR="$PWD/build/capture-build/native" \
.venv/bin/python benchmarks/qualify_camera.py \
  --artifacts artifacts/qualification-new-run
```

Add `--gpu` for three VGA frames through Vulkan Census/SGM and strict wide RGB-D.
This optional run is a numerical smoke with measured stage timings, not a 60 FPS
or metric-distance qualification. Use a new artifact destination for each run.

After the rotation-direction correction, three newly captured VGA (672×376 per
eye) frames passed a bounded Vulkan numerical smoke using 64 disparities, four
SGM paths, left/right consistency, and strict Vulkan wide RGB-D. Depth validity
was 30.06–30.66%; strict virtual RGB/depth coverage was 26.13–26.65%, with no
approximate pixels and finite positive values everywhere depth was valid.
Recordings replayed with exact RGB bytes and metadata. Evidence:
`artifacts/qualification-vulkan-vga/vulkan-camera-smoke.json` and its
`recording-vga` directory.

Per-frame pipeline wall times were 247.5 ms cold, then 160.6 and 167.8 ms. Stereo
including both directions used 93.2/43.5/53.3 ms; view synthesis including host
preparation used 152.2/116.2/113.4 ms. The JSON retains individual GPU timestamp
stages for the last directional stereo pass, submission, waits, and readbacks.
Those last-pass GPU timings are not whole-pipeline timings. These three frames
establish neither sustained throughput nor metric accuracy. Live IMU access was
still unavailable. Capture and GPU resources were closed after the run.


The corrected parser was rerun on the **original saved 720p RGB bytes**, with the
factory calibration deliberately reinterpreted under the new rotation convention.
The old checksum-bound recording and initial failed report remain unchanged.
All three rectification screens pass the unchanged 2 px vertical p95 limit;
positive horizontal disparity fractions are 100%, 100%, and 98.86%. CPU SGBM
validity increased to 22.77–23.95%; unknown-scene reported depths remain
unqualified. The corrected report is
`artifacts/qualification-corrected/camera-qualification.json`.

Reproduce that comparison without camera access:

```bash
PYTHONPATH=python .venv/bin/python benchmarks/qualify_camera.py \
  --replay-source artifacts/qualification-initial/recording-720p \
  --factory-calibration build/calibration-cache/SN26041639.conf \
  --artifacts artifacts/qualification-corrected-new-run
```

The following summary is retained in versioned documentation; recordings and full
JSON artifacts remain ignored local evidence and are not committed:

```json
{
  "scope": "bounded physical capture and numerical screening only",
  "corrected_saved_720p_status": "pass_with_limits",
  "source_calibration_id": "f2cb6844945b041bf24c0eba08fca99050822c1277e50a2bbdeef3da1c621e22",
  "corrected_calibration_id": "40ed5e0bce10cb535e5c252c6e028d2a34b374d9f8af6a08161fffbb750d9390",
  "source_manifest_sha256": "3be01c2863617a5a16921000be73a95bf863c07c4f8d86bec32d4010d7af4ebc",
  "rgb_bytes_unchanged": true,
  "frame_ids": [
    31,
    32,
    33
  ],
  "epipolar_likely_match_counts": [
    95,
    86,
    88
  ],
  "vertical_abs_p50_px": [
    0.464081,
    0.424759,
    0.472824
  ],
  "vertical_abs_p95_px": [
    1.455592,
    1.480534,
    1.38969
  ],
  "horizontal_positive_fraction": [
    1.0,
    1.0,
    0.988636
  ],
  "cpu_sgbm_valid_fraction": [
    0.236157,
    0.227695,
    0.239531
  ],
  "physical_metric_depth_qualified": false,
  "realtime_pipeline_qualified": false,
  "live_imu_qualified": false
}
```
