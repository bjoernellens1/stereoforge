# Camera control evidence — 2026-10-27

The pinned ZED Open Capture API (`93739154903a66c112ee52c3c6fbfd83b7218617`)
supports per-eye manual exposure/gain and auto-exposure ROI, shared ISP controls,
white balance and automatic flags. No proprietary SDK or additional dependency
was added. Exposure/gain are normalized 0–100 controls, not milliseconds or ISO.

## Bounded physical check

Identified `/dev/video7`, USB `2b03:f780`, 5000 Mb/s, ZED 2 serial 26041639.
The qualifier saved no images. IMU initialization still failed; this does not
qualify sensors, synchronization, image quality or physical exposure timing.

The first exercise rejected an invalid mixed request, but the automatic-exposure
flag changed between API snapshots. It stopped before the baseline exercise and
restored settings with no stable-field mismatches. This failed artifact remains
preserved; the observation is not evidence that validation performed a write.
Source inspection confirms complete validation precedes USB calls. Upstream
automatic boolean getters cannot distinguish I/O errors.

Source inspection also found that upstream `getLEDstatus()` changes GPIO direction.
StereoForge therefore omits that getter: `led` is null with an explicit limitation.
The qualifier excludes LED writes and restoration. Opening/closing upstream
capture retains its existing LED and automatic-default behavior.

After rebuilding, one warmed retry collected four read-only samples over one
second; stable fields matched. It then applied only the baseline's existing
restorable settings with automatic modes disabled, compared readbacks, and
restored manual values followed by automatic flags in `finally`.

- Baseline exposure left/right: 57/57; gain: 28/28; white balance: 4600.
- Brightness/contrast/saturation: 4; hue/sharpness: 0; gamma: 1.
- Both ROIs: `[0,0,1280,708]`; both automatic flags initially true.
- Invalid request rejected; stable-field mismatches: none.
- Applied baseline readback mismatches and read/write errors: none.
- Restored stable-field mismatches and errors: none; closed guard passed.
- Baseline read: 27.536 ms; restored read: 27.985 ms. Entire exercise through
  restoration: 231.640 ms, including reads and validation, not pure write latency.

This checks baseline API round trips only. Different manual values, ROI reset,
LED effects, photometric matching and short-exposure motion benefits remain
unqualified. Automatic exposure/gain/white balance can change between snapshots.

Ignored local artifacts and SHA256 identities:

| Artifact under `artifacts/` | SHA256 |
|---|---|
| `controls-readonly.json` | `1897792d28720af1a3d04a85e81669507491f75ec1b454f4806bddb76ddd9e99` |
| `controls-exercise.json` | `30a1a34574420548e32198900772514d4c9510a50e7bfc1d0727fc24f5be7b98` |
| `controls-exercise-warm.json` | `3f72a51618164db703078a6ead572d6a84480628fbb3892c008c438169516b44` |

Reproduce the bounded warmed exercise after building the capture extension:

```bash
STEREOFORGE_NATIVE_DIR=build/capture-build/native PYTHONPATH=python \
  .venv/bin/python benchmarks/qualify_controls.py --device-id 7 \
  --exercise --warmup-seconds 1 --output artifacts/controls-new.json
```

Reidentify the video node first. Without `--exercise`, the qualifier reads API
state only, subject to upstream capture initialization/shutdown behavior.

## Capture metadata performance

A separate performance review found no per-frame control-register queries or new
GPU waits. Cached host snapshots are copied at capture/rectification/recording
boundaries so later writes cannot alter prior frames. AST guards protect the
cached path, and the integration test counts exactly one initial control read
across repeated frame acquisitions.

`PYTHONPATH=python .venv/bin/python benchmarks/benchmark_control_metadata.py`
measured a 449-byte synthetic snapshot, 20 warmups and seven batches of 1000:

| Isolated operation | Median |
|---|---:|
| Locked cached snapshot copy | 8.47 µs |
| Rectifier metadata deepcopy | 6.93 µs |
| Recorder canonical JSON round trip | 5.52 µs |

These separate component measurements exclude USB, images, disk I/O and GPU
execution. They are not end-to-end overhead or a new capture-rate measurement.

## Verification

With the built native capture extension selected: 178 non-GPU Python tests
passed, with no skips; 10 AMD Vulkan tests passed separately. Native CPU CTest
passed 1/1 and Vulkan CTest 4/4. The latter includes honest rejection when the
validation layer is missing, not clean driver-validation qualification.
The Python wheel builds and includes `controls.py`; no binary was published.
Quality-aware automatic exposure and exposure-synchronous metadata are future work.
