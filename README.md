# StereoForge

Open stereo depth. Vulkan compute. Explicit geometry.

StereoForge implements ZED 2 acquisition, reproducible recordings, classical
stereo, depth-aware virtual views, and temporal/adaptive experiments. SlangPy
and the C++20 Slang RHI runtime execute shared Slang shaders on Vulkan. CPU
implementations provide references; there is no CUDA, HIP, ROCm, or TensorRT
execution path.

This is an initial research implementation, not a qualified real-time release.
See [hardware evidence](docs/hardware-evidence.md), [architecture](docs/architecture.md),
and the [implementation ledger](docs/implementation-status.md) for exact boundaries.

## Setup

Requires Python 3.11+, uv, CMake 3.24+, Ninja, a C++20 compiler, and a working
Vulkan driver for GPU execution. Optional native Vulkan builds also need the Vulkan development headers. Tested here with Python 3.12 and RADV on AMD8060S.

```bash
uv sync --locked --extra gpu --extra test --extra capture --python 3.12
uv run stereoforge doctor --gpu
uv run pytest -q
cmake --preset cpu
cmake --build --preset cpu
ctest --preset cpu
cmake --preset vulkan
cmake --build --preset vulkan
ctest --preset vulkan
./build/vulkan/native_geometry --projection cylindrical
```

CPU-only setup omits `--extra gpu --extra capture`; run
`uv run pytest -m 'not gpu' -q`. GPU tests require actual Vulkan execution and do
not silently switch to a CPU estimator. Native Python parity tests explicitly
skip when their optional C++ executable has not been built.

The Vulkan preset locates the pinned SlangPy wheel's matching RHI/compiler
headers and binaries. It downloads two SHA256-pinned compiler helper headers.
The wheel's RHI symbols are supplied by `libsgl.so`; standalone RHI binary
installation/export packaging is not yet supported.

## Try stereo and wide RGB-D

```bash
uv run stereoforge synthetic --backend vulkan --disparities 16 \
  --mode wide_rgbd --fusion-backend vulkan --output artifacts/synthetic

cmake -S . -B build/capture -G Ninja \
  -DSTEREOFORGE_CAPTURE=ON -DPython_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build/capture
export STEREOFORGE_NATIVE_DIR="$PWD/build/capture/native"

uv run stereoforge record --frames 30 --output artifacts/recording
uv run stereoforge replay artifacts/recording --config configs/wide-rgbd.json \
  --output artifacts/replay
```

Outputs contain `rgb.png`, lossless `depth.npz` planes, and `metrics.json`.
The adapter converts camera USB YUYV to left/right RGB. Recordings losslessly
preserve those RGB bytes, not the original packed YUYV or Bayer sensor data, plus
factory calibration, ordered IDs, timestamps/clock domains, available sensor
samples, configuration, and SHA256 checksums. Incomplete or corrupted recordings
are rejected.

Recording compression and processing can drop camera frames; capture metrics
report SDK frame-ID gaps. Acquisition rate is distinct from depth throughput.
The factory calibration is downloaded by numeric camera serial and cached.
Calibration failures stop processing; no nominal baseline/intrinsics substitute.

If sensor access is denied, inspect the identified camera's hidraw node. For
this ZED2 (`2b03:f781`), immediate access can be granted with
`sudo setfacl -m "u:$(id -un):rw" /dev/hidraw2` after verifying its current identity.
An example persistent rule is in [camera notes](docs/camera-calibration.md).

## Camera controls

Inspect host-sampled control readback with
`uv run stereoforge controls --output artifacts/controls-readback.json`.
Opening upstream capture resets automatic defaults; no StereoForge profile is
applied unless requested. Both `record` and `live` accept `--camera-controls`:

```bash
# Illustrative normalized values, not a qualified exposure/quality preset.
cat > controls.json <<'EOF'
{"profile":"stereo_optimized","exposure":25,"gain":30,"white_balance":4500}
EOF
uv run stereoforge record --frames 30 --camera-controls controls.json \
  --output artifacts/manual-recording
```

`motion_optimized` requires the same explicit exposure/gain/white-balance fields;
`automatic` enables automatic exposure/gain and white balance. Direct per-eye
settings and ROIs are supported by the pinned API; baseline readback/restoration
passed the [bounded camera check](docs/camera-controls-evidence.md). Image effects
and tuning remain unqualified. Recorded snapshots preserve requested values, readback,
matches and errors separately. They are cached host observations, not per-frame
exposure measurements. See [camera controls](docs/camera-calibration.md#camera-controls-and-readback)
for ranges, semantics and remaining work.

## Output modes

- `left_rgb`, `right_rgb`: original lens canvases after color conversion.
- `rectified_rgb`: rectified left RGB aligned with source disparity/depth.
- `wide_rgbd`: strict calibrated virtual view; unsupported pixels remain holes.
- `wide_rgb`: strict or explicitly marked approximate teleoperation preview.
- `stabilized_wide_rgb`: orientation stabilization with a matching `TimedPose`.

Perspective virtual depth is axial Z; cylindrical virtual depth is ray range.
Stereo disparity always remains in its source rectified geometry. RGB validity,
depth validity, approximation, confidence, source, and provenance are separate.

The CPU view oracle includes exposure matching, confidence seams, compatible
surface blending, and marked infinity-plane preview. The Vulkan renderer currently
provides strict deterministic nearest-surface projection. Select capabilities
explicitly; GPU preview is rejected. No view synthesizer invents metric depth.

Configure the virtual camera with `--virtual-camera configs/virtual-camera.json`.
That example explicitly uses the left-camera origin. Omitting the option derives
a calibrated canvas at the lens midpoint. Requested FOV is not measured coverage.

## Temporal and adaptive experiments

```bash
uv run stereoforge replay artifacts/recording --config configs/temporal-adaptive.json \
  --motions motions.json --output artifacts/temporal
uv run stereoforge ablate artifacts/recording --motions motions.json \
  --backend vulkan --disparities 64 --output artifacts/ablations.json
```

`motions.json` maps each destination frame ID to a 4x4 **previous unrectified left-camera
to current unrectified left-camera** transform in metres. Supply every transition for
ablations. Identity transforms assert a static rig; they are never assumed.
Stabilized output through the Python API requires a rig-to-world `TimedPose`
with the image's exact timestamp and clock domain. Pose estimation/VIO is external.

The four ablations isolate full stereo, fusion, adaptive search, and both.
The Python runner accepts independent ground truth in rectified-left axial metres;
without it, metric errors are null. It reports common-valid accuracy and fixed
truth-support completeness. Reduced candidate counts are not automatically reduced
SGM work: aggregation remains dense over the configured disparity range.

## Current limits

- Python research orchestration explicitly uploads/reads back data; it is not a
  GPU-resident real-time pipeline. Rectification, calibration remapping, and
  metadata handling still include CPU work.
- Native stereo and geometry stage APIs expose asynchronous GPU buffers, leases, and completion;
  an end-to-end native camera-to-mapping application is not qualified.
- No neural model is qualified. ncnn/IREE integration, ROS2, map feedback,
  calibrated uncertainty, and other GPU vendors remain gated extensions.
- No physical metric ground truth was available. Camera rectification screening
  and numerical parity do not establish metric surface accuracy.
- Khronos validation must be installed for a requested driver-validation run;
  native `--validation` fails when it is unavailable.

New code is Apache-2.0. The [license evaluation](docs/license-evaluation.md) clears
public source publication; bundled binaries require a separate review. See
[dependency notices](third_party/README.md) and [NOTICE](NOTICE).
