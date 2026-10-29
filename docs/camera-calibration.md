# Camera and calibration

The adapter opens ZED Open Capture with a selected physical camera and checks
serial and mode against calibration. Frame data is copied under the capture
library mutex; a bounded sensor thread collects raw new samples. Sensor unavailability
is explicit. Frame IDs/timestamps are library outputs; gap counts do not prove
zero hardware losses or a particular exposure timestamp accuracy.

Factory INI intrinsics are mode-specific. Baseline and suffixed TY/TZ are converted
from millimetres to metres; Rodrigues RX/CV/RZ is transposed when represented as
a right-camera pose. This rotation convention is covered by synthetic epipolar
regressions and same-byte live-image screening. Global unsuffixed TY/TZ fields are
not interpreted; their convention and other camera families require validation.

The official ZED2 factory calibration for serial `26041639` is tracked at
[`configs/cameras/zed2/SN26041639.conf`](../configs/cameras/zed2/SN26041639.conf).
It is a calibration source, not a ROS publisher manifest. To create the native
Daxa preprocessing fixture, first record packed YUYV frames and convert the
factory calibration to canonical JSON:

```bash
uv run python - <<'PY'
import json
import sys
sys.path.insert(0, "python")
from stereoforge.calibration import parse_zed_conf

calibration = parse_zed_conf(
    "configs/cameras/zed2/SN26041639.conf", 1280, 720, "26041639"
)
with open("/tmp/stereoforge-zed2-calibration.json", "w") as output:
    json.dump(calibration.to_dict(), output, sort_keys=True, separators=(",", ":"))
PY

build/ros-daxa/native_capture_benchmark \
  --frames 30 \
  --record-dir /path/to/zed2-capture \
  --output /path/to/zed2-capture-report.json

PYTHONPATH=python .venv/bin/python benchmarks/compare_daxa_preprocess.py \
  --export /path/to/zed2-capture/capture.sfp \
  --record-ordinal 0 \
  --calibration /tmp/stereoforge-zed2-calibration.json \
  --output /path/to/zed2-fixture
```

The resulting fixture `manifest.json`, rectification maps, and `capture.sfp`
remain runtime artifacts and are not tracked. A measured `clock-map.json` is
also required for ROS launch; factory calibration does not provide that
source-clock-to-ROS mapping.

The native capture options do not provide a safe serial override:
`--device-id` selects a video device, while `--calibration-id` supplies the
expected calibration identity hash after capture. Open Capture must first
report a positive hardware serial; otherwise StereoForge rejects the source
before calibration can be associated with it. Do not bypass that check by
assigning the tracked factory calibration to an unidentified video node.

To grant only the current user access to the identified ZED2 sensor interface:

```bash
sudo setfacl -m "u:$(id -un):rw" /dev/hidraw2
printf 'SUBSYSTEM=="hidraw", ATTRS{idVendor}=="2b03", ATTRS{idProduct}=="f781", OWNER="%s", MODE="0600"\n' "$(id -un)" |
  sudo tee /etc/udev/rules.d/99-stereoforge-zed2-imu.rules
sudo udevadm control --reload-rules
sudo udevadm trigger --action=add /sys/class/hidraw/hidraw2
```

Device node numbering can change. Reidentify the node before applying a temporary
ACL. Reopening capture after permission changes reruns sensor initialization.

## Camera and host processing boundary

The camera supplies ISP-processed, synchronized images, image controls, sensor
samples and USB transport. Its images are
[unrectified post-ISP video](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/recording),
not Bayer sensor raw. Rectification, matching and metric depth are
[host computations](https://support.stereolabs.com/articles/6815914373-how-does-the-zed-3d-perception-work);
StereoForge also performs temporal processing and view synthesis on the host.
Slang shaders cannot be deployed onto the camera through this capture interface.

Currently the adapter converts packed USB YUYV to RGB on the CPU. The pipeline
rectifies RGB and converts it to grayscale on the CPU before Vulkan stereo.
This is not GPU rectification or a zero-copy capture path. Image controls applied
before USB delivery are already reflected in the recorded pixels.

## Camera controls and readback

`stereoforge controls --output controls-readback.json` opens the camera and writes
an explicit control-register snapshot. Opening upstream capture resets automatic
defaults; StereoForge applies no additional profile unless requested. Both
`record` and `live` accept `--camera-controls request.json`.

A profile request uses `{"profile":"automatic"}` or a manual profile such as:

```json
{"profile":"stereo_optimized","exposure":25,"gain":30,"white_balance":4500,"sharpness":0}
```

These numbers are illustrative and unqualified. `stereo_optimized` and
`motion_optimized` both require explicit normalized exposure/gain (0–100) and
white balance (2800–6500); they disable automatic exposure/gain and white balance,
apply matched left/right values, and default sharpness to zero. Neither label
promises exposure duration, tuned image quality or motion performance. `automatic`
enables the two automatic flags and leaves other ISP settings unchanged.

Direct native-setting requests use `exposure_left`, `exposure_right`, `gain_left`,
`gain_right`, `roi_left` and `roi_right`, alongside supported shared ISP controls.
Brightness, contrast, saturation and sharpness accept integers 0–8; hue 0–11;
gamma 1–9. Exposure/gain accept 0–100 and white balance 2800–6500.
An ROI is `[x,y,width,height]` inside the per-eye source canvas with area greater
than 100 pixels; `null` resets that eye to its full frame. This interface is
verified against pinned upstream APIs. Baseline control readback/restoration
passed a [bounded camera check](camera-controls-evidence.md); image effects,
different ROI settings and ROI reset remain unqualified.

Reports separate `requested`, `readback`, `matches`, `read_errors` and
`write_errors`; requested values never stand in for measured readback. Snapshots
are sampled on open, writes and explicit reads, then cached without per-frame USB
queries. Their host-monotonic timestamp is not exposure-synchronous metadata.
Frames carry the cached snapshot through recording, replay and rectification.
Native requests are fully validated before USB writes, but an I/O failure can
leave a partially applied request. The CLI stops when readback cannot verify a
request. Automatic-mode boolean getters cannot distinguish upstream I/O failure;
their readbacks are API observations, not physical proof. LED writes are explicit
only: safe LED readback is unavailable because the upstream getter changes GPIO
direction. Reports leave `led` null and explain this in `readback_limitations`;
a requested LED write therefore cannot be reported as verified.

## Remaining priorities

These remain **unimplemented**, with separate correctness and transfer-inclusive
performance gates:

- Preserve packed YUYV, match directly from luminance, and create RGB lazily while
  checking image quality against the existing conversion path.
- Establish exposure-synchronous metadata and validate left/right image effects;
  a quality-aware automatic-exposure controller is future work.
- Move rectification to Vulkan and add a persistent upload ring to reduce copies.

These priorities do not establish sensor rates or measured depth accuracy.
