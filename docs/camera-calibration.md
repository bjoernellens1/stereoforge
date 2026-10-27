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

Priorities below remain **unimplemented**, with separate correctness and
transfer-inclusive performance gates:

- Preserve packed YUYV, match directly from luminance, and create RGB lazily while
  checking image quality against the existing conversion path.
- Expose capture controls and record exposure, gain and white balance; validate
  left/right consistency rather than assume matching settings.
- Move rectification to Vulkan and add a persistent upload ring to reduce copies.

These priorities do not establish control ranges, sensor rates or measured depth
accuracy.
