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
