#!/usr/bin/env python3
"""Probe the installed SDK's align depth semantics without opening a physical device.

A known 100 mm optical-Z translation distinguishes native depth Z from color Z.
API grounding: librealsense wrappers/python/examples/align-with-software-device.py
and src/proc/align.cpp. This tests geometry semantics, not camera accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np


def probe(rs):
    device = rs.software_device()
    intr = rs.intrinsics()
    intr.width = intr.height = 16
    intr.fx = intr.fy = 16.
    intr.ppx = intr.ppy = 7.5
    intr.model, intr.coeffs = rs.distortion.none, [0.] * 5
    sensors, profiles = [], []
    for uid, (kind, fmt, bpp) in enumerate(((rs.stream.depth, rs.format.z16, 2),
                                            (rs.stream.color, rs.format.rgb8, 3)), 1):
        sensor = device.add_sensor(str(kind))
        stream = rs.video_stream()
        stream.type, stream.fmt, stream.bpp = kind, fmt, bpp
        stream.width, stream.height, stream.fps = 16, 16, 30
        stream.index, stream.uid, stream.intrinsics = 0, uid, intr
        profiles.append(sensor.add_video_stream(stream))
        sensors.append(sensor)
    transform = rs.extrinsics()
    transform.rotation = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
    transform.translation = [0., 0., .1]
    profiles[0].register_extrinsics_to(profiles[1], transform)
    device.create_matcher(rs.matchers.default)
    sync = rs.syncer()
    for sensor, profile in zip(sensors, profiles):
        sensor.open(profile)
        sensor.start(sync)
    depth, color = np.full((16, 16), 1000, np.uint16), np.zeros((16, 16, 3), np.uint8)
    align = rs.align(rs.stream.color)
    try:
        aligned = None
        for number in range(1, 5):
            for i, pixels in enumerate((depth, color)):
                frame = rs.software_video_frame()
                frame.pixels, frame.bpp = pixels, (2 if i == 0 else 3)
                frame.stride = 16 * frame.bpp
                frame.timestamp, frame.frame_number = number * 1000. / 30, number
                frame.domain = rs.timestamp_domain.hardware_clock
                frame.profile = profiles[i].as_video_stream_profile()
                frame.depth_units = .001
                sensors[i].on_video_frame(frame)
            frames = sync.wait_for_frames(1000)
            if frames.get_depth_frame() and frames.get_color_frame():
                aligned = np.asarray(align.process(frames).get_depth_frame().get_data()).copy()
        if aligned is None:
            raise RuntimeError('software device produced no paired frames')
        values = np.unique(aligned[aligned != 0]).tolist()
        transformed = rs.rs2_transform_point_to_point(transform, [0., 0., 1.])
        if values != [1000] or not np.isclose(transformed[2], 1.1):
            raise RuntimeError(f'installed SDK alignment semantics differ: {values}, {transformed}')
        package = Path(rs.__file__).parent
        binaries = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('pyrealsense2*.so')}
        return dict(status='verified_installed_sdk_software_device',
                    sdk_version=importlib.metadata.version('pyrealsense2'), sdk_binaries_sha256=binaries,
                    aligned_depth_units=values, depth_scale_m=.001,
                    native_depth_z_m=1., transformed_color_z_m=float(transformed[2]),
                    conclusion='align transfers native depth Z to color pixels; do not deproject as color Z',
                    physical_accuracy_qualified=False)
    finally:
        for sensor in sensors:
            sensor.stop()
            sensor.close()


def main():
    import pyrealsense2 as rs
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = probe(rs)
    report['probe_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
