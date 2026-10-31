#!/usr/bin/env python3
"""Bounded loss-diagnostic RGB/depth ROS 2 sequence recorder (no SVO/SDK).

Run under the StereoForge Pixi ROS environment. Writes uncompressed per-frame
32FC1 and rgba8 buffers, plus paired publisher header timestamps and monotonic
subscriber receipt times. It neither validates the source camera clock nor
claims zero drops. Does not import NumPy or OpenCV in the ROS process.
"""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import time
import sys

SCHEMA = "stereoforge.ros_depth_sequence.v1"


def validate_image(msg, expected_encoding):
    if msg.encoding != expected_encoding:
        raise ValueError(f"expected {expected_encoding}, got {msg.encoding}")
    if msg.width <= 0 or msg.height <= 0 or msg.width > 8192 or msg.height > 8192:
        raise ValueError("invalid resolution")
    if msg.step != msg.width * 4 or len(msg.data) != msg.step * msg.height:
        raise ValueError("only tightly packed four-byte pixel rows are supported")
    if expected_encoding == "32FC1" and msg.is_bigendian:
        raise ValueError("big-endian 32FC1 unsupported")
    if msg.header.stamp.sec < 0 or not (0 <= msg.header.stamp.nanosec < 1_000_000_000):
        raise ValueError("invalid header timestamp")
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


class SequenceWriter:
    def __init__(self, output, frames=90, paired=True, max_pending=12):
        self.root = Path(output)
        self.root.mkdir(parents=True, exist_ok=False)  # never overwrite recordings
        if not (2 <= frames <= 10_000 and 1 <= max_pending <= 256):
            raise ValueError("invalid frame or pending limit")
        self.limit, self.paired, self.max_pending = frames, paired, max_pending
        self.depth = OrderedDict()
        self.rgb = OrderedDict()
        self.records = []
        self.unpaired_depth = 0
        self.unpaired_rgb = 0
        self.duplicates = 0
        self.first_shape = None
        self.started_ns = time.monotonic_ns()

    def _trim(self, cache, is_depth):
        while len(cache) > self.max_pending:
            cache.popitem(last=False)
            if is_depth:
                self.unpaired_depth += 1
            else:
                self.unpaired_rgb += 1

    def add(self, msg, kind, receive_ns=None):
        if len(self.records) >= self.limit:
            return
        if kind not in ("depth", "rgb"):
            raise ValueError("unknown image kind")
        if kind == "rgb" and not self.paired:
            return
        stamp = validate_image(msg, "32FC1" if kind == "depth" else "rgba8")
        receive_ns = time.monotonic_ns() if receive_ns is None else int(receive_ns)
        if receive_ns <= 0:
            raise ValueError("invalid receipt time")
        cache = self.depth if kind == "depth" else self.rgb
        if stamp in cache or (self.records and stamp <= self.records[-1]["stamp_ns"]):
            self.duplicates += 1
            return
        cache[stamp] = (msg, receive_ns)
        if not self.paired:
            self._write(stamp, self.depth.pop(stamp), None)
            return
        if stamp in self.depth and stamp in self.rgb:
            self._write(stamp, self.depth.pop(stamp), self.rgb.pop(stamp))
        self._trim(self.depth, True)
        self._trim(self.rgb, False)

    def _write(self, stamp, depth_pair, rgb_pair):
        depth, receive_depth = depth_pair
        width, height = int(depth.width), int(depth.height)
        if self.first_shape is None:
            self.first_shape = (height, width)
        elif self.first_shape != (height, width):
            raise ValueError("resolution changed during capture")
        if rgb_pair is not None:
            rgb, receive_rgb = rgb_pair
            if (rgb.width, rgb.height) != (width, height):
                raise ValueError("RGB/depth resolution mismatch")
            if rgb.header.frame_id != depth.header.frame_id:
                raise ValueError("RGB/depth frame ID mismatch")
        index = len(self.records)
        base = f"frame-{index:05d}"
        depth_name = f"{base}.depth.f32"
        with (self.root / depth_name).open("xb") as file:
            file.write(depth.data)
        row = {
            "index": index,
            "stamp_ns": stamp,
            "depth_received_ns": receive_depth,
            "depth_file": depth_name,
            "frame_id": depth.header.frame_id,
        }
        if rgb_pair is not None:
            rgb_name = f"{base}.rgb.rgba8"
            with (self.root / rgb_name).open("xb") as file:
                file.write(rgb.data)
            row["rgba_file"] = rgb_name
            row["rgb_received_ns"] = receive_rgb
        self.records.append(row)

    def finish(self):
        if self.first_shape is None:
            raise ValueError("no frames received")
        height, width = self.first_shape
        manifest = {
            "schema": SCHEMA,
            "width": width,
            "height": height,
            "paired_rgb": self.paired,
            "frames": self.records,
            "stats": {
                "written": len(self.records),
                "unpaired_depth_evicted": self.unpaired_depth,
                "unpaired_rgb_evicted": self.unpaired_rgb,
                "duplicate_or_out_of_order": self.duplicates,
                "pending_depth": len(self.depth),
                "pending_rgb": len(self.rgb),
                "elapsed_ns": time.monotonic_ns() - self.started_ns,
            },
            "clock_note": "header stamps are unvalidated publisher data; receipt uses local monotonic_ns",
        }
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--timeout-seconds", type=float, default=25.0)
    parser.add_argument("--depth-topic", default="/stereoforge/depth")
    parser.add_argument("--rgb-topic", default="/stereoforge/rectified_rgb")
    parser.add_argument("--depth-only", action="store_true")
    args = parser.parse_args()
    if not (0 < args.timeout_seconds <= 3600):
        parser.error("invalid timeout")
    # Import ROS only when used as a live capture utility, never during unit tests.
    try:
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
    except (ImportError, OSError) as exc:
        parser.error(
            f"ROS Python import failed under {sys.executable}: "
            f"{type(exc).__name__}: {exc}. Use the Pixi interpreter "
            "or scripts/record-ros-depth-video.sh."
        )
    writer = SequenceWriter(args.output_dir, frames=args.frames, paired=not args.depth_only)
    rclpy.init()
    node = rclpy.create_node("stereoforge_temporal_depth_recorder")
    subs = [node.create_subscription(Image, args.depth_topic,
               lambda msg: writer.add(msg, "depth"), qos_profile_sensor_data)]
    if not args.depth_only:
        subs.append(node.create_subscription(Image, args.rgb_topic,
                    lambda msg: writer.add(msg, "rgb"), qos_profile_sensor_data))
    deadline = time.monotonic() + args.timeout_seconds
    try:
        while rclpy.ok() and len(writer.records) < writer.limit and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.25)
        manifest = writer.finish()
        print(f"Recorded {len(writer.records)} frames to {writer.root}")
        print(json.dumps(manifest["stats"], indent=2))
        if len(writer.records) < 2:
            raise SystemExit("Too few frames to evaluate temporal behavior")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
