#!/usr/bin/env python3
"""Person-aware refinement of StereoForge completed depth.

A human segmentation model (U^2-Net human-seg, ONNX) runs on `/stereoforge/rectified_rgb`. Using the person mask:
  * depth at the person's depth that lies *outside* the mask (the halo that stereo matching grows over
    low-contrast background) is removed, and
  * holes and outliers *inside* the mask are filled from the valid depth inside the same mask (normalized
    convolution restricted to the mask, so nothing bleeds across the silhouette).

Depth frames are matched to the mask by `header.stamp`, so the output is aligned with the image it was computed
from (at the cost of the segmentation latency). Publishes `/stereoforge/depth_person` (32FC1) and
`/stereoforge/depth_person_colorized` (rgb8).
"""
from __future__ import annotations

import collections
import os
import sys
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from stereoforge.person_depth import colorize, refine_depth  # noqa: E402

MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx"
MODEL_MIN_BYTES = 100_000_000
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def ensure_model(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size < MODEL_MIN_BYTES:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        if tmp.stat().st_size < MODEL_MIN_BYTES:
            tmp.unlink()
            raise RuntimeError("segmentation model download is incomplete")
        os.replace(tmp, path)
    return path


def person_mask(session: ort.InferenceSession, rgb: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    small = cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    tensor = ((small - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None].astype(np.float32)
    out = session.run(None, {session.get_inputs()[0].name: tensor})[0][0, 0]
    out = (out - out.min()) / (out.max() - out.min() + 1e-9)
    return cv2.resize(out, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR) > threshold


class PersonDepthRefiner(Node):
    def __init__(self) -> None:
        super().__init__("person_depth_refiner")
        self.declare_parameter("model_path", str(Path.home() / ".cache" / "stereoforge" / "u2net_human_seg.onnx"))
        self.declare_parameter("z_min_m", 0.40)
        self.declare_parameter("z_max_m", 2.50)
        self.session = ort.InferenceSession(
            str(ensure_model(Path(self.get_parameter("model_path").value))), providers=["CPUExecutionProvider"]
        )
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub_depth = self.create_publisher(Image, "/stereoforge/depth_person", qos)
        self.pub_col = self.create_publisher(Image, "/stereoforge/depth_person_colorized", qos)
        self.create_subscription(Image, "/stereoforge/depth_completed", self.on_depth, qos)
        self.create_subscription(Image, "/stereoforge/rectified_rgb", self.on_rgb, qos)
        self.depth_ring: collections.OrderedDict[int, np.ndarray] = collections.OrderedDict()
        self.pending: tuple[Image, np.ndarray] | None = None
        self.cond = threading.Condition()
        threading.Thread(target=self.worker, daemon=True).start()

    @staticmethod
    def stamp_ns(msg: Image) -> int:
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def on_depth(self, msg: Image) -> None:
        z = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
        with self.cond:
            self.depth_ring[self.stamp_ns(msg)] = z
            while len(self.depth_ring) > 16:
                self.depth_ring.popitem(last=False)

    def on_rgb(self, msg: Image) -> None:
        rgba = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        with self.cond:
            self.pending = (msg, rgba[:, :, :3].copy())
            self.cond.notify()

    def worker(self) -> None:
        while rclpy.ok():
            with self.cond:
                while self.pending is None:
                    self.cond.wait(timeout=0.5)
                    if not rclpy.ok():
                        return
                msg, rgb = self.pending
                self.pending = None
                z = self.depth_ring.get(self.stamp_ns(msg))
            if z is None or z.shape != rgb.shape[:2]:
                continue
            refined = refine_depth(z, person_mask(self.session, rgb))
            self.publish(msg, refined)

    def publish(self, src: Image, depth: np.ndarray) -> None:
        d = Image()
        d.header = src.header
        d.height, d.width, d.encoding, d.is_bigendian, d.step = depth.shape[0], depth.shape[1], "32FC1", 0, depth.shape[1] * 4
        d.data = depth.astype(np.float32).tobytes()
        self.pub_depth.publish(d)
        col = colorize(depth.astype(np.float32), self.get_parameter("z_min_m").value, self.get_parameter("z_max_m").value)
        c = Image()
        c.header = src.header
        c.height, c.width, c.encoding, c.is_bigendian, c.step = col.shape[0], col.shape[1], "rgb8", 0, col.shape[1] * 3
        c.data = col.tobytes()
        self.pub_col.publish(c)


def main() -> None:
    rclpy.init()
    node = PersonDepthRefiner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
