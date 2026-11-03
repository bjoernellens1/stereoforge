#!/usr/bin/env python3
"""ROS 2 Dual-Camera (ZED 2 + RealSense D435i) Publisher & Colorized Depth Bridge.

Spawns the RealSense D435i pyrealsense2 capture worker in an isolated Python
environment (stripping PYTHONPATH and LD_LIBRARY_PATH so Pixi Python 3.14
site-packages never collide with miniforge3 Python 3.13 numpy/pyrealsense2)
and publishes both raw metric depth (32FC1) and colorized/RGB-registered views.
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
import threading
from typing import Any

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

RS_HELPER_CODE = r"""
import sys, struct, time
import numpy as np
import pyrealsense2 as rs

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgba8, 30)
prof = pipe.start(cfg)
depth_sensor = prof.get_device().first_depth_sensor()
if depth_sensor.supports(rs.option.emitter_enabled):
    depth_sensor.set_option(rs.option.emitter_enabled, 1.0)
if depth_sensor.supports(rs.option.laser_power):
    max_lp = depth_sensor.get_option_range(rs.option.laser_power).max
    depth_sensor.set_option(rs.option.laser_power, max_lp)
scale = float(depth_sensor.get_depth_scale())
align_to_color = rs.align(rs.stream.color)

for _ in range(10):
    pipe.wait_for_frames(5000)

d_intr = prof.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
out = sys.stdout.buffer
out.write(struct.pack("<IIfff", d_intr.width, d_intr.height, d_intr.fx, d_intr.fy, scale))
out.flush()

try:
    while True:
        fs = pipe.wait_for_frames(5000)
        df_raw = fs.get_depth_frame()
        aligned_fs = align_to_color.process(fs)
        df_aligned = aligned_fs.get_depth_frame()
        cf = aligned_fs.get_color_frame()
        if not df_raw or not df_aligned or not cf:
            continue
        d_raw_m = (np.asanyarray(df_raw.get_data()).astype(np.float32) * scale)
        d_raw_m[(d_raw_m < 0.15) | (d_raw_m > 10.0)] = 0.0
        d_al_m = (np.asanyarray(df_aligned.get_data()).astype(np.float32) * scale)
        d_al_m[(d_al_m < 0.15) | (d_al_m > 10.0)] = 0.0
        c_rgba = np.asanyarray(cf.get_data()).astype(np.uint8)
        ts_ns = int(time.time_ns())
        hdr = struct.pack("<QIII", ts_ns, d_raw_m.nbytes, d_al_m.nbytes, c_rgba.nbytes)
        out.write(hdr)
        out.write(d_raw_m.tobytes())
        out.write(d_al_m.tobytes())
        out.write(c_rgba.tobytes())
        out.flush()
except (BrokenPipeError, KeyboardInterrupt):
    pass
finally:
    pipe.stop()
"""

COLORMAPS = {
    "turbo": cv2.COLORMAP_TURBO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "inferno": cv2.COLORMAP_INFERNO,
    "plasma": cv2.COLORMAP_PLASMA,
    "magma": cv2.COLORMAP_MAGMA,
    "jet": cv2.COLORMAP_JET,
}


def build_natural_color_3d_lut() -> np.ndarray:
    """Precompute a 64x64x64 3D RGB lookup table for ZED 2 natural color restoration."""
    grid = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    rr, gg, bb = np.meshgrid(grid, grid, grid, indexing="ij")
    rgb = np.stack([rr, gg, bb], axis=-1).reshape(64, 64 * 64, 3)

    rgb_lifted = np.power(np.clip(rgb, 0.0, 1.0), 0.78)
    lum = 0.299 * rgb_lifted[:, :, 0] + 0.587 * rgb_lifted[:, :, 1] + 0.114 * rgb_lifted[:, :, 2]
    w_mid = np.clip((lum - 0.08) / 0.20, 0.0, 1.0)[:, :, None]
    wb_gains = np.array([0.95, 1.00, 1.18], dtype=np.float32)[None, None, :]
    rgb_wb = rgb_lifted * (1.0 + w_mid * (wb_gains - 1.0))
    rgb_wb = rgb_wb / (1.0 + 0.16 * np.maximum(0.0, rgb_wb - 0.75))
    bgr_u8 = (np.clip(rgb_wb[:, :, ::-1], 0.0, 1.0) * 255.0).astype(np.uint8)
    hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    warm_weight = np.exp(-0.5 * ((h - 22.0) / 10.0) ** 2)
    hsv[:, :, 1] = np.clip(s * 0.90 * (1.0 - 0.32 * warm_weight), 0.0, 255.0)
    nat_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return nat_rgb.reshape(64, 64, 64, 3)


NATURAL_LUT_64 = build_natural_color_3d_lut()


def restore_zed2_natural_rgb(rgba: np.ndarray) -> np.ndarray:
    q = rgba[:, :, :3] >> 2
    return NATURAL_LUT_64[q[:, :, 0], q[:, :, 1], q[:, :, 2]]


def stylize_depth_rgb(
    depth_m: np.ndarray,
    base_rgb: np.ndarray | None,
    cmap_name: str,
    register_rgb: bool,
    alpha: float,
    vmin: float,
    vmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    cmap_id = COLORMAPS.get(cmap_name.lower(), cv2.COLORMAP_TURBO)
    valid = np.isfinite(depth_m) & (depth_m > 0.20)
    norm = np.clip((np.where(valid, depth_m, vmin) - vmin) / max(vmax - vmin, 0.05), 0.0, 1.0)
    u8 = (255.0 * (1.0 - norm)).astype(np.uint8)
    col_bgr = cv2.applyColorMap(u8, cmap_id)
    col_rgb = np.ascontiguousarray(col_bgr[:, :, ::-1])
    col_rgb[~valid] = (20, 20, 20)

    if base_rgb is not None and register_rgb and base_rgb.shape[:2] == depth_m.shape:
        blended = cv2.addWeighted(base_rgb, 1.0 - alpha, col_rgb, alpha, 0.0)
        reg_rgb = base_rgb.copy()
        reg_rgb[valid] = blended[valid]
    else:
        reg_rgb = col_rgb
    return col_rgb, reg_rgb


class DualCameraDepthBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("dual_camera_depth_bridge")
        self.declare_parameter("colormap", "turbo")
        self.declare_parameter("rgb_registration", True)
        self.declare_parameter("overlay_alpha", 0.68)
        self.declare_parameter("z_min_m", 0.40)
        self.declare_parameter("z_max_m", 2.50)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub_rs_depth = self.create_publisher(Image, "/realsense/depth", qos)
        self.pub_rs_rgb = self.create_publisher(Image, "/realsense/rectified_rgb", qos)
        self.pub_rs_d_col = self.create_publisher(Image, "/realsense/depth_colorized", qos)
        self.pub_rs_d_reg = self.create_publisher(Image, "/realsense/depth_rgb_registered", qos)

        self.pub_zed_rgb_nat = self.create_publisher(Image, "/stereoforge/rectified_rgb_natural", qos)
        self.pub_zed_d_col = self.create_publisher(Image, "/stereoforge/depth_colorized", qos)
        self.pub_zed_d_reg = self.create_publisher(Image, "/stereoforge/depth_rgb_registered", qos)
        self.pub_zed_dc_col = self.create_publisher(Image, "/stereoforge/depth_completed_colorized", qos)
        self.pub_zed_dc_reg = self.create_publisher(Image, "/stereoforge/depth_completed_rgb_registered", qos)

        self.latest_zed_nat_rgb: np.ndarray | None = None
        self.pending_rs: tuple[int, np.ndarray, np.ndarray, np.ndarray] | None = None
        self.lock = threading.Lock()

        self.create_subscription(Image, "/stereoforge/rectified_rgb", self.on_zed_rgb, qos)
        self.create_subscription(Image, "/stereoforge/depth", self.on_zed_depth, qos)
        self.create_subscription(Image, "/stereoforge/depth_completed", self.on_zed_depth_completed, qos)
        self.create_timer(1.0 / 60.0, self.on_timer_flush_rs)

    def params(self) -> tuple[str, bool, float, float, float]:
        cmap = str(self.get_parameter("colormap").value)
        reg = bool(self.get_parameter("rgb_registration").value)
        alpha = float(self.get_parameter("overlay_alpha").value)
        zmin = float(self.get_parameter("z_min_m").value)
        zmax = float(self.get_parameter("z_max_m").value)
        return cmap, reg, alpha, zmin, zmax

    @staticmethod
    def make_rgb_msg(header: Any, rgb: np.ndarray) -> Image:
        h, w, _ = rgb.shape
        msg = Image()
        msg.header = header
        msg.height = h
        msg.width = w
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = w * 3
        msg.data = np.ascontiguousarray(rgb).tobytes()
        return msg

    def on_zed_rgb(self, msg: Image) -> None:
        raw_rgba = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        nat_rgb = restore_zed2_natural_rgb(raw_rgba)
        self.latest_zed_nat_rgb = nat_rgb
        self.pub_zed_rgb_nat.publish(self.make_rgb_msg(msg.header, nat_rgb))

    def on_zed_depth(self, msg: Image) -> None:
        d_m = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        nat_rgb = self.latest_zed_nat_rgb
        cmap, reg, alpha, zmin, zmax = self.params()
        pure_rgb, reg_rgb = stylize_depth_rgb(d_m, nat_rgb, cmap, reg, alpha, zmin, zmax)
        self.pub_zed_d_col.publish(self.make_rgb_msg(msg.header, pure_rgb))
        self.pub_zed_d_reg.publish(self.make_rgb_msg(msg.header, reg_rgb))

    def on_zed_depth_completed(self, msg: Image) -> None:
        d_m = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        nat_rgb = self.latest_zed_nat_rgb
        cmap, reg, alpha, zmin, zmax = self.params()
        pure_rgb, reg_rgb = stylize_depth_rgb(d_m, nat_rgb, cmap, reg, alpha, zmin, zmax)
        self.pub_zed_dc_col.publish(self.make_rgb_msg(msg.header, pure_rgb))
        self.pub_zed_dc_reg.publish(self.make_rgb_msg(msg.header, reg_rgb))

    def enqueue_rs(self, ts_ns: int, d_raw_m: np.ndarray, d_al_m: np.ndarray, c_rgba: np.ndarray) -> None:
        with self.lock:
            self.pending_rs = (ts_ns, d_raw_m, d_al_m, c_rgba)

    def on_timer_flush_rs(self) -> None:
        with self.lock:
            item = self.pending_rs
            self.pending_rs = None
        if item is None:
            return
        ts_ns, d_raw_m, d_al_m, c_rgba = item
        h, w = d_raw_m.shape
        d_msg = Image()
        d_msg.header.stamp.sec = int(ts_ns // 1_000_000_000)
        d_msg.header.stamp.nanosec = int(ts_ns % 1_000_000_000)
        d_msg.header.frame_id = "stereoforge_camera"
        d_msg.height = h
        d_msg.width = w
        d_msg.encoding = "32FC1"
        d_msg.is_bigendian = 0
        d_msg.step = w * 4
        d_msg.data = d_raw_m.tobytes()
        self.pub_rs_depth.publish(d_msg)

        c_rgb = np.ascontiguousarray(c_rgba[:, :, :3])
        self.pub_rs_rgb.publish(self.make_rgb_msg(d_msg.header, c_rgb))
        cmap, reg, alpha, zmin, zmax = self.params()
        pure_rgb, _ = stylize_depth_rgb(d_raw_m, None, cmap, False, alpha, zmin, zmax)
        _, reg_rgb = stylize_depth_rgb(d_al_m, c_rgb, cmap, reg, alpha, zmin, zmax)
        self.pub_rs_d_col.publish(self.make_rgb_msg(d_msg.header, pure_rgb))
        self.pub_rs_d_reg.publish(self.make_rgb_msg(d_msg.header, reg_rgb))


def read_exact(stream: Any, n: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < n:
        part = stream.read(n - len(chunks))
        if not part:
            break
        chunks.extend(part)
    return bytes(chunks)


def main() -> None:
    rclpy.init()
    node = DualCameraDepthBridgeNode()
    rs_python = os.environ.get("REALSENSE_PYTHON", "/home/bjoern/miniforge3/bin/python3")
    proc: subprocess.Popen[bytes] | None = None
    if os.path.exists(rs_python):
        sub_env = dict(os.environ)
        sub_env.pop("PYTHONPATH", None)
        sub_env.pop("LD_LIBRARY_PATH", None)
        proc = subprocess.Popen(
            [rs_python, "-c", RS_HELPER_CODE],
            stdout=subprocess.PIPE,
            env=sub_env,
        )
        assert proc.stdout is not None
        meta = read_exact(proc.stdout, 20)
        if len(meta) == 20:
            rw, rh, rfx, _, _ = struct.unpack("<IIfff", meta)
            print(f"RealSense D435i + ZED 2 Bridge active ({rw}x{rh}, fx={rfx:.1f}).", flush=True)

            def reader() -> None:
                assert proc is not None and proc.stdout is not None
                while rclpy.ok():
                    hdr = read_exact(proc.stdout, 20)
                    if len(hdr) != 20:
                        break
                    ts_ns, draw_len, dal_len, c_len = struct.unpack("<QIII", hdr)
                    d_raw = np.frombuffer(read_exact(proc.stdout, draw_len), dtype=np.float32).reshape(rh, rw).copy()
                    d_al = np.frombuffer(read_exact(proc.stdout, dal_len), dtype=np.float32).reshape(rh, rw).copy()
                    c_rgba = np.frombuffer(read_exact(proc.stdout, c_len), dtype=np.uint8).reshape(rh, rw, 4).copy()
                    node.enqueue_rs(ts_ns, d_raw, d_al, c_rgba)

            threading.Thread(target=reader, daemon=True).start()
        else:
            print("Warning: RealSense D435i stream unavailable; running ZED 2 stylizer only.", file=sys.stderr, flush=True)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if proc is not None:
            proc.terminate()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
