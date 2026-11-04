#!/usr/bin/env python3
"""ROS 2 Dual-Camera (ZED 2 + RealSense D435i) Publisher & Colorized Depth Bridge.

Spawns the RealSense D435i pyrealsense2 capture worker in an isolated Python
environment (stripping PYTHONPATH and LD_LIBRARY_PATH so Pixi Python 3.14
site-packages never collide with miniforge3 Python 3.13 numpy/pyrealsense2)
and publishes both raw metric depth (32FC1) and colorized/RGB-registered views.
"""
from __future__ import annotations

import array
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

cv2.setNumThreads(2)

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


def build_natural_color_1d_lut() -> np.ndarray:
    """Precompute a 256x1x3 1D per-channel RGB lookup table for fast OpenCV AVX2 cv2.LUT."""
    x = np.arange(256, dtype=np.float32) / 255.0
    lifted = np.power(x, 0.78)
    gains = np.array([0.96, 1.00, 1.14], dtype=np.float32)
    chans = []
    for g in gains:
        v = lifted * g
        v = v / (1.0 + 0.16 * np.maximum(0.0, v - 0.75))
        chans.append(np.clip(v * 255.0, 0.0, 255.0).astype(np.uint8))
    return np.stack(chans, axis=-1).reshape(256, 1, 3)


def build_rgb_colormap_luts() -> dict[str, np.ndarray]:
    """Precompute 256x1x3 RGB colormaps (inverted so 1=near/warm, 255=far/cool, 0=invalid dark gray)."""
    ramp = np.arange(255, -1, -1, dtype=np.uint8).reshape(256, 1)
    luts: dict[str, np.ndarray] = {}
    for name, cid in COLORMAPS.items():
        bgr = cv2.applyColorMap(ramp, cid)
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        rgb[0] = (20, 20, 20)
        luts[name] = rgb
    return luts


NATURAL_LUT_1D = build_natural_color_1d_lut()
RGB_COLORMAP_LUTS = build_rgb_colormap_luts()


def restore_zed2_natural_rgb(rgba: np.ndarray, dst: np.ndarray | None = None) -> np.ndarray:
    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB, dst=dst)
    return cv2.LUT(rgb, NATURAL_LUT_1D, dst=rgb)


def stylize_depth_rgb(
    depth_m: np.ndarray,
    base_rgb: np.ndarray | None,
    cmap_name: str,
    register_rgb: bool,
    alpha: float,
    vmin: float,
    vmax: float,
    dst_col: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    cmap_lut = RGB_COLORMAP_LUTS.get(cmap_name.lower(), RGB_COLORMAP_LUTS["turbo"])
    z_floor = max(vmin, 0.22)
    span = max(vmax - z_floor, 0.05)
    scale = 254.0 / span
    offset = 1.0 - z_floor * scale

    # Single AVX2 pass: NaN and <= z_floor saturate to 0; [z_floor, vmax] -> [1, 255]
    u8 = cv2.addWeighted(depth_m, scale, depth_m, 0.0, offset, dtype=cv2.CV_8U)
    col_rgb = cv2.applyColorMap(u8, cmap_lut, dst=dst_col)

    if register_rgb and base_rgb is not None and base_rgb.shape[:2] == depth_m.shape:
        blended = cv2.addWeighted(base_rgb, 1.0 - alpha, col_rgb, alpha, 0.0)
        reg_rgb = base_rgb.copy()
        cv2.copyTo(blended, u8, reg_rgb)
    else:
        reg_rgb = None
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
            depth=1,
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
        self.msg_pool: dict[str, tuple[Image, np.ndarray]] = {}
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

    def get_pooled_rgb_msg(self, key: str, header: Any, h: int, w: int) -> tuple[Image, np.ndarray]:
        entry = self.msg_pool.get(key)
        if entry is None or entry[0].height != h or entry[0].width != w:
            msg = Image()
            msg.height = h
            msg.width = w
            msg.encoding = "rgb8"
            msg.is_bigendian = 0
            msg.step = w * 3
            buf = array.array("B", bytes(h * w * 3))
            msg._data = buf
            view = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            entry = (msg, view)
            self.msg_pool[key] = entry
        entry[0].header = header
        return entry

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
        buf = array.array("B")
        buf.frombytes(np.ascontiguousarray(rgb).tobytes())
        msg._data = buf
        return msg

    def on_zed_rgb(self, msg: Image) -> None:
        need_nat = (
            self.pub_zed_rgb_nat.get_subscription_count() > 0
            or self.pub_zed_d_reg.get_subscription_count() > 0
            or self.pub_zed_dc_reg.get_subscription_count() > 0
        )
        if not need_nat:
            return
        raw_rgba = np.frombuffer(msg._data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        out_msg, out_view = self.get_pooled_rgb_msg("zed_nat", msg.header, msg.height, msg.width)
        restore_zed2_natural_rgb(raw_rgba, dst=out_view)
        self.latest_zed_nat_rgb = out_view
        if self.pub_zed_rgb_nat.get_subscription_count() > 0:
            self.pub_zed_rgb_nat.publish(out_msg)

    def _publish_stylized_zed_depth(self, msg: Image, pool_key: str, pub_col: Any, pub_reg: Any) -> None:
        need_col = pub_col.get_subscription_count() > 0
        need_reg = pub_reg.get_subscription_count() > 0
        if not need_col and not need_reg:
            return
        d_m = np.frombuffer(msg._data, dtype=np.float32).reshape(msg.height, msg.width)
        cmap, reg, alpha, zmin, zmax = self.params()
        out_msg, out_view = self.get_pooled_rgb_msg(pool_key, msg.header, msg.height, msg.width)
        _, reg_rgb = stylize_depth_rgb(
            d_m, self.latest_zed_nat_rgb, cmap, reg and need_reg, alpha, zmin, zmax, dst_col=out_view
        )
        if need_col:
            pub_col.publish(out_msg)
        if need_reg and reg_rgb is not None:
            pub_reg.publish(self.make_rgb_msg(msg.header, reg_rgb))

    def on_zed_depth(self, msg: Image) -> None:
        self._publish_stylized_zed_depth(msg, "zed_d_col", self.pub_zed_d_col, self.pub_zed_d_reg)

    def on_zed_depth_completed(self, msg: Image) -> None:
        self._publish_stylized_zed_depth(msg, "zed_dc_col", self.pub_zed_dc_col, self.pub_zed_dc_reg)

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
        if self.pub_rs_depth.get_subscription_count() > 0:
            d_msg.height = h
            d_msg.width = w
            d_msg.encoding = "32FC1"
            d_msg.is_bigendian = 0
            d_msg.step = w * 4
            d_msg.data = d_raw_m.tobytes()
            self.pub_rs_depth.publish(d_msg)

        rs_rgb_msg, rs_rgb_view = self.get_pooled_rgb_msg("rs_rgb", d_msg.header, h, w)
        cv2.cvtColor(c_rgba, cv2.COLOR_RGBA2RGB, dst=rs_rgb_view)
        if self.pub_rs_rgb.get_subscription_count() > 0:
            self.pub_rs_rgb.publish(rs_rgb_msg)
        need_col = self.pub_rs_d_col.get_subscription_count() > 0
        need_reg = self.pub_rs_d_reg.get_subscription_count() > 0
        if need_col or need_reg:
            cmap, reg, alpha, zmin, zmax = self.params()
            if need_col:
                rs_col_msg, rs_col_view = self.get_pooled_rgb_msg("rs_d_col", d_msg.header, h, w)
                stylize_depth_rgb(d_raw_m, None, cmap, False, alpha, zmin, zmax, dst_col=rs_col_view)
                self.pub_rs_d_col.publish(rs_col_msg)
            if need_reg:
                _, reg_rgb = stylize_depth_rgb(d_al_m, rs_rgb_view, cmap, reg, alpha, zmin, zmax)
                if reg_rgb is not None:
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
