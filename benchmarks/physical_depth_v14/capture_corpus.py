#!/usr/bin/env python3
"""Record one immutable, session-labelled native ZED stereo scene.

RealSense is optional comparison evidence. With --realsense, save native Z16,
RGB and both IR streams, factory geometry, scale, device clocks and host ingress.
Legacy rs_depth.npy is aligned preview only. Tape distance is unqualified until
its datum, axial/ray semantics, target support and uncertainty are surveyed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import shutil
import json
import os
import subprocess
import time
from pathlib import Path


def file_sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def create_scene(root: Path, scene: str, rig_session: str) -> Path:
    if not rig_session or not rig_session.strip():
        raise ValueError("rig session is required; assign a new identity after any remount")
    if not scene or Path(scene).name != scene or scene in (".", ".."):
        raise ValueError("scene must be a single directory name")
    out = root / scene
    out.mkdir(parents=True, exist_ok=False)
    return out


def validate_native_calibration(calibration: dict, height: int) -> None:
    width = 1280 if height == 720 else 1920
    for eye in ('left', 'right'):
        camera = calibration.get(eye, {})
        if (camera.get('width'), camera.get('height')) != (width, height):
            raise ValueError('factory calibration does not match requested native profile')


def extrinsics_matrix(rotation, translation):
    """SDK rs2_extrinsics.rotation is column-major; mapping is depth -> colour."""
    import numpy as np
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation).reshape(3, 3, order="F")
    transform[:3, 3] = translation
    return transform.tolist()


def zed_capture(out: Path, frames: int, device: str, height: int = 720) -> None:
    import cv2
    import numpy as np

    width, fps = (1280, 60) if height == 720 else (1920, 30)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width * 2)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        for _ in range(20):
            if not cap.read()[0]:
                raise RuntimeError("ZED warmup read failed")
        raw = np.empty((frames, height, width * 2, 2), np.uint8)
        stamps, ingress = np.empty(frames, np.int64), np.empty(frames, np.int64)
        for i in range(frames):
            ok, frame = cap.read()
            arrival = time.monotonic_ns()
            if not ok or frame.size != height * width * 4 or frame.dtype != np.uint8:
                raise RuntimeError("ZED read failed or negotiated a different native profile")
            raw[i] = frame.reshape(height, width * 2, 2)
            stamps[i], ingress[i] = time.time_ns(), arrival
    finally:
        cap.release()
    np.save(out / "zed_raw.npy", raw)
    np.save(out / "zed_t_ns.npy", stamps)  # legacy realtime arrival clock, not acquisition
    np.save(out / "zed_ingress_monotonic_ns.npy", ingress)
    (out / "zed_timing.json").write_text(json.dumps({
        "acquisition_clock_qualified": False, "acquisition_timestamps": None,
        "ingress_clock": "host_monotonic_ns_after_read", "native_width": width,
        "native_height": height, "requested_fps": fps}))


def realsense_worker(out: Path, fps: int = 15) -> None:
    """Keep native geometry and SDK timestamps; aligned depth is a legacy preview."""
    import numpy as np
    import pyrealsense2 as rs

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, fps)
    cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, fps)
    cfg.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, fps)
    profile = pipe.start(cfg)  # fail explicitly if the link cannot carry this profile
    try:
        sensor = profile.get_device().first_depth_sensor()
        if sensor.supports(rs.option.emitter_enabled):
            sensor.set_option(rs.option.emitter_enabled, 1.0)
        scale = sensor.get_depth_scale()
        streams = {"depth": profile.get_stream(rs.stream.depth).as_video_stream_profile(),
                   "color": profile.get_stream(rs.stream.color).as_video_stream_profile(),
                   "ir_left": profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile(),
                   "ir_right": profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()}
        def intrinsics(stream):
            intr = stream.get_intrinsics()
            return dict(width=intr.width, height=intr.height, fx=intr.fx, fy=intr.fy,
                        ppx=intr.ppx, ppy=intr.ppy, model=str(intr.model), coeffs=list(intr.coeffs),
                        fps=stream.fps(), format=str(stream.format()))
        factory = {key: intrinsics(stream) for key, stream in streams.items()}
        factory.update(schema="stereoforge.native_realsense_factory.v1",
                       usb_type=profile.get_device().get_info(rs.camera_info.usb_type_descriptor), serial=profile.get_device().get_info(rs.camera_info.serial_number),
                       sdk_version=importlib.metadata.version("pyrealsense2"), depth_scale_m=scale,
                       native_depth_semantics="axial Z in native depth optical frame; uint16 times depth_scale_m",
                       aligned_depth_semantics="legacy preview only; SDK source semantics need installed-version verification")
        for key, stream in streams.items():
            ext = streams["depth"].get_extrinsics_to(stream)
            factory["T_" + key + "_from_depth"] = extrinsics_matrix(ext.rotation, ext.translation)
        (out / "rs_factory.json").write_text(json.dumps(factory, indent=2))
        align = rs.align(rs.stream.color)
        arrays = {key: [] for key in ("depth", "native_depth", "color", "ir_left", "ir_right")}
        stamps, timing = [], []
        for _ in range(30):
            pipe.wait_for_frames(timeout_ms=5000)
        (out / ".ready").write_text("ok")
        while not (out / ".stop").exists() or len(stamps) < 2:
            fs = pipe.wait_for_frames(timeout_ms=5000)
            arrival, realtime = time.monotonic_ns(), time.time_ns()
            native = {"depth": fs.get_depth_frame(), "color": fs.get_color_frame(),
                      "ir_left": fs.get_infrared_frame(1), "ir_right": fs.get_infrared_frame(2)}
            if not all(native.values()):
                raise RuntimeError("missing required native RealSense stream")
            arrays["native_depth"].append(np.asarray(native["depth"].get_data()).copy())
            for key in ("color", "ir_left", "ir_right"):
                arrays[key].append(np.asarray(native[key].get_data()).copy())
            arrays["depth"].append(np.asarray(align.process(fs).get_depth_frame().get_data()).astype(np.float32) * scale)
            stamps.append(realtime)
            timing.append(dict(host_ingress_monotonic_ns=arrival, streams={key: dict(
                timestamp_ms=frame.get_timestamp(), domain=str(frame.get_frame_timestamp_domain()),
                frame_number=frame.get_frame_number()) for key, frame in native.items()}))
    finally:
        pipe.stop()
    for key, values in arrays.items():
        np.save(out / ("rs_" + key + ".npy"), np.stack(values))
    np.save(out / "rs_t_ns.npy", np.asarray(stamps, np.int64))
    (out / "rs_timing.json").write_text(json.dumps({"clock_mapping_qualified": False, "frames": timing}))
    (out / "rs_intrinsics.json").write_text(json.dumps(dict(factory["color"], depth_scale=scale)))


def report_overlap(out: Path) -> None:
    import numpy as np

    zt, rt = np.load(out / "zed_t_ns.npy"), np.load(out / "rs_t_ns.npy")
    overlap = (min(zt[-1], rt[-1]) - max(zt[0], rt[0])) / 1e9
    print(f"ZED {(zt[-1] - zt[0]) / 1e9:.2f} s, RealSense {(rt[-1] - rt[0]) / 1e9:.2f} s, overlap {overlap:.2f} s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene")
    ap.add_argument("--out", default=str(Path.home() / "stereoforge_corpus"))
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--rig-session")
    ap.add_argument("--room")
    ap.add_argument("--scene-group")
    ap.add_argument("--split", choices=("development", "validation", "holdout"), default="development")
    ap.add_argument("--zed-calibration", type=Path)
    ap.add_argument("--zed-height", type=int, choices=(720, 1080), default=720)
    ap.add_argument("--tape-m", type=float)
    ap.add_argument("--realsense", action="store_true")
    ap.add_argument("--realsense-fps", type=int, choices=(6, 15, 30), default=15,
                    help="Explicit reference capture rate; 15 carries depth/RGB/both IR on the verified USB2 link")
    ap.add_argument("--realsense-worker", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.realsense_worker:
        realsense_worker(Path(args.realsense_worker), args.realsense_fps)
        return
    if args.frames < 2 or not args.scene or not args.rig_session or not args.zed_calibration or not args.room or not args.scene_group:
        ap.error("--scene, --rig-session, --room, --scene-group, --zed-calibration and at least two frames are required")
    calibration = Path(args.zed_calibration).read_bytes()
    validate_native_calibration(json.loads(calibration), args.zed_height)
    out = create_scene(Path(args.out), args.scene, args.rig_session)
    shutil.copyfile(args.zed_calibration, out / "zed_factory_calibration.json")
    metadata = dict(scene=args.scene, frames=args.frames, tape_m=args.tape_m,
                    rig_session=args.rig_session, room=args.room, scene_group=args.scene_group,
                    split=args.split, realsense=args.realsense, realsense_fps=args.realsense_fps if args.realsense else None,
                    zed_native_height=args.zed_height,
                    zed_calibration_sha256=hashlib.sha256(calibration).hexdigest(),
                    capture_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    absolute_reference_qualified=False, status="incomplete",
                    captured_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (out / "scene.json").write_text(json.dumps(metadata, indent=2))
    proc = None
    try:
        if args.realsense:
            py = os.environ.get("REALSENSE_PYTHON", "/home/bjoern/miniforge3/bin/python3")
            proc = subprocess.Popen([py, __file__, "--realsense-worker", str(out), "--realsense-fps", str(args.realsense_fps)])
            deadline = time.monotonic() + 30
            while not (out / ".ready").exists():
                if proc.poll() is not None:
                    raise RuntimeError("RealSense worker failed to start")
                if time.monotonic() >= deadline:
                    raise RuntimeError("RealSense startup timed out")
                time.sleep(0.1)
        zed_capture(out, args.frames, args.device, args.zed_height)
    finally:
        if proc is not None:
            (out / ".stop").write_text("ok")
            try:
                code = proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise RuntimeError("RealSense worker did not stop")
            if code != 0:
                raise RuntimeError("RealSense worker failed; scene remains incomplete")
            (out / ".ready").unlink(missing_ok=True)
            (out / ".stop").unlink(missing_ok=True)
    if args.realsense:
        report_overlap(out)
    metadata["status"] = "captured_unqualified"
    metadata["files_sha256"] = {path.name: file_sha256(path)
                                for path in out.iterdir() if path.is_file() and path.name != "scene.json"}
    (out / "scene.json").write_text(json.dumps(metadata, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
