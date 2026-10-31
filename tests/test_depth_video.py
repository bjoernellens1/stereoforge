"""Self-contained video diagnostics tests; ROS is not imported by these tests."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

BASE = Path(__file__).resolve().parents[1] / "benchmarks"


def module(name):
    spec = importlib.util.spec_from_file_location(name, BASE / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


capture = module("ros_depth_video_capture")
evaluate = module("evaluate_depth_video")


def image(array, encoding, stamp, frame_id="zed2"):
    h,w = array.shape[:2]
    return SimpleNamespace(encoding=encoding, width=w, height=h, step=w*4,
        is_bigendian=False, data=array.tobytes(),
        header=SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=stamp), frame_id=frame_id))


def make(root, depths, rgbs=None):
    w = capture.SequenceWriter(root, frames=len(depths), paired=rgbs is not None)
    for i,d in enumerate(depths):
        stamp = 100_000_000 + i * 33_333_333
        w.add(image(d.astype("<f4"), "32FC1", stamp), "depth", receive_ns=1_000_000_000+i*33_333_333)
        if rgbs is not None:
            w.add(image(rgbs[i].astype(np.uint8), "rgba8", stamp), "rgb", receive_ns=1_000_010_000+i*33_333_333)
    assert w.finish()["stats"]["written"] == len(depths)
    return root


def test_static_video_is_stable(tmp_path):
    depths = [np.full((6,8),2.0,dtype="<f4") for _ in range(4)]
    root = make(tmp_path / "steady", depths)
    summary = evaluate.analyze(root, fxb=60)
    assert summary["frames"] == 4
    assert summary["valid_fraction_mean"] == 1.0
    assert summary["pixel_valid_80pct_frames_fraction"] == 1.0
    assert summary["validity_switch_fraction_per_transition"] == 0.0
    assert summary["pair_disparity_delta_p90_px_median"] == 0.0
    assert all(x["disparity_jump_fraction"] == 0.0 for x in summary["pair_metrics"])


def test_temporal_flicker_and_disparity_noise(tmp_path):
    frames=[]
    for i in range(4):
        d=np.full((6,8), 2.0 if i%2==0 else 60/(30+2),dtype="<f4")
        d[:,0]=0 if i%2 else 2.0
        frames.append(d)
    root=make(tmp_path / "unstable",frames)
    s=evaluate.analyze(root, fxb=60)
    assert s["valid_fraction_mean"] == pytest.approx(0.9375)
    assert s["validity_switch_fraction_per_transition"] == pytest.approx(1/8)
    assert s["pixel_valid_80pct_frames_fraction"] == pytest.approx(7/8)
    assert s["pair_disparity_delta_p90_px_median"] == pytest.approx(2.0,rel=1e-5)
    assert all(x["disparity_jump_fraction"] == 1.0 for x in s["pair_metrics"])


def test_paired_capture_and_motion_gate(tmp_path):
    rng=np.random.default_rng(3)
    rgba=np.dstack((rng.integers(0,255,(32,48,3),dtype=np.uint8),
                    np.full((32,48),255,dtype=np.uint8)))
    frames=[np.full((32,48),1.5,dtype="<f4") for _ in range(3)]
    rgbs=[rgba,rgba,rgba]
    root=make(tmp_path / "paired",frames,rgbs)
    s=evaluate.analyze(root,alignment="flow",fxb=63.04)
    assert s["frames"]==3
    assert s["pair_disparity_jump_fraction_mean"] == 0
    assert s["pair_metrics"][0]["flow_trusted_fraction"]>.6
    assert "Optical-flow" in " ".join(s["warnings"])


def test_reject_bad_payload_duplicate_and_existing_dir(tmp_path):
    root=make(tmp_path/"single",[np.ones((4,7),np.float32)]*2)
    with pytest.raises(FileExistsError): capture.SequenceWriter(root)
    p=root/"frame-00000.depth.f32"
    p.write_bytes(b"bad")
    with pytest.raises(ValueError, match="payload length"):
        evaluate.analyze(root)
    writer=capture.SequenceWriter(tmp_path/"dups",frames=2,paired=False)
    img=image(np.ones((4,7),np.float32),"32FC1",100)
    writer.add(img,"depth",receive_ns=1000)
    writer.add(img,"depth",receive_ns=1000)
    assert len(writer.records)==1
    assert writer.duplicates==1
    img.step=3
    with pytest.raises(ValueError,match="tightly packed"):
        writer.add(img,"depth",receive_ns=1100)


def test_manifest_calibration_and_motion_warning(tmp_path):
    calibration=tmp_path/"calib.json"
    calibration.write_text(json.dumps({"calibration":{"P2":[[530,0,640,-63.04],[0,530,360,0],[0,0,1,0]]}}))
    assert evaluate.load_fxb(calibration) == pytest.approx(63.04)
    depths=[np.full((8,10),2.0,np.float32) for _ in range(2)]
    rgb0=np.zeros((8,10,4),np.uint8)
    rgb1=np.full((8,10,4),255,np.uint8)
    root=make(tmp_path/"motion",depths,[rgb0,rgb1])
    summary=evaluate.analyze(root,calibration_manifest=calibration)
    assert any("RGB changes strongly" in w for w in summary["warnings"])
    assert summary["fxb_px_m"]==pytest.approx(63.04)
