#!/usr/bin/env python3
"""Measure observed temporal 32FC1 support stability; not metric accuracy.

Modes: pixel (stationary-camera only), flow (RGB optical-flow alignment,
exploratory under motion: not SE(3) depth reprojection). Ground truth and raw
stereo are NOT part of ROS depth publications. Report validity flicker,
disparity-domain jitter and frame receipt cadence without filtering pixels.
"""
import argparse
import json
from pathlib import Path
import numpy as np

SCHEMA = "stereoforge.ros_depth_sequence.v1"


def percentile(data, q):
    a = np.asarray(data)
    return float(np.percentile(a, q)) if a.size else None


def load_fxb(calibration_manifest):
    if calibration_manifest is None:
        return None
    info = json.loads(Path(calibration_manifest).read_text())
    p2 = info["calibration"]["P2"]
    value = abs(float(p2[0][3]))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("invalid rectified focal-baseline product")
    return value


def _file(root, name, dtype, shape):
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("invalid or nonlocal frame path")
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"frame missing or symlink: {name}")
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"incorrect payload length: {name}")
    return np.fromfile(path, dtype=dtype).reshape(shape)


def _metric_pair(previous, current, a, b, fxb, tolerance_px):
    common = a & b
    if not common.any():
        return {"common_valid_pixels": 0, "depth_delta_p50_m": None,
                "depth_delta_p90_m": None, "disparity_delta_p50_px": None,
                "disparity_delta_p90_px": None, "disparity_jump_fraction": None}
    x, y = previous[common].astype(np.float64), current[common].astype(np.float64)
    delta = np.abs(x-y)
    out = {"common_valid_pixels": int(x.size),
           "depth_delta_p50_m": percentile(delta, 50),
           "depth_delta_p90_m": percentile(delta, 90),
           "disparity_delta_p50_px": None,
           "disparity_delta_p90_px": None,
           "disparity_jump_fraction": None}
    if fxb is not None:
        disparity_jump = fxb * np.abs(1/x - 1/y)
        out["disparity_delta_p50_px"] = percentile(disparity_jump, 50)
        out["disparity_delta_p90_px"] = percentile(disparity_jump, 90)
        out["disparity_jump_fraction"] = float((disparity_jump > tolerance_px).mean())
    return out


def _flow_align(previous_rgb, rgb, current):
    import cv2
    gray_prev = cv2.cvtColor(previous_rgb, cv2.COLOR_RGBA2GRAY)
    gray_curr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_curr, None,
                                       0.5, 3, 21, 3, 5, 1.2, 0)
    backward = cv2.calcOpticalFlowFarneback(gray_curr, gray_prev, None,
                                           0.5, 3, 21, 3, 5, 1.2, 0)
    h,w = current.shape
    yy,xx = np.indices((h,w), dtype=np.float32)
    mapx = xx+flow[:,:,0]
    mapy = yy+flow[:,:,1]
    # Avoid interpolating across depth discontinuities; depth is sampled nearest.
    warped = cv2.remap(current, mapx, mapy, interpolation=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    bx = cv2.remap(backward[:,:,0], mapx, mapy, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT)
    by = cv2.remap(backward[:,:,1], mapx, mapy, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT)
    e = np.sqrt((flow[:,:,0]+bx)**2+(flow[:,:,1]+by)**2)
    gray_warped = cv2.remap(gray_curr, mapx, mapy, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
    photometric = np.abs(gray_prev.astype(np.float32) - gray_warped.astype(np.float32))
    in_bounds = (mapx>=0)&(mapx<w-1)&(mapy>=0)&(mapy<h-1)
    trust = in_bounds & (e<=1.5) & (photometric<=25)
    flow_norm = np.linalg.norm(flow, axis=2)
    return warped, trust, {"flow_p50_px": percentile(flow_norm,50),
                           "flow_p90_px": percentile(flow_norm,90),
                           "flow_trusted_fraction":float(trust.mean())}


def analyze(root, *, alignment="pixel", calibration_manifest=None, fxb=None,
            disparity_tolerance_px=1.0, max_disparity=128, heatmap=None, preview=None):
    root = Path(root)
    info = json.loads((root/"manifest.json").read_text())
    if info.get("schema") != SCHEMA:
        raise ValueError("unknown sequence schema")
    w,h = info["width"],info["height"]
    if not (isinstance(w,int) and isinstance(h,int) and 1<=w<=8192 and 1<=h<=8192):
        raise ValueError("invalid sequence dimensions")
    rows = info["frames"]
    if len(rows)<2:
        raise ValueError("at least two frames required")
    if alignment not in ("pixel","flow"):
        raise ValueError("choose pixel or flow alignment")
    if not (np.isfinite(disparity_tolerance_px) and disparity_tolerance_px>0):
        raise ValueError("disparity tolerance must be positive")
    if fxb is None:
        fxb = load_fxb(calibration_manifest)
    if fxb is not None and (not np.isfinite(fxb) or fxb<=0):
        raise ValueError("invalid fxb")
    valid_count = np.zeros((h,w), dtype=np.uint32)
    transitions = np.zeros((h,w), dtype=np.uint32)
    summaries = []
    pairs = []
    previous, previous_valid, previous_rgb, previous_row = None,None,None,None
    rgba_video = None
    preview_writer = None
    for i,row in enumerate(rows):
        if row.get("index") != i:
            raise ValueError("non-sequential frame identities")
        depth = _file(root,row["depth_file"],"<f4",(h,w))
        valid = np.isfinite(depth)&(depth>0)
        valid_count+=valid.astype(np.uint32)
        rgb = None
        if "rgba_file" in row:
            rgb = _file(root,row["rgba_file"],np.uint8,(h,w,4))
        if alignment=="flow" and rgb is None:
            raise ValueError("flow mode requires paired rectified RGB")
        if previous_row is not None:
            if row["stamp_ns"]<=previous_row["stamp_ns"] or row["depth_received_ns"]<=previous_row["depth_received_ns"]:
                raise ValueError("timestamps not increasing")
            transitions+=(previous_valid ^ valid).astype(np.uint32)
        valid_depth=depth[valid]
        boundary = None
        if fxb is not None and valid_depth.size:
            boundary=float(np.mean(fxb/valid_depth >= max_disparity-1.5))
        summaries.append({"frame":i, "valid_fraction":float(valid.mean()),
                          "valid_pixels":int(valid.sum()),
                          "zero_pixels":int((depth==0).sum()),
                          "nonfinite_pixels":int((~np.isfinite(depth)).sum()),
                          "depth_p50_m":percentile(valid_depth,50),
                          "depth_p90_m":percentile(valid_depth,90),
                          "at_high_disparity_boundary_fraction_of_valid":boundary,
                          "stamp_ns":row["stamp_ns"],"depth_received_ns":row["depth_received_ns"]})
        if previous is not None:
            test_depth, trust, flow_stats = depth, np.ones_like(valid), {}
            gray_difference=None
            if rgb is not None and previous_rgb is not None:
                import cv2
                a=cv2.cvtColor(previous_rgb,cv2.COLOR_RGBA2GRAY)
                b=cv2.cvtColor(rgb,cv2.COLOR_RGBA2GRAY)
                gray_difference=percentile(np.abs(a.astype(np.int16)-b.astype(np.int16)),90)
            if alignment=="flow":
                test_depth,trust,flow_stats=_flow_align(previous_rgb,rgb,depth)
            compare=_metric_pair(previous,test_depth,
                                 previous_valid&trust, (np.isfinite(test_depth)&(test_depth>0))&trust,
                                 fxb,disparity_tolerance_px)
            flip=float((previous_valid^valid).mean()) if alignment=="pixel" else None
            pair={"from_frame":i-1,"to_frame":i,
                  "receipt_interval_ms":(row["depth_received_ns"]-previous_row["depth_received_ns"])/1e6,
                  "header_interval_ms":(row["stamp_ns"]-previous_row["stamp_ns"])/1e6,
                  "raw_valid_flip_fraction":float((previous_valid^valid).mean()),
                  "pixel_aligned_valid_flip_fraction":flip,
                  "rgb_change_p90_gray":gray_difference,
                  **flow_stats,**compare}
            pairs.append(pair)
        if preview is not None:
            import cv2
            if preview_writer is None:
                width = w*2 if rgb is not None else w
                Path(preview).parent.mkdir(parents=True,exist_ok=True)
                if Path(preview).exists():raise FileExistsError(preview)
                preview_writer = cv2.VideoWriter(str(preview),cv2.VideoWriter_fourcc(*"mp4v"),
                                                 30.0,(width,h))
                if not preview_writer.isOpened():raise RuntimeError("VideoWriter failed to open output")
            # Fixed display scale: 0.5–8.0 m avoids artificial per-frame flicker.
            values=np.clip((depth-0.5)/7.5,0,1)
            values[~valid]=0
            depth_color=cv2.applyColorMap((values*255).astype(np.uint8),cv2.COLORMAP_TURBO)
            depth_color[~valid]=0
            display=np.concatenate([cv2.cvtColor(rgb,cv2.COLOR_RGBA2BGR),depth_color],axis=1) if rgb is not None else depth_color
            preview_writer.write(display)
        previous,previous_valid,previous_rgb,previous_row=depth,valid,rgb,row
    if preview_writer is not None:
        preview_writer.release()
    if heatmap is not None:
        import cv2
        p=Path(heatmap)
        if p.exists():raise FileExistsError(p)
        p.parent.mkdir(parents=True,exist_ok=True)
        rate=transitions.astype(np.float32)/(len(rows)-1)
        cv2.imwrite(str(p),cv2.applyColorMap((rate*255).astype(np.uint8),cv2.COLORMAP_TURBO))
    avg_valid=float(np.mean([x["valid_fraction"] for x in summaries]))
    ever_valid_fraction=float(np.mean(valid_count>0))
    persistent_eighty=int(np.count_nonzero(valid_count >= np.ceil(len(rows)*.8)))
    rgb_p90=[x["rgb_change_p90_gray"] for x in pairs if x["rgb_change_p90_gray"] is not None]
    warn=[]
    if alignment=="pixel" and rgb_p90 and percentile(rgb_p90,50)>20:
        warn.append("RGB changes strongly: pixel-aligned depth differences mix scene/camera motion with temporal instability")
    if alignment=="flow":
        warn.append("Optical-flow alignment is not 3D pose compensation; depth changes with camera translation, and moving objects violate static-scene assumptions")
    if fxb is None:
        warn.append("No calibrated focal-baseline product: disparity-domain instability and saturation unavailable")
    warn.append("Temporal consistency is not depth accuracy; stable incorrect disparities can pass")
    warn.append("ROS header stamps were not validated against physical capture time")
    result={"schema":"stereoforge.depth_video_quality.v1", "frames":len(rows),
            "width":w,"height":h,"alignment":alignment,"fxb_px_m":fxb,
            "valid_fraction_mean":avg_valid,
            "valid_fraction_min":float(min(x["valid_fraction"] for x in summaries)),
            "valid_fraction_max":float(max(x["valid_fraction"] for x in summaries)),
            "pixel_ever_valid_fraction":ever_valid_fraction,
            "pixel_valid_80pct_frames_fraction":persistent_eighty/(h*w),
            "persistent_80pct_fraction_among_ever_valid":(persistent_eighty/int(np.count_nonzero(valid_count>0)) if ever_valid_fraction else None),
            "validity_switch_fraction_per_transition":float(transitions.mean()/(len(rows)-1)),
            "pair_depth_delta_p90_m_median":percentile([x["depth_delta_p90_m"] for x in pairs if x["depth_delta_p90_m"] is not None],50),
            "pair_disparity_delta_p90_px_median":percentile([x["disparity_delta_p90_px"] for x in pairs if x["disparity_delta_p90_px"] is not None],50),
            "pair_disparity_jump_fraction_mean":float(np.mean([x["disparity_jump_fraction"] for x in pairs if x["disparity_jump_fraction"] is not None])) if any(x["disparity_jump_fraction"] is not None for x in pairs) else None,
            "rgb_change_p90_gray_median":percentile(rgb_p90,50),
            "receipt_interval_p50_ms":percentile([x["receipt_interval_ms"] for x in pairs],50),
            "receipt_interval_p95_ms":percentile([x["receipt_interval_ms"] for x in pairs],95),
            "mean_boundary_fraction_of_valid":percentile([x["at_high_disparity_boundary_fraction_of_valid"] for x in summaries if x["at_high_disparity_boundary_fraction_of_valid"] is not None],50),
            "warnings":warn,"frame_metrics":summaries,"pair_metrics":pairs}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("sequence",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--alignment",choices=["pixel","flow"],default="pixel")
    p.add_argument("--calibration-manifest",type=Path)
    p.add_argument("--fxb",type=float)
    p.add_argument("--max-disparity",type=int,default=128)
    p.add_argument("--tolerance-px",type=float,default=1.0)
    p.add_argument("--heatmap",type=Path)
    p.add_argument("--preview",type=Path)
    a=p.parse_args()
    if a.output.exists():p.error("output exists; choose a fresh path")
    result=analyze(a.sequence,alignment=a.alignment,calibration_manifest=a.calibration_manifest,
                   fxb=a.fxb, disparity_tolerance_px=a.tolerance_px,
                   max_disparity=a.max_disparity,heatmap=a.heatmap,preview=a.preview)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print("frames={} valid_mean={:.1%} persistent_80pct={:.1%} flip={:.1%} |disp_delta|_p90_median={} px".format(
        result["frames"],result["valid_fraction_mean"],
        result["pixel_valid_80pct_frames_fraction"],
        result["validity_switch_fraction_per_transition"],
        result["pair_disparity_delta_p90_px_median"]))
    for warning in result["warnings"]:print("CAUTION:",warning)
    print("Report:",a.output)


if __name__ == "__main__":
    main()
