#!/usr/bin/env python3
"""Evaluate native Daxa GPU dumped planes (Step 5 full-SGM and Step 6 multiresolution refinement)
on the ZED 2 fixture using the Step 2-4 correspondence and geometric metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

from benchmarks.lr_confidence_audit_v6 import read_fixture_luma
from benchmarks.matching_cost_inspector_v8 import (
    aggregate_sgm_4path_vectorized,
    build_cost_volumes,
    geometric_qualification_metrics,
    select_from_aggregated,
)
from benchmarks.stereo_reference_crosscheck_v5 import lr_pair
from benchmarks.temporal_consensus_v2 import (
    CONFIRMED_CURRENT,
    HELD_HISTORY,
    ConsensusConfig,
    TemporalConsensus,
)
from benchmarks.compare_daxa_preprocess import read_capture_record

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs/cameras/zed2/fixtures/SN26041639-fixture/manifest.json"


def evaluate_depth_dump(prefix: Path, report_json: Path, luma_l: np.ndarray) -> dict[str, Any]:
    report = json.loads(report_json.read_text())
    h, w = int(report["height"]), int(report["width"])
    lr_disp = np.fromfile(str(prefix) + ".lr_disparity.f32", dtype="<f4").reshape(2, h, w)
    lr_val = np.fromfile(str(prefix) + ".lr_validity.u32", dtype="<u4").reshape(2, h, w) != 0
    metric_val = np.fromfile(str(prefix) + ".metric_validity.u32", dtype="<u4").reshape(2, h, w) != 0
    left_disp = np.fromfile(str(prefix) + ".left.disparity.f32", dtype="<f4").reshape(h, w)
    left_val = np.fromfile(str(prefix) + ".left.validity.u32", dtype="<u4").reshape(h, w) != 0
    right_disp = np.fromfile(str(prefix) + ".right.disparity.f32", dtype="<f4").reshape(h, w)
    right_val = np.fromfile(str(prefix) + ".right.validity.u32", dtype="<u4").reshape(h, w) != 0

    ys, xs = np.divmod(np.arange(h * w, dtype=np.int32), w)
    sel_l = left_val.reshape(-1)
    rx = np.rint(xs.astype(np.float64) - np.where(sel_l, left_disp.reshape(-1), 0.0)).astype(np.int32)
    inb = sel_l & (rx >= 0) & (rx < w)
    both = np.zeros(h * w, dtype=bool)
    both[inb] = right_val[ys[inb], rx[inb]]
    lr_err = np.full(h * w, np.nan, dtype=np.float64)
    idx_b = np.flatnonzero(both)
    lr_err[idx_b] = np.abs(left_disp.reshape(-1)[idx_b] - right_disp[ys[idx_b], rx[idx_b]])

    geom = geometric_qualification_metrics(lr_disp[0], lr_val[0], luma_l)
    return {
        "descriptor_mode": report.get("descriptor_mode", "census"),
        "stereo_schedule": report.get("stereo_schedule"),
        "right_reference_mode": report.get("right_reference_mode", "independent-sgm"),
        "left_valid_pct": round(100.0 * float(left_val.mean()), 2),
        "right_valid_pct": round(100.0 * float(right_val.mean()), 2),
        "lr_valid_left_pct": round(100.0 * float(lr_val[0].mean()), 2),
        "metric_valid_left_pct": round(100.0 * float(metric_val[0].mean()), 2),
        "lr_agreement_among_both_selected_pct": round(100.0 * float(lr_val[0].sum()) / max(int(both.sum()), 1), 2),
        "median_lr_error_both_selected_px": round(float(np.nanmedian(lr_err)), 3) if idx_b.size else None,
        "reference_checks": {
            "lr_mask_metadata_reference_checked": report.get("lr_mask_metadata_reference_checked"),
            "both_stereo_integer_reference_checked": report.get("both_stereo_integer_reference_checked"),
            "left_stereo_integer_reference_checked": report.get("left_stereo_integer_reference_checked"),
            "derived_right_mask_reference_checked": report.get("derived_right_mask_reference_checked"),
            "validation_errors_after_teardown": report.get("validation_errors_after_teardown"),
            "validation_warnings_after_teardown": report.get("validation_warnings_after_teardown"),
        },
        "requested_combined_buffer_bytes": report.get("requested_combined_buffer_bytes"),
        "gpu_combined_queue_span_ms": report.get("gpu_combined_queue_span_ms"),
        "host_wall_ms": report.get("host_wall_ms"),
        "geometry": geom,
    }


def evaluate_refinement_dump(prefix: Path, report_json: Path, luma_l: np.ndarray) -> dict[str, Any]:
    report = json.loads(report_json.read_text())
    h, w = int(report["height"]), int(report["width"])
    disp_file = Path(str(prefix) + ".disparity.f32")
    out: dict[str, Any] = {
        "schema": report.get("schema"),
        "descriptor_mode": report.get("descriptor_mode", "census"),
        "fast_mode": report.get("fast_mode", False),
        "coarse_width": report.get("coarse_width"),
        "coarse_height": report.get("coarse_height"),
        "coarse_disparities": report.get("coarse_disparities"),
        "validation_errors_after_teardown": report.get("validation_errors_after_teardown"),
        "validation_warnings_after_teardown": report.get("validation_warnings_after_teardown"),
        "requested_refinement_buffer_bytes": report.get("requested_refinement_buffer_bytes"),
        "requested_total_buffer_bytes": report.get("requested_total_buffer_bytes"),
        "timings_ms": report.get("timings_ms"),
        "host_wall_ms": report.get("host_wall_ms"),
        "enqueue_cpu_ms": report.get("enqueue_cpu_ms"),
        "wait_cpu_ms": report.get("wait_cpu_ms"),
        "all_30_planes_exact_between_variants": report.get("all_30_planes_exact_between_variants"),
    }
    if disp_file.exists():
        lr_disp = np.fromfile(str(prefix) + ".disparity.f32", dtype="<f4").reshape(2, h, w)
        lr_val = np.fromfile(str(prefix) + ".validity.u32", dtype="<u4").reshape(2, h, w) != 0
        metric_val = np.fromfile(str(prefix) + ".metric_validity.u32", dtype="<u4").reshape(2, h, w) != 0
        raw_disp = np.fromfile(str(prefix) + ".raw_disparity.f32", dtype="<f4").reshape(2, h, w)
        raw_val = np.fromfile(str(prefix) + ".raw_validity.u32", dtype="<u4").reshape(2, h, w) != 0

        ys, xs = np.divmod(np.arange(h * w, dtype=np.int32), w)
        sel_l = raw_val[0].reshape(-1)
        rx = np.rint(xs.astype(np.float64) - np.where(sel_l, raw_disp[0].reshape(-1), 0.0)).astype(np.int32)
        inb = sel_l & (rx >= 0) & (rx < w)
        both = np.zeros(h * w, dtype=bool)
        both[inb] = raw_val[1, ys[inb], rx[inb]]

        out.update(
            {
                "left_raw_valid_pct": round(100.0 * float(raw_val[0].mean()), 2),
                "right_raw_valid_pct": round(100.0 * float(raw_val[1].mean()), 2),
                "lr_valid_left_pct": round(100.0 * float(lr_val[0].mean()), 2),
                "metric_valid_left_pct": round(100.0 * float(metric_val[0].mean()), 2),
                "lr_agreement_among_both_selected_pct": round(100.0 * float(lr_val[0].sum()) / max(int(both.sum()), 1), 2),
                "geometry": geometric_qualification_metrics(lr_disp[0], lr_val[0], luma_l),
            }
        )
    return out


def evaluate_5frame_temporal_stability(manifest_path: Path) -> dict[str, Any]:
    """Compare 5-frame temporal stability on SN26041639-fixture/capture/capture.sfp
    between baseline census_24bit and ortho_hybrid_u32."""
    import cv2

    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    cap_path = root / manifest["capture"]["file"]
    lx = np.fromfile(root / manifest["maps"]["left_x"]["file"], dtype="<f4").reshape(720, 1280)
    ly = np.fromfile(root / manifest["maps"]["left_y"]["file"], dtype="<f4").reshape(720, 1280)
    rx = np.fromfile(root / manifest["maps"]["right_x"]["file"], dtype="<f4").reshape(720, 1280)
    ry = np.fromfile(root / manifest["maps"]["right_y"]["file"], dtype="<f4").reshape(720, 1280)
    fxb = 527.790283203125 * 0.11944399774074554

    frames_rgb_l = []
    frames_luma_l = []
    frames_luma_r = []
    stamps_ns = []
    for ordinal in range(5):
        meta, payload = read_capture_record(cap_path, ordinal)
        yuyv = np.frombuffer(payload, dtype=np.uint8).reshape(720, 2560, 2)
        rgb = cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUYV)
        rl = cv2.remap(rgb[:, :1280], lx, ly, cv2.INTER_LINEAR)
        rr = cv2.remap(rgb[:, 1280:], rx, ry, cv2.INTER_LINEAR)
        frames_rgb_l.append(rl)
        frames_luma_l.append(cv2.cvtColor(rl, cv2.COLOR_RGB2GRAY).astype(np.uint32))
        frames_luma_r.append(cv2.cvtColor(rr, cv2.COLOR_RGB2GRAY).astype(np.uint32))
        stamps_ns.append(int(meta["camera_timestamp_ns"]))

    results: dict[str, Any] = {}
    for mode in ("census_24bit", "ortho_hybrid_u32"):
        tc = TemporalConsensus((720, 1280), fxb=fxb, config=ConsensusConfig(window_frames=5, min_hits=2, hold_ms=100.0))
        raw_covs = []
        conf_covs = []
        total_covs = []
        dropouts = []
        prev_lr = None
        for idx in range(5):
            cl, cr = build_cost_volumes(frames_luma_l[idx], frames_luma_r[idx], 128, mode)
            al = aggregate_sgm_4path_vectorized(cl, p1=8, p2=32)
            ar = aggregate_sgm_4path_vectorized(cr, p1=8, p2=32)
            dl, _, lv = select_from_aggregated(al, non_adjacent_runner_up=False, uniqueness_ratio=0.0, min_margin=1, denom_floor=1)
            dr, _, rv = select_from_aggregated(ar, non_adjacent_runner_up=False, uniqueness_ratio=0.0, min_margin=1, denom_floor=1)
            lr_ok, _, _ = lr_pair(dl, dr, lv, rv)
            depth = np.where(lr_ok & (dl > 0), fxb / dl, np.nan).astype(np.float32)
            receipt_ns = stamps_ns[0] + idx * 16_666_667
            cf = tc.update(depth, frames_rgb_l[idx], receipt_ns=receipt_ns, source_stamp_ns=stamps_ns[idx])
            raw_covs.append(float(lr_ok.mean()))
            if prev_lr is not None:
                dropouts.append(float((prev_lr & ~lr_ok).sum() / max(int(prev_lr.sum()), 1)))
            prev_lr = lr_ok
            if idx >= 1:
                conf_covs.append(float((cf.provenance == CONFIRMED_CURRENT).mean()))
                total_covs.append(float(((cf.provenance == CONFIRMED_CURRENT) | (cf.provenance == HELD_HISTORY)).mean()))
        results[mode] = {
            "mean_raw_lr_valid_pct": round(100.0 * float(np.mean(raw_covs)), 2),
            "mean_frame_to_frame_raw_dropout_pct": round(100.0 * float(np.mean(dropouts)), 2),
            "mean_confirmed_current_pct": round(100.0 * float(np.mean(conf_covs)), 2),
            "mean_confirmed_plus_100ms_hold_pct": round(100.0 * float(np.mean(total_covs)), 2),
        }
    return results


def main() -> None:
    luma_l, _ = read_fixture_luma(DEFAULT_MANIFEST.parent)
    step5_dir = ROOT / "artifacts/depth-video/v8-review-milestone/step5-dumps"
    step5_results: dict[str, Any] = {}
    for name in (
        "depth_census_sequential",
        "depth_ortho-hybrid_sequential",
        "depth_census_layered",
        "depth_ortho-hybrid_layered",
        "depth_census_derived",
        "depth_ortho-hybrid_derived",
    ):
        rpath = step5_dir / f"{name}.json"
        if rpath.exists():
            step5_results[name] = evaluate_depth_dump(step5_dir / name, rpath, luma_l)

    out5 = ROOT / "artifacts/depth-video/v8-review-milestone/step5-native-ortho-hybrid-depth-benchmark.json"
    out5_doc = json.dumps({"schema": "stereoforge.native_ortho_hybrid_depth_v8.v1", "runs": step5_results}, indent=2) + "\n"
    out5.write_text(out5_doc)
    docs_dir = ROOT / "docs/evidence/v8-review-milestone"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / out5.name).write_text(out5_doc)

    step6_dir = ROOT / "artifacts/depth-video/v8-review-milestone/step6-dumps"
    step6_results: dict[str, Any] = {}
    for name in (
        "ref2x_census_layered_fast1",
        "ref2x_ortho-hybrid_layered_fast1",
        "ref2x_census_derived_fast1",
        "ref2x_ortho-hybrid_derived_fast1",
        "ref4x_census_layered_fast1",
        "ref4x_ortho-hybrid_layered_fast1",
        "ref4x_census_derived_fast1",
        "ref4x_ortho-hybrid_derived_fast1",
        "dual2x_ortho-hybrid_layered_fast1",
        "dual2x_ortho-hybrid_derived_fast1",
        "dual4x_ortho-hybrid_layered_fast1",
        "dual4x_ortho-hybrid_derived_fast1",
    ):
        rpath = step6_dir / f"{name}.json"
        if rpath.exists():
            step6_results[name] = evaluate_refinement_dump(step6_dir / name, rpath, luma_l)

    out6 = ROOT / "artifacts/depth-video/v8-review-milestone/step6-multiresolution-refinement-240fps-v8.json"
    temporal_5f = evaluate_5frame_temporal_stability(DEFAULT_MANIFEST)
    out6_doc = (
        json.dumps(
            {
                "schema": "stereoforge.multiresolution_refinement_v8.v1",
                "runs": step6_results,
                "temporal_5frame_capture_stability": temporal_5f,
            },
            indent=2,
        )
        + "\n"
    )
    out6.write_text(out6_doc)
    (docs_dir / out6.name).write_text(out6_doc)
    print(f"Wrote {out5} ({len(step5_results)} runs) and {out6} ({len(step6_results)} runs)")


if __name__ == "__main__":
    main()
