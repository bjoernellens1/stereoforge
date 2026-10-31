#!/usr/bin/env python3
"""Audit and replay saved StereoForge depth/RGBA sequences, without ROS or a GPU.

Fresh evidence and held history have separate coverage/age/provenance metrics.
No metric in this tool establishes geometric accuracy without ground truth.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from temporal_consensus_v2 import (
    ConsensusConfig, TemporalConsensus, ConsensusFrame, valid_depth,
    corrected_disparity, raw_frame, RAW_CURRENT, CONFIRMED_CURRENT, HELD_HISTORY)

SEQUENCE_SCHEMA = "stereoforge.ros_depth_sequence.v1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def percentile(values, p):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if a.size else None


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def frame_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError(f"invalid frame basename: {name!r}")
    path = root / name
    if path.is_symlink():
        raise ValueError(f"frame symlinks are not accepted: {path}")
    return path


def load_plane(root: Path, name: str, dtype, shape):
    path = frame_path(root, name)
    expected = math.prod(shape) * np.dtype(dtype).itemsize
    if not path.is_file() or path.stat().st_size != expected:
        raise ValueError(f"missing or wrong-sized payload: {path}; expected {expected} bytes")
    return np.fromfile(path, dtype).reshape(shape)


def sequence_info(root: Path, *, complete=True) -> dict:
    manifest = root / "manifest.json"
    info = json.loads(manifest.read_text())
    if info.get("schema") != SEQUENCE_SCHEMA:
        raise ValueError("expected stereoforge.ros_depth_sequence.v1")
    w, h = info.get("width"), info.get("height")
    if any(type(x) is not int or not 1 <= x <= 8192 for x in (w, h)):
        raise ValueError("invalid sequence geometry")
    rows = info.get("frames")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("a sequence needs at least two frames")
    last_stamp = last_received = 0
    seen = set()
    missing = []
    for i, row in enumerate(rows):
        if row.get("index") != i:
            raise ValueError("frame indices must be ordered and sequential")
        for key, previous in (("stamp_ns", last_stamp), ("depth_received_ns", last_received)):
            value = row.get(key)
            if type(value) is not int or not previous < value <= np.iinfo(np.int64).max:
                raise ValueError(f"invalid, duplicate or backward {key} at frame {i}")
        last_stamp, last_received = row["stamp_ns"], row["depth_received_ns"]
        for key in ("depth_file", "rgba_file"):
            name = row.get(key)
            if key == "rgba_file" and name is None:
                if complete:
                    raise ValueError("temporal replay requires paired RGBA for motion guards")
                continue
            p = frame_path(root, name)
            if name in seen:
                raise ValueError(f"reused payload filename: {name}")
            seen.add(name)
            if not p.is_file():
                missing.append(name)
            elif p.stat().st_size != w * h * 4:
                raise ValueError(f"incorrect payload size: {p}")
    if complete and missing:
        raise ValueError(f"{len(missing)} raw payloads missing; first: {missing[0]}. "
                         "Use the original recording directory. MP4/color maps cannot replace float depth.")
    return info


class Metrics:
    def __init__(self, shape, fxb):
        self.n = math.prod(shape)
        self.fxb = fxb
        self.count = np.zeros(shape, np.uint32)
        self.frames = 0
        self.rows = []
        self.pairs = []
        self.previous = self.prev_raw = None
        self.prev_prov = None
        self.prev_time = None

    def update(self, output: ConsensusFrame, raw: np.ndarray, receipt_ns: int):
        depth, prov = output.depth, output.provenance
        valid, rv = valid_depth(depth), valid_depth(raw)
        self.count += valid
        self.frames += 1
        q = corrected_disparity(depth, self.fxb)
        rq = corrected_disparity(raw, self.fxb)
        held = prov == HELD_HISTORY
        fresh = valid & ~held
        ages = output.age_ms[held]
        self.rows.append({
            "valid_pixels": int(valid.sum()), "raw_valid_pixels": int(rv.sum()),
            "current_supported_pixels": int(fresh.sum()),
            "confirmed_current_pixels": int((prov == CONFIRMED_CURRENT).sum()),
            "raw_fallback_pixels": int((prov == RAW_CURRENT).sum()),
            "held_pixels": int(held.sum()),
            "held_age_p95_ms": percentile(ages, 95),
            "held_age_max_ms": float(ages.max()) if ages.size else None,
            **output.diagnostics,
        })
        if self.previous is not None:
            pv, pr = valid_depth(self.previous), valid_depth(self.prev_raw)
            common = valid & pv
            union = valid | pv
            fresh_common = common & (prov != HELD_HISTORY) & (self.prev_prov != HELD_HISTORY) & rv & pr
            pq = corrected_disparity(self.previous, self.fxb)
            prq = corrected_disparity(self.prev_raw, self.fxb)
            # Check a previously held estimate against the next available raw observation.
            # This is internal consistency, NOT ground-truth accuracy.
            returning = (self.prev_prov == HELD_HISTORY) & rv
            return_delta = np.abs(pq[returning] - rq[returning])
            self.pairs.append({
                "valid_lost": int((pv & ~valid).sum()),
                "previous_valid": int(pv.sum()),
                "valid_gained": int((~pv & valid).sum()),
                "previous_invalid": int((~pv).sum()),
                "union_valid": int(union.sum()),
                "common_valid": int(common.sum()),
                "flip_all_pixels": float((pv ^ valid).mean()),
                "all_output_common_p90_px": percentile(np.abs(q[common] - pq[common]), 90),
                "fresh_common_pixels": int(fresh_common.sum()),
                "raw_same_fresh_support_p90_px": percentile(np.abs(rq[fresh_common] - prq[fresh_common]), 90),
                "output_same_fresh_support_p90_px": percentile(np.abs(q[fresh_common] - pq[fresh_common]), 90),
                "returning_after_hold_pixels": int(returning.sum()),
                "returning_after_hold_disagree_gt1px": int((return_delta > 1).sum()),
                "receipt_interval_ms": (receipt_ns - self.prev_time) / 1e6,
            })
        self.previous, self.prev_raw = depth.copy(), raw.copy()
        self.prev_prov, self.prev_time = prov.copy(), receipt_ns

    def finish(self) -> dict:
        if self.frames == 0:
            raise ValueError("no evaluation frames")
        sums = lambda k: sum(x[k] for x in self.rows)
        psums = lambda k: sum(x[k] for x in self.pairs)
        med = lambda k: percentile([x[k] for x in self.pairs if x[k] is not None], 50)
        total = self.n * self.frames
        ever = self.count > 0
        return {
            "evaluated_frames": self.frames,
            "valid_fraction_mean": sums("valid_pixels") / total,
            "current_supported_fraction_mean": sums("current_supported_pixels") / total,
            "confirmed_current_fraction_mean": sums("confirmed_current_pixels") / total,
            "held_fraction_mean": sums("held_pixels") / total,
            "current_raw_retention": ratio(sums("current_supported_pixels"), sums("raw_valid_pixels")),
            "ever_valid_fraction": float(ever.mean()),
            "never_valid_fraction": float((~ever).mean()),
            "persistent_80pct_fraction": float((self.count >= math.ceil(self.frames * .8)).mean()),
            "conditional_dropout_probability": ratio(psums("valid_lost"), psums("previous_valid")),
            "conditional_reappearance_probability": ratio(psums("valid_gained"), psums("previous_invalid")),
            "flip_fraction_all_pixels": float(np.mean([x["flip_all_pixels"] for x in self.pairs])) if self.pairs else None,
            "receipt_interval_p50_ms": percentile([x["receipt_interval_ms"] for x in self.pairs], 50),
            "receipt_interval_p95_ms": percentile([x["receipt_interval_ms"] for x in self.pairs], 95),
            "flip_fraction_among_pair_union": ratio(psums("valid_lost") + psums("valid_gained"), psums("union_valid")),
            "all_output_common_p90_px_median": med("all_output_common_p90_px"),
            "fresh_common_pixel_transitions": psums("fresh_common_pixels"),
            "raw_same_fresh_support_p90_px_median": med("raw_same_fresh_support_p90_px"),
            "output_same_fresh_support_p90_px_median": med("output_same_fresh_support_p90_px"),
            "held_age_max_ms": percentile([x["held_age_max_ms"] for x in self.rows if x["held_age_max_ms"] is not None], 100),
            "returning_after_hold_pixels": psums("returning_after_hold_pixels"),
            "returning_after_hold_disagree_gt1px_fraction": ratio(psums("returning_after_hold_disagree_gt1px"), psums("returning_after_hold_pixels")),
            "cpu_reference_p50_ms": percentile([x["cpu_reference_ms"] for x in self.rows if "cpu_reference_ms" in x], 50),
            "cpu_reference_p95_ms": percentile([x["cpu_reference_ms"] for x in self.rows if "cpu_reference_ms" in x], 95),
            "motion_or_unobservable_frame_resets": sum(x.get("global_motion_guard") in ("reset", "unobservable_texture") for x in self.rows),
            "gap_frame_resets": sum(x.get("gap_reset", False) for x in self.rows),
            "contradiction_reset_pixels": sum(x.get("contradiction_reset_pixels", 0) for x in self.rows),
            "frame_metrics": self.rows, "pair_metrics": self.pairs,
        }


def summarize_existing(report: dict, manifest: dict | None = None) -> dict:
    frames = report["frame_metrics"]
    pairs = report["pair_metrics"]
    if len(frames) < 2 or len(pairs) != len(frames) - 1:
        raise ValueError("report has incomplete frame/pair summaries")
    n = report["width"] * report["height"]
    prev = curr = lost = gained = shared = union = 0
    for i, pair in enumerate(pairs):
        if pair["from_frame"] != i or pair["to_frame"] != i + 1:
            raise ValueError("report pair identities are not sequential")
        a, b, c = frames[i]["valid_pixels"], frames[i + 1]["valid_pixels"], pair["common_valid_pixels"]
        if not 0 <= c <= min(a, b) <= n:
            raise ValueError("invalid report support counts")
        prev += a; curr += b; shared += c; lost += a - c; gained += b - c; union += a + b - c
    receive = [f["depth_received_ns"] for f in frames]
    header = [f["stamp_ns"] for f in frames]
    gaps = np.diff(np.array(header, dtype=np.int64)) / 1e6
    gap_summary = {"near_33ms": int(((gaps > 30) & (gaps < 37)).sum()),
                   "near_67ms": int(((gaps > 63) & (gaps < 70)).sum())}
    out = {
        "frames": len(frames),
        "valid_fraction_mean": report["valid_fraction_mean"],
        "ever_valid_fraction": report["pixel_ever_valid_fraction"],
        "never_valid_fraction": 1 - report["pixel_ever_valid_fraction"],
        "persistent_80pct_fraction": report["pixel_valid_80pct_frames_fraction"],
        "conditional_dropout_probability": ratio(lost, prev),
        "conditional_valid_survival": ratio(shared, prev),
        "pair_common_valid_fraction": shared / (n * len(pairs)),
        "pair_union_valid_fraction": union / (n * len(pairs)),
        "flip_fraction_among_pair_union": ratio(lost + gained, union),
        "median_framewise_disparity_p50_px": percentile([p["disparity_delta_p50_px"] for p in pairs if p["disparity_delta_p50_px"] is not None], 50),
        "median_framewise_disparity_p90_px": report["pair_disparity_delta_p90_px_median"],
        "median_framewise_depth_p50_m": percentile([p["depth_delta_p50_m"] for p in pairs if p["depth_delta_p50_m"] is not None], 50),
        "mean_high_disparity_boundary_fraction_of_valid": float(np.mean([f["at_high_disparity_boundary_fraction_of_valid"] for f in frames])),
        "receipt_fps_first_to_last": (len(frames) - 1) * 1e9 / (receive[-1] - receive[0]),
        "header_fps_first_to_last_unvalidated": (len(frames) - 1) * 1e9 / (header[-1] - header[0]),
        "header_gap_counts_unvalidated": gap_summary,
        "warning": "Receipt cadence is not GPU time; global RGB changes are not a proof of geometric stationarity.",
    }
    if manifest is not None:
        if len(manifest["frames"]) != len(frames):
            raise ValueError("manifest and report frame counts differ")
        if any(m["stamp_ns"] != f["stamp_ns"] for m, f in zip(manifest["frames"], frames)):
            raise ValueError("manifest/report source stamps differ")
        arrivals = [(r["rgb_received_ns"] - r["depth_received_ns"]) / 1e6
                    for r in manifest["frames"] if "rgb_received_ns" in r]
        out["rgb_minus_depth_receipt_p50_ms"] = percentile(arrivals, 50)
        out["rgb_minus_depth_receipt_p95_ms"] = percentile(arrivals, 95)
    return out


def audit(args):
    report_path = args.report.resolve()
    report = json.loads(report_path.read_text())
    root = args.sequence.resolve() if args.sequence else None
    manifest = sequence_info(root, complete=False) if root else None
    result = {"schema": "stereoforge.temporal_evidence_audit.v2",
              "source_report_sha256": digest(report_path),
              "summary": summarize_existing(report, manifest)}
    if root:
        available = [r for r in manifest["frames"] if frame_path(root, r["depth_file"]).is_file()]
        result["raw_payload_availability"] = {"depth_frames_present": len(available),
                                               "depth_frames_listed": len(manifest["frames"])}
        if available:
            row = available[-1]
            z = load_plane(root, row["depth_file"], "<f4", (manifest["height"], manifest["width"]))
            v = valid_depth(z); values = z[v]
            q = corrected_disparity(z, report["fxb_px_m"])
            result["last_available_raw_frame"] = {
                "index": row["index"], "sha256": digest(root / row["depth_file"]),
                "valid_fraction": float(v.mean()),
                "depth_percentiles_m": {str(k): percentile(values, k) for k in (0, 1, 10, 50, 90, 99, 100)},
                "corrected_disparity_le1_fraction_of_valid": float((q[v] <= 1).mean()),
                "depth_gt8m_fraction_of_valid": float((values > 8).mean()),
                "depth_gt16m_fraction_of_valid": float((values > 16).mean()),
                "note": "Far values are not automatically wrong: window views include outdoor surfaces. No ground truth supplied."}
    if args.preview:
        import cv2
        cap = cv2.VideoCapture(str(args.preview))
        if not cap.isOpened():
            raise ValueError("cannot open preview")
        count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        result["preview_metadata"] = {"frames": count, "encoded_fps": fps,
                                     "playback_speed_relative_to_receipt": fps / result["summary"]["receipt_fps_first_to_last"],
                                     "note": "MP4 is visualization only, not reconstructible numeric depth."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


def calibration(args):
    cx = 0.0
    if args.calibration_manifest:
        data = json.loads(args.calibration_manifest.read_text())["calibration"]
        fxb = abs(float(data["P2"][0][3]))
        cx = float(data["P1"][0][2]) - float(data["P2"][0][2])
    else:
        fxb = args.fxb
    if fxb is None or not math.isfinite(fxb) or fxb <= 0 or not math.isfinite(cx):
        raise ValueError("supply a valid calibration manifest or --fxb")
    return fxb, cx


def replay(args):
    if not args.assume_static:
        raise ValueError("this reference is static-camera only; explicitly pass --assume-static")
    root = args.sequence.resolve()
    info = sequence_info(root, complete=True)
    fxb, cx = calibration(args)
    shape = (info["height"], info["width"])
    config = ConsensusConfig(window_frames=args.window, min_hits=args.min_hits,
                             agreement_px=args.agreement_px, evidence_ms=args.evidence_ms,
                             dominance=args.dominance, hold_ms=100.,
                             global_guard=not args.no_global_guard)
    config.validate()
    configs = {"confirm": replace(config, hold_ms=0),
               "hold100": replace(config, hold_ms=100),
               "hold200": replace(config, hold_ms=200)}
    for c in configs.values():
        c.validate()
    if args.save_sequence != "none" and args.save_sequence not in configs:
        raise ValueError("invalid saved profile")
    rows = info["frames"]
    warmup = args.window - 1
    if len(rows) < warmup + 2:
        raise ValueError(f"need at least {warmup + 2} frames for window={args.window}")
    # Never touch source data or overwrite an earlier experiment.
    out = args.output_dir.resolve()
    if out == root or root in out.parents:
        # A child results directory is allowed; none of its files replace raw input.
        if out == root:
            raise ValueError("output must differ from the raw sequence directory")
    out.mkdir(parents=True, exist_ok=False)
    writer = None
    try:
        filters = {name: TemporalConsensus(shape, fxb, c) for name, c in configs.items()}
        metrics = {name: Metrics(shape, fxb) for name in ("raw", *filters)}
        full_raw = Metrics(shape, fxb)
        saved_rows = []
        saved = None
        if args.save_sequence != "none":
            saved = out / args.save_sequence
            saved.mkdir()
        fps = (len(rows) - 1) * 1e9 / (rows[-1]["depth_received_ns"] - rows[0]["depth_received_ns"])
        if args.preview:
            import cv2
            pw = min(640, shape[1]); ph = max(2, round(shape[0] * pw / shape[1]))
            ph -= ph % 2; pw -= pw % 2
            writer = cv2.VideoWriter(str(out / "comparison.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (pw * 3, ph))
            if not writer.isOpened():
                raise RuntimeError("could not create MP4 preview")
        for i, row in enumerate(rows):
            z = load_plane(root, row["depth_file"], "<f4", shape)
            rgb = load_plane(root, row["rgba_file"], np.uint8, (*shape, 4))
            outputs = {"raw": raw_frame(z)}
            for name, model in filters.items():
                outputs[name] = model.update(z, rgb, receipt_ns=row["depth_received_ns"], source_stamp_ns=row["stamp_ns"])
            full_raw.update(outputs["raw"], z, row["depth_received_ns"])
            if i >= warmup:
                for name, value in outputs.items():
                    metrics[name].update(value, z, row["depth_received_ns"])
            if saved is not None:
                value = outputs[args.save_sequence]
                prefix = f"frame-{i:05d}"
                depth_file, rgb_file = prefix + ".depth.f32", prefix + ".rgb.rgba8"
                value.depth.astype("<f4").tofile(saved / depth_file)
                shutil.copyfile(root / row["rgba_file"], saved / rgb_file)
                age_file, provenance_file = prefix + ".age_ms.f32", prefix + ".provenance.u8"
                value.age_ms.astype("<f4").tofile(saved / age_file)
                value.provenance.tofile(saved / provenance_file)
                saved_rows.append({**row, "depth_file": depth_file, "rgba_file": rgb_file,
                                   "age_file": age_file, "provenance_file": provenance_file})
            if writer is not None:
                import cv2
                panels = [cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)]
                for name in ("raw", "hold100"):
                    d = outputs[name].depth
                    v = valid_depth(d)
                    tone = np.zeros(shape, np.uint8)
                    tone[v] = np.rint(np.clip((d[v] - .5) / 7.5, 0, 1) * 255).astype(np.uint8)
                    color = cv2.applyColorMap(tone, cv2.COLORMAP_TURBO)
                    color[~v] = 0
                    panels.append(color)
                panels = [cv2.resize(p, (pw, ph), interpolation=cv2.INTER_AREA) for p in panels]
                for panel, label in zip(panels, ("RGB", "RAW", "CONSENSUS + <=100 ms HISTORY")):
                    cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1, cv2.LINE_AA)
                writer.write(np.concatenate(panels, axis=1))
            if (i + 1) % 15 == 0 or i + 1 == len(rows):
                print(f"processed {i + 1}/{len(rows)}", flush=True)
        if writer is not None:
            writer.release(); writer = None
        if saved is not None:
            derived = {**info, "frames": saved_rows,
                       "processing": {"algorithm": "static_causal_disparity_consensus_v2", "profile": args.save_sequence,
                                      "config": asdict(configs[args.save_sequence]),
                                      "source_manifest_sha256": digest(root / "manifest.json"),
                                      "age_clock": "time since last corroborated input RECEIPT; not exposure age",
                                      "provenance_codes": {"0": "invalid", "1": "raw current on motion reset",
                                                           "2": "confirmed current", "3": "held history"}},
                       "stats": {"written": len(rows), "scope": "offline derived output, not live recorder statistics"}}
            write_json(saved / "manifest.json", derived)
        result = {"schema": "stereoforge.temporal_consensus_ablation.v2",
                  "source_manifest_sha256": digest(root / "manifest.json"),
                  "fxb_px_m": fxb, "cx_offset_px": cx,
                  "assumption": "fixed-pixel stationary scene; RGB motion tests are guards, not pose compensation",
                  "comparison_warmup_frames_excluded": warmup,
                  "preview_fps": fps, "preview_timing": "constant FPS equal to mean receipt cadence; not exact variable-frame timing",
                  "configs": {k: asdict(c) for k, c in configs.items()},
                  "raw_full_sequence": full_raw.finish(),
                  "profiles": {k: v.finish() for k, v in metrics.items()},
                  "warnings": ["No geometric ground truth: lower jitter does not establish accuracy.",
                               "Held output is historical evidence, never counted as a new camera observation.",
                               "Static-camera prototype. Do not enable on moving robots.",
                               "CPU reference latency is not a native Vulkan performance measurement.",
                               "Current source consists of post-LR ROS depth and left RGB, not both stereo views."]}
        write_json(out / "summary.json", result)
        print_summary(result)
    except Exception as exc:
        if writer is not None:
            writer.release()
        write_json(out / "FAILED.json", {"error": str(exc), "partial_outputs_not_a_complete_recording": True})
        raise


def print_summary(result):
    print("profile       valid    fresh     held  dropout    p90-out  p90-raw*  p90-out*  age-max")
    def fmt(x, style):
        return "   n/a" if x is None else format(x, style)
    for name, p in result["profiles"].items():
        print("{:<12} {:>7} {:>8} {:>8} {:>8} {:>10} {:>9} {:>9} {:>8}".format(
            name, fmt(p["valid_fraction_mean"], ".1%"), fmt(p["current_supported_fraction_mean"], ".1%"),
            fmt(p["held_fraction_mean"], ".1%"), fmt(p["conditional_dropout_probability"], ".1%"),
            fmt(p["all_output_common_p90_px_median"], ".3f"),
            fmt(p["raw_same_fresh_support_p90_px_median"], ".3f"),
            fmt(p["output_same_fresh_support_p90_px_median"], ".3f"),
            fmt(p["held_age_max_ms"], ".1f")))
    print("* Raw and output jitter on identical fresh-current pair support; no held pixels. Disparities in px; age in ms.")
    print("Lower full-output jitter can result from discarding hard pixels or holding history. Inspect fresh coverage and paired-support metrics.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit", help="re-evaluate the existing JSON report; raw clip may be incomplete")
    a.add_argument("--report", type=Path, required=True)
    a.add_argument("--sequence", type=Path)
    a.add_argument("--preview", type=Path)
    a.add_argument("--output", type=Path, required=True)
    a.set_defaults(func=audit)
    p = sub.add_parser("replay", help="same-input raw/confirmed/held ablation; all raw frames required")
    p.add_argument("sequence", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    calibration_group = p.add_mutually_exclusive_group(required=True)
    calibration_group.add_argument("--calibration-manifest", type=Path)
    calibration_group.add_argument("--fxb", type=float)
    p.add_argument("--assume-static", action="store_true")
    p.add_argument("--window", type=int, default=9)
    p.add_argument("--min-hits", type=int, default=3)
    p.add_argument("--agreement-px", type=float, default=1.0)
    p.add_argument("--dominance", type=float, default=.6)
    p.add_argument("--evidence-ms", type=float, default=500.)
    p.add_argument("--no-global-guard", action="store_true", help="synthetic/diagnostic only; weakens motion rejection")
    p.add_argument("--save-sequence", choices=("none", "confirm", "hold100", "hold200"), default="none")
    p.add_argument("--preview", action="store_true")
    p.set_defaults(func=replay)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
