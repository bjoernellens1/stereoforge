"""Read-only v11 promotion screen: real-frame coverage and reference disagreement.

This is a *post-hoc development diagnostic*, not sealed-test qualification.
It must not alter or invalidate the separate v11 coarse-stage performance pass.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "benchmarks/results/v11_0_qualification_and_coarse_replacement.json"
PIXELS_720P = 1280 * 720

# These are *proposed future development screens*, selected after seeing v11.
# They are not pre-registered gates and must be frozen on new development data.
MAX_COVERAGE_DROP_PP = 1.0
MAX_ZNCC_BAD2_INCREASE_PP = 0.5
MAX_NEWLY_GAINED_ZNCC_BAD2_PCT = 10.0


def _wilson_interval(success_count: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0 or success_count < 0 or success_count > n:
        raise ValueError("invalid binomial count")
    p = success_count / n
    denom = 1.0 + z*z / n
    mid = (p + z*z / (2*n)) / denom
    half = z * math.sqrt(p * (1-p) / n + z*z / (4*n*n)) / denom
    return (mid-half, mid+half)


def assess(data: dict[str, Any]) -> dict[str, Any]:
    zed = data["v11_0_ablation_zed2_cohorts_and_gpu_distributions"]["configs"]
    a0 = zed["A0_quality_layered_sgm_r4"]
    a7 = zed["A7_v11_fast_hierarchical_1to8_fallback"]
    syn = data["v11_0_ablation_synthetic_and_counterexamples"]
    a0_syn = syn["A0_quality_layered_sgm_r4"]
    a7_syn = syn["A7_v11_fast_hierarchical_1to8_fallback"]
    near = data["v11_0_near_field_d_coarse_64_qualification"]["measurements"]["d_coarse_64"]
    new = a7["measured_cohorts_vs_A0"]["newly_gained"]
    lost = a7["measured_cohorts_vs_A0"]["lost_evaluated_on_A0"]
    p95_meas = a7["gpu_measured_only"]["gpu_combined_queue_span_ms"]["p95"]
    p95_planar = a7["gpu_with_planar"]["gpu_combined_queue_span_ms"]["p95"]
    measured_delta_pp = 100 * (int(a7["measured_valid_pixels"]) - int(a0["measured_valid_pixels"])) / PIXELS_720P
    completed_delta_pp = 100 * (int(a7["completed_valid_pixels"]) - int(a0["completed_valid_pixels"])) / PIXELS_720P
    a0_zncc = a0["measured_overall_vs_zncc"]
    a7_zncc = a7["measured_overall_vs_zncc"]
    zncc_delta_pp = float(a7_zncc["vs_zncc_bad_2px_pct"]) - float(a0_zncc["vs_zncc_bad_2px_pct"])
    scene_deltas_pp = {
        scene: round(
            float(a7_syn["synthetic_per_scene"][scene]["unconditional_good_2px_pct"])
            - float(a0_syn["synthetic_per_scene"][scene]["unconditional_good_2px_pct"]), 3)
        for scene in a0_syn["synthetic_per_scene"]
    }
    flags = {
        "measured_coverage_dropped": measured_delta_pp < -MAX_COVERAGE_DROP_PP,
        "completed_coverage_dropped": completed_delta_pp < -MAX_COVERAGE_DROP_PP,
        "zncc_bad2_increased": zncc_delta_pp > MAX_ZNCC_BAD2_INCREASE_PP,
        "newly_gained_reference_risk": (
            int(new["zncc_overlap_pixels"]) < 1000
            or new["vs_zncc_bad_2px_pct"] is None
            or float(new["vs_zncc_bad_2px_pct"]) > MAX_NEWLY_GAINED_ZNCC_BAD2_PCT
        ),
        "synthetic_scene_regression": any(delta < -0.5 for delta in scene_deltas_pp.values()),
    }
    n = int(near["vs_zncc_d_gt_127_5_overlap_pixels"])
    bad_rate = near["vs_zncc_d_gt_127_5_bad_2px_pct"]
    estimated_bad_count = round(n * float(bad_rate) / 100) if n and bad_rate is not None else 0
    ci = _wilson_interval(estimated_bad_count, n) if n else None
    return {
        "schema": "stereoforge.v11.2.real_quality_screen.v1",
        "scope": "retained-single-frame ZED2 development artifact; not held-out or physical-ground-truth qualification",
        "not_preregistered": True,
        "performance": {
            "v11_coarse_stage_gate_as_reported": bool(data["v11_1_coarse_sgm_replacement_gate"]["gate_passed"]),
            "a7_gpu_measured_p95_ms": round(float(p95_meas), 4),
            "a7_gpu_planar_p95_ms": round(float(p95_planar), 4),
            "fps1000_gpu_p95_pass": float(p95_meas) <= 1.0 and float(p95_planar) <= 1.0,
        },
        "real_camera": {
            "a0_measured_valid_pixels": int(a0["measured_valid_pixels"]),
            "a7_measured_valid_pixels": int(a7["measured_valid_pixels"]),
            "measured_coverage_change_pp": round(measured_delta_pp, 3),
            "a0_completed_valid_pixels": int(a0["completed_valid_pixels"]),
            "a7_completed_valid_pixels": int(a7["completed_valid_pixels"]),
            "completed_coverage_change_pp": round(completed_delta_pp, 3),
            "a0_zncc_overlap_pixels": int(a0_zncc["zncc_overlap_pixels"]),
            "a7_zncc_overlap_pixels": int(a7_zncc["zncc_overlap_pixels"]),
            "zncc_bad2_increase_pp": round(zncc_delta_pp, 3),
            "newly_gained_pixels": int(new["cohort_pixels"]),
            "lost_pixels": int(lost["cohort_pixels"]),
            "newly_gained_zncc_overlap": int(new["zncc_overlap_pixels"]),
            "newly_gained_zncc_bad2_pct": new["vs_zncc_bad_2px_pct"],
            "a0_anchored_plane_caveat": "desk plane fit uses A0 pixels gated by ZNCC; not independent physical ground truth",
        },
        "synthetic_good2_scene_deltas_pp": scene_deltas_pp,
        "near_field": {
            "high_disparity_completed_pixels": int(near["near_field_completed_d_gt_127_5_pixels"]),
            "independent_overlap_pixels": n,
            "independent_overlap_as_pct_of_completed_high_disparity": (
                round(100*n/int(near["near_field_completed_d_gt_127_5_pixels"]), 3)
                if int(near["near_field_completed_d_gt_127_5_pixels"]) else None
            ),
            "reported_bad2_pct_on_overlap": bad_rate,
            "estimated_bad_count_from_rounded_pct": estimated_bad_count,
            "approx_wilson_bad2_95pct_interval": (
                [round(100*ci[0], 2), round(100*ci[1], 2)] if ci else None
            ),
            "scope_warning": "v11 harness imports v10 near-field evaluator; D64 test not run on A7 hierarchical",
        },
        "exploratory_screen_flags": flags,
        "quality_promotion": "HOLD" if any(flags.values()) else "FURTHER_VALIDATION_REQUIRED",
        "unmet_evidence": [
            "independent physical geometry or held-out multi-scene depth truth",
            "per-mode diverse frame evaluation and real interleaved ABBA timing",
            "A7-specific near-field D64 qualification with error tails",
            "distinct-frame sustained GPU throughput and end-to-end ROS latency",
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--output", type=Path, help="write a fresh report; do not replace source evidence")
    args = p.parse_args()
    result = assess(json.loads(args.input.read_text(encoding="utf-8")))
    s = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            p.error("output already exists; preserve the audit trail")
        args.output.write_text(s, encoding="utf-8")
    else:
        print(s, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
