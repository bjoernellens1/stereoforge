"""Read-only StereoForge 1,000-FPS evidence audit.

This evaluates *qualification evidence*, never measures a GPU or converts a
single-frame kernel time into a demonstrated sustained delivery rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "benchmarks/results/v10_ultra_fast_and_universal_qualification.json"
GPU_P95_BUDGET_MS = 1.0
HOST_P95_BUDGET_MS = 1.0
MIN_SUSTAINED_FRAMES = 1000
MIN_DISTINCT_INPUTS = 128
MIN_INDEPENDENT_NEAR_FIELD_OVERLAP = 100

# Engineering allocations, NOT measured lower bounds. Includes 0.03 ms
# unallocated for synchronization, queue gaps and runtime overhead.
STAGE_BUDGET_MS = {
    "dual_preprocess_total_ms": 0.18,
    "coarse_total_ms": 0.25,
    "refinement_census_ms": 0.10,
    "refinement_guided_match_ms": 0.23,
    "refinement_lr_ms": 0.03,
    "refinement_metric_ms": 0.05,
    "refinement_planar_recovery_ms": 0.13,
}


def assess_profile(profile: dict[str, Any], *, width: int = 1280, height: int = 720) -> dict[str, Any]:
    """Audit a v10 profile without claiming fresh-frame throughput or accuracy."""
    gpu_p50 = float(profile["gpu_combined_p50_ms"])
    gpu_p95 = float(profile["gpu_combined_p95_ms"])
    host_p95 = float(profile["host_wall_p95_ms"])
    if not (gpu_p50 > 0 and gpu_p95 > 0 and host_p95 > 0):
        raise ValueError("non-positive timing")
    stages = profile["stage_timings_ms"]
    stage_headroom = {
        name: {
            "observed_ms": round(float(stages[name]), 4),
            "target_ms": budget,
            "needed_speedup": round(float(stages[name]) / budget, 2),
        }
        for name, budget in STAGE_BUDGET_MS.items()
        if name in stages
    }
    # This benchmark repeats the same retained CaptureLease; its stage readings
    # are a last-sample snapshot, not a distribution of per-stage p95 values.
    return {
        "gpu_p50_ms": gpu_p50,
        "gpu_p95_ms": gpu_p95,
        "host_p95_ms": host_p95,
        "fps_from_gpu_p50_only": round(1000.0 / gpu_p50, 1),
        "gpu_p95_budget_pass": gpu_p95 <= GPU_P95_BUDGET_MS,
        "host_p95_budget_pass": host_p95 <= HOST_P95_BUDGET_MS,
        "measured_coverage_pct": round(100.0 * int(profile["measured_valid_pixels"]) / (width * height), 2),
        "completed_coverage_pct": round(100.0 * int(profile["completed_valid_pixels"]) / (width * height), 2),
        "timed_frames": int(profile["frames"]),
        "warmup_frames": int(profile["warmup"]),
        "stage_snapshot_vs_allocations": stage_headroom,
        "stage_timing_limitation": "last measured frame, not per-stage p95",
        "input_limitation": "one repeated retained camera record; distinct frames not demonstrated",
        "end_to_end_limitation": "excludes live acquisition, view and transport",
        "timing_verdict": "FAIL" if gpu_p95 > GPU_P95_BUDGET_MS or host_p95 > HOST_P95_BUDGET_MS else "UNQUALIFIED",
    }


def assess_near_field(cohort: dict[str, Any]) -> dict[str, Any]:
    """An accepted pixel is not evidence of *correct* near-field geometry."""
    overlap = int(cohort["vs_zncc_d_gt_127_5_overlap_pixels"])
    bad = cohort.get("vs_zncc_d_gt_127_5_bad_2px_pct")
    return {
        "reported_high_disparity_completed_pixels": int(cohort["near_field_completed_d_gt_127_5_pixels"]),
        "independent_reference_overlap_pixels": overlap,
        "independent_bad_2px_pct_on_overlap": bad,
        "independently_qualified": overlap >= MIN_INDEPENDENT_NEAR_FIELD_OVERLAP and bad is not None and float(bad) <= 5.0,
        "verdict": "UNQUALIFIED" if overlap < MIN_INDEPENDENT_NEAR_FIELD_OVERLAP else ("PASS_PRELIMINARY" if bad is not None and float(bad) <= 5.0 else "FAIL"),
        "limitation": "small overlap cannot establish accuracy or inaccuracy across the accepted cohort",
    }


def assess_v10(data: dict[str, Any]) -> dict[str, Any]:
    profiles = data["v10_2_ultra_fast_gpu_profiles"]
    profile_names = [
        name
        for name in (
            "quality_with_planar",
            "ultra_fast_with_planar",
            "ultra_fast_measured_only",
            "v11_fast_with_planar",
            "v11_fast_measured_only",
        )
        if name in profiles
    ]
    audits = {name: assess_profile(profiles[name]) for name in profile_names}
    nearfield = assess_near_field(data["v10_2_near_field_d_coarse_64_evaluation"]["d_coarse_64"])
    competition = data.get("v10_4_duty_cycle_governor", {}).get("competing_workload_coexistence", {})
    key_issue = not bool(competition)
    return {
        "schema": "stereoforge.v11.fps1000_evidence_audit.v1",
        "goal": {
            "resolution_per_eye": [1280, 720],
            "gpu_p95_budget_ms": GPU_P95_BUDGET_MS,
            "host_p95_budget_ms": HOST_P95_BUDGET_MS,
            "min_sustained_frames": MIN_SUSTAINED_FRAMES,
            "min_distinct_inputs": MIN_DISTINCT_INPUTS,
            "required": "independent fixed-support quality gates and p99 under varied input and contention",
        },
        "v10_profiles": audits,
        "near_field_d64": nearfield,
        "competing_workload_provenance": {
            "field_present": not key_issue,
            "diagnosis_if_missing": "v10 script reads joint_3metric_profiler, baseline contains joint_3_metric_tuple" if key_issue else None,
        },
        "speedup_comparison_caveat": "quality profile is benchmarked with layered-per-path coarse schedule, while ultra-fast uses shared-census-sequential and derived coarse right reference; not a matched algorithmic A/B",
        "global_verdict": "NO_GO",
        "why_no_go": [
            "all recorded 720p profiles exceed the 1.0 ms GPU p95 budget",
            "the v10 fixture repeats one source frame and excludes capture/view/transport",
            "independent near-field and fixed-truth quality acceptance is not established for ultra-fast",
            "p99 and sustained >=1000 distinct-timestamped-frame throughput are not measured",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, help="Optional fresh report JSON output")
    args = parser.parse_args()
    report = assess_v10(json.loads(args.input.read_text(encoding="utf-8")))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            parser.error("output exists; use a fresh filename to preserve provenance")
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
