"""Pure-CPU safeguards for 1,000-FPS qualification evidence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.fps1000_gate_v11 import (
    MIN_INDEPENDENT_NEAR_FIELD_OVERLAP,
    STAGE_BUDGET_MS,
    assess_near_field,
    assess_profile,
    assess_v10,
)

ROOT = Path(__file__).resolve().parents[1]
V10 = ROOT / "benchmarks/results/v10_ultra_fast_and_universal_qualification.json"


def _profile() -> dict:
    return {
        "gpu_combined_p50_ms": 2.5544,
        "gpu_combined_p95_ms": 2.5708,
        "host_wall_p95_ms": 2.8239,
        "measured_valid_pixels": 294942,
        "completed_valid_pixels": 294942,
        "frames": 180,
        "warmup": 20,
        "stage_timings_ms": {name: budget * 3 for name, budget in STAGE_BUDGET_MS.items()},
    }


def test_timing_is_not_sustained_throughput() -> None:
    result = assess_profile(_profile())
    assert result["timing_verdict"] == "FAIL"
    assert result["fps_from_gpu_p50_only"] == 391.5
    assert result["measured_coverage_pct"] == 32.0
    assert result["stage_snapshot_vs_allocations"]["coarse_total_ms"]["needed_speedup"] == 3.0
    assert "repeated" in result["input_limitation"]


def test_hypothetical_under_one_ms_is_still_unqualified() -> None:
    profile = _profile()
    profile.update(gpu_combined_p50_ms=0.7, gpu_combined_p95_ms=0.9, host_wall_p95_ms=0.95)
    assert assess_profile(profile)["timing_verdict"] == "UNQUALIFIED"


def test_nearfield_count_does_not_qualify_accuracy() -> None:
    obj = {
        "near_field_completed_d_gt_127_5_pixels": 25064,
        "vs_zncc_d_gt_127_5_overlap_pixels": 1,
        "vs_zncc_d_gt_127_5_bad_2px_pct": 100,
    }
    result = assess_near_field(obj)
    assert result["reported_high_disparity_completed_pixels"] > 20000
    assert result["verdict"] == "UNQUALIFIED"
    assert result["independently_qualified"] is False
    obj["vs_zncc_d_gt_127_5_overlap_pixels"] = MIN_INDEPENDENT_NEAR_FIELD_OVERLAP
    obj["vs_zncc_d_gt_127_5_bad_2px_pct"] = 0.5
    assert assess_near_field(obj)["verdict"] == "PASS_PRELIMINARY"


def test_invalid_timing_rejected() -> None:
    profile = _profile()
    profile["gpu_combined_p50_ms"] = 0.0
    with pytest.raises(ValueError):
        assess_profile(profile)


@pytest.mark.skipif(not V10.exists(), reason="v10 report not present in checkout")
def test_real_v10_is_no_go_and_reports_missing_competition_link() -> None:
    report = assess_v10(json.loads(V10.read_text(encoding="utf-8")))
    assert report["global_verdict"] == "NO_GO"
    assert report["near_field_d64"]["verdict"] == "UNQUALIFIED"
    assert isinstance(report["competing_workload_provenance"]["field_present"], bool)
    assert "not a matched algorithmic A/B" in report["speedup_comparison_caveat"]
