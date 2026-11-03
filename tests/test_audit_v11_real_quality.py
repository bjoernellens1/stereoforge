"""CPU-only tests for post-hoc v11 real-scene qualification diagnostics."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from benchmarks.audit_v11_real_quality import assess, _wilson_interval

V11 = Path(__file__).resolve().parents[1] / "benchmarks/results/v11_0_qualification_and_coarse_replacement.json"


def test_wilson_small_sample_spans_preliminary_threshold() -> None:
    low, high = _wilson_interval(5, 143)
    assert 0.01 < low < 0.03
    assert 0.07 < high < 0.09
    with pytest.raises(ValueError):
        _wilson_interval(2, 0)
    with pytest.raises(ValueError):
        _wilson_interval(-1, 10)


@pytest.mark.skipif(not V11.exists(), reason="v11 report not present in checkout")
def test_v11_coarse_speed_pass_does_not_promote_real_quality() -> None:
    report = assess(json.loads(V11.read_text(encoding="utf-8")))
    assert report["scope"].startswith("retained-single-frame")
    assert report["performance"]["v11_coarse_stage_gate_as_reported"] is True
    assert report["performance"]["fps1000_gpu_p95_pass"] is False
    assert report["quality_promotion"] == "HOLD"
    assert report["real_camera"]["measured_coverage_change_pp"] < -2.0
    assert report["real_camera"]["completed_coverage_change_pp"] < -3.0
    assert report["real_camera"]["zncc_bad2_increase_pp"] > 1.5
    assert report["real_camera"]["newly_gained_zncc_bad2_pct"] > 20.0
    assert report["near_field"]["independent_overlap_as_pct_of_completed_high_disparity"] < 1.0
    assert report["near_field"]["approx_wilson_bad2_95pct_interval"][1] > 5.0
    assert "v10" in report["near_field"]["scope_warning"]
    assert report["not_preregistered"] is True


@pytest.mark.skipif(not V11.exists(), reason="v11 report not present in checkout")
def test_screen_reports_provisional_only_even_when_heuristics_clear() -> None:
    data = deepcopy(json.loads(V11.read_text(encoding="utf-8")))
    z = data["v11_0_ablation_zed2_cohorts_and_gpu_distributions"]["configs"]
    baseline = z["A0_quality_layered_sgm_r4"]
    candidate = z["A7_v11_fast_hierarchical_1to8_fallback"]
    candidate["measured_valid_pixels"] = baseline["measured_valid_pixels"]
    candidate["completed_valid_pixels"] = baseline["completed_valid_pixels"]
    candidate["measured_overall_vs_zncc"]["vs_zncc_bad_2px_pct"] = baseline["measured_overall_vs_zncc"]["vs_zncc_bad_2px_pct"]
    candidate["measured_cohorts_vs_A0"]["newly_gained"]["vs_zncc_bad_2px_pct"] = 3.0
    s = assess(data)
    assert not any(s["exploratory_screen_flags"].values())
    assert s["quality_promotion"] == "FURTHER_VALIDATION_REQUIRED"
