"""Milestone v10.0 / v10.2 / v10.3 / v10.4 tests:
- Hardware roofline & 3-metric tuple (`T_frame`, `E_frame`, `Delta_T_competing_workload`)
- `--profile ultra-fast` vs `--profile quality` GPU execution & stage speedups
- Near-field `D_coarse = 64` (`256 px` full-res disparity) recovery on `y >= 463`
- `DutyCycleBudgetGovernor` C++ contract in `include/stereoforge/depth_rate_controller.hpp`
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
V10_0_JSON = ROOT / "benchmarks" / "results" / "v10_0_roofline_and_energy_baseline.json"
V10_QUAL_JSON = ROOT / "benchmarks" / "results" / "v10_ultra_fast_and_universal_qualification.json"


def test_duty_cycle_budget_governor_cpp_contract(tmp_path: Path) -> None:
    """Compiles and runs a C++ contract test for `DutyCycleBudgetGovernor` in `depth_rate_controller.hpp`."""
    cxx = shutil.which("g++") or shutil.which("c++")
    if not cxx:
        pytest.skip("C++ compiler unavailable")

    probe_cpp = tmp_path / "test_governor.cpp"
    probe_bin = tmp_path / "test_governor"
    probe_cpp.write_text(
        r"""
#include <stereoforge/depth_rate_controller.hpp>
#include <cassert>
#include <cmath>
#include <stdexcept>

using namespace stereoforge;

int main() {
    // 1. Validation contracts
    bool caught = false;
    try {
        DutyCycleBudgetGovernor bad({0.0, 60, 5, 60, 0.2});
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught);

    // 2. 5% GPU duty-cycle budget with 2.4 ms ultra-fast frame time -> 20 FPS (4.8% <= 5.0%)
    DutyCycleBudgetGovernor gov5({0.05, 60, 5, 60, 0.5});
    for (int i = 0; i < 10; ++i) {
        gov5.observe_gpu_ms(2.4);
    }
    assert(gov5.recommended_fps() == 20);
    assert(gov5.estimated_duty_fraction() <= 0.05 + 1e-9);
    assert(gov5.budget_throttles() == 10);

    // 3. 10% GPU duty-cycle budget with 2.4 ms ultra-fast frame time -> 30 FPS (7.2% <= 10.0%)
    DutyCycleBudgetGovernor gov10({0.10, 60, 5, 60, 0.5});
    for (int i = 0; i < 10; ++i) {
        gov10.observe_gpu_ms(2.4);
    }
    assert(gov10.recommended_fps() == 30);
    assert(gov10.estimated_duty_fraction() <= 0.10 + 1e-9);

    // 4. 20% GPU duty-cycle budget with 2.4 ms ultra-fast frame time -> 60 FPS (14.4% <= 20.0%)
    DutyCycleBudgetGovernor gov20({0.20, 60, 5, 60, 0.5});
    for (int i = 0; i < 10; ++i) {
        gov20.observe_gpu_ms(2.4);
    }
    assert(gov20.recommended_fps() == 60);
    assert(gov20.budget_throttles() == 0);

    return 0;
}
""",
        encoding="utf-8",
    )
    subprocess.run(
        [cxx, "-std=c++20", "-O2", "-I", str(ROOT / "include"), str(probe_cpp), "-o", str(probe_bin)],
        check=True,
    )
    subprocess.run([str(probe_bin)], check=True)


def test_v10_0_roofline_and_energy_baseline_artifact() -> None:
    """Verifies the v10.0 roofline and joint 3-metric baseline JSON."""
    assert V10_0_JSON.exists()
    data = json.loads(V10_0_JSON.read_text(encoding="utf-8"))
    roof = data["roofline_and_memory_traffic_model"]
    assert roof["total_pipeline_traffic_mb"] > 500.0
    assert roof["total_bandwidth_roofline_floor_ms"] > 1.5
    elim = roof["v10_2_architectural_traffic_reduction_plan"]
    assert elim["total_eliminable_global_memory_traffic_mb"] > 100.0

    prof = data["joint_3_metric_tuple"]
    assert prof["T_frame_ms"]["gpu_combined_p50_ms"] < 4.167
    assert prof["E_frame_mj"]["gross_energy_mj_per_frame"] > 50.0
    assert prof["Delta_T_competing_workload"]["delta_competing_gpu_p50_pct"] > 0.0


def test_v10_ultra_fast_and_universal_qualification_artifact() -> None:
    """Verifies the complete Milestone v10 qualification JSON across v10.1, v10.2, v10.3, and v10.4."""
    assert V10_QUAL_JSON.exists()
    data = json.loads(V10_QUAL_JSON.read_text(encoding="utf-8"))
    assert data["schema"] == "stereoforge.v10.ultra_fast_and_universal_qualification.v1"

    # v10.1 Portable Slang CPU backend
    cpu = data["v10_1_portable_slang_cpu_backend"]
    assert cpu["zero_cuda_rocm_vulkan_dependency"] is True
    assert {"rectify", "census", "match", "lr", "metric", "planar"}.issubset(set(cpu["entrypoints_compiled"]))
    assert cpu["bit_exact_parity_with_vulkan"]["census_and_validity_bit_exact_pct"] == 100.0
    assert cpu["bit_exact_parity_with_vulkan"]["lr_and_metric_validity_bit_exact_pct"] == 100.0

    # v10.2 Ultra-fast GPU profiles
    gpu = data["v10_2_ultra_fast_gpu_profiles"]
    q_p50 = gpu["quality_with_planar"]["gpu_combined_p50_ms"]
    u_p50 = gpu["ultra_fast_with_planar"]["gpu_combined_p50_ms"]
    um_p50 = gpu["ultra_fast_measured_only"]["gpu_combined_p50_ms"]
    assert u_p50 < q_p50
    assert um_p50 < u_p50
    assert gpu["speedup_ultra_fast_planar_vs_quality_planar"] > 1.18
    assert gpu["speedup_ultra_fast_measured_vs_quality_planar"] > 1.40
    assert gpu["metric_stage_reduction_ms"] > 0.20

    # v10.2 Near-field D_coarse=64 vs D_coarse=32 on y >= 463 (d > 127.5 px)
    nf = data["v10_2_near_field_d_coarse_64_evaluation"]
    assert nf["d_coarse_32"]["near_field_completed_d_gt_127_5_pixels"] == 0
    assert nf["d_coarse_64"]["near_field_completed_d_gt_127_5_pixels"] > 20000
    assert nf["d_coarse_64"]["near_field_max_disparity_px"] > 150.0
    assert nf["d_coarse_64"]["near_field_min_depth_m"] < 0.45
