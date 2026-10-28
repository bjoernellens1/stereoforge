import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    binary = tmp_path_factory.mktemp("thread-clock") / "bench"
    subprocess.run(["c++", "-std=c++20", "-O2", str(ROOT / "apps/thread_cpu_clock_microbench.cpp"), "-o", str(binary)], check=True)
    return binary


def test_real_cpu_benchmark(bench):
    result = subprocess.run([str(bench), "--iterations", "30"], text=True, capture_output=True, check=True)
    data = json.loads(result.stdout)
    assert data["bytes"] == 1280 * 720 * 4
    assert data["clock_failures"] == 0
    for name in ("copy_wall_off", "copy_wall_on", "publish_wall_off", "publish_wall_on", "total_wall_off", "total_wall_on", "copy_cpu_on", "publish_cpu_on", "copy_wall_paired_delta", "publish_wall_paired_delta", "total_wall_paired_delta", "clock_call_wall"):
        stats = data[name]
        assert stats["count"] == 30
        assert stats["p50_ns"] <= stats["p95_ns"] <= stats["p99_ns"] <= stats["max_ns"]
    assert "no ROS" in data["limitations"]


def test_clock_failures_reported_without_fake_cpu_samples(bench):
    result = subprocess.run([str(bench), "--iterations", "2", "--inject-clock-failure"], text=True, capture_output=True)
    assert result.returncode == 2
    data = json.loads(result.stdout)
    assert data["clock_failures"] == 10
    assert data["copy_cpu_on"]["count"] == 0
    assert data["copy_cpu_on"]["p50_ns"] is None


@pytest.mark.parametrize("arguments", [["--iterations", "0"], ["--iterations", "2x"], ["--unknown"]])
def test_invalid_arguments_fail(bench, arguments):
    result = subprocess.run([str(bench), *arguments], text=True, capture_output=True)
    assert result.returncode != 0
    assert not result.stdout
