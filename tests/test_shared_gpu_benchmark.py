"""CPU-only integration of the real shared GPU subprocess benchmark protocol."""
import json
import math
from pathlib import Path
import sys

import pytest

from benchmarks.shared_gpu_benchmark import (
    SCHEMA, _lookup, _positive, evaluate, percentile, run, validate_config
)


def make_config(p: Path):
    script = p / "benchmark_fixture.py"
    script.write_text(
        "import json,sys,time\n"
        "from pathlib import Path\n"
        "out,kind = sys.argv[1:3]\n"
        "time.sleep(0.20)\n"
        "if kind=='foreground':\n"
        "  result={'metrics':{'gpu_ms':[8.0]*24},'identity':{'profile':'same-input'}}\n"
        "else:\n"
        "  result={'metrics':{'throughput':100.0}}\n"
        "Path(out).write_text(json.dumps(result))\n"
    )
    return {
        "schema": SCHEMA,
        "foreground": {
            "argv": [sys.executable, str(script), "{report}", "foreground"],
            "metric_path": "metrics.gpu_ms",
            "identity_path": "identity",
        },
        "background": {
            "argv": [sys.executable, str(script), "{report}", "background"],
            "metric_path": "metrics.throughput",
        },
        "rounds": 1,
        "timeout_seconds": 10,
        "minimum_overlap_fraction": 0.5,
        "minimum_samples": 20,
    }


def test_run_actual_subprocesses_without_shell(tmp_path):
    config = make_config(tmp_path)
    result = run(config, tmp_path, tmp_path / "experiment")
    summary = result["summary"]
    assert summary["rounds"] == 1
    assert summary["overlap_qualified"]
    assert summary["identity_consistent"]
    assert summary["solo_foreground_worst_p99_ms"] == 8.0
    assert summary["paired_foreground_worst_p99_ms"] == 8.0
    assert summary["worst_background_throughput_retention"] == 1.0
    assert summary["performance_qualified"]
    assert not summary["end_to_end_or_quality_qualified"]
    assert (tmp_path / "experiment" / "results.json").exists()
    assert all("report_sha256" in report for report in (
        result["cycles"][0]["solo"]["foreground"],
        result["cycles"][0]["paired"]["background"],
    ))
    with pytest.raises(FileExistsError):
        run(config, tmp_path, tmp_path / "experiment")


def test_reject_unsafe_and_incomplete_config(tmp_path):
    conf = make_config(tmp_path)
    validate_config(conf)
    for mutation in (
        {"foreground": {"argv": ["echo", "oops"], "metric_path": "x"}},
        {"rounds": 0},
        {"timeout_seconds": 0},
        {"minimum_overlap_fraction": 0.2},
        {"minimum_samples": -1},
    ):
        altered = {**conf, **mutation}
        with pytest.raises(ValueError):
            validate_config(altered)


def test_stats_reject_invalid_values():
    assert percentile([1, 2, 10, 4], .99) == 10
    assert percentile([1, 2, 10, 4], .5) == 2
    for x in (float("nan"), float("inf"), 0, -2, True, "3"):
        with pytest.raises(ValueError):
            _positive(x)
    with pytest.raises(ValueError):
        _lookup({"a": 1}, "a.b")


def test_no_performance_qualification_without_overlap_or_identity(tmp_path):
    conf = make_config(tmp_path)
    solo_fg = {"samples_ms": [6.0]*20, "p99_ms": 6.0, "identity": None}
    pair_fg = {"samples_ms": [12.0]*20, "p99_ms": 12.0, "identity": None}
    rec = {"solo": {"foreground": solo_fg, "background": {"throughput_per_second": 100.0}},
           "paired": {"foreground": pair_fg, "background": {"throughput_per_second": 90.0},
                      "foreground_overlap_fraction": .1, "overlap_qualified": False}}
    result = evaluate(conf, [rec])
    assert not result["identity_consistent"]
    assert not result["overlap_qualified"]
    assert not result["performance_qualified"]
    assert result["performance_targets"]["background_retention_ge_80pct"]
    assert result["physical_depth_quality_status"] == "unverified"
    assert not result["end_to_end_or_quality_qualified"]


def test_early_background_exit_does_not_fake_concurrency(tmp_path):
    config = make_config(tmp_path)
    # Reuse the exact subprocess fixture while reducing background lifetime.
    script = tmp_path / "benchmark_fixture.py"
    source = script.read_text()
    source = source.replace("time.sleep(0.20)",
                            "time.sleep(0.02 if kind=='background' else 0.35)")
    script.write_text(source)
    config["minimum_overlap_fraction"] = .8
    result = run(config, tmp_path, tmp_path / "short-background")
    pair = result["cycles"][0]["paired"]
    assert pair["foreground_overlap_fraction"] < .5
    assert not pair["overlap_qualified"]
    assert not result["summary"]["performance_qualified"]
