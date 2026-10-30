"""Execute a bounded ABBA-style shared-GPU benchmark using existing native binaries.

The runner does not invent GPU-utilization measurements or declare stereo quality.
Commands are argv arrays; no shell is invoked. Reports are parsed only after exit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
from typing import Any

SCHEMA = "stereoforge.shared_gpu_benchmark.v1"
MAX_REPORT_BYTES = 16 * 1024 * 1024


def percentile(values: list[float], fraction: float) -> float:
    if not values or not 0 <= fraction <= 1:
        raise ValueError("invalid percentile input")
    return sorted(values)[max(0, math.ceil(fraction * len(values)) - 1)]


def _positive(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected numeric performance value")
    v = float(value)
    if not math.isfinite(v) or v <= 0 or v > 1e9:
        raise ValueError("invalid performance measurement")
    return v


def _lookup(data: Any, field: str) -> Any:
    result = data
    for segment in field.split("."):
        if not segment or not isinstance(result, dict) or segment not in result:
            raise ValueError("missing metric/identity field: " + field)
        result = result[segment]
    return result


def _load_report(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size > MAX_REPORT_BYTES:
        raise ValueError("missing or oversized benchmark report: " + str(path))
    report = json.loads(path.read_text())
    if not isinstance(report, dict):
        raise ValueError("benchmark report must be an object")
    return report


def _command(spec: Any) -> dict:
    if not isinstance(spec, dict) or set(spec) - {"argv", "metric_path", "identity_path"}:
        raise ValueError("invalid command fields")
    argv = spec.get("argv")
    metric_path = spec.get("metric_path")
    if (not isinstance(argv, list) or not 1 <= len(argv) <= 100 or
        not all(isinstance(x, str) and 0 < len(x) <= 4096 for x in argv) or
        "{report}" not in argv or not isinstance(metric_path, str) or
        not metric_path or len(metric_path) > 512):
        raise ValueError("command needs argv containing {report} and a metric_path")
    identity = spec.get("identity_path")
    if identity is not None and (not isinstance(identity, str) or not identity):
        raise ValueError("invalid report identity path")
    return spec


def validate_config(config: Any) -> dict:
    if not isinstance(config, dict) or set(config) != {
        "schema", "foreground", "background", "rounds",
        "timeout_seconds", "minimum_overlap_fraction", "minimum_samples"
    } or config["schema"] != SCHEMA:
        raise ValueError("unsupported shared GPU benchmark config")
    _command(config["foreground"])
    _command(config["background"])
    if (type(config["rounds"]) is not int or not 1 <= config["rounds"] <= 12 or
        type(config["timeout_seconds"]) is not int or not 1 <= config["timeout_seconds"] <= 3600 or
        type(config["minimum_samples"]) is not int or not 5 <= config["minimum_samples"] <= 100_000):
        raise ValueError("invalid bounded benchmark settings")
    fraction = config["minimum_overlap_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (float, int)) or not .5 <= fraction <= 1.:
        raise ValueError("minimum overlap must be in [0.5,1]")
    return config


def _launch(spec: dict, root: Path, directory: Path, name: str) -> dict:
    report = directory / (name + ".json")
    if report.exists():
        raise FileExistsError(report)
    argv = [str(report) if x == "{report}" else x for x in spec["argv"]]
    stdout = (directory / (name + ".stdout")).open("xb")
    stderr = (directory / (name + ".stderr")).open("xb")
    try:
        started = time.monotonic_ns()
        process = subprocess.Popen(argv, cwd=root, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
    except BaseException:
        stdout.close()
        stderr.close()
        raise
    return {"process": process, "started_ns": started, "report": report,
            "stdout": stdout, "stderr": stderr, "argv": argv, "spec": spec}


def _finish(job: dict, timeout_seconds: int) -> dict:
    process = job["process"]
    try:
        status = process.wait(timeout_seconds)
    except subprocess.TimeoutExpired as e:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise TimeoutError("benchmark command exceeded timeout") from e
    finally:
        job["stdout"].close()
        job["stderr"].close()
    finished = time.monotonic_ns()
    if status:
        raise RuntimeError(f"benchmark command exited {status}: {job['stderr'].name}")
    report = _load_report(job["report"])
    raw = _lookup(report, job["spec"]["metric_path"])
    identity_path = job["spec"].get("identity_path")
    identity = _lookup(report, identity_path) if identity_path else None
    if identity is not None and not isinstance(identity, (str, int, float, dict)):
        raise ValueError("unsupported identity value")
    result = {
        "started_ns": job["started_ns"],
        "finished_ns": finished,
        "report": job["report"].name,
        "report_sha256": hashlib.sha256(job["report"].read_bytes()).hexdigest(),
        "identity": identity,
    }
    if isinstance(raw, list):
        values = [_positive(x) for x in raw]
        result["samples_ms"] = values
        result["p50_ms"] = percentile(values, .5)
        result["p99_ms"] = percentile(values, .99)
    else:
        result["throughput_per_second"] = _positive(raw)
    return result


def _run_one(spec: dict, root: Path, directory: Path, name: str,
             timeout_seconds: int) -> dict:
    return _finish(_launch(spec, root, directory, name), timeout_seconds)


def _run_pair(config: dict, root: Path, directory: Path, cycle: int) -> dict:
    # Launch the competitor first so it has begun producing work before stereo.
    bg = _launch(config["background"], root, directory, f"pair{cycle}-background")
    try:
        fg = _launch(config["foreground"], root, directory, f"pair{cycle}-foreground")
    except BaseException:
        os.killpg(bg["process"].pid, signal.SIGKILL)
        bg["process"].wait()
        bg["stdout"].close()
        bg["stderr"].close()
        raise
    try:
        foreground = _finish(fg, config["timeout_seconds"])
        background = _finish(bg, config["timeout_seconds"])
    except BaseException:
        for job in (fg, bg):
            if job["process"].poll() is None:
                os.killpg(job["process"].pid, signal.SIGKILL)
                job["process"].wait()
            job["stdout"].close()
            job["stderr"].close()
        raise
    overlap = max(0, min(foreground["finished_ns"], background["finished_ns"]) -
                  max(foreground["started_ns"], background["started_ns"]))
    fg_duration = foreground["finished_ns"] - foreground["started_ns"]
    overlap_fraction = overlap / fg_duration
    return {
        "foreground": foreground,
        "background": background,
        "foreground_overlap_fraction": overlap_fraction,
        "overlap_qualified": overlap_fraction >= config["minimum_overlap_fraction"],
    }


def evaluate(config: dict, records: list[dict]) -> dict:
    fg_solo, bg_solo, pair_fg, pair_bg = [], [], [], []
    overlap = []
    identities = []
    for record in records:
        solo = record["solo"]
        pair = record["paired"]
        if "samples_ms" not in solo["foreground"] or "samples_ms" not in pair["foreground"]:
            raise ValueError("foreground metric must be an array of GPU-time samples")
        if "throughput_per_second" not in solo["background"] or "throughput_per_second" not in pair["background"]:
            raise ValueError("background metric must be a throughput scalar")
        for fg in (solo["foreground"], pair["foreground"]):
            if len(fg["samples_ms"]) < config["minimum_samples"]:
                raise ValueError("insufficient foreground samples")
            identities.append(fg["identity"])
        fg_solo.append(solo["foreground"]["p99_ms"])
        pair_fg.append(pair["foreground"]["p99_ms"])
        bg_solo.append(solo["background"]["throughput_per_second"])
        pair_bg.append(pair["background"]["throughput_per_second"])
        overlap.append(pair["foreground_overlap_fraction"])
    if not records:
        raise ValueError("no benchmark records")
    identity_verified = config["foreground"].get("identity_path") is not None and (
        all(item == identities[0] for item in identities))
    ratios = [during / alone for alone, during in zip(bg_solo, pair_bg)]
    return {
        "schema": SCHEMA,
        "rounds": len(records),
        "solo_foreground_worst_p99_ms": max(fg_solo),
        "paired_foreground_worst_p99_ms": max(pair_fg),
        "solo_background_median_throughput": statistics.median(bg_solo),
        "paired_background_median_throughput": statistics.median(pair_bg),
        "worst_background_throughput_retention": min(ratios),
        "minimum_foreground_process_overlap_fraction": min(overlap),
        "overlap_qualified": all(x["paired"]["overlap_qualified"] for x in records),
        "identity_consistent": identity_verified,
        "performance_targets": {
            "stereo_p99_le_15ms": max(pair_fg) <= 15.0,
            "background_retention_ge_80pct": min(ratios) >= .8,
        },
        # This is never a claim of stereo/SDK accuracy or exposure-to-consumer age.
        "physical_depth_quality_status": "unverified",
        "camera_clock_mapping_status": "unverified",
        "performance_qualified": all(x["paired"]["overlap_qualified"] for x in records)
                                 and identity_verified and max(pair_fg) <= 15.0
                                 and min(ratios) >= .8,
        "end_to_end_or_quality_qualified": False,
    }


def run(config: dict, root: Path, output: Path) -> dict:
    validate_config(config)
    output.mkdir(parents=True, exist_ok=False)
    (output / "input-config.json").write_text(json.dumps(config, indent=2) + "\n")
    records = []
    for index in range(config["rounds"]):
        directory = output / f"cycle-{index:02d}"
        directory.mkdir()
        # Balance order between solo and paired runs to expose thermal drift.
        if index % 2:
            paired = _run_pair(config, root, directory, index)
        else:
            paired = None
        foreground = _run_one(config["foreground"], root, directory,
                              f"solo{index}-foreground", config["timeout_seconds"])
        background = _run_one(config["background"], root, directory,
                              f"solo{index}-background", config["timeout_seconds"])
        if paired is None:
            paired = _run_pair(config, root, directory, index)
        records.append({"solo": {"foreground": foreground, "background": background},
                        "paired": paired})
    summary = evaluate(config, records)
    report = {"summary": summary, "cycles": records,
              "input_config_sha256": hashlib.sha256(
                  (output / "input-config.json").read_bytes()).hexdigest()}
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Must be a new directory; never overwritten")
    options = parser.parse_args(argv)
    config = validate_config(json.loads(options.config.read_text()))
    result = run(config, options.config.resolve().parent, options.output)
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
