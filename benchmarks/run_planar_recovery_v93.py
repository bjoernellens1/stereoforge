"""Run the experimental v9.3 planar CPU reference on an exact-grid input NPZ.

This is an offline comparison. It never publishes inferred pixels as measured ROS depth.
Input arrays:
  measured_disp, measured_valid                    [H,W] (full pixels)
  coarse_left, coarse_right                        [H/4,W/4] (coarse pixels)
  coarse_left_valid, coarse_right_valid            [H/4,W/4]
  rectified_left_gray, rectified_right_gray        [H,W] (0..255)
Use an independently computed coarse RIGHT disparity, not a derived left-aggregated
right reference, when making reciprocal quality claims.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.planar_recovery_v93 import PlanarRecoveryConfig, recover_planar_reference


REQUIRED = (
    "measured_disp", "measured_valid", "coarse_left", "coarse_right",
    "coarse_left_valid", "coarse_right_valid",
    "rectified_left_gray", "rectified_right_gray",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Exact-grid .npz with required arrays")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--max-disparity", type=float, default=127.0)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as arrays:
        missing = [name for name in REQUIRED if name not in arrays.files]
        if missing:
            parser.error("missing arrays: " + ", ".join(missing))
        inputs = [arrays[name] for name in REQUIRED]
    result = recover_planar_reference(
        *inputs, config=PlanarRecoveryConfig(max_disparity_px=args.max_disparity)
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz = args.output_prefix.with_suffix(".npz")
    report_path = args.output_prefix.with_suffix(".json")
    np.savez_compressed(
        npz, disparity=result.depth_disparity, provenance=result.provenance,
        inferred_confidence=result.inferred_confidence,
        inferred_mask=result.candidate_mask,
        coarse_prediction=result.coarse_prediction,
    )
    report = {
        "schema": "stereoforge.planar_recovery_v93_cpu_reference.v0",
        "scope": "offline_reference_only; no Slang GPU path or ROS publication",
        "input_path": str(args.input),
        "output_path": str(npz),
        "configuration": {"max_disparity_px": args.max_disparity},
        "guard_warning": "Full-resolution descriptor-at-proposal guard not yet implemented",
        "correctness_warning": "Photometric agreement and plane fit are not independent depth ground truth",
        "counts": result.diagnostics,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
