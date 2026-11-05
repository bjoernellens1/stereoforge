# v14 physical depth corpus and audits

Correctness is measured against physical references, not against the estimator itself. Corpus data lives outside git
(`~/stereoforge_corpus/<scene>/`, ~0.8 GB per 150-frame scene with RealSense).

## Record scenes

```bash
.venv/bin/python benchmarks/physical_depth_v14/capture_corpus.py --scene S1_target_1.0m --tape-m 1.0 --realsense
```

| Scene | What to show | Why |
|---|---|---|
| `S1_target_<d>m` (d = 0.5, 1.0, 1.5, 2.0, 3.0) | ChArUco/AprilTag board on a flat stand, tape-measured camera-to-board distance | absolute bias/precision vs range |
| `S2_white_wall` | empty white wall | textureless behaviour, boxes, prior locking |
| `S3_person_white_wall` | head and shoulders in front of a white wall | low-contrast silhouette / halo |
| `S4_textured_desk` | desk with objects, static | plane RMSE, temporal stability |
| `S5_person_moving` | slow arm/head motion | ghosting / wake |
| `S6_dim_room` | lights dimmed | exposure/noise behaviour |
| `S7_calib_board_poses` | board at ~12 varied poses/rotations visible to both sensors | ZED-RealSense extrinsics, rectification |

**Mount the ZED 2 and the RealSense rigidly side by side, pointing the same way, so both see the board and the scene** (check both views before recording). Targets must be fully visible to the ZED left camera. RealSense runs at 640x480 (USB 2.1 link). The tool starts the ZED only once the RealSense streams and prints the overlap; it should equal the ZED capture length.

## Audits (no ground truth needed)

```bash
PYTHONPATH=python:. .venv/bin/python benchmarks/physical_depth_v14/audit_rectification.py FRAMES.npy      # vertical disparity
PYTHONPATH=python:. .venv/bin/python benchmarks/physical_depth_v14/refine_calibration.py FRAMES.npy --out overlay.json
PYTHONPATH=python:benchmarks:. .venv/bin/python benchmarks/physical_depth_v14/baseline_cpu.py FRAMES.npy # coverage, pixel locking, temporal std
```

`FRAMES.npy` is `zed_raw.npy` (or a subsample) of a scene. Results of the first audit are in
`benchmarks/results/v14_phase0_baseline.json`.
