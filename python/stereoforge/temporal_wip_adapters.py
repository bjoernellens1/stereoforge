"""StereoForge Temporal Supersampling (`v12`) — WIP Dummy Adapters.

STATUS: WORK IN PROGRESS (DUMMY ADAPTERS)

These adapters provide explicit placeholder interfaces for the two remaining
items in the `v12` Temporal Supersampling roadmap:
1. `WipDummyLivePoseAdapter`: Dummy visual odometry (VIO) / IMU rig pose source
   for live real-camera multi-frame sequences (`P1`–`P3`) before an independent
   VIO/IMU pose estimator is connected.
2. `WipDummyMultiPairStereoRingAdapter`: Dummy GPU ring-buffer adapter for
   retaining previous rectified stereo image pairs for cross-frame `P3` cost
   aggregation (currently delegates to single-current-pair `P3` Census re-search
   guided by the pose-warped historical prior).
3. `WipDummyGpuSuperRes2xAdapter`: Dummy GPU 2x depth super-resolution adapter
   for `P4` before porting the Python/CPU 2x subpixel surface accumulator into
   a native Vulkan/Daxa compute shader.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Optional, Tuple

import numpy as np


@dataclass
class WipDummyLivePoseAdapter:
    """WIP dummy VIO / IMU rig pose adapter for live multi-frame camera sequences.

    WARNING: Emits synthetic constant-velocity or identity `world_from_rig` poses
    aligned to the capture timestamp and clock domain. Do NOT treat outputs driven
    by this adapter as physically measured camera motion on moving live sequences.
    """

    step_translation_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    clock_domain: str = "steady"
    anchor_frame_id: Optional[int] = None

    IS_WIP_DUMMY: bool = True
    STATUS: str = "WIP_DUMMY_POSE_ADAPTER_NO_LIVE_VIO_IMU"

    def reset(self) -> None:
        self.anchor_frame_id = None

    def pose_for_frame(self, frame_id: int, timestamp_ns: int) -> Dict[str, Any]:
        if self.anchor_frame_id is None:
            self.anchor_frame_id = int(frame_id)
        steps = float(max(0, int(frame_id) - self.anchor_frame_id))
        tx = steps * float(self.step_translation_m[0])
        ty = steps * float(self.step_translation_m[1])
        tz = steps * float(self.step_translation_m[2])
        world_from_rig = np.array(
            [
                [1.0, 0.0, 0.0, tx],
                [0.0, 1.0, 0.0, ty],
                [0.0, 0.0, 1.0, tz],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return {
            "is_wip_dummy": True,
            "status": self.STATUS,
            "frame_id": int(frame_id),
            "timestamp_ns": int(timestamp_ns),
            "clock_domain": self.clock_domain,
            "world_from_rig": world_from_rig,
            "tx_m": tx,
            "ty_m": ty,
            "tz_m": tz,
        }


@dataclass
class WipDummyMultiPairStereoRingAdapter:
    """WIP dummy adapter for `P3` cross-frame historical rectified image-pair matching.

    Retains a bounded ring buffer of previous `(left_u8, right_u8, pose)` pairs
    while delegating execution to the current-frame `P3` temporal Census re-search
    until multi-pair GPU cost aggregation is implemented in `temporal.slang`.
    """

    capacity: int = 4
    _ring: Deque[Dict[str, Any]] = field(default_factory=deque, init=False)

    IS_WIP_DUMMY: bool = True
    STATUS: str = "WIP_DUMMY_CURRENT_PAIR_CENSUS_ONLY_NO_HISTORICAL_IMAGE_RING"

    def __post_init__(self) -> None:
        if self.capacity < 1 or self.capacity > 8:
            raise ValueError("WIP multi-pair ring capacity must be in 1..8")
        self._ring = deque(maxlen=self.capacity)

    def reset(self) -> None:
        self._ring.clear()

    @property
    def buffered_pair_count(self) -> int:
        return len(self._ring)

    def push_and_run_p3(
        self,
        frame_id: int,
        left_u8: np.ndarray,
        right_u8: np.ndarray,
        pose: Dict[str, Any],
        p3_runner: Callable[[], Dict[str, Any]],
    ) -> Dict[str, Any]:
        self._ring.append(
            {
                "frame_id": int(frame_id),
                "left_shape": tuple(left_u8.shape),
                "right_shape": tuple(right_u8.shape),
                "pose": pose,
            }
        )
        result = dict(p3_runner())
        result["wip_adapter_metadata"] = {
            "is_wip_dummy": True,
            "status": self.STATUS,
            "buffered_historical_pairs": len(self._ring),
            "ring_capacity": self.capacity,
        }
        return result


@dataclass
class WipDummyGpuSuperRes2xAdapter:
    """WIP dummy adapter for on-GPU `P4` 2x depth super-resolution grid.

    Wraps the CPU/Python `P4` 2x reconstruction prototype and tags the output
    with explicit `WIP_DUMMY_HOST_PROTOTYPE_ONLY_NO_GPU_2X_SHADER` provenance.
    """

    scale_factor: int = 2
    IS_WIP_DUMMY: bool = True
    STATUS: str = "WIP_DUMMY_HOST_PROTOTYPE_ONLY_NO_GPU_2X_SHADER"

    def reconstruct_2x(
        self,
        host_reconstructor: Callable[..., Tuple[np.ndarray, np.ndarray]],
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        depth_2x, valid_2x = host_reconstructor(*args, **kwargs)
        return {
            "is_wip_dummy": True,
            "status": self.STATUS,
            "scale_factor": self.scale_factor,
            "depth_2x_m": depth_2x,
            "valid_2x": valid_2x,
            "grid_shape": tuple(depth_2x.shape),
        }
