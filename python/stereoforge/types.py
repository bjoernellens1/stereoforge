"""Host contracts. Optical coordinates: +X right, +Y down, +Z forward; metres."""
from dataclasses import dataclass, field
from enum import IntEnum
import hashlib
import json
import numpy as np

class Provenance(IntEnum):
    INVALID = 0
    STEREO = 1
    REPROJECTED = 2
    FUSED = 3
    NEURAL = 4

@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...] = ()

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0 or not np.isfinite([self.fx,self.fy,self.cx,self.cy,*self.distortion]).all() or min(self.fx,self.fy) <= 0:
            raise ValueError("invalid camera intrinsics")

    @property
    def K(self):
        return np.array([[self.fx,0,self.cx],[0,self.fy,self.cy],[0,0,1]], np.float64)

    def to_dict(self):
        return dict(width=self.width,height=self.height,fx=self.fx,fy=self.fy,cx=self.cx,cy=self.cy,distortion=list(self.distortion))

@dataclass(frozen=True)
class Calibration:
    left: CameraModel
    right: CameraModel
    right_to_left: np.ndarray
    serial: str = "synthetic"

    def __post_init__(self):
        t = np.array(self.right_to_left, dtype=np.float64, copy=True)
        validate_pose(t)
        if np.linalg.norm(t[:3,3]) <= 0:
            raise ValueError("stereo baseline must be positive")
        t.setflags(write=False)
        object.__setattr__(self,"right_to_left",t)

    @property
    def baseline_m(self):
        return float(np.linalg.norm(self.right_to_left[:3,3]))

    def to_dict(self):
        return dict(left=self.left.to_dict(),right=self.right.to_dict(),right_to_left=self.right_to_left.tolist(),serial=self.serial)

    @classmethod
    def from_dict(cls,d):
        return cls(CameraModel(**d["left"]),CameraModel(**d["right"]),np.asarray(d["right_to_left"]),d["serial"])

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True,separators=(",",":")).encode()).hexdigest()

def validate_pose(t):
    if t.shape != (4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1]) or not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(t[:3,:3]),1.,atol=1e-6):
        raise ValueError("pose must be a finite rigid 4x4 transform")

@dataclass
class StereoFrame:
    left: np.ndarray
    right: np.ndarray
    calibration: Calibration
    timestamp_ns: int
    frame_id: int
    clock_domain: str = "host_monotonic"
    sensor_samples: list[dict] = field(default_factory=list)

    def __post_init__(self):
        for image,c in [(self.left,self.calibration.left),(self.right,self.calibration.right)]:
            if image.shape != (c.height,c.width,3) or image.dtype != np.uint8:
                raise ValueError("RGB image shape/dtype does not match calibration")
        if self.timestamp_ns < 0 or self.frame_id < 0 or not self.clock_domain:
            raise ValueError("invalid frame identity/timestamp")

@dataclass
class DepthFrame:
    disparity: np.ndarray
    depth_m: np.ndarray
    confidence: np.ndarray
    valid: np.ndarray
    provenance: np.ndarray
    source_ids: np.ndarray
    age_ns: np.ndarray
    timestamp_ns: int
    calibration_id: str

    def __post_init__(self):
        if self.timestamp_ns<0 or not isinstance(self.calibration_id,str) or not self.calibration_id:
            raise ValueError("depth requires calibration identity and nonnegative timestamp")
        shape=self.depth_m.shape
        if len(shape)!=2 or any(a.shape!=shape for a in [self.disparity,self.confidence,self.valid,self.provenance,self.source_ids,self.age_ns]):
            raise ValueError("depth plane shape mismatch")
        if self.valid.dtype!=np.bool_ or np.any(self.valid & (~np.isfinite(self.depth_m) | (self.depth_m<=0))):
            raise ValueError("valid depth must be finite and positive")
        if not np.issubdtype(self.source_ids.dtype,np.integer) or not np.issubdtype(self.age_ns.dtype,np.integer) or np.any(self.valid & ((self.source_ids<0)|(self.provenance<1)|(self.provenance>4))):
            raise ValueError("valid depth requires observation identity and known provenance")
        if not np.isfinite(self.confidence).all() or np.any((self.confidence<0)|(self.confidence>1)) or np.any(self.age_ns<0):
            raise ValueError("invalid confidence or age")

@dataclass(frozen=True)
class TimedPose:
    """Rig-to-world rigid pose bound to the source clock and timestamp."""
    transform: np.ndarray
    timestamp_ns: int
    clock_domain: str

    def __post_init__(self):
        t=np.array(self.transform,dtype=np.float64,copy=True)
        validate_pose(t)
        if self.timestamp_ns<0 or not self.clock_domain:raise ValueError('invalid pose timestamp/clock')
        t.setflags(write=False)
        object.__setattr__(self,'transform',t)
