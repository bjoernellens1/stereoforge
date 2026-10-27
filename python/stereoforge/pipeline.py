"""Research orchestration: explicit backends, measured readbacks, no hidden motion."""
from dataclasses import dataclass, asdict
from time import perf_counter
import cv2
import numpy as np
from .types import DepthFrame, Provenance, TimedPose
from .calibration import Rectifier
from .reference import census, matching_cost, aggregate_sgm, select_disparity, depth_from_disparity
from .temporal import TemporalHistory, reproject_depth, adaptive_bounds, _copy as snapshot_depth
from .fusion import synthesize, default_virtual_camera, stabilize_virtual_camera

@dataclass
class PipelineConfig:
    backend: str = 'vulkan'
    max_disparity: int = 128
    paths: int = 4
    p1: int = 8
    p2: int = 32
    subpixel: bool = True
    lr_tolerance: float = 1.
    temporal: bool = False
    adaptive: bool = False
    refresh_interval: int = 10
    search_radius: int = 4
    output: str = 'rectified_rgb'
    preview: bool = False
    fusion_backend: str = 'cpu_reference'

    def __post_init__(self):
        if self.backend not in ('vulkan','cpu_sgbm','cpu_census') or self.paths not in (4,8) or not 2<=self.max_disparity<=256 or not 0<=self.p1<=self.p2<=10000:
            raise ValueError('invalid stereo configuration')
        if self.output not in ('left_rgb','right_rgb','rectified_rgb','wide_rgb','wide_rgbd','stabilized_wide_rgb'):
            raise ValueError('invalid output mode')
        if self.fusion_backend not in ('cpu_reference','vulkan') or self.refresh_interval<1 or self.search_radius<0 or self.lr_tolerance<0:
            raise ValueError('invalid processing configuration')
        if self.adaptive and self.backend=='cpu_sgbm': raise ValueError('OpenCV SGBM does not support adaptive bounds')
        if self.preview and self.output=='wide_rgbd': raise ValueError('wide_rgbd requires strict geometry')
        if self.preview and self.fusion_backend=='vulkan': raise ValueError('GPU preview unavailable; select cpu_reference explicitly')

@dataclass
class PipelineResult:
    depth: DepthFrame
    rgb: np.ndarray
    view: object | None
    metrics: dict

class Pipeline:
    def __init__(self,config=None,virtual_camera=None):
        self.config=config or PipelineConfig()
        self.virtual_camera=virtual_camera; self.rectifier=None
        self.history=TemporalHistory();self.gpu=None;self.view_renderer=None;self.temporal_gpu=None;self._clock=None
        self._sgbm=None
        if self.config.backend=='vulkan':
            from .gpu import VulkanStereo
            self.gpu=VulkanStereo(self.config.max_disparity,self.config.paths,self.config.p1,self.config.p2,self.config.subpixel)
        if self.gpu and (self.config.temporal or self.config.adaptive):
            from .gpu_temporal import VulkanTemporal
            self.temporal_gpu=VulkanTemporal(self.gpu.device)
        if self.config.fusion_backend=='vulkan':
            from .gpu_geometry import VulkanViewSynthesizer
            self.view_renderer=VulkanViewSynthesizer()

    def _estimate(self,left,right,bounds=None):
        cfg=self.config
        if self.gpu:
            args=() if bounds is None else bounds[:2]
            out=self.gpu.result(self.gpu.enqueue(left,right,*args))
            return out['disparity'],out['confidence'],out['valid']
        if cfg.backend=='cpu_sgbm':
            D=((cfg.max_disparity+15)//16)*16
            if left.shape[1]<=D+5: raise ValueError('image too narrow for configured OpenCV disparity range')
            if self._sgbm is None:self._sgbm=cv2.StereoSGBM_create(numDisparities=D,blockSize=5,P1=8*25,P2=32*25,uniquenessRatio=10,speckleWindowSize=50,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
            disp=self._sgbm.compute(left.astype(np.uint8),right.astype(np.uint8)).astype(np.float32)/16
            valid=(disp>0)&(disp<cfg.max_disparity);disp[~valid]=np.nan
            # A validity score, deliberately not calibrated uncertainty.
            return disp,valid.astype(np.float32)*.5,valid
        a,av=census(left);b,bv=census(right)
        cost=matching_cost(a,b,av,bv,cfg.max_disparity,*(bounds[:2] if bounds else ()))
        return select_disparity(aggregate_sgm(cost,cfg.paths,cfg.p1,cfg.p2),cfg.subpixel)

    def process(self,frame,previous_to_current=None,rig_to_world=None):
        start=perf_counter();cfg=self.config
        if cfg.output=='stabilized_wide_rgb':
            if not isinstance(rig_to_world,TimedPose):raise ValueError('stabilized RGB requires a timestamped TimedPose')
            if rig_to_world.timestamp_ns!=frame.timestamp_ns or rig_to_world.clock_domain!=frame.clock_domain:
                raise ValueError('pose timestamp/clock does not match RGB frame')
        if self.rectifier is None or self.rectifier.calibration.identity!=frame.calibration.identity:
            self.rectifier=Rectifier(frame.calibration);self.history.reset()
        if self._clock!=frame.clock_domain:
            self.history.reset();self._clock=frame.clock_domain
        rect=self.rectifier.rectify(frame);cal=rect.calibration
        left=cv2.cvtColor(rect.left,cv2.COLOR_RGB2GRAY).astype(np.uint32)
        right=cv2.cvtColor(rect.right,cv2.COLOR_RGB2GRAY).astype(np.uint32)
        rect_ms=(perf_counter()-start)*1000
        prediction=None
        motion_start=perf_counter()
        if (cfg.temporal or cfg.adaptive) and previous_to_current is not None and self.history.frames:
            previous=self.history.frames[-1]
            if frame.timestamp_ns>previous.timestamp_ns and frame.timestamp_ns-previous.timestamp_ns<=self.history.max_age_ns:
                # Caller motion is in raw left-camera coordinates; transform to rectified frame.
                R=np.eye(4);R[:3,:3]=self.rectifier.R1
                motion=R@np.asarray(previous_to_current)@np.linalg.inv(R)
                warp=self.temporal_gpu.reproject if self.temporal_gpu else reproject_depth
                prediction=warp(previous,cal.left,motion,frame.timestamp_ns,fx=cal.left.fx,baseline_m=cal.baseline_m,principal_offset=cal.left.cx-cal.right.cx)
        motion_ms=(perf_counter()-motion_start)*1000
        bounds=adaptive_bounds(prediction,cfg.max_disparity,frame.frame_id,refresh_interval=cfg.refresh_interval,radius=cfg.search_radius,shape=left.shape) if cfg.adaptive else None
        t=perf_counter();disp,conf,valid=self._estimate(left,right,bounds)
        # Right-reference estimate via mirrored stereo; positive disparity convention retained.
        rd,_,rv=self._estimate(np.fliplr(right).copy(),np.fliplr(left).copy())
        rd=np.fliplr(rd);rv=np.fliplr(rv)
        yy,xx=np.indices(left.shape);safe=np.where(np.isfinite(disp),disp,0)
        xr=np.floor(xx-safe+.5).astype(np.int64);inside=(xr>=0)&(xr<left.shape[1]);xrc=np.clip(xr,0,left.shape[1]-1)
        consistent=inside&rv[yy,xrc]&(np.abs(disp-rd[yy,xrc])<=cfg.lr_tolerance)
        valid &= consistent;disp[~valid]=np.nan;conf[~valid]=0
        z,metric_valid=depth_from_disparity(disp,cal.left.fx,cal.baseline_m,cal.left.cx-cal.right.cx)
        valid &= metric_valid
        depth=DepthFrame(disp,z,conf,valid,np.where(valid,Provenance.STEREO,0).astype(np.uint8),np.where(valid,frame.frame_id,-1).astype(np.int64),np.zeros(left.shape,np.int64),frame.timestamp_ns,cal.identity)
        stereo_ms=(perf_counter()-t)*1000
        temporal_start=perf_counter()
        if cfg.temporal:
            if self.temporal_gpu:
                if self.history.frames and (frame.timestamp_ns<=self.history.frames[-1].timestamp_ns or frame.timestamp_ns-self.history.frames[-1].timestamp_ns>self.history.max_age_ns):
                    self.history.reset();prediction=None
                depth=self.temporal_gpu.fuse(depth,prediction,self.history.max_age_ns,self.history.relative_depth_tolerance)
                while self.history.frames and frame.timestamp_ns-self.history.frames[0].timestamp_ns>self.history.max_age_ns:self.history.frames.popleft()
                self.history.frames.append(snapshot_depth(depth))
            else:depth=self.history.fuse(depth,prediction)
        else:
            self.history.reset();self.history.frames.append(snapshot_depth(depth))
        temporal_ms=(perf_counter()-temporal_start)*1000
        view=None;rgb={'left_rgb':frame.left,'right_rgb':frame.right}.get(cfg.output,rect.left)
        t=perf_counter()
        if cfg.output.startswith(('wide_','stabilized_')):
            virtual=self.virtual_camera or default_virtual_camera(frame.calibration)
            if cfg.output=='stabilized_wide_rgb':
                if rig_to_world is None: raise ValueError('stabilized RGB requires a timestamped rig pose')
                virtual=stabilize_virtual_camera(virtual,rig_to_world.transform,rig_to_world.timestamp_ns)
            raw_depth=self.rectifier.depth_to_raw(depth)
            if self.view_renderer:
                view=self.view_renderer.render(frame,raw_depth,virtual)
            else:
                view=synthesize(frame,raw_depth,virtual,mode='preview' if cfg.preview else 'strict')
            rgb=view.rgb
        metrics=dict(rectify_ms=rect_ms,motion_ms=motion_ms,temporal_ms=temporal_ms,stereo_including_lr_ms=stereo_ms,view_ms=(perf_counter()-t)*1000,total_ms=(perf_counter()-start)*1000,
            valid_fraction=float(valid.mean()),candidate_evaluations=bounds[2] if bounds else left.size*cfg.max_disparity,
            right_candidate_evaluations=left.size*cfg.max_disparity,aggregation_candidates=2*left.size*cfg.max_disparity*cfg.paths,
            motion_available=prediction is not None,backend=cfg.backend,fusion_backend=cfg.fusion_backend,config=asdict(cfg))
        if self.gpu: metrics['gpu_last_pass']=dict(self.gpu.last_timings)
        return PipelineResult(depth,rgb,view,metrics)
