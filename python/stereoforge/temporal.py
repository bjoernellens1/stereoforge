"""Bounded CPU temporal oracle; optical coordinates use +Z axial metres."""
from collections import deque
import numpy as np
from .types import CameraModel, DepthFrame, Provenance, validate_pose


def _copy(frame):
    return DepthFrame(*(getattr(frame,name).copy() for name in
        ('disparity','depth_m','confidence','valid','provenance','source_ids','age_ns')),
        frame.timestamp_ns,frame.calibration_id)


def reproject_depth(previous: DepthFrame, camera: CameraModel,
                    previous_to_current, timestamp_ns: int, *, fx=None, baseline_m=None,
                    principal_offset=0.) -> DepthFrame:
    """Nearest-pixel rigid warp with nearest-depth, stable source-order visibility."""
    metric_disparity=fx is not None or baseline_m is not None
    if not np.isfinite(principal_offset) or (metric_disparity and
            (fx is None or baseline_m is None or not np.isfinite([fx,baseline_m]).all()
             or fx<=0 or baseline_m<=0)):
        raise ValueError('metric disparity requires positive finite fx and baseline_m')
    pose=np.asarray(previous_to_current,dtype=np.float64)
    validate_pose(pose)
    if previous.depth_m.shape != (camera.height,camera.width):
        raise ValueError('depth dimensions do not match camera')
    if camera.distortion and np.any(np.asarray(camera.distortion)!=0):
        raise ValueError('reprojection requires rectified pinhole calibration')
    if timestamp_ns < previous.timestamp_ns:
        raise ValueError('timestamp moves backwards')
    out=_copy(previous)
    out.timestamp_ns=timestamp_ns
    out.depth_m[:]=np.nan; out.disparity[:]=np.nan; out.valid[:]=False
    out.confidence[:]=0; out.provenance[:]=Provenance.INVALID
    out.source_ids[:]=-1; out.age_ns[:]=0
    y,x=np.nonzero(previous.valid)
    z=previous.depth_m[y,x]
    points=np.stack(((x-camera.cx)*z/camera.fx,(y-camera.cy)*z/camera.fy,z),axis=1)
    points=points@pose[:3,:3].T+pose[:3,3]
    front=np.isfinite(points).all(axis=1)&(points[:,2]>0)
    points=points[front]; x=x[front]; y=y[front]
    u_float=camera.fx*points[:,0]/points[:,2]+camera.cx
    v_float=camera.fy*points[:,1]/points[:,2]+camera.cy
    inside=(u_float>=-.5)&(u_float<camera.width-.5)&(v_float>=-.5)&(v_float<camera.height-.5)
    points=points[inside]; x=x[inside]; y=y[inside]
    u=np.floor(u_float[inside]+.5).astype(np.int64)
    v=np.floor(v_float[inside]+.5).astype(np.int64)
    target=v*camera.width+u
    order=np.lexsort((np.arange(len(target)),points[:,2],target))
    target_sorted=target[order]
    chosen=order[np.r_[True,target_sorted[1:]!=target_sorted[:-1]]] if len(order) else order
    u=u[chosen]; v=v[chosen]; x=x[chosen]; y=y[chosen]
    out.depth_m[v,u]=points[chosen,2]; out.valid[v,u]=True
    out.confidence[v,u]=previous.confidence[y,x]
    out.provenance[v,u]=np.where(np.isin(previous.provenance[y,x],[Provenance.STEREO,Provenance.REPROJECTED]),
                                  Provenance.REPROJECTED,previous.provenance[y,x])
    if metric_disparity:
        out.disparity[v,u]=fx*baseline_m/out.depth_m[v,u]+principal_offset
    out.source_ids[v,u]=previous.source_ids[y,x]
    out.age_ns[v,u]=previous.age_ns[y,x]+timestamp_ns-previous.timestamp_ns
    return out


class TemporalHistory:
    """History storage and conservative fusion; confidence is never compounded."""
    def __init__(self,max_frames=4,max_age_ns=200000000,relative_depth_tolerance=.03):
        if max_frames<1 or max_age_ns<0 or relative_depth_tolerance<0:
            raise ValueError('invalid history bounds')
        self.frames=deque(maxlen=max_frames)
        self.max_age_ns=max_age_ns
        self.relative_depth_tolerance=relative_depth_tolerance

    def reset(self):
        self.frames.clear()

    def fuse(self,current: DepthFrame,prediction: DepthFrame|None) -> DepthFrame:
        discontinuity=bool(self.frames and (self.frames[-1].calibration_id!=current.calibration_id or
                           current.timestamp_ns<=self.frames[-1].timestamp_ns or
                           current.timestamp_ns-self.frames[-1].timestamp_ns>self.max_age_ns))
        if discontinuity:
            self.reset(); prediction=None
        if prediction is not None:
            if prediction.calibration_id!=current.calibration_id or prediction.timestamp_ns>current.timestamp_ns:
                self.reset(); prediction=None
            elif prediction.depth_m.shape!=current.depth_m.shape:
                raise ValueError('prediction dimensions differ')
        out=_copy(current)
        if prediction is not None:
            age=prediction.age_ns+current.timestamp_ns-prediction.timestamp_ns
            usable=prediction.valid & (age<=self.max_age_ns)
            fill=~current.valid & usable
            for name in ('depth_m','confidence','source_ids'):
                getattr(out,name)[fill]=getattr(prediction,name)[fill]
            out.valid[fill]=True; out.age_ns[fill]=age[fill]
            out.provenance[fill]=np.where(np.isin(prediction.provenance[fill],
                [Provenance.STEREO,Provenance.REPROJECTED]),Provenance.REPROJECTED,
                prediction.provenance[fill])
            out.disparity[fill]=np.nan
            both=current.valid & usable
            consistent=both & (np.abs(current.depth_m-prediction.depth_m)<=
                              self.relative_depth_tolerance*current.depth_m)
            simple_current=np.isin(current.provenance,[Provenance.STEREO,Provenance.REPROJECTED])
            simple_prediction=np.isin(prediction.provenance,[Provenance.STEREO,Provenance.REPROJECTED])
            combine=consistent & (current.source_ids!=prediction.source_ids) & simple_current & simple_prediction
            weight=current.confidence+prediction.confidence
            combine &= weight>0
            out.depth_m[combine]=(current.depth_m[combine]*current.confidence[combine]+
                prediction.depth_m[combine]*prediction.confidence[combine])/weight[combine]
            out.confidence[combine]=np.maximum(current.confidence[combine],prediction.confidence[combine])
            out.provenance[combine]=Provenance.FUSED; out.disparity[combine]=np.nan
        while self.frames and current.timestamp_ns-self.frames[0].timestamp_ns>self.max_age_ns:
            self.frames.popleft()
        self.frames.append(_copy(out))
        return out


def adaptive_bounds(prediction: DepthFrame|None,max_disparity: int,frame_index: int,
                    refresh_interval=10,radius=4,min_confidence=.7,max_age_ns=100000000,
                    shape=None):
    """Return exclusive disparity bounds and total candidate work, conservatively."""
    if max_disparity<1 or refresh_interval<1 or radius<0 or frame_index<0 or max_age_ns<0 or not 0<=min_confidence<=1:
        raise ValueError('invalid adaptive configuration')
    if prediction is not None:
        if shape is not None and tuple(shape)!=prediction.depth_m.shape:
            raise ValueError('shape differs from prediction')
        shape=prediction.depth_m.shape
    if shape is None or len(shape)!=2 or min(shape)<=0:
        raise ValueError('shape is required without prediction')
    lower=np.zeros(shape,np.int32); upper=np.full(shape,max_disparity,np.int32)
    if prediction is not None and frame_index%refresh_interval:
        d=prediction.disparity
        reliable=prediction.valid & np.isfinite(d) & (d>0) & (d<max_disparity) & \
            (prediction.confidence>=min_confidence) & (prediction.age_ns<=max_age_ns)
        lower[reliable]=np.clip(np.floor(d[reliable])-radius,0,max_disparity-1).astype(np.int32)
        upper[reliable]=np.clip(np.ceil(d[reliable])+radius+1,1,max_disparity).astype(np.int32)
    return lower,upper,int(np.sum(upper.astype(np.int64)-lower))
