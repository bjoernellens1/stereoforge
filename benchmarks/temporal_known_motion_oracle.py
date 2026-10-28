"""Independent Task7b CPU specification, not a production implementation.

Float64 scalar geometry is exact only for individually justified rational cases.
No GPU float-byte equivalence, dynamic-scene visibility, matcher-work reduction,
physical accuracy or held-out quality claim follows. No production helpers or
legacy temporal reference are imported. All planes are two rectified eye grids;
fresh disparity/LR/raw planes remain outside this module and are never rewritten.
"""
from dataclasses import dataclass, replace
import math
import numpy as np

SCHEMA='stereoforge.temporal_known_motion.cpu.v1'
I64_MAX=(1<<63)-1
CAPACITY=4
PLANE_NAMES=('depth','confidence','valid','provenance','source_ids','age_ns','ancestry_ids','count','oldest_age_ns','reuse','origin_mask')


def integer(value,name,maximum=I64_MAX):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not 0<=int(value)<=maximum:raise ValueError('invalid '+name)
    return int(value)


def rigid(matrix):
    m=np.asarray(matrix,dtype=np.float64)
    if m.shape!=(4,4) or not np.isfinite(m).all() or not np.array_equal(m[3],[0.,0.,0.,1.]) or not np.allclose(m[:3,:3].T@m[:3,:3],np.eye(3),rtol=0,atol=1e-10) or not math.isclose(float(np.linalg.det(m[:3,:3])),1.,rel_tol=0,abs_tol=1e-10):raise ValueError('finite proper rigid pose required')
    return m


@dataclass(frozen=True)
class Config:
    max_contributors:int=4
    max_reuse:int=4
    max_age_ns:int=200000000
    relative_z_tolerance:float=.03
    min_confidence:float=.05

    def __post_init__(self):
        if integer(self.max_contributors,'contributor cap',CAPACITY)<1:raise ValueError('empty contributor cap')
        integer(self.max_reuse,'reuse cap',(1<<32)-1);integer(self.max_age_ns,'age cap')
        for name in ('relative_z_tolerance','min_confidence'):
            v=getattr(self,name)
            if isinstance(v,(bool,str)) or not math.isfinite(v) or v<0:raise ValueError('invalid '+name)
        if self.min_confidence>1:raise ValueError('confidence threshold exceeds one')


@dataclass
class TimedPose:
    world_from_rig:np.ndarray
    timestamp_ns:int
    clock_domain:str


@dataclass
class Geometry:
    intrinsics:np.ndarray  # eye, fx/fy/cx/cy
    eye_to_rig:np.ndarray  # A_L=R1^T; A_R=right_to_left*R2^T
    support:np.ndarray     # eye,height,width; validated calibration support

    def __post_init__(self):
        self.intrinsics=np.asarray(self.intrinsics,dtype=np.float64).copy();self.eye_to_rig=np.asarray(self.eye_to_rig,dtype=np.float64).copy();self.support=np.asarray(self.support).copy()
        if self.intrinsics.shape!=(2,4) or not np.isfinite(self.intrinsics).all() or np.any(self.intrinsics[:,:2]<=0):raise ValueError('two finite positive eye pinholes required')
        if self.eye_to_rig.shape!=(2,4,4):raise ValueError('two eye-to-rig transforms required')
        for m in self.eye_to_rig:rigid(m)
        if self.support.ndim!=3 or self.support.shape[0]!=2 or min(self.support.shape[1:])<1 or self.support.dtype!=np.bool_:raise ValueError('binary two-eye support required')


@dataclass
class Frame:
    depth:np.ndarray
    confidence:np.ndarray
    valid:np.ndarray
    provenance:np.ndarray
    source_ids:np.ndarray
    age_ns:np.ndarray
    ancestry_ids:np.ndarray
    count:np.ndarray
    oldest_age_ns:np.ndarray
    reuse:np.ndarray
    origin_mask:np.ndarray
    timestamp_ns:int
    capture_id:int
    calibration_id:str
    device_id:str
    clock_domain:str

    def copy(self):return replace(self,**{n:getattr(self,n).copy() for n in PLANE_NAMES})

    def to_json(self):
        # JSON has no NaN: validity retains the canonical invalid distinction.
        planes={n:np.where(self.valid,self.depth,0.).tolist() if n=='depth' else getattr(self,n).tolist() for n in PLANE_NAMES}
        return dict(schema=SCHEMA,**planes,**{n:getattr(self,n) for n in ('timestamp_ns','capture_id','calibration_id','device_id','clock_domain')})


def empty_like(frame):
    shape=frame.depth.shape
    return replace(frame,depth=np.full(shape,np.nan,np.float64),confidence=np.zeros(shape,np.float64),valid=np.zeros(shape,bool),provenance=np.zeros(shape,np.uint32),source_ids=np.full(shape,-1,np.int64),age_ns=np.zeros(shape,np.int64),ancestry_ids=np.full(shape+(CAPACITY,),-1,np.int64),count=np.zeros(shape,np.uint32),oldest_age_ns=np.zeros(shape,np.int64),reuse=np.zeros(shape,np.uint32),origin_mask=np.zeros(shape,np.uint32))


def fresh_frame(depth,*,confidence=.5,provenance=1,source_ids=None,age_ns=0,timestamp_ns=0,capture_id=0,calibration_id='cal',device_id='device',clock_domain='host'):
    z=np.asarray(depth,dtype=np.float64)
    if z.ndim!=3 or z.shape[0]!=2:raise ValueError('two-eye depth shape required')
    integer(capture_id,'capture ID');integer(timestamp_ns,'timestamp')
    def plane(value,dtype):
        raw=np.asarray(value)
        if dtype in (np.int64,np.uint32) and raw.dtype.kind not in 'iu':raise ValueError('integer metadata required')
        if dtype==np.int64 and raw.dtype.kind=='u' and np.any(raw>I64_MAX):raise ValueError('int64 metadata overflow')
        return np.broadcast_to(np.asarray(value,dtype=dtype),z.shape).copy()
    valid=np.isfinite(z)&(z>0)
    f=Frame(z.copy(),plane(confidence,np.float64),valid,plane(provenance,np.uint32),plane(capture_id if source_ids is None else source_ids,np.int64),plane(age_ns,np.int64),np.full(z.shape+(4,),-1,np.int64),valid.astype(np.uint32),plane(age_ns,np.int64),np.zeros(z.shape,np.uint32),np.zeros(z.shape,np.uint32),timestamp_ns,capture_id,calibration_id,device_id,clock_domain)
    if np.any(valid&((f.provenance<1)|(f.provenance>5))):raise ValueError('unknown origin provenance')
    f.ancestry_ids[...,0][valid]=f.source_ids[valid]
    f.origin_mask[valid]=np.left_shift(np.uint32(1),f.provenance[valid])
    for index in np.ndindex(z.shape):
        if not valid[index]:clear_pixel(f,index)
    validate_frame(f);return f


def clear_pixel(frame,index):
    frame.depth[index]=np.nan;frame.confidence[index]=0;frame.valid[index]=False;frame.provenance[index]=0;frame.source_ids[index]=-1;frame.age_ns[index]=0;frame.ancestry_ids[index]=-1;frame.count[index]=0;frame.oldest_age_ns[index]=0;frame.reuse[index]=0;frame.origin_mask[index]=0


def validate_frame(frame):
    integer(frame.timestamp_ns,'timestamp');integer(frame.capture_id,'capture ID')
    for name in ('calibration_id','device_id','clock_domain'):
        value=getattr(frame,name)
        if not isinstance(value,str) or not value or value.lower() in ('unknown','unspecified'):raise ValueError('known '+name+' required')
    shape=frame.depth.shape
    if len(shape)!=3 or shape[0]!=2 or min(shape[1:])<1:raise ValueError('two-eye shape required')
    expected=dict(depth=np.float64,confidence=np.float64,valid=np.bool_,provenance=np.uint32,source_ids=np.int64,age_ns=np.int64,ancestry_ids=np.int64,count=np.uint32,oldest_age_ns=np.int64,reuse=np.uint32,origin_mask=np.uint32)
    for name,dtype in expected.items():
        value=getattr(frame,name)
        if value.shape!=shape+((4,) if name=='ancestry_ids' else ()) or value.dtype!=dtype:raise ValueError('plane shape/type mismatch '+name)
    for index in np.ndindex(shape):
        if not frame.valid[index]:
            if not np.isnan(frame.depth[index]) or frame.confidence[index]!=0 or frame.provenance[index]!=0 or frame.source_ids[index]!=-1 or frame.age_ns[index]!=0 or frame.count[index]!=0 or frame.oldest_age_ns[index]!=0 or frame.reuse[index]!=0 or frame.origin_mask[index]!=0 or np.any(frame.ancestry_ids[index]!=-1):raise ValueError('noncanonical invalid metadata')
            continue
        count=int(frame.count[index]);ids=frame.ancestry_ids[index];used=ids[:count].tolist()
        if not math.isfinite(frame.depth[index]) or frame.depth[index]<=0 or not math.isfinite(frame.confidence[index]) or not 0<=frame.confidence[index]<=1:raise ValueError('invalid metric value')
        if not 1<=count<=4 or used!=sorted(set(used)) or min(used)<0 or np.any(ids[count:]!=-1) or int(frame.source_ids[index]) not in used:raise ValueError('incomplete ancestry or source ID')
        if not 1<=frame.provenance[index]<=5 or not 0<int(frame.origin_mask[index])<=62 or int(frame.origin_mask[index])&1 or (frame.provenance[index]==3 and count<2) or (count>1 and frame.provenance[index]!=3) or frame.age_ns[index]<0 or frame.oldest_age_ns[index]<frame.age_ns[index]:raise ValueError('invalid age/origin metadata')


def validate_fresh_current(frame,config):
    """Stream intake is raw measured depth, never this stage's accumulated output.

    General validate_frame remains usable for history and standalone fusion.
    All checks precede state changes; no inferred/truncated ancestry is accepted.
    """
    validate_frame(frame)
    for index in np.ndindex(frame.depth.shape):
        if not frame.valid[index]:continue
        p=int(frame.provenance[index]);source=int(frame.source_ids[index])
        if p not in (1,5) or frame.count[index]!=1 or frame.count[index]>config.max_contributors or frame.reuse[index]!=0 or frame.oldest_age_ns[index]!=frame.age_ns[index] or frame.ancestry_ids[index].tolist()!=[source,-1,-1,-1] or frame.origin_mask[index]!=(1<<p):raise ValueError('fresh measured current requires one original ancestor, zero reuse and exact origin')


def bounded_weighted_depth(z,p,c,pc):
    """Bounded convex interpolation avoids multiplying metric extremes.

    Any unrepresentable result rejects the complete caller transaction before
    commit. It is never silently clamped or committed with valid nonpositive Z.
    """
    low=min(z,p);high=max(z,p);high_confidence=c if z>=p else pc
    result=low+(high-low)*(high_confidence/(c+pc))
    if not math.isfinite(result) or result<=0 or not low<=result<=high:raise ValueError('unrepresentable fused metric depth')
    return result


def eye_motion(geometry,world_previous,world_current):
    previous=rigid(world_previous);current=rigid(world_current)
    return np.array([np.linalg.solve(A,np.linalg.solve(current,previous@A)) for A in geometry.eye_to_rig])


def copy_pixel(output,target,source,index):
    for name in PLANE_NAMES:getattr(output,name)[target]=getattr(source,name)[index]


def work_counts():return dict(source_pixels_scanned=0,valid_prior_pixels=0,eligible_history_candidates=0,projected_history_candidates=0,visible_history_pixels=0,history_fill_pixels=0,fused_pixels=0)


def warp(previous,geometry,motions,timestamp_ns,config):
    """Global nearest-Z, then lowest original source index; no per-tile visibility."""
    validate_frame(previous)
    if previous.depth.shape!=geometry.support.shape:raise ValueError('warp geometry shape mismatch')
    motions=np.asarray(motions,dtype=np.float64)
    if motions.shape!=(2,4,4):raise ValueError('two rigid eye motions required')
    for matrix in motions:rigid(matrix)
    integer(timestamp_ns,'target timestamp');elapsed=timestamp_ns-previous.timestamp_ns
    if elapsed<0:raise ValueError('backward warp timestamp')
    out=empty_like(previous);out.timestamp_ns=timestamp_ns;work=work_counts();winners={};_,height,width=previous.depth.shape
    for index in np.ndindex(previous.depth.shape):
        work['source_pixels_scanned']+=1
        if not previous.valid[index]:continue
        work['valid_prior_pixels']+=1
        eye,y,x=index;age=int(previous.age_ns[index])+elapsed;oldest=int(previous.oldest_age_ns[index])+elapsed;reuse=int(previous.reuse[index])+1
        if not geometry.support[index] or age>I64_MAX or oldest>I64_MAX or oldest>config.max_age_ns or reuse>config.max_reuse or previous.confidence[index]<config.min_confidence or previous.count[index]>config.max_contributors:continue
        work['eligible_history_candidates']+=1
        fx,fy,cx,cy=geometry.intrinsics[eye];z=float(previous.depth[index]);point=motions[eye]@np.array([(x-cx)*z/fx,(y-cy)*z/fy,z,1.])
        if not np.isfinite(point).all() or point[2]<=0:continue
        u=fx*point[0]/point[2]+cx;v=fy*point[1]/point[2]+cy
        if not(-.5<=u<width-.5 and -.5<=v<height-.5):continue
        target=(eye,math.floor(v+.5),math.floor(u+.5))
        if not geometry.support[target]:continue
        work['projected_history_candidates']+=1;candidate=(float(point[2]),y*width+x,index,age,oldest,reuse)
        if target not in winners or candidate[:2]<winners[target][:2]:winners[target]=candidate
    for target,(z,_,index,age,oldest,reuse) in winners.items():
        copy_pixel(out,target,previous,index);out.depth[target]=z;out.age_ns[target]=age;out.oldest_age_ns[target]=oldest;out.reuse[target]=reuse
        if out.provenance[target] in (1,2):out.provenance[target]=2
    work['visible_history_pixels']=len(winners)
    return out,work


def fuse(current,prior,support,config):
    validate_frame(current)
    if support.shape!=current.depth.shape or support.dtype!=np.bool_:raise ValueError('fusion support shape/type differs')
    if prior is not None:
        validate_frame(prior)
        if prior.depth.shape!=current.depth.shape or prior.timestamp_ns!=current.timestamp_ns:raise ValueError('prediction must already use current geometry/time')
    out=current.copy();work=work_counts()
    for index in np.ndindex(current.depth.shape):
        if not support[index]:clear_pixel(out,index);continue
        if prior is None or not prior.valid[index] or prior.oldest_age_ns[index]>config.max_age_ns or prior.reuse[index]>config.max_reuse or prior.count[index]>config.max_contributors or prior.confidence[index]<config.min_confidence:continue
        if not current.valid[index]:
            copy_pixel(out,index,prior,index)
            if out.provenance[index] in (1,2):out.provenance[index]=2
            work['history_fill_pixels']+=1;continue
        z=float(current.depth[index]);p=float(prior.depth[index]);c=float(current.confidence[index]);pc=float(prior.confidence[index])
        if c<config.min_confidence or abs(z-p)>config.relative_z_tolerance*z:continue
        left=set(current.ancestry_ids[index][:int(current.count[index])].tolist());right=set(prior.ancestry_ids[index][:int(prior.count[index])].tolist());union=sorted(left|right)
        if left&right or len(union)>config.max_contributors or c+pc<=0:continue
        out.depth[index]=bounded_weighted_depth(z,p,c,pc);out.confidence[index]=max(c,pc);out.provenance[index]=3
        out.ancestry_ids[index]=-1;out.ancestry_ids[index][:len(union)]=union;out.count[index]=len(union);out.oldest_age_ns[index]=max(int(current.oldest_age_ns[index]),int(prior.oldest_age_ns[index]));out.reuse[index]=max(int(current.reuse[index]),int(prior.reuse[index]));out.origin_mask[index]=int(current.origin_mask[index])|int(prior.origin_mask[index]);work['fused_pixels']+=1
    return out,work


class TemporalOracle:
    """Transactional CPU state machine; configuration replacement is explicit."""
    def __init__(self,geometry,config=Config(),calibration_id=None):
        self.geometry=Geometry(geometry.intrinsics,geometry.eye_to_rig,geometry.support);self.config=config;self.calibration_id=calibration_id;self.history=None;self.pose=None

    def reconfigure(self,geometry,calibration_id=None):
        self.geometry=Geometry(geometry.intrinsics,geometry.eye_to_rig,geometry.support);self.calibration_id=calibration_id;self.history=None;self.pose=None

    def process(self,current,pose,reset=False):
        validate_fresh_current(current,self.config)
        if current.depth.shape!=self.geometry.support.shape:raise ValueError('shape requires explicit reconfigure')
        if self.calibration_id is not None and current.calibration_id!=self.calibration_id:raise ValueError('calibration requires explicit reconfigure')
        if pose is not None:
            rigid(pose.world_from_rig);integer(pose.timestamp_ns,'pose timestamp')
            if pose.timestamp_ns!=current.timestamp_ns or pose.clock_domain!=current.clock_domain:raise ValueError('pose timestamp/clock alignment differs')
        previous=self.history;reason=None;work=work_counts()
        epoch=previous is not None and (previous.clock_domain!=current.clock_domain or previous.device_id!=current.device_id)
        if previous is not None and previous.calibration_id!=current.calibration_id:raise ValueError('calibration requires explicit reconfigure')
        if previous is not None and not epoch and (current.timestamp_ns<=previous.timestamp_ns or current.capture_id<=previous.capture_id):raise ValueError('late/duplicate frame rejected')
        if reset:reason='explicit_reset'
        elif epoch:reason='epoch_change'
        elif previous is not None and current.timestamp_ns-previous.timestamp_ns>self.config.max_age_ns:reason='gap'
        if pose is None:reason='missing_motion'
        prediction=None
        if previous is not None and self.pose is not None and pose is not None and reason is None:
            prediction,work=warp(previous,self.geometry,eye_motion(self.geometry,self.pose.world_from_rig,pose.world_from_rig),current.timestamp_ns,self.config)
        out,fusion=fuse(current,prediction,self.geometry.support,self.config);work.update({key:fusion[key] for key in ('history_fill_pixels','fused_pixels')});work['reset_reason']=reason;work['scope']='CPU temporal pixels only; no SGM work reduction or GPU cost claim'
        # Commit after all checks and math. Never share mutable arrays with callers.
        validate_frame(out)
        self.history=out.copy();self.pose=None if pose is None else TimedPose(np.array(pose.world_from_rig,copy=True),pose.timestamp_ns,pose.clock_domain)
        if self.calibration_id is None:self.calibration_id=current.calibration_id
        return out,work


def world_point_sequence(geometry,points,poses,*,calibration_id='cal',device_id='device'):
    """Independent finite world-point witnesses, not surfaces or a physical renderer.

    Absolute world points are projected afresh into each eye/pose. Neither warp,
    fusion, motion composition nor estimator output supplies geometry truth.
    Per-point IDs label visibility truth; observation source IDs remain capture
    IDs and therefore are shared by all measured pixels in a capture.
    """
    frames=[];shape=geometry.support.shape;_,height,width=shape
    for capture_id,pose in enumerate(poses,1):
        rigid(pose.world_from_rig);z=np.full(shape,np.nan,np.float64);labels=np.full(shape,-1,np.int64);winner={}
        for eye in (0,1):
            fx,fy,cx,cy=geometry.intrinsics[eye]
            world_from_eye=pose.world_from_rig@geometry.eye_to_rig[eye]
            for point_id,world in points:
                integer(point_id,'world point ID');q=np.linalg.solve(world_from_eye,np.r_[np.asarray(world,dtype=np.float64),1.])
                if not np.isfinite(q).all() or q[2]<=0:continue
                u=fx*q[0]/q[2]+cx;v=fy*q[1]/q[2]+cy
                if not(-.5<=u<width-.5 and -.5<=v<height-.5):continue
                index=(eye,math.floor(v+.5),math.floor(u+.5))
                if not geometry.support[index]:continue
                candidate=(float(q[2]),point_id)
                if index not in winner or candidate<winner[index]:winner[index]=candidate
        for index,(depth,point_id) in winner.items():z[index]=depth;labels[index]=point_id
        frame=fresh_frame(z,timestamp_ns=pose.timestamp_ns,capture_id=capture_id,calibration_id=calibration_id,device_id=device_id,clock_domain=pose.clock_domain)
        frames.append(dict(frame=frame,pose=TimedPose(np.array(pose.world_from_rig,copy=True),pose.timestamp_ns,pose.clock_domain),world_point_ids=labels))
    return frames
