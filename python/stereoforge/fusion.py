"""Deterministic CPU reference for calibrated, visibility-aware RGB view synthesis.

Input depth must be aligned to each *raw RGB image*, with axial camera-Z in metres.
Rectified stereo depth must be mapped back to raw geometry before calling this API.
The source arrays are never modified. Preview infinity projection is explicitly marked
and never supplies metric geometry. This module is an oracle, not a real-time backend.
"""
from dataclasses import dataclass, field, replace
import cv2
import numpy as np
from .types import CameraModel, Calibration, StereoFrame, DepthFrame, validate_pose


@dataclass(frozen=True)
class VirtualCamera:
    width: int
    height: int
    hfov_deg: float
    projection: str = 'perspective'
    camera_to_rig: np.ndarray = field(default_factory=lambda: np.eye(4))
    pose_timestamp_ns: int | None = None

    def __post_init__(self):
        maximum = 180 if self.projection == 'perspective' else 360
        if self.projection not in ('perspective','cylindrical'):
            raise ValueError('unsupported virtual projection')
        if not isinstance(self.width,int) or not isinstance(self.height,int) or min(self.width,self.height)<=0:
            raise ValueError('virtual image dimensions must be positive integers')
        if not np.isfinite(self.hfov_deg) or not 0 < self.hfov_deg < maximum:
            raise ValueError('invalid virtual horizontal FOV')
        transform=np.array(self.camera_to_rig,dtype=np.float64,copy=True)
        validate_pose(transform)
        transform.setflags(write=False)
        object.__setattr__(self,'camera_to_rig',transform)
        if self.pose_timestamp_ns is not None and self.pose_timestamp_ns<0:
            raise ValueError('invalid pose timestamp')

    @property
    def focal(self):
        angle=np.deg2rad(self.hfov_deg)
        return self.width/(2*np.tan(angle/2)) if self.projection=='perspective' else self.width/angle


def default_virtual_camera(calibration: Calibration) -> VirtualCamera:
    """Left calibrated canvas, aligned with the rig, at the lens midpoint.

    This is a requested canvas, not a guarantee of measured coverage at any depth.
    Its pinhole FOV follows the left intrinsic matrix, not a nominal lens FOV.
    """
    left=calibration.left
    transform=np.eye(4)
    transform[:3,3]=calibration.right_to_left[:3,3]/2
    return VirtualCamera(left.width,left.height,float(np.rad2deg(2*np.arctan(left.width/(2*left.fx)))),camera_to_rig=transform)


@dataclass
class FusedView:
    rgb: np.ndarray
    depth_m: np.ndarray
    rgb_valid: np.ndarray
    depth_valid: np.ndarray
    confidence: np.ndarray
    provenance: np.ndarray
    source: np.ndarray
    approximate: np.ndarray
    timestamp_ns: int
    depth_convention: str
    requested_hfov_deg: float
    pose_timestamp_ns: int | None = None

    @property
    def coverage(self):
        return {'rgb_fraction':float(np.mean(self.rgb_valid)),
                'depth_fraction':float(np.mean(self.depth_valid)),
                'approximate_fraction':float(np.mean(self.approximate))}


def _transform(points, pose):
    return points @ pose[:3,:3].T + pose[:3,3]


def _source_cloud(image, depth, model, camera_to_rig):
    v,u=np.nonzero(depth.valid & np.isfinite(depth.depth_m) & (depth.depth_m>0))
    uv=np.column_stack((u,v)).astype(np.float64)
    if not len(uv):
        return np.empty((0,3)),np.empty((0,3),np.uint8),np.empty(0),np.empty(0,np.uint8)
    normalized=cv2.undistortPoints(uv[:,None,:],model.K,np.asarray(model.distortion or (0.,0.,0.,0.))).reshape(-1,2)
    xyz=np.column_stack((normalized,np.ones(len(uv))))*depth.depth_m[v,u,None]
    return _transform(xyz,camera_to_rig),image[v,u],depth.confidence[v,u],depth.provenance[v,u]


def _raw_projection(points, model):
    # OpenCV requires nonempty input and a supported distortion vector.
    uv=np.full((len(points),2),np.nan)
    positive=points[:,2]>0
    if positive.any():
        uv[positive]=cv2.projectPoints(points[positive],np.zeros(3),np.zeros(3),model.K,np.asarray(model.distortion or (0.,0.,0.,0.)))[0].reshape(-1,2)
    return uv


def _pixels(uv, width, height):
    finite=np.isfinite(uv).all(axis=1)
    pixels=np.zeros(uv.shape,dtype=np.int64)
    # Test continuous support before rounding to avoid accepting off-image rays.
    good=finite & (uv[:,0]>=-.5) & (uv[:,0]<width-.5) & (uv[:,1]>=-.5) & (uv[:,1]<height-.5)
    pixels[good]=np.floor(uv[good]+.5).astype(np.int64)
    return pixels,good


def _winner_indices(pixels, valid, depth, width):
    indices=np.flatnonzero(valid)
    keys=pixels[indices,1]*width+pixels[indices,0]
    # Original input index breaks equal-depth ties reproducibly.
    order=np.lexsort((indices,depth[indices],keys))
    sorted_indices=indices[order]; sorted_keys=keys[order]
    first=np.r_[True,sorted_keys[1:]!=sorted_keys[:-1]] if len(order) else np.empty(0,bool)
    return sorted_indices[first]


def _right_observations(cloud, frame, model):
    """Reuse only left-observed points visible in the right image z-buffer."""
    points,rgb,confidence,provenance=cloud
    right_points=_transform(points,np.linalg.inv(frame.calibration.right_to_left))
    uv=_raw_projection(right_points,model)
    pixels,valid=_pixels(uv,model.width,model.height)
    valid &= right_points[:,2]>0
    selected=_winner_indices(pixels,valid,right_points[:,2],model.width)
    p=pixels[selected]
    return points[selected],frame.right[p[:,1],p[:,0]],confidence[selected],provenance[selected]


def _project(points,camera):
    local=_transform(points,np.linalg.inv(camera.camera_to_rig))
    f=camera.focal; cx=(camera.width-1)/2; cy=(camera.height-1)/2
    with np.errstate(divide='ignore',invalid='ignore'):
        if camera.projection=='perspective':
            uv=np.column_stack((f*local[:,0]/local[:,2]+cx,f*local[:,1]/local[:,2]+cy))
            depth=local[:,2]
            supported=local[:,2]>0
        else:
            radius=np.hypot(local[:,0],local[:,2])
            uv=np.column_stack((f*np.arctan2(local[:,0],local[:,2])+cx,f*local[:,1]/radius+cy))
            depth=np.linalg.norm(local,axis=1)
            supported=radius>0
    pixels,valid=_pixels(uv,camera.width,camera.height)
    return pixels,valid & supported & np.isfinite(depth) & (depth>0),depth


def _render_layer(cloud,camera):
    points,rgb,confidence,provenance=cloud
    pixels,valid,depth=_project(points,camera)
    chosen=_winner_indices(pixels,valid,depth,camera.width)
    shape=(camera.height,camera.width)
    color=np.zeros((*shape,3),np.float64)
    z=np.full(shape,np.nan,np.float32)
    conf=np.zeros(shape,np.float32); prov=np.zeros(shape,np.uint8)
    p=pixels[chosen]; target=(p[:,1],p[:,0])
    color[target]=rgb[chosen]; z[target]=depth[chosen]
    conf[target]=confidence[chosen]; prov[target]=provenance[chosen]
    return color,z,conf,prov


def _infinity_layer(image,model,pose,camera):
    v,u=np.indices((camera.height,camera.width),dtype=np.float64)
    x=(u-(camera.width-1)/2)/camera.focal
    y=(v-(camera.height-1)/2)/camera.focal
    if camera.projection=='perspective':
        rays=np.stack((x,y,np.ones_like(x)),axis=-1).reshape(-1,3)
    else:
        rays=np.stack((np.sin(x),y,np.cos(x)),axis=-1).reshape(-1,3)
    # Directions transform by rotation only. Translation is intentionally ignored.
    directions=rays @ camera.camera_to_rig[:3,:3].T @ pose[:3,:3]
    uv=_raw_projection(directions,model)
    pixels,valid=_pixels(uv,model.width,model.height)
    color=np.zeros((len(rays),3),np.uint8)
    color[valid]=image[pixels[valid,1],pixels[valid,0]]
    return color.reshape(camera.height,camera.width,3),valid.reshape(camera.height,camera.width)


def _validate_depth(depth,frame,model):
    if depth.calibration_id!=frame.calibration.identity:
        raise ValueError('depth calibration identity mismatch')
    if depth.timestamp_ns!=frame.timestamp_ns:
        raise ValueError('depth timestamp mismatch')
    if depth.depth_m.shape!=(model.height,model.width):
        raise ValueError('depth must align with source RGB camera shape')
    # A DepthFrame is mutable; recheck its numerical invariants at this boundary.
    depth.__post_init__()


def synthesize(frame: StereoFrame, left_depth: DepthFrame, camera: VirtualCamera,
               right_depth: DepthFrame | None = None, mode='strict', exposure_match=True,
               left_camera: CameraModel | None = None, right_camera: CameraModel | None = None) -> FusedView:
    """Synthesize from depth aligned to raw source RGB (not rectified disparity).

    Optional camera overrides describe another intrinsic grid sharing the same
    calibrated optical poses. They do not change extrinsics or implicitly rectify,
    resample, or crop either source.
    Confidence chooses seams; equally confident, geometrically consistent samples
    blend. Infinity-plane preview fills carry no depth, confidence, or provenance.
    """
    if mode not in ('strict','preview'):
        raise ValueError('mode must be strict or preview')
    lc=left_camera or frame.calibration.left; rc=right_camera or frame.calibration.right
    for image,model in ((frame.left,lc),(frame.right,rc)):
        if image.shape!=(model.height,model.width,3):
            raise ValueError('source RGB camera shape mismatch')
    _validate_depth(left_depth,frame,lc)
    if right_depth is not None:
        _validate_depth(right_depth,frame,rc)
    left_cloud=_source_cloud(frame.left,left_depth,lc,np.eye(4))
    right_cloud=(_source_cloud(frame.right,right_depth,rc,frame.calibration.right_to_left)
                 if right_depth is not None else _right_observations(left_cloud,frame,rc))
    lrgb,lz,lcfd,lprov=_render_layer(left_cloud,camera)
    rrgb,rz,rcfd,rprov=_render_layer(right_cloud,camera)
    lv=np.isfinite(lz); rv=np.isfinite(rz)
    both=lv & rv
    compatible=both & (np.abs(lz-rz)<=.01+.01*np.minimum(lz,rz))
    if exposure_match and compatible.any():
        # Median ratios ignore dark/saturated samples and resist occluding outliers.
        for channel in range(3):
            usable=compatible & (lrgb[:,:,channel]>8) & (rrgb[:,:,channel]>8) & (lrgb[:,:,channel]<247) & (rrgb[:,:,channel]<247)
            if usable.any():
                gain=np.clip(np.median(lrgb[:,:,channel][usable]/rrgb[:,:,channel][usable]),.5,2.)
                rrgb[:,:,channel]*=gain
    # Depth ordering takes precedence over confidence at incompatible surfaces.
    right_wins=rv & (~lv | (both & ~compatible & (rz<lz)) | (compatible & (rcfd>lcfd)))
    rgb=np.where(right_wins[:,:,None],rrgb,lrgb)
    z=np.where(right_wins,rz,lz)
    conf=np.where(right_wins,rcfd,lcfd)
    prov=np.where(right_wins,rprov,lprov)
    valid=lv | rv
    source=np.where(right_wins,2,np.where(lv,1,0)).astype(np.uint8)
    blend=compatible & (np.abs(lcfd-rcfd)<=.1)
    # Average color only where geometry agrees. Keep winning depth, avoiding invented
    # surfaces when measured depths differ within the tolerance.
    weights=lcfd+rcfd
    alpha=np.divide(lcfd,weights,out=np.full_like(weights,.5),where=weights>0)
    rgb[blend]=(lrgb*alpha[:,:,None]+rrgb*(1-alpha[:,:,None]))[blend]
    source[blend]=3
    # Color blending does not fuse independent depth observations; preserve the
    # selected depth provenance (especially when both colors reuse one cloud).
    approximate=np.zeros(valid.shape,bool)
    rgb_valid=valid.copy()
    if mode=='preview':
        for source_id,image,model,pose in ((1,frame.left,lc,np.eye(4)),(2,frame.right,rc,frame.calibration.right_to_left)):
            preview,coverage=_infinity_layer(image,model,pose,camera)
            fill=coverage & ~rgb_valid
            rgb[fill]=preview[fill]; rgb_valid[fill]=True
            approximate[fill]=True; source[fill]=source_id
    return FusedView(np.clip(np.rint(rgb),0,255).astype(np.uint8),z.astype(np.float32),rgb_valid,valid,
                     conf.astype(np.float32),prov.astype(np.uint8),source,approximate,frame.timestamp_ns,
                     'axial_z' if camera.projection=='perspective' else 'ray_range',camera.hfov_deg,camera.pose_timestamp_ns)


def stabilize_virtual_camera(camera: VirtualCamera, rig_to_world_pose: np.ndarray, timestamp_ns: int) -> VirtualCamera:
    """Cancel rig rotation with a validated external pose; retain rig-relative position.

    This is orientation-only stabilization. It does not estimate motion, translate the
    viewpoint in world coordinates, or fill disocclusions.
    """
    pose=np.asarray(rig_to_world_pose,dtype=np.float64)
    validate_pose(pose)
    transform=np.array(camera.camera_to_rig,copy=True)
    transform[:3,:3]=pose[:3,:3].T @ camera.camera_to_rig[:3,:3]
    return replace(camera,camera_to_rig=transform,pose_timestamp_ns=timestamp_ns)
