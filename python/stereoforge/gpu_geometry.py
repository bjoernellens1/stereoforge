"""Strict Vulkan nearest-surface renderer. CPU upload/readback adapter is explicit.

The CPU oracle additionally supports seam blending, exposure matching and marked
preview. This renderer provides deterministic geometric projection, not those
photometric refinements. Source geometry is raw RGB-aligned axial depth.
"""
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from .fusion import FusedView, _validate_depth, _source_cloud, _raw_projection, _pixels, _winner_indices

class VulkanViewSynthesizer:
    def __init__(self):
        import slangpy as s
        self.s=s;self.device=s.create_device(s.DeviceType.vulkan,enable_cuda_interop=False)
        source=(Path(__file__).parent/'shaders/fusion/project.slang').read_text()
        module=self.device.load_module_from_source('stereoforge_project',source)
        self.kernels={n:self.device.create_compute_kernel(self.device.link_program([module],[module.entry_point(n)])) for n in ('clear','splat_depth','splat_owner','resolve')}
        self.buffers={};self.shape=None;self.last_timings={}
        self._ray_cache={}

    def _rays(self,camera):
        key=tuple(camera.to_dict()[k] for k in ('width','height','fx','fy','cx','cy'))+tuple(camera.distortion)
        if key not in self._ray_cache:
            y,x=np.indices((camera.height,camera.width))
            uv=np.stack((x,y),axis=-1).reshape(-1,1,2).astype(np.float64)
            points=cv2.undistortPoints(uv,camera.K,np.asarray(camera.distortion or (0.,0.,0.,0.))).reshape(-1,2)
            if len(self._ray_cache)>=2:self._ray_cache.pop(next(iter(self._ray_cache)))
            self._ray_cache[key]=np.column_stack((points,np.ones(len(points)))).astype(np.float32)
        return self._ray_cache[key]

    def render(self,frame,left_depth,camera,right_depth=None):
        start=perf_counter();cal=frame.calibration
        _validate_depth(left_depth,frame,cal.left)
        if right_depth is not None:_validate_depth(right_depth,frame,cal.right)
        h,w=left_depth.depth_m.shape;n=h*w;m=camera.width*camera.height
        if (cal.right.height,cal.right.width)!=(h,w):raise ValueError('GPU renderer requires equal source dimensions')
        if self.shape!=(h,w,camera.height,camera.width):self.device.wait_for_idle();self.buffers={};self.shape=(h,w,camera.height,camera.width)
        def buffer(name,count,stride=4):
            if name not in self.buffers:self.buffers[name]=self.device.create_buffer(element_count=count,struct_size=stride,usage=self.s.BufferUsage.shader_resource|self.s.BufferUsage.unordered_access)
            return self.buffers[name]
        rays_left=self._rays(cal.left);rays_right=self._rays(cal.right).copy()
        ld=np.where(left_depth.valid,left_depth.depth_m,np.nan).astype(np.float32)
        if right_depth is not None:
            rz=np.where(right_depth.valid,right_depth.depth_m,np.nan).astype(np.float32)
            rc=right_depth.confidence;rp=right_depth.provenance
        else:
            # Derive only visibility-checked right observations of measured left geometry.
            points,_,conf,prov=_source_cloud(frame.left,left_depth,cal.left,np.eye(4))
            inv=np.linalg.inv(cal.right_to_left);local=points@inv[:3,:3].T+inv[:3,3]
            uv=_raw_projection(local,cal.right);pixels,ok=_pixels(uv,w,h)
            ids=_winner_indices(pixels,ok&(local[:,2]>0),local[:,2],w)
            p=pixels[ids];z=local[ids,2]
            rz=np.full((h,w),np.nan,np.float32);rc=np.zeros((h,w),np.float32);rp=np.zeros((h,w),np.uint32)
            rz[p[:,1],p[:,0]]=z;rc[p[:,1],p[:,0]]=conf[ids];rp[p[:,1],p[:,0]]=prov[ids]
            # Retain actual projected geometry, rather than quantizing it to a new ray.
            rays_right[p[:,1]*w+p[:,0]]=(local[ids]/z[:,None]).astype(np.float32)
        def pack(image):
            a=image.astype(np.uint32);return a[:,:,0]|(a[:,:,1]<<8)|(a[:,:,2]<<16)
        values=dict(source_depth=np.concatenate((ld.ravel(),rz.ravel())),source_confidence=np.concatenate((left_depth.confidence.ravel(),rc.ravel())).astype(np.float32),
            source_rgb=np.concatenate((pack(frame.left).ravel(),pack(frame.right).ravel())),source_provenance=np.concatenate((left_depth.provenance.ravel(),rp.ravel())).astype(np.uint32),
            source_rays=np.concatenate((rays_left,rays_right)).astype(np.float32))
        for name,value in values.items():buffer(name,2*n,12 if name=='source_rays' else 4).copy_from_numpy(np.ascontiguousarray(value))
        outputs=('zbuffer','owner','output_rgb','output_valid','output_source','output_provenance','output_depth','output_confidence')
        for name in outputs:buffer(name,m)
        b=self.buffers;encoder=self.device.create_command_encoder()
        target=dict(target_width=camera.width,target_height=camera.height)
        project=dict(**target,src_width=w,src_height=h,projection=int(camera.projection=='cylindrical'),fx=camera.focal,fy=camera.focal,cx=(camera.width-1)/2,cy=(camera.height-1)/2,hfov_radians=float(np.deg2rad(camera.hfov_deg)),
            source_to_virtual_left=np.linalg.inv(camera.camera_to_rig).astype(np.float32),source_to_virtual_right=(np.linalg.inv(camera.camera_to_rig)@cal.right_to_left).astype(np.float32),source_depth=b['source_depth'],source_rays=b['source_rays'],zbuffer=b['zbuffer'])
        self.kernels['clear'].dispatch([m,1,1],vars=dict(**target,**{n:b[n] for n in outputs}),command_encoder=encoder)
        self.kernels['splat_depth'].dispatch([2*n,1,1],vars=project,command_encoder=encoder)
        self.kernels['splat_owner'].dispatch([2*n,1,1],vars=dict(**project,owner=b['owner']),command_encoder=encoder)
        self.kernels['resolve'].dispatch([m,1,1],vars=dict(**target,src_width=w,src_height=h,**{name:b[name] for name in ('owner','zbuffer','source_rgb','source_confidence','source_provenance',*outputs[2:])}),command_encoder=encoder)
        token=self.device.submit_command_buffer(encoder.finish());self.device.wait_for_submit(token)
        self.last_timings['upload_dispatch_wait_ms']=(perf_counter()-start)*1000
        shape=(camera.height,camera.width)
        def read(name,dtype):return b[name].to_numpy().view(dtype).reshape(shape).copy()
        packed=read('output_rgb',np.uint32)
        rgb=np.stack((packed&255,(packed>>8)&255,(packed>>16)&255),axis=-1).astype(np.uint8)
        valid=read('output_valid',np.uint32).astype(bool)
        result=FusedView(rgb,read('output_depth',np.float32),valid.copy(),valid,read('output_confidence',np.float32),read('output_provenance',np.uint32).astype(np.uint8),read('output_source',np.uint32).astype(np.uint8),np.zeros(shape,bool),frame.timestamp_ns,'axial_z' if camera.projection=='perspective' else 'ray_range',camera.hfov_deg,camera.pose_timestamp_ns)
        self.last_timings['total_with_readback_ms']=(perf_counter()-start)*1000
        return result
