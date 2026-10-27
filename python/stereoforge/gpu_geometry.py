"""Deterministic Vulkan geometry, compatible seams, exposure and marked preview.

CPU upload/readback adapter is explicit. Calibration-only OpenCV inverse maps
are bounded to two configurations; no image data is used in map construction.
"""
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from .fusion import FusedView, _validate_depth, _source_cloud, _raw_projection, _pixels, _winner_indices

class VulkanViewSynthesizer:
    def __init__(self,profile=False,*,device=None):
        import slangpy as s
        self.s=s;self.device=device if device is not None else s.create_device(s.DeviceType.vulkan,enable_cuda_interop=False)
        if self.device.info.type!=s.DeviceType.vulkan:raise ValueError("View synthesis requires a Vulkan device")
        source=(Path(__file__).parent/'shaders/fusion/project.slang').read_text()
        module=self.device.load_module_from_source('stereoforge_project',source)
        self.kernels={n:self.device.create_compute_kernel(self.device.link_program([module],[module.entry_point(n)])) for n in ('clear','splat_depth','splat_owner','resolve','clear_layers','splat_layer_depth','splat_layer_owner','histogram_ratios','reduce_ratios','resolve_layers','pack_readback')}
        self.buffers={};self.shape=None;self.last_timings={}
        self._timestamp_pool=self.device.create_query_pool(s.QueryType.timestamp,16) if profile else None
        self._ray_cache={};self._preview_cache={};self._uploaded_preview=None

    def _rays(self,camera):
        key=tuple(camera.to_dict()[k] for k in ('width','height','fx','fy','cx','cy'))+tuple(camera.distortion)
        if key not in self._ray_cache:
            y,x=np.indices((camera.height,camera.width))
            uv=np.stack((x,y),axis=-1).reshape(-1,1,2).astype(np.float64)
            points=cv2.undistortPoints(uv,camera.K,np.asarray(camera.distortion or (0.,0.,0.,0.))).reshape(-1,2)
            if len(self._ray_cache)>=2:self._ray_cache.pop(next(iter(self._ray_cache)))
            self._ray_cache[key]=np.column_stack((points,np.ones(len(points)))).astype(np.float32)
        return self._ray_cache[key]

    def _preview_map(self,cal,camera):
        # Translation intentionally omitted: these are infinity directions, not geometry.
        key=(cal.identity,camera.width,camera.height,camera.projection,camera.hfov_deg,
             camera.camera_to_rig[:3,:3].tobytes())
        if key not in self._preview_cache:
            v,u=np.indices((camera.height,camera.width),dtype=np.float64)
            x=(u-(camera.width-1)/2)/camera.focal
            y=(v-(camera.height-1)/2)/camera.focal
            rays=(np.stack((x,y,np.ones_like(x)),axis=-1) if camera.projection=='perspective'
                  else np.stack((np.sin(x),y,np.cos(x)),axis=-1)).reshape(-1,3)
            layers=[]
            for eye,model,pose in ((0,cal.left,np.eye(4)),(1,cal.right,cal.right_to_left)):
                directions=rays@camera.camera_to_rig[:3,:3].T@pose[:3,:3]
                pixels,valid=_pixels(_raw_projection(directions,model),model.width,model.height)
                indices=np.full(len(rays),0xffffffff,np.uint32)
                indices[valid]=(eye*model.width*model.height+pixels[valid,1]*model.width+pixels[valid,0]).astype(np.uint32)
                layers.append(indices)
            if len(self._preview_cache)>=2:self._preview_cache.pop(next(iter(self._preview_cache)))
            self._preview_cache[key]=np.concatenate(layers)
        return key,self._preview_cache[key]

    @staticmethod
    def _decode_readback(packed):
        color=packed[:,:,0];flags=packed[:,:,3]
        return dict(rgb=np.stack((color&255,(color>>8)&255,(color>>16)&255),axis=-1).astype(np.uint8),
            depth_m=packed[:,:,1].view(np.float32).copy(),confidence=packed[:,:,2].view(np.float32).copy(),
            depth_valid=(flags&1)!=0,rgb_valid=(flags&2)!=0,approximate=(flags&4)!=0,
            source=((flags>>8)&3).astype(np.uint8),provenance=((flags>>16)&255).astype(np.uint8))

    def render(self,frame,left_depth,camera,right_depth=None,mode="strict",exposure_match=True):
        if mode not in ("strict","preview"):raise ValueError("mode must be strict or preview")
        start=perf_counter();cal=frame.calibration
        _validate_depth(left_depth,frame,cal.left)
        if right_depth is not None:_validate_depth(right_depth,frame,cal.right)
        h,w=left_depth.depth_m.shape;n=h*w;m=camera.width*camera.height
        if (cal.right.height,cal.right.width)!=(h,w):raise ValueError('GPU renderer requires equal source dimensions')
        if self.shape!=(h,w,camera.height,camera.width):self.device.wait_for_idle();self.buffers={};self.shape=(h,w,camera.height,camera.width);self._uploaded_preview=None
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
        for name in ('layer_z','layer_owner','preview_indices'):buffer(name,2*m)
        for name in ('output_rgb_valid','output_approximate'):buffer(name,m)
        buffer('ratio_histogram',3*4096);buffer('exposure_gain',3);buffer('packed_readback',m,16)
        map_start=perf_counter()
        if mode=='preview':
            key,mapping=self._preview_map(cal,camera)
            if self._uploaded_preview!=key:
                self.buffers['preview_indices'].copy_from_numpy(mapping)
                self._uploaded_preview=key
        self.last_timings['calibration_preview_map_ms']=(perf_counter()-map_start)*1000
        b=self.buffers;encoder=self.device.create_command_encoder()
        query=self._timestamp_pool
        if query is not None:query.reset()
        def timing(index):
            return {} if query is None else dict(query_pool=query,query_index_before=2*index,query_index_after=2*index+1)
        target=dict(target_width=camera.width,target_height=camera.height)
        project=dict(**target,src_width=w,src_height=h,projection=int(camera.projection=='cylindrical'),fx=camera.focal,fy=camera.focal,cx=(camera.width-1)/2,cy=(camera.height-1)/2,hfov_radians=float(np.deg2rad(camera.hfov_deg)),
            source_to_virtual_left=np.linalg.inv(camera.camera_to_rig).astype(np.float32),source_to_virtual_right=(np.linalg.inv(camera.camera_to_rig)@cal.right_to_left).astype(np.float32),source_depth=b['source_depth'],source_rays=b['source_rays'],zbuffer=b['zbuffer'])
        self.kernels['clear'].dispatch([m,1,1],vars=dict(**target,**{n:b[n] for n in outputs}),command_encoder=encoder,**timing(0))
        self.kernels['clear_layers'].dispatch([max(2*m,3*4096),1,1],vars=dict(**target,**{name:b[name] for name in ('layer_z','layer_owner','ratio_histogram','exposure_gain','output_rgb_valid','output_approximate')}),command_encoder=encoder,**timing(1))
        layer_project=dict(project,layer_z=b['layer_z'])
        self.kernels['splat_layer_depth'].dispatch([2*n,1,1],vars=layer_project,command_encoder=encoder,**timing(2))
        self.kernels['splat_layer_owner'].dispatch([2*n,1,1],vars=dict(layer_project,layer_owner=b['layer_owner']),command_encoder=encoder,**timing(3))
        layers={name:b[name] for name in ('layer_z','layer_owner','source_rgb','source_confidence','source_provenance')}
        self.kernels['histogram_ratios'].dispatch([m,1,1],vars=dict(**target,**layers,exposure_enabled=int(exposure_match),ratio_histogram=b['ratio_histogram']),command_encoder=encoder,**timing(4))
        self.kernels['reduce_ratios'].dispatch([3,1,1],vars=dict(ratio_histogram=b['ratio_histogram'],exposure_gain=b['exposure_gain']),command_encoder=encoder,**timing(5))
        self.kernels['resolve_layers'].dispatch([m,1,1],vars=dict(**target,**layers,owner=b['owner'],preview_enabled=int(mode=='preview'),preview_indices=b['preview_indices'],exposure_gain=b['exposure_gain'],**{name:b[name] for name in (*outputs[2:],'output_rgb_valid','output_approximate')}),command_encoder=encoder,**timing(6))
        self.kernels['pack_readback'].dispatch([m,1,1],vars=dict(**target,**{name:b[name] for name in (*outputs[2:],'output_rgb_valid','output_approximate','packed_readback')}),command_encoder=encoder,**timing(7))
        token=self.device.submit_command_buffer(encoder.finish())
        self.last_timings['host_prepare_upload_encode_submit_ms']=(perf_counter()-start)*1000
        wait_start=perf_counter();self.device.wait_for_submit(token)
        self.last_timings['wait_ms']=(perf_counter()-wait_start)*1000
        self.last_timings['upload_dispatch_wait_ms']=(perf_counter()-start)*1000
        if query is not None:
            # Completed above: fixed-size timestamp readback introduces no extra wait.
            ticks=query.get_results(0,16)
            frequency=self.device.info.timestamp_frequency
            names=('clear','clear_layers','splat_layer_depth','splat_layer_owner','histogram_ratios','reduce_ratios','resolve_layers','pack_readback')
            self.last_timings['gpu_stages_ms']={name:(ticks[2*i+1]-ticks[2*i])*1000/frequency for i,name in enumerate(names)}
        readback_start=perf_counter()
        shape=(camera.height,camera.width)
        packed=b['packed_readback'].to_numpy().view(np.uint32).reshape(*shape,4)
        decoded=self._decode_readback(packed)
        result=FusedView(**decoded,timestamp_ns=frame.timestamp_ns,depth_convention='axial_z' if camera.projection=='perspective' else 'ray_range',requested_hfov_deg=camera.hfov_deg,pose_timestamp_ns=camera.pose_timestamp_ns)
        self.last_timings['readback_unpack_ms']=(perf_counter()-readback_start)*1000
        self.last_timings['total_with_readback_ms']=(perf_counter()-start)*1000
        return result
