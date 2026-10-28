"""Explicit Vulkan execution through SlangPy. No automatic backend selection."""
from pathlib import Path
from time import perf_counter
import numpy as np
from .reference import DIRECTIONS
from ._sintergpu_reduction import reduction_source

class VulkanStereo:
    def __init__(self,max_disparity=128,paths=4,p1=8,p2=32,subpixel=True,debug=False):
        if not 1<=max_disparity<=256 or paths not in (4,8) or not 0<=p1<=p2<=10000:
            raise ValueError('invalid SGM parameters')
        source,self.shader_source_identity=reduction_source(Path(__file__).parent/'shaders/stereo/sgm.slang')
        import slangpy as s
        self.s=s
        self.device=s.create_device(s.DeviceType.vulkan,enable_debug_layers=debug,enable_cuda_interop=False)
        self.device_name=self.device.info.adapter_name
        self.D=max_disparity; self.paths=paths; self.p1=p1;self.p2=p2;self.subpixel=subpixel
        module=self.device.load_module_from_source('stereoforge_sgm',source)
        self.kernels={name:self.device.create_compute_kernel(self.device.link_program([module],[module.entry_point(name)])) for name in ('census','cost','aggregate','select')}
        self.buffers={};self.shape=None;self.last_timings={};self._pending=None
        self.query_pool=self.device.create_query_pool(s.QueryType.timestamp,32)

    def _buffer(self,name,count,dtype=np.uint32):
        # Reallocate only on shape/configuration changes.
        if name not in self.buffers:
            self.buffers[name]=self.device.create_buffer(element_count=count,struct_size=4,usage=self.s.BufferUsage.shader_resource|self.s.BufferUsage.unordered_access)
        return self.buffers[name]

    def enqueue(self,left,right,lower=None,upper=None):
        """Submit without readback; returns device-owned buffers + submission id.

        Caller retains inputs until upload completes and must not enqueue another
        frame until consuming this single-slot result. Production rings are separate.
        """
        if left.shape!=right.shape or left.ndim!=2: raise ValueError('expected matching grayscale HxW')
        if self._pending is not None: raise RuntimeError("consume pending submission before enqueue")
        h,w=left.shape
        if min(h,w)<5: raise ValueError('Census requires dimensions >=5')
        if (lower is None)!=(upper is None): raise ValueError('both bounds required')
        if lower is not None and (lower.shape!=left.shape or upper.shape!=left.shape or np.any(lower<0) or np.any(upper>self.D) or np.any(lower>=upper)):
            raise ValueError('invalid disparity bounds')
        if self.shape!=(h,w): self.device.wait_for_idle(); self.buffers={};self.shape=(h,w)
        start=perf_counter(); n=h*w
        for name,data in [('left',left),('right',right)]:
            self._buffer(name,n).copy_from_numpy(np.ascontiguousarray(data,np.uint32))
        for name in ('ld','rd','lv','rv','lower','upper','disparity','confidence','validity'): self._buffer(name,n)
        for name in ('costs','sums'): self._buffer(name,n*self.D)
        if lower is not None:
            self.buffers['lower'].copy_from_numpy(np.ascontiguousarray(lower,np.uint32))
            self.buffers['upper'].copy_from_numpy(np.ascontiguousarray(upper,np.uint32))
        b=self.buffers; encoder=self.device.create_command_encoder()
        query_index=0;self._stages=[]
        for source,desc,valid in [('left','ld','lv'),('right','rd','rv')]:
            self.kernels['census'].dispatch([n,1,1],vars=dict(image=b[source],descriptor=b[desc],descriptor_valid=b[valid],width=w,height=h),command_encoder=encoder,query_pool=self.query_pool,query_index_before=query_index,query_index_after=query_index+1)
            self._stages.append(source+'_census');query_index+=2
        common=dict(width=w,height=h,disparity_count=self.D)
        self.kernels['cost'].dispatch([n*self.D,1,1],vars=dict(**common,left_desc=b['ld'],right_desc=b['rd'],left_valid=b['lv'],right_valid=b['rv'],lower=b['lower'],upper=b['upper'],bounded=int(lower is not None),costs=b['costs'],sums=b['sums']),command_encoder=encoder,query_pool=self.query_pool,query_index_before=query_index,query_index_after=query_index+1)
        self._stages.append('matching_cost');query_index+=2
        for dx,dy in DIRECTIONS[:self.paths]:
            lines=h if dy==0 else w if dx==0 else w+h-1
            self.kernels['aggregate'].dispatch([lines*256,1,1],vars=dict(**common,costs=b['costs'],sums=b['sums'],dx=dx,dy=dy,line_count=lines,p1=self.p1,p2=self.p2),command_encoder=encoder,query_pool=self.query_pool,query_index_before=query_index,query_index_after=query_index+1)
            self._stages.append(f'aggregate_{dx}_{dy}');query_index+=2
        self.kernels['select'].dispatch([n*4,1,1],vars=dict(**common,sums=b['sums'],disparity=b['disparity'],confidence=b['confidence'],validity=b['validity'],subpixel=int(self.subpixel)),command_encoder=encoder,query_pool=self.query_pool,query_index_before=query_index,query_index_after=query_index+1)
        self._stages.append('select')
        token=self.device.submit_command_buffer(encoder.finish())
        self._pending=token
        self.last_timings={'upload_and_enqueue_ms':(perf_counter()-start)*1000}
        return token

    def result(self,token,debug=False,*,debug_census=False):
        if debug_census and not debug: raise ValueError('debug_census requires debug=True')
        if self._pending is None or token!=self._pending: raise ValueError("unknown or consumed submission token")
        start=perf_counter();self.device.wait_for_submit(token)
        self._pending=None
        timestamps=self.query_pool.get_timestamp_results(0,len(self._stages)*2)
        self.last_timings["gpu_stages_ms"]={name:(timestamps[2*i+1]-timestamps[2*i])*1000 for i,name in enumerate(self._stages)}
        self.last_timings['wait_ms']=(perf_counter()-start)*1000
        start=perf_counter();h,w=self.shape;b=self.buffers
        out={name:b[name].to_numpy().view(dtype).reshape(h,w).copy() for name,dtype in [('disparity',np.float32),('confidence',np.float32),('validity',np.uint32)]}
        out['valid']=out.pop('validity').astype(bool)
        if debug:
            out['left_census']=b['ld'].to_numpy().view(np.uint32).reshape(h,w).copy()
            out['cost']=b['costs'].to_numpy().view(np.uint32).reshape(h,w,self.D).copy()
            out['aggregate']=b['sums'].to_numpy().view(np.uint32).reshape(h,w,self.D).copy()
            if debug_census:
                for name,buffer in [('right_census','rd'),('left_census_valid','lv'),('right_census_valid','rv')]:
                    out[name]=b[buffer].to_numpy().view(np.uint32).reshape(h,w).copy()
        self.last_timings['readback_ms']=(perf_counter()-start)*1000
        return out

    def debug_stages(self,left,right):
        return self.result(self.enqueue(left,right),debug=True)
