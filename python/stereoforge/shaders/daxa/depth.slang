#ifndef SF_METRIC_PROVENANCE
#define SF_METRIC_PROVENANCE 1
#endif
#include "../ops/stereo_observation.slang"
struct Parameters {uint64_t inputs[8];uint64_t outputs[9];uint width;uint height;float tolerance;float fx;float baseline;float principal_offset;uint source_low;uint source_high;};
struct Push {uint64_t parameters;};[[vk::push_constant]] Push pc;
#define p (*(Parameters*)pc.parameters)
[shader("compute")][numthreads(64,1,1)]
void lr_consistency(uint3 tid:SV_DispatchThreadID){
    uint n=p.width*p.height,t=tid.x;if(t>=2*n)return;uint eye=t/n,i=t%n,x=i%p.width,target=0;
    float* own=(float*)p.inputs[eye*3];float* other=(float*)p.inputs[(1-eye)*3];float* confidence=(float*)p.inputs[eye*3+1];uint* valid=(uint*)p.inputs[eye*3+2];uint* other_valid=(uint*)p.inputs[(1-eye)*3+2];
    float d=own[i];bool present=false;
    if(valid[i]!=0&&observation_target(p.width,x,d,eye!=0,target)){uint j=(i/p.width)*p.width+target;present=observation_lr(d,other[j],valid[i],other_valid[j],p.tolerance);}
    ((float*)p.outputs[0])[t]=present?d:asfloat(0x7fc00000);((float*)p.outputs[1])[t]=present?confidence[i]:0;((uint*)p.outputs[2])[t]=present?1:0;
}
[shader("compute")][numthreads(64,1,1)]
void metric_observation(uint3 tid:SV_DispatchThreadID){
    uint n=p.width*p.height,t=tid.x;if(t>=2*n)return;uint eye=t/n,i=t%n,target=0;float d=((float*)p.outputs[0])[t];
    bool valid=((uint*)p.outputs[2])[t]!=0;float z=asfloat(0x7fc00000);
    if(valid&&observation_target(p.width,i%p.width,d,eye!=0,target)){uint j=(i/p.width)*p.width+target;valid=((uint*)p.inputs[6+eye])[i]!=0&&((uint*)p.inputs[7-eye])[j]!=0;if(valid){z=observation_z(d,p.fx,p.baseline,p.principal_offset);valid=isfinite(z)&&z>0;}}else valid=false;
    ((float*)p.outputs[3])[t]=valid?z:asfloat(0x7fc00000);((float*)p.outputs[4])[t]=valid?((float*)p.outputs[1])[t]:0;((uint*)p.outputs[5])[t]=valid?1:0;((uint*)p.outputs[6])[t]=valid?SF_METRIC_PROVENANCE:0;
    ((uint2*)p.outputs[7])[t]=valid?uint2(p.source_low,p.source_high):uint2(0xffffffff,0xffffffff);((uint2*)p.outputs[8])[t]=uint2(0,0);
}
