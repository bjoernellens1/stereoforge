// Deterministic nearest-surface forward projection. Host invalidates unsupported
// source depths before upload and supplies undistorted, unit-Z source rays.
StructuredBuffer<float> source_depth;
StructuredBuffer<float> source_confidence;
StructuredBuffer<uint> source_rgb;
StructuredBuffer<uint> source_provenance;
StructuredBuffer<float3> source_rays;
RWStructuredBuffer<uint> zbuffer;
RWStructuredBuffer<uint> owner;
RWStructuredBuffer<uint> output_rgb;
RWStructuredBuffer<uint> output_valid;
RWStructuredBuffer<uint> output_source;
RWStructuredBuffer<uint> output_provenance;
RWStructuredBuffer<float> output_depth;
RWStructuredBuffer<float> output_confidence;

uniform uint src_width;
uniform uint src_height;
uniform uint target_width;
uniform uint target_height;
uniform uint projection;
uniform float fx;
uniform float fy;
uniform float cx;
uniform float cy;
uniform float hfov_radians;
uniform float4x4 source_to_virtual_left;
uniform float4x4 source_to_virtual_right;

bool project_source(uint index, out uint target, out float depth)
{
    target=0; depth=0;
    float axial=source_depth[index];
    if (!isfinite(axial) || axial<=0) return false;
    uint eye_size=src_width*src_height;
    float3 point=source_rays[index]*axial;
    float4 local=index<eye_size ? mul(source_to_virtual_left,float4(point,1))
                               : mul(source_to_virtual_right,float4(point,1));
    if (!all(isfinite(local))) return false;
    float u,v;
    if (projection==0)
    {
        if (local.z<=0) return false;
        u=fx*local.x/local.z+cx;
        v=fy*local.y/local.z+cy;
        depth=local.z;
    }
    else
    {
        float radius=length(local.xz);
        if (!isfinite(radius) || radius<=0) return false;
        u=(atan2(local.x,local.z)/hfov_radians+.5)*target_width-.5;
        v=fy*local.y/radius+cy;
        depth=length(local.xyz);
    }
    if (!isfinite(u) || !isfinite(v) || !isfinite(depth) || depth<=0 ||
        u<-.5 || u>=float(target_width)-.5 ||
        v<-.5 || v>=float(target_height)-.5) return false;
    uint x=uint(floor(u+.5));
    uint y=uint(floor(v+.5));
    target=y*target_width+x;
    return true;
}

[shader("compute")]
[numthreads(64,1,1)]
void clear(uint3 dispatch_id : SV_DispatchThreadID)
{
    uint index=dispatch_id.x;
    if (index>=target_width*target_height) return;
    zbuffer[index]=0x7f800000;
    owner[index]=0xffffffff;
    output_rgb[index]=0;
    output_valid[index]=0;
    output_source[index]=0;
    output_provenance[index]=0;
    output_depth[index]=asfloat(0x7fc00000);
    output_confidence[index]=0;
}

[shader("compute")]
[numthreads(64,1,1)]
void splat_depth(uint3 dispatch_id : SV_DispatchThreadID)
{
    uint index=dispatch_id.x;
    if (index>=src_width*src_height*2) return;
    uint target; float depth;
    if (!project_source(index,target,depth)) return;
    uint ignored;
    InterlockedMin(zbuffer[target],asuint(depth),ignored);
}

[shader("compute")]
[numthreads(64,1,1)]
void splat_owner(uint3 dispatch_id : SV_DispatchThreadID)
{
    uint index=dispatch_id.x;
    if (index>=src_width*src_height*2) return;
    uint target; float depth;
    if (!project_source(index,target,depth) || zbuffer[target]!=asuint(depth)) return;
    uint ignored;
    InterlockedMin(owner[target],index,ignored);
}

[shader("compute")]
[numthreads(64,1,1)]
void resolve(uint3 dispatch_id : SV_DispatchThreadID)
{
    uint target=dispatch_id.x;
    if (target>=target_width*target_height) return;
    uint index=owner[target];
    if (index==0xffffffff) return;
    output_rgb[target]=source_rgb[index];
    output_valid[target]=1;
    output_source[target]=index<src_width*src_height ? 1 : 2;
    output_provenance[target]=source_provenance[index];
    output_depth[target]=asfloat(zbuffer[target]);
    output_confidence[target]=source_confidence[index];
}
