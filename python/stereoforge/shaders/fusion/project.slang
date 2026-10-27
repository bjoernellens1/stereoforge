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

// Research fusion ABI: two target-sized uint depth/owner layers (left then
// right), 3*4096 uint histogram bins and three float gains. No intermediate
// readback. Positive float bits preserve depth ordering; source index breaks ties.
RWStructuredBuffer<uint> layer_z;
RWStructuredBuffer<uint> layer_owner;
RWStructuredBuffer<uint> ratio_histogram;
RWStructuredBuffer<float> exposure_gain;
StructuredBuffer<uint> preview_indices;
RWStructuredBuffer<uint> output_rgb_valid;
RWStructuredBuffer<uint> output_approximate;
uniform uint preview_enabled;
uniform uint exposure_enabled;

float3 unpack_color(uint packed)
{
    return float3(packed&255,(packed>>8)&255,(packed>>16)&255);
}
uint pack_color(float3 color)
{
    uint3 value=uint3(round(clamp(color,0.,255.)));
    return value.x|(value.y<<8)|(value.z<<16);
}
bool compatible_layers(uint target)
{
    uint count=target_width*target_height;
    float left=asfloat(layer_z[target]),right=asfloat(layer_z[count+target]);
    return isfinite(left)&&isfinite(right)&&abs(left-right)<=.01+.01*min(left,right);
}
[shader("compute")]
[numthreads(64,1,1)]
void clear_layers(uint3 tid:SV_DispatchThreadID)
{
    uint i=tid.x,count=target_width*target_height;
    if(i<2*count){layer_z[i]=0x7f800000;layer_owner[i]=0xffffffff;}
    if(i<3*4096) ratio_histogram[i]=0;
    if(i<3) exposure_gain[i]=1.;
    if(i<count){output_rgb_valid[i]=0;output_approximate[i]=0;}
}
[shader("compute")]
[numthreads(64,1,1)]
void splat_layer_depth(uint3 tid:SV_DispatchThreadID)
{
    uint i=tid.x,n=src_width*src_height;
    if(i>=2*n)return;
    uint target;float z;
    if(!project_source(i,target,z))return;
    uint ignored;
    InterlockedMin(layer_z[(i/n)*target_width*target_height+target],asuint(z),ignored);
}
[shader("compute")]
[numthreads(64,1,1)]
void splat_layer_owner(uint3 tid:SV_DispatchThreadID)
{
    uint i=tid.x,n=src_width*src_height;
    if(i>=2*n)return;
    uint target;float z;
    if(!project_source(i,target,z))return;
    uint cell=(i/n)*target_width*target_height+target;
    if(layer_z[cell]!=asuint(z))return;
    uint ignored;InterlockedMin(layer_owner[cell],i,ignored);
}
[shader("compute")]
[numthreads(64,1,1)]
void histogram_ratios(uint3 tid:SV_DispatchThreadID)
{
    uint t=tid.x,count=target_width*target_height;
    if(t>=count||exposure_enabled==0||!compatible_layers(t))return;
    float3 l=unpack_color(source_rgb[layer_owner[t]]);
    float3 r=unpack_color(source_rgb[layer_owner[count+t]]);
    for(uint c=0;c<3;c++)if(l[c]>8&&l[c]<247&&r[c]>8&&r[c]<247){
        // Valid integer colors bound ratios to [9/246,246/9]. Symmetric
        // log bins preserve both middle values before the final gain clamp.
        // Relative quantization error <=0.081%; color error <1 intensity.
        float limit=log2(246./9.);
        uint bin=uint(round((log2(l[c]/r[c])+limit)*(4095./(2*limit))));
        uint ignored;InterlockedAdd(ratio_histogram[c*4096+bin],1,ignored);
    }
}
[shader("compute")]
[numthreads(1,1,1)]
void reduce_ratios(uint3 tid:SV_DispatchThreadID)
{
    uint c=tid.x;if(c>=3)return;
    uint total=0;
    for(uint b=0;b<4096;b++)total+=ratio_histogram[c*4096+b];
    if(total==0)return;
    uint lower=(total-1)/2,upper=total/2,cumulative=0;
    float low=.5,high=.5;bool got_low=false;
    for(uint b=0;b<4096;b++){
        cumulative+=ratio_histogram[c*4096+b];
        float limit=log2(246./9.);
        float ratio=exp2(float(b)*(2*limit/4095.)-limit);
        if(!got_low&&cumulative>lower){low=ratio;got_low=true;}
        if(cumulative>upper){high=ratio;break;}
    }
    // Both middle order statistics are averaged for a true even sample median.
    exposure_gain[c]=clamp((low+high)*.5,.5,2.);
}
[shader("compute")]
[numthreads(64,1,1)]
void resolve_layers(uint3 tid:SV_DispatchThreadID)
{
    uint t=tid.x,count=target_width*target_height;if(t>=count)return;
    uint l=layer_owner[t],r=layer_owner[count+t];
    bool lv=l!=0xffffffff,rv=r!=0xffffffff;
    if(!lv&&!rv){
        if(preview_enabled!=0){
            uint index=preview_indices[t],source=1;
            if(index==0xffffffff){index=preview_indices[count+t];source=2;}
            if(index!=0xffffffff){owner[t]=index;output_rgb[t]=source_rgb[index];output_rgb_valid[t]=1;output_source[t]=source;output_approximate[t]=1;}
        }
        return;
    }
    bool compatible=lv&&rv&&compatible_layers(t);
    float lz=asfloat(layer_z[t]),rz=asfloat(layer_z[count+t]);
    bool right=rv&&(!lv||(!compatible&&rz<lz)||(compatible&&source_confidence[r]>source_confidence[l]));
    uint chosen=right?r:l;
    float3 gains=float3(exposure_gain[0],exposure_gain[1],exposure_gain[2]);
    float3 color=unpack_color(source_rgb[chosen])*(right?gains:float3(1));
    uint attribution=right?2:1;
    if(compatible&&abs(source_confidence[l]-source_confidence[r])<=.1){
        float weight=source_confidence[l]+source_confidence[r];
        float alpha=weight>0?source_confidence[l]/weight:.5;
        color=unpack_color(source_rgb[l])*alpha+unpack_color(source_rgb[r])*gains*(1-alpha);
        attribution=3;
    }
    owner[t]=chosen;output_rgb[t]=pack_color(color);output_valid[t]=1;output_rgb_valid[t]=1;
    output_source[t]=attribution;output_provenance[t]=source_provenance[chosen];
    output_depth[t]=right?rz:lz;output_confidence[t]=source_confidence[chosen];
}

// Python adapter readback ABI, one uint4 (16 bytes) per target pixel:
// RGB888, float axial/range depth bits, float confidence bits, metadata bits.
// Metadata: depth valid bit0, RGB valid bit1, approximate bit2,
// color source bits8..9, metric provenance bits16..23. Native output resources
// remain separate; this optional pass only batches the CPU research readback.
RWStructuredBuffer<uint4> packed_readback;
[shader("compute")]
[numthreads(64,1,1)]
void pack_readback(uint3 tid:SV_DispatchThreadID)
{
    uint t=tid.x;if(t>=target_width*target_height)return;
    uint flags=(output_valid[t]&1)|((output_rgb_valid[t]&1)<<1)|
        ((output_approximate[t]&1)<<2)|((output_source[t]&3)<<8)|
        ((output_provenance[t]&255)<<16);
    packed_readback[t]=uint4(output_rgb[t],asuint(output_depth[t]),asuint(output_confidence[t]),flags);
}
