// Experimental optional GPU feature atlas producer. Not yet a depth-graph default.
// Dispatch ceil(width/8),ceil(height/8),2 on the shared Vulkan/Daxa context.
#include "../ops/feature_atlas.slang"
struct FeatureAtlasParameters {
    uint64_t luma[2];
    uint64_t support[2];
    uint64_t atlas[2];
    uint width;
    uint height;
};
struct FeatureAtlasPush {uint64_t parameters;};
[[vk::push_constant]] FeatureAtlasPush pc;
#define p (*(FeatureAtlasParameters*)pc.parameters)
[shader("compute")][numthreads(8,8,1)]
void feature_atlas(uint3 tid:SV_DispatchThreadID){
    if(tid.z>=2||tid.x>=p.width||tid.y>=p.height)return;
    uint* luma=(uint*)p.luma[tid.z];
    uint* support=(uint*)p.support[tid.z];
    uint4* atlas=(uint4*)p.atlas[tid.z];
    uint index=tid.y*p.width+tid.x;
    atlas[index]=stereo_atlas_pixel(luma,support,p.width,p.height,tid.x,tid.y);
}
