// Test-only support-aware Census stage; shared refinement semantics.
#include "../ops/refinement.slang"
struct CensusParams { uint64_t luma; uint64_t support; uint64_t descriptor; uint64_t valid; uint width; uint height; };
struct CensusPush { uint64_t parameters; };
[[vk::push_constant]] ConstantBuffer<CensusPush> pc;
[shader("compute")][numthreads(64,1,1)]
void census_image(uint3 tid:SV_DispatchThreadID){
    CensusParams p=*(CensusParams*)pc.parameters;uint i=tid.x;if(i>=p.width*p.height)return;
    uint x=i%p.width,y=i/p.width;uint* gray=(uint*)p.luma;uint* support=(uint*)p.support;
    uint* descriptor=(uint*)p.descriptor;uint* valid=(uint*)p.valid;
    bool ok=refinement_support(support,p.width,p.height,x,y);
    descriptor[i]=ok?refinement_census(gray,p.width,x,y):0;valid[i]=uint(ok);
}
