#include "stereo_parameters.slang"
#include "../ops/sgm_specialized.slang"
// A separate wrapper ABI; legacy entrypoints retain their original 16-byte Push.
struct LayeredPush {uint64_t left_parameters,right_parameters;uint pass,groups_x;};
[[vk::push_constant]] ConstantBuffer<LayeredPush> layered_pc;

void aggregate_layered_body<let Components:uint,let Bits:uint>(uint3 gid){
    // Eye is workgroup-uniform. Reject before loading either parameter block or
    // reaching full-wave collectives; z never contributes to a scanline index.
    if(gid.z>=2)return;
    if(gid.z==0)aggregate_specialized_body<Components,Bits,false>(*(Parameters*)layered_pc.left_parameters,layered_pc.pass,layered_pc.groups_x,gid);
    else aggregate_specialized_body<Components,Bits,true>(*(Parameters*)layered_pc.right_parameters,layered_pc.pass,layered_pc.groups_x,gid);
}
void select_layered_body<let Components:uint,let Bits:uint>(uint3 gid,uint3 tid){
    // All 256 threads of one workgroup take the same direction branch, so every
    // lane reaches the unchanged shared-memory reduction barriers together.
    if(gid.z>=2)return;
    if(gid.z==0)select_specialized_body<Components,Bits,false>(*(Parameters*)layered_pc.left_parameters,layered_pc.groups_x,gid,tid);
    else select_specialized_body<Components,Bits,true>(*(Parameters*)layered_pc.right_parameters,layered_pc.groups_x,gid,tid);
}
[shader("compute")][numthreads(64,1,1)]
void aggregate_layered8(uint3 gid:SV_GroupID){aggregate_layered_body<4,8>(gid);}
[shader("compute")][numthreads(64,1,1)]
void aggregate_layered16(uint3 gid:SV_GroupID){aggregate_layered_body<2,16>(gid);}
[shader("compute")][numthreads(256,1,1)]
void select_layered8(uint3 gid:SV_GroupID,uint3 tid:SV_GroupThreadID){select_layered_body<4,8>(gid,tid);}
[shader("compute")][numthreads(256,1,1)]
void select_layered16(uint3 gid:SV_GroupID,uint3 tid:SV_GroupThreadID){select_layered_body<2,16>(gid,tid);}
