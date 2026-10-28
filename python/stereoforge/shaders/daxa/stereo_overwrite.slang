#include "stereo_parameters.slang"
#include "../ops/sgm_overwrite.slang"
struct Push {uint64_t parameters;uint pass;uint groups_x;};
[[vk::push_constant]] ConstantBuffer<Push> pc;
#define p (*(Parameters*)pc.parameters)
[shader("compute")][numthreads(64,1,1)] void aggregate_init_left8(uint3 gid:SV_GroupID){aggregate_overwrite_body<4,8,false>(p,pc.pass,pc.groups_x,gid);}
[shader("compute")][numthreads(64,1,1)] void aggregate_init_left16(uint3 gid:SV_GroupID){aggregate_overwrite_body<2,16,false>(p,pc.pass,pc.groups_x,gid);}
[shader("compute")][numthreads(64,1,1)] void aggregate_init_right8(uint3 gid:SV_GroupID){aggregate_overwrite_body<4,8,true>(p,pc.pass,pc.groups_x,gid);}
[shader("compute")][numthreads(64,1,1)] void aggregate_init_right16(uint3 gid:SV_GroupID){aggregate_overwrite_body<2,16,true>(p,pc.pass,pc.groups_x,gid);}
