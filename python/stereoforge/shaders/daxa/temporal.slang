#include "../ops/temporal.slang"
struct Push{uint64_t parameters;};[[vk::push_constant]] Push pc;
#define p (*(TemporalParameters*)pc.parameters)
[shader("compute")][numthreads(64,1,1)] void clear(uint3 t:SV_DispatchThreadID){temporal_clear(p,t.x);}
[shader("compute")][numthreads(64,1,1)] void splat_z(uint3 t:SV_DispatchThreadID){temporal_splat_z(p,t.x);}
[shader("compute")][numthreads(64,1,1)] void splat_owner(uint3 t:SV_DispatchThreadID){temporal_splat_owner(p,t.x);}
[shader("compute")][numthreads(64,1,1)] void gather_fuse(uint3 t:SV_DispatchThreadID){temporal_fuse(p,t.x);}
