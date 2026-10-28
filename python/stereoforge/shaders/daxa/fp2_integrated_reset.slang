// Test-only reset of the entire reusable packed cost capacity.
struct Push {uint64_t costs; uint words;};
[[vk::push_constant]] ConstantBuffer<Push> pc;
[shader("compute")][numthreads(64,1,1)]
void reset_cost(uint3 tid:SV_DispatchThreadID){
    uint stride=64*min((pc.words+63)/64,65535u);
    for(uint i=tid.x;i<pc.words;i+=stride)((uint*)pc.costs)[i]=0xffffffffu;
}
