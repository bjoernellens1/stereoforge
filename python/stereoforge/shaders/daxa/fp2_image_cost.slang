// Isolated test-only packed Census cost producer.
struct CostParams {
    uint64_t lo; uint64_t hi; uint64_t offsets; uint64_t costs;
    uint64_t own; uint64_t other; uint64_t own_support; uint64_t other_support;
    uint width; uint height; uint eye;
};
struct CostPush { uint64_t parameters; };
[[vk::push_constant]] ConstantBuffer<CostPush> pc;
[shader("compute")][numthreads(64,1,1)]
void construct_cost(uint3 tid:SV_DispatchThreadID) {
    CostParams p=*(CostParams*)pc.parameters;
    uint i=tid.x; if(i>=p.width*p.height)return;
    uint* lo=(uint*)p.lo; uint* hi=(uint*)p.hi; uint64_t* offsets=(uint64_t*)p.offsets;
    uint* costs=(uint*)p.costs; uint* own=(uint*)p.own; uint* other=(uint*)p.other;
    uint* support=(uint*)p.own_support; uint* other_support=(uint*)p.other_support;
    uint x=i%p.width; uint row=i/p.width;
    for(uint d=lo[i];d<hi[i];++d){
        int ox=p.eye==0?int(x)-int(d):int(x)+int(d); uint value=0xffffffffu;
        if(support[i]!=0 && ox>=0 && ox<int(p.width)){
            uint j=row*p.width+uint(ox);
            if(other_support[j]!=0)value=countbits(own[i]^other[j]);
        }
        costs[offsets[i]+d-lo[i]]=value;
    }
}
