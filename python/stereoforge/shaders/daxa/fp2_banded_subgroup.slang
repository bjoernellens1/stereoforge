// Isolated 128-lane scanline experiment; not work-proportional to compact widths.
struct Parameters {
    uint64_t lo; uint64_t hi; uint64_t offsets; uint64_t costs;
    uint64_t sums; uint64_t paths; uint64_t scratch; uint64_t line_offsets;
    uint width; uint height; uint disparities; uint p1; uint p2; uint max_lines;
};
struct Push { uint64_t parameters; uint pass; };
[[vk::push_constant]] ConstantBuffer<Push> pc;
#define p (*(Parameters*)pc.parameters)
static const uint INF=0xffffffffu;
groupshared uint previous[128];
groupshared uint current[128];
groupshared uint group_minimum;

[shader("compute")][numthreads(128,1,1)]
void aggregate(uint3 group:SV_GroupID, uint lane:SV_GroupIndex) {
    uint line=group.x; bool horizontal=pc.pass<2;
    // This condition is uniform across the workgroup.
    if(line >= (horizontal?p.height:p.width)) return;
    int2 dir=pc.pass==0?int2(1,0):pc.pass==1?int2(-1,0):pc.pass==2?int2(0,1):int2(0,-1);
    int2 xy=horizontal?int2(dir.x>0?0:int(p.width)-1,int(line)):int2(int(line),dir.y>0?0:int(p.height)-1);
    uint* lo=(uint*)p.lo; uint* hi=(uint*)p.hi; uint64_t* offsets=(uint64_t*)p.offsets;
    uint* cost=(uint*)p.costs; uint* sums=(uint*)p.sums; uint* paths=(uint*)p.paths;
    uint64_t allocated=offsets[uint64_t(p.width)*p.height];
    previous[lane]=INF;
    GroupMemoryBarrierWithGroupSync();
    while(xy.x>=0 && xy.y>=0 && xy.x<int(p.width) && xy.y<int(p.height)) {
        uint i=uint(xy.y)*p.width+uint(xy.x);
        if(lane==0) group_minimum=INF;
        GroupMemoryBarrierWithGroupSync();
        uint subgroup_minimum=WaveActiveMin(previous[lane]);
        if(WaveIsFirstLane()) InterlockedMin(group_minimum,subgroup_minimum);
        GroupMemoryBarrierWithGroupSync();
        uint minimum=group_minimum;
        uint value=INF;
        if(lane<p.disparities && lane>=lo[i] && lane<hi[i]) {
            uint64_t j=offsets[i]+lane-lo[i]; uint c=cost[j];
            if(c!=INF) {
                value=c;
                if(minimum!=INF) {
                    uint best=min(minimum+p.p2,previous[lane]);
                    if(lane>0 && previous[lane-1]!=INF) best=min(best,previous[lane-1]+p.p1);
                    if(lane+1<p.disparities && previous[lane+1]!=INF) best=min(best,previous[lane+1]+p.p1);
                    value+=best-minimum;
                }
                if(pc.pass==0) sums[j]=value; else if(sums[j]!=INF) sums[j]+=value;
                if(p.paths!=0) paths[uint64_t(pc.pass)*allocated+j]=value;
            }
        }
        current[lane]=value;
        GroupMemoryBarrierWithGroupSync();
        previous[lane]=current[lane];
        GroupMemoryBarrierWithGroupSync();
        xy+=dir;
    }
}
