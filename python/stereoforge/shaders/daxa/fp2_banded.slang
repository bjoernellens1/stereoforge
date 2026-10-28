// Isolated correctness prototype. One invocation owns one scanline.
struct Parameters {
    uint64_t lo; uint64_t hi; uint64_t offsets; uint64_t costs;
    uint64_t sums; uint64_t paths; uint64_t scratch; uint64_t line_offsets;
    uint width; uint height; uint disparities; uint p1; uint p2; uint max_lines;
};
struct Push { uint64_t parameters; uint pass; };
[[vk::push_constant]] ConstantBuffer<Push> pc;
#define p (*(Parameters*)pc.parameters)
static const uint INF=0xffffffffu;
[shader("compute")][numthreads(1,1,1)]
void aggregate(uint3 group:SV_GroupID) {
    uint line=group.x; bool horizontal=pc.pass<2;
    if(line >= (horizontal?p.height:p.width)) return;
    int2 dir=pc.pass==0?int2(1,0):pc.pass==1?int2(-1,0):pc.pass==2?int2(0,1):int2(0,-1);
    int2 xy=horizontal?int2(dir.x>0?0:int(p.width)-1,int(line)):int2(int(line),dir.y>0?0:int(p.height)-1);
    uint* lo=(uint*)p.lo; uint* hi=(uint*)p.hi; uint64_t* offsets=(uint64_t*)p.offsets;
    uint* cost=(uint*)p.costs; uint* sums=(uint*)p.sums; uint* paths=(uint*)p.paths;
    uint* scratch=(uint*)p.scratch; uint64_t* line_offsets=(uint64_t*)p.line_offsets;
    uint64_t line_index=uint64_t(pc.pass)*uint64_t(p.max_lines+1)+uint64_t(line);
    uint64_t base=line_offsets[line_index];uint64_t next=line_offsets[line_index+1];uint64_t cap=(next-base)/2;
    uint prev_lo=0,prev_width=0; bool swap=false;
    uint64_t allocated=offsets[uint64_t(p.width)*p.height];
    while(xy.x>=0 && xy.y>=0 && xy.x<int(p.width) && xy.y<int(p.height)) {
        uint i=uint(xy.y)*p.width+uint(xy.x);uint width=hi[i]-lo[i];
        uint64_t previous=base+(swap?cap:0);uint64_t current=base+(swap?0:cap);
        uint minimum=INF;
        for(uint t=0;t<prev_width;++t)minimum=min(minimum,scratch[previous+t]);
        for(uint t=0;t<width;++t){
            uint d=lo[i]+t;uint64_t j=offsets[i]+t;uint c=cost[j];uint v=INF;
            if(c!=INF){
                v=c;
                if(minimum!=INF){
                    uint best=minimum+p.p2;
                    if(d>=prev_lo&&d-prev_lo<prev_width)best=min(best,scratch[previous+(d-prev_lo)]);
                    if(d>0 && d-1>=prev_lo && d-1-prev_lo<prev_width){uint n=scratch[previous+(d-1-prev_lo)];if(n!=INF)best=min(best,n+p.p1);}
                    if(d+1>=prev_lo && d+1-prev_lo<prev_width){uint n=scratch[previous+(d+1-prev_lo)];if(n!=INF)best=min(best,n+p.p1);}
                    v+=best-minimum;
                }
                if(pc.pass==0)sums[j]=v;else if(sums[j]!=INF)sums[j]+=v;
                paths[uint64_t(pc.pass)*allocated+j]=v;
            }
            scratch[current+t]=v;
        }
        prev_lo=lo[i];prev_width=width;swap=!swap;xy+=dir;
    }
}
