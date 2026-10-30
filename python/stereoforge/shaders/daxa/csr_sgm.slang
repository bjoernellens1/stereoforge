// Experimental four-path work-proportional CSR SGM. GPU parity/performance OPEN.
// Dispatch scan_path with (height,1,1) for pass 0/1 and (width,1,1)
// for pass 2/3, each pass independently writing its own state region.
// Then dispatch sum_paths with ceil(total/128) groups.
struct CsrParameters {
    uint64_t offsets;        // uint32[width*height+1]
    uint64_t disparities;    // uint32[total], sorted, unique per pixel
    uint64_t costs;          // uint32[total], exact integer 0..10000
    uint64_t paths;          // uint32[4*total]
    uint64_t sums;           // uint32[total]
    uint width, height, disparity_range, total, p1, p2;
};
struct CsrPush {uint64_t parameters; uint pass;};
[[vk::push_constant]] CsrPush pc;
#define p (*(CsrParameters*)pc.parameters)

static const uint CSR_INF=0x3fffffff;
groupshared uint path_minimum[64];

uint find_previous(uint* hypotheses,uint* states,uint first,uint last,uint target){
    uint begin=first,end=last;
    while(begin<end){
        uint mid=begin+(end-begin)/2;
        uint d=hypotheses[mid];
        if(d<target)begin=mid+1;
        else end=mid;
    }
    return begin<last&&hypotheses[begin]==target?states[begin]:CSR_INF;
}
[shader("compute")][numthreads(64,1,1)]
void scan_path(uint3 gid:SV_GroupID,uint3 tid:SV_GroupThreadID){
    uint pass=pc.pass;
    if(pass>=4||p.width==0||p.height==0)return;
    uint line_count=pass<2?p.height:p.width;
    if(gid.x>=line_count)return; // Uniform early return across entire group.
    int dx=pass==0?1:pass==1?-1:0;
    int dy=pass==2?1:pass==3?-1:0;
    int x=pass==0?0:pass==1?int(p.width)-1:int(gid.x);
    int y=pass==2?0:pass==3?int(p.height)-1:int(gid.x);
    uint lane=tid.x;
    uint* offsets=(uint*)p.offsets;
    uint* hypotheses=(uint*)p.disparities;
    uint* costs=(uint*)p.costs;
    uint* states=(uint*)p.paths+pass*p.total;
    bool previous_exists=false;
    uint previous_pixel=0;
    while(x>=0&&y>=0&&x<int(p.width)&&y<int(p.height)){
        uint pixel=uint(y)*p.width+uint(x);
        uint begin=offsets[pixel],end=offsets[pixel+1];
        uint prev_begin=previous_exists?offsets[previous_pixel]:0;
        uint prev_end=previous_exists?offsets[previous_pixel+1]:0;
        uint local_min=CSR_INF;
        for(uint k=prev_begin+lane;k<prev_end;k+=64)local_min=min(local_min,states[k]);
        path_minimum[lane]=local_min;
        GroupMemoryBarrierWithGroupSync();
        for(uint step=32;step>0;step>>=1){
            if(lane<step)path_minimum[lane]=min(path_minimum[lane],path_minimum[lane+step]);
            GroupMemoryBarrierWithGroupSync();
        }
        uint minimum=path_minimum[0];
        for(uint k=begin+lane;k<end;k+=64){
            uint d=hypotheses[k];
            uint c=costs[k];
            uint value=c;
            if(minimum!=CSR_INF){
                uint own=find_previous(hypotheses,states,prev_begin,prev_end,d);
                uint lo=d>0?find_previous(hypotheses,states,prev_begin,prev_end,d-1)+p.p1:CSR_INF;
                uint hi=d+1<p.disparity_range?find_previous(hypotheses,states,prev_begin,prev_end,d+1)+p.p1:CSR_INF;
                value=c+min(min(own,lo),min(hi,minimum+p.p2))-minimum;
            }
            states[k]=value;
        }
        // Each later pixel depends on preceding writes from other lanes in this group.
        DeviceMemoryBarrierWithGroupSync();
        previous_exists=true;
        previous_pixel=pixel;
        x+=dx;y+=dy;
    }
}
[shader("compute")][numthreads(128,1,1)]
void sum_paths(uint3 gid:SV_GroupID,uint3 lane:SV_GroupThreadID){
    uint index=gid.x*128+lane.x;
    if(index>=p.total)return;
    uint* states=(uint*)p.paths;
    ((uint*)p.sums)[index]=states[index]+states[p.total+index]+
        states[2*p.total+index]+states[3*p.total+index];
}
