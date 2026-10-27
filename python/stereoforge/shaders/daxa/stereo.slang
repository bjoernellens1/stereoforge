#include "../ops/sgm.slang"
struct Parameters {
    uint64_t left_address,right_address,left_desc_address,right_desc_address,left_valid_address,right_valid_address;
    uint64_t costs_address,sums_address,disparity_address,confidence_address,validity_address,depth_address,depth_valid_address,counters_address;
    uint width,height,disparities,paths,p1,p2,subpixel,metric_enabled;
    float fx,baseline_m,cx_offset;
    uint packing_bits;
    uint64_t penalties_address;
};
struct Push {uint64_t parameters;uint pass;uint groups_x;};
[[vk::push_constant]] ConstantBuffer<Push> pc;
uint linear_group(uint3 gid){return gid.y*pc.groups_x+gid.x;}
#define p (*(Parameters*)pc.parameters)
#define costs ((uint*)p.costs_address)
#define sums ((uint*)p.sums_address)
#define disparity ((float*)p.disparity_address)
#define confidence ((float*)p.confidence_address)
#define validity ((uint*)p.validity_address)
#define depth ((float*)p.depth_address)
#define depth_valid ((uint*)p.depth_valid_address)
[shader("compute")][numthreads(64,1,1)]
void census(uint3 gid:SV_GroupID,uint3 local:SV_GroupThreadID){
    uint i=linear_group(gid)*64+local.x;if(i>=p.width*p.height)return;
    uint* image=(uint*)(pc.pass==0?p.left_address:p.right_address);
    uint* descriptor=(uint*)(pc.pass==0?p.left_desc_address:p.right_desc_address);
    uint* valid=(uint*)(pc.pass==0?p.left_valid_address:p.right_valid_address);
    bool supported=sgm_census_supported(p.width,p.height,i);uint value=0;if(supported)value=sgm_census(image,p.width,p.height,i);descriptor[i]=value;valid[i]=supported?1:0;
}
[shader("compute")][numthreads(64,1,1)]
void cost(uint3 gid:SV_GroupID,uint3 local:SV_GroupThreadID){
    uint index=linear_group(gid)*64+local.x;if(index>=p.width*p.height*p.disparities)return;
    uint i=index/p.disparities,d=index%p.disparities,x=i%p.width;
    uint* left_desc=(uint*)p.left_desc_address;uint* right_desc=(uint*)p.right_desc_address;
    uint* left_valid=(uint*)p.left_valid_address;uint* right_valid=(uint*)p.right_valid_address;
    bool ok=x>=d&&left_valid[i]!=0;if(ok)ok=right_valid[i-d]!=0;
    uint value=SGM_INVALID_COST;if(ok)value=sgm_cost(left_desc[i],right_desc[i-d],true);
    costs[index]=value;sums[index]=0;
}
groupshared uint previous[256],current[256],reduction[256];
// Frozen algorithm baseline: a workgroup per line and 11 barriers/pixel.
[shader("compute")][numthreads(256,1,1)]
void aggregate_baseline(uint3 gid:SV_GroupID,uint3 lane:SV_GroupThreadID){
    int2 dir=sgm_direction(pc.pass);uint line=linear_group(gid),d=lane.x;
    if(line>=sgm_line_count(p.width,p.height,dir))return;
    int2 xy=sgm_line_start(p.width,p.height,dir,line);previous[d]=0;GroupMemoryBarrierWithGroupSync();bool first=true;
    while(sgm_inside(xy,p.width,p.height)){
        reduction[d]=d<p.disparities?previous[d]:SGM_REDUCTION_INF;GroupMemoryBarrierWithGroupSync();
        for(uint stride=128;stride>0;stride>>=1){if(d<stride)reduction[d]=min(reduction[d],reduction[d+stride]);GroupMemoryBarrierWithGroupSync();}
        uint value=0;if(d<p.disparities){uint index=(uint(xy.y)*p.width+uint(xy.x))*p.disparities+d;
            value=sgm_step(costs[index],previous[d],d>0?previous[d>0?d-1:0]:SGM_REDUCTION_INF,d+1<p.disparities?previous[min(d+1,255u)]:SGM_REDUCTION_INF,reduction[0],p.p1,p.p2,first);sums[index]+=value;}
        current[d]=value;GroupMemoryBarrierWithGroupSync();previous[d]=current[d];GroupMemoryBarrierWithGroupSync();first=false;xy+=dir;
    }
}
// Candidate A: one actual wave per line, striped register chunks. Group64 is
// either one wave64 or two wave32 lines. Dispatch line_count groups deliberately
// overcovers wave32 and a uniform per-wave line guard masks excess work.
[shader("compute")][numthreads(64,1,1)]
void aggregate_wave(uint3 gid:SV_GroupID){
    uint lanes=WaveGetLaneCount(),lane=WaveGetLaneIndex();
    int2 dir=sgm_direction(pc.pass);uint line=linear_group(gid)*(64/lanes)+WaveGetWaveIndex();
    if(line>=sgm_line_count(p.width,p.height,dir))return; // Uniform within a wave.
    uint chunks=(p.disparities+lanes-1)/lanes;uint prev[8],next[8];
    [unroll]for(uint k=0;k<8;++k){prev[k]=0;next[k]=0;}
    int2 xy=sgm_line_start(p.width,p.height,dir,line);bool first=true;
    while(sgm_inside(xy,p.width,p.height)){
        uint local_min=SGM_REDUCTION_INF;
        [unroll]for(uint k=0;k<8;++k)if(k<chunks&&k*lanes+lane<p.disparities)local_min=min(local_min,prev[k]);
        uint minimum=WaveActiveMin(local_min);
        [unroll]for(uint k=0;k<8;++k)if(k<chunks){
            // Every participating lane executes each shuffle, including masked tails.
            uint left_same=WaveReadLaneAt(prev[k],(lane+lanes-1)%lanes);
            uint right_same=WaveReadLaneAt(prev[k],(lane+1)%lanes);
            uint left_cross=WaveReadLaneAt(k>0?prev[k>0?k-1:0]:SGM_REDUCTION_INF,lanes-1);
            uint right_cross=WaveReadLaneAt(k+1<chunks?prev[min(k+1,7u)]:SGM_REDUCTION_INF,0);
            uint d=k*lanes+lane;
            uint lo=d>0?(lane==0?left_cross:left_same):SGM_REDUCTION_INF;
            uint hi=d+1<p.disparities?(lane+1==lanes?right_cross:right_same):SGM_REDUCTION_INF;
            if(d<p.disparities){uint index=(uint(xy.y)*p.width+uint(xy.x))*p.disparities+d;
                next[k]=sgm_step(costs[index],prev[k],lo,hi,minimum,p.p1,p.p2,first);sums[index]+=next[k];}
        }
        [unroll]for(uint k=0;k<8;++k)prev[k]=next[k];first=false;xy+=dir;
    }
}
groupshared uint select_best_cost[256],select_second_cost[256],select_best_index[256];
[shader("compute")][numthreads(256,1,1)]
void select_output(uint3 gid:SV_GroupID,uint3 tid:SV_GroupThreadID){
    uint i=linear_group(gid)*64+tid.x/4,lane=tid.x%4,t=tid.x;
    uint offset=i*p.disparities,best=0xffffffff,best_cost=0xffffffff,second=0xffffffff;
    if(i<p.width*p.height)for(uint d=lane;d<p.disparities;d+=4){uint c=sums[offset+d];if(c<best_cost||(c==best_cost&&d<best)){second=best_cost;best_cost=c;best=d;}else second=min(second,c);}
    select_best_cost[t]=best_cost;select_second_cost[t]=second;select_best_index[t]=best;GroupMemoryBarrierWithGroupSync();
    for(uint stride=2;stride>0;stride/=2){if(lane<stride){uint other=t+stride;uint oc=select_best_cost[other],oi=select_best_index[other],own=select_best_cost[t],index=select_best_index[t];
        if(oc<own||(oc==own&&oi<index)){select_second_cost[t]=min(select_second_cost[other],own);select_best_cost[t]=oc;select_best_index[t]=oi;}else select_second_cost[t]=min(select_second_cost[t],oc);}
        GroupMemoryBarrierWithGroupSync();}
    if(lane!=0||i>=p.width*p.height)return;
    best=select_best_index[t];best_cost=select_best_cost[t];second=select_second_cost[t];if(p.disparities==1)second=best_cost;
    bool valid=sgm_selection_valid(best,best_cost,second);float value=float(best);
    if(p.subpixel!=0&&best>0&&best+1<p.disparities)value=sgm_subpixel(best,best_cost,sums[offset+best-1],sums[offset+best+1]);
    disparity[i]=valid?value:asfloat(0x7fc00000);confidence[i]=valid?float(second-best_cost)/float(max(second,1u)):0;validity[i]=valid?1:0;
}
[shader("compute")][numthreads(64,1,1)]
void depth_convert(uint3 gid:SV_GroupID,uint3 local:SV_GroupThreadID){uint i=linear_group(gid)*64+local.x;if(i>=p.width*p.height)return;float corrected=disparity[i]-p.cx_offset,z=p.fx*p.baseline_m/corrected;bool valid=validity[i]!=0&&isfinite(corrected)&&corrected>0&&isfinite(z)&&z>0;depth[i]=valid?z:asfloat(0x7fc00000);depth_valid[i]=valid?1:0;}
[shader("compute")][numthreads(64,1,1)]
void inspect(uint3 gid:SV_GroupID,uint3 local:SV_GroupThreadID){uint i=linear_group(gid)*64+local.x;if(i>=p.width*p.height)return;uint* counters=(uint*)p.counters_address;InterlockedAdd(counters[0],validity[i]);InterlockedXor(counters[1],asuint(disparity[i]));InterlockedXor(counters[2],asuint(confidence[i]));if(p.metric_enabled!=0)InterlockedAdd(counters[3],depth_valid[i]);}

// Candidate B has an explicitly different layout: each word owns 4x8-bit or
// 2x16-bit normalized penalties for adjacent disparities within ONE pixel.
// No dense cost/sum buffer exists. Paths are graph-ordered; a lane owns a whole
// word and the validated bound paths*P2 prevents carries between packed fields.
#define penalties ((uint*)p.penalties_address)
uint packed_components(){return 32/p.packing_bits;}
uint packed_words_per_pixel(){uint components=packed_components();return (p.disparities+components-1)/components;}
uint packed_cost(uint i,uint d){
    uint* lv=(uint*)p.left_valid_address;uint* rv=(uint*)p.right_valid_address;
    bool supported=i%p.width>=d&&lv[i]!=0;if(supported)supported=rv[i-d]!=0;
    uint cost=SGM_INVALID_COST;if(supported){uint* ld=(uint*)p.left_desc_address;uint* rd=(uint*)p.right_desc_address;cost=sgm_cost(ld[i],rd[i-d],true);}return cost;
}
uint packed_sum(uint i,uint d){uint components=packed_components();uint field=(penalties[i*packed_words_per_pixel()+d/components]>>((d%components)*p.packing_bits))&((1u<<p.packing_bits)-1);return p.paths*packed_cost(i,d)+field;}
[shader("compute")][numthreads(64,1,1)]
void clear_packed(uint3 gid:SV_GroupID,uint3 local:SV_GroupThreadID){uint i=linear_group(gid)*64+local.x;if(i<p.width*p.height*packed_words_per_pixel())penalties[i]=0;}
void aggregate_packed_body<let Components:uint,let Bits:uint>(uint3 gid){
    uint lanes=WaveGetLaneCount(),lane=WaveGetLaneIndex();int2 dir=sgm_direction(pc.pass);
    uint line=linear_group(gid)*(64/lanes)+WaveGetWaveIndex();if(line>=sgm_line_count(p.width,p.height,dir))return;
    uint row_words=(p.disparities+Components-1)/Components,chunks=(row_words+lanes-1)/lanes;
    uint prev[8],next[8];[unroll]for(uint r=0;r<8;++r){prev[r]=0;next[r]=0;}
    int2 xy=sgm_line_start(p.width,p.height,dir,line);bool first=true;
    while(sgm_inside(xy,p.width,p.height)){
        uint local_min=SGM_REDUCTION_INF;
        [unroll]for(uint k=0;k<8/Components;++k)if(k<chunks){[unroll]for(uint component=0;component<Components;++component)if((k*lanes+lane)*Components+component<p.disparities)local_min=min(local_min,prev[k*Components+component]);}
        uint minimum=WaveActiveMin(local_min),i=uint(xy.y)*p.width+uint(xy.x);
        [unroll]for(uint k=0;k<8/Components;++k)if(k<chunks){
            // Full-wave uniform shuffles occur even for lanes whose word/tail is masked.
            uint lo_same=WaveReadLaneAt(prev[k*Components+Components-1],(lane+lanes-1)%lanes);
            uint hi_same=WaveReadLaneAt(prev[k*Components],(lane+1)%lanes);
            uint lo_cross=WaveReadLaneAt(k>0?prev[(k>0?k-1:0)*Components+Components-1]:SGM_REDUCTION_INF,lanes-1);
            uint hi_cross=WaveReadLaneAt(k+1<chunks?prev[(k+1<8/Components?k+1:k)*Components]:SGM_REDUCTION_INF,0);
            uint word=k*lanes+lane,addition=0;
            [unroll]for(uint component=0;component<Components;++component){uint d=word*Components+component,r=k*Components+component;
                uint lo=component>0?prev[r>0?r-1:0]:(lane>0?lo_same:lo_cross);
                uint hi=component+1<Components?prev[min(r+1,7u)]:(lane+1<lanes?hi_same:hi_cross);
                if(d==0)lo=SGM_REDUCTION_INF;if(d+1>=p.disparities)hi=SGM_REDUCTION_INF;
                if(d<p.disparities){uint c=packed_cost(i,d);next[r]=sgm_step(c,prev[r],lo,hi,minimum,p.p1,p.p2,first);addition|=(next[r]-c)<<(component*Bits);}else next[r]=0;
            }
            if(word<row_words)penalties[i*row_words+word]+=addition;
        }
        [unroll]for(uint r=0;r<8;++r)prev[r]=next[r];first=false;xy+=dir;
    }
}
[shader("compute")][numthreads(64,1,1)]
void aggregate_packed8(uint3 gid:SV_GroupID){aggregate_packed_body<4,8>(gid);}
[shader("compute")][numthreads(64,1,1)]
void aggregate_packed16(uint3 gid:SV_GroupID){aggregate_packed_body<2,16>(gid);}

[shader("compute")][numthreads(256,1,1)]
void select_packed(uint3 gid:SV_GroupID,uint3 tid:SV_GroupThreadID){
    uint i=linear_group(gid)*64+tid.x/4,lane=tid.x%4,t=tid.x;
    uint offset=i*p.disparities,best=0xffffffff,best_cost=0xffffffff,second=0xffffffff;
    if(i<p.width*p.height)for(uint d=lane;d<p.disparities;d+=4){uint c=packed_sum(i,d);if(c<best_cost||(c==best_cost&&d<best)){second=best_cost;best_cost=c;best=d;}else second=min(second,c);}
    select_best_cost[t]=best_cost;select_second_cost[t]=second;select_best_index[t]=best;GroupMemoryBarrierWithGroupSync();
    for(uint stride=2;stride>0;stride/=2){if(lane<stride){uint other=t+stride;uint oc=select_best_cost[other],oi=select_best_index[other],own=select_best_cost[t],index=select_best_index[t];
        if(oc<own||(oc==own&&oi<index)){select_second_cost[t]=min(select_second_cost[other],own);select_best_cost[t]=oc;select_best_index[t]=oi;}else select_second_cost[t]=min(select_second_cost[t],oc);}
        GroupMemoryBarrierWithGroupSync();}
    if(lane!=0||i>=p.width*p.height)return;
    best=select_best_index[t];best_cost=select_best_cost[t];second=select_second_cost[t];if(p.disparities==1)second=best_cost;
    bool valid=sgm_selection_valid(best,best_cost,second);float value=float(best);
    if(p.subpixel!=0&&best>0&&best+1<p.disparities)value=sgm_subpixel(best,best_cost,packed_sum(i,best-1),packed_sum(i,best+1));
    disparity[i]=valid?value:asfloat(0x7fc00000);confidence[i]=valid?float(second-best_cost)/float(max(second,1u)):0;validity[i]=valid?1:0;
}

// Candidate C changes only address arithmetic and invariant loads. B entries
// above remain the causal reference; widths/layouts are not runtime divisors
// inside C's disparity loops.
uint specialized_cost(uint x,uint i,uint d,uint reference,bool reference_valid,uint* right_valid,uint* right_desc){
    bool supported=x>=d&&reference_valid;if(supported)supported=right_valid[i-d]!=0;
    uint cost=SGM_INVALID_COST;if(supported)cost=sgm_cost(reference,right_desc[i-d],true);return cost;
}
uint specialized_sum<let Components:uint,let Bits:uint>(uint x,uint i,uint d,uint word_base,uint paths,uint reference,bool reference_valid,uint* right_valid,uint* right_desc,uint* words){
    uint field=(words[word_base+d/Components]>>((d%Components)*Bits))&((1u<<Bits)-1);
    return paths*specialized_cost(x,i,d,reference,reference_valid,right_valid,right_desc)+field;
}
void aggregate_specialized_body<let Components:uint,let Bits:uint>(uint3 gid){
    uint width=p.width,height=p.height,disparities=p.disparities,p1=p.p1,p2=p.p2;
    uint* left_desc=(uint*)p.left_desc_address;uint* right_desc=(uint*)p.right_desc_address;
    uint* left_valid=(uint*)p.left_valid_address;uint* right_valid=(uint*)p.right_valid_address;
    uint* words=(uint*)p.penalties_address;
    uint lanes=WaveGetLaneCount(),lane=WaveGetLaneIndex();int2 dir=sgm_direction(pc.pass);
    uint line=linear_group(gid)*(64/lanes)+WaveGetWaveIndex();if(line>=sgm_line_count(width,height,dir))return;
    uint row_words=(disparities+Components-1)/Components,chunks=(row_words+lanes-1)/lanes;
    uint prev[8],next[8];[unroll]for(uint r=0;r<8;++r){prev[r]=0;next[r]=0;}
    int2 xy=sgm_line_start(width,height,dir,line);bool first=true;
    while(sgm_inside(xy,width,height)){
        uint local_min=SGM_REDUCTION_INF;
        [unroll]for(uint k=0;k<8/Components;++k)if(k<chunks){[unroll]for(uint component=0;component<Components;++component)if((k*lanes+lane)*Components+component<disparities)local_min=min(local_min,prev[k*Components+component]);}
        uint minimum=WaveActiveMin(local_min),x=uint(xy.x),i=uint(xy.y)*width+x;
        uint reference=left_desc[i];bool reference_valid=left_valid[i]!=0;
        [unroll]for(uint k=0;k<8/Components;++k)if(k<chunks){
            // Full-wave uniform shuffles occur even for lanes whose word/tail is masked.
            uint lo_same=WaveReadLaneAt(prev[k*Components+Components-1],(lane+lanes-1)%lanes);
            uint hi_same=WaveReadLaneAt(prev[k*Components],(lane+1)%lanes);
            uint lo_cross=WaveReadLaneAt(k>0?prev[(k>0?k-1:0)*Components+Components-1]:SGM_REDUCTION_INF,lanes-1);
            uint hi_cross=WaveReadLaneAt(k+1<chunks?prev[(k+1<8/Components?k+1:k)*Components]:SGM_REDUCTION_INF,0);
            uint word=k*lanes+lane,addition=0;
            [unroll]for(uint component=0;component<Components;++component){uint d=word*Components+component,r=k*Components+component;
                uint lo=component>0?prev[r>0?r-1:0]:(lane>0?lo_same:lo_cross);
                uint hi=component+1<Components?prev[min(r+1,7u)]:(lane+1<lanes?hi_same:hi_cross);
                if(d==0)lo=SGM_REDUCTION_INF;if(d+1>=disparities)hi=SGM_REDUCTION_INF;
                if(d<disparities){uint c=specialized_cost(x,i,d,reference,reference_valid,right_valid,right_desc);next[r]=sgm_step(c,prev[r],lo,hi,minimum,p1,p2,first);addition|=(next[r]-c)<<(component*Bits);}else next[r]=0;
            }
            if(word<row_words)words[i*row_words+word]+=addition;
        }
        [unroll]for(uint r=0;r<8;++r)prev[r]=next[r];first=false;xy+=dir;
    }
}
[shader("compute")][numthreads(64,1,1)]
void aggregate_specialized8(uint3 gid:SV_GroupID){aggregate_specialized_body<4,8>(gid);}
[shader("compute")][numthreads(64,1,1)]
void aggregate_specialized16(uint3 gid:SV_GroupID){aggregate_specialized_body<2,16>(gid);}
void select_specialized_body<let Components:uint,let Bits:uint>(uint3 gid,uint3 tid){
    uint i=linear_group(gid)*64+tid.x/4,lane=tid.x%4,t=tid.x;
    uint width=p.width,disparities=p.disparities,paths=p.paths,subpixel=p.subpixel;
    uint x=i%width,word_base=i*((disparities+Components-1)/Components);
    uint* left_desc=(uint*)p.left_desc_address;uint* right_desc=(uint*)p.right_desc_address;
    uint* left_valid=(uint*)p.left_valid_address;uint* right_valid=(uint*)p.right_valid_address;
    uint* words=(uint*)p.penalties_address;
    uint reference=0;bool reference_valid=false;
    if(i<width*p.height){reference=left_desc[i];reference_valid=left_valid[i]!=0;}
    uint best=0xffffffff,best_cost=0xffffffff,second=0xffffffff;
    if(i<width*p.height)for(uint d=lane;d<disparities;d+=4){uint c=specialized_sum<Components,Bits>(x,i,d,word_base,paths,reference,reference_valid,right_valid,right_desc,words);if(c<best_cost||(c==best_cost&&d<best)){second=best_cost;best_cost=c;best=d;}else second=min(second,c);}
    select_best_cost[t]=best_cost;select_second_cost[t]=second;select_best_index[t]=best;GroupMemoryBarrierWithGroupSync();
    for(uint stride=2;stride>0;stride/=2){if(lane<stride){uint other=t+stride;uint oc=select_best_cost[other],oi=select_best_index[other],own=select_best_cost[t],index=select_best_index[t];
        if(oc<own||(oc==own&&oi<index)){select_second_cost[t]=min(select_second_cost[other],own);select_best_cost[t]=oc;select_best_index[t]=oi;}else select_second_cost[t]=min(select_second_cost[t],oc);}
        GroupMemoryBarrierWithGroupSync();}
    if(lane!=0||i>=width*p.height)return;
    best=select_best_index[t];best_cost=select_best_cost[t];second=select_second_cost[t];if(disparities==1)second=best_cost;
    bool valid=sgm_selection_valid(best,best_cost,second);float value=float(best);
    if(subpixel!=0&&best>0&&best+1<disparities)value=sgm_subpixel(best,best_cost,specialized_sum<Components,Bits>(x,i,best-1,word_base,paths,reference,reference_valid,right_valid,right_desc,words),specialized_sum<Components,Bits>(x,i,best+1,word_base,paths,reference,reference_valid,right_valid,right_desc,words));
    disparity[i]=valid?value:asfloat(0x7fc00000);confidence[i]=valid?float(second-best_cost)/float(max(second,1u)):0;validity[i]=valid?1:0;
}

[shader("compute")][numthreads(256,1,1)]
void select_specialized8(uint3 gid:SV_GroupID,uint3 tid:SV_GroupThreadID){select_specialized_body<4,8>(gid,tid);}
[shader("compute")][numthreads(256,1,1)]
void select_specialized16(uint3 gid:SV_GroupID,uint3 tid:SV_GroupThreadID){select_specialized_body<2,16>(gid,tid);}

// Candidate D specializes only aggregation state/stride for exactly D128.
void aggregate_d128_body<let Components:uint,let Bits:uint>(uint3 gid){
    uint width=p.width,height=p.height,disparities=128,p1=p.p1,p2=p.p2;
    uint* left_desc=(uint*)p.left_desc_address;uint* right_desc=(uint*)p.right_desc_address;
    uint* left_valid=(uint*)p.left_valid_address;uint* right_valid=(uint*)p.right_valid_address;
    uint* words=(uint*)p.penalties_address;
    uint lanes=WaveGetLaneCount(),lane=WaveGetLaneIndex();int2 dir=sgm_direction(pc.pass);
    uint line=linear_group(gid)*(64/lanes)+WaveGetWaveIndex();if(line>=sgm_line_count(width,height,dir))return;
    uint row_words=(disparities+Components-1)/Components,chunks=(row_words+lanes-1)/lanes;
    uint prev[4],next[4];[unroll]for(uint r=0;r<4;++r){prev[r]=0;next[r]=0;}
    int2 xy=sgm_line_start(width,height,dir,line);bool first=true;
    while(sgm_inside(xy,width,height)){
        uint local_min=SGM_REDUCTION_INF;
        [unroll]for(uint k=0;k<4/Components;++k)if(k<chunks){[unroll]for(uint component=0;component<Components;++component)if((k*lanes+lane)*Components+component<disparities)local_min=min(local_min,prev[k*Components+component]);}
        uint minimum=WaveActiveMin(local_min),x=uint(xy.x),i=uint(xy.y)*width+x;
        uint reference=left_desc[i];bool reference_valid=left_valid[i]!=0;
        [unroll]for(uint k=0;k<4/Components;++k)if(k<chunks){
            // Full-wave uniform shuffles occur even for lanes whose word/tail is masked.
            uint lo_same=WaveReadLaneAt(prev[k*Components+Components-1],(lane+lanes-1)%lanes);
            uint hi_same=WaveReadLaneAt(prev[k*Components],(lane+1)%lanes);
            uint lo_cross=WaveReadLaneAt(k>0?prev[(k>0?k-1:0)*Components+Components-1]:SGM_REDUCTION_INF,lanes-1);
            uint hi_cross=WaveReadLaneAt(k+1<chunks?prev[(k+1<4/Components?k+1:k)*Components]:SGM_REDUCTION_INF,0);
            uint word=k*lanes+lane,addition=0;
            [unroll]for(uint component=0;component<Components;++component){uint d=word*Components+component,r=k*Components+component;
                uint lo=component>0?prev[r>0?r-1:0]:(lane>0?lo_same:lo_cross);
                uint hi=component+1<Components?prev[min(r+1,3u)]:(lane+1<lanes?hi_same:hi_cross);
                if(d==0)lo=SGM_REDUCTION_INF;if(d+1>=disparities)hi=SGM_REDUCTION_INF;
                if(d<disparities){uint c=specialized_cost(x,i,d,reference,reference_valid,right_valid,right_desc);next[r]=sgm_step(c,prev[r],lo,hi,minimum,p1,p2,first);addition|=(next[r]-c)<<(component*Bits);}else next[r]=0;
            }
            if(word<row_words)words[i*row_words+word]+=addition;
        }
        [unroll]for(uint r=0;r<4;++r)prev[r]=next[r];first=false;xy+=dir;
    }
}
[shader("compute")][numthreads(64,1,1)]
void aggregate_d1288(uint3 gid:SV_GroupID){aggregate_d128_body<4,8>(gid);}
[shader("compute")][numthreads(64,1,1)]
void aggregate_d12816(uint3 gid:SV_GroupID){aggregate_d128_body<2,16>(gid);}

// Candidate E changes only the selector merge/dispatch; Caggregate is reused.
void select_quad_body<let Components:uint,let Bits:uint>(uint3 gid){
    uint wave_size=WaveGetLaneCount(),wave_lane=WaveGetLaneIndex(),lane=wave_lane%4;
    uint i=linear_group(gid)*16+WaveGetWaveIndex()*(wave_size/4)+wave_lane/4;
    uint width=p.width,disparities=p.disparities,paths=p.paths,subpixel=p.subpixel;
    uint x=i%width,word_base=i*((disparities+Components-1)/Components);
    uint* left_desc=(uint*)p.left_desc_address;uint* right_desc=(uint*)p.right_desc_address;
    uint* left_valid=(uint*)p.left_valid_address;uint* right_valid=(uint*)p.right_valid_address;
    uint* words=(uint*)p.penalties_address;
    uint reference=0;bool reference_valid=false;
    if(i<width*p.height){reference=left_desc[i];reference_valid=left_valid[i]!=0;}
    uint best=0xffffffff,best_cost=0xffffffff,second=0xffffffff;
    if(i<width*p.height)for(uint d=lane;d<disparities;d+=4){uint c=specialized_sum<Components,Bits>(x,i,d,word_base,paths,reference,reference_valid,right_valid,right_desc,words);if(c<best_cost||(c==best_cost&&d<best)){second=best_cost;best_cost=c;best=d;}else second=min(second,c);}
    // Every full-wave lane executes every shuffle, including tail pixels.
    // XOR2 then XOR1 stays inside the same4-lane pixel for all active sources.
    [unroll]for(uint stride=2;stride>0;stride/=2){
        uint source=wave_lane^stride;
        uint other_cost=WaveReadLaneAt(best_cost,source);
        uint other_index=WaveReadLaneAt(best,source);
        uint other_second=WaveReadLaneAt(second,source);
        if(lane<stride){if(other_cost<best_cost||(other_cost==best_cost&&other_index<best)){second=min(other_second,best_cost);best_cost=other_cost;best=other_index;}else second=min(second,other_cost);}
    }
    if(lane!=0||i>=width*p.height)return;
    if(disparities==1)second=best_cost;
    bool valid=sgm_selection_valid(best,best_cost,second);float value=float(best);
    if(subpixel!=0&&best>0&&best+1<disparities)value=sgm_subpixel(best,best_cost,specialized_sum<Components,Bits>(x,i,best-1,word_base,paths,reference,reference_valid,right_valid,right_desc,words),specialized_sum<Components,Bits>(x,i,best+1,word_base,paths,reference,reference_valid,right_valid,right_desc,words));
    disparity[i]=valid?value:asfloat(0x7fc00000);confidence[i]=valid?float(second-best_cost)/float(max(second,1u)):0;validity[i]=valid?1:0;
}

[shader("compute")][numthreads(64,1,1)]
void select_quad8(uint3 gid:SV_GroupID){select_quad_body<4,8>(gid);}
[shader("compute")][numthreads(64,1,1)]
void select_quad16(uint3 gid:SV_GroupID){select_quad_body<2,16>(gid);}

// Candidate F replaces only the full-wave integer minimum primitive.
void aggregate_shuffle_body<let Components:uint,let Bits:uint>(uint3 gid){
    uint width=p.width,height=p.height,disparities=p.disparities,p1=p.p1,p2=p.p2;
    uint* left_desc=(uint*)p.left_desc_address;uint* right_desc=(uint*)p.right_desc_address;
    uint* left_valid=(uint*)p.left_valid_address;uint* right_valid=(uint*)p.right_valid_address;
    uint* words=(uint*)p.penalties_address;
    uint lanes=WaveGetLaneCount(),lane=WaveGetLaneIndex();int2 dir=sgm_direction(pc.pass);
    uint line=linear_group(gid)*(64/lanes)+WaveGetWaveIndex();if(line>=sgm_line_count(width,height,dir))return;
    uint row_words=(disparities+Components-1)/Components,chunks=(row_words+lanes-1)/lanes;
    uint prev[8],next[8];[unroll]for(uint r=0;r<8;++r){prev[r]=0;next[r]=0;}
    int2 xy=sgm_line_start(width,height,dir,line);bool first=true;
    while(sgm_inside(xy,width,height)){
        uint local_min=SGM_REDUCTION_INF;
        [unroll]for(uint k=0;k<8/Components;++k)if(k<chunks){[unroll]for(uint component=0;component<Components;++component)if((k*lanes+lane)*Components+component<disparities)local_min=min(local_min,prev[k*Components+component]);}
        uint minimum=local_min;
        [unroll]for(uint offset=1;offset<32;offset*=2){uint other=WaveReadLaneAt(minimum,lane^offset);minimum=min(minimum,other);}
        if(lanes==64){uint other=WaveReadLaneAt(minimum,lane^32);minimum=min(minimum,other);}
        uint x=uint(xy.x),i=uint(xy.y)*width+x;
        uint reference=left_desc[i];bool reference_valid=left_valid[i]!=0;
        [unroll]for(uint k=0;k<8/Components;++k)if(k<chunks){
            // Full-wave uniform shuffles occur even for lanes whose word/tail is masked.
            uint lo_same=WaveReadLaneAt(prev[k*Components+Components-1],(lane+lanes-1)%lanes);
            uint hi_same=WaveReadLaneAt(prev[k*Components],(lane+1)%lanes);
            uint lo_cross=WaveReadLaneAt(k>0?prev[(k>0?k-1:0)*Components+Components-1]:SGM_REDUCTION_INF,lanes-1);
            uint hi_cross=WaveReadLaneAt(k+1<chunks?prev[(k+1<8/Components?k+1:k)*Components]:SGM_REDUCTION_INF,0);
            uint word=k*lanes+lane,addition=0;
            [unroll]for(uint component=0;component<Components;++component){uint d=word*Components+component,r=k*Components+component;
                uint lo=component>0?prev[r>0?r-1:0]:(lane>0?lo_same:lo_cross);
                uint hi=component+1<Components?prev[min(r+1,7u)]:(lane+1<lanes?hi_same:hi_cross);
                if(d==0)lo=SGM_REDUCTION_INF;if(d+1>=disparities)hi=SGM_REDUCTION_INF;
                if(d<disparities){uint c=specialized_cost(x,i,d,reference,reference_valid,right_valid,right_desc);next[r]=sgm_step(c,prev[r],lo,hi,minimum,p1,p2,first);addition|=(next[r]-c)<<(component*Bits);}else next[r]=0;
            }
            if(word<row_words)words[i*row_words+word]+=addition;
        }
        [unroll]for(uint r=0;r<8;++r)prev[r]=next[r];first=false;xy+=dir;
    }
}
[shader("compute")][numthreads(64,1,1)]
void aggregate_shuffle8(uint3 gid:SV_GroupID){aggregate_shuffle_body<4,8>(gid);}
[shader("compute")][numthreads(64,1,1)]
void aggregate_shuffle16(uint3 gid:SV_GroupID){aggregate_shuffle_body<2,16>(gid);}
