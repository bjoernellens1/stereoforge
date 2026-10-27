// One shared C aggregate/selection algorithm for legacy and eye-layered wrappers.
// Inputs/output pointers are explicit; no push constant or reference global is read here.
#ifndef SF_SGM_SPECIALIZED
#define SF_SGM_SPECIALIZED
#include "sgm.slang"
#include "../daxa/stereo_parameters.slang"
groupshared uint select_best_cost[256],select_second_cost[256],select_best_index[256];
uint specialized_cost(uint x,uint i,uint d,uint reference,bool reference_valid,uint* right_valid,uint* right_desc){
    bool supported=x>=d&&reference_valid;if(supported)supported=right_valid[i-d]!=0;
    uint cost=SGM_INVALID_COST;if(supported)cost=sgm_cost(reference,right_desc[i-d],true);return cost;
}
uint specialized_sum<let Components:uint,let Bits:uint>(uint x,uint i,uint d,uint word_base,uint paths,uint reference,bool reference_valid,uint* right_valid,uint* right_desc,uint* words){
    uint field=(words[word_base+d/Components]>>((d%Components)*Bits))&((1u<<Bits)-1);
    return paths*specialized_cost(x,i,d,reference,reference_valid,right_valid,right_desc)+field;
}
// Compile-time geometry leaves the recurrence and packed/selection math shared.
uint reference_cost<let RightReference:bool>(uint width,uint x,uint i,uint d,uint reference,bool reference_valid,uint* target_valid,uint* target_desc){
    if(!RightReference)return specialized_cost(x,i,d,reference,reference_valid,target_valid,target_desc);
    bool supported=d<width-x&&reference_valid;if(supported)supported=target_valid[i+d]!=0;
    uint cost=SGM_INVALID_COST;if(supported)cost=sgm_cost(reference,target_desc[i+d],true);return cost;
}
uint reference_sum<let Components:uint,let Bits:uint,let RightReference:bool>(uint width,uint x,uint i,uint d,uint word_base,uint paths,uint reference,bool reference_valid,uint* target_valid,uint* target_desc,uint* words){
    uint field=(words[word_base+d/Components]>>((d%Components)*Bits))&((1u<<Bits)-1);
    return paths*reference_cost<RightReference>(width,x,i,d,reference,reference_valid,target_valid,target_desc)+field;
}
void aggregate_specialized_body<let Components:uint,let Bits:uint,let RightReference:bool>(Parameters parameters,uint path,uint groups_x,uint3 gid){
    uint width=parameters.width,height=parameters.height,disparities=parameters.disparities,p1=parameters.p1,p2=parameters.p2;
    uint* left_desc=(uint*)(RightReference?parameters.right_desc_address:parameters.left_desc_address);uint* right_desc=(uint*)(RightReference?parameters.left_desc_address:parameters.right_desc_address);
    uint* left_valid=(uint*)(RightReference?parameters.right_valid_address:parameters.left_valid_address);uint* right_valid=(uint*)(RightReference?parameters.left_valid_address:parameters.right_valid_address);
    uint* words=(uint*)parameters.penalties_address;
    uint lanes=WaveGetLaneCount(),lane=WaveGetLaneIndex();int2 dir=sgm_direction(path);
    uint line=(gid.y*groups_x+gid.x)*(64/lanes)+WaveGetWaveIndex();if(line>=sgm_line_count(width,height,dir))return;
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
                if(d<disparities){uint c=reference_cost<RightReference>(width,x,i,d,reference,reference_valid,right_valid,right_desc);next[r]=sgm_step(c,prev[r],lo,hi,minimum,p1,p2,first);addition|=(next[r]-c)<<(component*Bits);}else next[r]=0;
            }
            if(word<row_words)words[i*row_words+word]+=addition;
        }
        [unroll]for(uint r=0;r<8;++r)prev[r]=next[r];first=false;xy+=dir;
    }
}
void select_specialized_body<let Components:uint,let Bits:uint,let RightReference:bool>(Parameters parameters,uint groups_x,uint3 gid,uint3 tid){
    uint i=(gid.y*groups_x+gid.x)*64+tid.x/4,lane=tid.x%4,t=tid.x;
    uint width=parameters.width,disparities=parameters.disparities,paths=parameters.paths,subpixel=parameters.subpixel;
    uint x=i%width,word_base=i*((disparities+Components-1)/Components);
    uint* left_desc=(uint*)(RightReference?parameters.right_desc_address:parameters.left_desc_address);uint* right_desc=(uint*)(RightReference?parameters.left_desc_address:parameters.right_desc_address);
    uint* left_valid=(uint*)(RightReference?parameters.right_valid_address:parameters.left_valid_address);uint* right_valid=(uint*)(RightReference?parameters.left_valid_address:parameters.right_valid_address);
    uint* words=(uint*)parameters.penalties_address;
    uint reference=0;bool reference_valid=false;
    if(i<width*parameters.height){reference=left_desc[i];reference_valid=left_valid[i]!=0;}
    uint best=0xffffffff,best_cost=0xffffffff,second=0xffffffff;
    if(i<width*parameters.height)for(uint d=lane;d<disparities;d+=4){uint c=reference_sum<Components,Bits,RightReference>(width,x,i,d,word_base,paths,reference,reference_valid,right_valid,right_desc,words);if(c<best_cost||(c==best_cost&&d<best)){second=best_cost;best_cost=c;best=d;}else second=min(second,c);}
    select_best_cost[t]=best_cost;select_second_cost[t]=second;select_best_index[t]=best;GroupMemoryBarrierWithGroupSync();
    for(uint stride=2;stride>0;stride/=2){if(lane<stride){uint other=t+stride;uint oc=select_best_cost[other],oi=select_best_index[other],own=select_best_cost[t],index=select_best_index[t];
        if(oc<own||(oc==own&&oi<index)){select_second_cost[t]=min(select_second_cost[other],own);select_best_cost[t]=oc;select_best_index[t]=oi;}else select_second_cost[t]=min(select_second_cost[t],oc);}
        GroupMemoryBarrierWithGroupSync();}
    if(lane!=0||i>=width*parameters.height)return;
    best=select_best_index[t];best_cost=select_best_cost[t];second=select_second_cost[t];if(disparities==1)second=best_cost;
    bool valid=sgm_selection_valid(best,best_cost,second);float value=float(best);
    if(subpixel!=0&&best>0&&best+1<disparities)value=sgm_subpixel(best,best_cost,reference_sum<Components,Bits,RightReference>(width,x,i,best-1,word_base,paths,reference,reference_valid,right_valid,right_desc,words),reference_sum<Components,Bits,RightReference>(width,x,i,best+1,word_base,paths,reference,reference_valid,right_valid,right_desc,words));
    float* output_disparity=(float*)parameters.disparity_address;float* output_confidence=(float*)parameters.confidence_address;uint* output_validity=(uint*)parameters.validity_address;
    output_disparity[i]=valid?value:asfloat(0x7fc00000);output_confidence[i]=valid?float(second-best_cost)/float(max(second,1u)):0;output_validity[i]=valid?1:0;
}

#endif
