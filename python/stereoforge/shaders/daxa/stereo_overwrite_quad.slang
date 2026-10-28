#include "stereo_parameters.slang"
#include "../ops/sgm_specialized.slang"
struct Push {uint64_t parameters;uint pass;uint groups_x;};
[[vk::push_constant]] ConstantBuffer<Push> pc;
void select_quad_body<let Components:uint,let Bits:uint,let RightReference:bool>(Parameters parameters,uint3 gid){
    uint wave_size=WaveGetLaneCount(),wave_lane=WaveGetLaneIndex(),lane=wave_lane%4;
    uint i=(gid.y*pc.groups_x+gid.x)*16+WaveGetWaveIndex()*(wave_size/4)+wave_lane/4;
    uint width=parameters.width,disparities=parameters.disparities,paths=parameters.paths,subpixel=parameters.subpixel;
    uint x=i%width,word_base=i*((disparities+Components-1)/Components);
    uint* left_desc=(uint*)(RightReference?parameters.right_desc_address:parameters.left_desc_address);uint* right_desc=(uint*)(RightReference?parameters.left_desc_address:parameters.right_desc_address);
    uint* left_valid=(uint*)(RightReference?parameters.right_valid_address:parameters.left_valid_address);uint* right_valid=(uint*)(RightReference?parameters.left_valid_address:parameters.right_valid_address);
    uint* words=(uint*)parameters.penalties_address;
    uint reference=0;bool reference_valid=false;
    if(i<width*parameters.height){reference=left_desc[i];reference_valid=left_valid[i]!=0;}
    uint best=0xffffffff,best_cost=0xffffffff,second=0xffffffff;
    if(i<width*parameters.height)for(uint d=lane;d<disparities;d+=4){uint c=reference_sum<Components,Bits,RightReference>(width,x,i,d,word_base,paths,reference,reference_valid,right_valid,right_desc,words);if(c<best_cost||(c==best_cost&&d<best)){second=best_cost;best_cost=c;best=d;}else second=min(second,c);}
    // Every full-wave lane executes every shuffle, including tail pixels.
    // XOR2 then XOR1 stays inside the same4-lane pixel for all active sources.
    [unroll]for(uint stride=2;stride>0;stride/=2){
        uint source=wave_lane^stride;
        uint other_cost=WaveReadLaneAt(best_cost,source);
        uint other_index=WaveReadLaneAt(best,source);
        uint other_second=WaveReadLaneAt(second,source);
        if(lane<stride){if(other_cost<best_cost||(other_cost==best_cost&&other_index<best)){second=min(other_second,best_cost);best_cost=other_cost;best=other_index;}else second=min(second,other_cost);}
    }
    if(lane!=0||i>=width*parameters.height)return;
    if(disparities==1)second=best_cost;
    bool valid=sgm_selection_valid(best,best_cost,second);float value=float(best);
    if(subpixel!=0&&best>0&&best+1<disparities)value=sgm_subpixel(best,best_cost,reference_sum<Components,Bits,RightReference>(width,x,i,best-1,word_base,paths,reference,reference_valid,right_valid,right_desc,words),reference_sum<Components,Bits,RightReference>(width,x,i,best+1,word_base,paths,reference,reference_valid,right_valid,right_desc,words));
    float* disparity=(float*)parameters.disparity_address;float* confidence=(float*)parameters.confidence_address;uint* validity=(uint*)parameters.validity_address;
    disparity[i]=valid?value:asfloat(0x7fc00000);confidence[i]=valid?float(second-best_cost)/float(max(second,1u)):0;validity[i]=valid?1:0;
}

[shader("compute")][numthreads(64,1,1)]
void select_overwrite_quad_left8(uint3 gid:SV_GroupID){select_quad_body<4,8,false>(*(Parameters*)pc.parameters,gid);}
[shader("compute")][numthreads(64,1,1)]
void select_overwrite_quad_left16(uint3 gid:SV_GroupID){select_quad_body<2,16,false>(*(Parameters*)pc.parameters,gid);}
[shader("compute")][numthreads(64,1,1)]
void select_overwrite_quad_right8(uint3 gid:SV_GroupID){select_quad_body<4,8,true>(*(Parameters*)pc.parameters,gid);}
[shader("compute")][numthreads(64,1,1)]
void select_overwrite_quad_right16(uint3 gid:SV_GroupID){select_quad_body<2,16,true>(*(Parameters*)pc.parameters,gid);}
