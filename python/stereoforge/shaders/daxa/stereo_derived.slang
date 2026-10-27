#include "stereo_parameters.slang"
#include "../ops/sgm.slang"
struct Push {uint64_t parameters;uint pass;uint groups_x;};
[[vk::push_constant]] ConstantBuffer<Push> pc;
groupshared uint derived_best_cost[256],derived_second_cost[256],derived_best_index[256],derived_count[256];
// All support checks precede descriptor or packed-penalty loads. No raw remap veto.
bool derived_supported(Parameters parameters,uint i,uint d){
    if(i>=parameters.width*parameters.height||d>=parameters.disparities)return false;
    if(d>=parameters.width-i%parameters.width)return false;
    if(((uint*)parameters.right_valid_address)[i]==0)return false;
    return ((uint*)parameters.left_valid_address)[i+d]!=0;
}
uint derived_score<let Components:uint,let Bits:uint>(Parameters parameters,uint i,uint d){
    uint left=i+d,row_words=(parameters.disparities+Components-1)/Components;
    uint cost=sgm_cost(((uint*)parameters.left_desc_address)[left],((uint*)parameters.right_desc_address)[i],true);
    uint field=(((uint*)parameters.penalties_address)[left*row_words+d/Components]>>((d%Components)*Bits))&((1u<<Bits)-1);
    return parameters.paths*cost+field;
}
void derived_select<let Components:uint,let Bits:uint>(uint3 gid,uint3 tid){
    Parameters parameters=*(Parameters*)pc.parameters;
    uint i=(gid.y*pc.groups_x+gid.x)*64+tid.x/4,lane=tid.x%4,t=tid.x;
    uint best=0xffffffff,best_cost=0xffffffff,second=0xffffffff,count=0;
    for(uint d=lane;d<parameters.disparities;d+=4)if(derived_supported(parameters,i,d)){
        ++count;uint c=derived_score<Components,Bits>(parameters,i,d);
        if(c<best_cost||(c==best_cost&&d<best)){second=best_cost;best_cost=c;best=d;}else second=min(second,c);
    }
    derived_best_cost[t]=best_cost;derived_second_cost[t]=second;derived_best_index[t]=best;derived_count[t]=count;GroupMemoryBarrierWithGroupSync();
    for(uint stride=2;stride>0;stride/=2){if(lane<stride){uint other=t+stride;uint oc=derived_best_cost[other],oi=derived_best_index[other],own=derived_best_cost[t],index=derived_best_index[t];
        if(oc<own||(oc==own&&oi<index)){derived_second_cost[t]=min(derived_second_cost[other],own);derived_best_cost[t]=oc;derived_best_index[t]=oi;}else derived_second_cost[t]=min(derived_second_cost[t],oc);
        derived_count[t]+=derived_count[other];}GroupMemoryBarrierWithGroupSync();}
    if(lane!=0||i>=parameters.width*parameters.height)return;
    best=derived_best_index[t];best_cost=derived_best_cost[t];second=derived_second_cost[t];
    bool valid=derived_count[t]>=2&&sgm_selection_valid(best,best_cost,second);float value=float(best);
    if(parameters.subpixel!=0&&best>0&&best+1<parameters.disparities&&derived_supported(parameters,i,best-1)&&derived_supported(parameters,i,best+1))
        value=sgm_subpixel(best,best_cost,derived_score<Components,Bits>(parameters,i,best-1),derived_score<Components,Bits>(parameters,i,best+1));
    ((float*)parameters.disparity_address)[i]=valid?value:asfloat(0x7fc00000);
    ((float*)parameters.confidence_address)[i]=valid?float(second-best_cost)/float(max(second,1u)):0;
    ((uint*)parameters.validity_address)[i]=valid?1:0;
}
[shader("compute")][numthreads(256,1,1)]
void select_right_derived8(uint3 gid:SV_GroupID,uint3 tid:SV_GroupThreadID){derived_select<4,8>(gid,tid);}
[shader("compute")][numthreads(256,1,1)]
void select_right_derived16(uint3 gid:SV_GroupID,uint3 tid:SV_GroupThreadID){derived_select<2,16>(gid,tid);}
