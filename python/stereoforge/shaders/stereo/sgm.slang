// Shared explicit compute entry points. Row-major pixels, disparity fastest.
// Grayscale is uint32, Census 24-bit, costs/aggregation uint32, disparity float32.
StructuredBuffer<uint> image;
RWStructuredBuffer<uint> descriptor;
RWStructuredBuffer<uint> descriptor_valid;
StructuredBuffer<uint> left_desc;
StructuredBuffer<uint> right_desc;
StructuredBuffer<uint> left_valid;
StructuredBuffer<uint> right_valid;
StructuredBuffer<uint> lower;
StructuredBuffer<uint> upper;
RWStructuredBuffer<uint> costs;
RWStructuredBuffer<uint> sums;
RWStructuredBuffer<float> disparity;
RWStructuredBuffer<float> confidence;
RWStructuredBuffer<uint> validity;
uint width, height, disparity_count, path_count;
uint p1, p2, bounded, subpixel;
int dx,dy;
uint line_count;

[shader("compute")] [numthreads(64,1,1)]
void census(uint3 tid:SV_DispatchThreadID) {
    uint i=tid.x; if(i>=width*height)return;
    int x=int(i%width),y=int(i/width); uint value=0;
    bool valid=x>=2&&y>=2&&x<int(width)-2&&y<int(height)-2;
    if(valid) { uint bit=0; for(int oy=-2;oy<=2;oy++)for(int ox=-2;ox<=2;ox++) {
        if(ox==0&&oy==0)continue;
        if(image[(y+oy)*int(width)+x+ox]<image[i])value|=1u<<bit;
        bit++;
    }}
    descriptor[i]=value; descriptor_valid[i]=valid?1:0;
}

[shader("compute")] [numthreads(64,1,1)]
void cost(uint3 tid:SV_DispatchThreadID) {
    uint index=tid.x; if(index>=width*height*disparity_count)return;
    uint i=index/disparity_count,d=index%disparity_count,x=i%width;
    bool ok=x>=d && left_valid[i]!=0;
    if(ok)ok=right_valid[i-d]!=0;
    if(bounded!=0)ok=ok&&d>=lower[i]&&d<upper[i];
    costs[index]=ok?countbits(left_desc[i]^right_desc[i-d]):10000;
    sums[index]=0;
}

groupshared uint previous[256];
groupshared uint current[256];
groupshared uint reduction[256];
[shader("compute")] [numthreads(256,1,1)]
void aggregate(uint3 gid:SV_GroupID,uint3 lane:SV_GroupThreadID) {
    uint line=gid.x,d=lane.x; if(line>=line_count)return;
    int x,y;
    if(dy==0){x=dx>0?0:int(width)-1;y=int(line);}
    else if(dx==0){x=int(line);y=dy>0?0:int(height)-1;}
    else if(line<width){x=int(line);y=dy>0?0:int(height)-1;}
    else {x=dx>0?0:int(width)-1;y=dy>0?int(line-width)+1:int(height)-2-int(line-width);}
    previous[d]=0; GroupMemoryBarrierWithGroupSync();
    bool first=true;
    while(x>=0&&x<int(width)&&y>=0&&y<int(height)) {
        reduction[d]=d<disparity_count?previous[d]:1000000;
        GroupMemoryBarrierWithGroupSync();
#if SINTERGPU_USE_SHARED_REDUCTION
        SINTERGPU_REDUCE_256(reduction, d, 1, min)
#else
        for(uint stride=128;stride>0;stride>>=1){
            if(d<stride)reduction[d]=min(reduction[d],reduction[d+stride]);
            GroupMemoryBarrierWithGroupSync();
        }
#endif
        uint index=(uint(y)*width+uint(x))*disparity_count+d;
        uint value=0;
        if(d<disparity_count){
            uint best=min(previous[d],reduction[0]+p2);
            if(d>0)best=min(best,previous[d-1]+p1);
            if(d+1<disparity_count)best=min(best,previous[d+1]+p1);
            value=costs[index]+(first?0:best-reduction[0]);
            sums[index]+=value;
        }
        current[d]=value; GroupMemoryBarrierWithGroupSync();
        previous[d]=current[d]; GroupMemoryBarrierWithGroupSync();
        first=false; x+=dx;y+=dy;
    }
}

// Four adjacent lanes scan each pixel's disparity-major cost row.
// Padded pixels participate in every barrier and never access source buffers.
groupshared uint select_best_cost[256];
groupshared uint select_second_cost[256];
groupshared uint select_best_index[256];
[shader("compute")] [numthreads(256,1,1)]
void select(uint3 group_id:SV_GroupID,uint3 group_thread_id:SV_GroupThreadID){
    uint i=group_id.x*64+group_thread_id.x/4,lane=group_thread_id.x%4;
    uint thread=group_thread_id.x;
    uint offset=i*disparity_count,best=0xffffffff,best_cost=0xffffffff,second=0xffffffff;
    if(i<width*height)for(uint d=lane;d<disparity_count;d+=4){
        uint c=sums[offset+d];
        if(c<best_cost||(c==best_cost&&d<best)){
            second=best_cost;best_cost=c;best=d;
        }else second=min(second,c);
    }
    select_best_cost[thread]=best_cost;
    select_second_cost[thread]=second;
    select_best_index[thread]=best;
    GroupMemoryBarrierWithGroupSync();
    for(uint stride=2;stride>0;stride/=2){
        if(lane<stride){
            uint other=thread+stride;
            uint other_cost=select_best_cost[other],other_index=select_best_index[other];
            uint own_cost=select_best_cost[thread],own_index=select_best_index[thread];
            if(other_cost<own_cost||(other_cost==own_cost&&other_index<own_index)){
                select_second_cost[thread]=min(select_second_cost[other],own_cost);
                select_best_cost[thread]=other_cost;select_best_index[thread]=other_index;
            }else select_second_cost[thread]=min(select_second_cost[thread],other_cost);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if(lane!=0||i>=width*height)return;
    best=select_best_index[thread];best_cost=select_best_cost[thread];second=select_second_cost[thread];
    if(disparity_count==1)second=best_cost;
    bool valid=best>0&&best_cost<10000&&second>best_cost;
    float value=float(best);
    if(subpixel!=0&&best>0&&best+1<disparity_count){
        float lo=float(sums[offset+best-1]),hi=float(sums[offset+best+1]);
        float denominator=lo+hi-2*float(best_cost);
        if(denominator>0)value+=.5*(lo-hi)/denominator;
    }
    disparity[i]=valid?value:asfloat(0x7fc00000);
    confidence[i]=valid?float(second-best_cost)/float(max(second,1u)):0;
    validity[i]=valid?1:0;
}

// Optional metric output. Parameters require explicit calibrated units:
// focal length in pixels, baseline in metres, principal offset cx_left-cx_right.
float metric_fx, metric_baseline, metric_principal_offset;
RWStructuredBuffer<float> metric_depth_output;
RWStructuredBuffer<uint> metric_validity_output;
[shader("compute")] [numthreads(64,1,1)]
void depth_convert(uint3 tid:SV_DispatchThreadID) {
    uint i=tid.x; if(i>=width*height)return;
    float corrected=disparity[i]-metric_principal_offset;
    float z=metric_fx*metric_baseline/corrected;
    bool valid=validity[i]!=0 && isfinite(corrected) && corrected>0 && isfinite(z) && z>0;
    metric_depth_output[i]=valid ? z : asfloat(0x7fc00000);
    metric_validity_output[i]=valid ? 1 : 0;
}
