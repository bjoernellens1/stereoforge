// Independent dense frozen-RHI sum oracle. No packed production selector imports.
StructuredBuffer<uint> sums;
StructuredBuffer<uint> left_valid;
StructuredBuffer<uint> right_valid;
RWStructuredBuffer<float> disparity;
RWStructuredBuffer<float> confidence;
RWStructuredBuffer<uint> validity;
uniform uint width, height, disparity_count, subpixel;
bool supported(uint pixel,uint d) {
    uint x=pixel%width;
    return d<disparity_count && d<width-x && right_valid[pixel]!=0 && left_valid[pixel+d]!=0;
}
uint shifted_cost(uint pixel,uint d) { return sums[(pixel+d)*disparity_count+d]; }
[shader("compute")] [numthreads(64,1,1)]
void derived_right(uint3 tid:SV_DispatchThreadID) {
    uint pixel=tid.x;if(pixel>=width*height)return;
    uint best=0,best_cost=0xffffffff,second=0xffffffff,count=0;
    for(uint d=0;d<disparity_count;d++)if(supported(pixel,d)){
        count++;uint cost=shifted_cost(pixel,d);
        if(cost<best_cost){second=best_cost;best_cost=cost;best=d;}
        else second=min(second,cost);
    }
    bool valid=count>=2 && best>0 && best_cost<10000 && second>best_cost;
    float value=float(best);
    if(subpixel!=0 && best>0 && best+1<disparity_count && supported(pixel,best-1) && supported(pixel,best+1)){
        float lo=float(shifted_cost(pixel,best-1)),hi=float(shifted_cost(pixel,best+1));
        float denominator=lo+hi-2*float(best_cost);
        if(denominator>0)value+=.5*(lo-hi)/denominator;
    }
    disparity[pixel]=valid?value:asfloat(0x7fc00000);
    confidence[pixel]=valid?float(second-best_cost)/float(max(second,1u)):0;
    validity[pixel]=valid?1:0;
}
