// Optional research-only multi-pair confirmation. Raw cached pixels/metric
// evidence remain distinct from the accumulated temporal output.
struct TemporalStereoPair {
    uint64_t buffers[7];uint2 source_id;float current_to_left[32];
};
bool pair_inside(TemporalParameters p,float2 uv) {
    return all(isfinite(uv))&&all(uv>=2.)&&uv.x<float(p.width)-3.&&uv.y<float(p.height)-3.;
}
float pair_gray(TemporalParameters p,uint64_t image,int2 q) {
    uint c=((uint*)image)[q.y*p.width+q.x];return (float(c&255)*.299+float((c>>8)&255)*.587+float((c>>16)&255)*.114)/255.;
}
float pair_sample(TemporalParameters p,uint64_t image,float2 uv) {
    int2 q=int2(floor(uv));float2 f=uv-float2(q);
    return lerp(lerp(pair_gray(p,image,q),pair_gray(p,image,q+int2(1,0)),f.x),lerp(pair_gray(p,image,q+int2(0,1)),pair_gray(p,image,q+int2(1,1)),f.x),f.y);
}
bool pair_zncc(TemporalParameters p,uint64_t a,float2 uv,uint64_t b,float2 target) {
    if(!pair_inside(p,uv)||!pair_inside(p,target))return false;
    float sa=0,sb=0,saa=0,sbb=0,sab=0;
    for(int y=-1;y<=1;++y)for(int x=-1;x<=1;++x) {
        float2 delta=float2(x,y);float va=pair_sample(p,a,uv+delta),vb=pair_sample(p,b,target+delta);
        sa+=va;sb+=vb;saa+=va*va;sbb+=vb*vb;sab+=va*vb;
    }
    float aa=max(0.,saa-sa*sa/9.),bb=max(0.,sbb-sb*sb/9.);
    return aa>.0001&&bb>.0001&&(sab-sa*sb/9.)/sqrt(aa*bb)>=.9;
}
bool pair_metric(TemporalParameters p,TemporalStereoPair pair,uint eye,float2 uv,float predicted_z) {
    if(!pair_inside(p,uv)||!temporal_normal(predicted_z))return false;
    int2 q=int2(floor(uv));float lo=1e30,hi=0;
    for(uint k=0;k<4;++k) {
        uint i=eye*p.width*p.height+(q.y+int(k/2))*p.width+q.x+k%2;
        float z=((float*)pair.buffers[2])[i];uint prov=((uint*)pair.buffers[4])[i];
        if(((uint*)pair.buffers[3])[i]==0||(prov!=1&&prov!=5)||!temporal_normal(z)||
           any(((uint2*)pair.buffers[5])[i]!=pair.source_id)||any(((uint2*)pair.buffers[6])[i]!=uint2(0))||abs(z-predicted_z)>p.tolerance*min(z,predicted_z))return false;
        lo=min(lo,z);hi=max(hi,z);
    }
    return hi-lo<=.01*lo;
}
bool temporal_pairs_confirm(TemporalParameters p,uint t,float z) {
    temporal_counter(p,10);
    uint n=p.width*p.height,eye=t/n,loc=t%n,k=eye*4,other=1-eye,ko=other*4;
    float2 uv=float2(loc%p.width,loc/p.width);
    float3 point=float3((uv.x-p.intrinsics[k+2])*z/p.intrinsics[k],(uv.y-p.intrinsics[k+3])*z/p.intrinsics[k+1],z);
    float other_x=point.x+(eye!=0?p.baseline_m:-p.baseline_m);
    float2 other_uv=float2(p.intrinsics[ko]*other_x/z+p.intrinsics[ko+2],p.intrinsics[ko+1]*point.y/z+p.intrinsics[ko+3]);
    // Current independent stereo image evidence is required even for a hole.
    if(p.pair_count==0||!pair_zncc(p,p.current_images[eye],uv,p.current_images[other],other_uv)){temporal_counter(p,11);return false;}
    uint votes=0,required=min(2u,p.pair_count);
    for(uint h=0;h<p.pair_count;++h) {
        TemporalStereoPair pair=((TemporalStereoPair*)p.pair_parameters)[h];uint m=eye*16;
        float3 q;
        for(uint r=0;r<3;++r)q[r]=pair.current_to_left[m+r*4]*point.x+pair.current_to_left[m+r*4+1]*point.y+pair.current_to_left[m+r*4+2]*point.z+pair.current_to_left[m+r*4+3];
        if(!temporal_normal(q.z))continue;
        float2 left=float2(p.intrinsics[0]*q.x/q.z+p.intrinsics[2],p.intrinsics[1]*q.y/q.z+p.intrinsics[3]);
        float2 right=float2(p.intrinsics[4]*(q.x-p.baseline_m)/q.z+p.intrinsics[6],p.intrinsics[5]*q.y/q.z+p.intrinsics[7]);
        if(!pair_metric(p,pair,0,left,q.z)||!pair_metric(p,pair,1,right,q.z))continue;
        if(pair_zncc(p,p.current_images[eye],uv,pair.buffers[0],left)&&pair_zncc(p,p.current_images[eye],uv,pair.buffers[1],right))++votes;
        if(votes>=required)return true;
    }
    temporal_counter(p,11);return false;
}
