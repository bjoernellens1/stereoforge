#include "../ops/preprocess.slang"
#include "../ops/refinement.slang"
#include "../ops/stereo_observation.slang"
struct Parameters {
    uint64_t original[2];uint64_t coarse[3];uint64_t maps[4];uint64_t output[30];
    uint width;uint height;uint source_width;uint source_height;uint coarse_width;uint coarse_height;
    uint radius;uint disparities;uint min_margin;uint subpixel;
    float prior_floor;float max_span;float lr_tolerance;float fx;float baseline;float principal_offset;
    uint source_low;uint source_high;uint64_t borrowed[4];
};
struct Push{uint64_t parameters;};[[vk::push_constant]] Push pc;
#define p (*(Parameters*)pc.parameters)
uint* uplane(uint plane){return (uint*)p.output[plane];}float* fplane(uint plane){return (float*)p.output[plane];}
[shader("compute")][numthreads(64,1,1)]
void rectify(uint3 tid:SV_DispatchThreadID){uint n=p.width*p.height,t=tid.x;if(t>=2*n)return;uint eye=t/n,i=t%n;float* mx=(float*)p.maps[eye*2];float* my=(float*)p.maps[eye*2+1];int qx=preprocess_quantize(mx[i]),qy=preprocess_quantize(my[i]);int x=qx>>5,y=qy>>5;uint fx=uint(qx)&31,fy=uint(qy)&31;uint weights[4]={(32-fx)*(32-fy),fx*(32-fy),(32-fx)*fy,fx*fy};uint3 channels=0;bool full_support=true;uint* original=(uint*)p.original[eye];[unroll]for(uint tap=0;tap<4;++tap){int xx=x+int(tap&1),yy=y+int(tap>>1);bool inside=xx>=0&&yy>=0&&xx<int(p.source_width)&&yy<int(p.source_height);if(weights[tap]!=0&&!inside)full_support=false;if(inside){uint rgb=original[uint(yy)*p.source_width+uint(xx)];channels+=uint3(rgb&255,(rgb>>8)&255,(rgb>>16)&255)*weights[tap];}}channels=(channels+512)>>10;uint color=channels.x|(channels.y<<8)|(channels.z<<16)|0xff000000u;uplane(eye==0?26:29)[i]=color;uplane(27)[t]=preprocess_gray(color);uplane(28)[t]=full_support?1:0;}
[shader("compute")][numthreads(64,1,1)]
void census(uint3 tid:SV_DispatchThreadID){uint n=p.width*p.height,t=tid.x;if(t>=2*n)return;uint eye=t/n,i=t%n,x=i%p.width,y=i/p.width;uint* gray=p.borrowed[0]!=0?(uint*)p.borrowed[eye]:uplane(27)+eye*n;uint* support=p.borrowed[0]!=0?(uint*)p.borrowed[2+eye]:uplane(28)+eye*n;bool valid=refinement_support(support,p.width,p.height,x,y);uplane(24)[t]=valid?refinement_census(gray,p.width,x,y):0;uplane(25)[t]=uint(valid);}
[shader("compute")][numthreads(64,1,1)]
void match(uint3 tid:SV_DispatchThreadID){uint n=p.width*p.height,t=tid.x;if(t>=2*n)return;uint eye=t/n,i=t%n,x=i%p.width,y=i/p.width,cn=p.coarse_width*p.coarse_height;
    // Half-pixel 2x coordinates are exactly quarter-integers.
    float cx=(float(x)+.5f)*.5f-.5f,cy=(float(y)+.5f)*.5f-.5f;int ix=int(floor(cx)),iy=int(floor(cy));float wx=cx-float(ix),wy=cy-float(iy);float weights[4]={(1-wx)*(1-wy),wx*(1-wy),(1-wx)*wy,wx*wy};precise float predicted=0;float mn=3.402823466e38f,mx=-3.402823466e38f;uint reason=0;
    [unroll]for(uint tap=0;tap<4;++tap){if(weights[tap]==0)continue;int xx=ix+int(tap&1),yy=iy+int(tap>>1);if(xx<0||yy<0||xx>=int(p.coarse_width)||yy>=int(p.coarse_height)){reason|=REF_OUTSIDE;continue;}uint j=eye*cn+uint(yy)*p.coarse_width+uint(xx);float d=((float*)p.coarse[0])[j],confidence=((float*)p.coarse[1])[j];if(((uint*)p.coarse[2])[j]==0)reason|=REF_NO_PRIOR;if(!isfinite(d))reason|=REF_NONFINITE;if(confidence<p.prior_floor)reason|=REF_LOW_CONFIDENCE;if(isfinite(d)){if(d<0||d>float(p.disparities-1)*.5f)reason|=REF_OUTSIDE;mn=min(mn,d);mx=max(mx,d);predicted=predicted+d*weights[tap];}}
    if(mx-mn>p.max_span)reason|=REF_PRIOR_SPAN;bool prior=reason==0;predicted=predicted*2;fplane(9)[t]=prior?predicted:refinement_invalid();uplane(10)[t]=uint(prior);
    uint lo=0,hi=0xffffffff;if(prior){int centre=int(floor(predicted+.5f));lo=uint(max(0,centre-int(p.radius)));hi=uint(min(int(p.disparities)-1,centre+int(p.radius)));}uplane(11)[t]=lo;uplane(12)[t]=hi;
    uint best=0xffffffff,second=0xffffffff,winner=0xffffffff,count=0,work=0,minus=0xffffffff,plus=0xffffffff;
    uint saved_costs[17];[unroll]for(uint k=0;k<17;++k)saved_costs[k]=0xffffffff;bool supported=uplane(25)[t]!=0;if(prior&&!supported)reason|=REF_UNSUPPORTED;
    if(prior&&supported){for(uint d=lo;d<=hi;++d){int xx=int(x)+(eye!=0?int(d):-int(d));if(xx<0||xx>=int(p.width))continue;uint j=(1-eye)*n+y*p.width+uint(xx);if(uplane(25)[j]==0)continue;uint cost=countbits(uplane(24)[t]^uplane(24)[j]);saved_costs[d-lo]=cost;++count;++work;if(cost<best){second=best;best=cost;winner=d;}else second=min(second,cost);}
        if(winner!=0xffffffff){if(winner>lo)minus=saved_costs[winner-lo-1];if(winner<hi)plus=saved_costs[winner-lo+1];}
    }
    bool unique=count>=2&&second-best>=p.min_margin;if(count<2)reason|=REF_TOO_FEW;if(count>=2&&!unique)reason|=REF_AMBIGUOUS;bool restricted=count>0&&((winner==lo&&lo>0)||(winner==hi&&hi+1<p.disparities));if(restricted)reason|=REF_RESTRICTED;if(count>0&&(winner==0||winner+1==p.disparities))reason|=REF_GLOBAL;bool valid=unique&&!restricted;
    float d=float(winner),confidence=0;bool applied=false;if(valid){confidence=clamp(float(second-best)/float(max(second,1u)),0.f,1.f);if(p.subpixel!=0){if(minus!=0xffffffff&&plus!=0xffffffff){precise float denominator=(float(minus)-2.f*float(best))+float(plus);precise float numerator=.5f*(float(minus)-float(plus));precise float offset=denominator!=0?clamp(numerator/denominator,-1.f,1.f):0;d=d+offset;applied=true;}else reason|=REF_MISSING_SUBPIXEL;}}
    uplane(13)[t]=reason;uplane(14)[t]=prior?1:0;uplane(15)[t]=winner;uplane(16)[t]=best;uplane(17)[t]=second;uplane(18)[t]=count;uplane(19)[t]=work;uplane(20)[t]=uint(applied);fplane(21)[t]=valid?d:refinement_invalid();fplane(22)[t]=valid?confidence:0;uplane(23)[t]=uint(valid);
}
[shader("compute")][numthreads(64,1,1)]
void lr(uint3 tid:SV_DispatchThreadID){uint n=p.width*p.height,t=tid.x;if(t>=2*n)return;uint eye=t/n,i=t%n,target=0;float d=fplane(21)[t];bool valid=false;if(uplane(23)[t]!=0&&refinement_target(p.width,i%p.width,d,eye!=0,target)){uint j=(1-eye)*n+(i/p.width)*p.width+target;valid=observation_lr(d,fplane(21)[j],uplane(23)[t],uplane(23)[j],p.lr_tolerance);}if(uplane(23)[t]!=0&&!valid)uplane(13)[t]|=REF_LR_FAILED;fplane(0)[t]=valid?d:refinement_invalid();fplane(1)[t]=valid?fplane(22)[t]:0;uplane(2)[t]=uint(valid);}
[shader("compute")][numthreads(64,1,1)]
void metric(uint3 tid:SV_DispatchThreadID){uint n=p.width*p.height,t=tid.x;if(t>=2*n)return;uint eye=t/n,i=t%n,target=0;float z=refinement_invalid(),d=fplane(0)[t];bool valid=uplane(2)[t]!=0&&d>0&&(d-p.principal_offset)>0;if(valid&&refinement_target(p.width,i%p.width,d,eye!=0,target)){uint j=(1-eye)*n+(i/p.width)*p.width+target;valid=uplane(25)[t]!=0&&uplane(25)[j]!=0;if(valid){precise float numerator=p.fx*p.baseline;precise float denominator=d-p.principal_offset;z=numerator/denominator;valid=isfinite(z)&&z>0;}}else valid=false;fplane(3)[t]=valid?z:refinement_invalid();fplane(4)[t]=valid?fplane(1)[t]:0;uplane(5)[t]=uint(valid);uplane(6)[t]=valid?1:0;((uint2*)p.output[7])[t]=valid?uint2(p.source_low,p.source_high):uint2(0xffffffff,0xffffffff);((uint2*)p.output[8])[t]=uint2(0,0);}
