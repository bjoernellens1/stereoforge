// Isolated integer FP2 selection and metric observation experiment.
struct Raw { int winner; uint best; uint second; uint count; uint disparity; uint confidence; uint valid; };
struct Observation {uint lr_valid; uint depth; uint confidence; uint metric_valid; uint provenance; uint source_low; uint source_high; uint age_low; uint age_high;};
struct SelectParams {
    uint64_t lo; uint64_t hi; uint64_t offsets; uint64_t sums; uint64_t paths;
    uint64_t raw; uint64_t obs; uint64_t rect_support;
    uint width; uint height; uint disparities; uint eye; uint max_words;
    float numerator; float principal_offset; uint source_low; uint source_high; uint64_t status; uint64_t metric_lut;
};
struct Push {uint64_t parameters;};
[[vk::push_constant]] ConstantBuffer<Push> pc;
#define p (*(SelectParams*)pc.parameters)
static const uint INF=0xffffffffu;static const uint NAN_BITS=0x7fc00000u;
[shader("compute")][numthreads(64,1,1)]
void reset_workspace(uint3 tid:SV_DispatchThreadID){uint stride=64*min((p.max_words+63)/64,65535u);for(uint i=tid.x;i<p.max_words;i+=stride){((uint*)p.sums)[i]=INF;for(uint pass=0;pass<4;++pass)((uint*)p.paths)[pass*p.max_words+i]=INF;}}
uint confidence_rte(uint n,uint d){
    int e=0;while((uint64_t(n)<<uint(-e))<d)--e;
    uint64_t scaled=uint64_t(n)<<uint(23-e);uint64_t quotient=scaled/d,remainder=scaled%d;
    if(2*remainder>d||(2*remainder==d&&(quotient&1)!=0))++quotient;
    if(quotient==0x1000000){quotient>>=1;++e;}
    return uint((e+127)<<23)|uint(quotient-0x800000);
}
[shader("compute")][numthreads(64,1,1)]
void select_packed(uint3 tid:SV_DispatchThreadID){
    uint i=tid.x,n=p.width*p.height;if(i>=n)return;
    uint* lo=(uint*)p.lo;uint* hi=(uint*)p.hi;uint64_t* offsets=(uint64_t*)p.offsets;uint* sums=(uint*)p.sums;
    Raw r;r.winner=-1;r.best=INF;r.second=INF;r.count=0;r.disparity=NAN_BITS;r.confidence=0;r.valid=0;
    for(uint d=lo[i];d<hi[i];++d){uint v=sums[offsets[i]+d-lo[i]];if(v==INF)continue;if(v>224){uint old;InterlockedOr(((uint*)p.status)[0],1,old);continue;}++r.count;if(v<r.best){r.second=r.best;r.best=v;r.winner=int(d);}else if(v<r.second)r.second=v;}
    bool restricted=(r.winner==int(lo[i])&&lo[i]>0)||(r.winner==int(hi[i])-1&&hi[i]<p.disparities);
    if(r.count>=2&&r.winner>0&&r.second>r.best&&!restricted){r.valid=1;r.disparity=asuint(float(r.winner));r.confidence=confidence_rte(r.second-r.best,r.second);}
    ((Raw*)p.raw)[p.eye*n+i]=r;
}
[shader("compute")][numthreads(64,1,1)]
void consistency(uint3 tid:SV_DispatchThreadID){
    uint t=tid.x,n=p.width*p.height;if(t>=2*n)return;uint eye=t/n,i=t%n,x=i%p.width;Raw* raw=(Raw*)p.raw;Raw a=raw[t];Observation o;o.lr_valid=0;o.depth=NAN_BITS;o.confidence=0;o.metric_valid=0;o.provenance=0;o.source_low=INF;o.source_high=INF;o.age_low=0;o.age_high=0;
    if(a.valid!=0){int target=eye!=0?int(x)+a.winner:int(x)-a.winner;if(target>=0&&target<int(p.width)){Raw b=raw[(1-eye)*n+(i/p.width)*p.width+uint(target)];if(b.valid!=0&&abs(a.winner-b.winner)<=1)o.lr_valid=1;}}
    ((Observation*)p.obs)[t]=o;
}
[shader("compute")][numthreads(64,1,1)]
void metric(uint3 tid:SV_DispatchThreadID){
    uint t=tid.x,n=p.width*p.height;if(t>=2*n)return;Raw a=((Raw*)p.raw)[t];Observation* all=(Observation*)p.obs;Observation o=all[t];uint* support=(uint*)p.rect_support;
    if(o.lr_valid!=0&&support[t]!=0){if(a.winner<=0||a.winner>=int(p.disparities)){uint old;InterlockedOr(((uint*)p.status)[0],2,old);}else{uint2 entry=((uint2*)p.metric_lut)[uint(a.winner)];if(entry.y!=0){o.depth=entry.x;o.confidence=a.confidence;o.metric_valid=1;o.provenance=1;o.source_low=p.source_low;o.source_high=p.source_high;}}}
    all[t]=o;
}
