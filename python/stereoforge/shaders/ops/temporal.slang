// Float32 normal positive axial-Z domain; IDs/ages use exact paired uint32.
struct TemporalParameters {
    uint64_t fresh[6];uint64_t history[11];uint64_t output[11];uint64_t support[2];
    uint64_t zbuffer;uint64_t owners;uint64_t counters;
    uint width;uint height;uint warp;uint capacity;uint max_reuses;
    uint2 max_age;uint2 elapsed;float tolerance;float confidence_floor;
    float intrinsics[8];float transforms[32];
};
bool temporal_normal(float z){return isfinite(z)&&z>=asfloat(0x00800000);}
bool temporal_less(uint2 a,uint2 b){return a.y<b.y||(a.y==b.y&&a.x<b.x);}
bool temporal_equal(uint2 a,uint2 b){return all(a==b);}
bool temporal_add_age(uint2 a,uint2 b,out uint2 result){uint lo=a.x+b.x;uint hi=a.y+b.y+uint(lo<a.x);result=uint2(lo,hi);return (a.y&0x80000000)==0&&(hi&0x80000000)==0&&!temporal_less(result,a);}
void temporal_counter(TemporalParameters p,uint index){InterlockedAdd(((uint*)p.counters)[index],1);}
// Returns projected target and transformed Z. Only the Z pass records candidate counters.
bool temporal_project(TemporalParameters p,uint t,bool count,out uint target,out float z){
    target=0;z=0;if(p.warp==0)return false;if(count)temporal_counter(p,4);
    if(((uint*)p.history[2])[t]==0)return false;
    float old=((float*)p.history[0])[t];if(!temporal_normal(old)){if(count)temporal_counter(p,1);return false;}
    uint cap=p.capacity&7u;
    float conf=((float*)p.history[1])[t];uint reuse=((uint*)p.history[9])[t],count_ids=((uint*)p.history[7])[t];uint2 aged,oldest;
    if(count_ids==0||count_ids>cap||!isfinite(conf)||conf<p.confidence_floor||conf>1||reuse>=p.max_reuses||!temporal_add_age(((uint2*)p.history[5])[t],p.elapsed,aged)||!temporal_add_age(((uint2*)p.history[8])[t],p.elapsed,oldest)||temporal_less(p.max_age,oldest))return false;
    if(count)temporal_counter(p,5);
    uint n=p.width*p.height,eye=t/n,i=t%n,k=eye*4,m=eye*16;
    float x=(float(i%p.width)-p.intrinsics[k+2])*old/p.intrinsics[k];float y=(float(i/p.width)-p.intrinsics[k+3])*old/p.intrinsics[k+1];
    float px=p.transforms[m]*x+p.transforms[m+1]*y+p.transforms[m+2]*old+p.transforms[m+3];
    float py=p.transforms[m+4]*x+p.transforms[m+5]*y+p.transforms[m+6]*old+p.transforms[m+7];
    z=p.transforms[m+8]*x+p.transforms[m+9]*y+p.transforms[m+10]*old+p.transforms[m+11];
    if(!isfinite(z)||z<=0)return false;
    if(!temporal_normal(z)){if(count)temporal_counter(p,2);return false;}
    float u=floor(p.intrinsics[k]*px/z+p.intrinsics[k+2]+.5);float v=floor(p.intrinsics[k+1]*py/z+p.intrinsics[k+3]+.5);
    if(!isfinite(u)||!isfinite(v)||u<0||v<0||u>=float(p.width)||v>=float(p.height))return false;
    uint local=uint(v)*p.width+uint(u);if(((uint*)p.support[eye])[local]==0)return false;target=eye*n+local;
    if(count)temporal_counter(p,6);return true;
}
void temporal_clear(TemporalParameters p,uint t){uint eyes=(p.capacity&128u)!=0?1u:2u,n=eyes*p.width*p.height;if(t<n){((uint*)p.zbuffer)[t]=0x7f800000;((uint*)p.owners)[t]=0xffffffff;}if(t<10)((uint*)p.counters)[t]=0;}
void temporal_splat_z(TemporalParameters p,uint t){uint eyes=(p.capacity&128u)!=0?1u:2u;if(t>=eyes*p.width*p.height)return;uint target;float z;if(temporal_project(p,t,true,target,z))InterlockedMin(((uint*)p.zbuffer)[target],asuint(z));}
void temporal_splat_owner(TemporalParameters p,uint t){uint eyes=(p.capacity&128u)!=0?1u:2u;if(t>=eyes*p.width*p.height)return;uint target;float z;if(temporal_project(p,t,false,target,z)&&((uint*)p.zbuffer)[target]==asuint(z))InterlockedMin(((uint*)p.owners)[target],t);}
void temporal_fuse(TemporalParameters p,uint t){
    uint n=p.width*p.height,eyes=(p.capacity&128u)!=0?1u:2u;if(t>=eyes*n)return;
    uint cap=p.capacity&7u;bool inv_depth=(p.capacity&16u)!=0,disocc_guard=(p.capacity&32u)!=0,stereo_research=(p.capacity&64u)!=0,ext_mode=(p.capacity&0xf0u)!=0;
    float depth=asfloat(0x7fc00000),confidence=0;uint valid=0,provenance=0,count=0,reuse=0,origin=0;uint2 selected=uint2(0xffffffff),age=uint2(0),oldest=uint2(0);uint2 ids[4];for(uint i=0;i<4;++i)ids[i]=uint2(0xffffffff);
    bool current=((uint*)p.fresh[2])[t]!=0&&((uint*)p.support[t/n])[t%n]!=0;
    if(current){float z=((float*)p.fresh[0])[t];float c=((float*)p.fresh[1])[t];uint prov=((uint*)p.fresh[3])[t];uint2 sid=((uint2*)p.fresh[4])[t],a=((uint2*)p.fresh[5])[t];
        bool ok_prov=(prov==1||prov==5)||(ext_mode&&prov==6);
        if(!temporal_normal(z)){temporal_counter(p,0);current=false;}
        else if(!isfinite(c)||c<0||c>1||!ok_prov||(sid.y&0x80000000)!=0||(a.y&0x80000000)!=0){temporal_counter(p,3);current=false;}
        else{depth=z;confidence=c;valid=1;provenance=prov;selected=sid;age=a;oldest=a;ids[0]=sid;count=1;origin=1u<<prov;}
    }
    uint owner=((uint*)p.owners)[t];if(owner!=0xffffffff){temporal_counter(p,7);float hz=asfloat(((uint*)p.zbuffer)[t]);float hc=((float*)p.history[1])[owner];uint hcount=((uint*)p.history[7])[owner];uint hre=((uint*)p.history[9])[owner]+1;uint2 ha,ho;temporal_add_age(((uint2*)p.history[5])[owner],p.elapsed,ha);temporal_add_age(((uint2*)p.history[8])[owner],p.elapsed,ho);
        if(!current&&stereo_research){
            uint64_t cen_addr=uint64_t(asuint(p.transforms[12]))|(uint64_t(asuint(p.transforms[13]))<<32);
            uint64_t cval_addr=uint64_t(asuint(p.transforms[14]))|(uint64_t(asuint(p.transforms[15]))<<32);
            float fxb=p.transforms[28],cx_off=p.transforms[29];
            uint eye=t/n,loc=t%n,x=loc%p.width,y=loc/p.width;
            if(cen_addr!=0&&cval_addr!=0&&((uint*)p.support[eye])[loc]!=0&&fxb>0&&hz>0){
                uint* cen=(uint*)cen_addr;uint* cval=(uint*)cval_addr;
                float d_pred=(fxb/hz)+cx_off;
                if(isfinite(d_pred)&&d_pred>=1.f&&d_pred<127.f&&cval[t]!=0){
                    int c_d=int(floor(d_pred+.5f));
                    uint lo=uint(max(1,c_d-2)),hi=uint(min(126,c_d+2));
                    uint c0=cen[t],other_row=(1u-eye)*n+y*p.width;
                    uint best=0xffffffff,second=0xffffffff,win=0xffffffff,minus=0xffffffff,plus=0xffffffff,prev_c=0xffffffff;
                    for(uint d=lo;d<=hi;++d){
                        uint cost=0xffffffff;int xx=int(x)+(eye!=0?int(d):-int(d));
                        if(xx>=0&&xx<int(p.width)&&cval[other_row+uint(xx)]!=0){
                            cost=countbits(c0^cen[other_row+uint(xx)]);
                            if(cost<best){second=best;best=cost;win=d;minus=prev_c;plus=0xffffffff;}
                            else if(cost<second){second=cost;}
                        }
                        if(win!=0xffffffff&&d==win+1)plus=cost;
                        prev_c=cost;
                    }
                    if(win!=0xffffffff&&best<=10u&&second>=best+2u){
                        int rx=int(x)+(eye!=0?int(win):-int(win));
                        uint r_idx=other_row+uint(rx),r_cen=cen[r_idx],own_row=eye*n+y*p.width;
                        uint r_best=0xffffffff,r_win=0xffffffff;
                        uint r_lo=uint(max(1,int(win)-2)),r_hi=uint(min(126,int(win)+2));
                        for(uint rd=r_lo;rd<=r_hi;++rd){
                            int lx=rx+(eye!=0?-int(rd):int(rd));
                            if(lx>=0&&lx<int(p.width)&&cval[own_row+uint(lx)]!=0){
                                uint rc=countbits(r_cen^cen[own_row+uint(lx)]);
                                if(rc<r_best){r_best=rc;r_win=rd;}
                            }
                        }
                        if(r_win!=0xffffffff&&abs(int(win)-int(r_win))<=1){
                            float d_res=float(win);
                            if(minus!=0xffffffff&&plus!=0xffffffff){
                                float denom=(float(minus)-2.f*float(best))+float(plus);
                                if(denom>0)d_res+=clamp(.5f*(float(minus)-float(plus))/denom,-.75f,.75f);
                            }
                            if(d_res-cx_off>0){
                                float z_res=fxb/(d_res-cx_off);
                                if(temporal_normal(z_res)&&abs(z_res-hz)<=2.f*p.tolerance*hz){
                                    float c_res=clamp(float(second-best)/float(max(second,1u)),p.confidence_floor,1.f);
                                    uint2 cur_sid=uint2(asuint(p.transforms[30]),asuint(p.transforms[31]));
                                    current=true;depth=z_res;confidence=c_res;valid=1;provenance=1;
                                    selected=cur_sid;age=uint2(0,0);oldest=uint2(0,0);ids[0]=cur_sid;count=1;origin=2u;
                                }
                            }
                        }
                    }
                }
            }
        }
        if(!current){
            bool allow_fill=true;
            float hc_fill=hc;
            if(disocc_guard){
                float age_ratio=p.max_age.x>0?clamp(float(ha.x)/float(p.max_age.x),0.f,1.f):0.f;
                hc_fill=hc*(1.f-.35f*age_ratio);
                if(hc_fill<p.confidence_floor)allow_fill=false;
                uint eye=t/n,loc=t%n,x=loc%p.width,y=loc/p.width,row_base=eye*n;
                uint fg_closer=0;
                for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
                    int nx=int(x)+dx,ny=int(y)+dy;
                    if(nx>=0&&nx<int(p.width)&&ny>=0&&ny<int(p.height)){
                        uint nt=row_base+uint(ny)*p.width+uint(nx);
                        if(((uint*)p.fresh[2])[nt]!=0){
                            float nz=((float*)p.fresh[0])[nt];
                            if(temporal_normal(nz)&&nz<hz*(1.f-2.f*p.tolerance))++fg_closer;
                        }
                    }
                }
                if(fg_closer>=2u)allow_fill=false;
            }
            if(allow_fill){
                depth=hz;confidence=hc_fill;valid=1;provenance=((uint*)p.history[3])[owner];if(provenance==1)provenance=2;if(provenance==6)provenance=7;selected=((uint2*)p.history[4])[owner];age=ha;oldest=ho;count=hcount;reuse=hre;origin=((uint*)p.history[10])[owner];for(uint i=0;i<4;++i)ids[i]=((uint2*)p.history[6])[owner*4+i];temporal_counter(p,8);
            }
        }
        else if(confidence>=p.confidence_floor&&hc>=p.confidence_floor&&abs(hz-depth)<=p.tolerance*depth){
            bool disjoint=true;for(uint i=0;i<hcount&&i<4;++i)if(temporal_equal(selected,((uint2*)p.history[6])[owner*4+i]))disjoint=false;
            if(disjoint&&hcount>0&&hcount<4&&hcount+1<=cap){float sum=confidence+hc;
                if(sum>0&&isfinite(sum)){float low=min(depth,hz),high=max(depth,hz);
                    float fused=low+(high-low)*((depth>=hz?confidence:hc)/sum);
                    if(inv_depth){
                        float age_ratio=p.max_age.x>0?clamp(float(ha.x)/float(p.max_age.x),0.f,1.f):0.f;
                        float decay=1.f-.25f*age_ratio;
                        float w_c=confidence/max(depth*depth,1e-4f);
                        float w_h=(hc*decay*min(sqrt(float(hcount)),2.f))/max(hz*hz,1e-4f);
                        float w_sum=w_c+w_h;
                        if(w_sum>0&&isfinite(w_sum)){
                            float rho=(w_c*(1.f/depth)+w_h*(1.f/hz))/w_sum;
                            if(rho>0&&isfinite(rho))fused=1.f/rho;
                        }
                    }
                    if(isfinite(fused))fused=clamp(fused,low,high);
                    if(temporal_normal(fused)){depth=fused;confidence=max(confidence,hc);provenance=(provenance==6||((uint*)p.history[3])[owner]==6||((uint*)p.history[3])[owner]==7)?7:3;oldest=temporal_less(oldest,ho)?ho:oldest;reuse=hre;origin|=((uint*)p.history[10])[owner];count=hcount+1;
                        uint j=0;bool inserted=false;for(uint i=0;i<count;++i){uint2 next=j<hcount?((uint2*)p.history[6])[owner*4+j]:uint2(0xffffffff);if(!inserted&&temporal_less(selected,next)){ids[i]=selected;inserted=true;}else{ids[i]=next;++j;}}temporal_counter(p,9);
                    }else temporal_counter(p,3);
                }else if(!isfinite(sum))temporal_counter(p,3);
            }
        }
    }
    ((float*)p.output[0])[t]=depth;((float*)p.output[1])[t]=confidence;((uint*)p.output[2])[t]=valid;((uint*)p.output[3])[t]=provenance;((uint2*)p.output[4])[t]=selected;((uint2*)p.output[5])[t]=age;
    for(uint i=0;i<4;++i)((uint2*)p.output[6])[t*4+i]=ids[i];((uint*)p.output[7])[t]=count;((uint2*)p.output[8])[t]=oldest;((uint*)p.output[9])[t]=reuse;((uint*)p.output[10])[t]=origin;
}
