#include "../ops/preprocess.slang"
#include "../ops/refinement.slang"
#include "../ops/stereo_observation.slang"
struct Parameters {
    uint64_t original[2];uint64_t coarse[3];uint64_t maps[4];uint64_t output[30];
    uint width;uint height;uint source_width;uint source_height;uint coarse_width;uint coarse_height;
    uint radius;uint disparities;uint min_margin;uint subpixel;
    float prior_floor;float max_span;float lr_tolerance;float fx;float baseline;float principal_offset;
    uint source_low;uint source_high;uint64_t borrowed[4];
    uint64_t completed[7];
    uint planar_cfg;float planar_coarse_lr_tol;float planar_patch_luma_tol;float planar_max_rmse;
};
static const uint SF_PROVENANCE_STEREO = 1u;
static const uint SF_PROVENANCE_PLANAR_INFERRED = 6u;
struct Push{uint64_t parameters;};[[vk::push_constant]] Push pc;
#define p (*(Parameters*)pc.parameters)
uint* uplane(uint plane){return (uint*)p.output[plane];}float* fplane(uint plane){return (float*)p.output[plane];}
[shader("compute")][numthreads(64,1,1)]
void rectify(uint3 tid:SV_DispatchThreadID){uint n=p.width*p.height,t=tid.x;if(t>=2*n)return;uint eye=t/n,i=t%n;float* mx=(float*)p.maps[eye*2];float* my=(float*)p.maps[eye*2+1];int qx=preprocess_quantize(mx[i]),qy=preprocess_quantize(my[i]);int x=qx>>5,y=qy>>5;uint fx=uint(qx)&31,fy=uint(qy)&31;uint weights[4]={(32-fx)*(32-fy),fx*(32-fy),(32-fx)*fy,fx*fy};uint3 channels=0;bool full_support=true;uint* original=(uint*)p.original[eye];[unroll]for(uint tap=0;tap<4;++tap){int xx=x+int(tap&1),yy=y+int(tap>>1);bool inside=xx>=0&&yy>=0&&xx<int(p.source_width)&&yy<int(p.source_height);if(weights[tap]!=0&&!inside)full_support=false;if(inside){uint rgb=original[uint(yy)*p.source_width+uint(xx)];channels+=uint3(rgb&255,(rgb>>8)&255,(rgb>>16)&255)*weights[tap];}}channels=(channels+512)>>10;uint color=channels.x|(channels.y<<8)|(channels.z<<16)|0xff000000u;uplane(eye==0?26:29)[i]=color;uplane(27)[t]=preprocess_gray(color);uplane(28)[t]=full_support?1:0;}
[shader("compute")][numthreads(64,1,1)]
void census(uint3 tid:SV_DispatchThreadID){uint n=p.width*p.height,t=tid.x;if(t>=2*n)return;uint eye=t/n,i=t%n,x=i%p.width,y=i/p.width;uint* gray=p.borrowed[0]!=0?(uint*)p.borrowed[eye]:uplane(27)+eye*n;uint* support=p.borrowed[0]!=0?(uint*)p.borrowed[2+eye]:uplane(28)+eye*n;bool valid=refinement_support(support,p.width,p.height,x,y);uplane(24)[t]=valid?(((p.subpixel&2u)!=0)?refinement_ortho_thermo_census(gray,p.width,x,y):refinement_census(gray,p.width,x,y)):0;uplane(25)[t]=uint(valid);}
[shader("compute")][numthreads(64,1,1)]
void match(uint3 tid:SV_DispatchThreadID){uint n=p.width*p.height,t=tid.x;if(t>=2*n)return;uint eye=t/n,i=t%n,x=i%p.width,y=i/p.width,cn=p.coarse_width*p.coarse_height;
    // Half-pixel 2x/4x coordinates are exact binary32 fractions.
    bool scale4=(p.width==p.coarse_width*4u),hybrid=(p.subpixel&2u)!=0,skip_diag=(p.subpixel&4u)!=0,repair=(p.subpixel&8u)!=0;
    precise float step=scale4?.25f:.5f,mult=scale4?4.f:2.f;
    precise float cx=(float(x)+.5f)*step-.5f,cy=(float(y)+.5f)*step-.5f;int ix=int(floor(cx)),iy=int(floor(cy));precise float wx=cx-float(ix),wy=cy-float(iy);precise float weights[4]={(1.f-wx)*(1.f-wy),wx*(1.f-wy),(1.f-wx)*wy,wx*wy};precise float predicted=0,w_valid=0;float mn=3.402823466e38f,mx=-3.402823466e38f;uint reason=0;bool boundary_repair=false,hole_fallback=false,expanded_sweep=false;
    if(!hybrid){
        [unroll]for(uint tap=0;tap<4;++tap){if(weights[tap]==0)continue;int xx=ix+int(tap&1),yy=iy+int(tap>>1);if(xx<0||yy<0||xx>=int(p.coarse_width)||yy>=int(p.coarse_height)){reason|=REF_OUTSIDE;continue;}uint j=eye*cn+uint(yy)*p.coarse_width+uint(xx);float d=((float*)p.coarse[0])[j],confidence=((float*)p.coarse[1])[j];if(((uint*)p.coarse[2])[j]==0)reason|=REF_NO_PRIOR;if(!isfinite(d))reason|=REF_NONFINITE;if(confidence<p.prior_floor)reason|=REF_LOW_CONFIDENCE;if(isfinite(d)){if(d<0||d>float(p.disparities-1)*step)reason|=REF_OUTSIDE;mn=min(mn,d);mx=max(mx,d);predicted=predicted+d*weights[tap];}}
        if(mx-mn>p.max_span)reason|=REF_PRIOR_SPAN;
    }else{
        [unroll]for(uint tap=0;tap<4;++tap){int xx=ix+int(tap&1),yy=iy+int(tap>>1);if(xx<0||yy<0||xx>=int(p.coarse_width)||yy>=int(p.coarse_height))continue;uint j=eye*cn+uint(yy)*p.coarse_width+uint(xx);float d=((float*)p.coarse[0])[j];if(((uint*)p.coarse[2])[j]!=0&&isfinite(d)&&d>=0&&d<=float(p.disparities-1)*step){precise float w=weights[tap]+.001f;predicted=predicted+d*w;w_valid=w_valid+w;mn=min(mn,d);mx=max(mx,d);}}
        int rx=int(floor(cx+.5f)),ry=int(floor(cy+.5f));
        if(w_valid==0.f){[unroll]for(int dy=-1;dy<=1;++dy)[unroll]for(int dx=-1;dx<=1;++dx){int xx=rx+dx,yy=ry+dy;if(xx<0||yy<0||xx>=int(p.coarse_width)||yy>=int(p.coarse_height))continue;uint j=eye*cn+uint(yy)*p.coarse_width+uint(xx);float d=((float*)p.coarse[0])[j];if(((uint*)p.coarse[2])[j]!=0&&isfinite(d)&&d>=0&&d<=float(p.disparities-1)*step){predicted=predicted+d;w_valid=w_valid+1.f;mn=min(mn,d);mx=max(mx,d);}}}
        if(repair&&w_valid==0.f){
            [unroll]for(int dy=-2;dy<=2;dy+=2)[unroll]for(int dx=-2;dx<=2;dx+=2){int xx=rx+dx,yy=ry+dy;if(xx<0||yy<0||xx>=int(p.coarse_width)||yy>=int(p.coarse_height))continue;uint j=eye*cn+uint(yy)*p.coarse_width+uint(xx);float d=((float*)p.coarse[0])[j];if(((uint*)p.coarse[2])[j]!=0&&isfinite(d)&&d>=0&&d<=float(p.disparities-1)*step){predicted=predicted+d;w_valid=w_valid+1.f;mn=min(mn,d);mx=max(mx,d);}}
            if(w_valid<3.f||(mx-mn)>p.max_span*1.5f)w_valid=0.f;
            else hole_fallback=true;
        }
        if(w_valid>0.f){
            predicted=predicted/w_valid;
            float span_limit=repair?(p.max_span*3.0f):(p.max_span*2.5f);
            if(mx-mn>span_limit)reason|=REF_PRIOR_SPAN;
            else if(repair&&!hole_fallback&&(mx-mn>p.max_span*2.5f))boundary_repair=true;
        }else reason|=REF_NO_PRIOR;
    }
    bool prior=reason==0;predicted=predicted*mult;if(!skip_diag){fplane(9)[t]=prior?predicted:refinement_invalid();uplane(10)[t]=uint(prior);}
    int centre=0;uint lo=0,hi=0xffffffff;
    if(prior){
        centre=int(floor(predicted+.5f));
        int rad=int(p.radius);
        if(boundary_repair){
            int lo_b=max(int(floor(mn*mult))-1,centre-4);
            int hi_b=min(int(floor(mx*mult+1.f))+1,centre+4);
            lo=uint(clamp(lo_b,0,int(p.disparities)-1));
            hi=uint(clamp(hi_b,int(lo),int(p.disparities)-1));
        }else{
            lo=uint(max(0,centre-rad));
            hi=uint(min(int(p.disparities)-1,centre+rad));
        }
    }
    if(!skip_diag){uplane(11)[t]=lo;uplane(12)[t]=hi;}
    uint best=0xffffffff,second=0xffffffff,third=0xffffffff,winner=0xffffffff,count=0,work=0,minus=0xffffffff,plus=0xffffffff;
    uint prev_cost=0xffffffff;bool supported=uplane(25)[t]!=0;if(prior&&!supported)reason|=REF_UNSUPPORTED;
    bool plateau_tie=false,strict_reject=false;
    if(prior&&supported){
        uint c0=uplane(24)[t];
        uint c_l=(x>0&&uplane(25)[t-1]!=0)?uplane(24)[t-1]:c0;
        uint c_r=(x+1<p.width&&uplane(25)[t+1]!=0)?uplane(24)[t+1]:c0;
        uint c_u=(y>0&&uplane(25)[t-p.width]!=0)?uplane(24)[t-p.width]:c0;
        uint c_d=(y+1<p.height&&uplane(25)[t+p.width]!=0)?uplane(24)[t+p.width]:c0;
        for(uint pass_idx=0;pass_idx<2;++pass_idx){
            for(uint d=lo;d<=hi;++d){uint cost=0xffffffff;int xx=int(x)+(eye!=0?int(d):-int(d));if(xx>=0&&xx<int(p.width)){uint j=(1-eye)*n+y*p.width+uint(xx);if(uplane(25)[j]!=0){
                uint base=countbits(c0^uplane(24)[j]);
                if(!hybrid){cost=base;}else{
                    uint m_l=(xx>0&&uplane(25)[j-1]!=0)?countbits(c_l^uplane(24)[j-1]):base;
                    uint m_r=(xx+1<int(p.width)&&uplane(25)[j+1]!=0)?countbits(c_r^uplane(24)[j+1]):base;
                    uint m_u=(y>0&&uplane(25)[j-p.width]!=0)?countbits(c_u^uplane(24)[j-p.width]):base;
                    uint m_d=(y+1<p.height&&uplane(25)[j+p.width]!=0)?countbits(c_d^uplane(24)[j+p.width]):base;
                    uint dist=uint(abs(int(d)-centre));
                    cost=2u*base+m_l+m_r+m_u+m_d+dist;
                }
                ++count;++work;
                if(cost<best){third=second;second=best;best=cost;winner=d;minus=prev_cost;plus=0xffffffff;}
                else if(cost<second){third=second;second=cost;}
                else if(cost<third){third=cost;}
            }}if(winner!=0xffffffff&&d==winner+1)plus=cost;prev_cost=cost;}
            if(pass_idx==0&&repair&&!boundary_repair&&!hole_fallback&&second<best+p.min_margin&&best>32u){
                uint lo2=uint(max(0,int(lo)-2)),hi2=uint(min(int(p.disparities)-1,int(hi)+2));
                if(lo2<lo||hi2>hi){
                    lo=lo2;hi=hi2;expanded_sweep=true;
                    best=0xffffffff;second=0xffffffff;third=0xffffffff;winner=0xffffffff;count=0;work=0;minus=0xffffffff;plus=0xffffffff;prev_cost=0xffffffff;
                    if(!skip_diag){uplane(11)[t]=lo;uplane(12)[t]=hi;}
                    continue;
                }
            }
            break;
        }
        plateau_tie=repair&&!hole_fallback&&!expanded_sweep&&!boundary_repair&&count>=3&&second==best&&best<=14u&&(plus==best||minus==best)&&third>=best+max(p.min_margin+2u,3u);
        strict_reject=(hole_fallback||boundary_repair||expanded_sweep)&&(best>16u||second<best+p.min_margin+1u);
    }
    uint eff_margin=plateau_tie?p.min_margin:(second-best);
    uint eff_second=plateau_tie?(best+p.min_margin):second;
    bool unique=count>=2&&eff_margin>=p.min_margin&&!strict_reject;if(count<2)reason|=REF_TOO_FEW;if(count>=2&&!unique)reason|=REF_AMBIGUOUS;bool restricted=!hybrid&&count>0&&((winner==lo&&lo>0)||(winner==hi&&hi+1<p.disparities));if(restricted)reason|=REF_RESTRICTED;if(count>0&&(winner==0||winner+1==p.disparities))reason|=REF_GLOBAL;bool valid=unique&&!restricted;
    float d=float(winner),confidence=0;bool applied=false;if(valid){confidence=clamp(float(eff_margin)/float(max(eff_second,1u)),0.f,1.f);if((p.subpixel&1u)!=0){if(minus!=0xffffffff&&plus!=0xffffffff){precise float denominator=(float(minus)-2.f*float(best))+float(plus);precise float numerator=.5f*(float(minus)-float(plus));precise float offset=denominator!=0?clamp(numerator/denominator,-1.f,1.f):0;d=d+offset;applied=true;}else reason|=REF_MISSING_SUBPIXEL;}}
    if(!skip_diag){uplane(13)[t]=reason;uplane(14)[t]=prior?1:0;uplane(15)[t]=winner;uplane(16)[t]=best;uplane(17)[t]=second;uplane(18)[t]=count;uplane(19)[t]=work;uplane(20)[t]=uint(applied);}fplane(21)[t]=valid?d:refinement_invalid();fplane(22)[t]=valid?confidence:0;uplane(23)[t]=uint(valid);
}
[shader("compute")][numthreads(64,1,1)]
void lr(uint3 tid:SV_DispatchThreadID){uint n=p.width*p.height,t=tid.x;if(t>=2*n)return;uint eye=t/n,i=t%n,target=0;float d=fplane(21)[t];bool valid=false;if(uplane(23)[t]!=0&&refinement_target(p.width,i%p.width,d,eye!=0,target)){uint j=(1-eye)*n+(i/p.width)*p.width+target;valid=observation_lr(d,fplane(21)[j],uplane(23)[t],uplane(23)[j],p.lr_tolerance);}if((p.subpixel&4u)==0&&uplane(23)[t]!=0&&!valid)uplane(13)[t]|=REF_LR_FAILED;fplane(0)[t]=valid?d:refinement_invalid();fplane(1)[t]=valid?fplane(22)[t]:0;uplane(2)[t]=uint(valid);}
void write_completed(uint t,float z,float c,uint v,uint prov,uint2 sid,float d){
    ((float*)p.completed[0])[t]=z;((float*)p.completed[1])[t]=c;((uint*)p.completed[2])[t]=v;((uint*)p.completed[3])[t]=prov;
    ((uint2*)p.completed[4])[t]=sid;((float*)p.completed[6])[t]=d;
}
[shader("compute")][numthreads(64,1,1)]
void metric(uint3 tid:SV_DispatchThreadID){uint n=p.width*p.height,t=tid.x;if(t>=2*n)return;uint eye=t/n,i=t%n,target=0;float z=refinement_invalid(),d=fplane(0)[t];bool valid=uplane(2)[t]!=0&&d>0&&(d-p.principal_offset)>0;if(valid&&refinement_target(p.width,i%p.width,d,eye!=0,target)){uint j=(1-eye)*n+(i/p.width)*p.width+target;valid=uplane(25)[t]!=0&&uplane(25)[j]!=0;if(valid){precise float numerator=p.fx*p.baseline;precise float denominator=d-p.principal_offset;z=numerator/denominator;valid=isfinite(z)&&z>0;}}else valid=false;float z_out=valid?z:refinement_invalid(),c_out=valid?fplane(1)[t]:0;uint v_out=uint(valid),prov_out=valid?SF_PROVENANCE_STEREO:0u;uint2 sid=valid?uint2(p.source_low,p.source_high):uint2(0xffffffff,0xffffffff);fplane(3)[t]=z_out;fplane(4)[t]=c_out;uplane(5)[t]=v_out;uplane(6)[t]=prov_out;((uint2*)p.output[7])[t]=sid;if(p.completed[0]!=0)write_completed(t,z_out,c_out,v_out,prov_out,sid,valid?d:refinement_invalid());}
groupshared uint gs_has_candidate;
groupshared uint gs_has_valid_coarse;
groupshared float gs_disp[34][34];
groupshared float gs_luma[34][34];
groupshared float gs_coarse_d[6][6];
groupshared float4 gs_coarse_plane[4][4];
[shader("compute")][numthreads(16,16,1)]
void planar(uint3 tid:SV_DispatchThreadID,uint3 gid:SV_GroupID,uint3 gtid:SV_GroupThreadID,uint gidx:SV_GroupIndex){
    if(p.completed[0]==0)return;
    if(gidx==0u){gs_has_candidate=0u;gs_has_valid_coarse=0u;}
    GroupMemoryBarrierWithGroupSync();
    uint x=tid.x,y=tid.y,n=p.width*p.height,t=y*p.width+x;
    bool in_bounds=(x>=12u&&x+12u<p.width&&y>=12u&&y+12u<p.height);
    bool is_cand=in_bounds&&(uplane(5)[t]==0u)&&(uplane(25)[t]!=0u);
    if(is_cand){
        uint qpre=((uplane(2)[(y-9u)*p.width+(x-9u)]|uplane(2)[(y-5u)*p.width+(x-5u)])!=0?1u:0u)
                 |((uplane(2)[(y-9u)*p.width+(x+9u)]|uplane(2)[(y-5u)*p.width+(x+5u)])!=0?2u:0u)
                 |((uplane(2)[(y+9u)*p.width+(x-9u)]|uplane(2)[(y+5u)*p.width+(x-5u)])!=0?4u:0u)
                 |((uplane(2)[(y+9u)*p.width+(x+9u)]|uplane(2)[(y+5u)*p.width+(x+5u)])!=0?8u:0u);
        if(countbits(qpre)<2u)is_cand=false;
    }
    if(is_cand)InterlockedOr(gs_has_candidate,1u);
    GroupMemoryBarrierWithGroupSync();
    if(gs_has_candidate==0u)return;
    bool scale4=(p.width==p.coarse_width*4u);
    bool use_coarse_affine=(p.planar_cfg&256u)!=0;
    precise float step=scale4?.25f:.5f,mult=scale4?4.f:2.f;
    if(scale4&&use_coarse_affine){
        if(gidx<36u){
            uint cy_i=gidx/6u,cx_i=gidx%6u;
            int rcx0=clamp(int(gid.x*4u)+int(cx_i)-1,0,int(p.coarse_width)-1);
            int rcy0=clamp(int(gid.y*4u)+int(cy_i)-1,0,int(p.coarse_height)-1);
            uint cj=uint(rcy0)*p.coarse_width+uint(rcx0);
            float cd=((float*)p.coarse[0])[cj];
            bool ok=(((uint*)p.coarse[2])[cj]!=0)&&isfinite(cd)&&cd>0.f&&cd<=float(p.disparities-1)*step;
            gs_coarse_d[cy_i][cx_i]=ok?cd:-1.f;
        }
        GroupMemoryBarrierWithGroupSync();
        if(gidx<16u){
            uint py=gidx/4u,px=gidx%4u;
            uint c_cnt=0;
            float c_mn=3.402823466e38f,c_mx=-3.402823466e38f;
            float csx=0,csy=0,csxx=0,csyy=0,csxy=0,csd=0,csxd=0,csyd=0,csdd=0;
            for(int cdy=-1;cdy<=1;++cdy){
                float fcdy=float(cdy);
                for(int cdx=-1;cdx<=1;++cdx){
                    float cd=gs_coarse_d[int(py)+1+cdy][int(px)+1+cdx];
                    if(cd<=0.f)continue;
                    float fcdx=float(cdx);
                    ++c_cnt;c_mn=min(c_mn,cd);c_mx=max(c_mx,cd);
                    csx+=fcdx;csy+=fcdy;csxx+=fcdx*fcdx;csyy+=fcdy*fcdy;csxy+=fcdx*fcdy;
                    csd+=cd;csxd+=fcdx*cd;csyd+=fcdy*cd;csdd+=cd*cd;
                }
            }
            float4 plane=float4(0,0,0,-1.f);
            if(c_cnt>=5u&&(c_mx-c_mn)<=p.max_span*1.25f){
                float fcn=float(c_cnt),inv_cn=1.f/fcn;
                float cmx=csx*inv_cn,cmy=csy*inv_cn,cmd=csd*inv_cn;
                float ccxx=csxx-fcn*cmx*cmx,ccyy=csyy-fcn*cmy*cmy,ccxy=csxy-fcn*cmx*cmy;
                float cdet=ccxx*ccyy-ccxy*ccxy;
                float ac=0.f,bc=0.f;
                bool ok_fit=true;
                if(cdet>.25f){
                    float ccxd=csxd-fcn*cmx*cmd,ccyd=csyd-fcn*cmy*cmd,ccdd=csdd-fcn*cmd*cmd;
                    ac=(ccyy*ccxd-ccxy*ccyd)/cdet;
                    bc=(ccxx*ccyd-ccxy*ccxd)/cdet;
                    float cssr=max(0.f,ccdd-ac*ccxd-bc*ccyd);
                    if(abs(ac)>.60f||abs(bc)>.85f||sqrt(cssr*inv_cn)>.55f)ok_fit=false;
                }
                if(ok_fit)plane=float4(cmd-ac*cmx-bc*cmy,ac,bc,1.f);
            }
            gs_coarse_plane[py][px]=plane;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    uint min_anc=max(p.planar_cfg&0xffu,1u);
    bool ray_guard=(p.planar_cfg&512u)!=0;
    bool coarse_lr_guard=(p.planar_cfg&1024u)!=0;
    precise float cx=(float(x)+.5f)*step-.5f,cy=(float(y)+.5f)*step-.5f;
    uint cn=p.coarse_width*p.coarse_height;
    float d_coarse=0.f,ac_full=0.f,bc_full=0.f;
    if(is_cand){
        if(scale4&&use_coarse_affine){
            uint px=gtid.x>>2u,py=gtid.y>>2u;
            float4 plane=gs_coarse_plane[py][px];
            if(plane.w<0.f)is_cand=false;
            else{
                int rcx=clamp(int(gid.x*4u+px),1,int(p.coarse_width)-2);
                int rcy=clamp(int(gid.y*4u+py),1,int(p.coarse_height)-2);
                ac_full=plane.y;bc_full=plane.z;
                d_coarse=(plane.x+ac_full*(cx-float(rcx))+bc_full*(cy-float(rcy)))*mult;
            }
        }else if(use_coarse_affine){
            int rcx=clamp(int(floor(cx+.5f)),1,int(p.coarse_width)-2);
            int rcy=clamp(int(floor(cy+.5f)),1,int(p.coarse_height)-2);
            uint c_cnt=0;
            float c_mn=3.402823466e38f,c_mx=-3.402823466e38f;
            float csx=0,csy=0,csxx=0,csyy=0,csxy=0,csd=0,csxd=0,csyd=0,csdd=0;
            for(int cdy=-1;cdy<=1;++cdy){
                uint crow=uint(rcy+cdy)*p.coarse_width;
                float fcdy=float(cdy);
                for(int cdx=-1;cdx<=1;++cdx){
                    uint cj=crow+uint(rcx+cdx);
                    if(((uint*)p.coarse[2])[cj]==0)continue;
                    float cd=((float*)p.coarse[0])[cj];
                    if(!isfinite(cd)||cd<=0.f||cd>float(p.disparities-1)*step)continue;
                    float fcdx=float(cdx);
                    ++c_cnt;c_mn=min(c_mn,cd);c_mx=max(c_mx,cd);
                    csx+=fcdx;csy+=fcdy;csxx+=fcdx*fcdx;csyy+=fcdy*fcdy;csxy+=fcdx*fcdy;
                    csd+=cd;csxd+=fcdx*cd;csyd+=fcdy*cd;csdd+=cd*cd;
                }
            }
            if(c_cnt<5u||(c_mx-c_mn)>p.max_span*1.25f)is_cand=false;
            else{
                float fcn=float(c_cnt),inv_cn=1.f/fcn;
                float cmx=csx*inv_cn,cmy=csy*inv_cn,cmd=csd*inv_cn;
                float ccxx=csxx-fcn*cmx*cmx,ccyy=csyy-fcn*cmy*cmy,ccxy=csxy-fcn*cmx*cmy;
                float cdet=ccxx*ccyy-ccxy*ccxy;
                float ac=0.f,bc=0.f;
                if(cdet>.25f){
                    float ccxd=csxd-fcn*cmx*cmd,ccyd=csyd-fcn*cmy*cmd,ccdd=csdd-fcn*cmd*cmd;
                    ac=(ccyy*ccxd-ccxy*ccyd)/cdet;
                    bc=(ccxx*ccyd-ccxy*ccxd)/cdet;
                    float cssr=max(0.f,ccdd-ac*ccxd-bc*ccyd);
                    if(abs(ac)>.60f||abs(bc)>.85f||sqrt(cssr*inv_cn)>.55f)is_cand=false;
                }
                if(is_cand){
                    d_coarse=(cmd+ac*(cx-float(rcx)-cmx)+bc*(cy-float(rcy)-cmy))*mult;
                    ac_full=ac;bc_full=bc;
                }
            }
        }else{
            int ix=int(floor(cx)),iy=int(floor(cy));
            precise float wx=cx-float(ix),wy=cy-float(iy);
            precise float weights[4]={(1.f-wx)*(1.f-wy),wx*(1.f-wy),(1.f-wx)*wy,wx*wy};
            float mn=3.402823466e38f,mx=-3.402823466e38f;
            precise float predicted=0,w_valid=0;
            for(uint tap=0;tap<4;++tap){
                int xx=ix+int(tap&1),yy=iy+int(tap>>1);
                if(xx<0||yy<0||xx>=int(p.coarse_width)||yy>=int(p.coarse_height))continue;
                uint j=uint(yy)*p.coarse_width+uint(xx);
                float d=((float*)p.coarse[0])[j];
                if(((uint*)p.coarse[2])[j]!=0&&isfinite(d)&&d>=0&&d<=float(p.disparities-1)*step){
                    precise float w=weights[tap]+.001f;predicted=predicted+d*w;w_valid=w_valid+w;mn=min(mn,d);mx=max(mx,d);
                }
            }
            if(w_valid<.70f||(mx-mn)>p.max_span*1.25f)is_cand=false;
            else d_coarse=(predicted/w_valid)*mult;
        }
        if(is_cand&&(d_coarse<1.5f||d_coarse>=float(p.disparities-1)))is_cand=false;
        // [P0] Reciprocal right-eye coarse & full-res disparity check before loading full-res halo or stencil traversal.
        if(is_cand&&coarse_lr_guard){
            float cx_r=cx-d_coarse*step;
            int ix_r=int(floor(cx_r+.5f)),iy_r=clamp(int(floor(cy+.5f)),0,int(p.coarse_height)-1);
            if(ix_r<0||ix_r>=int(p.coarse_width))is_cand=false;
            else{
                uint jr_c=cn+uint(iy_r)*p.coarse_width+uint(ix_r);
                if(((uint*)p.coarse[2])[jr_c]==0)is_cand=false;
                else{
                    float dr_c=((float*)p.coarse[0])[jr_c]*mult;
                    if(!isfinite(dr_c)||abs(dr_c-d_coarse)>p.planar_coarse_lr_tol)is_cand=false;
                    else{
                        int rx_c=int(floor(float(x)-d_coarse+.5f));
                        if(rx_c>=0&&rx_c<int(p.width)){
                            uint jr_f=n+y*p.width+uint(rx_c);
                            if(uplane(23)[jr_f]!=0&&abs(fplane(21)[jr_f]-d_coarse)>2.0f)is_cand=false;
                        }
                    }
                }
            }
        }
        if(is_cand)InterlockedOr(gs_has_valid_coarse,1u);
    }
    GroupMemoryBarrierWithGroupSync();
    if(gs_has_valid_coarse==0u)return;
    uint* gray_l=p.borrowed[0]!=0?(uint*)p.borrowed[0]:uplane(27);
    uint* gray_r=p.borrowed[0]!=0?(uint*)p.borrowed[1]:uplane(27)+n;
    int base_x=int(gid.x*16u)-9,base_y=int(gid.y*16u)-9;
    for(uint k=gidx;k<34u*34u;k+=256u){
        uint hy=k/34u,hx=k%34u;
        int gx=clamp(base_x+int(hx),0,int(p.width)-1);
        int gy=clamp(base_y+int(hy),0,int(p.height)-1);
        uint g_idx=uint(gy)*p.width+uint(gx);
        gs_luma[hy][hx]=float(gray_l[g_idx]);
        gs_disp[hy][hx]=(uplane(2)[g_idx]!=0)?fplane(0)[g_idx]:-1.f;
    }
    GroupMemoryBarrierWithGroupSync();
    if(!is_cand)return;
    uint lx=gtid.x+9u,ly=gtid.y+9u;
    float i0=gs_luma[ly][lx];
    float l_xm1=gs_luma[ly][lx-1u],l_xp1=gs_luma[ly][lx+1u],l_ym1=gs_luma[ly-1u][lx],l_yp1=gs_luma[ly+1u][lx];
    if(abs(l_xp1-l_xm1)+abs(l_yp1-l_ym1)>28.f)return;
    // Single-pass 36-tap quadrant stencil dy,dx in {-9,-5,-2,2,5,9} reading from LDS (gs_disp & gs_luma)
    // with coarse-slope-centered residual u_res = da - dc_a and Huber/Cauchy outlier weighting.
    static const int kOff[6]={-9,-5,-2,2,5,9};
    uint n_anc=0,qmask=0;
    float anc_mn=3.402823466e38f,anc_mx=-3.402823466e38f;
    float sw=0,sx=0,sy=0,sxx=0,syy=0,sxy=0,su=0,sxu=0,syu=0,suu=0;
    float dc_tol=(d_coarse>=114.f)?2.6f:1.6f;
    for(uint iy_i=0;iy_i<6u;++iy_i){
        int dy=kOff[iy_i];
        uint qy=(dy>0)?2u:0u;
        float fy_d=float(dy);
        for(uint ix_i=0;ix_i<6u;++ix_i){
            int dx=kOff[ix_i];
            float da=gs_disp[int(ly)+dy][int(lx)+dx];
            if(da<=0.f)continue;
            float ga=gs_luma[int(ly)+dy][int(lx)+dx];
            if(abs(ga-i0)>18.f)continue;
            float fx_d=float(dx);
            float dc_a=d_coarse+ac_full*fx_d+bc_full*fy_d;
            float u_res=da-dc_a;
            if(abs(u_res)>dc_tol)continue;
            if(ray_guard&&(abs(dx)>=5||abs(dy)>=5)){
                float gm=gs_luma[int(ly)+(dy/2)][int(lx)+(dx/2)];
                if(abs(gm-i0)>18.f&&abs(gm-ga)>18.f)continue;
            }
            float w_a=1.f/(1.f+1.77f*u_res*u_res);
            ++n_anc;
            anc_mn=min(anc_mn,da);anc_mx=max(anc_mx,da);
            qmask|=1u<<(qy+(dx>0?1u:0u));
            sw+=w_a;
            sx+=w_a*fx_d;sy+=w_a*fy_d;
            sxx+=w_a*fx_d*fx_d;syy+=w_a*fy_d*fy_d;sxy+=w_a*fx_d*fy_d;
            su+=w_a*u_res;sxu+=w_a*fx_d*u_res;syu+=w_a*fy_d*u_res;suu+=w_a*u_res*u_res;
        }
        if(iy_i==2u&&(qmask==0u||n_anc*3u<min_anc))return;
    }
    if(n_anc<min_anc||countbits(qmask)<3u||(anc_mx-anc_mn)>p.max_span*mult*1.65f||sw<=1e-3f)return;
    float fn=float(n_anc),inv_w=1.f/sw;
    float mx_a=sx*inv_w,my_a=sy*inv_w,mu_a=su*inv_w;
    float cxx=sxx-sw*mx_a*mx_a,cyy=syy-sw*my_a*my_a,cxy=sxy-sw*mx_a*my_a;
    float det=cxx*cyy-cxy*cxy;
    if(det<2.0f*sw*(cxx+cyy)||det<=.5f)return;
    float cxu=sxu-sw*mx_a*mu_a,cyu=syu-sw*my_a*mu_a,cuu=suu-sw*mu_a*mu_a;
    float da_slope=(cyy*cxu-cxy*cyu)/det;
    float db_slope=(cxx*cyu-cxy*cxu)/det;
    float a=ac_full+da_slope,b=bc_full+db_slope;
    if(abs(a)>.25f||abs(b)>.55f)return;
    float ssr=max(0.f,cuu-da_slope*cxu-db_slope*cyu);
    float rmse=sqrt(ssr*inv_w);
    float max_rmse=min(p.planar_max_rmse,clamp(.32f+.0045f*d_coarse,.36f,.75f));
    if(rmse>max_rmse)return;
    float u_pred=mu_a-da_slope*mx_a-db_slope*my_a;
    if(abs(u_pred)>(d_coarse>=114.f?2.4f:1.6f))return;
    float d_plane=d_coarse+u_pred;
    if(d_plane<=max(.5f,p.principal_offset+.5f)||d_plane>=float(p.disparities-1))return;
    int d_int=int(floor(d_plane+.5f));
    int rx=int(x)-d_int;
    float xr_f=float(x)-d_plane;
    int rx0=int(floor(xr_f));
    if(rx<1||rx+1>=int(p.width)||rx0<0||rx0+1>=int(p.width))return;
    uint r_idx=y*p.width+uint(rx),jr=n+r_idx;
    if(uplane(25)[jr]==0)return;
    if(coarse_lr_guard&&uplane(23)[jr]!=0&&abs(fplane(21)[jr]-d_plane)>1.8f)return;
    float wf=xr_f-float(rx0);
    float r_luma=(1.f-wf)*float(gray_r[y*p.width+uint(rx0)])+wf*float(gray_r[y*p.width+uint(rx0+1)]);
    if(abs(i0-r_luma)>18.f)return;
    float l_mean5=.2f*(i0+l_xm1+l_xp1+l_ym1+l_yp1);
    float r_mean5=.2f*(float(gray_r[r_idx])+float(gray_r[r_idx-1u])+float(gray_r[r_idx+1u])+float(gray_r[r_idx-p.width])+float(gray_r[r_idx+p.width]));
    if(abs(l_mean5-r_mean5)>p.planar_patch_luma_tol)return;
    uint c0=uplane(24)[t],base=countbits(c0^uplane(24)[jr]);
    uint m_l=(uplane(25)[t-1]!=0&&uplane(25)[jr-1]!=0)?countbits(uplane(24)[t-1]^uplane(24)[jr-1]):base;
    uint m_r=(uplane(25)[t+1]!=0&&uplane(25)[jr+1]!=0)?countbits(uplane(24)[t+1]^uplane(24)[jr+1]):base;
    uint m_u=(uplane(25)[t-p.width]!=0&&uplane(25)[jr-p.width]!=0)?countbits(uplane(24)[t-p.width]^uplane(24)[jr-p.width]):base;
    uint m_d=(uplane(25)[t+p.width]!=0&&uplane(25)[jr+p.width]!=0)?countbits(uplane(24)[t+p.width]^uplane(24)[jr+p.width]):base;
    uint cost_plane=2u*base+m_l+m_r+m_u+m_d;
    if(cost_plane>28u)return;
    precise float numerator=p.fx*p.baseline;
    precise float denominator=d_plane-p.principal_offset;
    float z=numerator/denominator;
    if(!isfinite(z)||z<=0.f)return;
    float conf=clamp((1.f-rmse/.55f)*(1.f-float(cost_plane)/28.f)*min(fn/16.f,1.f),.05f,.85f);
    write_completed(t,z,conf,1u,SF_PROVENANCE_PLANAR_INFERRED,uint2(p.source_low,p.source_high),d_plane);
}
