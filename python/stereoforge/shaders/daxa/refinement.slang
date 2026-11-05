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
static const uint SF_PROVENANCE_TEMPORAL_PLANAR_INFERRED = 7u;
struct Push{uint64_t parameters;};[[vk::push_constant]] Push pc;
#define p (*(Parameters*)pc.parameters)
uint* uplane(uint plane){return (uint*)p.output[plane];}float* fplane(uint plane){return (float*)p.output[plane];}
bool refinement_photometric_consistent(uint eye,uint x,uint y,uint disparity){
    int xr=int(x)+(eye!=0?int(disparity):-int(disparity));
    if(x<1u||x+1u>=p.width||y<1u||y+1u>=p.height||xr<1||xr+1>=int(p.width))return false;
    uint n=p.width*p.height;
    uint* own=p.borrowed[0]!=0?(uint*)p.borrowed[eye]:uplane(27)+eye*n;
    uint* other=p.borrowed[0]!=0?(uint*)p.borrowed[1u-eye]:uplane(27)+(1u-eye)*n;
    SF_PRECISE float sl=0.f,sr=0.f,sll=0.f,srr=0.f,slr=0.f;
    for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
        uint row=uint(int(y)+dy)*p.width;
        SF_PRECISE float l=float(own[row+uint(int(x)+dx)]),r=float(other[row+uint(xr+dx)]);
        sl=sl+l;sr=sr+r;sll=sll+l*l;srr=srr+r*r;slr=slr+l*r;
    }
    SF_PRECISE float vl=9.f*sll-sl*sl,vr=9.f*srr-sr*sr,cross=9.f*slr-sl*sr;
    // Positive 3x3 ZNCC>=.95; gain/offset invariant. A low Census cost alone
    // can falsely confirm the same wrong coarse prior in both eyes.
    return vl>1.f&&vr>1.f&&cross>0.f&&cross*cross>=.9025f*(vl*vr);
}
[shader("compute")][numthreads(64,1,1)]
void rectify(uint3 tid:SV_DispatchThreadID){uint n=p.width*p.height,t=tid.x;if(t>=2*n)return;uint eye=t/n,i=t%n;float* mx=(float*)p.maps[eye*2];float* my=(float*)p.maps[eye*2+1];int qx=preprocess_quantize(mx[i]),qy=preprocess_quantize(my[i]);int x=qx>>5,y=qy>>5;uint fx=uint(qx)&31,fy=uint(qy)&31;uint weights[4]={(32-fx)*(32-fy),fx*(32-fy),(32-fx)*fy,fx*fy};uint3 channels=0;bool full_support=true;uint* original=(uint*)p.original[eye];[unroll]for(uint tap=0;tap<4;++tap){int xx=x+int(tap&1),yy=y+int(tap>>1);bool inside=xx>=0&&yy>=0&&xx<int(p.source_width)&&yy<int(p.source_height);if(weights[tap]!=0&&!inside)full_support=false;if(inside){uint rgb=original[uint(yy)*p.source_width+uint(xx)];channels+=uint3(rgb&255,(rgb>>8)&255,(rgb>>16)&255)*weights[tap];}}channels=(channels+512)>>10;uint color=channels.x|(channels.y<<8)|(channels.z<<16)|0xff000000u;uplane(eye==0?26:29)[i]=color;uplane(27)[t]=preprocess_gray(color);uplane(28)[t]=full_support?1:0;}
#if !defined(SF_CPU_TARGET)
groupshared uint gs_census_supp[68];
groupshared int gs_census_sum5[68];
groupshared uint gs_census_top3[68];
groupshared uint gs_census_bot3[68];
#endif
[shader("compute")][numthreads(64,1,1)]
void census(uint3 tid:SV_DispatchThreadID,uint3 gid:SV_GroupID,uint gidx:SV_GroupIndex){
    uint n=p.width*p.height,t=tid.x;
#if !defined(SF_CPU_TARGET)
    uint base_t=gid.x*64u;
    uint eye0=base_t/n,rem0=base_t%n,x0=rem0%p.width,y0=rem0/p.width;
    if((p.subpixel&514u)==2u&&eye0<2u&&x0+64u<=p.width&&y0>=2u&&y0+2u<p.height){
        uint* gray=p.borrowed[0]!=0?(uint*)p.borrowed[eye0]:uplane(27)+eye0*n;
        uint* support=p.borrowed[0]!=0?(uint*)p.borrowed[2+eye0]:uplane(28)+eye0*n;
        uint row_m2=(y0-2u)*p.width,row_m1=(y0-1u)*p.width,row_0=y0*p.width,row_p1=(y0+1u)*p.width,row_p2=(y0+2u)*p.width;
        {
            uint hx=gidx+2u;
            uint ux=x0+gidx;
            uint s_ok=uint(support[row_m2+ux]!=0&&support[row_m1+ux]!=0&&support[row_0+ux]!=0&&support[row_p1+ux]!=0&&support[row_p2+ux]!=0);
            uint gm2=gray[row_m2+ux],gm1=gray[row_m1+ux],g0=gray[row_0+ux],gp1=gray[row_p1+ux],gp2=gray[row_p2+ux];
            gs_census_supp[hx]=s_ok;
            gs_census_top3[hx]=gm2+gm1+g0;
            gs_census_bot3[hx]=g0+gp1+gp2;
            gs_census_sum5[hx]=int(gm2+gm1+g0+gp1+gp2);
            if(gidx>=32u&&gidx<36u){
                uint h_side=gidx-32u;
                uint h_dst=(h_side<2u)?h_side:(h_side+64u);
                int gx_h=int(x0)+int(h_dst)-2;
                if(gx_h>=0&&gx_h<int(p.width)){
                    uint ux_h=uint(gx_h);
                    uint s_h=uint(support[row_m2+ux_h]!=0&&support[row_m1+ux_h]!=0&&support[row_0+ux_h]!=0&&support[row_p1+ux_h]!=0&&support[row_p2+ux_h]!=0);
                    uint hm2=gray[row_m2+ux_h],hm1=gray[row_m1+ux_h],h0=gray[row_0+ux_h],hp1=gray[row_p1+ux_h],hp2=gray[row_p2+ux_h];
                    gs_census_supp[h_dst]=s_h;
                    gs_census_top3[h_dst]=hm2+hm1+h0;
                    gs_census_bot3[h_dst]=h0+hp1+hp2;
                    gs_census_sum5[h_dst]=int(hm2+hm1+h0+hp1+hp2);
                }else{
                    gs_census_supp[h_dst]=0u;
                    gs_census_top3[h_dst]=0u;
                    gs_census_bot3[h_dst]=0u;
                    gs_census_sum5[h_dst]=0;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        uint x=x0+gidx,lx=gidx+2u;
        bool valid=(x>=2u&&x+2u<p.width)&&(gs_census_supp[lx-2u]&gs_census_supp[lx-1u]&gs_census_supp[lx]&gs_census_supp[lx+1u]&gs_census_supp[lx+2u])!=0u;
        uint desc=0u;
        if(valid){
            int gxl=gs_census_sum5[lx]-gs_census_sum5[lx-2u];
            int gxr=gs_census_sum5[lx+2u]-gs_census_sum5[lx];
            uint lt=gs_census_top3[lx-2u]+gs_census_top3[lx-1u]+gs_census_top3[lx]+gs_census_top3[lx+1u]+gs_census_top3[lx+2u];
            uint lb=gs_census_bot3[lx-2u]+gs_census_bot3[lx-1u]+gs_census_bot3[lx]+gs_census_bot3[lx+1u]+gs_census_bot3[lx+2u];
            uint q_gxl=uint(clamp((gxl+60)/15,0,8));
            uint q_gxr=uint(clamp((gxr+60)/15,0,8));
            uint q_lt=min(lt/(15u*28u),8u);
            uint q_lb=min(lb/(15u*28u),8u);
            desc=refinement_thermo8(q_gxl)|(refinement_thermo8(q_gxr)<<8)|(refinement_thermo8(q_lt)<<16)|(refinement_thermo8(q_lb)<<24);
        }
        uplane(24)[t]=desc;
        uplane(25)[t]=uint(valid);
        return;
    }
#endif
    if(t>=2*n)return;uint eye=t/n,i=t%n,x=i%p.width,y=i/p.width;uint* gray=p.borrowed[0]!=0?(uint*)p.borrowed[eye]:uplane(27)+eye*n;uint* support=p.borrowed[0]!=0?(uint*)p.borrowed[2+eye]:uplane(28)+eye*n;bool valid=refinement_support(support,p.width,p.height,x,y);uplane(24)[t]=valid?(((p.subpixel&514u)==2u)?refinement_ortho_thermo_census(gray,p.width,x,y):refinement_census(gray,p.width,x,y)):0;uplane(25)[t]=uint(valid);
}
[shader("compute")][numthreads(64,1,1)]
void match(uint3 tid:SV_DispatchThreadID){uint n=p.width*p.height,t=tid.x;if(t>=2*n)return;uint eye=t/n,i=t%n,x=i%p.width,y=i/p.width,cn=p.coarse_width*p.coarse_height;
    // Half-pixel 2x/4x coordinates are exact binary32 fractions.
    bool scale4=(p.width==p.coarse_width*4u),hybrid=(p.subpixel&2u)!=0,ultra=(p.subpixel&16u)!=0,skip_diag=(p.subpixel&4u)!=0&&(p.subpixel&128u)==0,repair=(p.subpixel&8u)!=0,adaptive_tiles=(p.subpixel&64u)!=0;
    SF_PRECISE float step=scale4?.25f:.5f,mult=scale4?4.f:2.f;
    SF_PRECISE float cx=(float(x)+.5f)*step-.5f,cy=(float(y)+.5f)*step-.5f;int ix=int(floor(cx)),iy=int(floor(cy));SF_PRECISE float wx=cx-float(ix),wy=cy-float(iy);SF_PRECISE float weights[4]={(1.f-wx)*(1.f-wy),wx*(1.f-wy),(1.f-wx)*wy,wx*wy};SF_PRECISE float predicted=0,w_valid=0;float mn=3.402823466e38f,mx=-3.402823466e38f,min_c_conf=1.f;uint reason=0,valid_taps=0;bool boundary_repair=false,hole_fallback=false,expanded_sweep=false;
    if(!hybrid){
        [unroll]for(uint tap=0;tap<4;++tap){if(weights[tap]==0)continue;int xx=ix+int(tap&1),yy=iy+int(tap>>1);if(xx<0||yy<0||xx>=int(p.coarse_width)||yy>=int(p.coarse_height)){reason|=REF_OUTSIDE;continue;}uint j=eye*cn+uint(yy)*p.coarse_width+uint(xx);float d=((float*)p.coarse[0])[j],confidence=((float*)p.coarse[1])[j];if(((uint*)p.coarse[2])[j]==0)reason|=REF_NO_PRIOR;if(!isfinite(d))reason|=REF_NONFINITE;if(confidence<p.prior_floor)reason|=REF_LOW_CONFIDENCE;if(isfinite(d)){if(d<0||d>float(p.disparities-1)*step)reason|=REF_OUTSIDE;mn=min(mn,d);mx=max(mx,d);predicted=predicted+d*weights[tap];}}
        if(mx-mn>p.max_span)reason|=REF_PRIOR_SPAN;
    }else{
        [unroll]for(uint tap=0;tap<4;++tap){int xx=ix+int(tap&1),yy=iy+int(tap>>1);if(xx<0||yy<0||xx>=int(p.coarse_width)||yy>=int(p.coarse_height))continue;uint j=eye*cn+uint(yy)*p.coarse_width+uint(xx);float d=((float*)p.coarse[0])[j],cc=((float*)p.coarse[1])[j];if(((uint*)p.coarse[2])[j]!=0&&isfinite(d)&&d>=0&&d<=float(p.disparities-1)*step){SF_PRECISE float w=weights[tap]+.001f;predicted=predicted+d*w;w_valid=w_valid+w;mn=min(mn,d);mx=max(mx,d);min_c_conf=min(min_c_conf,cc);++valid_taps;}}
        int rx=int(floor(cx+.5f)),ry=int(floor(cy+.5f));
        if(w_valid==0.f){[unroll]for(int dy=-1;dy<=1;++dy)[unroll]for(int dx=-1;dx<=1;++dx){int xx=rx+dx,yy=ry+dy;if(xx<0||yy<0||xx>=int(p.coarse_width)||yy>=int(p.coarse_height))continue;uint j=eye*cn+uint(yy)*p.coarse_width+uint(xx);float d=((float*)p.coarse[0])[j];if(((uint*)p.coarse[2])[j]!=0&&isfinite(d)&&d>=0&&d<=float(p.disparities-1)*step){predicted=predicted+d;w_valid=w_valid+1.f;mn=min(mn,d);mx=max(mx,d);}}}
        if(repair&&!ultra&&w_valid==0.f){
            [unroll]for(int dy=-2;dy<=2;dy+=2)[unroll]for(int dx=-2;dx<=2;dx+=2){int xx=rx+dx,yy=ry+dy;if(xx<0||yy<0||xx>=int(p.coarse_width)||yy>=int(p.coarse_height))continue;uint j=eye*cn+uint(yy)*p.coarse_width+uint(xx);float d=((float*)p.coarse[0])[j];if(((uint*)p.coarse[2])[j]!=0&&isfinite(d)&&d>=0&&d<=float(p.disparities-1)*step){predicted=predicted+d;w_valid=w_valid+1.f;mn=min(mn,d);mx=max(mx,d);}}
            if(w_valid<3.f||(mx-mn)>p.max_span*1.5f)w_valid=0.f;
            else hole_fallback=true;
        }
        if(w_valid>0.f){
            predicted=predicted/w_valid;
            float span_limit=repair?(p.max_span*3.0f):(p.max_span*2.5f);
            if(mx-mn>span_limit)reason|=REF_PRIOR_SPAN;
            else if(repair&&!hole_fallback&&(mx-mn>p.max_span*(adaptive_tiles?1.8f:2.5f))){
                boundary_repair=true;
                if(adaptive_tiles&&valid_taps>=2u){
                    uint* gray_own=p.borrowed[0]!=0?(uint*)p.borrowed[eye]:uplane(27)+eye*n;
                    float luma_0=float(gray_own[i]),b_pred=0.f,b_w=0.f;
                    [unroll]for(uint tap=0;tap<4;++tap){
                        int xx=ix+int(tap&1),yy=iy+int(tap>>1);
                        if(xx<0||yy<0||xx>=int(p.coarse_width)||yy>=int(p.coarse_height))continue;
                        uint j=eye*cn+uint(yy)*p.coarse_width+uint(xx);
                        float d=((float*)p.coarse[0])[j];
                        if(((uint*)p.coarse[2])[j]!=0&&isfinite(d)&&d>=0.f&&d<=float(p.disparities-1)*step){
                            uint gx_c=min(uint(xx)*4u+2u,p.width-1u),gy_c=min(uint(yy)*4u+2u,p.height-1u);
                            float l_tap=float(gray_own[gy_c*p.width+gx_c]);
                            float wp=(weights[tap]+.08f)/(1.f+.18f*abs(luma_0-l_tap));
                            b_pred+=d*wp;b_w+=wp;
                        }
                    }
                    if(b_w>0.f)predicted=.35f*predicted+.65f*(b_pred/b_w);
                }
            }
        }else reason|=REF_NO_PRIOR;
    }
    bool search_fallback=(p.subpixel&256u)!=0;
    bool prior=reason==0;predicted=predicted*mult;if(!skip_diag){fplane(9)[t]=prior?predicted:refinement_invalid();uplane(10)[t]=uint(prior);}
    uint* cen=uplane(24);uint* cval=uplane(25);
    bool supported=cval[t]!=0;if((prior||search_fallback)&&!supported)reason|=REF_UNSUPPORTED;
    bool fallback_sweep=search_fallback&&!prior&&supported;
    bool coarse_planar_pixel=false,precision_pixel=false;
    if(adaptive_tiles&&prior&&supported){
        // 2-4 px fine-processing halo via 2x2 bilinear lookahead: require all 4 surrounding coarse taps to be confident & smooth.
        coarse_planar_pixel=(valid_taps==4u&&min_c_conf>=.25f&&(mx-mn)<=.35f&&predicted<=56.f);
        precision_pixel=!coarse_planar_pixel&&((mx-mn)>.70f||min_c_conf<.16f||predicted>64.f||boundary_repair||hole_fallback);
    }
    int centre=0;uint lo=0,hi=0xffffffff;
    if(prior){
        centre=int(floor(predicted+.5f));
        int rad=coarse_planar_pixel?1:(precision_pixel?max(int(p.radius),3):int(p.radius));
        if(boundary_repair){
            int max_b=adaptive_tiles?5:4;
            int lo_b=max(int(floor(mn*mult))-1,centre-max_b);
            int hi_b=min(int(floor(mx*mult+1.f))+1,centre+max_b);
            lo=uint(clamp(lo_b,0,int(p.disparities)-1));
            hi=uint(clamp(hi_b,int(lo),int(p.disparities)-1));
        }else if(coarse_planar_pixel&&!ultra&&abs(predicted-float(centre))>.18f){
            int d_fl=int(floor(predicted));
            lo=uint(max(0,d_fl-1));
            hi=uint(min(int(p.disparities)-1,d_fl+2));
        }else{
            lo=uint(max(0,centre-rad));
            hi=uint(min(int(p.disparities)-1,centre+rad));
        }
        uint min_d_b=(p.planar_cfg>>16)&0xffu,max_d_b=(p.planar_cfg>>24)&0xffu;
        if(min_d_b>0u)lo=max(lo,min_d_b);
        if(max_d_b>0u)hi=min(hi,max_d_b);
        if(adaptive_tiles){
            if(eye==0)hi=min(hi,x);
            else hi=min(hi,(p.width-1u)-x);
        }
    }
    if(fallback_sweep){
        lo=(p.planar_cfg>>16)&0xffu;uint bound=(p.planar_cfg>>24)&0xffu;
        hi=min(p.disparities-1u,eye==0?x:(p.width-1u)-x);
        if(bound>0u)hi=min(hi,bound);
        reason|=REF_FULL_SEARCH;
    }
    if(!skip_diag){uplane(11)[t]=lo;uplane(12)[t]=hi;}
    uint best=0xffffffff,second=0xffffffff,third=0xffffffff,winner=0xffffffff,count=0,work=0,minus=0xffffffff,plus=0xffffffff;
    uint prev_cost=0xffffffff;
    uint local_best=0xffffffff,local_second=0xffffffff,local_third=0xffffffff,local_winner=0xffffffff,local_count=0,local_minus=0xffffffff,local_plus=0xffffffff;
    bool saved_local=false,retained_local=false;
    bool plateau_tie=false,strict_reject=false,photo_confirmed=false;
    if((prior||fallback_sweep)&&supported&&lo<=hi){
        uint c0=cen[t],c_l=c0,c_r=c0,c_u=c0,c_d=c0;
        bool full_5tap=hybrid&&(search_fallback||(!(ultra&&coarse_planar_pixel)&&!(ultra&&eye!=0)));
        bool edge_clamp_l=false,edge_clamp_r=false,edge_clamp_u=false,edge_clamp_d=false;
        if(hybrid&&(!(ultra&&eye!=0)||search_fallback)){
            c_l=(x>0&&cval[t-1]!=0)?cen[t-1]:c0;
            c_r=(x+1<p.width&&cval[t+1]!=0)?cen[t+1]:c0;
            if(full_5tap){
                c_u=(y>0&&cval[t-p.width]!=0)?cen[t-p.width]:c0;
                c_d=(y+1<p.height&&cval[t+p.width]!=0)?cen[t+p.width]:c0;
                if(adaptive_tiles&&boundary_repair){
                    uint* gray_own=p.borrowed[0]!=0?(uint*)p.borrowed[eye]:uplane(27)+eye*n;
                    int g0=int(gray_own[i]);
                    if(x>0&&abs(int(gray_own[i-1])-g0)>18){c_l=c0;edge_clamp_l=true;}
                    if(x+1<p.width&&abs(int(gray_own[i+1])-g0)>18){c_r=c0;edge_clamp_r=true;}
                    if(y>0&&abs(int(gray_own[i-p.width])-g0)>18){c_u=c0;edge_clamp_u=true;}
                    if(y+1<p.height&&abs(int(gray_own[i+p.width])-g0)>18){c_d=c0;edge_clamp_d=true;}
                }
            }
        }
        uint other_row=(1-eye)*n+y*p.width;
        uint max_passes=(search_fallback||(repair&&!ultra))?2u:1u;
        for(uint pass_idx=0;pass_idx<max_passes;++pass_idx){
            if(hybrid&&eye==0){
                int xx0=int(x)-int(lo);
                bool v_right=(xx0+1>=0&&xx0+1<int(p.width))?(cval[other_row+uint(xx0+1)]!=0):false;
                uint d_right=v_right?cen[other_row+uint(xx0+1)]:0u;
                bool v_curr=(xx0>=0&&xx0<int(p.width))?(cval[other_row+uint(xx0)]!=0):false;
                uint d_curr=v_curr?cen[other_row+uint(xx0)]:0u;
                for(uint d=lo;d<=hi;++d){
                    uint cost=0xffffffff;
                    int xx=int(x)-int(d);
                    bool v_left=(xx>0&&xx-1<int(p.width))?(cval[other_row+uint(xx-1)]!=0):false;
                    uint d_left=v_left?cen[other_row+uint(xx-1)]:0u;
                    if(xx>=0&&xx<int(p.width)&&v_curr){
                        uint j=other_row+uint(xx);
                        uint base=countbits(c0^d_curr);
                        uint dist=fallback_sweep?0u:uint(abs(int(d)-centre));
                        uint m_l=(v_left&&!edge_clamp_l)?countbits(c_l^d_left):base;
                        uint m_r=(v_right&&!edge_clamp_r)?countbits(c_r^d_right):base;
                        if(coarse_planar_pixel&&ultra&&!fallback_sweep){
                            cost=4u*base+m_l+m_r+dist;
                        }else{
                            uint m_u=(y>0&&!edge_clamp_u&&cval[j-p.width]!=0)?countbits(c_u^cen[j-p.width]):base;
                            uint m_d=(y+1<p.height&&!edge_clamp_d&&cval[j+p.width]!=0)?countbits(c_d^cen[j+p.width]):base;
                            cost=2u*base+m_l+m_r+m_u+m_d+dist;
                        }
                        ++count;++work;
                        if(cost<best){third=second;second=best;best=cost;winner=d;minus=prev_cost;plus=0xffffffff;}
                        else if(cost<second){third=second;second=cost;}
                        else if(cost<third){third=cost;}
                    }
                    if(winner!=0xffffffff&&d==winner+1)plus=cost;
                    prev_cost=cost;
                    v_right=v_curr;d_right=d_curr;
                    v_curr=v_left;d_curr=d_left;
                }
            }else{
                for(uint d=lo;d<=hi;++d){uint cost=0xffffffff;int xx=int(x)+(eye!=0?int(d):-int(d));if(xx>=0&&xx<int(p.width)){uint j=other_row+uint(xx);if(cval[j]!=0){
                    uint base=countbits(c0^cen[j]);
                    if(!hybrid){cost=base;}else if(ultra&&(coarse_planar_pixel||eye!=0)&&!fallback_sweep){cost=6u*base+uint(abs(int(d)-centre));}else{
                        uint m_l=(xx>0&&cval[j-1]!=0)?countbits(c_l^cen[j-1]):base;
                        uint m_r=(xx+1<int(p.width)&&cval[j+1]!=0)?countbits(c_r^cen[j+1]):base;
                        uint m_u=(y>0&&cval[j-p.width]!=0)?countbits(c_u^cen[j-p.width]):base;
                        uint m_d=(y+1<p.height&&cval[j+p.width]!=0)?countbits(c_d^cen[j+p.width]):base;
                        uint dist=fallback_sweep?0u:uint(abs(int(d)-centre));
                        cost=2u*base+m_l+m_r+m_u+m_d+dist;
                    }
                    ++count;++work;
                    if(cost<best){third=second;second=best;best=cost;winner=d;minus=prev_cost;plus=0xffffffff;}
                    else if(cost<second){third=second;second=cost;}
                    else if(cost<third){third=cost;}
                }}if(winner!=0xffffffff&&d==winner+1)plus=cost;prev_cost=cost;}
            }
            if(search_fallback&&count>0u)photo_confirmed=refinement_photometric_consistent(eye,x,y,winner);
            bool truncated=(winner==lo&&lo>0u)||(winner==hi&&hi+1u<p.disparities);
            if(pass_idx==0&&search_fallback&&!fallback_sweep&&(count<2u||truncated||second<best+p.min_margin+1u||best>16u||!photo_confirmed)){
                local_best=best;local_second=second;local_third=third;local_winner=winner;local_count=count;local_minus=minus;local_plus=plus;saved_local=true;
                lo=(p.planar_cfg>>16)&0xffu;uint bound=(p.planar_cfg>>24)&0xffu;
                hi=min(p.disparities-1u,eye==0?x:(p.width-1u)-x);
                if(bound>0u)hi=min(hi,bound);
                fallback_sweep=true;reason|=REF_FULL_SEARCH;
                best=0xffffffff;second=0xffffffff;third=0xffffffff;winner=0xffffffff;count=0;minus=0xffffffff;plus=0xffffffff;prev_cost=0xffffffff;
                if(!skip_diag){uplane(11)[t]=lo;uplane(12)[t]=hi;}
                continue;
            }
            bool full_confirmed=count>=2u&&best<=16u&&second>=best+p.min_margin+1u&&photo_confirmed&&winner>0u&&winner+1u<p.disparities;
            if(fallback_sweep&&!full_confirmed){
                reason|=REF_SEARCH_UNCONFIRMED;
                if(count>0u&&!photo_confirmed)reason|=REF_PHOTOMETRIC;
            }
            if(fallback_sweep&&saved_local&&!full_confirmed){
                // A failed extra search must not discard an existing local
                // proposal. Its original uniqueness/repair rules still apply.
                best=local_best;second=local_second;third=local_third;winner=local_winner;count=local_count;minus=local_minus;plus=local_plus;
                fallback_sweep=false;retained_local=true;
            }
            if(pass_idx==0&&!fallback_sweep&&repair&&!ultra&&!boundary_repair&&!hole_fallback&&second<best+p.min_margin&&best>32u){
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
        strict_reject=((hole_fallback||boundary_repair||expanded_sweep||fallback_sweep)&&(best>16u||second<best+p.min_margin+1u))||(fallback_sweep&&!photo_confirmed);
    }
    uint eff_margin=plateau_tie?p.min_margin:(second-best);
    uint eff_second=plateau_tie?(best+p.min_margin):second;
    bool unique=(count>=2&&eff_margin>=p.min_margin+(adaptive_tiles?1u:0u)&&!strict_reject);if(count<2)reason|=REF_TOO_FEW;if(count>=2&&!unique)reason|=REF_AMBIGUOUS;bool restricted=!hybrid&&count>0&&((winner==lo&&lo>0)||(winner==hi&&hi+1<p.disparities));if(restricted)reason|=REF_RESTRICTED;if(count>0&&(winner==0||winner+1==p.disparities))reason|=REF_GLOBAL;bool valid=unique&&!restricted&&(!fallback_sweep||(winner>0u&&winner+1u<p.disparities));
    float d=float(winner),confidence=0;bool applied=false;
    if(valid){
        confidence=clamp(float(eff_margin)/float(max(eff_second,1u)),0.f,1.f);
        if((p.subpixel&1u)!=0&&!(ultra&&eye!=0)){
            SF_PRECISE float offset=0.f;
            if(minus!=0xffffffff&&plus!=0xffffffff){
                SF_PRECISE float denominator=(float(minus)-2.f*float(best))+float(plus);
                SF_PRECISE float numerator=.5f*(float(minus)-float(plus));
                offset=denominator!=0?clamp(numerator/denominator,-1.f,1.f):0.f;
                applied=true;
            }
            if((!ultra&&((p.subpixel&32u)!=0||precision_pixel||coarse_planar_pixel))||(ultra&&(p.subpixel&32u)!=0&&!adaptive_tiles)){
                int s_dir=eye!=0?1:-1;
                int xr=int(x)+s_dir*int(winner);
                if(x>=2u&&x+2u<p.width&&y>=1u&&y+1u<p.height&&xr>=2&&xr+2<int(p.width)){
                    uint* gray_own=p.borrowed[0]!=0?(uint*)p.borrowed[eye]:uplane(27)+eye*n;
                    uint* gray_oth=p.borrowed[0]!=0?(uint*)p.borrowed[1u-eye]:uplane(27)+(1u-eye)*n;
                    int dy_lim=1;
                    SF_PRECISE float inv_n=0.1111111111f;
                    SF_PRECISE float sum_l=0.f,sum_m2=0.f,sum_m1=0.f,sum_0=0.f,sum_p1=0.f,sum_p2=0.f,gx_energy=0.f;
                    for(int dy=-dy_lim;dy<=dy_lim;++dy){
                        uint row_l=uint(int(y)+dy)*p.width;
                        uint row_r=uint(int(y)+dy)*p.width;
                        SF_PRECISE float gl_m1=float(gray_own[row_l+uint(int(x)-1)]);
                        SF_PRECISE float gl_p1=float(gray_own[row_l+uint(int(x)+1)]);
                        SF_PRECISE float gdiff=gl_p1-gl_m1;
                        gx_energy=gx_energy+(gdiff*gdiff);
                        [unroll]for(int dx=-1;dx<=1;++dx){
                            SF_PRECISE float lv=float(gray_own[row_l+uint(int(x)+dx)]);
                            int rx=xr+dx;
                            SF_PRECISE float rv_0=float(gray_oth[row_r+uint(rx)]);
                            SF_PRECISE float rv_m=float(gray_oth[row_r+uint(rx-s_dir)]);
                            SF_PRECISE float rv_p=float(gray_oth[row_r+uint(rx+s_dir)]);
                            SF_PRECISE float hm=.5f*(rv_0+rv_m);
                            SF_PRECISE float hp=.5f*(rv_0+rv_p);
                            sum_l=sum_l+lv;sum_m2=sum_m2+rv_m;sum_m1=sum_m1+hm;sum_0=sum_0+rv_0;sum_p1=sum_p1+hp;sum_p2=sum_p2+rv_p;
                        }
                    }
                    SF_PRECISE float mean_l=sum_l*inv_n,mean_m2=sum_m2*inv_n,mean_m1=sum_m1*inv_n,mean_0=sum_0*inv_n,mean_p1=sum_p1*inv_n,mean_p2=sum_p2*inv_n;
                    SF_PRECISE float e_m2=0.f,e_m1=0.f,e_0=0.f,e_p1=0.f,e_p2=0.f;
                    for(int dy2=-dy_lim;dy2<=dy_lim;++dy2){
                        uint row_l=uint(int(y)+dy2)*p.width;
                        uint row_r=uint(int(y)+dy2)*p.width;
                        [unroll]for(int dx2=-1;dx2<=1;++dx2){
                            SF_PRECISE float lv=float(gray_own[row_l+uint(int(x)+dx2)]);
                            int rx=xr+dx2;
                            SF_PRECISE float rv_0=float(gray_oth[row_r+uint(rx)]);
                            SF_PRECISE float rv_m=float(gray_oth[row_r+uint(rx-s_dir)]);
                            SF_PRECISE float rv_p=float(gray_oth[row_r+uint(rx+s_dir)]);
                            SF_PRECISE float hm=.5f*(rv_0+rv_m);
                            SF_PRECISE float hp=.5f*(rv_0+rv_p);
                            SF_PRECISE float dl=lv-mean_l;
                            SF_PRECISE float dm2=dl-(rv_m-mean_m2);
                            SF_PRECISE float dm1=dl-(hm-mean_m1);
                            SF_PRECISE float d0=dl-(rv_0-mean_0);
                            SF_PRECISE float dp1=dl-(hp-mean_p1);
                            SF_PRECISE float dp2=dl-(rv_p-mean_p2);
                            e_m2=e_m2+(dm2*dm2);e_m1=e_m1+(dm1*dm1);e_0=e_0+(d0*d0);e_p1=e_p1+(dp1*dp1);e_p2=e_p2+(dp2*dp2);
                        }
                    }
                    SF_PRECISE float base_off=0.f,el=e_m1,ec=e_0,er=e_p1;
                    if(e_m1<ec&&e_m1<=e_p1){base_off=-.5f;el=e_m2;ec=e_m1;er=e_0;}
                    else if(e_p1<ec){base_off=.5f;el=e_0;ec=e_p1;er=e_p2;}
                    SF_PRECISE float p_denom=(el-2.f*ec)+er;
                    SF_PRECISE float p_num=.25f*(el-er);
                    if(gx_energy>4.f&&p_denom>1e-3f&&ec<=el&&ec<=er){
                        SF_PRECISE float p_off=clamp(base_off+clamp(p_num/p_denom,-.25f,.25f),-.85f,.85f);
                        if(!applied){
                            offset=p_off;applied=true;
                        }else{
                            SF_PRECISE float wp=(minus==plus||eff_margin<=8u)?.75f:.55f;
                            SF_PRECISE float wc=1.f-wp;
                            offset=clamp((wc*offset)+(wp*p_off),-1.f,1.f);
                        }
                    }
                }
            }
            if(applied)d=d+offset;
            else reason|=REF_MISSING_SUBPIXEL;
        }
    }
    if(!skip_diag){uplane(13)[t]=reason;uplane(14)[t]=retained_local?3u:(fallback_sweep?2u:(prior?1u:0u));uplane(15)[t]=winner;uplane(16)[t]=best;uplane(17)[t]=second;uplane(18)[t]=count;uplane(19)[t]=work;uplane(20)[t]=uint(applied);}fplane(21)[t]=valid?d:refinement_invalid();if((p.subpixel&20u)!=20u||eye==0)fplane(22)[t]=valid?confidence:0;uplane(23)[t]=uint(valid);
}
[shader("compute")][numthreads(64,1,1)]
void lr(uint3 tid:SV_DispatchThreadID){
    if((p.subpixel&84u)==84u)return;
    uint n=p.width*p.height,t=tid.x;if(t>=2*n)return;
    bool skip_r=((p.subpixel&20u)==20u);
    if(skip_r&&t>=n)return;
    uint eye=t/n,i=t%n,target=0;
    float d=fplane(21)[t];bool valid=false;
    if(uplane(23)[t]!=0&&refinement_target(p.width,i%p.width,d,eye!=0,target)){
        uint j=(1-eye)*n+(i/p.width)*p.width+target;
        valid=observation_lr(d,fplane(21)[j],uplane(23)[t],uplane(23)[j],p.lr_tolerance);
    }
    if(((p.subpixel&4u)==0||(p.subpixel&128u)!=0)&&uplane(23)[t]!=0&&!valid)uplane(13)[t]|=REF_LR_FAILED;
    fplane(0)[t]=valid?d:refinement_invalid();
    fplane(1)[t]=valid?fplane(22)[t]:0;
    uplane(2)[t]=uint(valid);
}
void write_completed(uint t,float z,float c,uint v,uint prov,uint2 sid,float d){
    ((float*)p.completed[0])[t]=z;((float*)p.completed[1])[t]=c;((uint*)p.completed[2])[t]=v;((uint*)p.completed[3])[t]=prov;
    ((uint2*)p.completed[4])[t]=sid;((float*)p.completed[6])[t]=d;
}
[shader("compute")][numthreads(64,1,1)]
void metric(uint3 tid:SV_DispatchThreadID){
    uint n=p.width*p.height,t=tid.x;if(t>=2*n)return;
    bool skip_r=((p.subpixel&20u)==20u);
    if(skip_r&&t>=n)return;
    float d=0.f,c_in=0.f;bool lr_valid=false;
    if((p.subpixel&84u)==84u){
        uint eye=t/n,i=t%n,target=0;
        d=fplane(21)[t];
        if(uplane(23)[t]!=0&&refinement_target(p.width,i%p.width,d,eye!=0,target)){
            uint j=(1-eye)*n+(i/p.width)*p.width+target;
            lr_valid=observation_lr(d,fplane(21)[j],uplane(23)[t],uplane(23)[j],p.lr_tolerance);
        }
        if((p.subpixel&128u)!=0&&uplane(23)[t]!=0&&!lr_valid)uplane(13)[t]|=REF_LR_FAILED;
        c_in=lr_valid?fplane(22)[t]:0.f;
        fplane(0)[t]=lr_valid?d:refinement_invalid();
        fplane(1)[t]=c_in;
        uplane(2)[t]=uint(lr_valid);
    }else{
        d=fplane(0)[t];
        c_in=fplane(1)[t];
        lr_valid=(uplane(2)[t]!=0);
    }
    float z=refinement_invalid();
    bool valid=lr_valid&&d>0&&(d-p.principal_offset)>0;
    if(valid){
        SF_PRECISE float numerator=p.fx*p.baseline;
        SF_PRECISE float denominator=d-p.principal_offset;
        z=numerator/denominator;
        valid=isfinite(z)&&z>0;
    }
    float z_out=valid?z:refinement_invalid(),c_out=valid?c_in:0;
    uint v_out=uint(valid),prov_out=valid?SF_PROVENANCE_STEREO:0u;
    uint2 sid=valid?uint2(p.source_low,p.source_high):uint2(0xffffffff,0xffffffff);
    if(t<n||!skip_r||p.completed[0]==0){
        fplane(3)[t]=z_out;fplane(4)[t]=c_out;uplane(5)[t]=v_out;uplane(6)[t]=prov_out;((uint2*)p.output[7])[t]=sid;
    }
    if(p.completed[0]!=0)write_completed(t,z_out,c_out,v_out,prov_out,sid,valid?d:refinement_invalid());
}
bool right_lr_check(uint n,uint y,uint rx,out float d_r){
    uint jr=n+y*p.width+rx;
    if((p.subpixel&20u)!=20u){
        d_r=fplane(0)[jr];
        return uplane(2)[jr]!=0u;
    }
    d_r=fplane(21)[jr];
    if(uplane(23)[jr]==0u)return false;
    uint target_l=0u;
    if(!refinement_target(p.width,rx,d_r,true,target_l))return false;
    uint jl=y*p.width+target_l;
    return observation_lr(d_r,fplane(21)[jl],uplane(23)[jr],uplane(23)[jl],p.lr_tolerance);
}
#if !defined(SF_CPU_TARGET)
groupshared uint gs_has_candidate;
groupshared uint gs_has_valid_coarse;
groupshared float gs_disp[34][34];
groupshared float gs_luma[34][34];
groupshared float gs_coarse_d[8][8];
groupshared float4 gs_coarse_plane[4][4];
#endif
float planar_tile_luma(uint* gray_l,uint lx,uint ly,uint x,uint y,int dx,int dy){
#if !defined(SF_CPU_TARGET)
    return gs_luma[int(ly)+dy][int(lx)+dx];
#else
    return float(gray_l[uint(int(y)+dy)*p.width+uint(int(x)+dx)]);
#endif
}
float planar_tile_disp(uint lx,uint ly,uint x,uint y,int dx,int dy){
#if !defined(SF_CPU_TARGET)
    return gs_disp[int(ly)+dy][int(lx)+dx];
#else
    uint idx=uint(int(y)+dy)*p.width+uint(int(x)+dx);
    return (uplane(2)[idx]!=0)?fplane(0)[idx]:-1.f;
#endif
}
[shader("compute")][numthreads(16,16,1)]
void planar(uint3 tid:SV_DispatchThreadID,uint3 gid:SV_GroupID,uint3 gtid:SV_GroupThreadID,uint gidx:SV_GroupIndex){
    if(p.completed[0]==0)return;
#if !defined(SF_CPU_TARGET)
    if(gidx==0u){gs_has_candidate=0u;gs_has_valid_coarse=0u;}
    GroupMemoryBarrierWithGroupSync();
#endif
    uint x=tid.x,y=tid.y,n=p.width*p.height,t=y*p.width+x;
    bool in_bounds=(x>=12u&&x+12u<p.width&&y>=12u&&y+12u<p.height);
    bool is_cand=in_bounds&&(((uint*)p.completed[2])[t]==0u)&&(uplane(25)[t]!=0u);
#if !defined(SF_CPU_TARGET)
    if(is_cand)InterlockedOr(gs_has_candidate,1u);
    GroupMemoryBarrierWithGroupSync();
    if(gs_has_candidate==0u)return;
#else
    if(!is_cand)return;
#endif
    bool scale4=(p.width==p.coarse_width*4u);
    bool use_coarse_affine=(p.planar_cfg&256u)!=0;
    bool ultra=(p.subpixel&16u)!=0;
    SF_PRECISE float step=scale4?.25f:.5f,mult=scale4?4.f:2.f;
#if !defined(SF_CPU_TARGET)
    if(scale4&&use_coarse_affine){
        if(gidx<64u){
            uint cy_i=gidx>>3u,cx_i=gidx&7u;
            int rcx0=clamp(int(gid.x*4u)+int(cx_i)-2,0,int(p.coarse_width)-1);
            int rcy0=clamp(int(gid.y*4u)+int(cy_i)-2,0,int(p.coarse_height)-1);
            uint cj=uint(rcy0)*p.coarse_width+uint(rcx0);
            float cd=((float*)p.coarse[0])[cj];
            bool ok=(((uint*)p.coarse[2])[cj]!=0)&&isfinite(cd)&&cd>0.f&&cd<=float(p.disparities-1)*step;
            gs_coarse_d[cy_i][cx_i]=ok?cd:-1.f;
        }
        GroupMemoryBarrierWithGroupSync();
        if(gidx<16u){
            uint py=gidx>>2u,px=gidx&3u;
            float4 plane=float4(0,0,0,-1.f);
            uint c_cnt=0;
            float c_mn=3.402823466e38f,c_mx=-3.402823466e38f;
            float csx=0,csy=0,csxx=0,csyy=0,csxy=0,csd=0,csxd=0,csyd=0,csdd=0;
            for(int cdy=-1;cdy<=1;++cdy){
                float fcdy=float(cdy);
                for(int cdx=-1;cdx<=1;++cdx){
                    float cd=gs_coarse_d[int(py)+2+cdy][int(px)+2+cdx];
                    if(cd<=0.f)continue;
                    float fcdx=float(cdx);
                    ++c_cnt;c_mn=min(c_mn,cd);c_mx=max(c_mx,cd);
                    csx+=fcdx;csy+=fcdy;csxx+=fcdx*fcdx;csyy+=fcdy*fcdy;csxy+=fcdx*fcdy;
                    csd+=cd;csxd+=fcdx*cd;csyd+=fcdy*cd;csdd+=cd*cd;
                }
            }
            float c_span_lim=((p.subpixel&64u)!=0)?(p.max_span*.90f):(p.max_span*1.25f);
            if(c_cnt>=5u&&(c_mx-c_mn)<=c_span_lim){
                float fcn=float(c_cnt),inv_cn=1.f/fcn;
                float cmx=csx*inv_cn,cmy=csy*inv_cn,cmd=csd*inv_cn;
                float ccxx=csxx-fcn*cmx*cmx,ccyy=csyy-fcn*cmy*cmy,ccxy=csxy-fcn*cmx*cmy;
                float cdet=ccxx*ccyy-ccxy*ccxy;
                float ac=0.f,bc=0.f,c_rmse=0.f;
                bool ok_fit=true;
                if(cdet>.25f){
                    float ccxd=csxd-fcn*cmx*cmd,ccyd=csyd-fcn*cmy*cmd,ccdd=csdd-fcn*cmd*cmd;
                    ac=(ccyy*ccxd-ccxy*ccyd)/cdet;
                    bc=(ccxx*ccyd-ccxy*ccxd)/cdet;
                    float cssr=max(0.f,ccdd-ac*ccxd-bc*ccyd);
                    c_rmse=sqrt(cssr*inv_cn);
                    if(abs(ac)>.60f||abs(bc)>.85f||c_rmse>.55f)ok_fit=false;
                }
                if(ok_fit){
                    float quality=(c_cnt>=6u&&c_rmse<=.42f)?2.f:1.f;
                    plane=float4(cmd-ac*cmx-bc*cmy,ac,bc,quality);
                }
            }
            gs_coarse_plane[py][px]=plane;
        }
        GroupMemoryBarrierWithGroupSync();
    }
#endif
    uint min_anc=max(p.planar_cfg&0xffu,1u);
    bool ray_guard=(p.planar_cfg&512u)!=0;
    bool coarse_lr_guard=(p.planar_cfg&1024u)!=0;
    SF_PRECISE float cx=(float(x)+.5f)*step-.5f,cy=(float(y)+.5f)*step-.5f;
    uint cn=p.coarse_width*p.coarse_height;
    float d_coarse=0.f,ac_full=0.f,bc_full=0.f;
    bool strong_coarse=false;
    if(is_cand){
#if !defined(SF_CPU_TARGET)
        if(scale4&&use_coarse_affine){
            uint px=gtid.x>>2u,py=gtid.y>>2u;
            float4 plane=gs_coarse_plane[py][px];
            if(plane.w<0.f)is_cand=false;
            else{
                int rcx=clamp(int(gid.x*4u+px),1,int(p.coarse_width)-2);
                int rcy=clamp(int(gid.y*4u+py),1,int(p.coarse_height)-2);
                ac_full=plane.y;bc_full=plane.z;
                d_coarse=(plane.x+ac_full*(cx-float(rcx))+bc_full*(cy-float(rcy)))*mult;
                strong_coarse=(plane.w>1.5f);
                if((p.subpixel&64u)!=0){
                    int sx=(gtid.x&3u)<2u?-1:1,sy=(gtid.y&3u)<2u?-1:1;
                    float wx=abs(float(gtid.x&3u)-1.5f)*.25f,wy=abs(float(gtid.y&3u)-1.5f)*.25f;
                    float w_sum=(1.f-wx)*(1.f-wy);
                    float d_sum=w_sum*d_coarse,ac_sum=w_sum*ac_full,bc_sum=w_sum*bc_full;
                    [unroll]for(uint tap=1u;tap<4u;++tap){
                        int tx=int(px)+((tap&1u)!=0u?sx:0),ty=int(py)+((tap&2u)!=0u?sy:0);
                        if(tx>=0&&tx<4&&ty>=0&&ty<4){
                            float4 pn=gs_coarse_plane[uint(ty)][uint(tx)];
                            if(pn.w>0.f){
                                int rtx=clamp(int(gid.x*4u)+tx,1,int(p.coarse_width)-2);
                                int rty=clamp(int(gid.y*4u)+ty,1,int(p.coarse_height)-2);
                                float dn=(pn.x+pn.y*(cx-float(rtx))+pn.z*(cy-float(rty)))*mult;
                                float slope_diff=abs(pn.y-ac_full)+abs(pn.z-bc_full);
                                if(abs(dn-d_coarse)<=.85f&&slope_diff<=.18f){
                                    float wn=((tap&1u)!=0u?wx:(1.f-wx))*((tap&2u)!=0u?wy:(1.f-wy));
                                    w_sum+=wn;d_sum+=wn*dn;ac_sum+=wn*pn.y;bc_sum+=wn*pn.z;
                                }
                            }
                        }
                    }
                    float inv_w=1.f/w_sum;
                    d_coarse=d_sum*inv_w;ac_full=ac_sum*inv_w;bc_full=bc_sum*inv_w;
                }
            }
        }else
#endif
        if(use_coarse_affine){
            int rcx=clamp(scale4?int(x>>2u):int(floor(cx+.5f)),1,int(p.coarse_width)-2);
            int rcy=clamp(scale4?int(y>>2u):int(floor(cy+.5f)),1,int(p.coarse_height)-2);
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
            float c_span_lim=((p.subpixel&64u)!=0)?(p.max_span*.90f):(p.max_span*1.25f);
            if(c_cnt<5u||(c_mx-c_mn)>c_span_lim)is_cand=false;
            else{
                float fcn=float(c_cnt),inv_cn=1.f/fcn;
                float cmx=csx*inv_cn,cmy=csy*inv_cn,cmd=csd*inv_cn;
                float ccxx=csxx-fcn*cmx*cmx,ccyy=csyy-fcn*cmy*cmy,ccxy=csxy-fcn*cmx*cmy;
                float cdet=ccxx*ccyy-ccxy*ccxy;
                float ac=0.f,bc=0.f,c_rmse=0.f;
                if(cdet>.25f){
                    float ccxd=csxd-fcn*cmx*cmd,ccyd=csyd-fcn*cmy*cmd,ccdd=csdd-fcn*cmd*cmd;
                    ac=(ccyy*ccxd-ccxy*ccyd)/cdet;
                    bc=(ccxx*ccyd-ccxy*ccxd)/cdet;
                    float cssr=max(0.f,ccdd-ac*ccxd-bc*ccyd);
                    c_rmse=sqrt(cssr*inv_cn);
                    if(abs(ac)>.60f||abs(bc)>.85f||c_rmse>.55f)is_cand=false;
                }
                if(is_cand){
                    float plane_x=cmd-ac*cmx-bc*cmy;
                    d_coarse=(plane_x+ac*(cx-float(rcx))+bc*(cy-float(rcy)))*mult;
                    ac_full=ac;bc_full=bc;
                    strong_coarse=(c_cnt>=6u&&c_rmse<=.42f);
                }
            }
        }else{
            int ix=int(floor(cx)),iy=int(floor(cy));
            SF_PRECISE float wx=cx-float(ix),wy=cy-float(iy);
            SF_PRECISE float weights[4]={(1.f-wx)*(1.f-wy),wx*(1.f-wy),(1.f-wx)*wy,wx*wy};
            float mn=3.402823466e38f,mx=-3.402823466e38f;
            SF_PRECISE float predicted=0,w_valid=0;
            for(uint tap=0;tap<4;++tap){
                int xx=ix+int(tap&1),yy=iy+int(tap>>1);
                if(xx<0||yy<0||xx>=int(p.coarse_width)||yy>=int(p.coarse_height))continue;
                uint j=uint(yy)*p.coarse_width+uint(xx);
                float d=((float*)p.coarse[0])[j];
                if(((uint*)p.coarse[2])[j]!=0&&isfinite(d)&&d>=0&&d<=float(p.disparities-1)*step){
                    SF_PRECISE float w=weights[tap]+.001f;predicted=predicted+d*w;w_valid=w_valid+w;mn=min(mn,d);mx=max(mx,d);
                }
            }
            if(w_valid<.70f||(mx-mn)>p.max_span*1.25f)is_cand=false;
            else d_coarse=(predicted/w_valid)*mult;
        }
        if(is_cand&&(d_coarse<1.5f||d_coarse>=float(p.disparities-1)))is_cand=false;
        // [P0] Reciprocal right-eye coarse & LR-verified full-res disparity check before loading full-res halo.
        if(is_cand&&coarse_lr_guard){
            float cx_r=cx-d_coarse*step;
            int ix_r=int(floor(cx_r+.5f)),iy_r=clamp(int(floor(cy+.5f)),0,int(p.coarse_height)-1);
            if(ix_r<0||ix_r>=int(p.coarse_width))is_cand=false;
            else{
                float clr_tol=max(p.planar_coarse_lr_tol,3.2f);
                uint jr_c0=cn+uint(iy_r)*p.coarse_width+uint(ix_r);
                bool r_agree=false,r_contra=false;
                if(((uint*)p.coarse[2])[jr_c0]!=0){
                    float dr_c0=((float*)p.coarse[0])[jr_c0]*mult;
                    if(isfinite(dr_c0)){
                        float diff0=abs(dr_c0-d_coarse);
                        if(diff0<=clr_tol)r_agree=true;
                        else if(diff0>max(clr_tol*1.4f,4.5f))r_contra=true;
                    }
                }
                if(!r_agree&&!r_contra){
                    for(int rdx=-1;rdx<=1;rdx+=2){
                        int rxx=clamp(ix_r+rdx,0,int(p.coarse_width)-1);
                        uint jr_c=cn+uint(iy_r)*p.coarse_width+uint(rxx);
                        if(((uint*)p.coarse[2])[jr_c]!=0){
                            float dr_c=((float*)p.coarse[0])[jr_c]*mult;
                            if(isfinite(dr_c)&&abs(dr_c-d_coarse)<=clr_tol)r_agree=true;
                        }
                    }
                }
                if(!r_agree||r_contra)is_cand=false;
                else{
                    int rx_c=int(floor(float(x)-d_coarse+.5f));
                    if(rx_c>=0&&rx_c<int(p.width)){
                        float dr_f=0.f;
                        if(right_lr_check(n,y,uint(rx_c),dr_f)&&abs(dr_f-d_coarse)>2.0f)is_cand=false;
                    }
                }
            }
        }
        if(is_cand){
            // Fast cardinal + diagonal quadrant pre-screen before loading LDS tile.
            uint qpre=0u;
            static const int2 kPre[12]={
                int2(-5,-2),int2(5,-2),int2(-5,2),int2(5,2),
                int2(-2,-5),int2(2,-5),int2(-2,5),int2(2,5),
                int2(-9,-5),int2(9,-5),int2(-9,5),int2(9,5)
            };
            for(uint qp=0u;qp<12u;++qp){
                int2 o=kPre[qp];
                uint idx_p=uint(int(y)+o.y)*p.width+uint(int(x)+o.x);
                if(uplane(2)[idx_p]!=0u)qpre|=1u<<(qp&3u);
            }
            if(countbits(qpre)<2u)is_cand=false;
#if !defined(SF_CPU_TARGET)
            else InterlockedOr(gs_has_valid_coarse,1u);
#endif
        }
    }
#if !defined(SF_CPU_TARGET)
    GroupMemoryBarrierWithGroupSync();
    if(gs_has_valid_coarse==0u)return;
#else
    if(!is_cand)return;
#endif
    uint* gray_l=p.borrowed[0]!=0?(uint*)p.borrowed[0]:uplane(27);
    uint* gray_r=p.borrowed[0]!=0?(uint*)p.borrowed[1]:uplane(27)+n;
    uint halo=ultra?8u:9u;
#if !defined(SF_CPU_TARGET)
    int base_x=int(gid.x*16u)-int(halo),base_y=int(gid.y*16u)-int(halo);
    if(ultra){
        for(uint k=gidx;k<1024u;k+=256u){
            uint hy=k>>5u,hx=k&31u;
            int gx=clamp(base_x+int(hx),0,int(p.width)-1);
            int gy=clamp(base_y+int(hy),0,int(p.height)-1);
            uint g_idx=uint(gy)*p.width+uint(gx);
            gs_luma[hy][hx]=float(gray_l[g_idx]);
            gs_disp[hy][hx]=(uplane(2)[g_idx]!=0)?fplane(0)[g_idx]:-1.f;
        }
    }else{
        for(uint k=gidx;k<34u*34u;k+=256u){
            uint hy=k/34u,hx=k%34u;
            int gx=clamp(base_x+int(hx),0,int(p.width)-1);
            int gy=clamp(base_y+int(hy),0,int(p.height)-1);
            uint g_idx=uint(gy)*p.width+uint(gx);
            gs_luma[hy][hx]=float(gray_l[g_idx]);
            gs_disp[hy][hx]=(uplane(2)[g_idx]!=0)?fplane(0)[g_idx]:-1.f;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if(!is_cand)return;
#endif
    uint lx=gtid.x+halo,ly=gtid.y+halo;
    float i0=planar_tile_luma(gray_l,lx,ly,x,y,0,0);
    float l_xm1=planar_tile_luma(gray_l,lx,ly,x,y,-1,0),l_xp1=planar_tile_luma(gray_l,lx,ly,x,y,1,0);
    float l_ym1=planar_tile_luma(gray_l,lx,ly,x,y,0,-1),l_yp1=planar_tile_luma(gray_l,lx,ly,x,y,0,1);
    float grad_1=abs(l_xp1-l_xm1)+abs(l_yp1-l_ym1);
    if(!use_coarse_affine&&grad_1>28.f)return;
    float max_ldiff=(use_coarse_affine&&strong_coarse)?(22.f+1.25f*grad_1):18.f;
    // Single-pass 36-tap (or 16-tap in ultra-fast mode) quadrant stencil reading from LDS/tile
    // with coarse-slope-centered residual u_res = da - dc_a and soft-bilateral Cauchy weighting.
    static const int kOff[6]={-9,-5,-2,2,5,9};
    static const int kOffFast[4]={-8,-3,3,8};
    uint n_taps=ultra?4u:6u;
    uint n_anc=0,n_inner=0,n_contra=0,qmask=0;
    float anc_mn=3.402823466e38f,anc_mx=-3.402823466e38f;
    float sw=0,sx=0,sy=0,sxx=0,syy=0,sxy=0,su=0,sxu=0,syu=0,suu=0;
    float dc_tol=(d_coarse>=114.f)?2.6f:1.8f;
    float contra_tol=2.0f+5.0f*(abs(ac_full)+abs(bc_full));
    if(ray_guard){
        static const int2 kCard[8]={int2(-2,0),int2(2,0),int2(0,-2),int2(0,2),int2(-4,0),int2(4,0),int2(0,-4),int2(0,4)};
        for(uint ci=0u;ci<8u;++ci){
            int2 co=kCard[ci];
            float sda=planar_tile_disp(lx,ly,x,y,co.x,co.y);
            if(sda>0.f&&abs(sda-(d_coarse+ac_full*float(co.x)+bc_full*float(co.y)))>contra_tol)return;
        }
    }
    for(uint iy_i=0;iy_i<n_taps;++iy_i){
        int dy=ultra?kOffFast[iy_i]:kOff[iy_i];
        uint qy=(dy>0)?2u:0u;
        float fy_d=float(dy);
        for(uint ix_i=0;ix_i<n_taps;++ix_i){
            int dx=ultra?kOffFast[ix_i]:kOff[ix_i];
            float da=planar_tile_disp(lx,ly,x,y,dx,dy);
            if(da<=0.f)continue;
            float ga=planar_tile_luma(gray_l,lx,ly,x,y,dx,dy);
            float l_diff=abs(ga-i0);
            if(l_diff>max_ldiff)continue;
            float fx_d=float(dx);
            float dc_a=d_coarse+ac_full*fx_d+bc_full*fy_d;
            float u_res=da-dc_a;
            if(abs(u_res)>dc_tol){
                if(abs(u_res)>contra_tol)++n_contra;
                continue;
            }
            if(ray_guard&&(abs(dx)>=5||abs(dy)>=5)){
                float gm=planar_tile_luma(gray_l,lx,ly,x,y,dx/2,dy/2);
                if(abs(gm-i0)>max(24.f,.85f*max_ldiff)||abs(gm-ga)>max(24.f,.85f*max_ldiff))continue;
            }
            float w_a=1.f/((1.f+1.77f*u_res*u_res)*(1.f+.0025f*l_diff*l_diff));
            ++n_anc;
            if(abs(dx)<=5&&abs(dy)<=5)++n_inner;
            anc_mn=min(anc_mn,da);anc_mx=max(anc_mx,da);
            qmask|=1u<<(qy+(dx>0?1u:0u));
            sw+=w_a;
            sx+=w_a*fx_d;sy+=w_a*fy_d;
            sxx+=w_a*fx_d*fx_d;syy+=w_a*fy_d*fy_d;sxy+=w_a*fx_d*fy_d;
            su+=w_a*u_res;sxu+=w_a*fx_d*u_res;syu+=w_a*fy_d*u_res;suu+=w_a*u_res*u_res;
        }
    }
    if(ray_guard&&(n_contra>0u||n_inner==0u))return;
    uint qcnt=countbits(qmask);
    bool h_two_sided=((qmask&5u)!=0u)&&((qmask&10u)!=0u);
    float anc_span_lim=p.max_span*mult*(((p.subpixel&64u)!=0)?1.10f:1.65f);
    if(sw<=1e-3f||(anc_mx-anc_mn)>anc_span_lim)return;
    float fn=float(n_anc),inv_w=1.f/sw;
    float mx_a=sx*inv_w,my_a=sy*inv_w,mu_a=su*inv_w;
    float cxx=sxx-sw*mx_a*mx_a,cyy=syy-sw*my_a*my_a,cxy=sxy-sw*mx_a*my_a;
    float det=cxx*cyy-cxy*cxy;
    uint eff_min=ultra?max(min_anc*2u/3u,3u):min_anc;
    uint min_tier1=strong_coarse?max(eff_min*3u/4u,ultra?3u:4u):eff_min;
    bool well_cond=(n_anc>=min_tier1)&&(qcnt>=3u)&&(det>=1.2f*sw*(cxx+cyy))&&(det>.5f);
    bool tier2_ok=use_coarse_affine&&ray_guard&&coarse_lr_guard&&(!well_cond)&&h_two_sided&&(((n_anc>=max(eff_min/2u,2u))&&(qcnt>=2u))||(strong_coarse&&(n_anc>=max(eff_min/4u,2u))));
    if(!well_cond&&!tier2_ok)return;
    float cxu=sxu-sw*mx_a*mu_a,cyu=syu-sw*my_a*mu_a,cuu=suu-sw*mu_a*mu_a;
    float da_slope=well_cond?((cyy*cxu-cxy*cyu)/det):0.f;
    float db_slope=well_cond?((cxx*cyu-cxy*cxu)/det):0.f;
    float a=ac_full+da_slope,b=bc_full+db_slope;
    if(abs(a)>.25f||abs(b)>.55f)return;
    float ssr=well_cond?max(0.f,cuu-da_slope*cxu-db_slope*cyu):max(0.f,cuu);
    float rmse=sqrt(ssr*inv_w);
    float max_rmse=min(p.planar_max_rmse,clamp(.35f+.0045f*d_coarse,.40f,.75f));
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
    if(coarse_lr_guard){
        float dr_f=0.f;
        if(right_lr_check(n,y,uint(rx),dr_f)&&abs(dr_f-d_plane)>1.8f)return;
    }
    float wf=xr_f-float(rx0);
    float r_luma=(1.f-wf)*float(gray_r[y*p.width+uint(rx0)])+wf*float(gray_r[y*p.width+uint(rx0+1)]);
    float l_mean5=.2f*(i0+l_xm1+l_xp1+l_ym1+l_yp1);
    float grad_mag=use_coarse_affine?max(.5f*(abs(l_xp1-l_xm1)+abs(l_yp1-l_ym1)),1.5f*abs(i0-l_mean5)):0.f;
    if(abs(i0-r_luma)>18.f+.65f*grad_mag)return;
    float r_mean5=.2f*(float(gray_r[r_idx])+float(gray_r[r_idx-1u])+float(gray_r[r_idx+1u])+float(gray_r[r_idx-p.width])+float(gray_r[r_idx+p.width]));
    if(abs(l_mean5-r_mean5)>p.planar_patch_luma_tol+.40f*grad_mag)return;
    uint c0=uplane(24)[t],base=countbits(c0^uplane(24)[jr]);
    uint m_l=(uplane(25)[t-1]!=0&&uplane(25)[jr-1]!=0)?countbits(uplane(24)[t-1]^uplane(24)[jr-1]):base;
    uint m_r=(uplane(25)[t+1]!=0&&uplane(25)[jr+1]!=0)?countbits(uplane(24)[t+1]^uplane(24)[jr+1]):base;
    uint m_u=(uplane(25)[t-p.width]!=0&&uplane(25)[jr-p.width]!=0)?countbits(uplane(24)[t-p.width]^uplane(24)[jr-p.width]):base;
    uint m_d=(uplane(25)[t+p.width]!=0&&uplane(25)[jr+p.width]!=0)?countbits(uplane(24)[t+p.width]^uplane(24)[jr+p.width]):base;
    uint cost_plane=2u*base+m_l+m_r+m_u+m_d;
    uint max_cost=use_coarse_affine?(grad_mag>16.f?34u:30u):28u;
    if(cost_plane>max_cost)return;
    SF_PRECISE float numerator=p.fx*p.baseline;
    SF_PRECISE float denominator=d_plane-p.principal_offset;
    float z=numerator/denominator;
    if(!isfinite(z)||z<=0.f)return;
    float conf=clamp((1.f-rmse/.55f)*(1.f-float(cost_plane)/float(max_cost))*min(fn/12.f,1.f),.05f,.85f);
    write_completed(t,z,conf,1u,SF_PROVENANCE_PLANAR_INFERRED,uint2(p.source_low,p.source_high),d_plane);
}
