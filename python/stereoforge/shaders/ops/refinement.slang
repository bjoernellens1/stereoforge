#ifndef SF_REFINEMENT_OPS
#define SF_REFINEMENT_OPS
#if defined(SF_CPU_TARGET)
#define SF_PRECISE
#else
#define SF_PRECISE precise
#endif
// Integer diagnostics are a stable wire contract. NaN is canonical on invalid outputs.
static const uint REF_NO_PRIOR=1,REF_LOW_CONFIDENCE=2,REF_PRIOR_SPAN=4,REF_NONFINITE=8,REF_OUTSIDE=16,REF_UNSUPPORTED=32,REF_TOO_FEW=64,REF_AMBIGUOUS=128,REF_RESTRICTED=256,REF_GLOBAL=512,REF_MISSING_SUBPIXEL=1024,REF_LR_FAILED=2048;
float refinement_invalid(){return asfloat(0x7fc00000);}
uint refinement_census(uint* gray,uint width,uint x,uint y){uint centre=gray[y*width+x],bits=0,bit=0;[unroll]for(int dy=-2;dy<=2;++dy)[unroll]for(int dx=-2;dx<=2;++dx){if(dx==0&&dy==0)continue;bits|=uint(gray[uint(int(y)+dy)*width+uint(int(x)+dx)]<centre)<<(bit++);}const int2 sym[8]={int2(0,1),int2(0,-1),int2(1,0),int2(-1,0),int2(0,2),int2(0,-2),int2(2,0),int2(-2,0)};[unroll]for(uint k=0;k<8;++k){int2 d=sym[k];bits|=uint(gray[uint(int(y)+d.x)*width+uint(int(x)+d.y)]<gray[uint(int(y)-d.x)*width+uint(int(x)-d.y)])<<(24u+k);}return bits;}
uint refinement_thermo8(uint v){uint c=min(v,8u);return (1u<<c)-1u;}
uint refinement_ortho_thermo_census(uint* gray,uint width,uint x,uint y){int ix=int(x),iy=int(y),w=int(width);int gxl=0,gxr=0;uint lt=0,lb=0;[unroll]for(int dy=-2;dy<=2;++dy){int row=(iy+dy)*w+ix;uint p_m2=gray[row-2],p_m1=gray[row-1],p_0=gray[row],p_p1=gray[row+1],p_p2=gray[row+2];gxl+=int(p_0)-int(p_m2);gxr+=int(p_p2)-int(p_0);if(dy<=0)lt+=p_m2+p_m1+p_0+p_p1+p_p2;if(dy>=0)lb+=p_m2+p_m1+p_0+p_p1+p_p2;}uint q_gxl=uint(clamp((gxl+60)/15,0,8));uint q_gxr=uint(clamp((gxr+60)/15,0,8));uint q_lt=min(lt/(15u*28u),8u);uint q_lb=min(lb/(15u*28u),8u);return refinement_thermo8(q_gxl)|(refinement_thermo8(q_gxr)<<8)|(refinement_thermo8(q_lt)<<16)|(refinement_thermo8(q_lb)<<24);}
bool refinement_support(uint* support,uint width,uint height,uint x,uint y){if(x<2||y<2||x+2>=width||y+2>=height)return false;[unroll]for(int dy=-2;dy<=2;++dy)[unroll]for(int dx=-2;dx<=2;++dx)if(support[uint(int(y)+dy)*width+uint(int(x)+dx)]==0)return false;return true;}
// Experimental refinement pins the nominal binary32 pixel-coordinate graph.
// Keep frozen SGM/Depth observation_target semantics unchanged.
bool refinement_target(uint width,uint x,float disparity,bool right,out uint target){
    target=0;if(!isfinite(disparity)||abs(disparity)>float(width+1))return false;
    SF_PRECISE float shifted=right?float(x)+disparity:float(x)-disparity;
    SF_PRECISE float rounded=shifted+.5f;
    float coordinate=floor(rounded);
    if(coordinate<0||coordinate>=float(width))return false;
    target=uint(coordinate);return true;
}
#endif
