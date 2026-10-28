#ifndef SF_REFINEMENT_OPS
#define SF_REFINEMENT_OPS
// Integer diagnostics are a stable wire contract. NaN is canonical on invalid outputs.
static const uint REF_NO_PRIOR=1,REF_LOW_CONFIDENCE=2,REF_PRIOR_SPAN=4,REF_NONFINITE=8,REF_OUTSIDE=16,REF_UNSUPPORTED=32,REF_TOO_FEW=64,REF_AMBIGUOUS=128,REF_RESTRICTED=256,REF_GLOBAL=512,REF_MISSING_SUBPIXEL=1024,REF_LR_FAILED=2048;
float refinement_invalid(){return asfloat(0x7fc00000);}
uint refinement_census(uint* gray,uint width,uint x,uint y){uint centre=gray[y*width+x],bits=0,bit=0;[unroll]for(int dy=-2;dy<=2;++dy)[unroll]for(int dx=-2;dx<=2;++dx){if(dx==0&&dy==0)continue;bits|=uint(gray[uint(int(y)+dy)*width+uint(int(x)+dx)]<centre)<<(bit++);}return bits;}
bool refinement_support(uint* support,uint width,uint height,uint x,uint y){if(x<2||y<2||x+2>=width||y+2>=height)return false;[unroll]for(int dy=-2;dy<=2;++dy)[unroll]for(int dx=-2;dx<=2;++dx)if(support[uint(int(y)+dy)*width+uint(int(x)+dx)]==0)return false;return true;}
// Experimental refinement pins the nominal binary32 pixel-coordinate graph.
// Keep frozen SGM/Depth observation_target semantics unchanged.
bool refinement_target(uint width,uint x,float disparity,bool right,out uint target){
    target=0;if(!isfinite(disparity)||abs(disparity)>float(width+1))return false;
    precise float shifted=right?float(x)+disparity:float(x)-disparity;
    precise float rounded=shifted+.5f;
    float coordinate=floor(rounded);
    if(coordinate<0||coordinate>=float(width))return false;
    target=uint(coordinate);return true;
}
#endif
