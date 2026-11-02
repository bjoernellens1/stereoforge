// Binding-independent, exact uint32 Census/SGM scalar operations. The dense
// layout is row-major pixel, disparity fastest. Census is 5x5/24 bits, strict
// less-than, and the two-pixel border is unsupported. Invalid cost is 10000.
// P1<=P2<=10000, D<=256, 4/8 paths. Normalization keeps path values <=20000,
// so uint32 sums <=160000; no precision/packed-width change is authorized.
#ifndef SF_SGM_OPS
#define SF_SGM_OPS
static const uint SGM_INVALID_COST=10000;
static const uint SGM_REDUCTION_INF=1000000;
uint sgm_cost(uint reference,uint matching,bool supported){return supported?countbits(reference^matching):SGM_INVALID_COST;}
uint sgm_step(uint cost,uint own,uint left,uint right,uint minimum,uint p1,uint p2,bool first){
    uint best=min(own,minimum+p2);best=min(best,left+p1);best=min(best,right+p1);
    return cost+(first?0:best-minimum);
}
int2 sgm_direction(uint path){
    const int2 directions[8]={int2(1,0),int2(-1,0),int2(0,1),int2(0,-1),int2(1,1),int2(-1,-1),int2(1,-1),int2(-1,1)};
    return directions[path];
}
uint sgm_line_count(uint width,uint height,int2 dir){return dir.y==0?height:dir.x==0?width:width+height-1;}
int2 sgm_line_start(uint width,uint height,int2 dir,uint line){
    if(dir.y==0)return int2(dir.x>0?0:int(width)-1,int(line));
    if(dir.x==0)return int2(int(line),dir.y>0?0:int(height)-1);
    if(line<width)return int2(int(line),dir.y>0?0:int(height)-1);
    return int2(dir.x>0?0:int(width)-1,dir.y>0?int(line-width)+1:int(height)-2-int(line-width));
}
bool sgm_inside(int2 xy,uint width,uint height){return xy.x>=0&&xy.y>=0&&xy.x<int(width)&&xy.y<int(height);}
uint sgm_census(uint* image,uint width,uint height,uint i){
    int x=int(i%width),y=int(i/width);uint value=0,bit=0;
    for(int oy=-2;oy<=2;++oy)for(int ox=-2;ox<=2;++ox){if(ox==0&&oy==0)continue;if(image[(y+oy)*int(width)+x+ox]<image[i])value|=1u<<bit;++bit;}
    const int2 sym[8]={int2(0,1),int2(0,-1),int2(1,0),int2(-1,0),int2(0,2),int2(0,-2),int2(2,0),int2(-2,0)};
    for(uint k=0;k<8;++k){int2 d=sym[k];if(image[(y+d.x)*int(width)+x+d.y]<image[(y-d.x)*int(width)+x-d.y])value|=1u<<(24u+k);}
    return value;
}
uint sgm_thermo8(uint v){uint c=min(v,8u);return (1u<<c)-1u;}
uint sgm_ortho_thermo_census(uint* image,uint width,uint height,uint i){
    int x=int(i%width),y=int(i/width),w=int(width);
    int gxl=0,gxr=0;uint lt=0,lb=0;
    [unroll]for(int dy=-2;dy<=2;++dy){
        int row=(y+dy)*w+x;
        uint p_m2=image[row-2],p_m1=image[row-1],p_0=image[row],p_p1=image[row+1],p_p2=image[row+2];
        gxl+=int(p_0)-int(p_m2);gxr+=int(p_p2)-int(p_0);
        if(dy<=0)lt+=p_m2+p_m1+p_0+p_p1+p_p2;
        if(dy>=0)lb+=p_m2+p_m1+p_0+p_p1+p_p2;
    }
    uint q_gxl=uint(clamp((gxl+60)/15,0,8));
    uint q_gxr=uint(clamp((gxr+60)/15,0,8));
    uint q_lt=min(lt/(15u*28u),8u);
    uint q_lb=min(lb/(15u*28u),8u);
    return sgm_thermo8(q_gxl)|(sgm_thermo8(q_gxr)<<8)|(sgm_thermo8(q_lt)<<16)|(sgm_thermo8(q_lb)<<24);
}
bool sgm_census_supported(uint width,uint height,uint i){uint x=i%width,y=i/width;return x>=2&&y>=2&&x<width-2&&y<height-2;}
bool sgm_selection_valid(uint best,uint best_cost,uint second){return best>0&&best_cost<SGM_INVALID_COST&&second>best_cost;}
float sgm_subpixel(uint best,uint best_cost,uint lo,uint hi){
    // Preserve the frozen RHI control flow so compiler contraction retains its
    // execution profile: the conditional controls the update, not a ratio phi.
    float value=float(best);
    float low=float(lo),high=float(hi);
    float denominator=low+high-2*float(best_cost);
    if(denominator>0)value+=.5*(low-high)/denominator;
    return value;
}
#endif
