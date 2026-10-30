#include <stereoforge/feature_atlas.hpp>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
using namespace stereoforge;
namespace {
template<class F> bool fails(F f){try{f();}catch(const std::invalid_argument&){return true;}return false;}
void constant_and_support(){
    constexpr unsigned w=9,h=7;
    std::vector<std::uint32_t> gray(w*h,100),support(w*h,1);
    std::vector<StereoFeaturePixel> out(w*h);
    stereo_feature_atlas_reference(gray,support,w,h,out);
    for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){
        const auto &p=out[y*w+x];
        if(x<2||y<2||x+2>=w||y+2>=h){
            assert(!p.valid&&!p.census24&&!p.sobel_l1&&!p.texture_l1);
        }else{
            assert(p.valid==1&&!p.census24&&!p.sobel_l1&&!p.texture_l1);
        }
    }
    support[3*w+4]=0;
    stereo_feature_atlas_reference(gray,support,w,h,out);
    assert(out[3*w+4].valid==0);
    assert(out[2*w+2].valid==0); // its 5x5 support includes the hole.
}
void ramp_and_census(){
    constexpr unsigned w=9,h=9;
    std::vector<std::uint32_t> gray(w*h),support(w*h,1);
    for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)gray[y*w+x]=x*10;
    std::vector<StereoFeaturePixel> out(w*h);
    stereo_feature_atlas_reference(gray,support,w,h,out);
    const auto p=out[4*w+4];
    assert(p.valid==1 && p.sobel_l1==80 && p.texture_l1==60);
    unsigned bit=0;std::uint32_t expected=0;
    for(int dy=-2;dy<=2;++dy)for(int dx=-2;dx<=2;++dx){
        if(!dx&&!dy)continue;
        if(dx<0)expected|=1u<<bit;
        ++bit;
    }
    assert(p.census24==expected);
}
void all_smaller_and_invalid(){
    for(unsigned h=1;h<=4;++h){
        std::vector<std::uint32_t> g(7*h,50),s(7*h,1);
        std::vector<StereoFeaturePixel> out(7*h);
        stereo_feature_atlas_reference(g,s,7,h,out);
        for(auto const& p:out)assert(p.valid==0);
    }
    std::vector<std::uint32_t> g(7*7,0),s(7*7,1);
    std::vector<StereoFeaturePixel> out(7*7);
    g[3*7+3]=256;
    stereo_feature_atlas_reference(g,s,7,7,out);
    assert(out[3*7+3].valid==0);
    assert(fails([&]{stereo_feature_atlas_reference(g,s,7,8,out);}));
    assert(fails([&]{stereo_feature_atlas_reference(g,s,0,7,out);}));
    assert(fails([&]{stereo_feature_atlas_reference(g,s,UINT32_MAX,UINT32_MAX,out);}));
}
}
int main(){constant_and_support();ramp_and_census();all_smaller_and_invalid();}
