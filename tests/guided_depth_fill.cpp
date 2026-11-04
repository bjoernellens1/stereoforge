#include <stereoforge/guided_depth_fill.hpp>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>
using namespace stereoforge;
int main(){
    constexpr unsigned w=40,h=9;
    const float nan=std::numeric_limits<float>::quiet_NaN();
    std::vector<float> d(w*h,nan);
    std::vector<std::byte> rgba(w*h*4);
    for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){
        const std::uint8_t l=x<20?200:40;
        for(unsigned c=0;c<3;++c)rgba[(y*w+x)*4+c]=static_cast<std::byte>(l);
        rgba[(y*w+x)*4+3]=static_cast<std::byte>(255);
        const bool anchor=(x==0||x==w-1||y==0||y==h-1);
        if(anchor)d[y*w+x]=x<20?1.8f:0.7f;
    }
    GuidedDepthFill fill({w,h,48});
    std::vector<std::uint8_t> filled;
    const auto n=fill.apply(d,rgba,filled);
    assert(n>0);
    // Wall holes copy wall depth, object holes copy object depth; the edge at x=20 is never bridged.
    assert(std::abs(d[4*w+5]-1.8f)<1e-4f&&filled[4*w+5]==1);
    assert(std::abs(d[4*w+34]-0.7f)<1e-4f&&filled[4*w+34]==1);
    for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){
        if(!filled[y*w+x])continue;
        assert(x<20?std::abs(d[y*w+x]-1.8f)<1e-4f:std::abs(d[y*w+x]-0.7f)<1e-4f);
    }
    // A hole reachable by a single ray only is left invalid.
    std::vector<float> one(w*h,nan);one[4*w+0]=1.f;
    GuidedDepthFill single({w,h,48});
    single.apply(one,rgba,filled);
    assert(!std::isfinite(one[4*w+3])&&filled[4*w+3]==0);
    bool rejected=false;try{fill.apply(std::span<float>(d.data(),3),rgba,filled);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
    rejected=false;try{GuidedDepthFill bad({w,h,0});}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
    return 0;
}
