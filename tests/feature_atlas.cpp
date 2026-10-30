#include <stereoforge/feature_atlas.hpp>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
using namespace stereoforge::features;
int main() {
    constexpr std::uint32_t w=9,h=8;
    std::vector<std::uint8_t> src(h*(w+3),253);
    for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)
        src[y*(w+3)+x]=std::uint8_t(x*7+y*3);
    auto a=build(src,w,h,w+3);
    assert(a.width==w&&a.height==h);
    assert(a.luma[3*w+4]==37);
    for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x) {
        const auto i=y*w+x;
        const bool supported=x>=2&&x+2<w&&y>=2&&y+2<h;
        assert(bool(a.valid[i])==supported);
        if (!supported)assert(a.census[i]==0);
        if (x>0&&x+1<w&&y>0&&y+1<h)
            assert(a.gradient[i]==20); // |7-(-7)| + |3-(-3)|
    }
    auto b=build(src,w,h,w+3);
    assert(equivalent(a,b));
    b.census[3*w+4]^=1u;
    assert(!equivalent(a,b));
    assert(hamming24(0,0x00ffffffu)==24);
    assert(hamming24(0,0xff000000u)==0);
    bool rejected=false;
    try { (void)build(src,w,h,w-1); } catch(const std::invalid_argument&) {rejected=true;}
    assert(rejected);
    rejected=false;
    try { (void)build(src,w,h+100,w+3); } catch(const std::invalid_argument&) {rejected=true;}
    assert(rejected);
    std::vector<std::uint8_t> flat(25,0);
    auto equal=build(flat,5,5,5);
    assert(equal.valid[2*5+2]==1&&equal.census[2*5+2]==0);
    flat[2*5+2]=200;
    auto high=build(flat,5,5,5);
    assert(high.census[2*5+2]==0x00ffffffu);
}
