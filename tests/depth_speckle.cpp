#include <stereoforge/depth_speckle.hpp>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
using namespace stereoforge;
int main(){
    constexpr unsigned w=12,h=10;
    std::vector<float> d(w*h,std::numeric_limits<float>::quiet_NaN());
    for(unsigned y=2;y<8;++y)for(unsigned x=2;x<9;++x)d[y*w+x]=1.25f;
    d[0]=3.7f;d[1]=3.71f;d[9*w+11]=0.f;
    std::vector<std::byte> bytes(d.size()*sizeof(float));
    std::memcpy(bytes.data(),d.data(),bytes.size());
    DepthSpeckleFilter filter({w,h,10,.05f});
    auto stats=filter.apply(bytes);
    assert(stats.removed_components==1&&stats.invalidated_pixels==2);
    assert(stats.input_valid_pixels==44&&stats.output_valid_pixels==42);
    assert(stats.retained_components==1);
    std::memcpy(d.data(),bytes.data(),bytes.size());
    assert(std::isnan(d[0])&&std::isnan(d[1]));
    assert(d[3*w+3]==1.25f&&d[5*w+6]==1.25f);
    assert(d[9*w+11]==0.f);
    stats=filter.apply(bytes);assert(stats.invalidated_pixels==0);
    assert(stats.input_valid_pixels==42&&stats.output_valid_pixels==42);
    bool rejected=false;try{filter.apply(std::span<std::byte>(bytes.data(),3));}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
    rejected=false;try{DepthSpeckleFilter f({w,h,10,-1.f});}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
    std::vector<float> step(w*h,1.f);
    for(unsigned y=0;y<h;++y)for(unsigned x=6;x<w;++x)step[y*w+x]=2.f;
    std::memcpy(bytes.data(),step.data(),bytes.size());
    stats=filter.apply(bytes);assert(stats.removed_components==0&&stats.retained_components==2);
    assert(stats.input_valid_pixels==120&&stats.output_valid_pixels==120);
}
