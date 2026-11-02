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

    // Disparity-domain 8-connected test: slanted surface from Z=2.5m to Z=4.8m stays connected
    constexpr float fx_b=63.041385f;
    std::vector<float> slant(w*h,std::numeric_limits<float>::quiet_NaN());
    for(unsigned y=1;y<6;++y)for(unsigned x=0;x<w;++x) {
        const float disp=25.0f-0.9f*float(x);
        slant[y*w+x]=fx_b/disp;
    }
    // Diagonal 3-pixel speckle at Z=1.1m
    slant[7*w+1]=1.1f; slant[8*w+2]=1.1f; slant[9*w+3]=1.1f;
    std::memcpy(bytes.data(),slant.data(),bytes.size());
    DepthSpeckleFilter disp_filter({w,h,20,0.05f,fx_b,1.25f,true});
    stats=disp_filter.apply(bytes);
    assert(stats.input_valid_pixels==63&&stats.output_valid_pixels==60);
    assert(stats.retained_components==1&&stats.removed_components==1&&stats.invalidated_pixels==3);
    // 3x3 local support severs 1-pixel filament bridges before BFS
    std::vector<float> bridged(w*h,std::numeric_limits<float>::quiet_NaN());
    for(unsigned y=1;y<4;++y)for(unsigned x=1;x<4;++x)bridged[y*w+x]=2.0f; // 9 px island
    bridged[2*w+4]=2.0f; bridged[2*w+5]=2.0f; // 1-px wide bridge (each has <=2 neighbors)
    for(unsigned y=1;y<4;++y)for(unsigned x=6;x<9;++x)bridged[y*w+x]=2.0f; // 9 px island (total 20 px if bridged)
    std::memcpy(bytes.data(),bridged.data(),bytes.size());
    DepthSpeckleFilter support_filter({w,h,15,0.05f,fx_b,1.25f,true,3,1.0f});
    stats=support_filter.apply(bytes);
    assert(stats.input_valid_pixels==20&&stats.output_valid_pixels==0);
    // Max depth ceiling & 5x5 local luma contrast gate reject textureless / far-range blobs
    std::vector<float> far_or_flat(w*h,2.0f);
    for(unsigned i=0;i<w*h/2;++i)far_or_flat[i]=25.0f; // beyond max_depth_m=8.0
    std::vector<std::uint8_t> rgba(w*h*4,20); // completely flat luma=20 everywhere
    std::memcpy(bytes.data(),far_or_flat.data(),bytes.size());
    DepthSpeckleFilter gate_filter({w,h,10,0.05f,fx_b,1.25f,true,3,0.8f,8.0f,6u});
    stats=gate_filter.apply(bytes,std::as_bytes(std::span(rgba)));
    assert(stats.input_valid_pixels==w*h&&stats.output_valid_pixels==0);
    rejected=false;try{DepthSpeckleFilter f({w,h,10,0.05f,-1.f,1.f,true});}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
    rejected=false;try{DepthSpeckleFilter f({w,h,10,0.05f,fx_b,1.f,true,9,0.f});}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
}
