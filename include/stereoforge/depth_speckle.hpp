#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>
namespace stereoforge {
struct DepthSpeckleConfig {
    std::uint32_t width=0,height=0,min_component_pixels=50;
    float max_neighbor_delta_m=0.05f;
    float fx_baseline_px_m=0.0f;
    float max_neighbor_disparity_px=0.0f;
    bool eight_connected=false;
    std::uint32_t min_local_support_neighbors=0;
    float local_smooth_disparity_px=0.0f;
    float max_depth_m=0.0f;
    std::uint32_t min_local_luma_range=0;
};
struct DepthSpeckleStats { std::uint64_t input_valid_pixels=0,output_valid_pixels=0,invalidated_pixels=0,removed_components=0,retained_components=0; };
class DepthSpeckleFilter {
    DepthSpeckleConfig c_;
    std::vector<std::uint8_t> seen_;
    std::vector<std::uint8_t> luma_;
    std::vector<std::uint32_t> queue_;
    std::vector<float> values_;
    std::vector<float> scratch_;
public:
    explicit DepthSpeckleFilter(DepthSpeckleConfig c):c_(c) {
        if(!c.width||!c.height||c.width>8192||c.height>8192||
           std::uint64_t(c.width)*c.height>16'777'216ull||
           !c.min_component_pixels||c.min_component_pixels>std::uint64_t(c.width)*c.height||
           !std::isfinite(c.max_neighbor_delta_m)||c.max_neighbor_delta_m<0||
           !std::isfinite(c.fx_baseline_px_m)||c.fx_baseline_px_m<0.f||
           !std::isfinite(c.max_neighbor_disparity_px)||c.max_neighbor_disparity_px<0.f||
           c.min_local_support_neighbors>8||
           !std::isfinite(c.local_smooth_disparity_px)||c.local_smooth_disparity_px<0.f||
           !std::isfinite(c.max_depth_m)||c.max_depth_m<0.f||
           c.min_local_luma_range>255)
            throw std::invalid_argument("invalid depth speckle configuration");
        const auto size=std::size_t(c.width)*c.height;
        seen_.resize(size);luma_.resize(size);queue_.resize(size);values_.resize(size);scratch_.resize(size);
    }
    DepthSpeckleStats apply(std::span<std::byte> image, std::span<const std::byte> rgba = {}) {
        const auto count=std::size_t(c_.width)*c_.height;
        if(image.size()!=count*sizeof(float))throw std::invalid_argument("32FC1 depth byte extent mismatch");
        const bool has_rgba=!rgba.empty();
        if(has_rgba&&rgba.size()!=count*4)throw std::invalid_argument("RGBA8 byte extent mismatch");
        const bool disparity_domain=c_.fx_baseline_px_m>0.f&&c_.max_neighbor_disparity_px>0.f;
        const float max_delta=disparity_domain?c_.max_neighbor_disparity_px:c_.max_neighbor_delta_m;
        const float invalid=std::numeric_limits<float>::quiet_NaN();
        if(has_rgba&&c_.min_local_luma_range>0) {
            const auto* px=reinterpret_cast<const std::uint8_t*>(rgba.data());
            for(std::size_t i=0;i<count;++i) {
                const unsigned r=px[i*4],g=px[i*4+1],b=px[i*4+2];
                luma_[i]=static_cast<std::uint8_t>((r+(g<<1)+b)>>2);
            }
        }
        DepthSpeckleStats result;
        for(std::size_t i=0;i<count;++i) {
            float z=0.f;
            std::memcpy(&z,image.data()+i*sizeof(float),sizeof(float));
            if(std::isfinite(z)&&z>0.f) {
                ++result.input_valid_pixels;
                if(c_.max_depth_m>0.f&&z>c_.max_depth_m) {
                    values_[i]=invalid;
                    ++result.invalidated_pixels;
                    std::memcpy(image.data()+i*sizeof(float),&invalid,sizeof(float));
                } else {
                    values_[i]=disparity_domain?(c_.fx_baseline_px_m/z):z;
                }
            } else {
                values_[i]=invalid;
            }
        }
        if(has_rgba&&c_.min_local_luma_range>0) {
            const std::uint32_t w=c_.width,h=c_.height;
            for(std::uint32_t y=0;y<h;++y) {
                const std::uint32_t y0=y>=2?y-2:0,y1=y+2<h?y+2:h-1;
                for(std::uint32_t x=0;x<w;++x) {
                    const std::size_t idx=std::size_t(y)*w+x;
                    if(!std::isfinite(values_[idx])||values_[idx]<=0.f)continue;
                    const std::uint32_t x0=x>=2?x-2:0,x1=x+2<w?x+2:w-1;
                    std::uint8_t lo=255,hi=0;
                    for(std::uint32_t ny=y0;ny<=y1;++ny) {
                        const auto* row=luma_.data()+std::size_t(ny)*w;
                        for(std::uint32_t nx=x0;nx<=x1;++nx) {
                            const std::uint8_t v=row[nx];
                            if(v<lo)lo=v;
                            if(v>hi)hi=v;
                        }
                    }
                    if(unsigned(hi-lo)<c_.min_local_luma_range) {
                        values_[idx]=invalid;
                        ++result.invalidated_pixels;
                        std::memcpy(image.data()+idx*sizeof(float),&invalid,sizeof(float));
                    }
                }
            }
        }
        if(disparity_domain&&(c_.min_local_support_neighbors>0||c_.local_smooth_disparity_px>0.f)) {
            const std::uint32_t w=c_.width,h=c_.height;
            for(std::uint32_t y=0;y<h;++y) {
                const std::uint32_t y0=y?y-1:0,y1=y+1<h?y+1:h-1;
                for(std::uint32_t x=0;x<w;++x) {
                    const std::size_t idx=std::size_t(y)*w+x;
                    const float center=values_[idx];
                    if(!std::isfinite(center)||center<=0.f){scratch_[idx]=invalid;continue;}
                    const std::uint32_t x0=x?x-1:0,x1=x+1<w?x+1:w-1;
                    std::uint32_t agree=0;
                    float sum=center,weight=1.f;
                    for(std::uint32_t ny=y0;ny<=y1;++ny) {
                        for(std::uint32_t nx=x0;nx<=x1;++nx) {
                            if(nx==x&&ny==y)continue;
                            const float nb=values_[std::size_t(ny)*w+nx];
                            if(!std::isfinite(nb)||nb<=0.f)continue;
                            const float diff=std::fabs(nb-center);
                            if(diff<=max_delta)++agree;
                            if(c_.local_smooth_disparity_px>0.f&&diff<=c_.local_smooth_disparity_px) {
                                sum+=0.5f*nb;
                                weight+=0.5f;
                            }
                        }
                    }
                    if(agree<c_.min_local_support_neighbors) {
                        scratch_[idx]=invalid;
                        ++result.invalidated_pixels;
                        std::memcpy(image.data()+idx*sizeof(float),&invalid,sizeof(float));
                    } else {
                        const float refined=sum/weight;
                        scratch_[idx]=refined;
                        if(c_.local_smooth_disparity_px>0.f&&refined>0.f) {
                            const float z_refined=c_.fx_baseline_px_m/refined;
                            std::memcpy(image.data()+idx*sizeof(float),&z_refined,sizeof(float));
                        }
                    }
                }
            }
            values_.swap(scratch_);
        }
        std::fill(seen_.begin(),seen_.end(),0);
        for(std::uint32_t start=0;start<count;++start) {
            if(seen_[start]||!std::isfinite(values_[start])||values_[start]<=0.f)continue;
            std::size_t head=0,tail=1;queue_[0]=start;seen_[start]=1;
            while(head<tail) {
                const auto at=queue_[head++];const auto x=at%c_.width,y=at/c_.width;
                auto visit=[&](std::size_t next) {
                    if(seen_[next])return;
                    const float other=values_[next];
                    if(!std::isfinite(other)||other<=0.f||
                       std::fabs(other-values_[at])>max_delta)return;
                    seen_[next]=1;queue_[tail++]=std::uint32_t(next);
                };
                if(x)visit(at-1);
                if(x+1<c_.width)visit(at+1);
                if(y)visit(at-c_.width);
                if(y+1<c_.height)visit(at+c_.width);
                if(c_.eight_connected) {
                    if(x&&y)visit(at-c_.width-1);
                    if(x+1<c_.width&&y)visit(at-c_.width+1);
                    if(x&&y+1<c_.height)visit(at+c_.width-1);
                    if(x+1<c_.width&&y+1<c_.height)visit(at+c_.width+1);
                }
            }
            if(tail>=c_.min_component_pixels) {++result.retained_components;continue;}
            ++result.removed_components;result.invalidated_pixels+=tail;
            for(std::size_t k=0;k<tail;++k)
                std::memcpy(image.data()+std::size_t(queue_[k])*sizeof(float),&invalid,sizeof(float));
        }
        result.output_valid_pixels=result.input_valid_pixels-result.invalidated_pixels;
        return result;
    }
};
} // namespace stereoforge
