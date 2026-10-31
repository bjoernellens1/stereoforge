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
struct DepthSpeckleConfig { std::uint32_t width=0,height=0,min_component_pixels=50; float max_neighbor_delta_m=0.05f; };
struct DepthSpeckleStats { std::uint64_t input_valid_pixels=0,output_valid_pixels=0,invalidated_pixels=0,removed_components=0,retained_components=0; };
class DepthSpeckleFilter {
    DepthSpeckleConfig c_;
    std::vector<std::uint8_t> seen_;
    std::vector<std::uint32_t> queue_;
    std::vector<float> values_;
public:
    explicit DepthSpeckleFilter(DepthSpeckleConfig c):c_(c) {
        if(!c.width||!c.height||c.width>8192||c.height>8192||
           std::uint64_t(c.width)*c.height>16'777'216ull||
           !c.min_component_pixels||c.min_component_pixels>std::uint64_t(c.width)*c.height||
           !std::isfinite(c.max_neighbor_delta_m)||c.max_neighbor_delta_m<0)
            throw std::invalid_argument("invalid depth speckle configuration");
        const auto size=std::size_t(c.width)*c.height;
        seen_.resize(size);queue_.resize(size);values_.resize(size);
    }
    DepthSpeckleStats apply(std::span<std::byte> image) {
        const auto count=std::size_t(c_.width)*c_.height;
        if(image.size()!=count*sizeof(float))throw std::invalid_argument("32FC1 depth byte extent mismatch");
        DepthSpeckleStats result;
        for(std::size_t i=0;i<count;++i) {
            std::memcpy(&values_[i],image.data()+i*sizeof(float),sizeof(float));
            if(std::isfinite(values_[i])&&values_[i]>0.f)++result.input_valid_pixels;
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
                       std::fabs(other-values_[at])>c_.max_neighbor_delta_m)return;
                    seen_[next]=1;queue_[tail++]=std::uint32_t(next);
                };
                if(x)visit(at-1);
                if(x+1<c_.width)visit(at+1);
                if(y)visit(at-c_.width);
                if(y+1<c_.height)visit(at+c_.width);
            }
            if(tail>=c_.min_component_pixels) {++result.retained_components;continue;}
            ++result.removed_components;result.invalidated_pixels+=tail;
            const float invalid=std::numeric_limits<float>::quiet_NaN();
            for(std::size_t k=0;k<tail;++k)
                std::memcpy(image.data()+std::size_t(queue_[k])*sizeof(float),&invalid,sizeof(float));
        }
        result.output_valid_pixels=result.input_valid_pixels-result.invalidated_pixels;
        return result;
    }
};
} // namespace stereoforge
