#pragma once
// CPU oracle and caller-owned storage contract for a shared, GPU-resident stereo
// feature atlas. One 16-byte record per pixel per eye. No heap allocation here.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <span>
#include <stdexcept>

namespace stereoforge {

struct alignas(16) StereoFeaturePixel {
    std::uint32_t census24 = 0;
    std::uint32_t sobel_l1 = 0;
    std::uint32_t texture_l1 = 0;
    std::uint32_t valid = 0;
};
static_assert(sizeof(StereoFeaturePixel) == 16);

inline void stereo_feature_atlas_reference(
    std::span<const std::uint32_t> gray,
    std::span<const std::uint32_t> support,
    std::uint32_t width,
    std::uint32_t height,
    std::span<StereoFeaturePixel> output) {

    const auto n = std::uint64_t(width) * height;
    if (!width || !height || n > std::numeric_limits<std::uint32_t>::max() ||
        gray.size() != n || support.size() != n || output.size() != n)
        throw std::invalid_argument("feature atlas extent mismatch");

    const auto at = [&](std::uint32_t x,std::uint32_t y) -> std::uint32_t {
        return gray[std::size_t(y) * width + x];
    };
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const auto index = std::size_t(y) * width + x;
            auto& pixel = output[index];
            pixel = {};
            if (x < 2 || y < 2 || x + 2 >= width || y + 2 >= height)
                continue;
            bool complete=true;
            for (int dy=-2;dy<=2;++dy)
                for (int dx=-2;dx<=2;++dx) {
                    const auto pos=(std::size_t(int(y)+dy)*width)+std::size_t(int(x)+dx);
                    if (support[pos]!=1 || gray[pos]>255) complete=false;
                }
            if (!complete) continue;
            const auto center=at(x,y);
            std::uint32_t bits=0;
            unsigned bit=0;
            for (int dy=-2;dy<=2;++dy)
                for (int dx=-2;dx<=2;++dx) {
                    if (dx==0&&dy==0) continue;
                    if (gray[std::size_t(int(y)+dy)*width+std::size_t(int(x)+dx)]<center)
                        bits|=1u<<bit;
                    ++bit;
                }
            const int gx =
                int(at(x+1,y-1))+2*int(at(x+1,y))+int(at(x+1,y+1)) -
                int(at(x-1,y-1))-2*int(at(x-1,y))-int(at(x-1,y+1));
            const int gy =
                int(at(x-1,y+1))+2*int(at(x,y+1))+int(at(x+1,y+1)) -
                int(at(x-1,y-1))-2*int(at(x,y-1))-int(at(x+1,y-1));
            std::uint32_t texture=0;
            for(int dy=-1;dy<=1;++dy)
                for(int dx=-1;dx<=1;++dx)
                    if(dx||dy)
                        texture+=std::uint32_t(std::abs(
                            int(gray[std::size_t(int(y)+dy)*width+std::size_t(int(x)+dx)])-int(center)));
            pixel={bits,std::uint32_t(std::abs(gx)+std::abs(gy)),texture,1};
        }
    }
}

} // namespace stereoforge
