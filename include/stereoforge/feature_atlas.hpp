#pragma once
// Portable CPU reference for a reusable luminance/gradient/Census feature atlas.
// This is a correctness oracle and API experiment; it is NOT a GPU speedup.
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace stereoforge::features {
struct Atlas {
    std::uint32_t width=0, height=0;
    // Row-major 8-bit luma, unsigned gradient magnitude, Census 24-neighbour bits,
    // and explicit 5x5 support. Zero Census on unsupported borders.
    std::vector<std::uint8_t> luma, gradient, valid;
    std::vector<std::uint32_t> census;
};
inline Atlas build(std::span<const std::uint8_t> luminance,
                   std::uint32_t width, std::uint32_t height,
                   std::size_t stride) {
    if (!width || !height || width > 8192 || height > 8192 ||
        stride < width || (height-1 > 0 && stride > (std::numeric_limits<std::size_t>::max()-width)/(height-1)) ||
        (height-1)*stride+width > luminance.size())
        throw std::invalid_argument("invalid atlas dimensions/stride/input extent");
    const std::size_t count=std::size_t(width)*height;
    Atlas out; out.width=width;out.height=height;
    out.luma.resize(count);out.gradient.assign(count,0);out.valid.assign(count,0);
    out.census.assign(count,0);
    for (std::uint32_t y=0;y<height;++y)
        for (std::uint32_t x=0;x<width;++x)
            out.luma[std::size_t(y)*width+x]=luminance[std::size_t(y)*stride+x];
    for (std::uint32_t y=1;y+1<height;++y)
        for (std::uint32_t x=1;x+1<width;++x) {
            const std::size_t i=std::size_t(y)*width+x;
            const int dx=int(out.luma[i+1])-int(out.luma[i-1]);
            const int dy=int(out.luma[i+width])-int(out.luma[i-width]);
            out.gradient[i]=std::uint8_t(std::min(255,std::abs(dx)+std::abs(dy)));
        }
    for (std::uint32_t y=2;y+2<height;++y)
        for (std::uint32_t x=2;x+2<width;++x) {
            const std::size_t i=std::size_t(y)*width+x;
            const auto centre=out.luma[i]; std::uint32_t bits=0; unsigned shift=0;
            for (int oy=-2;oy<=2;++oy)
                for (int ox=-2;ox<=2;++ox) {
                    if (ox==0 && oy==0)continue;
                    const auto j=std::size_t(int(y)+oy)*width+std::size_t(int(x)+ox);
                    bits |= std::uint32_t(out.luma[j]<centre)<<shift++;
                }
            out.census[i]=bits;out.valid[i]=1;
        }
    return out;
}
inline std::uint32_t hamming24(std::uint32_t a,std::uint32_t b) noexcept {
    return std::popcount((a^b)&0x00ffffffu);
}
inline bool equivalent(const Atlas& a,const Atlas& b) noexcept {
    return a.width==b.width&&a.height==b.height&&a.luma==b.luma&&
           a.gradient==b.gradient&&a.valid==b.valid&&a.census==b.census;
}
} // namespace stereoforge::features
