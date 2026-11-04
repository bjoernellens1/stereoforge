#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>
namespace stereoforge {
struct GuidedDepthFillConfig {
    std::uint32_t width=0,height=0,max_distance_px=48;
    // A ray stops when luma jumps between neighbours (object edge) or drifts away from the anchor.
    std::uint32_t step_luma_threshold=10,anchor_luma_threshold=28;
    float agreement_tolerance=0.04f;
    std::uint32_t min_agreeing_rays=3;
};
// Edge-aware hole filling of a 32FC1 depth image. Each invalid pixel looks along the four axes for the nearest
// valid depth reachable without crossing an RGB edge and is filled only when at least two rays agree on one
// surface, so flat walls inherit depth from their textured parts and silhouettes are never bridged.
class GuidedDepthFill {
    GuidedDepthFillConfig c_;
    std::vector<std::uint8_t> luma_,dist_[4];
    std::vector<float> cand_[4];
    struct Ray {
        float anchor=std::numeric_limits<float>::quiet_NaN();
        std::uint8_t anchor_luma=0,last_luma=0;std::uint32_t run=0;
    };
    // Advances one ray over pixel i (depth d, luma l) and records the candidate for direction `dir`.
    void step(Ray& r,std::size_t dir,std::size_t i,float d) {
        const int l=luma_[i];
        float out=std::numeric_limits<float>::quiet_NaN();
        if(d>0.f&&std::isfinite(d)) {
            r.anchor=d;r.anchor_luma=static_cast<std::uint8_t>(l);r.run=0;
        } else if(std::isfinite(r.anchor)) {
            if(++r.run>c_.max_distance_px||std::abs(l-int(r.last_luma))>int(c_.step_luma_threshold)||
               std::abs(l-int(r.anchor_luma))>int(c_.anchor_luma_threshold))
                r.anchor=std::numeric_limits<float>::quiet_NaN();
            else {out=r.anchor;dist_[dir][i]=static_cast<std::uint8_t>(r.run);}
        }
        cand_[dir][i]=out;
        r.last_luma=static_cast<std::uint8_t>(l);
    }
public:
    explicit GuidedDepthFill(GuidedDepthFillConfig c):c_(c) {
        if(!c.width||!c.height||c.width>8192||c.height>8192||!c.max_distance_px||c.max_distance_px>255||
           c.step_luma_threshold>255||c.anchor_luma_threshold>255||!std::isfinite(c.agreement_tolerance)||c.agreement_tolerance<=0.f)
            throw std::invalid_argument("invalid guided depth fill configuration");
        const auto n=std::size_t(c.width)*c.height;
        luma_.resize(n);
        for(auto& v:cand_)v.resize(n);
        for(auto& v:dist_)v.resize(n);
    }
    // Fills `depth` in place, sets filled[i]=1 for filled pixels (filled is resized/cleared) and returns the count.
    std::size_t apply(std::span<float> depth,std::span<const std::byte> rgba,std::vector<std::uint8_t>& filled) {
        const auto w=std::size_t(c_.width),h=std::size_t(c_.height),n=w*h;
        if(depth.size()!=n||rgba.size()!=n*4)throw std::invalid_argument("guided depth fill extent mismatch");
        const auto* px=reinterpret_cast<const std::uint8_t*>(rgba.data());
        const float* d_in=depth.data();
        filled.assign(n,0);
        const unsigned bands=std::max(1u,std::min(4u,std::thread::hardware_concurrency()));
        std::vector<std::size_t> counts(bands,0);
        auto parallel=[&](auto&& fn) {
            std::vector<std::thread> pool;
            for(unsigned b=1;b<bands;++b)pool.emplace_back(fn,b);
            fn(0u);
            for(auto& t:pool)t.join();
        };
        auto span_of=[&](unsigned b,std::size_t len){return std::pair<std::size_t,std::size_t>{len*b/bands,len*(b+1)/bands};};
        parallel([&](unsigned b) {
            const auto [y0,y1]=span_of(b,h);
            for(std::size_t y=y0;y<y1;++y) {
                Ray left,right;
                const auto row=y*w;
                for(std::size_t x=0;x<w;++x) {const auto i=row+x;luma_[i]=static_cast<std::uint8_t>((px[i*4]*77u+px[i*4+1]*150u+px[i*4+2]*29u)>>8);}
                for(std::size_t x=0;x<w;++x)step(left,0,row+x,d_in[row+x]);
                for(std::size_t x=w;x-->0;)step(right,1,row+x,d_in[row+x]);
            }
        });
        // Vertical rays advance row by row within a column band to stay cache friendly.
        parallel([&](unsigned b) {
            const auto [x0,x1]=span_of(b,w);
            std::vector<Ray> up(x1-x0),down(x1-x0);
            for(std::size_t y=0;y<h;++y)for(std::size_t x=x0;x<x1;++x)step(up[x-x0],2,y*w+x,d_in[y*w+x]);
            for(std::size_t y=h;y-->0;)for(std::size_t x=x0;x<x1;++x)step(down[x-x0],3,y*w+x,d_in[y*w+x]);
        });
        parallel([&](unsigned b) {
            const auto [y0,y1]=span_of(b,h);
            for(std::size_t i=y0*w;i<y1*w;++i) {
                const float d0=depth[i];
                if(std::isfinite(d0)&&d0>0.f)continue;
                float best_sum=0,best_w=0;unsigned best_n=0;
                for(unsigned a=0;a<4;++a) {
                    const float da=cand_[a][i];
                    if(!std::isfinite(da))continue;
                    float sum=0,wt=0;unsigned cnt=0;
                    for(unsigned c=0;c<4;++c) {
                        const float db=cand_[c][i];
                        if(!std::isfinite(db)||std::abs(db-da)>c_.agreement_tolerance*da)continue;
                        const float wb=1.f/(1.f+float(dist_[c][i]));
                        sum+=db*wb;wt+=wb;++cnt;
                    }
                    if(cnt>best_n){best_n=cnt;best_sum=sum;best_w=wt;}
                }
                if(best_n>=c_.min_agreeing_rays){depth[i]=best_sum/best_w;filled[i]=1;++counts[b];}
            }
        });
        std::size_t count=0;
        for(auto c:counts)count+=c;
        return count;
    }
};
}
