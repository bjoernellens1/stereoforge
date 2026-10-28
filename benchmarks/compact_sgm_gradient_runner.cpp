// Offline image-gradient-P2 ablation runner. Accepts explicit costs or
// computes Census costs from input descriptors. Not a production GPU implementation.
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr uint32_t INF = std::numeric_limits<uint32_t>::max();
template<class T> std::vector<T> read(const std::string& path, size_t n) {
    std::vector<T> v(n); std::ifstream f(path, std::ios::binary);
    if (!f.read(reinterpret_cast<char*>(v.data()), n * sizeof(T)) || f.peek() != EOF)
        throw std::runtime_error("incorrect input size: " + path);
    return v;
}
template<class T> void write(const std::string& path, const std::vector<T>& v) {
    std::ofstream f(path, std::ios::binary);
    if (!f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T)))
        throw std::runtime_error("write failed: " + path);
}
int main(int argc, char** argv) {
    try {
        if (argc != 7 && argc != 8) throw std::runtime_error("usage: runner directory H W D P1 P2 [eye]");
        std::string dir = argv[1];
        int h = std::stoi(argv[2]), w = std::stoi(argv[3]), d = std::stoi(argv[4]);
        int p1 = std::stoi(argv[5]), p2 = std::stoi(argv[6]);
        if (h <= 0 || w <= 0 || d <= 0 || p1 < 0 || p1 > p2 || p2 > 10000)
            throw std::runtime_error("invalid dimensions or penalties");
        size_t n = size_t(h)*w, nd = n*d;
        bool image = argc == 8;
        int eye = image ? std::stoi(argv[7]) : 0;
        if (eye != 0 && eye != 1) throw std::runtime_error("invalid eye");
        std::vector<uint32_t> cost, own, other, own_support, other_support;
        if (image) {
            own=read<uint32_t>(dir+"/own", n); other=read<uint32_t>(dir+"/other", n);
            own_support=read<uint32_t>(dir+"/own_support", n); other_support=read<uint32_t>(dir+"/other_support", n);
            for (auto v:own_support) if (v>1) throw std::runtime_error("invalid support");
            for (auto v:other_support) if (v>1) throw std::runtime_error("invalid support");
        } else cost=read<uint32_t>(dir+"/cost", nd);
        auto get_cost = [&](size_t i, uint32_t q) -> uint32_t {
            if (!image) return cost[i*d+q];
            int x=int(i%w), ox=eye ? x+int(q) : x-int(q);
            if (!own_support[i] || ox<0 || ox>=w) return INF;
            size_t j=(i/w)*w+ox;
            if (!other_support[j]) return INF;
            return std::popcount(own[i]^other[j]);
        };
        auto lo = read<uint32_t>(dir+"/lo", n), hi = read<uint32_t>(dir+"/hi", n);
        auto luma = read<uint32_t>(dir+"/luma", n);
        for (auto v:luma) if (v>255) throw std::runtime_error("invalid luma");
        for (size_t i=0;i<n;++i) if (lo[i]>hi[i] || hi[i]>uint32_t(d)) throw std::runtime_error("invalid interval");
        for (auto c:cost) if (c!=INF && c>10000) throw std::runtime_error("invalid cost");
        std::vector<uint32_t> paths(image ? 0 : 4*nd, INF), sums(nd, INF);
        uint64_t updates=0, slots=0;
        const int dx[4]={1,-1,0,0},dy[4]={0,0,1,-1};
        std::vector<uint32_t> prev(d,INF),cur(d,INF);
        for (int k=0;k<4;++k) for(int sy=0;sy<h;++sy) for(int sx=0;sx<w;++sx) {
            int px=sx-dx[k],py=sy-dy[k];
            if(px>=0&&px<w&&py>=0&&py<h) continue;
            std::fill(prev.begin(),prev.end(),INF);
            for(int x=sx,y=sy;x>=0&&x<w&&y>=0&&y<h;x+=dx[k],y+=dy[k]) {
                auto m=*std::min_element(prev.begin(),prev.end());
                std::fill(cur.begin(),cur.end(),INF);
                size_t i=size_t(y)*w+x;
                for(uint32_t q=lo[i];q<hi[i];++q) {
                    ++slots; uint32_t c=get_cost(i,q); if(c==INF) continue; ++updates;
                    uint32_t v=c;
                    if(m!=INF) {
                        int predecessor_x=x-dx[k], predecessor_y=y-dy[k];
                        uint32_t delta=std::abs(int(luma[i])-int(luma[size_t(predecessor_y)*w+predecessor_x]));
                        uint32_t adaptive_p2=std::max(uint32_t(p1), uint32_t(p2)*16/(16+delta));
                        uint32_t best=std::min(prev[q],m+adaptive_p2);
                        if(q && prev[q-1]!=INF) best=std::min(best,prev[q-1]+uint32_t(p1));
                        if(q+1<uint32_t(d) && prev[q+1]!=INF) best=std::min(best,prev[q+1]+uint32_t(p1));
                        v += best-m;
                    }
                    cur[q]=v;
                    size_t j=i*d+q;
                    if (!image) paths[size_t(k)*nd+j]=v;
                    if (k==0) sums[j]=v; else if (sums[j]!=INF) sums[j]+=v;
                }
                prev.swap(cur);
            }
        }
        std::vector<int32_t> winners(n,-1);
        std::vector<uint32_t> best(n, INF), second(n, INF), count(n, 0);
        for(size_t i=0;i<n;++i) for(int q=0;q<d;++q) {
            size_t j=i*d+q;
            uint32_t v=sums[j]; if(v==INF) continue;
            ++count[i];
            if(v<best[i]) { second[i]=best[i]; best[i]=v; winners[i]=q; }
            else if(v<second[i]) second[i]=v;
        }
        write(dir+"/best",best); write(dir+"/second",second); write(dir+"/count",count);
        std::ofstream stats(dir+"/stats"); stats << updates << " " << slots << "\n";

        if (!image) write(dir+"/paths",paths); if (!image || std::getenv("COMPACT_DUMP_SCORES")) write(dir+"/sums",sums); write(dir+"/winners",winners);
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
