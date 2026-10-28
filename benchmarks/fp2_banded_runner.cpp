// Offline packed-band SGM reference; not a production GPU implementation.
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

static_assert(sizeof(size_t)==8, "packed offset wire requires 64-bit size_t");
static constexpr uint32_t INF=std::numeric_limits<uint32_t>::max();
template<class T> std::vector<T> read(const std::string& path,size_t n){
    if(n>std::numeric_limits<size_t>::max()/sizeof(T))throw std::runtime_error("read extent overflow");
    std::vector<T> v(n);std::ifstream f(path,std::ios::binary);
    if(!f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T))||f.peek()!=EOF)throw std::runtime_error("invalid input size: "+path);
    return v;
}
template<class T> void write(const std::string& path,const std::vector<T>& v){std::ofstream f(path,std::ios::binary);if(!f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T)))throw std::runtime_error("write failed");}
static uint32_t bucket(uint32_t width){if(!width)return 0;if(width<=16)return 16;if(width<=32)return 32;if(width<=64)return 64;if(width<=128)return 128;throw std::runtime_error("band exceeds 128");}
int main(int argc,char** argv){try{
    if(argc!=7&&argc!=8)throw std::runtime_error("usage: runner dir H W D P1 P2 [eye]");
    std::string dir=argv[1];int h=std::stoi(argv[2]),w=std::stoi(argv[3]),d=std::stoi(argv[4]),p1=std::stoi(argv[5]),p2=std::stoi(argv[6]);
    if(h<=0||w<=0||d<=0||d>128||p1<0||p1>p2||p2>10000)throw std::runtime_error("invalid dimensions or penalties");
    if(size_t(h)>std::numeric_limits<size_t>::max()/size_t(w))throw std::runtime_error("dimension overflow");
    size_t n=size_t(h)*size_t(w);bool image=argc==8;int eye=image?std::stoi(argv[7]):0;if(eye!=0&&eye!=1)throw std::runtime_error("bad eye");
    auto lo=read<uint32_t>(dir+"/lo",n),hi=read<uint32_t>(dir+"/hi",n);std::vector<size_t> offset(n+1,0);uint64_t active=0;uint32_t max_width=0;uint64_t hist[5]={};
    for(size_t i=0;i<n;++i){
        if(lo[i]>hi[i]||hi[i]>uint32_t(d))throw std::runtime_error("invalid band");uint32_t width=hi[i]-lo[i],cap=bucket(width);
        if(offset[i]>std::numeric_limits<size_t>::max()-cap)throw std::runtime_error("offset overflow");offset[i+1]=offset[i]+cap;active+=width;max_width=std::max(max_width,width);
        ++hist[cap==0?0:cap==16?1:cap==32?2:cap==64?3:4];
    }
    size_t allocated=offset[n];if(allocated>std::numeric_limits<size_t>::max()/16)throw std::runtime_error("allocation extent overflow");
    std::vector<uint32_t> cost,own,other,own_support,other_support;
    if(image){own=read<uint32_t>(dir+"/own",n);other=read<uint32_t>(dir+"/other",n);own_support=read<uint32_t>(dir+"/own_support",n);other_support=read<uint32_t>(dir+"/other_support",n);for(auto x:own_support)if(x>1)throw std::runtime_error("bad support");for(auto x:other_support)if(x>1)throw std::runtime_error("bad support");}
    else {cost=read<uint32_t>(dir+"/cost",allocated);for(size_t i=0;i<n;++i)for(uint32_t t=0;t<hi[i]-lo[i];++t){auto c=cost[offset[i]+t];if(c!=INF&&c>10000)throw std::runtime_error("invalid cost");}}
    auto get_cost=[&](size_t i,uint32_t q){
        if(!image)return cost[offset[i]+q-lo[i]];
        int x=int(i%size_t(w)),ox=eye?x+int(q):x-int(q);if(!own_support[i]||ox<0||ox>=w)return INF;
        size_t j=(i/size_t(w))*size_t(w)+size_t(ox);if(!other_support[j])return INF;return uint32_t(std::popcount(own[i]^other[j]));
    };
    bool dump=std::getenv("FP2_DUMP_PATHS")!=nullptr;std::vector<uint32_t> sums(allocated,INF),paths(dump?4*allocated:0,INF),prev(max_width,INF),cur(max_width,INF);
    uint64_t cost_checks=0,valid_updates=0,prev_min_visits=0,aggregate_writes=0,winner_visits=0;const int dx[4]={1,-1,0,0},dy[4]={0,0,1,-1};
    for(int k=0;k<4;++k)for(int sy=0;sy<h;++sy)for(int sx=0;sx<w;++sx){
        int px=sx-dx[k],py=sy-dy[k];if(px>=0&&px<w&&py>=0&&py<h)continue;
        uint32_t prev_lo=0,prev_width=0;
        for(int x=sx,y=sy;x>=0&&x<w&&y>=0&&y<h;x+=dx[k],y+=dy[k]){
            uint32_t m=INF;for(uint32_t t=0;t<prev_width;++t){++prev_min_visits;m=std::min(m,prev[t]);}
            size_t i=size_t(y)*size_t(w)+size_t(x);uint32_t width=hi[i]-lo[i];
            for(uint32_t t=0;t<width;++t){
                uint32_t q=lo[i]+t;++cost_checks;uint32_t c=get_cost(i,q);cur[t]=INF;if(c==INF)continue;++valid_updates;uint32_t v=c;
                if(m!=INF){
                    uint32_t best=m+uint32_t(p2);
                    auto at=[&](int64_t global){return global>=prev_lo&&global<int64_t(prev_lo)+prev_width?prev[size_t(global-prev_lo)]:INF;};
                    auto same=at(q),lower=at(int64_t(q)-1),upper=at(int64_t(q)+1);best=std::min(best,same);if(lower!=INF)best=std::min(best,lower+uint32_t(p1));if(upper!=INF)best=std::min(best,upper+uint32_t(p1));v+=best-m;
                }
                cur[t]=v;size_t j=offset[i]+t;if(dump)paths[size_t(k)*allocated+j]=v;
                if(k==0)sums[j]=v;else if(sums[j]!=INF)sums[j]+=v;++aggregate_writes;
            }
            prev.swap(cur);prev_lo=lo[i];prev_width=width;
        }
    }
    std::vector<int32_t>winner(n,-1);std::vector<uint32_t>best(n,INF),second(n,INF),count(n,0);
    for(size_t i=0;i<n;++i)for(uint32_t t=0;t<hi[i]-lo[i];++t){++winner_visits;auto v=sums[offset[i]+t];if(v==INF)continue;++count[i];if(v<best[i]){second[i]=best[i];best[i]=v;winner[i]=int32_t(lo[i]+t);}else if(v<second[i])second[i]=v;}
    write(dir+"/winner",winner);write(dir+"/best",best);write(dir+"/second",second);write(dir+"/count",count);write(dir+"/offset",offset);
    if(dump){write(dir+"/paths",paths);write(dir+"/sums",sums);}
    std::ofstream f(dir+"/stats.json");f<<"{\"active_slots\":"<<active<<",\"bucket_slots\":"<<allocated<<",\"padding_slots\":"<<allocated-active<<",\"bucket_histogram\":["<<hist[0]<<","<<hist[1]<<","<<hist[2]<<","<<hist[3]<<","<<hist[4]<<"],\"cost_checks\":"<<cost_checks<<",\"valid_updates\":"<<valid_updates<<",\"prev_min_visits\":"<<prev_min_visits<<",\"aggregate_writes\":"<<aggregate_writes<<",\"winner_visits\":"<<winner_visits<<",\"aggregate_bytes\":"<<allocated*sizeof(uint32_t)<<",\"offset_bytes\":"<<(n+1)*sizeof(size_t)<<",\"bounds_bytes\":"<<2*n*sizeof(uint32_t)<<",\"scratch_bytes\":"<<2*max_width*sizeof(uint32_t)<<",\"packed_input_cost_bytes\":"<<cost.size()*sizeof(uint32_t)<<",\"image_planes_bytes\":"<<4*own.size()*sizeof(uint32_t)<<",\"winner_planes_bytes\":"<<4*n*sizeof(uint32_t)<<",\"diagnostic_paths_bytes\":"<<paths.size()*sizeof(uint32_t)<<"}\n";if(!f)throw std::runtime_error("stats write failed");
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}return 0;}
