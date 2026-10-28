// Standalone tiny FP2 banded recurrence fixture. Never used by the live graph.
#include <stereoforge/daxa_foundation.hpp>
#include "fp2_identity.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace stereoforge;
constexpr uint32_t INF=std::numeric_limits<uint32_t>::max();
static void require(bool v,const char* s){if(!v)throw std::runtime_error(s);}
static uint32_t bucket(uint32_t v){return !v?0:v<=16?16:v<=32?32:v<=64?64:v<=128?128:throw std::runtime_error("band width");}
struct Fixture {
    uint32_t w=7,h=5,d=128,p1=8,p2=32,max_lines=7;
    std::vector<uint32_t> lo,hi,cost;std::vector<uint64_t> offsets,line_offsets;
    uint64_t scratch_words=0;
};
static Fixture fixture(uint32_t variant=0){
    Fixture f;size_t n=size_t(f.w)*f.h;f.lo.resize(n);f.hi.resize(n);f.offsets.resize(n+1);
    for(size_t i=0;i<n;++i){auto lo=uint32_t((i*17)%90);auto len=uint32_t((i*11)%39);if(i==0)len=0;if(i==1){lo=0;len=128;}if(i==2){lo=127;len=1;}if(i==3){lo=15;len=16;}if(i==4){lo=10;len=32;}if(i==5){lo=30;len=64;}if(i==6){lo=0;len=128;}if(i==8){lo=0;len=16;}if(i==9){lo=96;len=16;}f.lo[i]=lo;f.hi[i]=lo+len;f.offsets[i+1]=f.offsets[i]+bucket(len);}
    f.cost.assign(f.offsets.back(),INF);
    for(size_t i=0;i<n;++i)for(uint32_t t=0;t<f.hi[i]-f.lo[i];++t){auto q=f.lo[i]+t;if(i!=8 && (i+q+variant)%7)f.cost[f.offsets[i]+t]=uint32_t((i*13+q*7+variant*3)%25);}
    f.line_offsets.resize(4*(f.max_lines+1));uint64_t current=0;
    for(uint32_t pass=0;pass<4;++pass){uint32_t lines=pass<2?f.h:f.w;
        for(uint32_t line=0;line<f.max_lines;++line){f.line_offsets[pass*(f.max_lines+1)+line]=current;uint32_t largest=0;
            if(line<lines){for(uint32_t k=0;k<(pass<2?f.w:f.h);++k){size_t i=pass<2?size_t(line)*f.w+k:size_t(k)*f.w+line;largest=std::max(largest,f.hi[i]-f.lo[i]);}current+=2*largest;}}
        f.line_offsets[pass*(f.max_lines+1)+f.max_lines]=current;
    }f.scratch_words=current;return f;
}
static void validate(const Fixture& f){
    require(f.w&&f.h&&f.w<=256&&f.h<=256&&f.d&&f.d<=128&&f.p1<=f.p2&&f.p2<=10000,"invalid dimensions/penalties");
    size_t n=size_t(f.w)*f.h;require(f.lo.size()==n&&f.hi.size()==n&&f.offsets.size()==n+1&&f.offsets[0]==0&&f.max_lines==std::max(f.w,f.h),"invalid layout");
    for(size_t i=0;i<n;++i){require(f.lo[i]<=f.hi[i]&&f.hi[i]<=f.d,"bad band");require(f.offsets[i+1]>=f.offsets[i]&&f.offsets[i+1]-f.offsets[i]==bucket(f.hi[i]-f.lo[i]),"bad bucket offset");}
    require(f.offsets.back()<=1000000&&f.cost.size()==f.offsets.back(),"invalid cost extent");
    for(auto c:f.cost)require(c==INF||c<=10000,"invalid cost");
    require(f.line_offsets.size()==4*(f.max_lines+1)&&f.scratch_words<=1000000,"invalid scratch");uint64_t end=0;
    for(uint32_t pass=0;pass<4;++pass){uint32_t lines=pass<2?f.h:f.w;
        for(uint32_t line=0;line<f.max_lines;++line){auto base=f.line_offsets[pass*(f.max_lines+1)+line],next=f.line_offsets[pass*(f.max_lines+1)+line+1];require(base==end&&next>=base&&next<=f.scratch_words&&(next-base)%2==0,"scratch overlap");uint32_t largest=0;
            if(line<lines){for(uint32_t k=0;k<(pass<2?f.w:f.h);++k){size_t i=pass<2?size_t(line)*f.w+k:size_t(k)*f.w+line;largest=std::max(largest,f.hi[i]-f.lo[i]);}}
            require(next-base==2*largest,"scratch capacity");end=next;}}
    require(end==f.scratch_words,"scratch extent");
}
struct Output {std::vector<uint32_t> paths,sums;};
// Independent dense global-disparity oracle, used on tiny host fixture only.
static Output oracle(const Fixture& f){
    size_t alloc=f.offsets.back();Output out{std::vector<uint32_t>(4*alloc,INF),std::vector<uint32_t>(alloc,INF)};int dx[4]={1,-1,0,0},dy[4]={0,0,1,-1};
    for(int pass=0;pass<4;++pass)for(int sy=0;sy<int(f.h);++sy)for(int sx=0;sx<int(f.w);++sx){int px=sx-dx[pass],py=sy-dy[pass];if(px>=0&&px<int(f.w)&&py>=0&&py<int(f.h))continue;std::array<uint32_t,128> prev{},cur{};prev.fill(INF);
        for(int x=sx,y=sy;x>=0&&x<int(f.w)&&y>=0&&y<int(f.h);x+=dx[pass],y+=dy[pass]){auto m=*std::min_element(prev.begin(),prev.end());cur.fill(INF);size_t i=size_t(y)*f.w+size_t(x);
            for(uint32_t q=0;q<f.d;++q){if(q<f.lo[i]||q>=f.hi[i])continue;auto j=f.offsets[i]+q-f.lo[i];auto c=f.cost[j];if(c==INF)continue;uint32_t v=c;if(m!=INF){auto b=std::min(prev[q],m+f.p2);if(q&&prev[q-1]!=INF)b=std::min(b,prev[q-1]+f.p1);if(q+1<f.d&&prev[q+1]!=INF)b=std::min(b,prev[q+1]+f.p1);v+=b-m;}cur[q]=v;out.paths[size_t(pass)*alloc+j]=v;if(!pass)out.sums[j]=v;else if(out.sums[j]!=INF)out.sums[j]+=v;}prev=cur;}}
    return out;
}
struct Region {uint64_t base,size;};
static void validate_regions(const std::vector<Region>& regions){
    for(size_t i=0;i<regions.size();++i){const auto a=regions[i];require(a.base&&a.size&&a.base<=UINT64_MAX-a.size,"invalid region");
        for(size_t j=0;j<i;++j){const auto b=regions[j];require(a.base>=b.base+b.size||b.base>=a.base+a.size,"overlapping regions");}}
}
static void cpu_tests(){for(uint32_t variant=0;variant<3;++variant){auto f=fixture(variant);validate(f);auto expected=oracle(f);require(!expected.paths.empty(),"empty oracle");
    // Pixel 0 is empty; the next pixel on the first horizontal path must restart.
    for(uint32_t t=0;t<f.hi[1]-f.lo[1];++t){auto j=f.offsets[1]+t;require(expected.paths[j]==f.cost[j],"empty predecessor restart");}
    for(size_t i=0;i<f.lo.size();++i)for(uint32_t t=0;t<f.hi[i]-f.lo[i];++t){auto j=f.offsets[i]+t;for(uint32_t pass=0;pass<4;++pass){auto v=expected.paths[size_t(pass)*f.offsets.back()+j];if(f.cost[j]==INF)require(v==INF,"invalid path sentinel");else require(v>=f.cost[j]&&v<=f.cost[j]+f.p2,"invalid path bound");}}
    // Pixel 8 has an all-invalid predecessor; pixel 9 is a disjoint band.
    for(uint32_t t=0;t<f.hi[9]-f.lo[9];++t){auto j=f.offsets[9]+t;require(expected.paths[j]==f.cost[j],"all-invalid predecessor restart");}
    validate_regions({{1024,64},{2048,64}});
    try{validate_regions({{1024,64},{1080,64}});throw std::runtime_error("alias accepted");}catch(const std::runtime_error& e){require(std::string(e.what())!="alias accepted","alias accepted");}
    auto bad=f;bad.offsets[3]++;try{validate(bad);throw std::runtime_error("bad offsets accepted");}catch(const std::runtime_error& e){require(std::string(e.what())!="bad offsets accepted","bad offsets accepted");}
    bad=f;bad.line_offsets[2]++;try{validate(bad);throw std::runtime_error("bad scratch accepted");}catch(const std::runtime_error& e){require(std::string(e.what())!="bad scratch accepted","bad scratch accepted");}}}
struct Parameters {uint64_t lo,hi,offsets,costs,sums,paths,scratch,line_offsets;uint32_t width,height,disparities,p1,p2,max_lines;};
struct Push {uint64_t parameters;uint32_t pass;};
static_assert(sizeof(Parameters)==88 && sizeof(Push)==16);
struct ResourceOwner {
    daxa::Device& device;std::vector<daxa::MemoryBlock>& memory;std::vector<daxa::BufferId>& buffers;bool released=false;
    void release(){if(released)return;device.wait_idle();for(auto b:buffers){device.destroy_buffer(b);}device.collect_garbage();memory.clear();released=true;}
    ~ResourceOwner() noexcept {try{release();}catch(...){}}
};
static void gpu(const Fixture& f){
    validate(f);auto expected=oracle(f);DaxaFoundation context(true);auto& d=context.device;std::vector<daxa::MemoryBlock> memory;std::vector<daxa::BufferId> buffers;ResourceOwner owner{d,memory,buffers};
    auto allocate=[&](size_t bytes,const void* data){require(bytes>0,"empty buffer");daxa::BufferInfo info{.size=bytes,.name="fp2 isolated fixture"};auto m=d.create_memory({.requirements=context.coherent_readback_requirements(info),.flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});auto b=d.create_buffer_from_memory_block({.buffer_info=info,.memory_block=m});memory.push_back(std::move(m));buffers.push_back(b);if(data)std::memcpy(d.buffer_host_address(b).value(),data,bytes);return b;};
    auto lo=allocate(f.lo.size()*4,f.lo.data()),hi=allocate(f.hi.size()*4,f.hi.data()),offset=allocate(f.offsets.size()*8,f.offsets.data()),cost=allocate(f.cost.size()*4,f.cost.data());
    auto sums=allocate(expected.sums.size()*4,expected.sums.data()),paths=allocate(expected.paths.size()*4,expected.paths.data());
    // Reset diagnostic outputs; initialized sentinel permits unwritten padding/invalid entries.
    std::memset(d.buffer_host_address(sums).value(),0xff,expected.sums.size()*4);std::memset(d.buffer_host_address(paths).value(),0xff,expected.paths.size()*4);
    auto scratch=allocate(f.scratch_words*4,nullptr),line=allocate(f.line_offsets.size()*8,f.line_offsets.data());
    Parameters params{d.device_address(lo).value(),d.device_address(hi).value(),d.device_address(offset).value(),d.device_address(cost).value(),d.device_address(sums).value(),d.device_address(paths).value(),d.device_address(scratch).value(),d.device_address(line).value(),f.w,f.h,f.d,f.p1,f.p2,f.max_lines};auto parameter=allocate(sizeof(params),&params);
    validate_regions({{params.lo,f.lo.size()*4},{params.hi,f.hi.size()*4},{params.offsets,f.offsets.size()*8},{params.costs,f.cost.size()*4},
        {params.sums,expected.sums.size()*4},{params.paths,expected.paths.size()*4},{params.scratch,f.scratch_words*4},
        {params.line_offsets,f.line_offsets.size()*8},{d.device_address(parameter).value(),sizeof(params)}});
    std::ifstream file(SF_FP2_SPIRV_PATH,std::ios::binary|std::ios::ate);require(bool(file),"missing SPIR-V");size_t bytes=size_t(file.tellg());require(bytes&&bytes%4==0&&bytes<=UINT32_MAX,"invalid SPIR-V");std::vector<uint32_t> words(bytes/4);file.seekg(0);require(bool(file.read(reinterpret_cast<char*>(words.data()),std::streamsize(bytes))),"SPIR-V read");
    { auto pipeline=d.create_compute_pipeline({.shader_info={.byte_code=words.data(),.byte_code_size=uint32_t(words.size())},.push_constant_size=sizeof(Push),.name="fp2 recurrence prototype"});
    auto recorder=d.create_command_recorder({.name="fp2 four ordered scanlines"});recorder.pipeline_barrier({.src_access=daxa::AccessConsts::HOST_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE});
    for(uint32_t pass=0;pass<4;++pass){if(pass)recorder.pipeline_barrier({.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE});recorder.set_pipeline(pipeline);recorder.push_constant(Push{d.device_address(parameter).value(),pass});recorder.dispatch({pass<2?f.h:f.w,1,1});}
    recorder.pipeline_barrier({.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::HOST_READ});auto commands=recorder.complete_current_commands();d.submit_commands({.command_lists={&commands,1}});d.wait_idle();
    require(std::memcmp(d.buffer_host_address(sums).value(),expected.sums.data(),expected.sums.size()*4)==0,"GPU sum mismatch");require(std::memcmp(d.buffer_host_address(paths).value(),expected.paths.data(),expected.paths.size()*4)==0,"GPU paths mismatch"); }
    owner.release();context.close();require(context.validation_errors.load()==0&&context.validation_warnings.load()==0,"validation diagnostics");
}
int main(int argc,char** argv){try{require(argc==2,"use --cpu-only or --gpu");std::string mode=argv[1];if(mode=="--identity"){std::cout<<"{\"shader_source_sha256\":\""<<SF_FP2_SHADER_SHA256<<"\",\"host_source_sha256\":\""<<SF_FP2_HOST_SHA256<<"\",\"spirv_sha256\":\""<<SF_FP2_SPIRV_SHA256<<"\",\"compiler_sha256\":\""<<SF_DAXA_COMPILER_SHA256<<"\",\"compiler_path\":\""<<SF_FP2_COMPILER_PATH<<"\",\"spirv_path\":\""<<SF_FP2_SPIRV_PATH<<"\"}\n";return 0;}cpu_tests();if(mode=="--gpu"){for(uint32_t variant=0;variant<3;++variant)gpu(fixture(variant));}else require(mode=="--cpu-only","unknown mode");std::cout<<"FP2 prototype "<<mode<<" contracts passed\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
