// Isolated image-derived cost + recurrence fixture, not a production path.
#define main fp2_tiny_fixture_main
#include "fp2_prototype_fixture.cpp"
#undef main
#include "fp2_cost_identity.hpp"
#include <bit>
static Fixture image_fixture(uint32_t seed){
    Fixture f;f.w=320;f.h=180;f.d=128;f.max_lines=320;size_t n=size_t(f.w)*f.h;f.lo.resize(n);f.hi.resize(n);f.offsets.resize(n+1);
    const uint32_t widths[]={0,1,16,17,32,33,64,65,128};
    for(size_t i=0;i<n;++i){uint32_t width=widths[(i+seed)%9];uint32_t low=width==128?0:uint32_t((i*13+seed*29)%(129-width));if(i==10){width=16;low=0;}if(i==11){width=16;low=96;}if(i==12){width=0;low=0;}if(i==13){width=1;low=127;}f.lo[i]=low;f.hi[i]=low+width;f.offsets[i+1]=f.offsets[i]+bucket(width);}
    f.cost.assign(f.offsets.back(),INF);
    for(size_t i=0;i<n;++i)for(uint32_t t=0;t<f.hi[i]-f.lo[i];++t){auto q=f.lo[i]+t;if(i!=10&&(i+q+seed)%7)f.cost[f.offsets[i]+t]=uint32_t((i*13+q*7+seed*11)%25);}
    f.line_offsets.resize(4*(f.max_lines+1));uint64_t current=0;
    for(uint32_t pass=0;pass<4;++pass){uint32_t lines=pass<2?f.h:f.w;
        for(uint32_t line=0;line<f.max_lines;++line){f.line_offsets[pass*(f.max_lines+1)+line]=current;uint32_t largest=0;
            if(line<lines){for(uint32_t k=0;k<(pass<2?f.w:f.h);++k){size_t i=pass<2?size_t(line)*f.w+k:size_t(k)*f.w+line;largest=std::max(largest,f.hi[i]-f.lo[i]);}current+=2*largest;}}
        f.line_offsets[pass*(f.max_lines+1)+f.max_lines]=current;
    }f.scratch_words=current;return f;
}
static void validate_image(const Fixture& f){
    require(f.w==320&&f.h==180&&f.d==128&&f.p1==8&&f.p2==32&&f.max_lines==320,"bad fixed geometry");size_t n=size_t(f.w)*f.h;
    require(f.lo.size()==n&&f.hi.size()==n&&f.offsets.size()==n+1&&f.offsets[0]==0,"bad pixel layout");
    for(size_t i=0;i<n;++i)require(f.lo[i]<=f.hi[i]&&f.hi[i]<=f.d&&f.offsets[i+1]>=f.offsets[i]&&f.offsets[i+1]-f.offsets[i]==bucket(f.hi[i]-f.lo[i]),"bad band");
    require(f.offsets.back()<=8000000&&f.cost.size()==f.offsets.back(),"bad cost extent");for(auto c:f.cost)require(c==INF||c<=10000,"invalid cost");
    require(f.line_offsets.size()==4*(f.max_lines+1)&&f.scratch_words<=2000000,"bad scratch extent");uint64_t end=0;
    for(uint32_t pass=0;pass<4;++pass)for(uint32_t line=0;line<f.max_lines;++line){auto base=f.line_offsets[pass*(f.max_lines+1)+line],next=f.line_offsets[pass*(f.max_lines+1)+line+1];require(base==end&&next>=base&&next<=f.scratch_words,"overlapping scratch");uint32_t largest=0;
        if(line<(pass<2?f.h:f.w))for(uint32_t k=0;k<(pass<2?f.w:f.h);++k){size_t i=pass<2?size_t(line)*f.w+k:size_t(k)*f.w+line;largest=std::max(largest,f.hi[i]-f.lo[i]);}
        require(next-base==2*largest,"bad line scratch");end=next;}
    require(end==f.scratch_words,"scratch terminal mismatch");
}
struct Images {std::vector<uint32_t> own,other,own_support,other_support;};
static Images images(const Fixture& f,uint32_t eye){size_t n=size_t(f.w)*f.h;Images a;for(auto* v:{&a.own,&a.other,&a.own_support,&a.other_support})v->resize(n);for(size_t i=0;i<n;++i){a.own[i]=uint32_t(i*2654435761u+eye*7123u);a.other[i]=uint32_t(i*2246822519u+eye*997u);a.own_support[i]=(i%11)!=0;a.other_support[i]=(i%13)!=0;}return a;}
// Scalar reference walks candidates independently of GPU invocation structure.
static std::vector<uint32_t> image_cost(const Fixture& f,const Images& a,uint32_t eye){
    require(eye<2,"bad eye");size_t n=size_t(f.w)*f.h;require(a.own.size()==n&&a.other.size()==n&&a.own_support.size()==n&&a.other_support.size()==n,"image extent");
    std::vector<uint32_t> out(f.offsets.back(),INF);
    for(size_t i=0;i<n;++i){require(a.own_support[i]<=1&&a.other_support[i]<=1,"support not binary");for(uint32_t q=f.lo[i];q<f.hi[i];++q){int x=int(i%f.w);int target=eye?x+int(q):x-int(q);if(a.own_support[i]&&target>=0&&target<int(f.w)){size_t j=(i/f.w)*f.w+size_t(target);if(a.other_support[j])out[f.offsets[i]+q-f.lo[i]]=std::popcount(a.own[i]^a.other[j]);}}}return out;
}
struct CostParams {uint64_t lo,hi,offsets,costs,own,other,own_support,other_support;uint32_t width,height,eye;};
struct CostPush {uint64_t parameters;};
static_assert(sizeof(CostParams)==80&&sizeof(CostPush)==8);
#include "fp2_census_identity.hpp"
struct Plane {std::vector<uint32_t> luma,support,descriptor,valid;};
static Plane cpu_census(uint32_t w,uint32_t h,const std::vector<uint32_t>& luma,const std::vector<uint32_t>& support){
    size_t n=size_t(w)*h;require(w&&h&&luma.size()==n&&support.size()==n,"census extent");Plane out{luma,support,std::vector<uint32_t>(n),std::vector<uint32_t>(n)};
    for(size_t i=0;i<n;++i)require(luma[i]<=255&&support[i]<=1,"invalid luma/support");
    for(uint32_t y=0;y<h;++y)for(uint32_t x=0;x<w;++x){if(x<2||y<2||x+2>=w||y+2>=h)continue;size_t i=size_t(y)*w+x;uint32_t bits=0,bit=0;bool ok=true;
        for(int dy=-2;dy<=2;++dy)for(int dx=-2;dx<=2;++dx){size_t j=size_t(int(y)+dy)*w+size_t(int(x)+dx);ok=ok&&support[j];if(dx||dy){if(luma[j]<luma[i])bits|=1u<<bit;++bit;}}
        if(ok){out.valid[i]=1;out.descriptor[i]=bits;}}
    return out;
}
static Plane make_plane(const Fixture& f,uint32_t side){size_t n=size_t(f.w)*f.h;std::vector<uint32_t> gray(n),support(n,1);for(size_t i=0;i<n;++i)gray[i]=uint32_t((i*73+i/f.w*19+side*37)%256);if(f.w>7){support[size_t(20)*f.w+20]=0;support[size_t(91)*f.w+180]=0;}return cpu_census(f.w,f.h,gray,support);}
static Fixture census_fixture(bool big,uint32_t seed){auto f=big?image_fixture(seed):fixture(seed);if(!big){for(size_t i: {size_t(17),size_t(18)}){auto width=f.hi[i]-f.lo[i];f.lo[i]=0;f.hi[i]=width;}}return f;}
static Images references(const Plane& a,const Plane& b){return {a.descriptor,b.descriptor,a.valid,b.valid};}
static void cpu_census_tests(){
    std::vector<uint32_t> g(25,100),s(25,1);g[0]=0;g[24]=1;auto a=cpu_census(5,5,g,s);require(a.valid[12]&&a.descriptor[12]==((1u<<0)|(1u<<23)),"bit order");
    for(size_t i=0;i<25;++i){auto t=s;t[i]=0;auto z=cpu_census(5,5,g,t);require(z.valid[12]==0&&z.descriptor[12]==0,"footprint hole");}
    auto small=cpu_census(3,3,std::vector<uint32_t>(9),std::vector<uint32_t>(9,1));require(std::count(small.valid.begin(),small.valid.end(),1)==0,"small border");
    auto bad=g;bad[0]=256;try{cpu_census(5,5,bad,s);throw std::runtime_error("bad luma accepted");}catch(const std::runtime_error& e){require(std::string(e.what())!="bad luma accepted","bad luma accepted");}
    auto mask=s;mask[0]=2;try{cpu_census(5,5,g,mask);throw std::runtime_error("bad mask accepted");}catch(const std::runtime_error& e){require(std::string(e.what())!="bad mask accepted","bad mask accepted");}
    for(uint32_t eye=0;eye<2;++eye)for(bool big:{false,true}){auto f=census_fixture(big,eye);if(big)validate_image(f);else validate(f);auto own=make_plane(f,eye),other=make_plane(f,1-eye);require(std::count(own.valid.begin(),own.valid.end(),1)>0&&std::count(other.valid.begin(),other.valid.end(),1)>0,"no supported census");f.cost=image_cost(f,references(own,other),eye);require(std::count_if(f.cost.begin(),f.cost.end(),[](uint32_t x){return x!=INF;})>0,"no valid costs");auto o=oracle(f);require(o.sums.size()==f.cost.size(),"oracle extent");}
}
struct CensusParams {uint64_t luma,support,descriptor,valid;uint32_t width,height;};struct CensusPush {uint64_t parameters;};static_assert(sizeof(CensusParams)==40&&sizeof(CensusPush)==8);
static std::vector<uint32_t> read_spirv(const char* path){std::ifstream f(path,std::ios::binary|std::ios::ate);require(bool(f),"missing SPIR-V");size_t bytes=size_t(f.tellg());require(bytes&&bytes%4==0&&bytes<=UINT32_MAX,"invalid SPIR-V");std::vector<uint32_t> words(bytes/4);f.seekg(0);require(bool(f.read(reinterpret_cast<char*>(words.data()),std::streamsize(bytes))),"SPIR-V read");return words;}
#include <filesystem>
#include <bit>
#include <sys/sysinfo.h>
#include <daxa/c/device.h>
namespace fs=std::filesystem;

template<class T> static std::vector<T> load(const fs::path& path,size_t n){require(n<=SIZE_MAX/sizeof(T),"input overflow");require(fs::file_size(path)==n*sizeof(T),"input extent");std::vector<T> v(n);std::ifstream in(path,std::ios::binary);require(bool(in.read(reinterpret_cast<char*>(v.data()),std::streamsize(n*sizeof(T)))),"input read");return v;}
static void compare_file(const fs::path& path,const void* data,uint64_t bytes){require(fs::file_size(path)==bytes,"reference extent");std::ifstream in(path,std::ios::binary);require(bool(in),"reference open");std::vector<char> chunk(8*1024*1024);auto ptr=static_cast<const char*>(data);while(bytes){auto n=std::min<uint64_t>(bytes,chunk.size());require(bool(in.read(chunk.data(),std::streamsize(n))),"reference read");require(std::memcmp(ptr,chunk.data(),n)==0,"reference mismatch");ptr+=n;bytes-=n;}}
static Fixture input_fixture(const fs::path& dir,uint32_t w,uint32_t h){
    require(w&&h&&w<=1280&&h<=720,"dimension bound");Fixture f;f.w=w;f.h=h;f.d=128;f.max_lines=std::max(w,h);size_t n=size_t(w)*h;f.lo=load<uint32_t>(dir/"lo",n);f.hi=load<uint32_t>(dir/"hi",n);f.offsets.resize(n+1);
    for(size_t i=0;i<n;++i){require(f.lo[i]<=f.hi[i]&&f.hi[i]<=128,"invalid band");f.offsets[i+1]=f.offsets[i]+bucket(f.hi[i]-f.lo[i]);}
    require(f.offsets.back()<=uint64_t(n)*128&&f.offsets.back()<=65000000,"packed budget");auto offset_file=load<uint64_t>(dir/"offset",n+1);require(f.offsets==offset_file,"offset mismatch");
    f.line_offsets.resize(4*(f.max_lines+1));uint64_t current=0;
    for(uint32_t pass=0;pass<4;++pass){uint32_t lines=pass<2?h:w;for(uint32_t line=0;line<f.max_lines;++line){f.line_offsets[pass*(f.max_lines+1)+line]=current;uint32_t largest=0;if(line<lines)for(uint32_t k=0;k<(pass<2?w:h);++k){size_t i=pass<2?size_t(line)*w+k:size_t(k)*w+line;largest=std::max(largest,f.hi[i]-f.lo[i]);}current+=2*largest;}f.line_offsets[pass*(f.max_lines+1)+f.max_lines]=current;}
    f.scratch_words=current;return f;
}
static void admission(DaxaFoundation& context,const std::vector<size_t>& sizes){
    auto phys=daxa_dvc_get_vk_physical_device(context.device.get());uint32_t extension_count=0;require(vkEnumerateDeviceExtensionProperties(phys,nullptr,&extension_count,nullptr)==VK_SUCCESS,"device extensions");std::vector<VkExtensionProperties> extensions(extension_count);require(vkEnumerateDeviceExtensionProperties(phys,nullptr,&extension_count,extensions.data())==VK_SUCCESS,"device extension read");bool budget=false;for(auto& e:extensions)if(std::string(e.extensionName)==VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)budget=true;require(budget,"memory budget unavailable");
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget_data{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};VkPhysicalDeviceMemoryProperties2 mem{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};mem.pNext=&budget_data;vkGetPhysicalDeviceMemoryProperties2(phys,&mem);
    VkPhysicalDeviceMaintenance3Properties limits{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};VkPhysicalDeviceProperties2 prop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};prop.pNext=&limits;vkGetPhysicalDeviceProperties2(phys,&prop);
    std::array<uint64_t,VK_MAX_MEMORY_HEAPS> heaps{};uint64_t total=0;for(size_t n:sizes){require(n&&n<=1'100'000'000ull&&n<=limits.maxMemoryAllocationSize&&total<=2'000'000'000ull-n,"buffer cap");total+=n;daxa::BufferInfo info{.size=n,.name="fp2 admission"};auto req=context.coherent_readback_requirements(info);uint32_t type=uint32_t(std::countr_zero(req.memory_type_bits));require(type<mem.memoryProperties.memoryTypeCount,"memory type");auto heap=mem.memoryProperties.memoryTypes[type].heapIndex;require(heap<mem.memoryProperties.memoryHeapCount&&heaps[heap]<=UINT64_MAX-req.size,"heap overflow");heaps[heap]+=req.size;}
    for(uint32_t i=0;i<mem.memoryProperties.memoryHeapCount;++i)if(heaps[i]){require(budget_data.heapBudget[i]>=budget_data.heapUsage[i],"invalid heap budget");require(budget_data.heapBudget[i]-budget_data.heapUsage[i]>=heaps[i]+512'000'000ull,"insufficient GPU heap budget");}
    struct sysinfo info{};require(sysinfo(&info)==0,"system memory info");require(uint64_t(info.freeram)*info.mem_unit>=total+512'000'000ull,"insufficient free system RAM");std::cout<<"requested_buffer_bytes="<<total<<" max_memory_allocation="<<limits.maxMemoryAllocationSize<<'\n';
}
static void gpu_720(Fixture f,uint32_t eye,const fs::path& dir){
    require(eye<2,"invalid eye");size_t n=size_t(f.w)*f.h;auto l0=load<uint32_t>(dir/"own_luma",n),l1=load<uint32_t>(dir/"other_luma",n),s0=load<uint32_t>(dir/"own_rect_support",n),s1=load<uint32_t>(dir/"other_rect_support",n);auto left=cpu_census(f.w,f.h,l0,s0),right=cpu_census(f.w,f.h,l1,s1);auto words=f.offsets.back();require(words&&words<=65000000,"packed cost bound");
    compare_file(dir/"own",left.descriptor.data(),n*4);compare_file(dir/"other",right.descriptor.data(),n*4);compare_file(dir/"own_support",left.valid.data(),n*4);compare_file(dir/"other_support",right.valid.data(),n*4);require(std::count(left.valid.begin(),left.valid.end(),1)>0&&std::count(right.valid.begin(),right.valid.end(),1)>0,"empty census");DaxaFoundation context(true);auto& d=context.device;std::vector<daxa::MemoryBlock> memory;std::vector<daxa::BufferId> buffers;ResourceOwner owner{d,memory,buffers};std::vector<Region> regions;uint64_t total=0;
    auto alloc=[&](size_t bytes,const void* data){require(bytes&&bytes<=1100000000&&total<=2000000000-bytes,"resource budget");total+=bytes;daxa::BufferInfo info{.size=bytes,.name="fp2 image fixture"};auto m=d.create_memory({.requirements=context.coherent_readback_requirements(info),.flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});auto b=d.create_buffer_from_memory_block({.buffer_info=info,.memory_block=m});memory.push_back(std::move(m));buffers.push_back(b);if(data)std::memcpy(d.buffer_host_address(b).value(),data,bytes);regions.push_back({d.device_address(b).value(),bytes});return b;};
    std::vector<size_t> sizes={n*4,n*4,(n+1)*8,size_t(words)*4,size_t(words)*4,size_t(words)*16,size_t(f.scratch_words)*4,f.line_offsets.size()*8,n*4,n*4,n*4,n*4,n*4,n*4,n*4,n*4,sizeof(CensusParams),sizeof(CensusParams),sizeof(Parameters),sizeof(CostParams)};admission(context,sizes);
    auto lo=alloc(f.lo.size()*4,f.lo.data()),hi=alloc(f.hi.size()*4,f.hi.data()),offset=alloc(f.offsets.size()*8,f.offsets.data()),cost=alloc(size_t(words)*4,nullptr);std::memset(d.buffer_host_address(cost).value(),0xff,size_t(words)*4);
    auto sums=alloc(size_t(words)*4,nullptr),paths=alloc(size_t(words)*16,nullptr);std::memset(d.buffer_host_address(sums).value(),0xff,size_t(words)*4);std::memset(d.buffer_host_address(paths).value(),0xff,size_t(words)*16);
    auto scratch=alloc(f.scratch_words*4,nullptr),line=alloc(f.line_offsets.size()*8,f.line_offsets.data());
    auto own=alloc(n*4,nullptr),other=alloc(n*4,nullptr),own_support=alloc(n*4,nullptr),other_support=alloc(n*4,nullptr);
    auto luma0=alloc(left.luma.size()*4,left.luma.data()),luma1=alloc(right.luma.size()*4,right.luma.data()),support0=alloc(left.support.size()*4,left.support.data()),support1=alloc(right.support.size()*4,right.support.data());
    CensusParams c0{d.device_address(luma0).value(),d.device_address(support0).value(),d.device_address(own).value(),d.device_address(own_support).value(),f.w,f.h};CensusParams c1{d.device_address(luma1).value(),d.device_address(support1).value(),d.device_address(other).value(),d.device_address(other_support).value(),f.w,f.h};auto census0=alloc(sizeof(c0),&c0),census1=alloc(sizeof(c1),&c1);
    Parameters p{d.device_address(lo).value(),d.device_address(hi).value(),d.device_address(offset).value(),d.device_address(cost).value(),d.device_address(sums).value(),d.device_address(paths).value(),d.device_address(scratch).value(),d.device_address(line).value(),f.w,f.h,f.d,f.p1,f.p2,f.max_lines};auto parameter=alloc(sizeof(p),&p);
    CostParams cp{p.lo,p.hi,p.offsets,p.costs,d.device_address(own).value(),d.device_address(other).value(),d.device_address(own_support).value(),d.device_address(other_support).value(),f.w,f.h,eye};auto cparam=alloc(sizeof(cp),&cp);validate_regions(regions);
    auto cw=read_spirv(SF_FP2_COST_SPIRV_PATH),rw=read_spirv(SF_FP2_SPIRV_PATH),sw=read_spirv(SF_FP2_CENSUS_SPIRV_PATH);
    {auto census=d.create_compute_pipeline({.shader_info={.byte_code=sw.data(),.byte_code_size=uint32_t(sw.size())},.push_constant_size=sizeof(CensusPush),.name="fp2 census"});auto producer=d.create_compute_pipeline({.shader_info={.byte_code=cw.data(),.byte_code_size=uint32_t(cw.size())},.push_constant_size=sizeof(CostPush),.name="fp2 cost"});auto recurrence=d.create_compute_pipeline({.shader_info={.byte_code=rw.data(),.byte_code_size=uint32_t(rw.size())},.push_constant_size=sizeof(Push),.name="fp2 recurrence"});
     auto recorder=d.create_command_recorder({.name="fp2 image producer and recurrence"});recorder.pipeline_barrier({.src_access=daxa::AccessConsts::HOST_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE});recorder.set_pipeline(census);recorder.push_constant(CensusPush{d.device_address(census0).value()});recorder.dispatch({(f.w*f.h+63)/64,1,1});recorder.push_constant(CensusPush{d.device_address(census1).value()});recorder.dispatch({(f.w*f.h+63)/64,1,1});recorder.pipeline_barrier({.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE});recorder.set_pipeline(producer);recorder.push_constant(CostPush{d.device_address(cparam).value()});recorder.dispatch({(f.w*f.h+63)/64,1,1});
     recorder.pipeline_barrier({.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE});
     for(uint32_t pass=0;pass<4;++pass){if(pass)recorder.pipeline_barrier({.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE});recorder.set_pipeline(recurrence);recorder.push_constant(Push{d.device_address(parameter).value(),pass});recorder.dispatch({pass<2?f.h:f.w,1,1});}
     recorder.pipeline_barrier({.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::HOST_READ});auto commands=recorder.complete_current_commands();d.submit_commands({.command_lists={&commands,1}});d.wait_idle();
     require(std::memcmp(d.buffer_host_address(own).value(),left.descriptor.data(),left.descriptor.size()*4)==0,"own descriptor mismatch");require(std::memcmp(d.buffer_host_address(other).value(),right.descriptor.data(),right.descriptor.size()*4)==0,"other descriptor mismatch");require(std::memcmp(d.buffer_host_address(own_support).value(),left.valid.data(),left.valid.size()*4)==0,"own census support mismatch");require(std::memcmp(d.buffer_host_address(other_support).value(),right.valid.data(),right.valid.size()*4)==0,"other census support mismatch");auto* cost_ptr=reinterpret_cast<uint32_t*>(d.buffer_host_address(cost).value());auto* sum_ptr=reinterpret_cast<uint32_t*>(d.buffer_host_address(sums).value());uint64_t valid_costs=0;
     for(size_t i=0;i<n;++i){auto x=int(i%f.w);for(uint32_t t=0;t<bucket(f.hi[i]-f.lo[i]);++t){uint32_t expected=INF;if(t<f.hi[i]-f.lo[i]){auto q=f.lo[i]+t;int other_x=eye?x+int(q):x-int(q);if(left.valid[i]&&other_x>=0&&other_x<int(f.w)){auto j=(i/f.w)*f.w+size_t(other_x);if(right.valid[j]){expected=std::popcount(left.descriptor[i]^right.descriptor[j]);++valid_costs;}}}require(cost_ptr[f.offsets[i]+t]==expected,"cost or padding mismatch");}}
     require(valid_costs>0,"no valid costs");compare_file(dir/"sums",sum_ptr,words*4);compare_file(dir/"paths",d.buffer_host_address(paths).value(),words*16);
     auto winner=load<int32_t>(dir/"winner",n);auto best=load<uint32_t>(dir/"best",n),second=load<uint32_t>(dir/"second",n),count=load<uint32_t>(dir/"count",n);
     for(size_t i=0;i<n;++i){int32_t w=-1;uint32_t b=INF,s=INF,c=0;for(uint32_t t=0;t<f.hi[i]-f.lo[i];++t){auto v=sum_ptr[f.offsets[i]+t];if(v==INF)continue;++c;if(v<b){s=b;b=v;w=int32_t(f.lo[i]+t);}else if(v<s)s=v;}require(w==winner[i]&&b==best[i]&&s==second[i]&&c==count[i],"winner plane mismatch");}
     std::cout<<"valid_costs="<<valid_costs<<'\n';}
    owner.release();context.close();require(context.validation_errors.load()==0&&context.validation_warnings.load()==0,"validation diagnostics");std::cout<<"eye="<<eye<<" dimensions="<<f.w<<"x"<<f.h<<" valid_own="<<std::count(left.valid.begin(),left.valid.end(),1)<<" valid_other="<<std::count(right.valid.begin(),right.valid.end(),1)<<" requested_buffer_bytes="<<total<<'\n';
}
int main(int argc,char** argv){try{
    require(argc>=2,"use --identity or --check/--gpu directory eye width height");std::string mode=argv[1];
    if(mode=="--identity"){require(argc==2,"identity arguments");std::cout<<"{\"host_sha256\":\""<<SF_FP2_720_HOST_SHA256<<"\",\"included_host_sha256\":\""<<SF_FP2_HOST_SHA256<<"\",\"census_shader_sha256\":\""<<SF_FP2_CENSUS_SHADER_SHA256<<"\",\"census_helper_sha256\":\""<<SF_FP2_CENSUS_HELPER_SHA256<<"\",\"census_spirv_sha256\":\""<<SF_FP2_CENSUS_SPIRV_SHA256<<"\",\"census_spirv_path\":\""<<SF_FP2_CENSUS_SPIRV_PATH<<"\",\"cost_shader_sha256\":\""<<SF_FP2_COST_SHADER_SHA256<<"\",\"cost_spirv_sha256\":\""<<SF_FP2_COST_SPIRV_SHA256<<"\",\"cost_spirv_path\":\""<<SF_FP2_COST_SPIRV_PATH<<"\",\"recurrence_shader_sha256\":\""<<SF_FP2_SHADER_SHA256<<"\",\"recurrence_spirv_sha256\":\""<<SF_FP2_SPIRV_SHA256<<"\",\"recurrence_spirv_path\":\""<<SF_FP2_SPIRV_PATH<<"\",\"compiler_sha256\":\""<<SF_DAXA_COMPILER_SHA256<<"\",\"compiler_path\":\""<<SF_FP2_COMPILER_PATH<<"\"}\n";return 0;}
    require(argc==6&&(mode=="--check"||mode=="--gpu"),"invalid arguments");fs::path dir=argv[2];auto eye=std::stoul(argv[3]),w=std::stoul(argv[4]),h=std::stoul(argv[5]);require(eye<2&&w<=1280&&h<=720,"invalid parameters");auto f=input_fixture(dir,uint32_t(w),uint32_t(h));
    auto n=size_t(w)*h;for(auto name:{"own_luma","other_luma","own_rect_support","other_rect_support","own","other","own_support","other_support","winner","best","second","count"})require(fs::file_size(dir/name)==n*4,"plane extent");require(fs::file_size(dir/"paths")==f.offsets.back()*16&&fs::file_size(dir/"sums")==f.offsets.back()*4,"oracle extent");
    if(mode=="--gpu"){require(w==1280&&h==720,"GPU fixture is 720p only");gpu_720(f,uint32_t(eye),dir);}std::cout<<"FP2 720p "<<mode<<" contracts passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
