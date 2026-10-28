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
static void cpu_image_tests(){
    for(uint32_t eye=0;eye<2;++eye)for(uint32_t big=0;big<2;++big){auto f=big?image_fixture(eye):fixture(eye);if(big)validate_image(f);else validate(f);auto a=images(f,eye);f.cost=image_cost(f,a,eye);auto out=oracle(f);require(out.sums.size()==f.cost.size(),"oracle extent");for(size_t i=0;i<f.lo.size();++i)for(uint64_t j=f.offsets[i]+f.hi[i]-f.lo[i];j<f.offsets[i+1];++j)require(f.cost[j]==INF&&out.sums[j]==INF,"padding");}
    auto f=fixture();f.lo[1]=1;f.hi[1]=2; // dedicated sign and support test uses original capacity
    auto a=images(f,0);std::fill(a.own.begin(),a.own.end(),0);std::fill(a.other.begin(),a.other.end(),0);std::fill(a.own_support.begin(),a.own_support.end(),1);std::fill(a.other_support.begin(),a.other_support.end(),1);a.other[0]=1;a.other[2]=3;
    require(image_cost(f,a,0)[f.offsets[1]]==1&&image_cost(f,a,1)[f.offsets[1]]==2,"eye direction");a.other_support[0]=0;require(image_cost(f,a,0)[f.offsets[1]]==INF,"target support");a.own_support[1]=0;require(image_cost(f,a,1)[f.offsets[1]]==INF,"reference support");
}
static std::vector<uint32_t> read_spirv(const char* path){std::ifstream f(path,std::ios::binary|std::ios::ate);require(bool(f),"missing SPIR-V");size_t bytes=size_t(f.tellg());require(bytes&&bytes%4==0&&bytes<=UINT32_MAX,"invalid SPIR-V");std::vector<uint32_t> words(bytes/4);f.seekg(0);require(bool(f.read(reinterpret_cast<char*>(words.data()),std::streamsize(bytes))),"SPIR-V read");return words;}
static void gpu_image(Fixture f,uint32_t eye,bool big){
    if(big)validate_image(f);else validate(f);auto im=images(f,eye);f.cost=image_cost(f,im,eye);auto expected=oracle(f);DaxaFoundation context(true);auto& d=context.device;std::vector<daxa::MemoryBlock> memory;std::vector<daxa::BufferId> buffers;ResourceOwner owner{d,memory,buffers};std::vector<Region> regions;uint64_t total=0;
    auto alloc=[&](size_t bytes,const void* data){require(bytes&&bytes<=256000000&&total<=512000000-bytes,"resource budget");total+=bytes;daxa::BufferInfo info{.size=bytes,.name="fp2 image fixture"};auto m=d.create_memory({.requirements=context.coherent_readback_requirements(info),.flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});auto b=d.create_buffer_from_memory_block({.buffer_info=info,.memory_block=m});memory.push_back(std::move(m));buffers.push_back(b);if(data)std::memcpy(d.buffer_host_address(b).value(),data,bytes);regions.push_back({d.device_address(b).value(),bytes});return b;};
    auto lo=alloc(f.lo.size()*4,f.lo.data()),hi=alloc(f.hi.size()*4,f.hi.data()),offset=alloc(f.offsets.size()*8,f.offsets.data()),cost=alloc(f.cost.size()*4,nullptr);std::memset(d.buffer_host_address(cost).value(),0xff,f.cost.size()*4);
    auto sums=alloc(expected.sums.size()*4,nullptr),paths=alloc(expected.paths.size()*4,nullptr);std::memset(d.buffer_host_address(sums).value(),0xff,expected.sums.size()*4);std::memset(d.buffer_host_address(paths).value(),0xff,expected.paths.size()*4);
    auto scratch=alloc(f.scratch_words*4,nullptr),line=alloc(f.line_offsets.size()*8,f.line_offsets.data());
    auto own=alloc(im.own.size()*4,im.own.data()),other=alloc(im.other.size()*4,im.other.data()),own_support=alloc(im.own_support.size()*4,im.own_support.data()),other_support=alloc(im.other_support.size()*4,im.other_support.data());
    Parameters p{d.device_address(lo).value(),d.device_address(hi).value(),d.device_address(offset).value(),d.device_address(cost).value(),d.device_address(sums).value(),d.device_address(paths).value(),d.device_address(scratch).value(),d.device_address(line).value(),f.w,f.h,f.d,f.p1,f.p2,f.max_lines};auto parameter=alloc(sizeof(p),&p);
    CostParams cp{p.lo,p.hi,p.offsets,p.costs,d.device_address(own).value(),d.device_address(other).value(),d.device_address(own_support).value(),d.device_address(other_support).value(),f.w,f.h,eye};auto cparam=alloc(sizeof(cp),&cp);validate_regions(regions);
    auto cw=read_spirv(SF_FP2_COST_SPIRV_PATH),rw=read_spirv(SF_FP2_SPIRV_PATH);
    {auto producer=d.create_compute_pipeline({.shader_info={.byte_code=cw.data(),.byte_code_size=uint32_t(cw.size())},.push_constant_size=sizeof(CostPush),.name="fp2 cost"});auto recurrence=d.create_compute_pipeline({.shader_info={.byte_code=rw.data(),.byte_code_size=uint32_t(rw.size())},.push_constant_size=sizeof(Push),.name="fp2 recurrence"});
     auto recorder=d.create_command_recorder({.name="fp2 image producer and recurrence"});recorder.pipeline_barrier({.src_access=daxa::AccessConsts::HOST_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE});recorder.set_pipeline(producer);recorder.push_constant(CostPush{d.device_address(cparam).value()});recorder.dispatch({(f.w*f.h+63)/64,1,1});
     recorder.pipeline_barrier({.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE});
     for(uint32_t pass=0;pass<4;++pass){if(pass)recorder.pipeline_barrier({.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE});recorder.set_pipeline(recurrence);recorder.push_constant(Push{d.device_address(parameter).value(),pass});recorder.dispatch({pass<2?f.h:f.w,1,1});}
     recorder.pipeline_barrier({.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::HOST_READ});auto commands=recorder.complete_current_commands();d.submit_commands({.command_lists={&commands,1}});d.wait_idle();
     require(std::memcmp(d.buffer_host_address(cost).value(),f.cost.data(),f.cost.size()*4)==0,"cost mismatch");require(std::memcmp(d.buffer_host_address(sums).value(),expected.sums.data(),expected.sums.size()*4)==0,"sum mismatch");require(std::memcmp(d.buffer_host_address(paths).value(),expected.paths.data(),expected.paths.size()*4)==0,"path mismatch");}
    owner.release();context.close();require(context.validation_errors.load()==0&&context.validation_warnings.load()==0,"validation diagnostics");std::cout<<"eye="<<eye<<" dimensions="<<f.w<<"x"<<f.h<<" requested_buffer_bytes="<<total<<'\n';
}
int main(int argc,char** argv){try{require(argc==2,"use --identity, --cpu-only, --gpu");std::string mode=argv[1];if(mode=="--identity"){std::cout<<"{\"cost_host_sha256\":\""<<SF_FP2_COST_HOST_SHA256<<"\",\"cost_shader_sha256\":\""<<SF_FP2_COST_SHADER_SHA256<<"\",\"cost_spirv_sha256\":\""<<SF_FP2_COST_SPIRV_SHA256<<"\",\"cost_spirv_path\":\""<<SF_FP2_COST_SPIRV_PATH<<"\",\"recurrence_host_sha256\":\""<<SF_FP2_HOST_SHA256<<"\",\"slots_host_sha256\":\""<<SF_FP2_SLOTS_HOST_SHA256<<"\",\"recurrence_shader_sha256\":\""<<SF_FP2_SHADER_SHA256<<"\",\"recurrence_spirv_sha256\":\""<<SF_FP2_SPIRV_SHA256<<"\",\"recurrence_spirv_path\":\""<<SF_FP2_SPIRV_PATH<<"\",\"compiler_sha256\":\""<<SF_DAXA_COMPILER_SHA256<<"\",\"compiler_path\":\""<<SF_FP2_COMPILER_PATH<<"\"}\n";return 0;}cpu_image_tests();if(mode=="--gpu")for(uint32_t eye=0;eye<2;++eye){gpu_image(fixture(eye),eye,false);gpu_image(image_fixture(eye),eye,true);}else require(mode=="--cpu-only","unknown mode");std::cout<<"FP2 image "<<mode<<" contracts passed\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
