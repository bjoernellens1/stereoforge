// Isolated synthetic 320x180 two-slot recurrence parity fixture.
#define main fp2_tiny_fixture_main
#include "fp2_prototype_fixture.cpp"
#undef main

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
static void cpu_slots(){
    auto a=image_fixture(0),b=image_fixture(1);validate_image(a);validate_image(b);require(a.cost!=b.cost&&a.lo!=b.lo,"slots must differ");
    auto bad=a;bad.offsets[2]++;try{validate_image(bad);throw std::runtime_error("bad offset accepted");}catch(const std::runtime_error& e){require(std::string(e.what())!="bad offset accepted","bad offset accepted");}
    bad=a;bad.line_offsets[1]++;try{validate_image(bad);throw std::runtime_error("bad scratch accepted");}catch(const std::runtime_error& e){require(std::string(e.what())!="bad scratch accepted","bad scratch accepted");}
    auto oa=oracle(a),ob=oracle(b);require(oa.sums!=ob.sums&&oa.paths!=ob.paths,"oracle outputs must differ");
}
struct Slot {daxa::BufferId sums,paths,parameter;Parameters params;};
static void gpu_slots(){
    std::array<Fixture,2> f={image_fixture(0),image_fixture(1)};for(auto const& x:f)validate_image(x);
    std::array<Output,2> expected={oracle(f[0]),oracle(f[1])};require(expected[0].sums!=expected[1].sums,"distinct oracle results required");
    DaxaFoundation context(true);auto& d=context.device;std::vector<daxa::MemoryBlock> memory;std::vector<daxa::BufferId> buffers;ResourceOwner owner{d,memory,buffers};std::vector<Region> regions;std::array<Slot,2> slots;
    auto allocate=[&](size_t bytes,const void* data){require(bytes>0&&bytes<=256000000,"buffer budget");daxa::BufferInfo info{.size=bytes,.name="fp2 isolated slot"};auto m=d.create_memory({.requirements=context.coherent_readback_requirements(info),.flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});auto b=d.create_buffer_from_memory_block({.buffer_info=info,.memory_block=m});memory.push_back(std::move(m));buffers.push_back(b);if(data)std::memcpy(d.buffer_host_address(b).value(),data,bytes);regions.push_back({d.device_address(b).value(),bytes});return b;};
    for(size_t k=0;k<2;++k){auto& x=f[k];auto lo=allocate(x.lo.size()*4,x.lo.data()),hi=allocate(x.hi.size()*4,x.hi.data()),offset=allocate(x.offsets.size()*8,x.offsets.data()),cost=allocate(x.cost.size()*4,x.cost.data());
        auto sums=allocate(expected[k].sums.size()*4,nullptr),paths=allocate(expected[k].paths.size()*4,nullptr);std::memset(d.buffer_host_address(sums).value(),0xff,expected[k].sums.size()*4);std::memset(d.buffer_host_address(paths).value(),0xff,expected[k].paths.size()*4);
        auto scratch=allocate(x.scratch_words*4,nullptr),line=allocate(x.line_offsets.size()*8,x.line_offsets.data());
        Parameters params{d.device_address(lo).value(),d.device_address(hi).value(),d.device_address(offset).value(),d.device_address(cost).value(),d.device_address(sums).value(),d.device_address(paths).value(),d.device_address(scratch).value(),d.device_address(line).value(),x.w,x.h,x.d,x.p1,x.p2,x.max_lines};auto parameter=allocate(sizeof(params),&params);slots[k]={sums,paths,parameter,params};}
    validate_regions(regions);
    std::ifstream file(SF_FP2_SPIRV_PATH,std::ios::binary|std::ios::ate);require(bool(file),"missing SPIR-V");size_t bytes=size_t(file.tellg());require(bytes&&bytes%4==0&&bytes<=UINT32_MAX,"invalid SPIR-V");std::vector<uint32_t> words(bytes/4);file.seekg(0);require(bool(file.read(reinterpret_cast<char*>(words.data()),std::streamsize(bytes))),"SPIR-V read");
    {auto pipeline=d.create_compute_pipeline({.shader_info={.byte_code=words.data(),.byte_code_size=uint32_t(words.size())},.push_constant_size=sizeof(Push),.name="fp2 slots"});std::array<daxa::ExecutableCommandList,2> commands;
        for(size_t k=0;k<2;++k){auto recorder=d.create_command_recorder({.name="fp2 slot"});recorder.pipeline_barrier({.src_access=daxa::AccessConsts::HOST_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE});
            for(uint32_t pass=0;pass<4;++pass){if(pass)recorder.pipeline_barrier({.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE});recorder.set_pipeline(pipeline);recorder.push_constant(Push{d.device_address(slots[k].parameter).value(),pass});recorder.dispatch({pass<2?f[k].h:f[k].w,1,1});}
            recorder.pipeline_barrier({.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::HOST_READ});commands[k]=recorder.complete_current_commands();}
        std::array<uint64_t,2> submitted{};
        for(size_t k=0;k<2;++k){d.submit_commands({.command_lists={&commands[k],1}});submitted[k]=d.latest_queue_submit_index(daxa::QUEUE_MAIN);}
        require(submitted[0]<submitted[1],"submission identity");
        for(size_t k=0;k<2;++k){d.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=submitted[k],.timeout=30'000'000'000ull});require(std::memcmp(d.buffer_host_address(slots[k].sums).value(),expected[k].sums.data(),expected[k].sums.size()*4)==0,"slot sum mismatch");require(std::memcmp(d.buffer_host_address(slots[k].paths).value(),expected[k].paths.data(),expected[k].paths.size()*4)==0,"slot path mismatch");}
    }
    owner.release();context.close();require(context.validation_errors.load()==0&&context.validation_warnings.load()==0,"validation diagnostics");
}
int main(int argc,char** argv){try{require(argc==2,"use --cpu-only, --identity or --gpu");std::string mode=argv[1];if(mode=="--identity"){std::cout<<"{\"slots_source_sha256\":\""<<SF_FP2_SLOTS_HOST_SHA256<<"\",\"tiny_source_sha256\":\""<<SF_FP2_HOST_SHA256<<"\",\"shader_source_sha256\":\""<<SF_FP2_SHADER_SHA256<<"\",\"spirv_sha256\":\""<<SF_FP2_SPIRV_SHA256<<"\",\"compiler_sha256\":\""<<SF_DAXA_COMPILER_SHA256<<"\",\"compiler_path\":\""<<SF_FP2_COMPILER_PATH<<"\",\"spirv_path\":\""<<SF_FP2_SPIRV_PATH<<"\"}\n";return 0;}if(mode=="--cpu-only")cpu_slots();else if(mode=="--gpu")gpu_slots();else throw std::runtime_error("unknown mode");std::cout<<"FP2 two-slot "<<mode<<" contracts passed\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
