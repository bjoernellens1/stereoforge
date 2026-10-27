// Standalone native microbenchmark: frozen metadata gather only.
#include <stereoforge/daxa_foundation.hpp>
#include <daxa/utils/task_graph.hpp>
#include <daxa/c/device.h>
#include <array>
#include <bit>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string_view>
#include <stdexcept>
#include <vector>

namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
using namespace stereoforge;
static void require(bool value,char const* message){if(!value)throw std::runtime_error(message);}
static double elapsed(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
static std::string quote(std::string const& s){std::ostringstream o;o<<'"';for(char raw:s){auto c=static_cast<unsigned char>(raw);if(c=='"'||c=='\\')o<<'\\'<<c;else if(c<32)o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c)<<std::dec;else o<<c;}o<<'"';return o.str();}
struct Options {std::string variant="metadata-fused-64",readback_memory="first-coherent";uint32_t width=1280,height=720,frames=30,warmup=3;fs::path output,dump;bool validation=false;};
static uint32_t number(std::string const& s){size_t used=0;auto n=std::stoull(s,&used);if(s.empty()||s[0]=='-'||used!=s.size()||!n||n>UINT32_MAX)throw std::invalid_argument("invalid positive integer");return uint32_t(n);}
static Options options(int argc,char** argv){Options o;for(int i=1;i<argc;++i){std::string key=argv[i];if(key=="--validation"){o.validation=true;continue;}if(key!="--readback-memory"&&key!="--variant"&&key!="--width"&&key!="--height"&&key!="--frames"&&key!="--warmup"&&key!="--output"&&key!="--dump-prefix")throw std::invalid_argument("unknown argument: "+key);if(++i==argc)throw std::invalid_argument("missing argument value");std::string s=argv[i];if(key=="--readback-memory")o.readback_memory=s;else if(key=="--variant")o.variant=s;else if(key=="--width")o.width=number(s);else if(key=="--height")o.height=number(s);else if(key=="--frames")o.frames=number(s);else if(key=="--warmup")o.warmup=number(s);else if(key=="--output")o.output=s;else o.dump=s;}
    if(o.variant!="metadata-fused-64"&&o.variant!="metadata-split-64")throw std::invalid_argument("unsupported metadata variant");
    if(o.readback_memory!="first-coherent"&&o.readback_memory!="cached-coherent")throw std::invalid_argument("unsupported readback memory policy");
    if(o.frames<30||o.frames>300||o.warmup<3||o.warmup>30||o.output.empty()||uint64_t(o.width)*o.height>1920*1080)throw std::invalid_argument("requires --output, frames30..300, warmup3..30 and dimensions no larger than1080p");
    return o;}
struct Fixture {
    std::vector<uint32_t> owner,valid;
    std::vector<int64_t> ids,ages,expected_ids,expected_ages,result_ids,result_ages;
    uint64_t valid_count=0;
    explicit Fixture(size_t n):owner(n),valid(n),ids(n),ages(n),expected_ids(n),expected_ages(n),result_ids(n),result_ages(n){
        for(size_t i=0;i<n;++i){ids[i]=int64_t(((uint64_t(0x12345678u)+(i%1009))<<32)|(uint32_t(i)*2654435761u+17u));ages[i]=int64_t(((uint64_t(0x02468aceu)+(i%1013))<<32)|(uint32_t(i)*2246822519u+31u));}
        for(size_t i=0;i<n;++i){owner[i]=i%7==0?UINT32_MAX:(i%11==0?uint32_t(n):(i%13==0?uint32_t(n+1):uint32_t((uint64_t(i)*7919+23)%n)));valid[i]=i%5==0?0:(i%3==0?7:1);bool present=valid[i]!=0&&owner[i]!=UINT32_MAX&&owner[i]<n;valid_count+=present;expected_ids[i]=present?ids[owner[i]]:-1;expected_ages[i]=present?ages[owner[i]]:0;}
    }
    bool parity()const{return !std::memcmp(result_ids.data(),expected_ids.data(),ids.size()*8)&&!std::memcmp(result_ages.data(),expected_ages.data(),ages.size()*8);}
};
struct Parameters {uint64_t owner,valid,ids,ages,out_ids,out_ages;uint32_t count;};
static_assert(offsetof(Parameters,count)==48);
struct Sample {double input_stage,enqueue,completion_wait,readback_copy,query,gpu,total;};
static uint32_t select_readback_memory_type(uint32_t compatible,VkPhysicalDeviceMemoryProperties const& memory,std::string_view policy){
    require(policy=="first-coherent"||policy=="cached-coherent","unsupported readback memory policy");
    VkMemoryPropertyFlags required=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if(policy=="cached-coherent")required|=VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    for(uint32_t i=0;i<memory.memoryTypeCount;++i)
        if((compatible&(1u<<i))&&(memory.memoryTypes[i].propertyFlags&required)==required)return i;
    throw std::runtime_error("no compatible memory type satisfies requested coherent readback policy");
}
// Daxa requires names unique across every external resource in one graph.
// All twelve are fixed setup metadata; no per-frame name construction occurs.
static constexpr std::array<std::string_view,12> metadata_resource_names{
    "metadata source owner", "metadata source validity", "metadata source IDs", "metadata source ages",
    "metadata upload owner", "metadata upload validity", "metadata upload IDs", "metadata upload ages",
    "metadata output IDs", "metadata output ages", "metadata readback", "metadata parameters"
};
static_assert([] {
    for(size_t i=0;i<metadata_resource_names.size();++i){
        if(metadata_resource_names[i].empty()||metadata_resource_names[i].size()>=64)return false;
        for(size_t j=0;j<i;++j)if(metadata_resource_names[i]==metadata_resource_names[j])return false;
    }
    return true;
}(),"Daxa external resource names must be unique and fit SmallString");
struct Engine {
    DaxaFoundation context;
    std::vector<daxa::BufferId> owned;
    std::vector<daxa::MemoryBlock> memory;
    std::array<daxa::BufferId,4> inputs,uploads;
    std::array<daxa::BufferId,2> outputs;
    daxa::BufferId readback,parameters;
    std::array<daxa::ComputePipeline,3> pipelines;
    daxa::TimelineQueryPool queries;
    std::unique_ptr<daxa::TaskGraph> resident,transfer,diagnostic;
    std::array<daxa::ExternalTaskBuffer,4> input_external,upload_external;
    std::array<daxa::ExternalTaskBuffer,2> output_external;
    daxa::ExternalTaskBuffer readback_external,parameters_external;
    std::array<void*,4> upload_ptrs{};void* readback_ptr=nullptr;
    uint32_t readback_type_index=0,readback_property_flags=0;
    size_t n;uint64_t allocated=0;uint32_t timestamp_bits=0,queue_family=0,visited=0;float timestamp_period=0;bool split,closed=false;
    explicit Engine(Options const& o,Fixture const& f):context(o.validation),n(f.ids.size()),split(o.variant=="metadata-split-64"){
        auto& d=context.device;
        auto physical=daxa_dvc_get_vk_physical_device(d.get());VkPhysicalDeviceProperties props{};vkGetPhysicalDeviceProperties(physical,&props);timestamp_period=props.limits.timestampPeriod;
        uint32_t count=0;vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,nullptr);std::vector<VkQueueFamilyProperties> families(count);vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,families.data());
        require(daxa_dvc_get_vk_queue(d.get(),{DAXA_QUEUE_TYPE_MAIN,0},nullptr,&queue_family)==DAXA_RESULT_SUCCESS&&queue_family<count,"cannot query actual main queue");timestamp_bits=families[queue_family].timestampValidBits;require(timestamp_bits&&timestamp_period>0,"main queue timestamps unsupported");
        for(size_t i=0;i<4;++i){size_t bytes=n*(i<2?4:8);inputs[i]=allocate(bytes,false);uploads[i]=allocate(bytes,true);upload_ptrs[i]=d.buffer_host_address(uploads[i]).value();input_external[i]=daxa::ExternalTaskBuffer({.buffer=inputs[i],.name=metadata_resource_names[i]});upload_external[i]=daxa::ExternalTaskBuffer({.buffer=uploads[i],.name=metadata_resource_names[i+4]});}
        for(size_t i=0;i<2;++i){outputs[i]=allocate(n*8,false);output_external[i]=daxa::ExternalTaskBuffer({.buffer=outputs[i],.name=metadata_resource_names[i+8]});}
        readback=allocate_readback(n*16,o.readback_memory);readback_ptr=d.buffer_host_address(readback).value();readback_external=daxa::ExternalTaskBuffer({.buffer=readback,.name=metadata_resource_names[10]});
        Parameters p{d.device_address(inputs[0]).value(),d.device_address(inputs[1]).value(),d.device_address(inputs[2]).value(),d.device_address(inputs[3]).value(),d.device_address(outputs[0]).value(),d.device_address(outputs[1]).value(),uint32_t(n)};
        parameters=allocate(sizeof(p),true);std::memcpy(d.buffer_host_address(parameters).value(),&p,sizeof(p));parameters_external=daxa::ExternalTaskBuffer({.buffer=parameters,.name=metadata_resource_names[11]});
        const char* entries[]={"metadata_fused","metadata_ids","metadata_ages"};
        for(size_t i=0;i<3;++i){auto path=shader_path(entries[i]);std::ifstream in(path,std::ios::binary|std::ios::ate);require(bool(in),"missing offline metadata SPIR-V");auto size=in.tellg();require(size>0&&size%4==0,"invalid SPIR-V size");std::vector<uint32_t> words(size_t(size)/4);in.seekg(0);in.read(reinterpret_cast<char*>(words.data()),size);require(bool(in),"cannot read SPIR-V");pipelines[i]=d.create_compute_pipeline({.shader_info={.byte_code=words.data(),.byte_code_size=uint32_t(words.size())},.push_constant_size=8,.name=entries[i]});}
        queries=d.create_timeline_query_pool({.query_count=2,.name="metadata compute timestamps"});resident=graph(false);transfer=graph(true);diagnostic=graph(true,false);
    }
    static fs::path shader_path(char const* entry){return fs::path(SF_DAXA_METADATA_SPIRV_DIRECTORY)/(std::string("daxa-metadata-")+entry+".spv");}
    daxa::BufferId allocate(size_t bytes,bool host){daxa::BufferInfo info{.size=bytes,.name="metadata retained buffer"};auto& d=context.device;daxa::BufferId id;
        if(host){auto block=d.create_memory({.requirements=context.coherent_readback_requirements(info),.flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});id=d.create_buffer_from_memory_block({.buffer_info=info,.memory_block=block});memory.push_back(std::move(block));}else id=d.create_buffer(info);owned.push_back(id);allocated+=bytes;return id;}
    daxa::BufferId allocate_readback(size_t bytes,std::string_view policy){
        auto& d=context.device;daxa::BufferInfo info{.size=bytes,.name="metadata diagnostic readback"};
        auto requirements=d.buffer_memory_requirements(info);
        VkPhysicalDeviceMemoryProperties properties{};
        vkGetPhysicalDeviceMemoryProperties(daxa_dvc_get_vk_physical_device(d.get()),&properties);
        readback_type_index=select_readback_memory_type(requirements.memory_type_bits,properties,policy);
        readback_property_flags=properties.memoryTypes[readback_type_index].propertyFlags;
        requirements.memory_type_bits=1u<<readback_type_index;
        auto block=d.create_memory({.requirements=requirements,.flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});
        auto id=d.create_buffer_from_memory_block({.buffer_info=info,.memory_block=block});
        memory.push_back(std::move(block));owned.push_back(id);allocated+=bytes;return id;
    }
    std::unique_ptr<daxa::TaskGraph> graph(bool copy,bool dispatch=true){
        auto g=std::make_unique<daxa::TaskGraph>(daxa::TaskGraphInfo{.device=context.device,.reorder_tasks=true,.name=copy?"metadata transfer inclusive":"metadata resident"});
        std::array<daxa::TaskBufferView,4> in,up;std::array<daxa::TaskBufferView,2> out;
        for(size_t i=0;i<4;++i){in[i]=g->register_buffer(input_external[i]);if(copy)up[i]=g->register_buffer(upload_external[i]);}for(size_t i=0;i<2;++i)out[i]=g->register_buffer(output_external[i]);auto par=g->register_buffer(parameters_external);
        if(copy&&dispatch){auto task=daxa::InlineTask::Transfer("upload all metadata inputs");for(size_t i=0;i<4;++i)task.reads(up[i]).writes(in[i]);g->add_task(task.executes([this,in,up](daxa::TaskInterface ti){for(size_t i=0;i<4;++i)ti.recorder.copy_buffer_to_buffer({.src_buffer=ti.get(up[i]).id,.dst_buffer=ti.get(in[i]).id,.size=n*(i<2?4:8)});}));}
        for(unsigned pass=0;dispatch&&pass<(split?2u:1u);++pass){
            auto task=daxa::InlineTask::Compute(split?(pass?"metadata_ages":"metadata_ids"):"metadata_fused");
            task.reads(par).reads(in[0]).reads(in[1]);
            if(split)task.reads(in[pass+2]).writes(out[pass]);
            else task.reads(in[2]).reads(in[3]).writes(out[0]).writes(out[1]);
            g->add_task(task.executes([this,pass](daxa::TaskInterface ti){
                // ID and age passes are independent. Timestamp whichever callback
                // Daxa actually schedules first/last, without an output dependency.
                require(visited<(split?2u:1u),"too many metadata dispatch callbacks");
                if(visited++==0){
                    ti.recorder.reset_timestamps({.query_pool=queries,.start_index=0,.count=2});
                    ti.recorder.write_timestamp({.query_pool=queries,.pipeline_stage=daxa::PipelineStageFlagBits::TOP_OF_PIPE,.query_index=0});
                }
                ti.recorder.set_pipeline(pipelines[split?pass+1:0]);
                ti.recorder.push_constant(context.device.device_address(parameters).value());
                ti.recorder.dispatch({uint32_t((n+63)/64),1,1});
                if(visited==(split?2u:1u))ti.recorder.write_timestamp({.query_pool=queries,.pipeline_stage=daxa::PipelineStageFlagBits::BOTTOM_OF_PIPE,.query_index=1});
            }));
        }
        if(copy){auto rb=g->register_buffer(readback_external);g->add_task(daxa::InlineTask::Transfer("download both metadata outputs").reads(out[0]).reads(out[1]).writes(rb).executes([this,out,rb](daxa::TaskInterface ti){for(size_t i=0;i<2;++i)ti.recorder.copy_buffer_to_buffer({.src_buffer=ti.get(out[i]).id,.dst_buffer=ti.get(rb).id,.dst_offset=i*n*8,.size=n*8});}));g->add_task(daxa::InlineTask::Transfer("readback host visibility").host.reads(rb).executes([](daxa::TaskInterface){}));}
        g->submit({});g->complete({});return g;
    }
    void stage(Fixture const& f){std::memcpy(upload_ptrs[0],f.owner.data(),n*4);std::memcpy(upload_ptrs[1],f.valid.data(),n*4);std::memcpy(upload_ptrs[2],f.ids.data(),n*8);std::memcpy(upload_ptrs[3],f.ages.data(),n*8);}
    void initialize(Fixture const& f){stage(f);auto& d=context.device;auto recorder=d.create_command_recorder({.name="metadata setup upload"});for(size_t i=0;i<4;++i)recorder.copy_buffer_to_buffer({.src_buffer=uploads[i],.dst_buffer=inputs[i],.size=n*(i<2?4:8)});recorder.pipeline_barrier({.src_access=daxa::AccessConsts::TRANSFER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ});auto command=recorder.complete_current_commands();d.submit_commands({.command_lists={&command,1}});d.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=d.latest_queue_submit_index(daxa::QUEUE_MAIN),.timeout=30'000'000'000ull});d.collect_garbage();}
    void unpack(Fixture& f){std::memcpy(f.result_ids.data(),readback_ptr,n*8);std::memcpy(f.result_ages.data(),static_cast<std::byte*>(readback_ptr)+n*8,n*8);}
    Sample run(bool copy,Fixture& f){
        auto& d=context.device;
        auto start=Clock::now();
        if(copy)stage(f);
        auto staged=Clock::now();
        d.collect_garbage();visited=0;(copy?transfer:resident)->execute({});
        require(visited==(split?2u:1u),"metadata dispatch count changed");
        auto submitted=Clock::now();
        d.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=d.latest_queue_submit_index(daxa::QUEUE_MAIN),.timeout=30'000'000'000ull});
        d.collect_garbage();
        auto completed=Clock::now();
        if(copy)unpack(f);
        auto copied=Clock::now();
        // The C++ convenience getter allocates a vector per sample. The pinned
        // C API fills timestamp/availability pairs into this fixed stack array.
        std::array<uint64_t,4> values{};
        require(daxa_timeline_query_pool_query_results(queries.get(),0,2,values.data())==DAXA_RESULT_SUCCESS&&values[1]&&values[3],"timestamp unavailable");
        require(timestamp_bits>0&&timestamp_bits<=64,"invalid timestamp bit count");
        uint64_t mask=timestamp_bits==64?~uint64_t(0):(uint64_t(1)<<timestamp_bits)-1;
        auto gpu=double((values[2]-values[0])&mask)*double(timestamp_period)/1e6;
        auto end=Clock::now(); // CPU query retrieval/validation is included.
        return {elapsed(start,staged),elapsed(staged,submitted),elapsed(submitted,completed),elapsed(completed,copied),elapsed(copied,end),gpu,elapsed(start,end)};
    }
    void read_outputs(Fixture& f){auto& d=context.device;d.collect_garbage();diagnostic->execute({});d.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=d.latest_queue_submit_index(daxa::QUEUE_MAIN),.timeout=30'000'000'000ull});d.collect_garbage();unpack(f);}
    void close(){if(closed)return;auto& d=context.device;d.wait_idle();resident.reset();transfer.reset();diagnostic.reset();input_external={};upload_external={};output_external={};readback_external={};parameters_external={};pipelines={};queries={};for(auto id:owned)d.destroy_buffer(id);d.collect_garbage();memory.clear();context.close();closed=true;}
    ~Engine(){if(!closed)close();}
};
template<class T>static void binary(fs::path const& path,std::vector<T> const& values){std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<char const*>(values.data()),static_cast<std::streamsize>(values.size()*sizeof(T)));require(bool(out),"cannot write metadata dump");}
static void samples(std::ostream& out,std::vector<Sample> const& rows,double Sample::*field){out<<'[';for(size_t i=0;i<rows.size();++i){if(i)out<<',';out<<rows[i].*field;}out<<']';}
static void phase_rows(std::ostream& out,std::vector<Sample> const& rows){
    out<<'[';for(size_t i=0;i<rows.size();++i){if(i)out<<',';auto const& s=rows[i];
        out<<"{\"input_stage\":"<<s.input_stage<<",\"enqueue\":"<<s.enqueue<<",\"completion_wait\":"<<s.completion_wait
            <<",\"readback_copy\":"<<s.readback_copy<<",\"query\":"<<s.query<<",\"total\":"<<s.total<<'}';
    }out<<']';
}
int main(int argc,char** argv){try{
    if(argc==2&&std::string(argv[1])=="--shader-directory"){std::cout<<SF_DAXA_METADATA_SPIRV_DIRECTORY<<'\n';return 0;}
    if(argc==2&&std::string(argv[1])=="--daxa-source-directory"){std::cout<<SF_DAXA_METADATA_DAXA_SOURCE_DIRECTORY<<'\n';return 0;}
    static_assert(std::endian::native==std::endian::little,"metadata fixture ABI requires little endian");
    auto o=options(argc,argv);Fixture f(size_t(o.width)*o.height);auto init=Clock::now();Engine e(o,f);e.initialize(f);auto setup_ms=elapsed(init,Clock::now());
    for(unsigned i=0;i<o.warmup;++i)e.run(false,f);
    std::vector<Sample> resident(o.frames),transfer(o.frames);
    for(auto& s:resident)s=e.run(false,f);
    // Diagnostic full output comparison after resident sampling, outside its timing.
    e.read_outputs(f);bool parity=f.parity();require(parity,"resident metadata CPU oracle byte mismatch");
    for(unsigned i=0;i<o.warmup;++i)e.run(true,f);
    for(auto& s:transfer)s=e.run(true,f);
    require(f.parity(),"transfer metadata CPU oracle byte mismatch");
    auto dump_start=Clock::now();if(!o.dump.empty()){if(!o.dump.parent_path().empty())fs::create_directories(o.dump.parent_path());binary(o.dump.string()+".source_ids.i64",f.result_ids);binary(o.dump.string()+".age_ns.i64",f.result_ages);}auto dump_ms=elapsed(dump_start,Clock::now());
    auto identity=e.context.report();auto allocated=e.allocated;auto bits=e.timestamp_bits;auto period=e.timestamp_period;auto family=e.queue_family;e.close();auto errors=e.context.validation_errors.load(),warnings=e.context.validation_warnings.load();require(!errors&&!warnings,"metadata Vulkan validation errors or warnings (including teardown)");
    if(!o.output.parent_path().empty())fs::create_directories(o.output.parent_path());
    std::ofstream out(o.output);require(bool(out),"cannot create metadata report");out<<std::setprecision(12);
    out<<"{\"schema\":2,\"variant\":"<<quote(o.variant)<<",\"width\":"<<o.width<<",\"height\":"<<o.height<<",\"frames\":"<<o.frames<<",\"warmup\":"<<o.warmup<<",\"validation_enabled\":"<<(o.validation?"true":"false")<<",\"validation_errors\":"<<errors<<",\"validation_warnings\":"<<warnings<<",\"validation_includes_teardown\":true,\"numeric_parity\":true,\"oracle\":\"Full byte comparison of both signed int64 planes against deterministic CPU reference; zero tolerance\",\"allocated_bytes\":"<<allocated<<",\"application_requested_buffer_growth_bytes\":0,\"requested_buffer_bytes_scope\":\"Fixed application-requested buffers only: retained input, output, upload, readback and parameters; excludes allocator padding, Daxa internals, Vulkan and process memory; total memory growth not measured\",\"initialization_and_upload_ms\":"<<setup_ms<<",\"diagnostic_dump_ms\":"<<dump_ms<<",\"precision\":\"signed-int64-bit-exact\",\"quality\":\"strict\",\"workgroup\":[64,1,1],\"subgroup_policy\":\"device-default\",\"dispatch_count\":"<<(e.split?2:1)<<",\"timestamp_queue_family\":"<<family<<",\"timestamp_valid_bits\":"<<bits<<",\"timestamp_period_ns\":"<<period<<",\"device_report\":"<<identity<<",\"offline_shader_directory\":"<<quote(SF_DAXA_METADATA_SPIRV_DIRECTORY)<<",\"readback_scope\":\"Only the diagnostic readback allocation changes memory policy; uploads and parameters retain foundation first-coherent allocations\"";
    out<<",\"readback_memory_policy\":"<<quote(o.readback_memory)<<",\"readback_memory_type_index\":"<<e.readback_type_index<<",\"readback_memory_property_flags\":"<<e.readback_property_flags<<",\"resident_cpu_phase_ms\":";phase_rows(out,resident);out<<",\"transfer_cpu_phase_ms\":";phase_rows(out,transfer);out<<",\"cpu_phase_scope\":\"Contiguous nonoverlapping stages: input CPU copy, enqueue including completed-resource GC, completion wait including post-wait GC, CPU readback copy, timestamp query/validation; total spans all stages\",\"shader_files\":[";
    if(e.split)out<<quote(Engine::shader_path("metadata_ids").string())<<','<<quote(Engine::shader_path("metadata_ages").string());else out<<quote(Engine::shader_path("metadata_fused").string());out<<"],\"memory_traffic_model\":{\"scope\":\"Logical shader bytes, excludes caching, coalescing, parameters and barriers; fused avoids duplicate owner and validity loads\",\"owner_valid_read_bytes\":"<<f.ids.size()*(e.split?16:8)<<",\"source_read_bytes\":"<<f.valid_count*16<<",\"output_write_bytes\":"<<f.ids.size()*16<<"},\"transfer_upload_bytes_per_frame\":"<<f.ids.size()*24<<",\"transfer_readback_bytes_per_frame\":"<<f.ids.size()*16<<",\"gpu_timing_scope\":\"Compute dispatches between actual first and last scheduled compute callbacks; split output planes are independent with no artificial read hazard; excludes uploads and readback\",\"resident_scope\":\"Retained inputs; enqueue plus completion wait and CPU timestamp retrieval; setup upload and diagnostic readback excluded\",\"transfer_inclusive_scope\":\"Each frame copies CPU fixture to preallocated coherent host staging, uploads all four inputs, dispatches, copies both outputs to coherent host staging, waits, copies both outputs to preallocated CPU vectors, retrieves timestamps; no amortization\",\"scope\":\"Frozen projection metadata only; no camera, full stereo or 60 Hz qualification\",\"gpu_ms_samples\":";samples(out,resident,&Sample::gpu);out<<",\"cpu_enqueue_ms_samples\":";samples(out,resident,&Sample::enqueue);out<<",\"cpu_wait_ms_samples\":";samples(out,resident,&Sample::completion_wait);out<<",\"resident_ms_samples\":";samples(out,resident,&Sample::total);out<<",\"transfer_inclusive_ms_samples\":";samples(out,transfer,&Sample::total);out<<",\"transfer_gpu_ms_samples\":";samples(out,transfer,&Sample::gpu);out<<",\"transfer_cpu_enqueue_ms_samples\":";samples(out,transfer,&Sample::enqueue);out<<",\"transfer_cpu_completion_wait_ms_samples\":";samples(out,transfer,&Sample::completion_wait);out<<"}\n";require(bool(out),"cannot write metadata report");return 0;
}catch(std::exception const& e){std::cerr<<"daxa_variant_benchmark: "<<e.what()<<'\n';return 2;}}
