#include <stereoforge/daxa_view.hpp>
#include <stereoforge/daxa_view_parameters.hpp>
#include <daxa/utils/task_graph.hpp>
#include <daxa/c/device.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>

namespace stereoforge {
namespace {
using Parameters=detail::ViewParameters;
constexpr const char* entries[]={"clear","clear_layers","splat_layer_depth","splat_layer_owner","histogram_ratios","reduce_ratios","resolve_layers","gather_project_metadata","inspect_outputs"};
void require(bool ok,char const* error){if(!ok)throw std::invalid_argument(error);}
std::vector<uint32_t> shader(char const* entry){
    auto path=std::filesystem::path(SF_DAXA_VIEW_SPIRV_DIRECTORY)/(std::string("daxa-view-")+entry+".spv");
    std::ifstream in(path,std::ios::binary|std::ios::ate);
    if(!in)throw std::runtime_error("missing offline Daxa view shader: "+path.string());
    auto bytes=in.tellg();require(bytes>0&&bytes%4==0,"invalid offline SPIR-V size");
    std::vector<uint32_t> words(size_t(bytes)/4);in.seekg(0);in.read(reinterpret_cast<char*>(words.data()),bytes);
    if(!in)throw std::runtime_error("cannot read offline Daxa view shader");
    return words;
}
}
void DaxaView::validate(GeometryConfig c,GeometrySources const& s,VirtualView const& v,unsigned slots){
    uint64_t n=uint64_t(c.source_width)*c.source_height,m=uint64_t(c.target_width)*c.target_height;
    require(n&&m&&n<=UINT32_MAX/2&&m<=UINT32_MAX/2,"invalid Daxa view dimensions");
    require(slots>=1&&slots<=3,"Daxa view ring requires one to three slots");
    require(s.timestamp_ns>=0&&!s.calibration_id.empty(),"projection timestamp and calibration identity required");
    require(s.depth.size()==2*n&&s.confidence.size()==2*n&&s.rgb.size()==2*n&&s.provenance.size()==2*n&&s.rays.size()==2*n&&s.source_ids.size()==2*n&&s.age_ns.size()==2*n,"geometry source dimensions mismatch");
    require(s.preview_indices.empty()||s.preview_indices.size()==2*m,"preview calibration map dimensions mismatch");
    for(size_t i=0;i<s.preview_indices.size();++i){auto x=s.preview_indices[i];require(x==UINT32_MAX||(x<2*n&&x/n==i/m),"preview source index belongs to wrong eye or is out of range");}
    for(size_t i=0;i<2*n;++i){auto r=s.rays[i];require(std::isfinite(r.x)&&std::isfinite(r.y)&&r.z==1&&s.age_ns[i]>=0,"unit-Z finite rays and nonnegative ages required");
        if(std::isfinite(s.depth[i])&&s.depth[i]>0)require(std::isfinite(s.confidence[i])&&s.confidence[i]>=0&&s.confidence[i]<=1&&s.provenance[i]&&s.source_ids[i]>=0,"invalid source confidence, identity, or provenance");}
    require(v.mode==ViewMode::Strict||v.mode==ViewMode::Preview,"unsupported view mode");
    require(v.mode!=ViewMode::Preview||s.preview_indices.size()==2*m,"preview requires calibration map");
    require(v.projection==Projection::Perspective||v.projection==Projection::Cylindrical,"unsupported projection");
    require(std::isfinite(v.hfov_radians)&&v.hfov_radians>0&&v.hfov_radians<6.28318530718f,"invalid virtual horizontal FOV");
    require(std::isfinite(v.intrinsics.fx)&&std::isfinite(v.intrinsics.fy)&&v.intrinsics.fx>0&&v.intrinsics.fy>0&&std::isfinite(v.intrinsics.cx)&&std::isfinite(v.intrinsics.cy),"invalid virtual intrinsics");
    for(auto const* pose:{&v.source_to_virtual_left,&v.source_to_virtual_right}){
        auto const& p=*pose;
        for(float x:p)require(std::isfinite(x),"finite rigid pose required");
        require(std::abs(p[12])+std::abs(p[13])+std::abs(p[14])+std::abs(p[15]-1)<=1e-5f,"affine rigid pose required");
        for(int i=0;i<3;++i)for(int j=0;j<3;++j){float dot=0;for(int k=0;k<3;++k)dot+=p[k*4+i]*p[k*4+j];require(std::abs(dot-(i==j?1.f:0.f))<=1e-4f,"orthonormal rigid pose required");}
        const float determinant=p[0]*(p[5]*p[10]-p[6]*p[9])-p[1]*(p[4]*p[10]-p[6]*p[8])+p[2]*(p[4]*p[9]-p[5]*p[8]);
        require(determinant>0,"proper rigid rotation required");
    }
}
struct DaxaView::Impl {
    DaxaFoundation context;
    GeometryConfig config;
    std::shared_ptr<Diagnostics> diagnostics=std::make_shared<Diagnostics>();
    uint64_t bytes=0,next_generation=1;
    inline static std::atomic<uint64_t> next_context{1};
    uint64_t identity=next_context.fetch_add(1);
    int64_t timestamp=0;
    std::string calibration;
    float timestamp_period=0;
    uint32_t timestamp_bits=0;
    std::map<std::string,daxa::BufferId> inputs;
    std::vector<daxa::MemoryBlock> memory;
    std::vector<daxa::BufferId> owned;
    std::array<daxa::ComputePipeline,9> pipelines;
    struct Slot {
        std::map<std::string,daxa::BufferId> buffers;
        std::map<std::string,daxa::ExternalTaskBuffer> external;
        std::unique_ptr<daxa::TaskGraph> graph;
        daxa::TimelineQueryPool queries;
        std::weak_ptr<Lease> lease;
        uint64_t submit=0,generation=0;
        unsigned visited=0,finished=0;
    };
    std::vector<std::unique_ptr<Slot>> slots;
    Impl(GeometryConfig c,bool validation):context(validation),config(c){
        auto physical=daxa_dvc_get_vk_physical_device(context.device.get());
        VkPhysicalDeviceProperties p{};vkGetPhysicalDeviceProperties(physical,&p);timestamp_period=p.limits.timestampPeriod;
        uint32_t count=0;vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,nullptr);
        std::vector<VkQueueFamilyProperties> queues(count);vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,queues.data());
        uint32_t family=UINT32_MAX;
        auto result=daxa_dvc_get_vk_queue(context.device.get(),{DAXA_QUEUE_TYPE_MAIN,0},nullptr,&family);
        require(result==DAXA_RESULT_SUCCESS&&family<queues.size(),"cannot query actual Daxa main queue family");
        timestamp_bits=queues[family].timestampValidBits;
        require(timestamp_bits>0,"main queue has no supported GPU timestamps");
    }
    ~Impl(){
        // Teardown boundary only: leases hold this owner until downstream use ends.
        context.device.wait_idle();slots.clear();pipelines={};
        for(auto id:owned)if(context.device.is_id_valid(id))context.device.destroy_buffer(id);
        context.device.collect_garbage();memory.clear();
        context.close(); // Messenger remains alive through resource/device teardown.
        diagnostics->errors=context.validation_errors.load();diagnostics->warnings=context.validation_warnings.load();diagnostics->closed=true;
    }
    daxa::BufferId allocate(size_t size,std::string const& name,bool host=false){
        auto& d=context.device;daxa::BufferId id;
        daxa::BufferInfo info{.size=size,.name=name};
        if(host){auto block=d.create_memory({.requirements=context.coherent_readback_requirements(info),.flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});
            id=d.create_buffer_from_memory_block({.buffer_info=info,.memory_block=block});memory.push_back(std::move(block));}
        else id=d.create_buffer(info);
        owned.push_back(id);bytes+=size;return id;
    }
    void upload(daxa::CommandRecorder& recorder,std::string const& name,void const* data,size_t size){
        auto id=allocate(size,name);inputs[name]=id;auto staging=allocate(size,"setup upload "+name,true);
        std::memcpy(context.device.buffer_host_address(staging).value(),data,size);
        recorder.copy_buffer_to_buffer({.src_buffer=staging,.dst_buffer=id,.size=size});
    }
    void setup(GeometrySources const& s,VirtualView const& v,unsigned count);
};
struct DaxaView::Lease {
    std::shared_ptr<Impl> owner;
    size_t slot;
    uint64_t generation,submit;
    void validate() const {require(owner&&slot<owner->slots.size()&&owner->slots[slot]->generation==generation,"stale Daxa frame generation");}
};
void DaxaView::Impl::setup(GeometrySources const& s,VirtualView const& v,unsigned count){
    timestamp=s.timestamp_ns;calibration=s.calibration_id;
    auto& d=context.device;auto recorder=d.create_command_recorder({.name="one-time source upload"});
    auto put=[&](char const* name,auto span){upload(recorder,name,span.data(),span.size_bytes());};
    put("source_depth",s.depth);put("source_confidence",s.confidence);put("source_rgb",s.rgb);put("source_provenance",s.provenance);put("source_rays",s.rays);put("source_ids",s.source_ids);put("source_ages",s.age_ns);
    if(!s.preview_indices.empty())put("preview_indices",s.preview_indices);
    else {std::vector<uint32_t> invalid(2ull*config.target_width*config.target_height,UINT32_MAX);put("preview_indices",std::span<const uint32_t>(invalid));}
    recorder.pipeline_barrier({.src_access=daxa::AccessConsts::TRANSFER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ});
    auto command=recorder.complete_current_commands();d.submit_commands({.command_lists={&command,1}});
    d.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=d.latest_queue_submit_index(daxa::QUEUE_MAIN),.timeout=30'000'000'000ull});d.collect_garbage();
    for(size_t i=0;i<9;++i){auto words=shader(entries[i]);pipelines[i]=d.create_compute_pipeline({.shader_info={.byte_code=words.data(),.byte_code_size=uint32_t(words.size())},.push_constant_size=8,.name=entries[i]});}
    const size_t n=size_t(config.source_width)*config.source_height,m=size_t(config.target_width)*config.target_height;
    for(unsigned i=0;i<count;++i){
        slots.push_back(std::make_unique<Slot>());auto& slot=*slots.back();slot.buffers=inputs;
        for(char const* name:{"zbuffer","owner","output_rgb","output_valid","output_source","output_provenance","output_depth","output_confidence","output_rgb_valid","output_approximate"})slot.buffers[name]=allocate(4*m,name);
        for(char const* name:{"layer_z","layer_owner","output_source_ids","output_age_ns"})slot.buffers[name]=allocate(8*m,name);
        slot.buffers["ratio_histogram"]=allocate(3*4096*4,"ratio_histogram");slot.buffers["exposure_gain"]=allocate(12,"exposure_gain");
        slot.buffers["packed_readback"]=allocate(16,"downstream consumer counters");
        Parameters p{};
        p.address_packed_readback=d.device_address(slot.buffers.at("packed_readback")).value();
        p.address_source_depth=d.device_address(slot.buffers.at("source_depth")).value();
        p.address_source_confidence=d.device_address(slot.buffers.at("source_confidence")).value();
        p.address_source_rgb=d.device_address(slot.buffers.at("source_rgb")).value();
        p.address_source_provenance=d.device_address(slot.buffers.at("source_provenance")).value();
        p.address_source_rays=d.device_address(slot.buffers.at("source_rays")).value();
        p.address_zbuffer=d.device_address(slot.buffers.at("zbuffer")).value();
        p.address_owner=d.device_address(slot.buffers.at("owner")).value();
        p.address_output_rgb=d.device_address(slot.buffers.at("output_rgb")).value();
        p.address_output_valid=d.device_address(slot.buffers.at("output_valid")).value();
        p.address_output_source=d.device_address(slot.buffers.at("output_source")).value();
        p.address_output_provenance=d.device_address(slot.buffers.at("output_provenance")).value();
        p.address_output_depth=d.device_address(slot.buffers.at("output_depth")).value();
        p.address_output_confidence=d.device_address(slot.buffers.at("output_confidence")).value();
        p.address_layer_z=d.device_address(slot.buffers.at("layer_z")).value();
        p.address_layer_owner=d.device_address(slot.buffers.at("layer_owner")).value();
        p.address_ratio_histogram=d.device_address(slot.buffers.at("ratio_histogram")).value();
        p.address_exposure_gain=d.device_address(slot.buffers.at("exposure_gain")).value();
        p.address_preview_indices=d.device_address(slot.buffers.at("preview_indices")).value();
        p.address_output_rgb_valid=d.device_address(slot.buffers.at("output_rgb_valid")).value();
        p.address_output_approximate=d.device_address(slot.buffers.at("output_approximate")).value();
        p.address_source_ids=d.device_address(slot.buffers.at("source_ids")).value();
        p.address_source_ages=d.device_address(slot.buffers.at("source_ages")).value();
        p.address_output_source_ids=d.device_address(slot.buffers.at("output_source_ids")).value();
        p.address_output_age_ns=d.device_address(slot.buffers.at("output_age_ns")).value();
        p.field_src_width=config.source_width;p.field_src_height=config.source_height;p.field_target_width=config.target_width;p.field_target_height=config.target_height;
        p.field_projection=v.projection==Projection::Cylindrical;p.field_fx=v.intrinsics.fx;p.field_fy=v.intrinsics.fy;p.field_cx=v.intrinsics.cx;p.field_cy=v.intrinsics.cy;
        p.field_hfov_radians=v.hfov_radians;p.field_source_to_virtual_left=v.source_to_virtual_left;p.field_source_to_virtual_right=v.source_to_virtual_right;
        p.field_preview_enabled=v.mode==ViewMode::Preview;p.field_exposure_enabled=v.exposure_match;
        auto parameters=slot.buffers["parameters"]=allocate(sizeof(p),"immutable shader parameters",true);
        std::memcpy(d.buffer_host_address(parameters).value(),&p,sizeof(p));
        slot.queries=d.create_timeline_query_pool({.query_count=2,.name="compositor GPU timestamps"});
        auto* timed_slot=&slot;
        slot.graph=std::make_unique<daxa::TaskGraph>(daxa::TaskGraphInfo{.device=d,
            .pre_task_callback=[timed_slot](daxa::TaskInterface ti){
                if(timed_slot->visited++==0){ti.recorder.reset_timestamps({.query_pool=timed_slot->queries,.start_index=0,.count=2});ti.recorder.write_timestamp({.query_pool=timed_slot->queries,.pipeline_stage=daxa::PipelineStageFlagBits::TOP_OF_PIPE,.query_index=0});}
            },
            .post_task_callback=[timed_slot](daxa::TaskInterface ti){
                if(++timed_slot->finished==8)ti.recorder.write_timestamp({.query_pool=timed_slot->queries,.pipeline_stage=daxa::PipelineStageFlagBits::BOTTOM_OF_PIPE,.query_index=1});
            },.name="resident compositor"});auto& graph=*slot.graph;
        std::map<std::string,daxa::TaskBufferView> views;
        for(auto const& [name,id]:slot.buffers){slot.external.emplace(name,daxa::ExternalTaskBuffer({.buffer=id,.name=name}));views[name]=graph.register_buffer(slot.external.at(name));}
        // Exact per-entry access declarations. Physical pointers do not bypass graph hazards.
        struct Pass {std::vector<const char*> reads,writes,readwrites;size_t threads;};
        std::array<Pass,8> passes{{
            {{},{"zbuffer","owner","output_rgb","output_valid","output_source","output_provenance","output_depth","output_confidence"},{},m},
            {{},{"layer_z","layer_owner","ratio_histogram","exposure_gain","output_rgb_valid","output_approximate"},{},std::max(2*m,size_t(3*4096))},
            {{"source_depth","source_rays"},{},{"layer_z"},2*n},
            {{"source_depth","source_rays","layer_z"},{},{"layer_owner"},2*n},
            {{"layer_z","layer_owner","source_rgb"},{},{"ratio_histogram"},m},
            {{"ratio_histogram"},{},{"exposure_gain"},3},
            {{"layer_z","layer_owner","source_rgb","source_confidence","source_provenance","exposure_gain","preview_indices"},{},{"owner","output_rgb","output_valid","output_rgb_valid","output_source","output_approximate","output_provenance","output_depth","output_confidence"},m},
            {{"owner","output_valid","source_ids","source_ages"},{"output_source_ids","output_age_ns"},{},m}
        }};
        for(size_t pass=0;pass<passes.size();++pass){auto task=daxa::InlineTask::Compute(entries[pass]);
            task.reads(views.at("parameters"));
            for(auto name:passes[pass].reads)task.reads(views.at(name));
            for(auto name:passes[pass].writes)task.writes(views.at(name));
            for(auto name:passes[pass].readwrites)task.reads_writes(views.at(name));
            auto address=d.device_address(parameters).value();auto threads=passes[pass].threads;
            graph.add_task(task.executes([this,pass,address,threads](daxa::TaskInterface ti){
                ti.recorder.set_pipeline(pipelines[pass]);ti.recorder.push_constant(address);
                ti.recorder.dispatch({uint32_t((threads+(pass==5?1:64)-1)/(pass==5?1:64)),1,1});
            }));
        }
        graph.submit({});graph.complete({});
    }
}
DaxaView::DaxaView(GeometryConfig c,GeometrySources const& s,VirtualView const& v,unsigned slots,bool validation){
    validate(c,s,v,slots);impl_=std::make_shared<Impl>(c,validation);impl_->setup(s,v,slots);
}
DaxaView::Frame DaxaView::project(){
    auto& e=*impl_;e.context.device.collect_garbage();
    auto pending=e.context.device.oldest_pending_submit_index();
    for(size_t i=0;i<e.slots.size();++i){auto& s=*e.slots[i];if(!s.lease.expired()||(s.submit&&s.submit>=pending))continue;
        s.generation=e.next_generation++;s.visited=0;s.finished=0;s.graph->execute({});
        require(s.visited==8&&s.finished==8,"Daxa compositor scheduled callback count changed");s.submit=e.context.device.latest_queue_submit_index(daxa::QUEUE_MAIN);
        auto lease=std::make_shared<Lease>(Lease{impl_,i,s.generation,s.submit});s.lease=lease;return Frame(std::move(lease));}
    throw std::runtime_error("Daxa output ring exhausted: retained consumer or pending completion");
}
void DaxaView::Frame::wait()const{require(bool(lease_),"empty Daxa frame");lease_->validate();auto& d=lease_->owner->context.device;d.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=lease_->submit,.timeout=30'000'000'000ull});d.collect_garbage();}
uint64_t DaxaView::Frame::generation()const{require(bool(lease_),"empty Daxa frame");lease_->validate();return lease_->generation;}
daxa::BufferId DaxaView::Frame::buffer(std::string const& name)const{generation();return lease_->owner->slots[lease_->slot]->buffers.at(name);}
double DaxaView::Frame::gpu_ms()const{
    require(bool(lease_),"empty Daxa frame");lease_->validate();auto& e=*lease_->owner;require(lease_->submit<e.context.device.oldest_pending_submit_index(),"GPU timestamps require completed frame");auto values=e.slots[lease_->slot]->queries.get_query_results(0,2);
    require(values.size()==4&&values[1]&&values[3],"GPU timestamp results unavailable");
    uint64_t mask=e.timestamp_bits==64?~uint64_t(0):(uint64_t(1)<<e.timestamp_bits)-1;
    return double((values[2]-values[0])&mask)*double(e.timestamp_period)/1e6;
}
GeometryReadback DaxaView::readback(Frame const& frame)const{
    require(frame.lease_&&frame.lease_->owner==impl_,"foreign or empty Daxa frame context");frame.wait();auto& e=*impl_;auto& d=e.context.device;auto& slot=*e.slots[frame.lease_->slot];
    const size_t m=size_t(e.config.target_width)*e.config.target_height;
    // One explicit diagnostic allocation/copy outside the measured compositor.
    const size_t size=m*(9*4+2*8);
    daxa::BufferInfo info{.size=size,.name="diagnostic output readback"};
    auto memory=d.create_memory({.requirements=e.context.coherent_readback_requirements(info),.flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});
    auto host=d.create_buffer_from_memory_block({.buffer_info=info,.memory_block=memory});
    struct Cleanup {daxa::Device& device;daxa::BufferId id;~Cleanup(){device.destroy_buffer(id);device.collect_garbage();}} cleanup{d,host};
    daxa::ExternalTaskBuffer external({.buffer=host,.name="diagnostic host readback"});
    daxa::TaskGraph graph({.device=d,.name="diagnostic plane export"});auto dst=graph.register_buffer(external);
    struct Copy {daxa::TaskBufferView view;size_t offset,size;};std::vector<Copy> copies;
    size_t offset=0;
    for(char const* name:{"output_depth","output_confidence","output_rgb","output_valid","output_rgb_valid","output_provenance","output_source","output_approximate","owner","output_source_ids","output_age_ns"}){
        auto view=graph.register_buffer(slot.external.at(name));size_t bytes=m*(std::string(name)=="output_source_ids"||std::string(name)=="output_age_ns"?8:4);copies.push_back({view,offset,bytes});offset+=bytes;}
    auto task=daxa::InlineTask::Transfer("export eleven output planes");for(auto c:copies)task.reads(c.view);task.writes(dst);
    graph.add_task(task.executes([&copies,dst](daxa::TaskInterface ti){for(auto c:copies)ti.recorder.copy_buffer_to_buffer({.src_buffer=ti.get(c.view).id,.dst_buffer=ti.get(dst).id,.dst_offset=c.offset,.size=c.size});}));
    graph.add_task(daxa::InlineTask::Transfer("host visibility").host.reads(dst).executes([](daxa::TaskInterface){}));graph.submit({});graph.complete({});graph.execute({});
    d.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=d.latest_queue_submit_index(daxa::QUEUE_MAIN),.timeout=30'000'000'000ull});
    auto* ptr=static_cast<std::byte*>(d.buffer_host_address(host).value());offset=0;GeometryReadback out;
    auto get=[&](auto& values){values.resize(m);auto bytes=m*sizeof(values[0]);std::memcpy(values.data(),ptr+offset,bytes);offset+=bytes;};
    get(out.depth);get(out.confidence);get(out.rgb);get(out.validity);get(out.rgb_validity);get(out.provenance);get(out.source);get(out.approximate);get(out.owner);get(out.source_ids);get(out.age_ns);
    return out;
}
std::string DaxaView::device_report()const{return impl_->context.report();}
uint64_t DaxaView::allocated_bytes()const{return impl_->bytes;}
unsigned DaxaView::validation_errors()const{return impl_->context.validation_errors.load();}
}

namespace stereoforge { std::shared_ptr<DaxaView::Diagnostics const> DaxaView::diagnostics()const{return impl_->diagnostics;} }

namespace stereoforge {
std::array<uint32_t,4> DaxaView::inspect(Frame const& frame)const {
    require(frame.lease_&&frame.lease_->owner==impl_,"foreign or empty Daxa consumer context");
    frame.wait();auto& e=*impl_;auto& slot=*e.slots[frame.lease_->slot];auto& d=e.context.device;
    daxa::BufferInfo info{.size=16,.name="downstream counters host"};
    auto memory=d.create_memory({.requirements=e.context.coherent_readback_requirements(info),.flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});
    auto host=d.create_buffer_from_memory_block({.buffer_info=info,.memory_block=memory});
    struct Cleanup {daxa::Device& device;daxa::BufferId id;~Cleanup(){device.destroy_buffer(id);device.collect_garbage();}} cleanup{d,host};
    daxa::ExternalTaskBuffer external({.buffer=host,.name="downstream readback"});
    daxa::TaskGraph graph({.device=d,.name="downstream GPU consumer"});
    auto counter=graph.register_buffer(slot.external.at("packed_readback"));auto destination=graph.register_buffer(external);
    auto parameters=graph.register_buffer(slot.external.at("parameters"));
    graph.add_task(daxa::InlineTask::Transfer("clear consumer counters").writes(counter).executes([counter](daxa::TaskInterface ti){ti.recorder.clear_buffer({.buffer=ti.get(counter).id,.offset=0,.size=16,.clear_value=0});}));
    auto task=daxa::InlineTask::Compute("consume retained final output");task.reads(parameters).reads_writes(counter);
    for(char const* name:{"output_valid","output_approximate","output_source_ids","output_age_ns"})task.reads(graph.register_buffer(slot.external.at(name)));
    auto address=d.device_address(slot.buffers.at("parameters")).value();auto count=e.config.target_width*e.config.target_height;
    graph.add_task(task.executes([&e,address,count](daxa::TaskInterface ti){ti.recorder.set_pipeline(e.pipelines[8]);ti.recorder.push_constant(address);ti.recorder.dispatch({(count+63)/64,1,1});}));
    graph.add_task(daxa::InlineTask::Transfer("export four counters").reads(counter).writes(destination).executes([counter,destination](daxa::TaskInterface ti){ti.recorder.copy_buffer_to_buffer({.src_buffer=ti.get(counter).id,.dst_buffer=ti.get(destination).id,.size=16});}));
    graph.add_task(daxa::InlineTask::Transfer("consumer host visibility").host.reads(destination).executes([](daxa::TaskInterface){}));
    graph.submit({});graph.complete({});graph.execute({});
    d.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=d.latest_queue_submit_index(daxa::QUEUE_MAIN),.timeout=30'000'000'000ull});
    std::array<uint32_t,4> result{};std::memcpy(result.data(),d.buffer_host_address(host).value(),16);return result;
}
}

namespace stereoforge {
uint64_t DaxaView::Frame::context_id()const{generation();return lease_->owner->identity;}
int64_t DaxaView::Frame::timestamp_ns()const{generation();return lease_->owner->timestamp;}
std::string DaxaView::Frame::calibration_id()const{generation();return lease_->owner->calibration;}
GeometryConfig DaxaView::Frame::config()const{generation();return lease_->owner->config;}
}

namespace stereoforge { std::string DaxaView::shader_directory(){return SF_DAXA_VIEW_SPIRV_DIRECTORY;} }
