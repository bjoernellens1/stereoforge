#include <stereoforge/daxa_view.hpp>
#include <stereoforge/daxa_depth.hpp>
#include <stereoforge/daxa_view_parameters.hpp>
#include <stereoforge/daxa_view_calibration.hpp>
#include <thread>
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
std::vector<uint32_t> shader(char const* entry,bool typed=false){
    auto path=std::filesystem::path(SF_DAXA_VIEW_SPIRV_DIRECTORY)/(std::string(typed?"daxa-view-depth-":"daxa-view-")+entry+".spv");
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
struct DaxaView::Slot {
    std::map<std::string,daxa::BufferId,std::less<>> buffers;
    std::map<std::string,daxa::ExternalTaskBuffer,std::less<>> external;
    std::unique_ptr<daxa::TaskGraph> graph;daxa::TimelineQueryPool queries;
    DaxaDepthPipeline::Frame input;CaptureLease source;
    int64_t timestamp=0;uint64_t submit=0,generation=0;unsigned visited=0,finished=0;
};
struct DaxaView::Impl {
    std::shared_ptr<DaxaFoundation> context;
    bool owns_context=false,typed=false;
    std::shared_ptr<const DaxaRectification> maps;
    DaxaViewCalibration cache;
    VirtualView virtual_view;
    const std::thread::id thread=std::this_thread::get_id();
    void require_thread()const{require(thread==std::this_thread::get_id(),"view requires one submission thread");}
    GeometryConfig config;
    std::shared_ptr<Diagnostics> diagnostics=std::make_shared<Diagnostics>();
    uint64_t bytes=0,next_generation=1;
    int64_t timestamp=0;
    std::string calibration;
    float timestamp_period=0;
    uint32_t timestamp_bits=0;uint64_t storage_range=0;uint32_t max_dispatch_x=0;
    std::map<std::string,daxa::BufferId> inputs;
    std::vector<daxa::MemoryBlock> memory;
    std::vector<daxa::BufferId> owned;
    std::array<daxa::ComputePipeline,9> pipelines;
    std::vector<std::shared_ptr<Slot>> slots;
    Impl(GeometryConfig c,bool validation,std::shared_ptr<DaxaFoundation> supplied={}):context(supplied?supplied:std::make_shared<DaxaFoundation>(validation)),owns_context(!supplied),config(c){
        auto physical=daxa_dvc_get_vk_physical_device(context->device.get());
        VkPhysicalDeviceProperties p{};vkGetPhysicalDeviceProperties(physical,&p);timestamp_period=p.limits.timestampPeriod;storage_range=p.limits.maxStorageBufferRange;max_dispatch_x=p.limits.maxComputeWorkGroupCount[0];require(p.limits.maxComputeWorkGroupInvocations>=64&&p.limits.maxComputeWorkGroupSize[0]>=64,"view workgroup64 unavailable");
        uint32_t count=0;vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,nullptr);
        std::vector<VkQueueFamilyProperties> queues(count);vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,queues.data());
        uint32_t family=UINT32_MAX;
        auto result=daxa_dvc_get_vk_queue(context->device.get(),{DAXA_QUEUE_TYPE_MAIN,0},nullptr,&family);
        require(result==DAXA_RESULT_SUCCESS&&family<queues.size(),"cannot query actual Daxa main queue family");
        timestamp_bits=queues[family].timestampValidBits;
        require(timestamp_bits>0,"main queue has no supported GPU timestamps");
    }
    ~Impl(){
        // Teardown boundary only: leases hold this owner until downstream use ends.
        context->device.wait_idle();slots.clear();pipelines={};
        for(auto id:owned)if(context->device.is_id_valid(id))context->device.destroy_buffer(id);
        context->device.collect_garbage();memory.clear();
        if(owns_context)context->close(); // Messenger remains alive through resource/device teardown.
        diagnostics->errors=context->validation_errors.load();diagnostics->warnings=context->validation_warnings.load();diagnostics->closed=true;
    }
    daxa::BufferId allocate(size_t size,std::string const& name,bool host=false){
        require(size&&size<=storage_range&&bytes+size<2ull*1024*1024*1024,"view buffer allocation exceeds checked storage/2GiB budget");auto& d=context->device;daxa::BufferId id;
        daxa::BufferInfo info{.size=size,.name=name};
        if(host){auto block=d.create_memory({.requirements=context->coherent_readback_requirements(info),.flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});
            id=d.create_buffer_from_memory_block({.buffer_info=info,.memory_block=block});memory.push_back(std::move(block));}
        else id=d.create_buffer(info);
        owned.push_back(id);bytes+=size;return id;
    }
    void upload(daxa::CommandRecorder& recorder,std::string const& name,void const* data,size_t size){
        auto id=allocate(size,name);inputs[name]=id;auto staging=allocate(size,"setup upload "+name,true);
        std::memcpy(context->device.buffer_host_address(staging).value(),data,size);
        recorder.copy_buffer_to_buffer({.src_buffer=staging,.dst_buffer=id,.size=size});
    }
    #ifdef SF_DAXA_VIEW_HAS_DEPTH
    size_t collect_at(uint64_t pending){size_t retired=0;for(auto& slot:slots)if(slot.use_count()==1&&(!slot->submit||slot->submit<pending)&&slot->input){slot->input={};slot->source.reset();++retired;}return retired;}
    #else
    size_t collect_at(uint64_t){return 0;}
    #endif
    void setup(GeometrySources const& s,VirtualView const& v,unsigned count);
    void setup_slots(VirtualView const&,unsigned);
    void validate_frame(DaxaDepthPipeline::Frame const&)const;
    void bind(Slot&,DaxaDepthPipeline::Frame const&);
};
DaxaView::Frame::Frame(std::shared_ptr<Impl> owner,std::shared_ptr<Slot> slot):owner_(std::move(owner)),slot_(std::move(slot)),generation_(slot_->generation){}
DaxaView::Frame& DaxaView::Frame::operator=(Frame other)noexcept{
    // Keep both old references together: the temporary destroys its slot before
    // its owner, including when this frame is the last owner of the context.
    owner_.swap(other.owner_);slot_.swap(other.slot_);std::swap(generation_,other.generation_);return *this;
}
void DaxaView::Frame::validate()const{require(owner_&&slot_&&generation_==slot_->generation,"stale/empty Daxa view frame");}
void DaxaView::Impl::setup(GeometrySources const& s,VirtualView const& v,unsigned count){
    timestamp=s.timestamp_ns;calibration=s.calibration_id;
    auto& d=context->device;auto recorder=d.create_command_recorder({.name="one-time source upload"});
    auto put=[&](char const* name,auto span){upload(recorder,name,span.data(),span.size_bytes());};
    put("source_depth",s.depth);put("source_confidence",s.confidence);put("source_rgb",s.rgb);put("source_provenance",s.provenance);put("source_rays",s.rays);put("source_ids",s.source_ids);put("source_ages",s.age_ns);
    if(!s.preview_indices.empty())put("preview_indices",s.preview_indices);
    else {std::vector<uint32_t> invalid(2ull*config.target_width*config.target_height,UINT32_MAX);put("preview_indices",std::span<const uint32_t>(invalid));}
    recorder.pipeline_barrier({.src_access=daxa::AccessConsts::TRANSFER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ});
    auto command=recorder.complete_current_commands();d.submit_commands({.command_lists={&command,1}});
    d.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=d.latest_queue_submit_index(daxa::QUEUE_MAIN),.timeout=30'000'000'000ull});d.collect_garbage();
    setup_slots(v,count);
}
void DaxaView::Impl::setup_slots(VirtualView const& v,unsigned count){
    virtual_view=v;auto& d=context->device;
    for(size_t i=0;i<9;++i){auto words=shader(entries[i],typed);pipelines[i]=d.create_compute_pipeline({.shader_info={.byte_code=words.data(),.byte_code_size=uint32_t(words.size())},.push_constant_size=8,.name=entries[i]});}
    const size_t n=size_t(config.source_width)*config.source_height,m=size_t(config.target_width)*config.target_height;
    for(unsigned i=0;i<count;++i){
        slots.push_back(std::make_shared<Slot>());auto& slot=*slots.back();for(auto const& [name,id]:inputs)slot.buffers[name]=id;slot.timestamp=timestamp;
        for(char const* name:{"zbuffer","owner","output_rgb","output_valid","output_source","output_provenance","output_depth","output_confidence","output_rgb_valid","output_approximate"})slot.buffers[name]=allocate(4*m,name);
        for(char const* name:{"layer_z","layer_owner","output_source_ids","output_age_ns"})slot.buffers[name]=allocate(8*m,name);
        slot.buffers["ratio_histogram"]=allocate(3*4096*4,"ratio_histogram");slot.buffers["exposure_gain"]=allocate(12,"exposure_gain");
        slot.buffers["packed_readback"]=allocate(16,"downstream consumer counters");
        detail::DepthViewParameters extended{};auto& p=extended.view;
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
        if(typed){extended.rectified_right_rgb=d.device_address(slot.buffers.at("rectified_right_rgb")).value();extended.original_left_rgb=d.device_address(slot.buffers.at("original_left_rgb")).value();extended.original_right_rgb=d.device_address(slot.buffers.at("original_right_rgb")).value();extended.original_width=cache.original_width;extended.original_height=cache.original_height;}
        auto parameters=slot.buffers["parameters"]=allocate(typed?sizeof(extended):sizeof(p),"immutable shader parameters",true);
        std::memcpy(d.buffer_host_address(parameters).value(),&extended,typed?sizeof(extended):sizeof(p));
        slot.queries=d.create_timeline_query_pool({.query_count=2,.name="compositor GPU timestamps"});
        auto* timed_slot=&slot;
        slot.graph=std::make_unique<daxa::TaskGraph>(daxa::TaskGraphInfo{.device=d,
            .pre_task_callback=[timed_slot](daxa::TaskInterface ti){
                if(timed_slot->visited++==0){ti.recorder.pipeline_barrier({.src_access=daxa::AccessConsts::HOST_WRITE|daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ});ti.recorder.reset_timestamps({.query_pool=timed_slot->queries,.start_index=0,.count=2});ti.recorder.write_timestamp({.query_pool=timed_slot->queries,.pipeline_stage=daxa::PipelineStageFlagBits::TOP_OF_PIPE,.query_index=0});}
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
            if(typed&&(pass==4||pass==6))task.reads(views.at("rectified_right_rgb"));
            if(typed&&pass==6)task.reads(views.at("original_left_rgb")).reads(views.at("original_right_rgb"));
            for(auto name:passes[pass].writes)task.writes(views.at(name));
            for(auto name:passes[pass].readwrites)task.reads_writes(views.at(name));
            auto address=d.device_address(parameters).value();auto threads=passes[pass].threads;require((threads+(pass==5?1:64)-1)/(pass==5?1:64)<=max_dispatch_x,"view dispatch exceeds actual device group limit");
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
#ifdef SF_DAXA_VIEW_HAS_DEPTH
void DaxaView::Impl::validate_frame(DaxaDepthPipeline::Frame const& frame)const{
    require(frame&&frame.context_id()==reinterpret_cast<uintptr_t>(context->device.get()),"foreign depth context");require(frame.calibration()==maps,"foreign immutable map/model owner");require(frame.width()==config.source_width&&frame.height()==config.source_height&&frame.layers()==2,"typed depth dimensions mismatch");auto n=size_t(config.source_width)*config.source_height;for(auto plane:{DaxaDepthPlane::Depth,DaxaDepthPlane::MetricConfidence,DaxaDepthPlane::MetricValidity,DaxaDepthPlane::Provenance})require(frame.size_bytes(plane)==2*n*4,"typed depth plane size mismatch");for(auto plane:{DaxaDepthPlane::SourceIds,DaxaDepthPlane::AgeNs})require(frame.size_bytes(plane)==2*n*8,"typed metadata plane size mismatch");require(frame.source()&&frame.source()->host_ingress_timestamp_ns<=uint64_t(INT64_MAX),"view timestamp outside signed64 domain");
}
void DaxaView::Impl::bind(Slot& slot,DaxaDepthPipeline::Frame const& frame){
    auto& d=context->device;auto* extended=reinterpret_cast<detail::DepthViewParameters*>(d.buffer_host_address(slot.buffers.find("parameters")->second).value());auto& p=extended->view;
    auto set=[&](const char* name,daxa::BufferId id,uint64_t& address){slot.buffers.find(name)->second=id;slot.external.find(name)->second.set_buffer(id);address=d.device_address(id).value();};
    set("source_depth",frame.buffer(DaxaDepthPlane::Depth),p.address_source_depth);set("source_confidence",frame.buffer(DaxaDepthPlane::MetricConfidence),p.address_source_confidence);set("source_provenance",frame.buffer(DaxaDepthPlane::Provenance),p.address_source_provenance);set("source_ids",frame.buffer(DaxaDepthPlane::SourceIds),p.address_source_ids);set("source_ages",frame.buffer(DaxaDepthPlane::AgeNs),p.address_source_ages);
    set("source_rgb",frame.image(DaxaPreprocessPlane::RectifiedLeft).buffer(),p.address_source_rgb);set("rectified_right_rgb",frame.image(DaxaPreprocessPlane::RectifiedRight).buffer(),extended->rectified_right_rgb);set("original_left_rgb",frame.image(DaxaPreprocessPlane::OriginalLeft).buffer(),extended->original_left_rgb);set("original_right_rgb",frame.image(DaxaPreprocessPlane::OriginalRight).buffer(),extended->original_right_rgb);
    slot.input=frame;slot.source=frame.source();slot.timestamp=int64_t(slot.source->host_ingress_timestamp_ns);
}
DaxaView::DaxaView(GeometryConfig c,VirtualView const& v,std::shared_ptr<DaxaFoundation> context,DaxaDepthPipeline::Frame const& prototype,unsigned count){
    require(context&&prototype&&prototype.context_id()==reinterpret_cast<uintptr_t>(context->device.get()),"shared foundation typed intake requires matching context");require(count>=1&&count<=3,"view ring requires1..3 slots");auto maps=prototype.calibration();auto cache=prepare_daxa_view_calibration(*maps,c,v);
    impl_=std::make_shared<Impl>(c,false,std::move(context));auto& e=*impl_;e.typed=true;e.maps=std::move(maps);e.cache=std::move(cache);e.calibration=e.maps->rectified_identity;e.validate_frame(prototype);auto& d=e.context->device;
    e.inputs["source_depth"]=prototype.buffer(DaxaDepthPlane::Depth);e.inputs["source_confidence"]=prototype.buffer(DaxaDepthPlane::MetricConfidence);e.inputs["source_provenance"]=prototype.buffer(DaxaDepthPlane::Provenance);e.inputs["source_ids"]=prototype.buffer(DaxaDepthPlane::SourceIds);e.inputs["source_ages"]=prototype.buffer(DaxaDepthPlane::AgeNs);e.inputs["source_rgb"]=prototype.image(DaxaPreprocessPlane::RectifiedLeft).buffer();e.inputs["rectified_right_rgb"]=prototype.image(DaxaPreprocessPlane::RectifiedRight).buffer();e.inputs["original_left_rgb"]=prototype.image(DaxaPreprocessPlane::OriginalLeft).buffer();e.inputs["original_right_rgb"]=prototype.image(DaxaPreprocessPlane::OriginalRight).buffer();
    auto recorder=d.create_command_recorder({.name="one-time calibrated rays and preview map"});e.upload(recorder,"source_rays",e.cache.rays.data(),e.cache.rays.size()*sizeof(Ray3));e.upload(recorder,"preview_indices",e.cache.preview_indices.data(),e.cache.preview_indices.size()*4);recorder.pipeline_barrier({.src_access=daxa::AccessConsts::TRANSFER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ});auto commands=recorder.complete_current_commands();d.submit_commands({.command_lists={&commands,1}});d.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=d.latest_queue_submit_index(daxa::QUEUE_MAIN),.timeout=30'000'000'000ull});e.setup_slots(e.cache.effective_view,count);
}
#endif
size_t DaxaView::collect(){auto& e=*impl_;e.require_thread();return e.collect_at(e.context->device.oldest_pending_submit_index());}
bool DaxaView::can_project()const{auto& e=*impl_;e.require_thread();auto pending=e.context->device.oldest_pending_submit_index();e.collect_at(pending);for(auto const& slot:e.slots)if(slot.use_count()==1&&(!slot->submit||slot->submit<pending))return true;return false;}
#ifdef SF_DAXA_VIEW_HAS_DEPTH
DaxaView::Frame DaxaView::project(DaxaDepthPipeline::Frame const& input){
    auto& e=*impl_;e.require_thread();require(e.typed,"typed project requires calibrated shared constructor");e.validate_frame(input);auto pending=e.context->device.oldest_pending_submit_index();e.collect_at(pending);
    for(auto& s:e.slots)if(s.use_count()==1&&(!s->submit||s->submit<pending)){e.bind(*s,input);s->generation=e.next_generation++;s->visited=s->finished=0;e.context->device.collect_garbage();try{s->graph->execute({});}catch(...){s->submit=e.context->device.latest_queue_submit_index(daxa::QUEUE_MAIN);throw;}s->submit=e.context->device.latest_queue_submit_index(daxa::QUEUE_MAIN);require(s->visited==8&&s->finished==8,"view callback count changed");return Frame(impl_,s);}throw std::runtime_error("view output ring exhausted before projection");
}
#endif
DaxaView::Frame DaxaView::project(){
    auto& e=*impl_;e.require_thread();require(!e.typed,"calibrated resident view requires explicit DepthFrame");auto pending=e.context->device.oldest_pending_submit_index();
    for(auto& s:e.slots)if(s.use_count()==1&&(!s->submit||s->submit<pending)){s->generation=e.next_generation++;s->visited=s->finished=0;e.context->device.collect_garbage();try{s->graph->execute({});}catch(...){s->submit=e.context->device.latest_queue_submit_index(daxa::QUEUE_MAIN);throw;}s->submit=e.context->device.latest_queue_submit_index(daxa::QUEUE_MAIN);require(s->visited==8&&s->finished==8,"view callback count changed");return Frame(impl_,s);}throw std::runtime_error("Daxa output ring exhausted: retained consumer or pending completion");
}
void DaxaView::Frame::wait()const{validate();auto& d=owner_->context->device;d.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=slot_->submit,.timeout=30'000'000'000ull});d.collect_garbage();}
uint64_t DaxaView::Frame::generation()const{validate();return generation_;}
daxa::BufferId DaxaView::Frame::buffer(std::string const& name)const{validate();return slot_->buffers.at(name);}
uint64_t DaxaView::Frame::start_tick()const{validate();require(slot_->submit<owner_->context->device.oldest_pending_submit_index(),"view query requires completion");std::array<uint64_t,4> values{};require(daxa_timeline_query_pool_query_results(slot_->queries.get(),0,2,values.data())==DAXA_RESULT_SUCCESS&&values[1]&&values[3],"view timestamps unavailable");return values[0];}
uint64_t DaxaView::Frame::end_tick()const{validate();require(slot_->submit<owner_->context->device.oldest_pending_submit_index(),"view query requires completion");std::array<uint64_t,4> values{};require(daxa_timeline_query_pool_query_results(slot_->queries.get(),0,2,values.data())==DAXA_RESULT_SUCCESS&&values[1]&&values[3],"view timestamps unavailable");return values[2];}
double DaxaView::Frame::gpu_ms()const{validate();auto mask=owner_->timestamp_bits==64?UINT64_MAX:(uint64_t(1)<<owner_->timestamp_bits)-1;return double((end_tick()-start_tick())&mask)*owner_->timestamp_period/1e6;}
GeometryReadback DaxaView::readback(Frame const& frame)const{
    require(frame.owner_==impl_&&frame.slot_,"foreign or empty Daxa frame context");frame.wait();auto& e=*impl_;auto& d=e.context->device;auto& slot=*frame.slot_;
    const size_t m=size_t(e.config.target_width)*e.config.target_height;
    // One explicit diagnostic allocation/copy outside the measured compositor.
    const size_t size=m*(9*4+2*8);
    daxa::BufferInfo info{.size=size,.name="diagnostic output readback"};
    auto memory=d.create_memory({.requirements=e.context->coherent_readback_requirements(info),.flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});
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
std::string DaxaView::device_report()const{return impl_->context->report();}
uint64_t DaxaView::allocated_bytes()const{return impl_->bytes;}
unsigned DaxaView::validation_errors()const{return impl_->context->validation_errors.load();}
}

namespace stereoforge { std::shared_ptr<DaxaView::Diagnostics const> DaxaView::diagnostics()const{return impl_->diagnostics;} }

namespace stereoforge {
std::array<uint32_t,4> DaxaView::inspect(Frame const& frame)const {
    require(frame.owner_==impl_&&frame.slot_,"foreign or empty Daxa consumer context");
    frame.wait();auto& e=*impl_;auto& slot=*frame.slot_;auto& d=e.context->device;
    daxa::BufferInfo info{.size=16,.name="downstream counters host"};
    auto memory=d.create_memory({.requirements=e.context->coherent_readback_requirements(info),.flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});
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
uint64_t DaxaView::Frame::context_id()const{generation();return reinterpret_cast<uintptr_t>(owner_->context->device.get());}
int64_t DaxaView::Frame::timestamp_ns()const{generation();return slot_->timestamp;}
std::string DaxaView::Frame::calibration_id()const{generation();return owner_->calibration;}
GeometryConfig DaxaView::Frame::config()const{generation();return owner_->config;}
}

namespace stereoforge { std::string DaxaView::shader_directory(){return SF_DAXA_VIEW_SPIRV_DIRECTORY;} }

namespace stereoforge {
CaptureLease DaxaView::Frame::source()const{validate();return slot_->source;}
DepthConvention DaxaView::Frame::depth_convention()const{validate();return owner_->virtual_view.projection==Projection::Perspective?DepthConvention::AxialZ:DepthConvention::RayRange;}
std::string DaxaView::depth_shader_identity(){return std::string(SF_DAXA_VIEW_DEPTH_SHA256)+":"+SF_DAXA_VIEW_PROJECTION_SHA256;}
}
