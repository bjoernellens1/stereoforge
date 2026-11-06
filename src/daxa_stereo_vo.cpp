#include <stereoforge/daxa_stereo_vo.hpp>
#include <stereoforge/daxa_preprocess.hpp>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>

namespace stereoforge {
namespace {
void require(bool value,char const* message) {if(!value)throw std::invalid_argument(message);}
struct Parameters {
    uint64_t current_image=0,previous_image=0,depth=0,validity=0,provenance=0,previous_features=0,current_features=0,tracks=0,source_ids=0,ages=0;
    uint32_t width=0,height=0,columns=0,rows=0,has_previous=0;
    float fx=0,fy=0,cx=0,cy=0;
    std::array<float,9> rig_from_rect{};
    std::array<uint32_t,2> source_id{};
};
struct Feature {float u,v,x,y,z;uint32_t valid;};
struct Track {float px,py,pz,cx,cy,cz,pu,pv,cu,cv;uint32_t valid,pad;};
static_assert(sizeof(Parameters)==160&&offsetof(Parameters,rig_from_rect)==116&&sizeof(Feature)==24&&sizeof(Track)==48);
std::vector<uint32_t> shader(char const* entry) {
    std::ifstream file(std::filesystem::path(SF_DAXA_VO_SPIRV_DIRECTORY)/(std::string("daxa-vo-")+entry+".spv"),std::ios::binary|std::ios::ate);
    require(bool(file),"missing offline VO shader");auto size=file.tellg();require(size>0&&size%4==0,"invalid VO SPIRV size");
    std::vector<uint32_t> words(std::size_t(size)/4);file.seekg(0);file.read(reinterpret_cast<char*>(words.data()),size);require(bool(file),"VO SPIRV read failed");return words;
}
}
struct DaxaStereoVo::Impl {
    std::shared_ptr<DaxaFoundation> context;
    std::shared_ptr<DaxaRectification const> maps;
    DaxaStereoVoConfig config;
    Parameters constants;
    std::vector<daxa::BufferId> owned;
    std::vector<daxa::MemoryBlock> memory;
    daxa::BufferId previous_image{},previous_features{},current_features{},tracks{},parameters{},readback{};
    std::array<daxa::ComputePipeline,2> pipelines;
    CaptureLease previous_source;
    DaxaObservationFrame pending_source;
    uint64_t ticket=0;
    bool poisoned=false;
    bool unresolved_mixed_motion=false;
    std::thread::id thread=std::this_thread::get_id();
    ~Impl() noexcept {
        try {wait();for(auto b:owned)context->device.destroy_buffer(b);context->device.collect_garbage();}
        catch(...) {std::terminate();}
    }
    void wait() {
        if(ticket)context->device.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=ticket,.timeout=30'000'000'000ull});
        pending_source={};ticket=0;
    }
    void require_thread() const {require(thread==std::this_thread::get_id(),"VO requires its shared device submission thread");}
    daxa::BufferId allocate(std::size_t size,bool host=false) {
        auto& d=context->device;daxa::BufferInfo info{.size=size,.name="bounded stereo VO"};daxa::BufferId id;
        if(host) {
            auto requirements=context->coherent_readback_requirements(info);
            memory.push_back(d.create_memory({.requirements=requirements,.flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM}));
            id=d.create_buffer_from_memory_block({.buffer_info=info,.memory_block=memory.back()});
        }else id=d.create_buffer(info);
        owned.push_back(id);return id;
    }
};
DaxaStereoVo::DaxaStereoVo(std::shared_ptr<DaxaFoundation> context,DaxaObservationFrame const& prototype,DaxaStereoVoConfig config) {
    require(context&&context->device.get()&&bool(prototype),"VO needs resident observation and context");
    require(prototype.context_id()==reinterpret_cast<uintptr_t>(context->device.get()),"foreign VO context");
    config.motion.validate();config.guard.validate();require(config.columns>0&&config.rows>0&&uint64_t(config.columns)*config.rows<=config.motion.max_tracks,"invalid bounded VO grid");
    require(config.max_ingress_gap_ns>0&&config.max_ingress_gap_ns<=200000000,"VO history age must be at most 200ms");
    auto maps=prototype.calibration();require(maps&&maps->geometry.has_value(),"VO requires native metric rig geometry");
    require(prototype.width()>=config.columns*8&&prototype.height()>=config.rows*8,"VO grid too dense for image support");
    auto e=std::make_unique<Impl>();e->context=std::move(context);e->maps=std::move(maps);e->config=config;
    auto& d=e->context->device;auto& p=e->constants;
    p.width=prototype.width();p.height=prototype.height();p.columns=config.columns;p.rows=config.rows;
    auto const& g=*e->maps->geometry;auto const& k=g.rectified.eyes[0];
    p.fx=float(k.fx);p.fy=float(k.fy);p.cx=float(k.cx);p.cy=float(k.cy);
    for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)p.rig_from_rect[i*3+j]=float(g.R1[j*3+i]);
    auto n=std::size_t(config.columns)*config.rows;
    e->previous_image=e->allocate(std::size_t(p.width)*p.height*4);
    e->previous_features=e->allocate(n*sizeof(Feature));e->current_features=e->allocate(n*sizeof(Feature));
    e->tracks=e->allocate(n*sizeof(Track));e->readback=e->allocate(n*sizeof(Track),true);e->parameters=e->allocate(sizeof(Parameters),true);
    for(unsigned i=0;i<2;++i) {
        auto words=shader(i==0?"detect":"track");
        e->pipelines[i]=d.create_compute_pipeline({.shader_info={.byte_code=words.data(),.byte_code_size=uint32_t(words.size())},.push_constant_size=8,.name="native stereo VO"});
    }
    impl_=std::move(e);
}
DaxaStereoVo::~DaxaStereoVo()=default;
void DaxaStereoVo::reset() {auto& e=*impl_;e.require_thread();require(!e.poisoned,"VO submission failed; reconstruct estimator");e.wait();e.previous_source={};e.unresolved_mixed_motion=false;}
DaxaStereoVoResult DaxaStereoVo::process(DaxaObservationFrame const& input) {
    auto& e=*impl_;e.require_thread();require(!e.poisoned,"VO submission failed; reconstruct estimator");
    require(bool(input)&&input.context_id()==reinterpret_cast<uintptr_t>(e.context->device.get()),"foreign VO observation context");
    require(input.calibration()==e.maps&&input.width()==e.constants.width&&input.height()==e.constants.height,"VO calibration/grid mismatch");
    auto src=input.source();require(bool(src),"VO observation has no exact source");
    auto image=input.image(DaxaObservationImage::RectifiedLeft);require(image.width()==input.width()&&image.height()==input.height(),"VO RGB registration mismatch");
    auto identity_for=[&](CaptureFrame const& frame) {
        require(frame.calibration_identity==e.maps->original_identity,"VO capture/calibration identity mismatch");
        StereoFrameIdentity id{frame.frame_id,frame.camera_timestamp_ns,frame.host_ingress_timestamp_ns,
            frame.device_identity,e.maps->original_identity,e.maps->rectified_identity,e.config.rig_session_identity,
            frame.camera_clock_domain,frame.host_clock_domain,{}};
        id.validate();return id;
    };
    DaxaStereoVoResult result;result.source=identity_for(*src);
    if(e.previous_source) {
        auto const& prev=*e.previous_source;
        bool same=prev.device_identity==src->device_identity&&prev.host_clock_domain==src->host_clock_domain&&prev.camera_clock_domain==src->camera_clock_domain;
        require(!same||(src->frame_id>prev.frame_id&&src->host_ingress_timestamp_ns>prev.host_ingress_timestamp_ns),"duplicate/late VO source");
        bool acquisition_reversed=src->camera_clock_domain!="unknown"&&src->camera_clock_domain!="unspecified"&&!src->camera_clock_domain.empty()&&
            prev.camera_timestamp_ns&&src->camera_timestamp_ns<=prev.camera_timestamp_ns;
        if(!same||acquisition_reversed||src->host_ingress_timestamp_ns-prev.host_ingress_timestamp_ns>e.config.max_ingress_gap_ns)e.previous_source={};
        else result.previous_source=identity_for(prev);
    }
    auto& d=e.context->device;auto p=e.constants;
    p.current_image=d.device_address(image.buffer()).value();p.previous_image=d.device_address(e.previous_image).value();
    p.depth=d.device_address(input.buffer(DaxaObservationPlane::Depth)).value();p.validity=d.device_address(input.buffer(DaxaObservationPlane::Validity)).value();p.provenance=d.device_address(input.buffer(DaxaObservationPlane::Provenance)).value();
    p.previous_features=d.device_address(e.previous_features).value();p.current_features=d.device_address(e.current_features).value();p.tracks=d.device_address(e.tracks).value();p.has_previous=bool(e.previous_source);
    p.source_ids=d.device_address(input.buffer(DaxaObservationPlane::SourceIds)).value();p.ages=d.device_address(input.buffer(DaxaObservationPlane::AgeNs)).value();
    p.source_id={uint32_t(src->frame_id),uint32_t(src->frame_id>>32)};
    std::memcpy(d.buffer_host_address(e.parameters).value(),&p,sizeof(p));
    auto n=std::size_t(p.columns)*p.rows;auto before=d.latest_queue_submit_index(daxa::QUEUE_MAIN);
    e.pending_source=input;
    try {
        auto recorder=d.create_command_recorder({.name="bounded GPU stereo VO"});
        recorder.pipeline_barrier({.src_access=daxa::AccessConsts::HOST_WRITE|daxa::AccessConsts::COMPUTE_SHADER_WRITE|daxa::AccessConsts::TRANSFER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE});
        for(auto const& pipeline:e.pipelines) {
            recorder.set_pipeline(pipeline);recorder.push_constant(d.device_address(e.parameters).value());recorder.dispatch({uint32_t((n+63)/64),1,1});
        }
        recorder.pipeline_barrier({.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE|daxa::AccessConsts::COMPUTE_SHADER_READ,.dst_access=daxa::AccessConsts::TRANSFER_READ|daxa::AccessConsts::TRANSFER_WRITE});
        recorder.copy_buffer_to_buffer({.src_buffer=e.tracks,.dst_buffer=e.readback,.size=n*sizeof(Track)});
        recorder.copy_buffer_to_buffer({.src_buffer=e.current_features,.dst_buffer=e.previous_features,.size=n*sizeof(Feature)});
        recorder.copy_buffer_to_buffer({.src_buffer=image.buffer(),.dst_buffer=e.previous_image,.size=std::size_t(p.width)*p.height*4});
        recorder.pipeline_barrier({.src_access=daxa::AccessConsts::TRANSFER_WRITE,.dst_access=daxa::AccessConsts::HOST_READ});
        auto command=recorder.complete_current_commands();e.ticket=d.submit_commands({.command_lists={&command,1}});e.wait();
    }catch(...) {
        // Retain input and own buffers through any partially successful submit.
        e.poisoned=true;try {e.ticket=std::max(e.ticket,d.latest_queue_submit_index(daxa::QUEUE_MAIN));if(e.ticket<=before)e.ticket=0;}catch(...){std::terminate();}
        throw;
    }
    // The copied image/features now belong to this source, even if a later
    // bounded CPU allocation or solve fails. Never pair them with old metadata.
    e.previous_source=src;
    std::vector<StereoMetricTrack> points;points.reserve(n);
    std::vector<StereoImageTrack> pixels;pixels.reserve(n);
    std::vector<Track> records(n);
    std::memcpy(records.data(),d.buffer_host_address(e.readback).value(),n*sizeof(Track));
    for(std::size_t i=0;i<n;++i)if(records[i].valid) {
        auto const& t=records[i];points.push_back({{t.px,t.py,t.pz},{t.cx,t.cy,t.cz},.005});
        pixels.push_back({{double(t.pu)/double(p.width-1),double(t.pv)/double(p.height-1)},{double(t.cu)/double(p.width-1),double(t.cv)/double(p.height-1)}});
    }
    result.tracked_features=points.size();result.motion=estimate_stereo_motion(points,e.config.motion);
    result.guard=guard_stereo_motion(points,pixels,result.motion,e.config.motion,e.config.guard);
    if(e.config.guard.enabled) {
        if(result.guard.status==StereoMotionGuardStatus::MultipleMotions)e.unresolved_mixed_motion=true;
        else if(result.guard.status==StereoMotionGuardStatus::Accepted&&e.unresolved_mixed_motion) {
            // A skipped/degenerate secondary search cannot clear known mixed
            // motion. Keep fresh stereo until one model explains every track.
            // This remains an unqualified body-motion fit, not a world-motion proof.
            if(result.motion.inliers.size()==points.size())e.unresolved_mixed_motion=false;
            else result.guard.status=StereoMotionGuardStatus::UnresolvedMixedMotion;
        }
    }
    if(result.motion.status==StereoMotionStatus::Tracked&&e.config.guard.enabled&&result.guard.status!=StereoMotionGuardStatus::Accepted) {
        result.motion.status=(result.guard.status==StereoMotionGuardStatus::MultipleMotions||result.guard.status==StereoMotionGuardStatus::UnresolvedMixedMotion)?StereoMotionStatus::AmbiguousMotion:
            (result.guard.status==StereoMotionGuardStatus::InsufficientImageSupport?StereoMotionStatus::InsufficientImageSupport:StereoMotionStatus::InvalidInput);
        result.motion.current_from_previous={};result.motion.covariance={};result.motion.inliers.clear();
    }
    return result;
}
}
