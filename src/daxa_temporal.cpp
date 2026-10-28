#include <stereoforge/detail/daxa_temporal_state.hpp>
#include <daxa/utils/task_graph.hpp>
#include <daxa/c/device.h>
#include <daxa/c/types.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>
namespace stereoforge {
namespace {
using detail::temporal_require;using detail::TemporalMatrix;using detail::temporal_multiply;using detail::temporal_inverse;
void identity(std::string_view s){temporal_require(!s.empty()&&s.size()<=65536&&s!="unknown"&&s!="unspecified","temporal requires known bounded device/clock identity");}
constexpr std::array<unsigned,11> bytes{4,4,4,4,8,8,32,4,8,4,4};
struct Parameters {
    std::array<uint64_t,6> fresh{};std::array<uint64_t,11> history{},output{};std::array<uint64_t,2> support{};
    uint64_t zbuffer=0,owners=0,counters=0;
    uint32_t width=0,height=0,warp=0,capacity=0,max_reuses=0;
    std::array<uint32_t,2> max_age{},elapsed{};float tolerance=0,confidence_floor=0;
    std::array<float,8> intrinsics{};std::array<float,32> transforms{};
};
static_assert(offsetof(Parameters,max_age)==284&&offsetof(Parameters,intrinsics)==308&&offsetof(Parameters,transforms)==340&&sizeof(Parameters)==472);
static_assert(sizeof(DaxaTemporalCounters)==40);
struct Push{uint64_t parameters;};
std::vector<uint32_t> shader(char const* entry){std::ifstream f(std::filesystem::path(SF_DAXA_TEMPORAL_SPIRV_DIRECTORY)/(std::string("daxa-temporal-")+entry+".spv"),std::ios::binary|std::ios::ate);temporal_require(bool(f),"missing offline temporal shader");auto size=f.tellg();temporal_require(size>0&&size%4==0,"invalid temporal SPIRV");std::vector<uint32_t> out(size_t(size)/4);f.seekg(0);f.read(reinterpret_cast<char*>(out.data()),size);temporal_require(bool(f),"temporal SPIRV read failed");return out;}
std::array<uint32_t,2> split(uint64_t v){return {uint32_t(v),uint32_t(v>>32)};}
}
void DaxaTemporalConfig::validate()const{temporal_require(slots>=2&&slots<=3,"temporal slots must be2..3");temporal_require(max_contributors>=1&&max_contributors<=4,"temporal contributor capacity must be1..4");temporal_require(max_reuses>0&&max_age_ns>=0,"invalid temporal reuse/age bound");temporal_require(std::isfinite(relative_depth_tolerance)&&relative_depth_tolerance>=0,"invalid temporal depth tolerance");temporal_require(std::isfinite(confidence_floor)&&confidence_floor>=0&&confidence_floor<=1,"invalid temporal confidence floor");}
void DaxaTimedRigPose::validate(CaptureFrame const& c)const{
    temporal_require(timestamp_ns>=0&&c.frame_id<=uint64_t(INT64_MAX)&&c.host_ingress_timestamp_ns<=uint64_t(INT64_MAX),"temporal timestamp/source ID outside signed64 domain");identity(c.device_identity);identity(c.host_clock_domain);identity(clock_domain);
    temporal_require(clock_domain==c.host_clock_domain&&timestamp_ns==int64_t(c.host_ingress_timestamp_ns),"temporal pose/capture clock mismatch");
    auto const& a=world_from_rig;for(auto v:a)temporal_require(std::isfinite(v),"nonfinite temporal pose");temporal_require(a[12]==0&&a[13]==0&&a[14]==0&&a[15]==1,"malformed temporal homogeneous pose");
    for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j){double dot=0;for(unsigned k=0;k<3;++k)dot+=a[k*4+i]*a[k*4+j];temporal_require(std::abs(dot-double(i==j))<=1e-10,"nonrigid temporal pose");}
    auto det=a[0]*(a[5]*a[10]-a[6]*a[9])-a[1]*(a[4]*a[10]-a[6]*a[8])+a[2]*(a[4]*a[9]-a[5]*a[8]);temporal_require(std::abs(det-1)<=1e-10,"reflected temporal pose");
}
detail::TemporalSequence::Admission detail::TemporalSequence::prepare(CaptureFrame const& c,std::optional<DaxaTimedRigPose> const& p,int64_t max_age)const{
    identity(c.device_identity);identity(c.host_clock_domain);temporal_require(c.frame_id<=uint64_t(INT64_MAX)&&c.host_ingress_timestamp_ns<=uint64_t(INT64_MAX),"temporal timestamp/source ID outside signed64 domain");if(p)p->validate(c);
    Admission a;a.epoch=!accepted||device!=c.device_identity||clock!=c.host_clock_domain;
    if(!a.epoch){temporal_require(c.frame_id>id&&c.host_ingress_timestamp_ns>time,"duplicate/late temporal capture");a.elapsed=c.host_ingress_timestamp_ns-time;}
    a.warp=!a.epoch&&has_pose&&p&&a.elapsed<=uint64_t(max_age);return a;
}
struct DaxaTemporal::Slot {
    std::array<daxa::BufferId,11> outputs{};daxa::BufferId parameters{},zbuffer{},owners{},counters{};
    std::array<daxa::ExternalTaskBuffer,6> fresh_nodes;std::array<daxa::ExternalTaskBuffer,11> history_nodes,output_nodes;
    std::array<daxa::ExternalTaskBuffer,2> support_nodes;std::array<daxa::ExternalTaskBuffer,4> scratch_nodes;
    std::unique_ptr<daxa::TaskGraph> graph;daxa::TimelineQueryPool queries;DaxaDepthPipeline::Frame input;
    uint64_t generation=0,last_write=0,last_read=0;
};
struct DaxaTemporal::Impl {
    std::shared_ptr<DaxaFoundation> context;std::shared_ptr<const DaxaRectification> maps;DaxaTemporalConfig config;
    std::vector<std::shared_ptr<Slot>> slots;std::shared_ptr<Slot> history;
    std::vector<daxa::BufferId> owned;std::vector<daxa::MemoryBlock> memory;std::array<daxa::ComputePipeline,4> pipelines;
    detail::TemporalSequence sequence;std::array<TemporalMatrix,2> eye_to_rig,rig_to_eye;Parameters constants;
    uint64_t allocated=0,next_generation=1,retired=0;uint32_t timestamp_bits=0;float timestamp_period=0;
    const std::thread::id thread=std::this_thread::get_id();
    void require_thread()const{temporal_require(thread==std::this_thread::get_id(),"temporal requires one shared submission thread");}
    bool complete(Slot const& s,uint64_t oldest)const{return detail::temporal_tickets_complete(s.last_write,s.last_read,oldest);}
    long internal(Slot const* s)const{return detail::temporal_internal_references(history.get()==s);}
    [[noreturn]] void fail_backpressure(uint64_t boundary)const{
        // Cold failure path only: retain the exact admission snapshot, not a
        // later completion poll that could conceal a boundary equality.
        std::ostringstream message;message<<"temporal backpressure: all nonhistory output slots retained/inflight; completion_boundary="<<boundary<<" latest_main_submit="<<context->device.latest_queue_submit_index(daxa::QUEUE_MAIN);
        for(size_t i=0;i<slots.size();++i){auto const& s=slots[i];message<<" slot["<<i<<"]{history="<<(s==history)<<",refs="<<s.use_count()<<",generation="<<s->generation<<",last_write="<<s->last_write<<",last_read="<<s->last_read<<",input="<<bool(s->input)<<"}";}
        throw std::runtime_error(message.str());
    }
    size_t collect_at(uint64_t oldest){size_t n=0;for(auto& s:slots)if(s->input&&s.use_count()==internal(s.get())&&complete(*s,oldest)){s->input={};++n;++retired;}return n;}
    daxa::BufferId allocate(size_t size,std::string const& name,bool host=false){auto& d=context->device;daxa::BufferInfo info{.size=size,.name=name};daxa::BufferId out;if(host){auto m=d.create_memory({.requirements=context->coherent_readback_requirements(info),.flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});out=d.create_buffer_from_memory_block({.buffer_info=info,.memory_block=m});memory.push_back(std::move(m));}else out=d.create_buffer(info);owned.push_back(out);allocated+=size;return out;}
    ~Impl(){if(!context||!context->device.get())return;auto& d=context->device;d.wait_idle();history.reset();slots.clear();pipelines={};for(auto b:owned)if(d.is_id_valid(b))d.destroy_buffer(b);d.collect_garbage();memory.clear();}
    Impl(std::shared_ptr<DaxaFoundation> c,DaxaDepthPipeline::Frame const& prototype,DaxaTemporalConfig cfg):context(std::move(c)),config(cfg){
        cfg.validate();temporal_require(context&&context->device.get(),"temporal requires foundation");auto observation=prototype.observations();temporal_require(observation.context()==context,"temporal prototype context mismatch");maps=observation.map_owner();maps->metric_calibration();temporal_require(maps->geometry.has_value(),"temporal requires calibrated metric geometry");auto const& geometry=*maps->geometry;geometry.validate_map_association({maps->maps[0],maps->maps[1],maps->maps[2],maps->maps[3]});
        uint64_t count=uint64_t(maps->width)*maps->height*2;temporal_require(count&&count<=UINT32_MAX,"temporal index/counter overflow");
        auto& d=context->device;auto physical=daxa_dvc_get_vk_physical_device(d.get());VkPhysicalDeviceProperties properties{};vkGetPhysicalDeviceProperties(physical,&properties);timestamp_period=properties.limits.timestampPeriod;temporal_require(properties.limits.maxComputeWorkGroupSize[0]>=64&&properties.limits.maxComputeWorkGroupInvocations>=64&&(count+63)/64<=properties.limits.maxComputeWorkGroupCount[0],"temporal dispatch unsupported");for(auto b:bytes)temporal_require(count*b<=properties.limits.maxStorageBufferRange,"temporal plane exceeds storage range");
        uint32_t family=0,families_count=0;temporal_require(daxa_dvc_get_vk_queue(d.get(),{DAXA_QUEUE_TYPE_MAIN,0},nullptr,&family)==DAXA_RESULT_SUCCESS,"missing temporal main queue");vkGetPhysicalDeviceQueueFamilyProperties(physical,&families_count,nullptr);std::vector<VkQueueFamilyProperties> families(families_count);vkGetPhysicalDeviceQueueFamilyProperties(physical,&families_count,families.data());temporal_require(family<families_count&&families[family].timestampValidBits,"temporal timestamps unavailable");timestamp_bits=families[family].timestampValidBits;
        constants.width=maps->width;constants.height=maps->height;constants.capacity=cfg.max_contributors;constants.max_reuses=cfg.max_reuses;constants.max_age=split(uint64_t(cfg.max_age_ns));constants.tolerance=cfg.relative_depth_tolerance;constants.confidence_floor=cfg.confidence_floor;
        for(unsigned eye=0;eye<2;++eye){auto const& k=geometry.rectified.eyes[eye];auto intrinsic=detail::temporal_pinhole_f32(k);std::copy(intrinsic.begin(),intrinsic.end(),constants.intrinsics.begin()+eye*4);TemporalMatrix rotation{};rotation[15]=1;auto const& r=eye?geometry.R2:geometry.R1;for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)rotation[i*4+j]=r[j*3+i];eye_to_rig[eye]=eye?temporal_multiply(geometry.original.right_to_left,rotation):rotation;rig_to_eye[eye]=temporal_inverse(eye_to_rig[eye]);}
        for(auto v:constants.intrinsics)temporal_require(std::isfinite(v),"temporal pinhole outside float32 domain");
        constexpr std::array<char const*,4> entries{"clear","splat_z","splat_owner","gather_fuse"};for(unsigned i=0;i<4;++i){auto code=shader(entries[i]);pipelines[i]=d.create_compute_pipeline({.shader_info={.byte_code=code.data(),.byte_code_size=uint32_t(code.size())},.push_constant_size=sizeof(Push),.name=entries[i]});}
        slots.reserve(cfg.slots);owned.reserve(cfg.slots*15);memory.reserve(cfg.slots);
        for(unsigned index=0;index<cfg.slots;++index){auto s=std::make_shared<Slot>();auto prefix="temporal_"+std::to_string(index)+"_";auto params=constants;
            for(unsigned p=0;p<11;++p){s->outputs[p]=allocate(count*bytes[p],prefix+"output_"+std::to_string(p));params.output[p]=d.device_address(s->outputs[p]).value();s->output_nodes[p]=daxa::ExternalTaskBuffer({.buffer=s->outputs[p],.name=prefix+"out_"+std::to_string(p)});s->history_nodes[p]=daxa::ExternalTaskBuffer({.buffer=s->outputs[p],.name=prefix+"history_"+std::to_string(p)});params.history[p]=params.output[p];}
            for(unsigned p=0;p<6;++p){auto b=observation.buffer(DaxaObservationPlane(p));s->fresh_nodes[p]=daxa::ExternalTaskBuffer({.buffer=b,.name=prefix+"fresh_"+std::to_string(p)});params.fresh[p]=d.device_address(b).value();}
            for(unsigned p=0;p<2;++p){auto b=prototype.image(p?DaxaPreprocessPlane::RightSupport:DaxaPreprocessPlane::LeftSupport).buffer();s->support_nodes[p]=daxa::ExternalTaskBuffer({.buffer=b,.name=prefix+"support_"+std::to_string(p)});params.support[p]=d.device_address(b).value();}
            s->zbuffer=allocate(count*4,prefix+"z");s->owners=allocate(count*4,prefix+"owner");s->counters=allocate(sizeof(DaxaTemporalCounters),prefix+"counters");s->parameters=allocate(sizeof(Parameters),prefix+"parameters",true);params.zbuffer=d.device_address(s->zbuffer).value();params.owners=d.device_address(s->owners).value();params.counters=d.device_address(s->counters).value();std::memcpy(d.buffer_host_address(s->parameters).value(),&params,sizeof(params));
            std::array<daxa::BufferId,4> scratch{s->parameters,s->zbuffer,s->owners,s->counters};for(unsigned p=0;p<4;++p)s->scratch_nodes[p]=daxa::ExternalTaskBuffer({.buffer=scratch[p],.name=prefix+"scratch_"+std::to_string(p)});
            s->queries=d.create_timeline_query_pool({.query_count=8,.name=prefix+"queries"});s->graph=std::make_unique<daxa::TaskGraph>(daxa::TaskGraphInfo{.device=d,.name=prefix+"graph"});auto& graph=*s->graph;
            std::array<daxa::TaskBufferView,6> fresh;std::array<daxa::TaskBufferView,11> old,out;std::array<daxa::TaskBufferView,2> support;std::array<daxa::TaskBufferView,4> tmp;
            for(unsigned p=0;p<6;++p)fresh[p]=graph.register_buffer(s->fresh_nodes[p]);for(unsigned p=0;p<11;++p){old[p]=graph.register_buffer(s->history_nodes[p]);out[p]=graph.register_buffer(s->output_nodes[p]);}for(unsigned p=0;p<2;++p)support[p]=graph.register_buffer(s->support_nodes[p]);for(unsigned p=0;p<4;++p)tmp[p]=graph.register_buffer(s->scratch_nodes[p]);
            auto* raw=s.get();auto address=d.device_address(s->parameters).value();for(unsigned pass=0;pass<4;++pass){auto task=daxa::InlineTask::Compute(entries[pass]);task.reads(tmp[0]);if(pass==0){task.writes(tmp[1]).writes(tmp[2]).writes(tmp[3]);}else{for(auto v:old)task.reads(v);for(auto v:support)task.reads(v);task.reads_writes(tmp[3]);if(pass==1)task.reads_writes(tmp[1]);if(pass==2)task.reads(tmp[1]).reads_writes(tmp[2]);if(pass==3){task.reads(tmp[1]).reads(tmp[2]);for(auto v:fresh)task.reads(v);for(auto v:out)task.writes(v);}}
                graph.add_task(task.executes([this,raw,address,pass,count](daxa::TaskInterface ti){if(pass==0){ti.recorder.pipeline_barrier({.src_access=daxa::AccessConsts::HOST_WRITE|daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE});ti.recorder.reset_timestamps({.query_pool=raw->queries,.start_index=0,.count=8});}ti.recorder.write_timestamp({.query_pool=raw->queries,.pipeline_stage=daxa::PipelineStageFlagBits::COMPUTE_SHADER,.query_index=2*pass});ti.recorder.set_pipeline(pipelines[pass]);ti.recorder.push_constant(Push{address});ti.recorder.dispatch({uint32_t((std::max<uint64_t>(count,10)+63)/64),1,1});ti.recorder.write_timestamp({.query_pool=raw->queries,.pipeline_stage=daxa::PipelineStageFlagBits::COMPUTE_SHADER,.query_index=2*pass+1});}));}
            graph.submit({});graph.complete({});slots.push_back(std::move(s));
        }
    }
};
DaxaTemporal::DaxaTemporal(std::shared_ptr<DaxaFoundation> context,DaxaDepthPipeline::Frame const& prototype,DaxaTemporalConfig config):impl_(std::make_shared<Impl>(std::move(context),prototype,config)){}
DaxaTemporal::Frame::Frame(std::shared_ptr<Impl> owner,std::shared_ptr<Slot> slot):owner_(std::move(owner)),slot_(std::move(slot)),generation_(slot_->generation){}
DaxaTemporal::Frame& DaxaTemporal::Frame::operator=(Frame other)noexcept{owner_.swap(other.owner_);slot_.swap(other.slot_);std::swap(generation_,other.generation_);return *this;}
void DaxaTemporal::Frame::validate()const{temporal_require(owner_&&slot_&&generation_==slot_->generation&&slot_->input,"empty/stale temporal frame");}
DaxaTemporal::Frame DaxaTemporal::process(DaxaDepthPipeline::Frame const& input,std::optional<DaxaTimedRigPose> const& pose){
    auto& e=*impl_;e.require_thread();auto fresh=input.observations();temporal_require(fresh.context()==e.context&&fresh.map_owner()==e.maps&&fresh.width()==e.maps->width&&fresh.height()==e.maps->height,"temporal incompatible context/calibration/grid");auto capture=input.source();auto admission=e.sequence.prepare(*capture,pose,e.config.max_age_ns);
    std::array<float,32> transforms{};
    if(admission.warp){auto relative=temporal_multiply(temporal_inverse(pose->world_from_rig),e.sequence.previous);for(unsigned eye=0;eye<2;++eye){auto m=temporal_multiply(temporal_multiply(e.rig_to_eye[eye],relative),e.eye_to_rig[eye]);for(unsigned j=0;j<16;++j){transforms[eye*16+j]=float(m[j]);temporal_require(std::isfinite(transforms[eye*16+j]),"temporal motion outside float32 domain");}}}
    auto& d=e.context->device;auto oldest=d.oldest_pending_submit_index();e.collect_at(oldest);
    for(auto& s:e.slots){if(s==e.history||s.use_count()!=1||!e.complete(*s,oldest))continue;
        auto* params=reinterpret_cast<Parameters*>(d.buffer_host_address(s->parameters).value());params->warp=admission.warp&&bool(e.history);params->elapsed=split(admission.elapsed);params->transforms=transforms;
        for(unsigned p=0;p<6;++p){auto b=fresh.buffer(DaxaObservationPlane(p));s->fresh_nodes[p].set_buffer(b);params->fresh[p]=d.device_address(b).value();}
        for(unsigned p=0;p<2;++p){auto b=input.image(p?DaxaPreprocessPlane::RightSupport:DaxaPreprocessPlane::LeftSupport).buffer();s->support_nodes[p].set_buffer(b);params->support[p]=d.device_address(b).value();}
        auto* old=params->warp?e.history.get():(e.slots[0].get()==s.get()?e.slots[1].get():e.slots[0].get());
        for(unsigned p=0;p<11;++p){s->history_nodes[p].set_buffer(old->outputs[p]);params->history[p]=d.device_address(old->outputs[p]).value();}
        s->input=input;s->generation=e.next_generation++;
        // execute may submit before throwing. Pin both candidate input and the
        // last historical read until the actual shared-queue ticket completes.
        try{s->graph->execute({});}catch(...){s->last_write=d.latest_queue_submit_index(daxa::QUEUE_MAIN);if(params->warp)e.history->last_read=std::max(e.history->last_read,s->last_write);throw;}
        s->last_write=d.latest_queue_submit_index(daxa::QUEUE_MAIN);if(params->warp)e.history->last_read=std::max(e.history->last_read,s->last_write);
        e.sequence.commit(*capture,pose);if(pose)e.history=s;else e.history.reset();return Frame(impl_,s);
    }
    e.fail_backpressure(oldest);
}
size_t DaxaTemporal::collect(){impl_->require_thread();auto& d=impl_->context->device;auto retired=impl_->collect_at(d.oldest_pending_submit_index());d.collect_garbage();return retired;}
bool DaxaTemporal::can_process()const{auto& e=*impl_;e.require_thread();auto oldest=e.context->device.oldest_pending_submit_index();for(auto const& s:e.slots)if(s!=e.history&&s.use_count()==1&&e.complete(*s,oldest))return true;return false;}
void DaxaTemporal::reset(){impl_->require_thread();impl_->history.reset();impl_->sequence.reset();}
void DaxaTemporal::Frame::wait()const{validate();owner_->context->device.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=slot_->last_write,.timeout=30'000'000'000ull});}
bool DaxaTemporal::Frame::ready()const{validate();return detail::completed_submission(slot_->last_write,owner_->context->device.oldest_pending_submit_index());}
uint32_t DaxaTemporal::Frame::width()const{validate();return owner_->maps->width;}uint32_t DaxaTemporal::Frame::height()const{validate();return owner_->maps->height;}
uint64_t DaxaTemporal::Frame::generation()const{validate();return generation_;}uint64_t DaxaTemporal::Frame::context_id()const{validate();return reinterpret_cast<uintptr_t>(owner_->context->device.get());}uint64_t DaxaTemporal::Frame::submit_ticket()const{validate();return slot_->last_write;}
DaxaObservationAncestryView DaxaTemporal::Frame::ancestry()const{validate();std::array<daxa::BufferId,5> sidecars{};std::copy(slot_->outputs.begin()+6,slot_->outputs.end(),sidecars.begin());return DaxaObservationAncestryView(size_t(owner_->maps->width)*owner_->maps->height*2,owner_->config.max_contributors,sidecars);}
DaxaObservationFrame DaxaTemporal::Frame::observations()const{validate();auto input=slot_->input.observations();std::array<daxa::BufferId,6> planes{};std::copy_n(slot_->outputs.begin(),6,planes.begin());std::array<DaxaObservationImageView,4> images{};for(unsigned i=0;i<4;++i)images[i]=input.image(DaxaObservationImage(i));return DaxaObservationFrame(owner_->context,owner_,slot_,&slot_->generation,generation_,slot_->last_write,owner_->maps,input.source(),planes,images,ancestry());}
DaxaTemporalTiming DaxaTemporal::Frame::timings()const{validate();owner_->require_thread();temporal_require(ready(),"temporal completion required before timing");std::array<uint64_t,16> results{};temporal_require(daxa_timeline_query_pool_query_results(slot_->queries.get(),0,8,results.data())==DAXA_RESULT_SUCCESS,"temporal timestamp query failed");for(unsigned i=0;i<8;++i)temporal_require(results[i*2+1],"temporal timestamp unavailable");auto mask=owner_->timestamp_bits==64?UINT64_MAX:(uint64_t{1}<<owner_->timestamp_bits)-1;auto ms=[&](unsigned a,unsigned b){return double((results[b*2]-results[a*2])&mask)*owner_->timestamp_period/1e6;};DaxaTemporalTiming out;out.clear_gpu_ms=ms(0,1);out.z_gpu_ms=ms(2,3);out.owner_gpu_ms=ms(4,5);out.fuse_gpu_ms=ms(6,7);out.total_gpu_ms=ms(0,7);out.start_tick=results[0];out.end_tick=results[14];out.timestamp_bits=owner_->timestamp_bits;out.timestamp_period=owner_->timestamp_period;return out;}
DaxaTemporalReadback DaxaTemporal::readback(Frame const& f)const{
    f.validate();impl_->require_thread();temporal_require(f.owner_==impl_,"foreign temporal diagnostic frame");f.wait();auto& d=impl_->context->device;DaxaTemporalReadback out;
    auto copy=[&](void* target,size_t size,daxa::BufferId source){daxa::BufferInfo info{.size=size,.name="temporal scoped diagnostic"};auto memory=d.create_memory({.requirements=impl_->context->coherent_readback_requirements(info),.flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});auto host=d.create_buffer_from_memory_block({.buffer_info=info,.memory_block=memory});struct Cleanup{daxa::Device& d;daxa::BufferId b;~Cleanup(){d.destroy_buffer(b);d.collect_garbage();}}cleanup{d,host};auto recorder=d.create_command_recorder({.name="temporal diagnostic transfer"});recorder.pipeline_barrier({.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::TRANSFER_READ});recorder.copy_buffer_to_buffer({.src_buffer=source,.dst_buffer=host,.size=size});recorder.pipeline_barrier({.src_access=daxa::AccessConsts::TRANSFER_WRITE,.dst_access=daxa::AccessConsts::HOST_READ});auto command=recorder.complete_current_commands();d.submit_commands({.command_lists={&command,1}});d.wait_idle();std::memcpy(target,d.buffer_host_address(host).value(),size);};
    auto plane=[&](auto& values,unsigned p){auto size=size_t(impl_->maps->width)*impl_->maps->height*2*bytes[p];values.resize(size/sizeof(typename std::remove_reference_t<decltype(values)>::value_type));copy(values.data(),size,f.slot_->outputs[p]);};
    plane(out.depth,0);plane(out.confidence,1);plane(out.validity,2);plane(out.provenance,3);plane(out.source_ids,4);plane(out.age_ns,5);plane(out.contributor_ids,6);plane(out.count,7);plane(out.oldest_age_ns,8);plane(out.reuse,9);plane(out.origin_mask,10);copy(&out.counters,sizeof(out.counters),f.slot_->counters);return out;
}
uint64_t DaxaTemporal::requested_buffer_bytes()const{return impl_->allocated;}uint64_t DaxaTemporal::completed_source_retirements()const{return impl_->retired;}
std::string DaxaTemporal::shader_directory(){return SF_DAXA_TEMPORAL_SPIRV_DIRECTORY;}std::string DaxaTemporal::shader_identity(){return std::string(SF_DAXA_TEMPORAL_WRAPPER_SHA256)+":"+SF_DAXA_TEMPORAL_OPS_SHA256;}
}
