#include <stereoforge/detail/daxa_temporal_state.hpp>
#include <stereoforge/canonical_rgb.hpp>
#include <stereoforge/daxa_stereo_history.hpp>
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
    uint64_t pair_parameters=0;std::array<uint64_t,2> current_images{};uint32_t pair_count=0;float baseline_m=0;uint32_t require_pair_support=0;
};
static_assert(offsetof(Parameters,max_age)==284&&offsetof(Parameters,intrinsics)==308&&offsetof(Parameters,transforms)==340&&sizeof(Parameters)==512);
static_assert(sizeof(DaxaTemporalCounters)==48);
struct PairParameters {std::array<uint64_t,7> buffers{};std::array<uint32_t,2> source_id{};std::array<float,32> current_to_left{};};
static_assert(sizeof(PairParameters)==192);
struct PairContext {StereoWorldPose const* pose=nullptr;std::vector<DaxaStereoHistory::Selection> const* pairs=nullptr;};
struct Push{uint64_t parameters;};
std::vector<uint32_t> shader(char const* entry){std::ifstream f(std::filesystem::path(SF_DAXA_TEMPORAL_SPIRV_DIRECTORY)/(std::string("daxa-temporal-")+entry+".spv"),std::ios::binary|std::ios::ate);temporal_require(bool(f),"missing offline temporal shader");auto size=f.tellg();temporal_require(size>0&&size%4==0,"invalid temporal SPIRV");std::vector<uint32_t> out(size_t(size)/4);f.seekg(0);f.read(reinterpret_cast<char*>(out.data()),size);temporal_require(bool(f),"temporal SPIRV read failed");return out;}
std::array<uint32_t,2> split(uint64_t v){return {uint32_t(v),uint32_t(v>>32)};}
}
void DaxaTemporalConfig::validate()const{temporal_require(!multi_pair_research||odometry_research,"multi-pair matching requires explicit odometry research");temporal_require(slots>=2&&slots<=3,"temporal slots must be2..3");temporal_require(max_contributors>=1&&max_contributors<=4,"temporal contributor capacity must be1..4");temporal_require(max_reuses>0&&max_age_ns>=0,"invalid temporal reuse/age bound");temporal_require(std::isfinite(relative_depth_tolerance)&&relative_depth_tolerance>=0,"invalid temporal depth tolerance");temporal_require(std::isfinite(confidence_floor)&&confidence_floor>=0&&confidence_floor<=1,"invalid temporal confidence floor");}
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
    std::array<daxa::ExternalTaskBuffer,2> census_nodes;
    std::unique_ptr<daxa::TaskGraph> graph;daxa::TimelineQueryPool queries;DaxaDepthPipeline::Frame input;DaxaObservationFrame obs;
    daxa::BufferId pair_parameters{};
    std::array<daxa::ExternalTaskBuffer,4> pair_nodes;
    std::array<daxa::ExternalTaskBuffer,2> image_nodes;
    daxa::ExternalTaskBuffer pair_parameter_node;
    std::vector<DaxaStereoHistory::Frame> pairs;
    uint64_t generation=0,last_write=0,last_read=0;
};
struct DaxaTemporal::Impl {
    std::shared_ptr<DaxaFoundation> context;std::shared_ptr<const DaxaRectification> maps;DaxaTemporalConfig config;
    std::vector<std::shared_ptr<Slot>> slots;std::shared_ptr<Slot> history;
    std::vector<daxa::BufferId> owned;std::vector<daxa::MemoryBlock> memory;std::array<daxa::ComputePipeline,4> pipelines;
    std::optional<std::pair<uint64_t,uint64_t>> vo_epoch;bool vo_dispatch=false,vo_suppress_history=false;
    std::shared_ptr<DaxaStereoHistory> pair_history;std::optional<StereoFrameIdentity> vo_origin;PairContext pair_context;
    detail::TemporalSequence sequence;std::array<TemporalMatrix,2> eye_to_rig,rig_to_eye;Parameters constants;
    uint64_t allocated=0,next_generation=1,retired=0;uint32_t timestamp_bits=0;float timestamp_period=0;bool refinement_mode=false;
    const std::thread::id thread=std::this_thread::get_id();
    void require_thread()const{temporal_require(thread==std::this_thread::get_id(),"temporal requires one shared submission thread");}
    bool complete(Slot const& s,uint64_t oldest)const{return detail::temporal_tickets_complete(s.last_write,s.last_read,oldest);}
    long internal(Slot const* s)const{return detail::temporal_internal_references(history.get()==s);}
    [[noreturn]] void fail_backpressure(uint64_t boundary)const{
        // Cold failure path only: retain the exact admission snapshot, not a
        // later completion poll that could conceal a boundary equality.
        std::ostringstream message;message<<"temporal backpressure: all nonhistory output slots retained/inflight; completion_boundary="<<boundary<<" latest_main_submit="<<context->device.latest_queue_submit_index(daxa::QUEUE_MAIN);
        for(size_t i=0;i<slots.size();++i){auto const& s=slots[i];message<<" slot["<<i<<"]{history="<<(s==history)<<",refs="<<s.use_count()<<",generation="<<s->generation<<",last_write="<<s->last_write<<",last_read="<<s->last_read<<",input="<<(s->input||s->obs)<<"}";}
        throw std::runtime_error(message.str());
    }
    size_t collect_at(uint64_t oldest){size_t n=0;for(auto& s:slots){
        // Cached pair leases are internal dispatch inputs, not part of an exported
        // temporal frame. Retire them after their own read even if output is held.
        if(detail::completed_submission(s->last_write,oldest))s->pairs.clear();
        if((s->input||s->obs)&&s.use_count()==internal(s.get())&&complete(*s,oldest)){s->input={};s->obs={};++n;++retired;}
    }return n;}
    daxa::BufferId allocate(size_t size,std::string const& name,bool host=false){auto& d=context->device;daxa::BufferInfo info{.size=size,.name=name};daxa::BufferId out;if(host){auto m=d.create_memory({.requirements=context->coherent_readback_requirements(info),.flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});out=d.create_buffer_from_memory_block({.buffer_info=info,.memory_block=m});memory.push_back(std::move(m));}else out=d.create_buffer(info);owned.push_back(out);allocated+=size;return out;}
    ~Impl(){if(!context||!context->device.get())return;auto& d=context->device;d.wait_idle();history.reset();slots.clear();pipelines={};for(auto b:owned)if(d.is_id_valid(b))d.destroy_buffer(b);d.collect_garbage();memory.clear();}
    void prepare_pairs(Slot& slot,Parameters& params,DaxaObservationFrame const& fresh) {
        params.pair_count=0;params.require_pair_support=vo_dispatch;
        slot.pairs.clear();
        for(unsigned eye=0;eye<2;++eye) {
            auto image=fresh.image(eye?DaxaObservationImage::RectifiedRight:DaxaObservationImage::RectifiedLeft).buffer();
            slot.image_nodes[eye].set_buffer(image);params.current_images[eye]=context->device.device_address(image).value();
        }
        std::array<PairParameters,4> data{};
        for(unsigned h=0;h<4;++h)slot.pair_nodes[h].set_buffer(slot.pair_parameters);
        if(vo_dispatch&&pair_context.pose&&pair_context.pairs)for(auto const& selected:*pair_context.pairs) {
            unsigned h=params.pair_count++;slot.pairs.push_back(selected.frame);auto& record=data[h];
            slot.pair_nodes[h].set_buffer(selected.frame.buffer());
            for(unsigned p=0;p<7;++p) {
                auto view=selected.frame.plane(DaxaStereoHistoryPlane(p));record.buffers[p]=context->device.device_address(view.buffer).value()+view.offset;
            }
            record.source_id=split(selected.frame.pose().source.frame_id);
            auto relative=temporal_multiply(temporal_inverse(*selected.frame.pose().world_from_rig),*pair_context.pose->world_from_rig);
            for(unsigned eye=0;eye<2;++eye) {
                auto matrix=temporal_multiply(temporal_multiply(rig_to_eye[0],relative),eye_to_rig[eye]);
                for(unsigned i=0;i<16;++i){record.current_to_left[eye*16+i]=float(matrix[i]);temporal_require(std::isfinite(record.current_to_left[eye*16+i]),"history transform outside float32 domain");}
            }
        }
        std::memcpy(context->device.buffer_host_address(slot.pair_parameters).value(),data.data(),sizeof(data));
    }
    void consume_pairs(Slot& slot){for(auto const& frame:slot.pairs)frame.mark_consumed(slot.last_write);}
    void setup(DaxaObservationFrame const& observation,std::array<daxa::BufferId,2> sup_bufs,std::array<uint64_t,2> sup_offs,std::array<daxa::BufferId,2> cen_bufs){
        config.validate();temporal_require(config.multi_pair_research||!pair_history,"shared pair history requires multi-pair configuration");temporal_require(context&&context->device.get(),"temporal requires foundation");temporal_require(observation.context()==context,"temporal prototype context mismatch");maps=observation.map_owner();maps->metric_calibration();temporal_require(maps->geometry.has_value(),"temporal requires calibrated metric geometry");auto const& geometry=*maps->geometry;
        bool mapless=std::all_of(maps->maps.begin(),maps->maps.end(),[](auto const& m){return m.empty();});
        if(mapless) {
            auto source=observation.source();
            temporal_require(source&&source->source_kind==CaptureSourceKind::CanonicalRgbMetadata&&
                canonical_rgb_grid_association(*source,maps->eye_width,maps->source_height,maps->width,maps->height,maps->original_identity,maps->rectified_identity),
                "mapless temporal input requires admitted canonical RGB grid association");
            geometry.validate();
        }else geometry.validate_map_association({maps->maps[0],maps->maps[1],maps->maps[2],maps->maps[3]});
        uint64_t eyes=config.left_eye_only?1ull:2ull,count=uint64_t(maps->width)*maps->height*eyes;temporal_require(count&&count<=UINT32_MAX,"temporal index/counter overflow");
        if(config.multi_pair_research){DaxaStereoHistoryConfig hc;hc.allow_unqualified_research=true;hc.max_age_ns=uint64_t(config.max_age_ns);if(!pair_history)pair_history=std::make_shared<DaxaStereoHistory>(context,observation,hc);temporal_require(pair_history->compatible(observation),"foreign shared stereo history");}
        auto& d=context->device;auto physical=daxa_dvc_get_vk_physical_device(d.get());VkPhysicalDeviceProperties properties{};vkGetPhysicalDeviceProperties(physical,&properties);timestamp_period=properties.limits.timestampPeriod;temporal_require(properties.limits.maxComputeWorkGroupSize[0]>=64&&properties.limits.maxComputeWorkGroupInvocations>=64&&(count+63)/64<=properties.limits.maxComputeWorkGroupCount[0],"temporal dispatch unsupported");for(auto b:bytes)temporal_require(count*b<=properties.limits.maxStorageBufferRange,"temporal plane exceeds storage range");
        uint32_t family=0,families_count=0;temporal_require(daxa_dvc_get_vk_queue(d.get(),{DAXA_QUEUE_TYPE_MAIN,0},nullptr,&family)==DAXA_RESULT_SUCCESS,"missing temporal main queue");vkGetPhysicalDeviceQueueFamilyProperties(physical,&families_count,nullptr);std::vector<VkQueueFamilyProperties> families(families_count);vkGetPhysicalDeviceQueueFamilyProperties(physical,&families_count,families.data());temporal_require(family<families_count&&families[family].timestampValidBits,"temporal timestamps unavailable");timestamp_bits=families[family].timestampValidBits;
        constants.baseline_m=float(maps->metric_calibration().baseline_m);constants.width=maps->width;constants.height=maps->height;constants.capacity=config.max_contributors|(config.inverse_depth_fusion?16u:0u)|(config.disocclusion_guard?32u:0u)|(config.temporal_stereo_research?64u:0u)|(config.left_eye_only?128u:0u)|((std::min<unsigned>(config.max_disparity?config.max_disparity:128u,4095u))<<16);constants.max_reuses=config.max_reuses;constants.max_age=split(uint64_t(config.max_age_ns));constants.tolerance=config.relative_depth_tolerance;constants.confidence_floor=config.confidence_floor;
        for(unsigned eye=0;eye<2;++eye){auto const& k=geometry.rectified.eyes[eye];auto intrinsic=detail::temporal_pinhole_f32(k);std::copy(intrinsic.begin(),intrinsic.end(),constants.intrinsics.begin()+eye*4);TemporalMatrix rotation{};rotation[15]=1;auto const& r=eye?geometry.R2:geometry.R1;for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)rotation[i*4+j]=r[j*3+i];eye_to_rig[eye]=eye?temporal_multiply(geometry.original.right_to_left,rotation):rotation;rig_to_eye[eye]=temporal_inverse(eye_to_rig[eye]);}
        for(auto v:constants.intrinsics)temporal_require(std::isfinite(v),"temporal pinhole outside float32 domain");
        constexpr std::array<char const*,4> entries{"clear","splat_z","splat_owner","gather_fuse"};for(unsigned i=0;i<4;++i){auto code=shader(entries[i]);pipelines[i]=d.create_compute_pipeline({.shader_info={.byte_code=code.data(),.byte_code_size=uint32_t(code.size())},.push_constant_size=sizeof(Push),.name=entries[i]});}
        slots.reserve(config.slots);owned.reserve(config.slots*(15+unsigned(config.multi_pair_research)));memory.reserve(config.slots);
        bool use_census=refinement_mode&&(config.temporal_stereo_research||config.disocclusion_guard);
        for(unsigned index=0;index<config.slots;++index){auto s=std::make_shared<Slot>();auto prefix="temporal_"+std::to_string(index)+"_";auto params=constants;
            for(unsigned p=0;p<11;++p){s->outputs[p]=allocate(count*bytes[p],prefix+"output_"+std::to_string(p));params.output[p]=d.device_address(s->outputs[p]).value();s->output_nodes[p]=daxa::ExternalTaskBuffer({.buffer=s->outputs[p],.name=prefix+"out_"+std::to_string(p)});s->history_nodes[p]=daxa::ExternalTaskBuffer({.buffer=s->outputs[p],.name=prefix+"history_"+std::to_string(p)});params.history[p]=params.output[p];}
            for(unsigned p=0;p<6;++p){auto b=observation.buffer(DaxaObservationPlane(p));s->fresh_nodes[p]=daxa::ExternalTaskBuffer({.buffer=b,.name=prefix+"fresh_"+std::to_string(p)});params.fresh[p]=d.device_address(b).value();}
            for(unsigned p=0;p<2;++p){s->support_nodes[p]=daxa::ExternalTaskBuffer({.buffer=sup_bufs[p],.name=prefix+"support_"+std::to_string(p)});params.support[p]=d.device_address(sup_bufs[p]).value()+sup_offs[p];}
            if(use_census)for(unsigned p=0;p<2;++p)s->census_nodes[p]=daxa::ExternalTaskBuffer({.buffer=cen_bufs[p],.name=prefix+"census_"+std::to_string(p)});
            s->zbuffer=allocate(count*4,prefix+"z");s->owners=allocate(count*4,prefix+"owner");s->counters=allocate(sizeof(DaxaTemporalCounters),prefix+"counters");s->parameters=allocate(sizeof(Parameters),prefix+"parameters",true);params.zbuffer=d.device_address(s->zbuffer).value();params.owners=d.device_address(s->owners).value();params.counters=d.device_address(s->counters).value();std::memcpy(d.buffer_host_address(s->parameters).value(),&params,sizeof(params));
            if(config.multi_pair_research) {
                s->pairs.reserve(4);s->pair_parameters=allocate(sizeof(PairParameters)*4,prefix+"pair_parameters",true);params.pair_parameters=d.device_address(s->pair_parameters).value();
                s->pair_parameter_node=daxa::ExternalTaskBuffer({.buffer=s->pair_parameters,.name=prefix+"pair parameters"});
                for(unsigned j=0;j<4;++j)s->pair_nodes[j]=daxa::ExternalTaskBuffer({.buffer=s->pair_parameters,.name=prefix+"pair_"+std::to_string(j)});
                for(unsigned j=0;j<2;++j){auto b=observation.image(j?DaxaObservationImage::RectifiedRight:DaxaObservationImage::RectifiedLeft).buffer();s->image_nodes[j]=daxa::ExternalTaskBuffer({.buffer=b,.name=prefix+"image_"+std::to_string(j)});params.current_images[j]=d.device_address(b).value();}
                std::memcpy(d.buffer_host_address(s->parameters).value(),&params,sizeof(params));
            }
            std::array<daxa::BufferId,4> scratch{s->parameters,s->zbuffer,s->owners,s->counters};for(unsigned p=0;p<4;++p)s->scratch_nodes[p]=daxa::ExternalTaskBuffer({.buffer=scratch[p],.name=prefix+"scratch_"+std::to_string(p)});
            s->queries=d.create_timeline_query_pool({.query_count=8,.name=prefix+"queries"});s->graph=std::make_unique<daxa::TaskGraph>(daxa::TaskGraphInfo{.device=d,.name=prefix+"graph"});auto& graph=*s->graph;
            std::array<daxa::TaskBufferView,6> fresh;std::array<daxa::TaskBufferView,11> old,out;std::array<daxa::TaskBufferView,2> support,cen{};std::array<daxa::TaskBufferView,4> tmp;
            for(unsigned p=0;p<6;++p){fresh[p]=graph.register_buffer(s->fresh_nodes[p]);}
            for(unsigned p=0;p<11;++p){old[p]=graph.register_buffer(s->history_nodes[p]);out[p]=graph.register_buffer(s->output_nodes[p]);}
            for(unsigned p=0;p<2;++p){support[p]=graph.register_buffer(s->support_nodes[p]);}
            if(use_census)for(unsigned p=0;p<2;++p){cen[p]=graph.register_buffer(s->census_nodes[p]);}
            for(unsigned p=0;p<4;++p){tmp[p]=graph.register_buffer(s->scratch_nodes[p]);}
            std::array<daxa::TaskBufferView,4> pair_views;std::array<daxa::TaskBufferView,2> image_views;daxa::TaskBufferView pair_parameters_view;
            if(config.multi_pair_research){pair_parameters_view=graph.register_buffer(s->pair_parameter_node);for(unsigned j=0;j<4;++j)pair_views[j]=graph.register_buffer(s->pair_nodes[j]);for(unsigned j=0;j<2;++j)image_views[j]=graph.register_buffer(s->image_nodes[j]);}
            auto* raw=s.get();auto address=d.device_address(s->parameters).value();for(unsigned pass=0;pass<4;++pass){auto task=daxa::InlineTask::Compute(entries[pass]);task.reads(tmp[0]);if(pass==0){task.writes(tmp[1]).writes(tmp[2]).writes(tmp[3]);}else{for(auto v:old)task.reads(v);for(auto v:support)task.reads(v);task.reads_writes(tmp[3]);if(pass==1)task.reads_writes(tmp[1]);if(pass==2)task.reads(tmp[1]).reads_writes(tmp[2]);if(pass==3){if(config.multi_pair_research){task.reads(pair_parameters_view);for(auto v:pair_views)task.reads(v);for(auto v:image_views)task.reads(v);}task.reads(tmp[1]).reads(tmp[2]);for(auto v:fresh)task.reads(v);if(use_census){task.reads(cen[0]).reads(cen[1]);}for(auto v:out)task.writes(v);}}
                graph.add_task(task.executes([this,raw,address,pass,count](daxa::TaskInterface ti){if(pass==3&&config.multi_pair_research)ti.recorder.pipeline_barrier({.src_access=daxa::AccessConsts::TRANSFER_WRITE|daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ});if(pass==0){ti.recorder.pipeline_barrier({.src_access=daxa::AccessConsts::HOST_WRITE|daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE});ti.recorder.reset_timestamps({.query_pool=raw->queries,.start_index=0,.count=8});}ti.recorder.write_timestamp({.query_pool=raw->queries,.pipeline_stage=daxa::PipelineStageFlagBits::COMPUTE_SHADER,.query_index=2*pass});ti.recorder.set_pipeline(pipelines[pass]);ti.recorder.push_constant(Push{address});ti.recorder.dispatch({uint32_t((std::max<uint64_t>(count,12)+63)/64),1,1});ti.recorder.write_timestamp({.query_pool=raw->queries,.pipeline_stage=daxa::PipelineStageFlagBits::COMPUTE_SHADER,.query_index=2*pass+1});}));}
            graph.submit({});graph.complete({});slots.push_back(std::move(s));
        }
    }
    Impl(std::shared_ptr<DaxaFoundation> c,DaxaDepthPipeline::Frame const& prototype,DaxaTemporalConfig cfg,std::shared_ptr<DaxaStereoHistory> shared_history):context(std::move(c)),config(cfg),pair_history(std::move(shared_history)),refinement_mode(false){
        setup(prototype.observations(),{prototype.image(DaxaPreprocessPlane::LeftSupport).buffer(),prototype.image(DaxaPreprocessPlane::RightSupport).buffer()},{0,0},{});
    }
    Impl(std::shared_ptr<DaxaFoundation> c,DaxaRefinement::Frame const& prototype,DaxaTemporalConfig cfg,std::shared_ptr<DaxaStereoHistory> shared_history):context(std::move(c)),config(cfg),pair_history(std::move(shared_history)),refinement_mode(true){
        setup(prototype.observations(),prototype.support_buffers(),prototype.support_offsets(),{prototype.buffer(DaxaRefinementPlane::Census),prototype.buffer(DaxaRefinementPlane::CensusValidity)});
    }
};
DaxaTemporal::DaxaTemporal(std::shared_ptr<DaxaFoundation> context,DaxaDepthPipeline::Frame const& prototype,DaxaTemporalConfig config,std::shared_ptr<DaxaStereoHistory> shared_history):impl_(std::make_shared<Impl>(std::move(context),prototype,config,std::move(shared_history))){}
DaxaTemporal::DaxaTemporal(std::shared_ptr<DaxaFoundation> context,DaxaRefinement::Frame const& prototype,DaxaTemporalConfig config,std::shared_ptr<DaxaStereoHistory> shared_history):impl_(std::make_shared<Impl>(std::move(context),prototype,config,std::move(shared_history))){}
namespace {
bool earlier_acquisition_lower_bound(StereoFrameIdentity const& a,StereoFrameIdentity const& b) {
    auto ua=*a.clock_mapping_uncertainty_ns,ub=*b.clock_mapping_uncertainty_ns;
    bool negative_a=a.acquisition_timestamp_ns<ua,negative_b=b.acquisition_timestamp_ns<ub;
    if(negative_a!=negative_b)return negative_a;
    if(negative_a)return ua-a.acquisition_timestamp_ns>ub-b.acquisition_timestamp_ns;
    return a.acquisition_timestamp_ns-ua<b.acquisition_timestamp_ns-ub;
}
std::optional<DaxaTimedRigPose> odometry_pose(DaxaObservationFrame const& input,StereoWorldPose const& pose,bool research) {
    auto source=input.source();auto maps=input.calibration();auto const& id=pose.source;
    if(!source||!maps||!stereo_pose_usable(pose,research)||
       id.frame_id!=source->frame_id||id.acquisition_timestamp_ns!=source->camera_timestamp_ns||id.ingress_timestamp_ns!=source->host_ingress_timestamp_ns||
       id.device!=source->device_identity||id.acquisition_clock!=source->camera_clock_domain||id.ingress_clock!=source->host_clock_domain||
       id.original_calibration!=maps->original_identity||id.original_calibration!=source->calibration_identity||id.rectified_calibration!=maps->rectified_identity||
       id.ingress_timestamp_ns>uint64_t(INT64_MAX))return {};
    DaxaTimedRigPose result{*pose.world_from_rig,int64_t(id.ingress_timestamp_ns),source->host_clock_domain};
    try {result.validate(*source);}catch(std::invalid_argument const&){return {};}
    return result;
}
struct OdometryDispatch {
    bool& active;bool& suppress;PairContext& context;
    OdometryDispatch(bool& dispatch,bool& suppression,bool candidate_suppression,PairContext& destination,PairContext inputs):active(dispatch),suppress(suppression),context(destination){
        temporal_require(!active,"recursive odometry submission");active=true;suppress=candidate_suppression;context=inputs;
    }
    ~OdometryDispatch(){active=false;suppress=false;context={};}
};
}
namespace {
template<class State,class Submit>
DaxaTemporal::Frame submit_odometry(State& e,DaxaObservationFrame const& fresh,StereoWorldPose const& pose,Submit const& submit) {
    e.require_thread();
    auto p=odometry_pose(fresh,pose,e.config.odometry_research);
    auto epoch=std::pair{pose.trajectory_id,pose.epoch};
    bool suppress=!p||e.vo_epoch!=epoch||pose.status==StereoTrajectoryStatus::Anchored;
    std::optional<StereoFrameIdentity> origin;
    if(p&&!e.config.odometry_research) {
        if(!pose.source.clock_mapping_uncertainty_ns||*pose.source.clock_mapping_uncertainty_ns>uint64_t(e.config.max_age_ns))p.reset();
        if(p&&!suppress&&e.vo_origin) {
            auto bound=stereo_acquisition_age_bound(*e.vo_origin,pose.source);
            if(!bound||*bound>uint64_t(e.config.max_age_ns))suppress=true;
        }
        if(p)origin=suppress||!e.vo_origin||earlier_acquisition_lower_bound(pose.source,*e.vo_origin)?pose.source:*e.vo_origin;
        else suppress=true;
    }
    auto selected=e.pair_history?e.pair_history->select(pose):std::vector<DaxaStereoHistory::Selection>{};
    std::erase_if(selected,[&](auto const& pair){return pair.ingress_age_ns>uint64_t(e.config.max_age_ns)||
        (pair.acquisition_age_upper_bound_ns&&*pair.acquisition_age_upper_bound_ns>uint64_t(e.config.max_age_ns));});
    OdometryDispatch guard(e.vo_dispatch,e.vo_suppress_history,suppress,e.pair_context,{&pose,&selected});auto output=submit(p);
    if(p)e.vo_epoch=epoch;else e.vo_epoch.reset();e.vo_origin=std::move(origin);if(e.pair_history){
        // Initial research mode serializes at this output fence to reuse exactly
        // four owned cache slots. No device-wide wait or image readback occurs.
        output.wait();e.collect_at(e.context->device.oldest_pending_submit_index());selected.clear();e.pair_history->collect();if(p)e.pair_history->store(fresh,pose);else e.pair_history->clear();
    }return output;
}
}
DaxaTemporal::Frame DaxaTemporal::process_odometry(DaxaDepthPipeline::Frame const& input,StereoWorldPose const& pose) {
    return submit_odometry(*impl_,input.observations(),pose,[&](auto const& p){return process(input,p);});
}
DaxaTemporal::Frame DaxaTemporal::process_odometry(DaxaRefinement::Frame const& input,StereoWorldPose const& pose,bool use_completed) {
    return submit_odometry(*impl_,input.observations(),pose,[&](auto const& p){return process(input,p,use_completed);});
}
DaxaTemporal::Frame::Frame(std::shared_ptr<Impl> owner,std::shared_ptr<Slot> slot):owner_(std::move(owner)),slot_(std::move(slot)),generation_(slot_->generation){}
DaxaTemporal::Frame& DaxaTemporal::Frame::operator=(Frame other)noexcept{owner_.swap(other.owner_);slot_.swap(other.slot_);std::swap(generation_,other.generation_);return *this;}
void DaxaTemporal::Frame::validate()const{temporal_require(owner_&&slot_&&generation_==slot_->generation&&(slot_->input||slot_->obs),"empty/stale temporal frame");}
DaxaTemporal::Frame DaxaTemporal::process(DaxaDepthPipeline::Frame const& input,std::optional<DaxaTimedRigPose> const& pose){
    auto& e=*impl_;e.require_thread();auto fresh=input.observations();temporal_require(fresh.context()==e.context&&fresh.map_owner()==e.maps&&fresh.width()==e.maps->width&&fresh.height()==e.maps->height,"temporal incompatible context/calibration/grid");auto capture=input.source();auto admission=e.sequence.prepare(*capture,pose,e.config.max_age_ns);
    admission.warp=admission.warp&&!e.vo_suppress_history&&(e.vo_dispatch||!e.vo_epoch);
    std::array<float,32> transforms{};
    if(admission.warp){auto relative=temporal_multiply(temporal_inverse(pose->world_from_rig),e.sequence.previous);for(unsigned eye=0;eye<2;++eye){auto m=temporal_multiply(temporal_multiply(e.rig_to_eye[eye],relative),e.eye_to_rig[eye]);for(unsigned j=0;j<16;++j){transforms[eye*16+j]=float(m[j]);temporal_require(std::isfinite(transforms[eye*16+j]),"temporal motion outside float32 domain");}}}
    auto& d=e.context->device;auto oldest=d.oldest_pending_submit_index();e.collect_at(oldest);
    for(auto& s:e.slots){if(s==e.history||s.use_count()!=1||!e.complete(*s,oldest))continue;
        auto* params=reinterpret_cast<Parameters*>(d.buffer_host_address(s->parameters).value());params->warp=admission.warp&&bool(e.history);params->elapsed=split(admission.elapsed);params->transforms=transforms;
        for(unsigned p=0;p<6;++p){auto b=fresh.buffer(DaxaObservationPlane(p));s->fresh_nodes[p].set_buffer(b);params->fresh[p]=d.device_address(b).value();}
        for(unsigned p=0;p<2;++p){auto b=input.image(p?DaxaPreprocessPlane::RightSupport:DaxaPreprocessPlane::LeftSupport).buffer();s->support_nodes[p].set_buffer(b);params->support[p]=d.device_address(b).value();}
        auto* old=params->warp?e.history.get():(e.slots[0].get()==s.get()?e.slots[1].get():e.slots[0].get());
        for(unsigned p=0;p<11;++p){s->history_nodes[p].set_buffer(old->outputs[p]);params->history[p]=d.device_address(old->outputs[p]).value();}
        if(e.config.multi_pair_research)e.prepare_pairs(*s,*params,fresh);
        s->input=input;s->generation=e.next_generation++;
        // execute may submit before throwing. Pin both candidate input and the
        // last historical read until the actual shared-queue ticket completes.
        try{s->graph->execute({});}catch(...){s->last_write=d.latest_queue_submit_index(daxa::QUEUE_MAIN);if(e.config.multi_pair_research)e.consume_pairs(*s);if(params->warp)e.history->last_read=std::max(e.history->last_read,s->last_write);throw;}
        s->last_write=d.latest_queue_submit_index(daxa::QUEUE_MAIN);if(e.config.multi_pair_research)e.consume_pairs(*s);if(params->warp)e.history->last_read=std::max(e.history->last_read,s->last_write);
        e.sequence.commit(*capture,pose);if(!e.vo_dispatch){e.vo_epoch.reset();e.vo_origin.reset();if(e.pair_history)e.pair_history->clear();}if(pose)e.history=s;else e.history.reset();return Frame(impl_,s);
    }
    e.fail_backpressure(oldest);
}
DaxaTemporal::Frame DaxaTemporal::process(DaxaRefinement::Frame const& input,std::optional<DaxaTimedRigPose> const& pose,bool use_completed){
    auto& e=*impl_;e.require_thread();temporal_require(e.refinement_mode,"temporal requires refinement prototype for refinement input");
    auto fresh=use_completed?input.completed_observations():input.observations();
    temporal_require(fresh.context()==e.context&&fresh.map_owner()==e.maps&&fresh.width()==e.maps->width&&fresh.height()==e.maps->height,"temporal incompatible context/calibration/grid");
    auto capture=fresh.source();auto admission=e.sequence.prepare(*capture,pose,e.config.max_age_ns);
    admission.warp=admission.warp&&!e.vo_suppress_history&&(e.vo_dispatch||!e.vo_epoch);
    std::array<float,32> transforms{};
    if(admission.warp){auto relative=temporal_multiply(temporal_inverse(pose->world_from_rig),e.sequence.previous);for(unsigned eye=0;eye<2;++eye){auto m=temporal_multiply(temporal_multiply(e.rig_to_eye[eye],relative),e.eye_to_rig[eye]);for(unsigned j=0;j<16;++j){transforms[eye*16+j]=float(m[j]);temporal_require(std::isfinite(transforms[eye*16+j]),"temporal motion outside float32 domain");}}}
    auto& d=e.context->device;
    if(e.config.temporal_stereo_research||e.config.disocclusion_guard){
        auto cen_b=input.buffer(DaxaRefinementPlane::Census),cval_b=input.buffer(DaxaRefinementPlane::CensusValidity);
        auto cen_a=d.device_address(cen_b).value(),cval_a=d.device_address(cval_b).value();
        auto c0=split(cen_a),c1=split(cval_a),sid=split(capture->frame_id);
        auto metric=e.maps->metric_calibration();
        float fxb=metric.fx*metric.baseline_m,cxo=metric.cx_offset;
        std::memcpy(&transforms[12],&c0[0],4);std::memcpy(&transforms[13],&c0[1],4);
        std::memcpy(&transforms[14],&c1[0],4);std::memcpy(&transforms[15],&c1[1],4);
        transforms[28]=fxb;transforms[29]=cxo;
        std::memcpy(&transforms[30],&sid[0],4);std::memcpy(&transforms[31],&sid[1],4);
    }
    auto oldest=d.oldest_pending_submit_index();e.collect_at(oldest);
    auto sup_bufs=input.support_buffers();auto sup_offs=input.support_offsets();
    for(auto& s:e.slots){if(s==e.history||s.use_count()!=1||!e.complete(*s,oldest))continue;
        auto* params=reinterpret_cast<Parameters*>(d.buffer_host_address(s->parameters).value());params->warp=admission.warp&&bool(e.history);params->elapsed=split(admission.elapsed);params->transforms=transforms;
        for(unsigned p=0;p<6;++p){auto b=fresh.buffer(DaxaObservationPlane(p));s->fresh_nodes[p].set_buffer(b);params->fresh[p]=d.device_address(b).value();}
        for(unsigned p=0;p<2;++p){s->support_nodes[p].set_buffer(sup_bufs[p]);params->support[p]=d.device_address(sup_bufs[p]).value()+sup_offs[p];}
        if(e.config.temporal_stereo_research||e.config.disocclusion_guard){s->census_nodes[0].set_buffer(input.buffer(DaxaRefinementPlane::Census));s->census_nodes[1].set_buffer(input.buffer(DaxaRefinementPlane::CensusValidity));}
        auto* old=params->warp?e.history.get():(e.slots[0].get()==s.get()?e.slots[1].get():e.slots[0].get());
        for(unsigned p=0;p<11;++p){s->history_nodes[p].set_buffer(old->outputs[p]);params->history[p]=d.device_address(old->outputs[p]).value();}
        if(e.config.multi_pair_research)e.prepare_pairs(*s,*params,fresh);
        s->obs=fresh;s->generation=e.next_generation++;
        try{s->graph->execute({});}catch(...){s->last_write=d.latest_queue_submit_index(daxa::QUEUE_MAIN);if(e.config.multi_pair_research)e.consume_pairs(*s);if(params->warp)e.history->last_read=std::max(e.history->last_read,s->last_write);throw;}
        s->last_write=d.latest_queue_submit_index(daxa::QUEUE_MAIN);if(e.config.multi_pair_research)e.consume_pairs(*s);if(params->warp)e.history->last_read=std::max(e.history->last_read,s->last_write);
        e.sequence.commit(*capture,pose);if(!e.vo_dispatch){e.vo_epoch.reset();e.vo_origin.reset();if(e.pair_history)e.pair_history->clear();}if(pose)e.history=s;else e.history.reset();return Frame(impl_,s);
    }
    e.fail_backpressure(oldest);
}
size_t DaxaTemporal::collect(){impl_->require_thread();auto& d=impl_->context->device;auto retired=impl_->collect_at(d.oldest_pending_submit_index());if(impl_->pair_history)impl_->pair_history->collect();d.collect_garbage();return retired;}
bool DaxaTemporal::can_process()const{auto& e=*impl_;e.require_thread();auto oldest=e.context->device.oldest_pending_submit_index();for(auto const& s:e.slots)if(s!=e.history&&s.use_count()==1&&e.complete(*s,oldest))return true;return false;}
void DaxaTemporal::reset(){impl_->require_thread();impl_->vo_epoch.reset();impl_->vo_origin.reset();if(impl_->pair_history)impl_->pair_history->clear();impl_->history.reset();impl_->sequence.reset();}
void DaxaTemporal::Frame::wait()const{validate();owner_->context->device.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=slot_->last_write,.timeout=30'000'000'000ull});}
bool DaxaTemporal::Frame::ready()const{validate();return detail::completed_submission(slot_->last_write,owner_->context->device.oldest_pending_submit_index());}
uint32_t DaxaTemporal::Frame::width()const{validate();return owner_->maps->width;}uint32_t DaxaTemporal::Frame::height()const{validate();return owner_->maps->height;}
uint64_t DaxaTemporal::Frame::generation()const{validate();return generation_;}uint64_t DaxaTemporal::Frame::context_id()const{validate();return reinterpret_cast<uintptr_t>(owner_->context->device.get());}uint64_t DaxaTemporal::Frame::submit_ticket()const{validate();return slot_->last_write;}
DaxaObservationAncestryView DaxaTemporal::Frame::ancestry()const{validate();std::array<daxa::BufferId,5> sidecars{};std::copy(slot_->outputs.begin()+6,slot_->outputs.end(),sidecars.begin());size_t eyes=owner_->config.left_eye_only?1:2;return DaxaObservationAncestryView(size_t(owner_->maps->width)*owner_->maps->height*eyes,owner_->config.max_contributors,sidecars);}
DaxaObservationFrame DaxaTemporal::Frame::observations()const{validate();temporal_require(!owner_->config.left_eye_only,"left_eye_only temporal frame requires left_observations()");auto input=slot_->obs?slot_->obs:slot_->input.observations();std::array<daxa::BufferId,6> planes{};std::copy_n(slot_->outputs.begin(),6,planes.begin());std::array<DaxaObservationImageView,4> images{};for(unsigned i=0;i<4;++i)images[i]=input.image(DaxaObservationImage(i));return DaxaObservationFrame(owner_->context,owner_,slot_,&slot_->generation,generation_,slot_->last_write,owner_->maps,input.source(),planes,images,ancestry());}
void DaxaLeftObservationFrame::validate()const{temporal_require(context_&&owner_&&lease_&&live_generation_&&generation_==*live_generation_,"empty/stale left observation frame");}
daxa::BufferId DaxaLeftObservationFrame::buffer(DaxaObservationPlane p)const{validate();temporal_require(unsigned(p)<6,"invalid left observation plane");return planes_[unsigned(p)];}
size_t DaxaLeftObservationFrame::size_bytes(DaxaObservationPlane p)const{validate();temporal_require(unsigned(p)<6,"invalid left observation plane");return size_t(width_)*height_*(unsigned(p)<4?4:8);}
DaxaObservationImageView DaxaLeftObservationFrame::image(unsigned i)const{validate();temporal_require(i<2,"invalid left observation image");return images_[i];}
DaxaLeftObservationFrame DaxaTemporal::Frame::left_observations()const{validate();temporal_require(owner_->config.left_eye_only,"left_observations() requires left_eye_only temporal frame");auto input=slot_->obs?slot_->obs:slot_->input.observations();DaxaLeftObservationFrame out;out.context_=owner_->context;out.owner_=owner_;out.lease_=slot_;out.maps_=owner_->maps;out.source_=input.source();out.live_generation_=&slot_->generation;out.generation_=generation_;out.submit_=slot_->last_write;out.width_=owner_->maps->width;out.height_=owner_->maps->height;std::copy_n(slot_->outputs.begin(),6,out.planes_.begin());out.images_={input.image(DaxaObservationImage::OriginalLeft),input.image(DaxaObservationImage::RectifiedLeft)};out.ancestry_=ancestry();return out;}
DaxaTemporalTiming DaxaTemporal::Frame::timings()const{validate();owner_->require_thread();temporal_require(ready(),"temporal completion required before timing");std::array<uint64_t,16> results{};temporal_require(daxa_timeline_query_pool_query_results(slot_->queries.get(),0,8,results.data())==DAXA_RESULT_SUCCESS,"temporal timestamp query failed");for(unsigned i=0;i<8;++i)temporal_require(results[i*2+1],"temporal timestamp unavailable");auto mask=owner_->timestamp_bits==64?UINT64_MAX:(uint64_t{1}<<owner_->timestamp_bits)-1;auto ms=[&](unsigned a,unsigned b){return double((results[b*2]-results[a*2])&mask)*double(owner_->timestamp_period)/1e6;};DaxaTemporalTiming out;out.clear_gpu_ms=ms(0,1);out.z_gpu_ms=ms(2,3);out.owner_gpu_ms=ms(4,5);out.fuse_gpu_ms=ms(6,7);out.total_gpu_ms=ms(0,7);out.start_tick=results[0];out.end_tick=results[14];out.timestamp_bits=owner_->timestamp_bits;out.timestamp_period=owner_->timestamp_period;return out;}
DaxaTemporalReadback DaxaTemporal::readback(Frame const& f)const{
    f.validate();impl_->require_thread();temporal_require(f.owner_==impl_,"foreign temporal diagnostic frame");f.wait();auto& d=impl_->context->device;DaxaTemporalReadback out;
    auto copy=[&](void* target,size_t size,daxa::BufferId source){daxa::BufferInfo info{.size=size,.name="temporal scoped diagnostic"};auto memory=d.create_memory({.requirements=impl_->context->coherent_readback_requirements(info),.flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});auto host=d.create_buffer_from_memory_block({.buffer_info=info,.memory_block=memory});struct Cleanup{daxa::Device& d;daxa::BufferId b;~Cleanup(){d.destroy_buffer(b);d.collect_garbage();}}cleanup{d,host};auto recorder=d.create_command_recorder({.name="temporal diagnostic transfer"});recorder.pipeline_barrier({.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::TRANSFER_READ});recorder.copy_buffer_to_buffer({.src_buffer=source,.dst_buffer=host,.size=size});recorder.pipeline_barrier({.src_access=daxa::AccessConsts::TRANSFER_WRITE,.dst_access=daxa::AccessConsts::HOST_READ});auto command=recorder.complete_current_commands();d.submit_commands({.command_lists={&command,1}});d.wait_idle();std::memcpy(target,d.buffer_host_address(host).value(),size);};
    size_t eyes=impl_->config.left_eye_only?1:2;
    auto plane=[&](auto& values,unsigned p){auto size=size_t(impl_->maps->width)*impl_->maps->height*eyes*bytes[p];values.resize(size/sizeof(typename std::remove_reference_t<decltype(values)>::value_type));copy(values.data(),size,f.slot_->outputs[p]);};
    plane(out.depth,0);plane(out.confidence,1);plane(out.validity,2);plane(out.provenance,3);plane(out.source_ids,4);plane(out.age_ns,5);plane(out.contributor_ids,6);plane(out.count,7);plane(out.oldest_age_ns,8);plane(out.reuse,9);plane(out.origin_mask,10);copy(&out.counters,sizeof(out.counters),f.slot_->counters);return out;
}
uint64_t DaxaTemporal::requested_buffer_bytes()const{return impl_->allocated+(impl_->pair_history?impl_->pair_history->requested_bytes():0);}uint64_t DaxaTemporal::completed_source_retirements()const{return impl_->retired;}
std::string DaxaTemporal::shader_directory(){return SF_DAXA_TEMPORAL_SPIRV_DIRECTORY;}std::string DaxaTemporal::shader_identity(){return std::string(SF_DAXA_TEMPORAL_WRAPPER_SHA256)+":"+SF_DAXA_TEMPORAL_OPS_SHA256+":"+SF_DAXA_TEMPORAL_PAIRS_SHA256;}
}
