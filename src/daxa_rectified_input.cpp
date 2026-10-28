#include <stereoforge/daxa_rectified_input.hpp>
#include <daxa/utils/task_graph.hpp>
#include <daxa/c/device.h>
#include <daxa/c/types.h>
#include <fstream>
#include <cstring>
#include <cmath>
#include <thread>
namespace stereoforge {
namespace {
void require(bool v,const char* m){if(!v)throw std::invalid_argument(m);}
uint32_t gray(uint32_t c){return (((c&255)*9798+((c>>8)&255)*19235+((c>>16)&255)*3735+16384)>>15);}
uint32_t color(const CanonicalRgbPlane& p,unsigned x,unsigned y){auto q=p.bytes.data()+size_t(y)*p.stride_bytes+size_t(x)*3;return q[0]|(uint32_t(q[1])<<8)|(uint32_t(q[2])<<16)|0xff000000u;}
bool close(double a,double b){return std::abs(a-b)<=1e-6+1e-5*std::abs(b);}
void grid(const DaxaRectification& c){require(c.eye_width&&c.source_height&&c.width&&c.height&&c.eye_width<32767&&c.source_height<32767&&c.width<32767&&c.height<32767&&c.stride_bytes==0,"invalid canonical grid dimensions/stride");for(auto const& m:c.maps)require(m.empty(),"canonical RGB must not carry raw rectification maps");c.metric_calibration();}
}
void DaxaRectifiedInputConfig::validate()const{require(slots==2,"canonical producer requires exactly two slots");}
void validate_daxa_rectified_input_geometry(const DaxaRectification& full,const std::optional<DaxaRectification>& coarse,const DaxaRectifiedInputConfig& cfg,const RectifiedRgbFrame& frame){
    cfg.validate();require(frame.admitted(),"offline-admitted canonical prototype required");validate_canonical_rgb_source(*frame.source);grid(full);auto const& s=*frame.source->canonical_rgb;
    require(canonical_rgb_grid_association(*frame.source,full.eye_width,full.source_height,full.width,full.height,full.original_identity,full.rectified_identity)&&s.rectified_calibration_identity==full.rectified_identity,"full canonical calibration association mismatch");
    require(cfg.coarse_2x==coarse.has_value(),"canonical coarse configuration mismatch");
    if(!coarse)return; // A record may retain a derived grid that this consumer does not request.
    auto const& c=*coarse;grid(c);require(!(full.width&1)&&!(full.height&1)&&c.width*2==full.width&&c.height*2==full.height,"canonical coarse requires even exact2x dimensions");
    require(canonical_rgb_grid_association(*frame.source,c.eye_width,c.source_height,c.width,c.height,c.original_identity,c.rectified_identity)&&s.coarse_rectified_calibration_identity==c.rectified_identity&&c.rectified_identity!=full.rectified_identity,"coarse canonical calibration association mismatch");
    auto const& a=*c.geometry;auto const& b=*full.geometry;
    require(a.original.right_to_left==b.original.right_to_left&&a.rectified.right_to_left==b.rectified.right_to_left&&a.R1==b.R1&&a.R2==b.R2,"canonical rig/rotation mismatch");
    for(unsigned eye=0;eye<2;++eye){auto const& o=a.original.eyes[eye];auto const& p=b.original.eyes[eye];require(o.width==p.width&&o.height==p.height&&o.fx==p.fx&&o.fy==p.fy&&o.cx==p.cx&&o.cy==p.cy&&o.distortion==p.distortion,"canonical original camera mismatch");auto const& x=a.rectified.eyes[eye];auto const& y=b.rectified.eyes[eye];require(close(y.fx,2*x.fx)&&close(y.fy,2*x.fy)&&close(y.cx,2*(x.cx+.5)-.5)&&close(y.cy,2*(x.cy+.5)-.5)&&x.distortion==y.distortion,"canonical halfpixel pinhole mismatch");auto const& cp=eye?a.P2:a.P1;auto const& fp=eye?b.P2:b.P1;for(unsigned j=0;j<12;++j){double expected=j<8?2*cp[j]:cp[j];if(j==2||j==6)expected=2*(cp[j]+.5)-.5;require(close(fp[j],expected),"canonical halfpixel projection mismatch");}}
}
DaxaRectifiedInputReference daxa_rectified_input_reference(const RectifiedRgbFrame& f,const DaxaRectification& full,const std::optional<DaxaRectification>& coarse,DaxaRectifiedInputConfig cfg){
    validate_daxa_rectified_input_geometry(full,coarse,cfg,f);auto const& s=*f.source->canonical_rgb;DaxaRectifiedInputReference out;
    for(unsigned p=0;p<4;++p){auto const& in=s.rgb[p];auto& dst=out.full.planes[p];dst.resize(size_t(in.width)*in.height);for(unsigned y=0;y<in.height;++y)for(unsigned x=0;x<in.width;++x)dst[size_t(y)*in.width+x]=color(in,x,y);}
    for(unsigned eye=0;eye<2;++eye){auto n=size_t(full.width)*full.height;out.full.planes[4+eye].resize(n);out.full.planes[6+eye].resize(n);for(unsigned y=0;y<full.height;++y)for(unsigned x=0;x<full.width;++x){size_t i=size_t(y)*full.width+x;out.full.planes[4+eye][i]=gray(out.full.planes[2+eye][i]);out.full.planes[6+eye][i]=s.support[eye].bytes[size_t(y)*s.support[eye].stride_bytes+x]==1;}}
    if(coarse){out.coarse.emplace();auto& dst=*out.coarse;for(unsigned eye=0;eye<2;++eye){dst.planes[eye]=out.full.planes[eye];for(unsigned p:{2+eye,4+eye,6+eye})dst.planes[p].resize(size_t(coarse->width)*coarse->height);for(unsigned y=0;y<coarse->height;++y)for(unsigned x=0;x<coarse->width;++x){uint32_t c=0xff000000u;unsigned sums[3]{};bool support=true;for(unsigned t=0;t<4;++t){size_t i=size_t(y*2+(t>>1))*full.width+x*2+(t&1);auto rgb=out.full.planes[2+eye][i];for(unsigned ch=0;ch<3;++ch)sums[ch]+=(rgb>>(ch*8))&255;support&=out.full.planes[6+eye][i]==1;}for(unsigned ch=0;ch<3;++ch)c|=((sums[ch]+2)/4)<<(ch*8);size_t i=size_t(y)*coarse->width+x;dst.planes[2+eye][i]=c;dst.planes[4+eye][i]=gray(c);dst.planes[6+eye][i]=support;}}}return out;
}
namespace {
struct Parameters {std::array<uint64_t,6> raw;std::array<uint64_t,8> full;std::array<uint64_t,6> coarse;std::array<uint32_t,6> stride;uint32_t eye_width,source_height,width,height;};
static_assert(sizeof(Parameters)==200&&offsetof(Parameters,eye_width)==184);
struct Push {uint64_t parameters;};
std::vector<uint32_t> shader(const char* entry){std::ifstream f(std::filesystem::path(SF_DAXA_RECTIFIED_INPUT_SPIRV_DIRECTORY)/(std::string("daxa-rectified-input-")+entry+".spv"),std::ios::binary|std::ios::ate);require(bool(f),"canonical offline shader missing");auto bytes=f.tellg();require(bytes>0&&bytes%4==0,"canonical SPIRV invalid");std::vector<uint32_t> out(size_t(bytes)/4);f.seekg(0);f.read(reinterpret_cast<char*>(out.data()),bytes);require(bool(f),"canonical SPIRV read failed");return out;}
}
uint64_t daxa_rectified_input_required_bytes(const DaxaRectification& full,const std::optional<DaxaRectification>& coarse,const DaxaRectifiedInputConfig& cfg,const RectifiedRgbFrame& prototype){
    validate_daxa_rectified_input_geometry(full,coarse,cfg,prototype);
    auto add=[](uint64_t a,uint64_t b){require(b<=UINT64_MAX-a,"canonical allocation size addition overflow");return a+b;};
    auto mul=[](uint64_t a,uint64_t b){require(!a||b<=UINT64_MAX/a,"canonical allocation size multiplication overflow");return a*b;};
    auto const& source=*prototype.source->canonical_rgb;uint64_t upload=0;
    for(unsigned i=0;i<6;++i){auto const& p=i<4?source.rgb[i]:source.support[i-4];auto extent=add(mul(p.stride_bytes,p.height-1),mul(p.width,i<4?3:1));upload=add(upload,add(extent,3)&~uint64_t{3});}
    auto per_slot=add(mul(upload,2),sizeof(Parameters));
    per_slot=add(per_slot,mul(mul(full.eye_width,full.source_height),8));
    per_slot=add(per_slot,mul(mul(full.width,full.height),24));
    if(coarse)per_slot=add(per_slot,mul(mul(coarse->width,coarse->height),24));
    auto total=mul(per_slot,cfg.slots);require(total<2ull*1024*1024*1024,"canonical total allocation ceiling exceeded");return total;
}
struct DaxaRectifiedInput::Impl:detail::DaxaFrameOwner {
    // This member is declared before graphs/pipelines, so constructor failure
    // destroys those users first, then releases every tracked device resource.
    struct Resources {
        std::shared_ptr<DaxaFoundation> context;
        std::vector<daxa::BufferId> owned;
        std::vector<daxa::MemoryBlock> memory;
        uint64_t bytes=0,planned_bytes=0;
        void prepare(std::shared_ptr<DaxaFoundation> foundation,uint64_t planned,size_t buffers,size_t blocks){
            require(planned&&planned<2ull*1024*1024*1024,"canonical allocation preflight budget invalid");
            owned.reserve(buffers);memory.reserve(blocks);planned_bytes=planned;context=std::move(foundation);
        }
        void wait()noexcept{if(context)try{context->device.wait_idle();}catch(...){} }
        void release()noexcept{
            if(!context)return;
            auto& d=context->device;
            for(auto b:owned)try{if(d.is_id_valid(b))d.destroy_buffer(b);}catch(...){}
            owned.clear();try{d.collect_garbage();}catch(...){}memory.clear();bytes=0;context.reset();
        }
        ~Resources()noexcept{wait();release();}
        daxa::BufferId allocate(size_t size,const std::string& name,bool host,uint64_t storage_limit){
            require(size&&size<=storage_limit,"canonical buffer exceeds storage limit");
            require(bytes<=planned_bytes&&size<=planned_bytes-bytes,"canonical allocation exceeds preflight budget");
            require(owned.size()<owned.capacity()&&(!host||memory.size()<memory.capacity()),"canonical allocation tracking not reserved");
            auto& d=context->device;daxa::BufferInfo info{.size=size,.name=name};daxa::BufferId b;
            if(host){auto block=d.create_memory({.requirements=context->coherent_readback_requirements(info),.flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});b=d.create_buffer_from_memory_block({.buffer_info=info,.memory_block=block});memory.push_back(std::move(block));}
            else b=d.create_buffer(info);
            owned.push_back(b);bytes+=size;return b;
        }
    } resources;
    struct Slot {std::shared_ptr<detail::DaxaFrameSlot> full=std::make_shared<detail::DaxaFrameSlot>(),coarse;daxa::BufferId staging,raw,params;std::unique_ptr<daxa::TaskGraph> graph;std::vector<daxa::ExternalTaskBuffer> external;daxa::TimelineQueryPool queries;uint64_t submit=0;};
    DaxaRectifiedInputConfig config;std::shared_ptr<const DaxaRectification> full,coarse;std::array<CanonicalRgbPlane,6> descriptors;std::array<size_t,6> offsets{},extents{};size_t upload_bytes=0;uint64_t next_generation=1,storage_limit=0;double upload_cpu_ms=0;uint32_t timestamp_bits=0;float timestamp_period=0;std::vector<std::shared_ptr<Slot>> slots;std::array<daxa::ComputePipeline,3> pipelines;const std::thread::id thread=std::this_thread::get_id();
    void require_thread()const{require(thread==std::this_thread::get_id(),"single canonical submission thread required");}
    bool available(const std::shared_ptr<Slot>& s,uint64_t oldest)const{return s.use_count()==1&&s->full.use_count()==1&&(!s->coarse||s->coarse.use_count()==1)&&detail::completed_submission(s->submit,oldest);}
    size_t collect_at(uint64_t oldest){size_t retired=0;for(auto const& s:slots)if(available(s,oldest)&&s->full->source){s->full->source.reset();if(s->coarse)s->coarse->source.reset();++retired;}return retired;}
    daxa::BufferId allocate(size_t size,const std::string& name,bool host=false){return resources.allocate(size,name,host,storage_limit);}
    ~Impl()noexcept{resources.wait();slots.clear();pipelines={};resources.release();}
    Impl(DaxaRectification f,std::optional<DaxaRectification> c,DaxaRectifiedInputConfig cfg,std::shared_ptr<DaxaFoundation> foundation,const RectifiedRgbFrame& prototype):config(cfg){
        auto planned=daxa_rectified_input_required_bytes(f,c,cfg,prototype);require(foundation&&foundation->device.get(),"canonical producer requires foundation");context=std::move(foundation);resources.prepare(context,planned,cfg.slots*(c?17u:11u),cfg.slots*2u);slots.reserve(cfg.slots);full=std::make_shared<const DaxaRectification>(std::move(f));if(c)coarse=std::make_shared<const DaxaRectification>(std::move(*c));auto const& source=*prototype.source->canonical_rgb;for(unsigned i=0;i<6;++i){descriptors[i]=i<4?source.rgb[i]:source.support[i-4];offsets[i]=upload_bytes;extents[i]=size_t(descriptors[i].stride_bytes)*(descriptors[i].height-1)+size_t(descriptors[i].width)*(i<4?3:1);upload_bytes+=(extents[i]+3)&~size_t{3};descriptors[i].bytes={};}
        auto& d=context->device;auto physical=daxa_dvc_get_vk_physical_device(d.get());VkPhysicalDeviceProperties properties{};vkGetPhysicalDeviceProperties(physical,&properties);storage_limit=properties.limits.maxStorageBufferRange;timestamp_period=properties.limits.timestampPeriod;require(properties.limits.maxComputeWorkGroupInvocations>=64&&properties.limits.maxComputeWorkGroupSize[0]>=8&&properties.limits.maxComputeWorkGroupSize[1]>=8,"canonical 8x8 workgroup unavailable");for(auto shape:{std::array<uint32_t,2>{full->eye_width,full->source_height},std::array<uint32_t,2>{full->width,full->height}})require((shape[0]+7)/8<=properties.limits.maxComputeWorkGroupCount[0]&&(shape[1]+7)/8<=properties.limits.maxComputeWorkGroupCount[1]&&properties.limits.maxComputeWorkGroupCount[2]>=2,"canonical dispatch limit exceeded");uint32_t family=0;require(daxa_dvc_get_vk_queue(d.get(),{DAXA_QUEUE_TYPE_MAIN,0},nullptr,&family)==DAXA_RESULT_SUCCESS,"canonical main queue missing");uint32_t count=0;vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,nullptr);std::vector<VkQueueFamilyProperties> families(count);vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,families.data());require(family<count&&families[family].timestampValidBits,"canonical timestamps unavailable");timestamp_bits=families[family].timestampValidBits;
        const char* entries[]={"pack_original","pack_rectified","coarse_box"};for(unsigned i=0;i<(coarse?3u:2u);++i){auto words=shader(entries[i]);pipelines[i]=d.create_compute_pipeline({.shader_info={.byte_code=words.data(),.byte_code_size=uint32_t(words.size())},.push_constant_size=sizeof(Push),.name=entries[i]});}
        for(unsigned index=0;index<cfg.slots;++index){auto s=std::make_shared<Slot>();s->full->calibration=full;if(coarse){s->coarse=std::make_shared<detail::DaxaFrameSlot>();s->coarse->calibration=coarse;}auto prefix="canonical_"+std::to_string(index)+"_";s->staging=allocate(upload_bytes,prefix+"staging",true);s->raw=allocate(upload_bytes,prefix+"pitched_rgb8_support");Parameters params{};for(unsigned i=0;i<6;++i){params.raw[i]=d.device_address(s->raw).value()+offsets[i];params.stride[i]=descriptors[i].stride_bytes;}for(unsigned i=0;i<8;++i){auto n=uint64_t(i<2?full->eye_width:full->width)*(i<2?full->source_height:full->height)*4;s->full->planes[i]=allocate(n,prefix+"full_"+std::to_string(i));params.full[i]=d.device_address(s->full->planes[i]).value();}if(coarse){s->coarse->planes[0]=s->full->planes[0];s->coarse->planes[1]=s->full->planes[1];for(unsigned i=2;i<8;++i){s->coarse->planes[i]=allocate(uint64_t(coarse->width)*coarse->height*4,prefix+"coarse_"+std::to_string(i));params.coarse[i-2]=d.device_address(s->coarse->planes[i]).value();}}params.eye_width=full->eye_width;params.source_height=full->source_height;params.width=full->width;params.height=full->height;s->params=allocate(sizeof(params),prefix+"immutable_params",true);std::memcpy(d.buffer_host_address(s->params).value(),&params,sizeof(params));auto address=d.device_address(s->params).value();s->queries=d.create_timeline_query_pool({.query_count=8,.name=prefix+"timings"});s->graph=std::make_unique<daxa::TaskGraph>(daxa::TaskGraphInfo{.device=d,.name=prefix+"canonical_pack_box"});auto& graph=*s->graph;
            std::vector<daxa::BufferId> ids{s->staging,s->raw,s->params};for(auto id:s->full->planes)ids.push_back(id);if(coarse)for(unsigned i=2;i<8;++i)ids.push_back(s->coarse->planes[i]);s->external.reserve(ids.size());std::vector<daxa::TaskBufferView> views;for(unsigned i=0;i<ids.size();++i){s->external.emplace_back(daxa::ExternalTaskBuffer({.buffer=ids[i],.name=prefix+std::to_string(i)}));views.push_back(graph.register_buffer(s->external.back()));}auto* slot=s.get();
            graph.add_task(daxa::InlineTask::Transfer("canonical coherent host writes").host.writes(views[0]).executes([](daxa::TaskInterface){}));
            graph.add_task(daxa::InlineTask::Transfer("canonical fixed pitched upload").reads(views[0]).writes(views[1]).executes([slot,source_view=views[0],destination_view=views[1],size=upload_bytes](daxa::TaskInterface ti){ti.recorder.reset_timestamps({.query_pool=slot->queries,.start_index=0,.count=8});ti.recorder.write_timestamp({.query_pool=slot->queries,.pipeline_stage=daxa::PipelineStageFlagBits::TOP_OF_PIPE,.query_index=0});ti.recorder.copy_buffer_to_buffer({.src_buffer=ti.get(source_view).id,.dst_buffer=ti.get(destination_view).id,.size=size});ti.recorder.write_timestamp({.query_pool=slot->queries,.pipeline_stage=daxa::PipelineStageFlagBits::TRANSFER,.query_index=1});}));
            graph.add_task(daxa::InlineTask::Compute("canonical original RGB8 pack").reads(views[1],views[2]).writes(views[3],views[4]).executes([this,slot,address](daxa::TaskInterface ti){dispatch(ti,slot,address,0,full->eye_width,full->source_height);}));
            graph.add_task(daxa::InlineTask::Compute("canonical rectified RGB8 gray15 support").reads(views[1],views[2]).writes(views[5],views[6],views[7],views[8],views[9],views[10]).executes([this,slot,address](daxa::TaskInterface ti){dispatch(ti,slot,address,1,full->width,full->height);}));
            if(coarse)graph.add_task(daxa::InlineTask::Compute("canonical coarse RGB box2x2 support AND4").reads(views[2],views[5],views[6],views[9],views[10]).writes(views[11],views[12],views[13],views[14],views[15],views[16]).executes([this,slot,address](daxa::TaskInterface ti){dispatch(ti,slot,address,2,coarse->width,coarse->height);}));graph.submit({});graph.complete({});slots.push_back(std::move(s));
        }
        require(resources.bytes==planned,"canonical allocation accounting differs from preflight");
    }
    void dispatch(daxa::TaskInterface ti,Slot* s,uint64_t address,unsigned pipeline,uint32_t width,uint32_t height){ti.recorder.write_timestamp({.query_pool=s->queries,.pipeline_stage=daxa::PipelineStageFlagBits::COMPUTE_SHADER,.query_index=2+pipeline*2});ti.recorder.set_pipeline(pipelines[pipeline]);ti.recorder.push_constant(Push{address});ti.recorder.dispatch({(width+7)/8,(height+7)/8,2});ti.recorder.write_timestamp({.query_pool=s->queries,.pipeline_stage=daxa::PipelineStageFlagBits::COMPUTE_SHADER,.query_index=3+pipeline*2});}
};
DaxaRectifiedInput::DaxaRectifiedInput(DaxaRectification f,std::optional<DaxaRectification> c,DaxaRectifiedInputConfig cfg,std::shared_ptr<DaxaFoundation> context,const RectifiedRgbFrame& prototype):impl_(std::make_shared<Impl>(std::move(f),std::move(c),cfg,std::move(context),prototype)){}
DaxaRectifiedInput::Frame::Frame(std::shared_ptr<Impl> owner,std::shared_ptr<void> slot,DaxaFrame full,DaxaFrame coarse):owner_(std::move(owner)),slot_(std::move(slot)),full_(std::move(full)),coarse_(std::move(coarse)){}
DaxaFrame DaxaRectifiedInput::Frame::full()const{require(bool(full_),"empty canonical frame");return full_;}
DaxaFrame DaxaRectifiedInput::Frame::coarse()const{require(bool(coarse_),"canonical coarse grid unavailable");return coarse_;}
size_t DaxaRectifiedInput::collect(){auto& e=*impl_;e.require_thread();return e.collect_at(e.context->device.oldest_pending_submit_index());}
DaxaRectifiedInput::Frame DaxaRectifiedInput::enqueue(const RectifiedRgbFrame& input){
    auto& e=*impl_;e.require_thread();require(input.admitted(),"offline-admitted canonical source required");validate_canonical_rgb_source(*input.source);auto const& s=*input.source->canonical_rgb;
    require(canonical_rgb_grid_association(*input.source,e.full->eye_width,e.full->source_height,e.full->width,e.full->height,e.full->original_identity,e.full->rectified_identity)&&s.rectified_calibration_identity==e.full->rectified_identity&&(!e.coarse||s.coarse_rectified_calibration_identity==e.coarse->rectified_identity),"canonical bound calibration mismatch");
    for(unsigned i=0;i<6;++i){auto const& p=i<4?s.rgb[i]:s.support[i-4];auto const& d=e.descriptors[i];require(p.width==d.width&&p.height==d.height&&p.stride_bytes==d.stride_bytes&&p.format==d.format&&p.bytes.size()>=e.extents[i],"canonical bound descriptor mismatch");}
    auto oldest=e.context->device.oldest_pending_submit_index();e.collect_at(oldest);
    for(auto const& slot:e.slots)if(e.available(slot,oldest)){
        // Retain source/backing before any task execution, including a partial
        // submission failure. Child/image leases independently prevent reuse.
        slot->full->source=input.source;slot->full->generation=e.next_generation++;if(slot->coarse){slot->coarse->source=input.source;slot->coarse->generation=slot->full->generation;}
        auto start=std::chrono::steady_clock::now();auto* target=reinterpret_cast<uint8_t*>(e.context->device.buffer_host_address(slot->staging).value());
        for(unsigned i=0;i<6;++i){auto const& p=i<4?s.rgb[i]:s.support[i-4];std::memcpy(target+e.offsets[i],p.bytes.data(),e.extents[i]);std::memset(target+e.offsets[i]+e.extents[i],0,((e.extents[i]+3)&~size_t{3})-e.extents[i]);}
        e.upload_cpu_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        try{slot->graph->execute({});}catch(...){slot->submit=e.context->device.latest_queue_submit_index(daxa::QUEUE_MAIN);slot->full->submit=slot->submit;if(slot->coarse)slot->coarse->submit=slot->submit;throw;}
        slot->submit=e.context->device.latest_queue_submit_index(daxa::QUEUE_MAIN);slot->full->submit=slot->submit;if(slot->coarse)slot->coarse->submit=slot->submit;
        return Frame(impl_,slot,DaxaFrame(impl_,slot->full),slot->coarse?DaxaFrame(impl_,slot->coarse):DaxaFrame{});
    }throw std::runtime_error("canonical producer backpressure: all slots retained/inflight");
}
uint64_t DaxaRectifiedInput::requested_bytes()const{return impl_->resources.bytes;}
double DaxaRectifiedInput::last_upload_cpu_ms()const{return impl_->upload_cpu_ms;}
std::shared_ptr<DaxaFoundation> DaxaRectifiedInput::context()const{return impl_->context;}
DaxaRectifiedInputTiming DaxaRectifiedInput::timings(const Frame& frame)const{auto& e=*impl_;e.require_thread();require(frame.owner_==impl_&&frame.slot_,"foreign canonical timing frame");require(frame.full_.ready(),"canonical timestamp completion required");auto slot=std::static_pointer_cast<Impl::Slot>(frame.slot_);unsigned count=e.coarse?8:6;std::array<uint64_t,16> values{};require(daxa_timeline_query_pool_query_results(slot->queries.get(),0,count,values.data())==DAXA_RESULT_SUCCESS,"canonical timing query failed");for(unsigned i=0;i<count;++i)require(values[i*2+1],"canonical timestamp unavailable");auto mask=e.timestamp_bits==64?UINT64_MAX:(uint64_t{1}<<e.timestamp_bits)-1;auto ms=[&](unsigned a,unsigned b){return double((values[b*2]-values[a*2])&mask)*e.timestamp_period/1e6;};return {ms(0,1),ms(2,3),ms(4,5),e.coarse?ms(6,7):0,ms(0,count-1),values[0],values[(count-1)*2],e.timestamp_period,e.timestamp_bits};}
DaxaPreprocessReadback DaxaRectifiedInput::readback(const DaxaFrame& frame)const{auto& e=*impl_;e.require_thread();frame.validate();require(frame.owner_.get()==impl_.get(),"foreign canonical readback frame");frame.wait();auto& d=e.context->device;DaxaPreprocessReadback out;for(unsigned p=0;p<8;++p){auto view=frame.plane(DaxaPreprocessPlane(p));size_t size=view.size_bytes();daxa::BufferInfo info{.size=size,.name="explicit canonical diagnostic readback"};auto memory=d.create_memory({.requirements=e.context->coherent_readback_requirements(info),.flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});auto host=d.create_buffer_from_memory_block({.buffer_info=info,.memory_block=memory});struct Cleanup{daxa::Device& d;daxa::BufferId b;~Cleanup(){d.destroy_buffer(b);d.collect_garbage();}}cleanup{d,host};auto recorder=d.create_command_recorder({.name="canonical diagnostic copy"});recorder.pipeline_barrier({.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::TRANSFER_READ});recorder.copy_buffer_to_buffer({.src_buffer=view.buffer(),.dst_buffer=host,.size=size});recorder.pipeline_barrier({.src_access=daxa::AccessConsts::TRANSFER_WRITE,.dst_access=daxa::AccessConsts::HOST_READ});auto command=recorder.complete_current_commands();d.submit_commands({.command_lists={&command,1}});d.wait_idle();out.planes[p].resize(size/4);std::memcpy(out.planes[p].data(),d.buffer_host_address(host).value(),size);}return out;}
std::string DaxaRectifiedInput::shader_directory(){return SF_DAXA_RECTIFIED_INPUT_SPIRV_DIRECTORY;}
std::string DaxaRectifiedInput::shader_identity(){return SF_DAXA_RECTIFIED_INPUT_SHADER_SHA256;}
}
