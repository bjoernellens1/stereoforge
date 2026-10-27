#include <stereoforge/geometry.hpp>
#include <stereoforge/detail/vulkan_validation.hpp>
#include <slang-rhi.h>
#include <slang-rhi/shader-cursor.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <unordered_map>

namespace stereoforge {
namespace {
using rhi::ComPtr;
void checked(SlangResult result,const char* operation) {
    if (SLANG_FAILED(result)) throw std::runtime_error(std::string(operation)+" failed ("+std::to_string(result)+")");
}
std::atomic<std::uint64_t> next_geometry_device{1ULL<<63};
void validate_camera(Pinhole c) {
    if (!std::isfinite(c.fx)||!std::isfinite(c.fy)||!std::isfinite(c.cx)||!std::isfinite(c.cy)||c.fx<=0||c.fy<=0)
        throw std::invalid_argument("positive finite rectified pinhole intrinsics required");
}
void validate_pose(const Pose4& p) {
    for (auto v:p) if (!std::isfinite(v)) throw std::invalid_argument("finite rigid pose required");
    if (std::abs(p[12])+std::abs(p[13])+std::abs(p[14])+std::abs(p[15]-1)>1e-5f)
        throw std::invalid_argument("affine rigid pose required");
    for (int i=0;i<3;++i) for(int j=0;j<3;++j) {
        float dot=0;for(int k=0;k<3;++k)dot+=p[k*4+i]*p[k*4+j];
        if(std::abs(dot-(i==j?1.f:0.f))>1e-4f)throw std::invalid_argument("orthonormal rigid pose required");
    }
    const float determinant=p[0]*(p[5]*p[10]-p[6]*p[9])-p[1]*(p[4]*p[10]-p[6]*p[8])+p[2]*(p[4]*p[9]-p[5]*p[8]);
    if(determinant<=0)throw std::invalid_argument("proper rigid rotation required");
}
std::int64_t add_age(std::int64_t age,std::int64_t elapsed) {
    if(age<0||elapsed<0||age>std::numeric_limits<std::int64_t>::max()-elapsed)
        throw std::invalid_argument("invalid or overflowing observation age");
    return age+elapsed;
}
}
struct GeometryMetadata {
    enum class Kind { Project,Reproject,Fuse } kind;
    std::vector<std::int64_t> current_ids,current_ages,previous_ids,previous_ages;
    std::int64_t elapsed=0;
};
struct NativeGeometry::Impl : std::enable_shared_from_this<NativeGeometry::Impl> {
    struct Slot {
        std::unordered_map<std::string,ComPtr<rhi::IBuffer>> buffers;
        ComPtr<rhi::ICommandBuffer> commands;
        std::weak_ptr<CompletionState> lease;
        std::uint64_t generation=0;
    };
    struct Lease final : CompletionState {
        std::shared_ptr<Impl> engine;
        std::uint64_t value;
        Lease(std::shared_ptr<Impl> owner,std::uint64_t v):CompletionState(owner->identity),engine(std::move(owner)),value(v){}
        bool ready() const override {std::uint64_t v=0;checked(engine->fence->getCurrentValue(&v),"query geometry completion");return v>=value;}
        void wait() override {rhi::IFence* fences[]={engine->fence};checked(engine->device->waitForFences(1,fences,&value,true,rhi::kTimeoutInfinite),"wait geometry completion");}
    };
    GeometryConfig config;
    std::size_t n,m,output_capacity;
    const std::uint64_t identity=next_geometry_device.fetch_add(1);
    ComPtr<rhi::IDevice> device;
    ComPtr<rhi::ICommandQueue> queue;
    ComPtr<rhi::IFence> fence;
    std::unordered_map<std::string,ComPtr<rhi::IComputePipeline>> pipelines;
    std::array<Slot,2> slots;
    std::uint64_t generation=0;
    Impl(GeometryConfig c,const std::string& shaders,bool validation):config(c) {
        n=std::uint64_t(c.source_width)*c.source_height;m=std::uint64_t(c.target_width)*c.target_height;
        if(!n||!m||n>UINT32_MAX/2||m>UINT32_MAX/2)throw std::invalid_argument("invalid native geometry dimensions");
        output_capacity=std::max(n,m);
        if(validation)detail::require_vulkan_validation();
        const std::string fusion=shaders+"/fusion",temporal=shaders+"/temporal";
        const char* paths[]={fusion.c_str(),temporal.c_str()};
        rhi::DeviceDesc desc{};desc.deviceType=rhi::DeviceType::Vulkan;desc.enableValidation=true;
        desc.enableCUDALaunchFromGfx=false;desc.enableRayTracing=false;
        desc.slang.searchPaths=paths;desc.slang.searchPathCount=2;desc.slang.targetProfile="sm_6_0";
        checked(rhi::getRHI()->createDevice(desc,device.writeRef()),"create geometry Vulkan device");
        checked(device->getQueue(rhi::QueueType::Graphics,queue.writeRef()),"get geometry queue");
        checked(device->createFence({},fence.writeRef()),"create geometry fence");
        auto session=device->getSlangSession();
        for(const char* module_name:{"project","history"}) {
            ComPtr<slang::IBlob> diagnostics;
            ComPtr<slang::IModule> module(session->loadModule(module_name,diagnostics.writeRef()));
            if(!module)throw std::runtime_error(diagnostics?static_cast<const char*>(diagnostics->getBufferPointer()):"geometry module failed");
            const std::vector<const char*> entry_names=std::strcmp(module_name,"project")==0
                ?std::vector<const char*>{"clear","clear_layers","splat_layer_depth","splat_layer_owner","histogram_ratios","reduce_ratios","resolve_layers"}
                :std::vector<const char*>{"clear","splat_depth","splat_owner","resolve","fuse"};
            for(const char* entry_name:entry_names) {
                ComPtr<slang::IEntryPoint> entry;checked(module->findEntryPointByName(entry_name,entry.writeRef()),"find geometry entry point");
                slang::IComponentType* entries[]={entry};rhi::ShaderProgramDesc program_desc{};
                program_desc.slangGlobalScope=module;program_desc.slangEntryPoints=entries;program_desc.slangEntryPointCount=1;
                ComPtr<rhi::IShaderProgram> program;
                auto result=device->createShaderProgram(program_desc,program.writeRef(),diagnostics.writeRef());
                if(SLANG_FAILED(result))throw std::runtime_error(diagnostics?static_cast<const char*>(diagnostics->getBufferPointer()):"geometry compile failed");
                rhi::ComputePipelineDesc pipeline_desc{};pipeline_desc.program=program;
                checked(device->createComputePipeline(pipeline_desc,pipelines[std::string(module_name)+"."+entry_name].writeRef()),"create geometry pipeline");
            }
        }
    }
    ~Impl(){if(queue)queue->waitOnHost();}
    Slot& acquire() {
        std::uint64_t completed=0;checked(fence->getCurrentValue(&completed),"query geometry slots");
        for(auto& slot:slots)if(slot.lease.expired()&&slot.generation<=completed){slot.commands=nullptr;return slot;}
        throw std::runtime_error("native geometry queue full: release completed frame leases");
    }
    rhi::IBuffer* buffer(Slot& slot,const std::string& name,std::size_t count,std::uint32_t stride=4) {
        if(!slot.buffers.contains(name)) {
            rhi::BufferDesc desc{};desc.size=count*stride;desc.elementSize=stride;desc.label=name.c_str();
            desc.usage=rhi::BufferUsage::ShaderResource|rhi::BufferUsage::UnorderedAccess|rhi::BufferUsage::CopySource|rhi::BufferUsage::CopyDestination;
            desc.defaultState=rhi::ResourceState::UnorderedAccess;
            checked(device->createBuffer(desc,nullptr,slot.buffers[name].writeRef()),"allocate persistent geometry buffer");
        }
        auto b=slot.buffers.at(name).get();
        if(b->getDesc().size<count*stride||b->getDesc().elementSize!=stride)throw std::runtime_error("geometry buffer layout mismatch");
        return b;
    }
    template<class T> void upload(rhi::ICommandEncoder* encoder,Slot& slot,const char* name,std::span<const T> values) {
        auto b=buffer(slot,name,values.size(),sizeof(T));
        checked(encoder->uploadBufferData(b,0,values.size_bytes(),values.data()),name);
    }
    void outputs(Slot& slot) {
        for(const char* name:{"zbuffer","owner","output_depth","output_confidence","output_disparity","output_valid","output_provenance","output_rgb","output_source","choose_previous","output_rgb_valid","output_approximate"})buffer(slot,name,output_capacity);
    }
    template<class F> void dispatch(rhi::ICommandEncoder* encoder,Slot& slot,const char* pipeline,std::size_t count,F parameters,std::uint32_t threads=64) {
        auto pass=encoder->beginComputePass();rhi::ShaderCursor cursor(pass->bindPipeline(pipelines.at(pipeline)));
        for(auto& [name,b]:slot.buffers){auto field=cursor[name.c_str()];if(field.isValid())checked(field.setBinding(rhi::Binding(b)),name.c_str());}
        parameters(cursor);pass->dispatchCompute(std::uint32_t((count+threads-1)/threads),1,1);pass->end();encoder->globalBarrier();
    }
    template<class T> static void set(rhi::ShaderCursor cursor,const char* name,const T& value) {
        auto field=cursor[name];if(field.isValid())checked(field.setData(value),name);
    }
    GeometryFrame submit(Slot& slot,rhi::ICommandEncoder* encoder,std::shared_ptr<GeometryMetadata> metadata,std::uint32_t w,std::uint32_t h) {
        checked(encoder->finish(slot.commands.writeRef()),"finish geometry commands");
        const auto value=++generation;rhi::ICommandBuffer* commands[]={slot.commands};rhi::IFence* fences[]={fence};
        rhi::SubmitDesc desc{};desc.commandBuffers=commands;desc.commandBufferCount=1;
        desc.signalFences=fences;desc.signalFenceValues=&value;desc.signalFenceCount=1;
        checked(queue->submit(desc),"submit geometry Vulkan work");slot.generation=value;
        auto lease=std::make_shared<Lease>(shared_from_this(),value);slot.lease=lease;CompletionToken token(lease);
        auto view=[&](const char* name,std::size_t elements,std::uint32_t stride=4){
            if(!slot.buffers.contains(name))return BufferView{};
            return BufferView{slot.buffers.at(name).get(),0,elements*stride,stride,identity,lease,token};
        };
        const auto count=std::size_t(w)*h;
        GeometryFrame frame;
        frame.depth=view("output_depth",count);frame.confidence=view("output_confidence",count);frame.disparity=view("output_disparity",count);
        frame.validity=view("output_valid",count);frame.provenance=view("output_provenance",count);
        frame.owner=view("owner",count);frame.choose_previous=view("choose_previous",count);
        frame.rgb=view("output_rgb",count);frame.source=view("output_source",count);
        if(metadata->kind==GeometryMetadata::Kind::Project){frame.previous_ids=view("source_ids",2*n,8);frame.previous_ages=view("source_ages",2*n,8);}
        else {
            frame.previous_ids=view("previous_ids",n,8);frame.previous_ages=view("previous_ages",n,8);
            if(metadata->kind==GeometryMetadata::Kind::Fuse){frame.current_ids=view("current_ids",n,8);frame.current_ages=view("current_ages",n,8);}
        }
        // Irrelevant outputs are absent, never stale buffers from a prior stage.
        if(metadata->kind==GeometryMetadata::Kind::Project){frame.disparity={};frame.choose_previous={};frame.rgb_validity=view("output_rgb_valid",count);frame.approximate=view("output_approximate",count);}
        else {frame.rgb={};frame.source={};if(metadata->kind==GeometryMetadata::Kind::Reproject)frame.choose_previous={};else frame.owner={};}
        frame.completion=token;frame.width=w;frame.height=h;frame.metadata=std::move(metadata);return frame;
    }
    void temporal_input(const TemporalInput& f) const {
        if(f.depth.size()!=n||f.confidence.size()!=n||f.disparity.size()!=n||f.validity.size()!=n||f.provenance.size()!=n||f.source_ids.size()!=n||f.age_ns.size()!=n)
            throw std::invalid_argument("temporal input dimensions mismatch");
        if(f.timestamp_ns<0||f.calibration_id.empty())throw std::invalid_argument("temporal timestamp and calibration identity required");
        for(std::size_t i=0;i<n;++i) {
            if(f.age_ns[i]<0)throw std::invalid_argument("negative observation age");
            if(f.validity[i]&&(!std::isfinite(f.depth[i])||f.depth[i]<=0||!std::isfinite(f.confidence[i])||f.confidence[i]<0||f.confidence[i]>1||f.source_ids[i]<0||f.provenance[i]==0))
                throw std::invalid_argument("invalid temporal observation");
        }
    }
    void upload_temporal(rhi::ICommandEncoder* encoder,Slot& slot,const char* prefix,const TemporalInput& f,std::span<const std::uint32_t> valid_override={}) {
        const auto name=[&](const char* suffix){return std::string(prefix)+suffix;};
        upload(encoder,slot,name("_depth").c_str(),f.depth);upload(encoder,slot,name("_confidence").c_str(),f.confidence);
        upload(encoder,slot,name("_disparity").c_str(),f.disparity);upload(encoder,slot,name("_valid").c_str(),valid_override.empty()?f.validity:valid_override);
        upload(encoder,slot,name("_provenance").c_str(),f.provenance);upload(encoder,slot,name("_ids").c_str(),f.source_ids);upload(encoder,slot,name("_ages").c_str(),f.age_ns);
    }
};
NativeGeometry::NativeGeometry(GeometryConfig config,const std::string& shaders,bool validation):impl_(std::make_shared<Impl>(config,shaders,validation)){}
std::uint64_t NativeGeometry::device_id()const{return impl_->identity;}
std::string NativeGeometry::device_name()const{return impl_->device->getInfo().adapterName;}
GeometryFrame NativeGeometry::project(const GeometrySources& src,const VirtualView& view) {
    auto& e=*impl_;const auto count=2*e.n;
    if(src.timestamp_ns<0||src.calibration_id.empty())throw std::invalid_argument("projection timestamp and calibration identity required");
    if(view.mode!=ViewMode::Strict&&view.mode!=ViewMode::Preview)throw std::invalid_argument("unsupported native view mode");
    if((view.mode==ViewMode::Preview&&src.preview_indices.size()!=2*e.m)||
        (!src.preview_indices.empty()&&src.preview_indices.size()!=2*e.m))
        throw std::invalid_argument("preview requires two target-sized calibration map layers");
    for(std::size_t i=0;i<src.preview_indices.size();++i){
        const auto index=src.preview_indices[i];
        if(index==0xffffffff)continue;
        const auto eye=i/e.m;
        if(index>=count||index/e.n!=eye)throw std::invalid_argument("preview source index is out of range or belongs to the wrong eye");
    }
    validate_camera(view.intrinsics);validate_pose(view.source_to_virtual_left);validate_pose(view.source_to_virtual_right);
    if(view.projection!=Projection::Perspective&&view.projection!=Projection::Cylindrical)throw std::invalid_argument("unsupported native projection");
    if(!std::isfinite(view.hfov_radians)||view.hfov_radians<=0||view.hfov_radians>=6.28318530718f)throw std::invalid_argument("invalid virtual horizontal FOV");
    if(src.depth.size()!=count||src.confidence.size()!=count||src.rgb.size()!=count||src.provenance.size()!=count||src.rays.size()!=count||src.source_ids.size()!=count||src.age_ns.size()!=count)
        throw std::invalid_argument("geometry source dimensions mismatch");
    for(std::size_t i=0;i<count;++i) {
        const auto ray=src.rays[i];
        if(!std::isfinite(ray.x)||!std::isfinite(ray.y)||ray.z!=1||src.age_ns[i]<0)throw std::invalid_argument("unit-Z finite rays and nonnegative ages required");
        if(std::isfinite(src.depth[i])&&src.depth[i]>0&&(!std::isfinite(src.confidence[i])||src.confidence[i]<0||src.confidence[i]>1||src.provenance[i]==0||src.source_ids[i]<0))
            throw std::invalid_argument("invalid source confidence, identity, or provenance");
    }
    auto& slot=e.acquire();auto encoder=e.queue->createCommandEncoder();if(!encoder)throw std::runtime_error("geometry command encoder failed");
    e.upload(encoder,slot,"source_depth",src.depth);e.upload(encoder,slot,"source_confidence",src.confidence);
    e.upload(encoder,slot,"source_rgb",src.rgb);e.upload(encoder,slot,"source_provenance",src.provenance);
    e.upload(encoder,slot,"source_rays",src.rays);e.upload(encoder,slot,"source_ids",src.source_ids);e.upload(encoder,slot,"source_ages",src.age_ns);
    e.outputs(slot);
    for(const char* name:{"layer_z","layer_owner","preview_indices"})e.buffer(slot,name,2*e.m);
    e.buffer(slot,"ratio_histogram",3*4096);e.buffer(slot,"exposure_gain",3);
    if(!src.preview_indices.empty())e.upload(encoder,slot,"preview_indices",src.preview_indices);
    auto parameters=[&](rhi::ShaderCursor cursor){
        e.set(cursor,"src_width",e.config.source_width);e.set(cursor,"src_height",e.config.source_height);
        e.set(cursor,"target_width",e.config.target_width);e.set(cursor,"target_height",e.config.target_height);
        e.set(cursor,"projection",std::uint32_t(view.projection==Projection::Cylindrical));
        e.set(cursor,"fx",view.intrinsics.fx);e.set(cursor,"fy",view.intrinsics.fy);e.set(cursor,"cx",view.intrinsics.cx);e.set(cursor,"cy",view.intrinsics.cy);
        e.set(cursor,"hfov_radians",view.hfov_radians);e.set(cursor,"source_to_virtual_left",view.source_to_virtual_left);e.set(cursor,"source_to_virtual_right",view.source_to_virtual_right);
        e.set(cursor,"preview_enabled",std::uint32_t(view.mode==ViewMode::Preview));
        e.set(cursor,"exposure_enabled",std::uint32_t(view.exposure_match));
    };
    e.dispatch(encoder,slot,"project.clear",e.m,parameters);
    e.dispatch(encoder,slot,"project.clear_layers",std::max(2*e.m,std::size_t(3*4096)),parameters);
    e.dispatch(encoder,slot,"project.splat_layer_depth",count,parameters);
    e.dispatch(encoder,slot,"project.splat_layer_owner",count,parameters);
    e.dispatch(encoder,slot,"project.histogram_ratios",e.m,parameters);
    // reduce_ratios has numthreads(1,1,1), unlike the 64-thread projection
    // passes. Three groups are required so every color channel is reduced.
    e.dispatch(encoder,slot,"project.reduce_ratios",3,parameters,1);
    e.dispatch(encoder,slot,"project.resolve_layers",e.m,parameters);
    auto metadata=std::make_shared<GeometryMetadata>();metadata->kind=GeometryMetadata::Kind::Project;
    metadata->previous_ids={src.source_ids.begin(),src.source_ids.end()};metadata->previous_ages={src.age_ns.begin(),src.age_ns.end()};
    auto frame=e.submit(slot,encoder,std::move(metadata),e.config.target_width,e.config.target_height);
    frame.operation=GeometryOperation::Project;
    frame.depth_convention=view.projection==Projection::Perspective?DepthConvention::AxialZ:DepthConvention::RayRange;
    frame.timestamp_ns=src.timestamp_ns;frame.calibration_id=src.calibration_id;return frame;
}
GeometryFrame NativeGeometry::reproject(const TemporalInput& previous,Pinhole camera,const Pose4& pose,std::int64_t timestamp,std::optional<MetricDisparity> metric) {
    auto& e=*impl_;e.temporal_input(previous);validate_camera(camera);validate_pose(pose);
    if(timestamp<previous.timestamp_ns)throw std::invalid_argument("temporal timestamp moves backwards");
    if(metric&&(!std::isfinite(metric->fx)||metric->fx<=0||!std::isfinite(metric->baseline_m)||metric->baseline_m<=0||!std::isfinite(metric->principal_offset)))throw std::invalid_argument("invalid metric disparity calibration");
    const auto elapsed=timestamp-previous.timestamp_ns;for(auto age:previous.age_ns)add_age(age,elapsed);
    auto& slot=e.acquire();auto encoder=e.queue->createCommandEncoder();if(!encoder)throw std::runtime_error("geometry command encoder failed");
    e.upload_temporal(encoder,slot,"previous",previous);e.outputs(slot);
    auto parameters=[&](rhi::ShaderCursor cursor){
        e.set(cursor,"width",e.config.source_width);e.set(cursor,"height",e.config.source_height);
        e.set(cursor,"fx",camera.fx);e.set(cursor,"fy",camera.fy);e.set(cursor,"cx",camera.cx);e.set(cursor,"cy",camera.cy);e.set(cursor,"pose",pose);
        e.set(cursor,"metric_disparity",std::uint32_t(metric.has_value()));e.set(cursor,"disparity_fx",metric?metric->fx:0.f);
        e.set(cursor,"baseline_m",metric?metric->baseline_m:0.f);e.set(cursor,"principal_offset",metric?metric->principal_offset:0.f);
    };
    for(const char* stage:{"history.clear","history.splat_depth","history.splat_owner","history.resolve"})e.dispatch(encoder,slot,stage,e.n,parameters);
    auto metadata=std::make_shared<GeometryMetadata>();metadata->kind=GeometryMetadata::Kind::Reproject;metadata->elapsed=elapsed;
    metadata->previous_ids={previous.source_ids.begin(),previous.source_ids.end()};metadata->previous_ages={previous.age_ns.begin(),previous.age_ns.end()};
    auto frame=e.submit(slot,encoder,std::move(metadata),e.config.source_width,e.config.source_height);
    frame.operation=GeometryOperation::Reproject;frame.timestamp_ns=timestamp;frame.calibration_id=previous.calibration_id;
    frame.observation_elapsed_ns=elapsed;return frame;
}
GeometryFrame NativeGeometry::fuse(const TemporalInput& current,const TemporalInput& prediction,std::int64_t max_age,float tolerance) {
    auto& e=*impl_;e.temporal_input(current);e.temporal_input(prediction);
    if(max_age<0||!std::isfinite(tolerance)||tolerance<0)throw std::invalid_argument("invalid temporal fusion limits");
    const bool compatible=prediction.calibration_id==current.calibration_id&&prediction.timestamp_ns<=current.timestamp_ns;
    const auto elapsed=compatible?current.timestamp_ns-prediction.timestamp_ns:0;
    std::vector<std::uint32_t> eligible(e.n,0);
    if(compatible)for(std::size_t i=0;i<e.n;++i)eligible[i]=prediction.validity[i]&&add_age(prediction.age_ns[i],elapsed)<=max_age;
    auto& slot=e.acquire();auto encoder=e.queue->createCommandEncoder();if(!encoder)throw std::runtime_error("geometry command encoder failed");
    e.upload_temporal(encoder,slot,"current",current);e.upload_temporal(encoder,slot,"previous",prediction,eligible);e.outputs(slot);
    auto parameters=[&](rhi::ShaderCursor cursor){e.set(cursor,"width",e.config.source_width);e.set(cursor,"height",e.config.source_height);e.set(cursor,"tolerance",tolerance);};
    e.dispatch(encoder,slot,"history.fuse",e.n,parameters);
    auto metadata=std::make_shared<GeometryMetadata>();metadata->kind=GeometryMetadata::Kind::Fuse;metadata->elapsed=elapsed;
    metadata->current_ids={current.source_ids.begin(),current.source_ids.end()};metadata->current_ages={current.age_ns.begin(),current.age_ns.end()};
    metadata->previous_ids={prediction.source_ids.begin(),prediction.source_ids.end()};metadata->previous_ages={prediction.age_ns.begin(),prediction.age_ns.end()};
    auto frame=e.submit(slot,encoder,std::move(metadata),e.config.source_width,e.config.source_height);
    frame.operation=GeometryOperation::Fuse;frame.timestamp_ns=current.timestamp_ns;frame.calibration_id=current.calibration_id;
    frame.observation_elapsed_ns=elapsed;return frame;
}
GeometryReadback NativeGeometry::readback(const GeometryFrame& frame)const {
    auto& e=*impl_;frame.depth.validate(device_id());frame.validity.validate(device_id());
    if(!frame.metadata)throw std::invalid_argument("geometry frame metadata missing");frame.completion.wait();
    const auto count=std::size_t(frame.width)*frame.height;GeometryReadback result;
    auto read=[&]<class T>(const BufferView& view,std::vector<T>& target){if(!view.native)return;view.validate(device_id());if(view.size_bytes!=count*sizeof(T))throw std::invalid_argument("geometry readback dimensions mismatch");target.resize(count);checked(e.device->readBuffer(static_cast<rhi::IBuffer*>(view.native),0,view.size_bytes,target.data()),"read geometry buffer");};
    read(frame.depth,result.depth);read(frame.confidence,result.confidence);read(frame.disparity,result.disparity);
    read(frame.validity,result.validity);read(frame.provenance,result.provenance);read(frame.rgb,result.rgb);read(frame.source,result.source);
    read(frame.owner,result.owner);read(frame.choose_previous,result.choose_previous);
    read(frame.rgb_validity,result.rgb_validity);read(frame.approximate,result.approximate);
    result.source_ids.assign(count,-1);result.age_ns.assign(count,0);const auto& metadata=*frame.metadata;
    for(std::size_t i=0;i<count;++i) {
        if(metadata.kind==GeometryMetadata::Kind::Fuse) {
            const bool previous=result.choose_previous[i]!=0;
            result.source_ids[i]=previous?metadata.previous_ids[i]:metadata.current_ids[i];
            result.age_ns[i]=previous?add_age(metadata.previous_ages[i],metadata.elapsed):metadata.current_ages[i];
        } else if(result.validity[i]) {
            const auto source=result.owner[i];if(source>=metadata.previous_ids.size())throw std::runtime_error("invalid GPU source owner");
            result.source_ids[i]=metadata.previous_ids[source];result.age_ns[i]=add_age(metadata.previous_ages[source],metadata.elapsed);
        }
    }
    return result;
}
} // namespace stereoforge
