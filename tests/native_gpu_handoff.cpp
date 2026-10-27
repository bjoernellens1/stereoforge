// Producer -> bridge -> compositor -> GPU consumer on one Vulkan context/queue.
// Only a sixteen-byte counter readback follows each complete chain; no depth,
// color, validity, identity or age plane ever returns to the CPU in the chain.
#include <stereoforge/native_device.hpp>
#include <stereoforge/runtime.hpp>
#include <stereoforge/geometry.hpp>
#include <slang-rhi.h>
#include <slang-rhi/shader-cursor.h>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
using namespace stereoforge;
using rhi::ComPtr;
namespace {
void checked(SlangResult result,const char* detail){if(SLANG_FAILED(result))throw std::runtime_error(detail);}
void require(bool ok,const char* detail){if(!ok)throw std::runtime_error(detail);}
rhi::IBuffer* buffer(const BufferView& view){return static_cast<rhi::IBuffer*>(view.native);}
struct ProducerOwner {
    NativeFrame stereo;
    GeometryDeviceSources initial;
    ComPtr<rhi::ICommandBuffer> command;
};
struct BridgeCompletion final:CompletionState {
    NativeDevice shared;
    ComPtr<rhi::IFence> fence;
    std::shared_ptr<ProducerOwner> owner;
    BridgeCompletion(NativeDevice d,ComPtr<rhi::IFence> f,std::shared_ptr<ProducerOwner> o):CompletionState(d.id()),shared(d),fence(f),owner(std::move(o)){}
    bool ready()const override{std::uint64_t value=0;checked(fence->getCurrentValue(&value),"query bridge completion");return value>=1;}
    void wait()override{rhi::IFence* f[]={fence};std::uint64_t v=1;checked(shared.rhi_device()->waitForFences(1,f,&v,true,rhi::kTimeoutInfinite),"wait bridge completion");}
};
void submit(rhi::ICommandQueue* queue,rhi::ICommandBuffer* command,rhi::IFence* fence){
    rhi::ICommandBuffer* commands[]={command};rhi::IFence* fences[]={fence};std::uint64_t value=1;
    rhi::SubmitDesc desc{};desc.commandBuffers=commands;desc.commandBufferCount=1;
    desc.signalFences=fences;desc.signalFenceValues=&value;desc.signalFenceCount=1;
    checked(queue->submit(desc),"submit handoff command");
}
ComPtr<rhi::IComputePipeline> pipeline(rhi::IDevice* device,slang::IModule* module,const char* name){
    ComPtr<slang::IEntryPoint> entry;checked(module->findEntryPointByName(name,entry.writeRef()),"handoff entry point");
    slang::IComponentType* entries[]={entry};rhi::ShaderProgramDesc program_desc{};
    program_desc.slangGlobalScope=module;program_desc.slangEntryPoints=entries;program_desc.slangEntryPointCount=1;
    ComPtr<rhi::IShaderProgram> program;ComPtr<slang::IBlob> diagnostics;
    auto result=device->createShaderProgram(program_desc,program.writeRef(),diagnostics.writeRef());
    if(SLANG_FAILED(result))throw std::runtime_error(diagnostics?static_cast<const char*>(diagnostics->getBufferPointer()):"handoff shader program");
    rhi::ComputePipelineDesc desc{};desc.program=program;ComPtr<rhi::IComputePipeline> out;
    checked(device->createComputePipeline(desc,out.writeRef()),"handoff pipeline");return out;
}
void bind(rhi::ShaderCursor cursor,const char* name,rhi::IBuffer* value){checked(cursor[name].setBinding(rhi::Binding(value)),name);}
template<class T>void set(rhi::ShaderCursor cursor,const char* name,T value){checked(cursor[name].setData(value),name);}
constexpr const char* shader=R"(
StructuredBuffer<float> stereo_depth;
StructuredBuffer<float> stereo_confidence;
StructuredBuffer<uint> stereo_valid;
RWStructuredBuffer<float> source_depth;
RWStructuredBuffer<float> source_confidence;
RWStructuredBuffer<uint> source_rgb;
RWStructuredBuffer<uint> source_provenance;
RWStructuredBuffer<uint2> source_ids;
RWStructuredBuffer<uint2> source_ages;
uniform uint count;
uniform uint frame_index;
uniform uint expected_rgb;
[shader("compute")][numthreads(64,1,1)]
void bridge(uint3 tid:SV_DispatchThreadID){
 uint i=tid.x;if(i>=2*count)return;
 bool valid=false;
 if(i<count)valid=stereo_valid[i]!=0&&isfinite(stereo_depth[i])&&stereo_depth[i]>0;
 source_depth[i]=valid?stereo_depth[i]:asfloat(0x7fc00000);
 source_confidence[i]=valid?stereo_confidence[i]:0;
 source_rgb[i]=expected_rgb;
 source_provenance[i]=valid?1:0;
 source_ids[i]=valid?uint2(frame_index,256):uint2(0xffffffff);
 source_ages[i]=uint2(0);
}
StructuredBuffer<float> final_depth;
StructuredBuffer<float> final_confidence;
StructuredBuffer<uint> final_valid;
StructuredBuffer<uint> final_rgb_valid;
StructuredBuffer<uint> final_rgb;
StructuredBuffer<uint> final_source;
StructuredBuffer<uint> final_provenance;
StructuredBuffer<uint> final_approximate;
StructuredBuffer<uint2> final_ids;
StructuredBuffer<uint2> final_ages;
RWStructuredBuffer<uint> counters;
[shader("compute")][numthreads(64,1,1)]
void consume(uint3 tid:SV_DispatchThreadID){
 uint i=tid.x;if(i>=count)return;
 bool valid=final_valid[i]!=0;
 uint ignored;InterlockedAdd(counters[valid?0:1],1,ignored);
 bool correct=all(final_ages[i]==uint2(0));
 if(valid){
  correct=correct&&all(final_ids[i]==uint2(frame_index,256))&&final_source[i]==1&&final_provenance[i]==1&&
   final_rgb[i]==expected_rgb&&final_rgb_valid[i]==1&&final_approximate[i]==0&&
   isfinite(final_depth[i])&&final_depth[i]>0&&isfinite(final_confidence[i])&&final_confidence[i]>=0&&final_confidence[i]<=1;
  if(abs(final_depth[i]-1.8)<.0001)InterlockedAdd(counters[3],1,ignored);
 }else correct=correct&&all(final_ids[i]==uint2(0xffffffff));
 if(!correct)InterlockedAdd(counters[2],1,ignored);
}
)";
}
int main(int argc,char**){try{
    constexpr std::uint32_t width=64,height=24,n=width*height;
    NativeDevice shared(argc>1);
    NativeStereo stereo({width,height,8,4,8,32,false},STEREOFORGE_STEREO_SHADER_DIRECTORY,shared,MetricCalibration{60,.12f,0});
    NativeGeometry geometry({width,height,width,height},STEREOFORGE_SHADER_DIRECTORY,shared);
    require(stereo.device_id()==shared.id()&&geometry.device_id()==shared.id(),"engines do not share context identity");
    auto device=shared.rhi_device();auto queue=shared.rhi_queue();
    ComPtr<slang::IBlob> diagnostics;
    ComPtr<slang::IModule> module(device->getSlangSession()->loadModuleFromSourceString("native_handoff","native_handoff.slang",shader,diagnostics.writeRef()));
    if(!module)throw std::runtime_error(diagnostics?static_cast<const char*>(diagnostics->getBufferPointer()):"handoff module");
    auto bridge=pipeline(device,module,"bridge"),consumer=pipeline(device,module,"consume");
    std::vector<float> z(2*n,NAN),confidence(2*n,0);
    std::vector<std::uint32_t> rgb(2*n,0),provenance(2*n,0),preview(2*n,0xffffffff);
    std::vector<std::int64_t> ids(2*n,-1),ages(2*n,0);std::vector<Ray3> rays(2*n);
    for(unsigned eye=0;eye<2;++eye)for(unsigned y=0;y<height;++y)for(unsigned x=0;x<width;++x)rays[eye*n+y*width+x]={(x-31.5f)/60,(y-11.5f)/60,1};
    GeometrySources host{z,confidence,rgb,provenance,rays,ids,ages,0,"synthetic-parallel",preview};
    const auto initial=geometry.upload_sources(host);initial.completion.wait(); // Setup only, before all chains.
    std::vector<std::uint32_t> left(n),right(n);std::uint32_t random=42;
    for(auto& value:left){random=1664525u*random+1013904223u;value=random>>24;}
    for(unsigned y=0;y<height;++y)for(unsigned x=0;x<width-4;++x)right[y*width+x]=left[y*width+x+4];
    rhi::BufferDesc desc{};desc.size=4*sizeof(std::uint32_t);desc.elementSize=4;
    desc.usage=rhi::BufferUsage::ShaderResource|rhi::BufferUsage::UnorderedAccess|rhi::BufferUsage::CopySource|rhi::BufferUsage::CopyDestination;
    desc.defaultState=rhi::ResourceState::UnorderedAccess;
    ComPtr<rhi::IBuffer> counts;checked(device->createBuffer(desc,nullptr,counts.writeRef()),"create handoff counters");
    VirtualView view;view.intrinsics={60,60,31.5f,11.5f};view.hfov_radians=2*std::atan(width/120.f);view.exposure_match=false;
    for(std::uint32_t frame_index=1;frame_index<=3;++frame_index){
        auto owner=std::make_shared<ProducerOwner>();owner->initial=initial;
        owner->stereo=stereo.enqueue(left,right);
        auto source=initial;source.timestamp_ns=frame_index;
        require(owner->stereo.depth.device_id==shared.id(),"stereo depth not on shared device");
        const std::uint32_t color=(20+frame_index)|((40+2*frame_index)<<8)|((60+3*frame_index)<<16);
        auto encoder=queue->createCommandEncoder();encoder->globalBarrier();auto pass=encoder->beginComputePass();rhi::ShaderCursor c(pass->bindPipeline(bridge));
        bind(c,"stereo_depth",buffer(owner->stereo.depth));bind(c,"stereo_confidence",buffer(owner->stereo.confidence));bind(c,"stereo_valid",buffer(owner->stereo.depth_validity));
        bind(c,"source_depth",buffer(source.depth));bind(c,"source_confidence",buffer(source.confidence));bind(c,"source_rgb",buffer(source.rgb));bind(c,"source_provenance",buffer(source.provenance));
        bind(c,"source_ids",buffer(source.source_ids));bind(c,"source_ages",buffer(source.age_ns));
        set(c,"count",n);set(c,"frame_index",frame_index);set(c,"expected_rgb",color);
        pass->dispatchCompute((2*n+63)/64,1,1);pass->end();encoder->globalBarrier();
        checked(encoder->finish(owner->command.writeRef()),"finish bridge");ComPtr<rhi::IFence> bridge_fence;checked(device->createFence({},bridge_fence.writeRef()),"create bridge fence");
        submit(queue,owner->command,bridge_fence);
        CompletionToken bridge_done(std::make_shared<BridgeCompletion>(shared,bridge_fence,owner));
        for(auto* value:{&source.depth,&source.confidence,&source.rgb,&source.provenance,&source.rays,&source.source_ids,&source.age_ns,&source.preview_indices})value->completion=bridge_done;
        source.completion=bridge_done;
        auto result=geometry.project(source,view);
        require(result.depth.device_id==shared.id()&&result.source_ids.device_id==shared.id(),"view outputs not on producer context");
        auto consumer_encoder=queue->createCommandEncoder();consumer_encoder->globalBarrier();consumer_encoder->clearBuffer(counts);consumer_encoder->globalBarrier();
        auto consume_pass=consumer_encoder->beginComputePass();rhi::ShaderCursor output(consume_pass->bindPipeline(consumer));
        bind(output,"final_depth",buffer(result.depth));bind(output,"final_confidence",buffer(result.confidence));bind(output,"final_valid",buffer(result.validity));
        bind(output,"final_rgb_valid",buffer(result.rgb_validity));bind(output,"final_rgb",buffer(result.rgb));bind(output,"final_source",buffer(result.source));
        bind(output,"final_provenance",buffer(result.provenance));bind(output,"final_approximate",buffer(result.approximate));
        bind(output,"final_ids",buffer(result.source_ids));bind(output,"final_ages",buffer(result.age_ns));bind(output,"counters",counts);
        set(output,"count",n);set(output,"frame_index",frame_index);set(output,"expected_rgb",color);
        consume_pass->dispatchCompute((n+63)/64,1,1);consume_pass->end();consumer_encoder->globalBarrier();
        ComPtr<rhi::ICommandBuffer> consume_command;checked(consumer_encoder->finish(consume_command.writeRef()),"finish consumer");
        ComPtr<rhi::IFence> consume_fence;checked(device->createFence({},consume_fence.writeRef()),"create consumer fence");submit(queue,consume_command,consume_fence);
        // Drop every high-level producer/result snapshot after submission. RHI
        // consumer bindings/queued commands retain the actual GPU resources.
        // Attempt producer recycling on the same queue before waiting: later
        // writes remain ordered after this consumer. Fence-based backpressure
        // is permitted when neither stereo slot has completed yet.
        result={};source={};bridge_done={};owner.reset();
        unsigned reuse_submissions=0,reuse_backpressure=0;
        std::vector<NativeFrame> recycled_frames;
        for(unsigned attempt=0;attempt<2;++attempt){
            try{recycled_frames.push_back(stereo.enqueue(left,right));++reuse_submissions;}
            catch(const std::runtime_error& error){
                if(std::string(error.what()).find("queue full")==std::string::npos)throw;
                ++reuse_backpressure;
            }
        }
        // The first CPU wait in this frame's producer/bridge/view/consumer chain.
        std::uint64_t one=1;rhi::IFence* fence_array[]={consume_fence};checked(device->waitForFences(1,fence_array,&one,true,rhi::kTimeoutInfinite),"wait final consumer");
        std::array<std::uint32_t,4> values{};checked(device->readBuffer(counts,0,sizeof(values),values.data()),"read sixteen-byte counters");
        require(values[0]>100&&values[1]>0&&values[0]+values[1]==n,"missing measured or invalid support");
        require(values[2]==0&&values[3]>100,"stale identity/color, invalid metadata or missing known metric depth");
        // Cleanup later reuse submissions only after the measured chain and tiny
        // counter download are complete; leave both producer slots ready next round.
        for(auto& recycled:recycled_frames)recycled.completion.wait();
        std::cout<<"frame="<<frame_index<<" measured="<<values[0]<<" invalid="<<values[1]<<" errors="<<values[2]<<" known_depth="<<values[3]<<" reuse_submissions="<<reuse_submissions<<" reuse_backpressure="<<reuse_backpressure<<'\n';
    }
    std::cout<<"shared-device GPU stereo-to-view-to-consumer handoff passed; synthetic correctness only\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
