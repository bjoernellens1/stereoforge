#include <stereoforge/runtime.hpp>
#include <slang-rhi.h>
#include <stereoforge/detail/vulkan_validation.hpp>
#include <slang-rhi/shader-cursor.h>
#include <array>
#include <atomic>
#include <iostream>
#include <unordered_map>

namespace stereoforge {
namespace {
void check(SlangResult result, const char* operation) {
    if (SLANG_FAILED(result)) throw std::runtime_error(std::string(operation) + " failed (" + std::to_string(result) + ")");
}
using rhi::ComPtr;
std::atomic<std::uint64_t> next_device{1};
}
struct NativeStereo::Impl : std::enable_shared_from_this<NativeStereo::Impl> {
    struct Slot {
        std::unordered_map<std::string, ComPtr<rhi::IBuffer>> buffers;
        ComPtr<rhi::ICommandBuffer> commands;
        std::weak_ptr<CompletionState> lease;
        std::uint64_t fence_value = 0;
    };
    struct Lease final : CompletionState {
        std::shared_ptr<Impl> engine;
        std::uint64_t value;
        Lease(std::shared_ptr<Impl> owner, std::uint64_t generation)
            : CompletionState(owner->identity), engine(std::move(owner)), value(generation) {}
        bool ready() const override {
            std::uint64_t completed = 0;
            check(engine->fence->getCurrentValue(&completed), "query completion fence");
            return completed >= value;
        }
        void wait() override {
            rhi::IFence* fences[] = {engine->fence};
            check(engine->device->waitForFences(1, fences, &value, true, rhi::kTimeoutInfinite), "wait completion");
        }
    };
    StereoConfig config;
    std::optional<MetricCalibration> metric;
    const std::uint64_t identity = next_device.fetch_add(1);
    ComPtr<rhi::IDevice> device;
    ComPtr<rhi::ICommandQueue> queue;
    ComPtr<rhi::IFence> fence;
    std::unordered_map<std::string, ComPtr<rhi::IComputePipeline>> pipelines;
    std::array<Slot, 2> slots;
    std::uint64_t generation = 0;
    Impl(StereoConfig c, const std::string& shader_directory, bool validation, std::optional<MetricCalibration> calibration) : config(c),metric(calibration) {
        config.validate();
        if(metric)metric->validate();
        if (validation) detail::require_vulkan_validation();
        const char* search[] = {shader_directory.c_str()};
        rhi::DeviceDesc desc{};
        desc.deviceType = rhi::DeviceType::Vulkan;
        desc.enableValidation = true;
        desc.enableCUDALaunchFromGfx = false;
        desc.enableRayTracing = false;
        desc.slang.searchPaths = search;
        desc.slang.searchPathCount = 1;
        desc.slang.targetProfile = "sm_6_0";
        check(rhi::getRHI()->createDevice(desc, device.writeRef()), "create Vulkan device");
        check(device->getQueue(rhi::QueueType::Graphics, queue.writeRef()), "create Vulkan queue");
        check(device->createFence({}, fence.writeRef()), "create completion fence");
        auto session = device->getSlangSession();
        ComPtr<slang::IBlob> diagnostics;
        ComPtr<slang::IModule> module(session->loadModule("sgm", diagnostics.writeRef()));
        if (!module) throw std::runtime_error(diagnostics ? static_cast<const char*>(diagnostics->getBufferPointer()) : "load shared sgm module failed");
        for (const char* name : {"census", "cost", "aggregate", "select", "depth_convert"}) {
            if(std::string(name)=="depth_convert"&&!metric)continue;
            ComPtr<slang::IEntryPoint> entry;
            check(module->findEntryPointByName(name, entry.writeRef()), "find shared entry point");
            slang::IComponentType* entries[] = {entry};
            rhi::ShaderProgramDesc pd{};
            pd.slangGlobalScope = module;
            pd.slangEntryPoints = entries;
            pd.slangEntryPointCount = 1;
            ComPtr<rhi::IShaderProgram> program;
            auto result = device->createShaderProgram(pd, program.writeRef(), diagnostics.writeRef());
            if (SLANG_FAILED(result) && diagnostics)
                std::cerr << static_cast<const char*>(diagnostics->getBufferPointer()) << '\n';
            check(result, "compile shared shader");
            rhi::ComputePipelineDesc pipeline{};
            pipeline.program = program;
            check(device->createComputePipeline(pipeline, pipelines[name].writeRef()), "create compute pipeline");
        }
        const std::uint64_t pixels = std::uint64_t(config.width) * config.height;
        for (auto& slot : slots) {
            for (const char* name : {"left", "right", "left_desc", "right_desc", "left_valid", "right_valid", "disparity", "confidence", "validity", "costs", "sums", "metric_depth_output", "metric_validity_output"}) {
                if((std::string(name)=="metric_depth_output"||std::string(name)=="metric_validity_output")&&!metric)continue;
                rhi::BufferDesc b{};
                const bool volume = std::string(name) == "costs" || std::string(name) == "sums";
                b.size = pixels * (volume ? config.disparities : 1) * 4;
                b.elementSize = 4;
                b.usage = rhi::BufferUsage::ShaderResource | rhi::BufferUsage::UnorderedAccess |
                          rhi::BufferUsage::CopySource | rhi::BufferUsage::CopyDestination;
                b.defaultState = rhi::ResourceState::UnorderedAccess;
                b.label = name;
                check(device->createBuffer(b, nullptr, slot.buffers[name].writeRef()), "allocate persistent buffer");
            }
        }
    }
    ~Impl() { if (queue) queue->waitOnHost(); }
    void dispatch(rhi::ICommandEncoder* encoder, Slot& slot, const char* stage,
                  std::uint32_t groups, bool right = false, int dx = 0, int dy = 0, std::uint32_t lines = 0) {
        auto pass = encoder->beginComputePass();
        rhi::ShaderCursor root(pass->bindPipeline(pipelines.at(stage)));
        auto bind = [&](const char* parameter, const char* buffer) {
            auto field = root[parameter];
            if (field.isValid()) check(field.setBinding(rhi::Binding(slot.buffers.at(buffer))), parameter);
        };
        bind("image", right ? "right" : "left");
        bind("descriptor", right ? "right_desc" : "left_desc");
        bind("descriptor_valid", right ? "right_valid" : "left_valid");
        for (const char* name : {"left_desc", "right_desc", "left_valid", "right_valid", "costs", "sums", "disparity", "confidence", "validity"}) bind(name, name);
        bind("lower", "left_valid"); bind("upper", "right_valid");
        if(metric){bind("metric_depth_output","metric_depth_output");bind("metric_validity_output","metric_validity_output");}
        auto scalar = [&](const char* name, auto value) {
            auto field = root[name];
            if (field.isValid()) check(field.setData(value), name);
        };
        scalar("width", config.width); scalar("height", config.height);
        scalar("disparity_count", config.disparities); scalar("path_count", config.paths);
        scalar("p1", config.p1); scalar("p2", config.p2);
        scalar("bounded", 0u); scalar("subpixel", std::uint32_t(config.subpixel));
        scalar("dx", dx); scalar("dy", dy); scalar("line_count", lines);
        if(metric){scalar("metric_fx",metric->fx);scalar("metric_baseline",metric->baseline_m);scalar("metric_principal_offset",metric->cx_offset);}
        pass->dispatchCompute(groups, 1, 1);
        pass->end();
        // Paths add into the same sums volume. Explicitly order storage writes.
        encoder->globalBarrier();
    }
    NativeFrame enqueue(std::span<const std::uint32_t> left, std::span<const std::uint32_t> right) {
        const auto count = std::size_t(config.width) * config.height;
        if (left.size() != count || right.size() != count) throw std::invalid_argument("stereo input size mismatch");
        std::uint64_t completed = 0;
        check(fence->getCurrentValue(&completed), "query reusable slots");
        Slot* available = nullptr;
        for (auto& slot : slots) if (slot.lease.expired() && slot.fence_value <= completed) { available = &slot; break; }
        if (!available) throw std::runtime_error("native stereo queue full: release completed frame leases");
        auto& slot = *available;
        slot.commands = nullptr;
        auto encoder = queue->createCommandEncoder();
        if (!encoder) throw std::runtime_error("create command encoder failed");
        check(encoder->uploadBufferData(slot.buffers.at("left"), 0, count*4, left.data()), "upload left");
        check(encoder->uploadBufferData(slot.buffers.at("right"), 0, count*4, right.data()), "upload right");
        dispatch(encoder, slot, "census", std::uint32_t((count+63)/64));
        dispatch(encoder, slot, "census", std::uint32_t((count+63)/64), true);
        dispatch(encoder, slot, "cost", std::uint32_t((count*config.disparities+63)/64));
        constexpr int directions[8][2] = {{1,0},{-1,0},{0,1},{0,-1},{1,1},{-1,-1},{1,-1},{-1,1}};
        for (std::uint32_t path=0; path<config.paths; ++path) {
            const int dx = directions[path][0], dy = directions[path][1];
            const auto lines = dx == 0 ? config.width : (dy == 0 ? config.height : config.width+config.height-1);
            dispatch(encoder, slot, "aggregate", lines, false, dx, dy, lines);
        }
        dispatch(encoder, slot, "select", std::uint32_t((count+63)/64));
        if(metric)dispatch(encoder,slot,"depth_convert",std::uint32_t((count+63)/64));
        check(encoder->finish(slot.commands.writeRef()), "finish command buffer");
        const auto value = ++generation;
        rhi::ICommandBuffer* commands[] = {slot.commands};
        rhi::IFence* fences[] = {fence};
        rhi::SubmitDesc submit{};
        submit.commandBuffers = commands; submit.commandBufferCount = 1;
        submit.signalFences = fences; submit.signalFenceValues = &value; submit.signalFenceCount = 1;
        check(queue->submit(submit), "submit Vulkan stereo");
        slot.fence_value = value;
        auto lease = std::make_shared<Lease>(shared_from_this(), value);
        slot.lease = lease;
        CompletionToken token(lease);
        auto view = [&](const char* name) {
            return BufferView{slot.buffers.at(name).get(), 0, count*4, 4, identity, lease, token};
        };
        return {view("disparity"), view("confidence"), view("validity"), token, config.width, config.height,metric?view("metric_depth_output"):BufferView{},metric?view("metric_validity_output"):BufferView{}};
    }
};
NativeStereo::NativeStereo(StereoConfig config, const std::string& shader_directory, bool validation, std::optional<MetricCalibration> metric)
    : impl_(std::make_shared<Impl>(config, shader_directory, validation, metric)) {}
NativeFrame NativeStereo::enqueue(std::span<const std::uint32_t> left, std::span<const std::uint32_t> right) {
    return impl_->enqueue(left, right);
}
std::uint64_t NativeStereo::device_id() const { return impl_->identity; }
std::string NativeStereo::device_name() const { return impl_->device->getInfo().adapterName; }
StereoReadback NativeStereo::readback(const NativeFrame& frame) const {
    frame.disparity.validate(device_id()); frame.confidence.validate(device_id()); frame.validity.validate(device_id());
    if (frame.width != impl_->config.width || frame.height != impl_->config.height)
        throw std::invalid_argument("readback shape mismatch");
    frame.completion.wait();
    const auto count = std::size_t(frame.width) * frame.height;
    StereoReadback result{std::vector<float>(count), std::vector<float>(count), std::vector<std::uint32_t>(count)};
    check(impl_->device->readBuffer(static_cast<rhi::IBuffer*>(frame.disparity.native), 0, count*4, result.disparity.data()), "read disparity");
    check(impl_->device->readBuffer(static_cast<rhi::IBuffer*>(frame.confidence.native), 0, count*4, result.confidence.data()), "read confidence");
    check(impl_->device->readBuffer(static_cast<rhi::IBuffer*>(frame.validity.native), 0, count*4, result.validity.data()), "read validity");
    if(frame.depth.native){
        frame.depth.validate(device_id());frame.depth_validity.validate(device_id());
        result.depth.resize(count);result.depth_validity.resize(count);
        check(impl_->device->readBuffer(static_cast<rhi::IBuffer*>(frame.depth.native),0,count*4,result.depth.data()),"read metric depth");
        check(impl_->device->readBuffer(static_cast<rhi::IBuffer*>(frame.depth_validity.native),0,count*4,result.depth_validity.data()),"read metric validity");
    }
    return result;
}
} // namespace stereoforge
