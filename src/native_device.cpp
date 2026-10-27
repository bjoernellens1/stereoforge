#include <stereoforge/native_device.hpp>
#include <stereoforge/detail/vulkan_validation.hpp>
#include <slang-rhi.h>
#include <atomic>
#include <stdexcept>

namespace stereoforge {
namespace {std::atomic<std::uint64_t> next_context{1};}
struct NativeDevice::Impl {
    const std::uint64_t identity=next_context.fetch_add(1);
    rhi::ComPtr<rhi::IDevice> device;
    rhi::ComPtr<rhi::ICommandQueue> queue;
    explicit Impl(bool validation) {
        if(validation)detail::require_vulkan_validation();
        rhi::DeviceDesc desc{};
        desc.deviceType=rhi::DeviceType::Vulkan;desc.enableValidation=true;
        desc.enableCUDALaunchFromGfx=false;desc.enableRayTracing=false;
        desc.slang.targetProfile="sm_6_0";
        if(SLANG_FAILED(rhi::getRHI()->createDevice(desc,device.writeRef())))
            throw std::runtime_error("create shared Vulkan device failed");
        // This pinned RHI exposes its Vulkan compute-capable queue as Graphics.
        if(SLANG_FAILED(device->getQueue(rhi::QueueType::Graphics,queue.writeRef())))
            throw std::runtime_error("create shared Vulkan queue failed");
    }
    ~Impl(){if(queue)queue->waitOnHost();}
};
NativeDevice::NativeDevice(bool validation):impl_(std::make_shared<Impl>(validation)){}
std::uint64_t NativeDevice::id()const{return impl_?impl_->identity:0;}
std::string NativeDevice::name()const {
    if(!impl_)throw std::invalid_argument("empty native context");
    const auto name=impl_->device->getInfo().adapterName;return name?name:"unknown Vulkan adapter";
}
rhi::IDevice* NativeDevice::rhi_device()const{return impl_?impl_->device.get():nullptr;}
rhi::ICommandQueue* NativeDevice::rhi_queue()const{return impl_?impl_->queue.get():nullptr;}
} // namespace stereoforge
