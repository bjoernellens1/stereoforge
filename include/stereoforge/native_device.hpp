#pragma once
#include <cstdint>
#include <memory>
#include <string>

namespace rhi { class IDevice; class ICommandQueue; }
namespace stereoforge {
// Copyable shared Vulkan context. id() identifies this context and its single
// compute-capable queue, not merely the physical GPU. All native producers and
// consumers must submit on rhi_queue(); cross-queue imports are unsupported.
class NativeDevice {
    struct Impl;
    std::shared_ptr<Impl> impl_;
public:
    explicit NativeDevice(bool vulkan_validation=false);
    std::uint64_t id() const;
    std::string name() const;
    // Borrowed pointers stay alive while a NativeDevice copy is retained.
    rhi::IDevice* rhi_device() const;
    rhi::ICommandQueue* rhi_queue() const;
};
} // namespace stereoforge
