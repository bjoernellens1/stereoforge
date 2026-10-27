#pragma once
#include <daxa/daxa.hpp>
#include <vulkan/vulkan.h>
#include <atomic>
#include <string>
namespace stereoforge {
// Only the foundation diagnostic owns this context. No foreign resource import.
struct DaxaFoundation {
    daxa::Instance instance;
    daxa::Device device;
    VkDebugUtilsMessengerEXT messenger{};
    std::atomic<unsigned> validation_errors{0};
    std::atomic<unsigned> validation_warnings{0};
    explicit DaxaFoundation(bool validation);
    ~DaxaFoundation() noexcept;
    void close();
    DaxaFoundation(DaxaFoundation const&) = delete;
    DaxaFoundation& operator=(DaxaFoundation const&) = delete;
    std::string report() const;
    daxa::MemoryRequirements coherent_readback_requirements(daxa::BufferInfo const&) const;
};
void require_validation_layer();
void run_daxa_foundation(DaxaFoundation&, bool benchmark = false);
}
