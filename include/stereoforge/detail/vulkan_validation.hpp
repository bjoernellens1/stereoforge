#pragma once
#include <slang-rhi.h>
#include <vulkan/vulkan_core.h>
#include <dlfcn.h>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace stereoforge::detail {
// The pinned RHI silently omits an undiscovered Khronos layer even when its
// DebugLayerOptions.required is true. Check discovery independently first.
inline void require_vulkan_validation() {
    struct CloseLoader { void operator()(void* handle) const { dlclose(handle); } };
    std::unique_ptr<void, CloseLoader> loader(dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL));
    if (!loader) throw std::runtime_error("Vulkan validation requires the Vulkan loader libvulkan.so.1");
    auto enumerate = reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(dlsym(loader.get(), "vkEnumerateInstanceLayerProperties"));
    if (!enumerate) throw std::runtime_error("Vulkan loader cannot enumerate validation layers");
    bool found = false;
    for (int attempt=0; attempt<3; ++attempt) {
        std::uint32_t count = 0;
        if (enumerate(&count, nullptr) != VK_SUCCESS) throw std::runtime_error("cannot enumerate Vulkan instance layers");
        std::vector<VkLayerProperties> layers(count);
        const auto result = enumerate(&count, layers.data());
        if (result == VK_INCOMPLETE) continue;
        if (result != VK_SUCCESS) throw std::runtime_error("cannot read Vulkan instance layers");
        layers.resize(count);
        for (const auto& layer : layers) if (std::strcmp(layer.layerName,"VK_LAYER_KHRONOS_validation") == 0) found = true;
        break;
    }
    if (!found) throw std::runtime_error("Vulkan validation requested but VK_LAYER_KHRONOS_validation is unavailable; validation was not performed");
    rhi::DebugLayerOptions options{};
    options.required = true;
    options.coreValidation = true;
    if (rhi::getRHI()->getDebugLayerOptions() != options && SLANG_FAILED(rhi::getRHI()->setDebugLayerOptions(options)))
        throw std::runtime_error("Vulkan validation must be configured before any RHI device is created");
}
} // namespace stereoforge::detail
