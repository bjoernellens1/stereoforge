// Read-only physical-device capability probe. This does not establish IPC.
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::string quote(const char* s) {
    std::ostringstream out;
    out << '"';
    for (auto p = reinterpret_cast<const unsigned char*>(s); *p; ++p) {
        if (*p == '"' || *p == '\\') out << '\\' << *p;
        else if (*p < 0x20) out << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << unsigned(*p) << std::dec;
        else out << *p;
    }
    out << '"';
    return out.str();
}
std::string hex_bytes(const uint8_t* data, size_t n) {
    std::ostringstream out;
    out << '"' << std::hex << std::setfill('0');
    for (size_t i = 0; i < n; ++i) out << std::setw(2) << unsigned(data[i]);
    out << '"';
    return out.str();
}
bool has_extension(const std::vector<VkExtensionProperties>& exts, const char* name) {
    return std::any_of(exts.begin(), exts.end(), [name](const auto& x) { return std::strcmp(x.extensionName, name) == 0; });
}
const char* boolean(bool v) { return v ? "true" : "false"; }
void buffer(VkPhysicalDevice physical, VkExternalMemoryHandleTypeFlagBits handle, bool available) {
    if (!available) { std::cout << "{\"queried\":false}"; return; }
    VkPhysicalDeviceExternalBufferInfo input{};
    input.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
    input.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    input.handleType = handle;
    VkExternalBufferProperties output{};
    output.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
    vkGetPhysicalDeviceExternalBufferProperties(physical, &input, &output);
    const auto& p = output.externalMemoryProperties;
    std::cout << "{\"queried\":true,\"importable\":" << boolean(p.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT)
              << ",\"exportable\":" << boolean(p.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT)
              << ",\"dedicated_only\":" << boolean(p.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT)
              << ",\"compatible_handle_types\":" << p.compatibleHandleTypes
              << ",\"export_from_imported_handle_types\":" << p.exportFromImportedHandleTypes << '}';
}
void semaphore(VkPhysicalDevice physical, VkExternalSemaphoreHandleTypeFlagBits handle, bool available) {
    if (!available) { std::cout << "{\"queried\":false}"; return; }
    VkPhysicalDeviceExternalSemaphoreInfo input{};
    input.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO;
    input.handleType = handle;
    VkExternalSemaphoreProperties output{};
    output.sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES;
    vkGetPhysicalDeviceExternalSemaphoreProperties(physical, &input, &output);
    std::cout << "{\"queried\":true,\"importable\":" << boolean(output.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT)
              << ",\"exportable\":" << boolean(output.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT)
              << ",\"compatible_handle_types\":" << output.compatibleHandleTypes
              << ",\"export_from_imported_handle_types\":" << output.exportFromImportedHandleTypes << '}';
}
}
int main() {
    uint32_t loader_version = VK_API_VERSION_1_0;
    auto enumerate_version = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
    if (enumerate_version && enumerate_version(&loader_version) != VK_SUCCESS) return 1;
    if (loader_version < VK_API_VERSION_1_1) { std::cerr << "Vulkan 1.1 loader required\n"; return 1; }
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "StereoForge external capability probe";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo create{};
    create.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&create, nullptr, &instance) != VK_SUCCESS) { std::cerr << "Cannot create Vulkan instance\n"; return 1; }
    uint32_t count = 0;
    VkResult result = vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (result != VK_SUCCESS) { std::cerr << "Cannot enumerate physical devices\n"; vkDestroyInstance(instance, nullptr); return 1; }
    std::vector<VkPhysicalDevice> devices(count);
    result = vkEnumeratePhysicalDevices(instance, &count, devices.data());
    if (result != VK_SUCCESS) { std::cerr << "Physical device enumeration failed or changed\n"; vkDestroyInstance(instance, nullptr); return 1; }
    devices.resize(count);
    std::cout << "{\"schema\":\"stereoforge.vulkan_external_capabilities.v1\",\"scope\":\"physical_device_properties_only\",\"timeline_compatibility\":\"not_queried\",\"devices\":[";
    bool first = true;
    for (auto device : devices) {
        VkPhysicalDeviceIDProperties identity{};
        identity.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
        VkPhysicalDeviceProperties2 props{};
        props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props.pNext = &identity;
        vkGetPhysicalDeviceProperties2(device, &props);
        uint32_t ext_count = 0;
        if (vkEnumerateDeviceExtensionProperties(device, nullptr, &ext_count, nullptr) != VK_SUCCESS) { std::cerr << "Cannot enumerate device extensions\n"; vkDestroyInstance(instance, nullptr); return 1; }
        std::vector<VkExtensionProperties> exts(ext_count);
        if (vkEnumerateDeviceExtensionProperties(device, nullptr, &ext_count, exts.data()) != VK_SUCCESS) { std::cerr << "Device extension enumeration failed or changed\n"; vkDestroyInstance(instance, nullptr); return 1; }
        exts.resize(ext_count);
        if (!first) std::cout << ',';
        first = false;
        std::cout << "{\"name\":" << quote(props.properties.deviceName)
                  << ",\"vendor_id\":" << props.properties.vendorID << ",\"device_id\":" << props.properties.deviceID
                  << ",\"api_version\":" << props.properties.apiVersion << ",\"driver_version\":" << props.properties.driverVersion
                  << ",\"device_uuid\":" << hex_bytes(identity.deviceUUID, VK_UUID_SIZE)
                  << ",\"driver_uuid\":" << hex_bytes(identity.driverUUID, VK_UUID_SIZE)
                  << ",\"buffer_usage\":[\"storage_buffer\",\"transfer_src\",\"transfer_dst\"],\"external_buffers\":{\"opaque_fd\":";
        bool memory_fd = has_extension(exts, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
        bool dma_buf = has_extension(exts, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);
        buffer(device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT, memory_fd);
        std::cout << ",\"dma_buf\":";
        buffer(device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, memory_fd && dma_buf);
        bool semaphore_fd = has_extension(exts, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
        std::cout << "},\"external_semaphores\":{\"opaque_fd\":";
        semaphore(device, VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT, semaphore_fd);
        std::cout << ",\"sync_fd\":";
        semaphore(device, VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT, semaphore_fd);
        std::cout << "}}";
    }
    std::cout << "]}\n";
    vkDestroyInstance(instance, nullptr);
}
