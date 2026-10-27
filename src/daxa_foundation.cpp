#include <stereoforge/daxa_foundation.hpp>
#include <daxa/utils/task_graph.hpp>
#include <daxa/c/instance.h>
#include <daxa/c/device.h>
#include <array>
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace stereoforge {
namespace {
void check(bool condition, char const* message) {
    if(!condition) {
        std::cerr << "[foundation contract] " << message << '\n';
        throw std::runtime_error(message);
    }
}
VKAPI_ATTR VkBool32 VKAPI_CALL validation_message(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT, VkDebugUtilsMessengerCallbackDataEXT const* data, void* owner) {
    if(severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        static_cast<DaxaFoundation*>(owner)->validation_errors.fetch_add(1);
    if(severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        static_cast<DaxaFoundation*>(owner)->validation_warnings.fetch_add(1);
    std::cerr << "[Khronos] " << data->pMessage << '\n';
    return VK_FALSE;
}
std::string_view missing_feature_name(daxa::MissingRequiredVkFeature feature) {
    static constexpr std::array names{
        std::string_view{"NONE"},
        std::string_view{"IMAGE_CUBE_ARRAY"},
        std::string_view{"INDEPENDENT_BLEND"},
        std::string_view{"TESSELLATION_SHADER"},
        std::string_view{"MULTI_DRAW_INDIRECT"},
        std::string_view{"DEPTH_CLAMP"},
        std::string_view{"FILL_MODE_NON_SOLID"},
        std::string_view{"WIDE_LINES"},
        std::string_view{"SAMPLER_ANISOTROPY"},
        std::string_view{"FRAGMENT_STORES_AND_ATOMICS"},
        std::string_view{"SHADER_STORAGE_IMAGE_MULTISAMPLE"},
        std::string_view{"SHADER_STORAGE_IMAGE_READ_WITHOUT_FORMAT"},
        std::string_view{"SHADER_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT"},
        std::string_view{"SHADER_INT64"},
        std::string_view{"IMAGE_GATHER_EXTENDED"},
        std::string_view{"VARIABLE_POINTERS_STORAGE_BUFFER"},
        std::string_view{"VARIABLE_POINTERS"},
        std::string_view{"BUFFER_DEVICE_ADDRESS"},
        std::string_view{"BUFFER_DEVICE_ADDRESS_CAPTURE_REPLAY"},
        std::string_view{"BUFFER_DEVICE_ADDRESS_MULTI_DEVICE"},
        std::string_view{"SHADER_SAMPLED_IMAGE_ARRAY_NON_UNIFORM_INDEXING"},
        std::string_view{"SHADER_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING"},
        std::string_view{"SHADER_STORAGE_IMAGE_ARRAY_NON_UNIFORM_INDEXING"},
        std::string_view{"DESCRIPTOR_BINDING_SAMPLED_IMAGE_UPDATE_AFTER_BIND"},
        std::string_view{"DESCRIPTOR_BINDING_STORAGE_IMAGE_UPDATE_AFTER_BIND"},
        std::string_view{"DESCRIPTOR_BINDING_STORAGE_BUFFER_UPDATE_AFTER_BIND"},
        std::string_view{"DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING"},
        std::string_view{"DESCRIPTOR_BINDING_PARTIALLY_BOUND"},
        std::string_view{"RUNTIME_DESCRIPTOR_ARRAY"},
        std::string_view{"HOST_QUERY_RESET"},
        std::string_view{"DYNAMIC_RENDERING"},
        std::string_view{"SYNCHRONIZATION2"},
        std::string_view{"TIMELINE_SEMAPHORE"},
        std::string_view{"SUBGROUP_SIZE_CONTROL"},
        std::string_view{"COMPUTE_FULL_SUBGROUPS"},
        std::string_view{"SCALAR_BLOCK_LAYOUT"},
        std::string_view{"ACCELERATION_STRUCTURE_CAPTURE_REPLAY"},
        std::string_view{"VULKAN_MEMORY_MODEL"},
        std::string_view{"ROBUST_BUFFER_ACCESS2"},
        std::string_view{"ROBUST_IMAGE_ACCESS2"},
        std::string_view{"MAX_ENUM"},
    };
    auto index=static_cast<unsigned>(feature);
    return index<names.size() ? names[index] : std::string_view{"UNKNOWN_PINNED_FEATURE"};
}
std::vector<daxa::u32> shader() {
    std::ifstream file(SF_DAXA_SPIRV, std::ios::binary | std::ios::ate);
    check(bool(file), "missing offline Slang SPIR-V: rebuild daxa_foundation_shader");
    auto bytes = file.tellg();
    check(bytes > 0 && bytes % 4 == 0, "invalid SPIR-V byte length");
    std::vector<daxa::u32> words(static_cast<std::size_t>(bytes) / 4);
    file.seekg(0); file.read(reinterpret_cast<char*>(words.data()), bytes);
    check(bool(file) && words[0] == 0x07230203, "invalid SPIR-V module");
    return words;
}
}
void require_validation_layer() {
    uint32_t count = 0;
    check(vkEnumerateInstanceLayerProperties(&count,nullptr) == VK_SUCCESS, "cannot enumerate Vulkan layers");
    std::vector<VkLayerProperties> layers(count);
    check(vkEnumerateInstanceLayerProperties(&count,layers.data()) == VK_SUCCESS, "cannot enumerate Vulkan layers");
    bool found = false;
    for(auto const& layer:layers) found |= std::strcmp(layer.layerName,"VK_LAYER_KHRONOS_validation") == 0;
    check(found, "VK_LAYER_KHRONOS_validation is unavailable; install the Vulkan validation layers or set VK_LAYER_PATH");
    // Upstream 3.6 deliberately enables no layers. Loader activation is explicit.
    auto previous = std::getenv("VK_INSTANCE_LAYERS");
    std::string enabled = previous ? previous : "";
    if(enabled.find("VK_LAYER_KHRONOS_validation") == std::string::npos) {
        if(!enabled.empty()) enabled += ':';
        enabled += "VK_LAYER_KHRONOS_validation";
    }
    check(setenv("VK_INSTANCE_LAYERS", enabled.c_str(), 1) == 0, "cannot enable Khronos validation");
}
DaxaFoundation::DaxaFoundation(bool validation) {
    try {
    if(validation) require_validation_layer();
    instance = daxa::create_instance({.app_name="StereoForge Daxa foundation"});
    if(validation) {
        auto vk_instance = daxa_instance_get_vk_instance(instance.get());
        auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(vk_instance,"vkCreateDebugUtilsMessengerEXT"));
        check(create != nullptr, "VK_EXT_debug_utils is unavailable; cannot monitor validation");
        VkDebugUtilsMessengerCreateInfoEXT info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT;
        info.pfnUserCallback=validation_message; info.pUserData=this;
        check(create(vk_instance,&info,nullptr,&messenger) == VK_SUCCESS,"cannot create validation messenger");
    }
    auto devices = instance.list_devices_properties();
    for(uint32_t i=0;i<devices.size();++i) {
        auto const& properties=devices[i];
        if(properties.vendor_id != 0x1002) continue;
        if(properties.missing_required_feature != daxa::MissingRequiredVkFeature::NONE) {
            throw std::runtime_error("AMD device misses Daxa required feature " + std::string(missing_feature_name(properties.missing_required_feature)) + "; use a Vulkan 1.3 driver/device supporting the pinned Daxa required feature set");
        }
        device=instance.create_device_2({.physical_device_index=i,.name="StereoForge Daxa main compute owner"});
        return;
    }
    throw std::runtime_error("no AMD Vulkan physical device available; foundation requires the actual AMD target");
    } catch(...) {
        if(messenger) {
            auto vk_instance=daxa_instance_get_vk_instance(instance.get());
            auto destroy=reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(vk_instance,"vkDestroyDebugUtilsMessengerEXT"));
            if(destroy) destroy(vk_instance,messenger,nullptr);
            messenger={};
        }
        throw;
    }
}
void DaxaFoundation::close() {
    if(!device.get()) return;
    // Explicit teardown: retain the validation messenger through device destruction.
    check(daxa_dvc_wait_idle(device.get())==DAXA_RESULT_SUCCESS,"Daxa device teardown wait failed");
    check(daxa_dvc_collect_garbage(device.get())==DAXA_RESULT_SUCCESS,"Daxa device teardown collection failed");
    device={};
}
DaxaFoundation::~DaxaFoundation() noexcept {
    try { close(); }
    catch(std::exception const& error) { std::cerr << "Daxa fallback teardown: " << error.what() << '\n'; }
    if(messenger) {
        auto vk_instance=daxa_instance_get_vk_instance(instance.get());
        auto destroy=reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(vk_instance,"vkDestroyDebugUtilsMessengerEXT"));
        if(destroy) destroy(vk_instance,messenger,nullptr);
        messenger={};
    }
}
daxa::MemoryRequirements DaxaFoundation::coherent_readback_requirements(daxa::BufferInfo const& info) const {
    auto requirements=device.buffer_memory_requirements(info);
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(daxa_dvc_get_vk_physical_device(device.get()),&memory);
    for(uint32_t i=0;i<memory.memoryTypeCount;++i) {
        auto flags=memory.memoryTypes[i].propertyFlags;
        if((requirements.memory_type_bits & (1u<<i)) &&
           (flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
           (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            requirements.memory_type_bits=1u<<i;
            return requirements;
        }
    }
    throw std::runtime_error("no compatible HOST_VISIBLE|HOST_COHERENT readback memory type; explicit noncoherent invalidate path is not qualified");
}
std::string DaxaFoundation::report() const {
    auto physical=daxa_dvc_get_vk_physical_device(device.get());
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext=&f11; f11.pNext=&f12; f12.pNext=&f13;
    vkGetPhysicalDeviceFeatures2(physical,&features);
    VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceSubgroupSizeControlProperties control{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
    VkPhysicalDeviceIDProperties identity{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    control.pNext=&identity; identity.pNext=&driver;
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties.pNext=&subgroup; subgroup.pNext=&control;
    vkGetPhysicalDeviceProperties2(physical,&properties);
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(physical,&memory);
    auto const& p=properties.properties;
    auto hex=[](auto const& bytes) { std::ostringstream text; for(auto byte:bytes) text << std::hex << std::setfill('0') << std::setw(2) << unsigned(byte); return text.str(); };
    std::ostringstream out;
    out << "{\n  \"daxa_commit\": \"" SF_DAXA_REVISION "\",\n  \"slang_version\": \"" SF_DAXA_SLANG_VERSION "\",\n"
        << "  \"compiler_route\": \"official-linux-slangc-offline-spirv\",\n  \"slangc_sha256\": \"" SF_DAXA_COMPILER_SHA256 "\",\n"
        << "  \"device\": " << std::quoted(p.deviceName) << ",\n  \"vendor_id\": " << p.vendorID
        << ",\n  \"device_id\": " << p.deviceID << ",\n  \"driver_version\": " << p.driverVersion
        << ",\n  \"device_uuid\": " << std::quoted(hex(identity.deviceUUID)) << ",\n  \"driver_uuid\": " << std::quoted(hex(identity.driverUUID))
        << ",\n  \"driver_name\": " << std::quoted(driver.driverName) << ",\n  \"driver_info\": " << std::quoted(driver.driverInfo)
        << ",\n  \"vulkan_api_version\": " << p.apiVersion << ",\n  \"required_features\": {";
    bool first=true;
    auto feature=[&](char const* name, VkBool32 available) {
        out << (first ? "\n" : ",\n") << "    " << std::quoted(name) << ": {\"required\": true, \"available\": " << (available ? "true" : "false") << '}'; first=false;
    };
    feature("shaderStorageImageMultisample",features.features.shaderStorageImageMultisample);
    feature("imageCubeArray",features.features.imageCubeArray);
    feature("independentBlend",features.features.independentBlend);
    feature("tessellationShader",features.features.tessellationShader);
    feature("multiDrawIndirect",features.features.multiDrawIndirect);
    feature("depthClamp",features.features.depthClamp);
    feature("fillModeNonSolid",features.features.fillModeNonSolid);
    feature("wideLines",features.features.wideLines);
    feature("samplerAnisotropy",features.features.samplerAnisotropy);
    feature("fragmentStoresAndAtomics",features.features.fragmentStoresAndAtomics);
    feature("shaderStorageImageReadWithoutFormat",features.features.shaderStorageImageReadWithoutFormat);
    feature("shaderStorageImageWriteWithoutFormat",features.features.shaderStorageImageWriteWithoutFormat);
    feature("shaderInt64",features.features.shaderInt64);
    feature("shaderImageGatherExtended",features.features.shaderImageGatherExtended);
    feature("variablePointersStorageBuffer",f11.variablePointersStorageBuffer);
    feature("variablePointers",f11.variablePointers);
    feature("bufferDeviceAddress",f12.bufferDeviceAddress);
    feature("shaderSampledImageArrayNonUniformIndexing",f12.shaderSampledImageArrayNonUniformIndexing);
    feature("shaderStorageBufferArrayNonUniformIndexing",f12.shaderStorageBufferArrayNonUniformIndexing);
    feature("shaderStorageImageArrayNonUniformIndexing",f12.shaderStorageImageArrayNonUniformIndexing);
    feature("descriptorBindingSampledImageUpdateAfterBind",f12.descriptorBindingSampledImageUpdateAfterBind);
    feature("descriptorBindingStorageImageUpdateAfterBind",f12.descriptorBindingStorageImageUpdateAfterBind);
    feature("descriptorBindingStorageBufferUpdateAfterBind",f12.descriptorBindingStorageBufferUpdateAfterBind);
    feature("descriptorBindingUpdateUnusedWhilePending",f12.descriptorBindingUpdateUnusedWhilePending);
    feature("descriptorBindingPartiallyBound",f12.descriptorBindingPartiallyBound);
    feature("runtimeDescriptorArray",f12.runtimeDescriptorArray);
    feature("hostQueryReset",f12.hostQueryReset);
    feature("dynamicRendering",f13.dynamicRendering);
    feature("synchronization2",f13.synchronization2);
    feature("timelineSemaphore",f12.timelineSemaphore);
    feature("subgroupSizeControl",f13.subgroupSizeControl);
    feature("computeFullSubgroups",f13.computeFullSubgroups);
    feature("scalarBlockLayout",f12.scalarBlockLayout);
    out << "\n  },\n  \"missing_features\": [],\n  \"subgroups\": {\"default_size\": " << subgroup.subgroupSize << ", \"min_size\": " << control.minSubgroupSize
        << ", \"max_size\": " << control.maxSubgroupSize << ", \"supported_stages\": " << subgroup.supportedStages
        << ", \"supported_operations\": " << subgroup.supportedOperations << "},\n  \"limits\": {\"max_compute_invocations\": " << p.limits.maxComputeWorkGroupInvocations
        << ", \"max_compute_workgroup_size\": [" << p.limits.maxComputeWorkGroupSize[0] << ',' << p.limits.maxComputeWorkGroupSize[1] << ',' << p.limits.maxComputeWorkGroupSize[2]
        << "], \"max_compute_shared_memory_bytes\": " << p.limits.maxComputeSharedMemorySize
        << ", \"max_storage_buffer_range\": " << p.limits.maxStorageBufferRange
        << ", \"max_push_constants_bytes\": " << p.limits.maxPushConstantsSize
        << ", \"noncoherent_atom_size\": " << p.limits.nonCoherentAtomSize << "},\n  \"memory_types\": [";
    for(uint32_t i=0;i<memory.memoryTypeCount;++i) {
        auto flags=memory.memoryTypes[i].propertyFlags;
        out << (i ? "," : "") << "\n    {\"index\": " << i << ", \"heap\": " << memory.memoryTypes[i].heapIndex
            << ", \"flags\": " << flags << ", \"host_visible\": " << ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? "true":"false")
            << ", \"host_coherent\": " << ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? "true":"false") << '}';
    }
    auto readback=coherent_readback_requirements({.size=12,.memory_flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});
    out << "\n  ],\n  \"readback_memory_type_bits\": " << readback.memory_type_bits
        << ",\n  \"readback_policy\": \"explicit coherent memory type; wait final completion before CPU access\",\n  \"queue\": \"Daxa main compute-capable queue\",\n  \"validation_errors\": " << validation_errors.load() << ",\n  \"validation_warnings\": " << validation_warnings.load() << "\n}\n";
    return out.str();
}
void execute_foundation_frame(daxa::TaskGraph& graph, daxa::Device& device) {
    graph.execute({});
    device.collect_garbage(); // Poll completed Daxa submissions; reclaim retired command pools.
}
void run_daxa_foundation(DaxaFoundation& context, bool benchmark) {
    auto& device=context.device;
    auto words=shader();
    auto pipeline=device.create_compute_pipeline({.shader_info={.byte_code=words.data(),.byte_code_size=static_cast<uint32_t>(words.size())},.push_constant_size=24,.name="offline Slang 2026.12 foundation"});
    // This fixed diagnostic owns raw Daxa IDs immediately, including on exceptions.
    // Graphs/external wrappers are declared later and therefore unwind first.
    struct FoundationBuffers {
        daxa::Device& device;
        std::array<daxa::BufferId,4> ids{};
        daxa::MemoryBlock readback_memory;
        bool completion_reached=false;
        ~FoundationBuffers() {
            if(!completion_reached) {
                auto result=daxa_dvc_wait_idle(device.get()); // Failure/teardown boundary only.
                if(result!=DAXA_RESULT_SUCCESS) std::cerr << "Daxa cleanup wait failed: " << result << '\n';
            }
            for(auto id:ids) {
                if(!id.is_empty() && device.is_id_valid(id)) device.destroy_buffer(id);
            }
            device.collect_garbage();
            // readback_memory releases after its raw buffer has been destroyed.
        }
    } owned{device};
    auto resident=owned.ids[0]=device.create_buffer({.size=64,.name="application resident buffer"});
    auto counters=owned.ids[1]=device.create_buffer({.size=12,.name="GPU final counters"});
    daxa::BufferInfo readback_info{.size=12,.name="12-byte coherent final readback"};
    owned.readback_memory=device.create_memory({.requirements=context.coherent_readback_requirements(readback_info),.flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});
    // Block allocation owns memory flags; buffer flags must be zero for this API.
    auto readback=owned.ids[2]=device.create_buffer_from_memory_block({.buffer_info=readback_info,.memory_block=owned.readback_memory});
    daxa::ExternalTaskBuffer resident_external({.buffer=resident,.name="resident external"});
    daxa::ExternalTaskBuffer counters_external({.buffer=counters,.name="counters external"});
    daxa::ExternalTaskBuffer readback_external({.buffer=readback,.name="readback external"});
    uint32_t frame=0;
    daxa::BufferId observed_resident{};
    struct Push { uint64_t resident,counters; uint32_t frame,phase; };
    static_assert(sizeof(Push)==24);
    {
        daxa::TaskGraph graph({.device=device,.name="resident external view regression"});
        // Deliberately create a preceding resource: graph identity != resource index.
        auto anchor=graph.create_task_buffer({.size=4,.name="identity anchor"});
        auto view=graph.register_buffer(resident_external);
        check(view.task_graph_index==anchor.task_graph_index && view.index!=anchor.index && view.double_buffer_index==0,
              "external registered view has malformed graph identity/index/generation");
        check(device.buffer_info(resident).value().size==64,"resident byte range changed");
        auto counter_view=graph.register_buffer(counters_external);
        graph.add_task(daxa::InlineTask::Compute("changing GPU resident writes")
            .writes(view).executes([&](daxa::TaskInterface ti) {
                observed_resident=ti.get(view).id;
                check(observed_resident==resident,"external view lost resident buffer identity/generation");
                ti.recorder.set_pipeline(pipeline);
                ti.recorder.push_constant(Push{device.device_address(ti.get(view).id).value(),device.device_address(counters).value(),frame,0});
                ti.recorder.dispatch({1,1,1});
            }));
        graph.add_task(daxa::InlineTask::Compute("GPU consumer verifies fresh frame")
            .reads(view).reads_writes(counter_view).executes([&](daxa::TaskInterface ti) {
                ti.recorder.set_pipeline(pipeline);
                ti.recorder.push_constant(Push{device.device_address(ti.get(view).id).value(),device.device_address(ti.get(counter_view).id).value(),frame,1});
                ti.recorder.dispatch({1,1,1});
            }));
        graph.submit({}); graph.complete({});
        daxa::TaskGraph final({.device=device,.name="final tiny readback"});
        auto source=final.register_buffer(counters_external);
        auto destination=final.register_buffer(readback_external);
        final.add_task(daxa::InlineTask::Transfer("copy only final GPU counters")
            .reads(source).writes(destination).executes([&](daxa::TaskInterface ti) {
                ti.recorder.copy_buffer_to_buffer({.src_buffer=ti.get(source).id,.dst_buffer=ti.get(destination).id,.size=12});
            }));
        final.add_task(daxa::InlineTask::Transfer("make final counters host visible").host.reads(destination).executes([](daxa::TaskInterface){}));
        // Pinned 3.6 TaskGraph::submit ignores TaskSubmitInfo, including extra signals.
        // Completion uses Daxa's implemented per-queue submission timeline instead.
        final.submit({}); final.complete({});
        // No StereoForge host wait/resource allocation/copy/readback in these three executions.
        for(frame=1;frame<=3;++frame) {
            execute_foundation_frame(graph,device);
        }
        std::vector<double> enqueue_us;
        double backpressure_wait_ms=0, batch_reclaim_cpu_ms=0;
        uint32_t backpressure_wait_count=0;
        constexpr uint32_t max_inflight_frames=16; // Safely below pinned 128 command pools.
        std::chrono::steady_clock::time_point measurement_start;
        if(benchmark) {
            auto warmup_submit=device.latest_queue_submit_index(daxa::QUEUE_MAIN);
            device.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=warmup_submit,.timeout=30'000'000'000ull});
            device.collect_garbage(); // Warmup completion/reclamation outside measurement.
            enqueue_us.reserve(300);
            measurement_start=std::chrono::steady_clock::now();
            for(uint32_t sample=0;sample<300;++sample) {
                frame=sample%3+1;
                auto start=std::chrono::steady_clock::now();
                execute_foundation_frame(graph,device);
                auto end=std::chrono::steady_clock::now();
                enqueue_us.push_back(std::chrono::duration<double,std::micro>(end-start).count());
                if((sample+1)%max_inflight_frames==0) {
                    auto submitted=device.latest_queue_submit_index(daxa::QUEUE_MAIN);
                    auto wait_start=std::chrono::steady_clock::now();
                    device.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=submitted,.timeout=30'000'000'000ull});
                    auto wait_end=std::chrono::steady_clock::now();
                    device.collect_garbage();
                    auto reclaim_end=std::chrono::steady_clock::now();
                    backpressure_wait_ms+=std::chrono::duration<double,std::milli>(wait_end-wait_start).count();
                    batch_reclaim_cpu_ms+=std::chrono::duration<double,std::milli>(reclaim_end-wait_end).count();
                    ++backpressure_wait_count;
                }
            }
        }
        auto before_final_submit=device.latest_queue_submit_index(daxa::QUEUE_MAIN);
        final.execute({});
        auto enqueue_end=std::chrono::steady_clock::now();
        auto final_submit=device.latest_queue_submit_index(daxa::QUEUE_MAIN);
        check(final_submit>before_final_submit,"final graph did not submit work on Daxa main queue");
        device.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=final_submit,.timeout=30'000'000'000ull});
        owned.completion_reached=true;
        if(benchmark) {
            auto completion=std::chrono::steady_clock::now();
            std::sort(enqueue_us.begin(),enqueue_us.end());
            std::cout << "{\"benchmark\":\"foundation-primitive\",\"samples\":300,\"warmup_frames\":3,\"cpu_enqueue_p50_us\":" << enqueue_us[150]
                << ",\"cpu_enqueue_p95_us\":" << enqueue_us[285]
                << ",\"cpu_enqueue_includes_gc\":true,\"max_inflight_frames\":" << max_inflight_frames
                << ",\"backpressure_wait_count\":" << backpressure_wait_count
                << ",\"backpressure_wait_ms\":" << backpressure_wait_ms
                << ",\"batch_reclaim_cpu_ms\":" << batch_reclaim_cpu_ms
                << ",\"final_completion_wait_ms\":" << std::chrono::duration<double,std::milli>(completion-enqueue_end).count()
                << ",\"enqueue_through_completion_ms\":" << std::chrono::duration<double,std::milli>(completion-measurement_start).count()
                << ",\"gpu_timing\":\"unavailable-not-instrumented\",\"readback_bytes\":12}\n";
        }
        check(device.buffer_info(observed_resident).value().size==64,"resolved external buffer byte range changed during execution");
        auto values=device.buffer_host_address_as<uint32_t>(readback).value();
        check(values[0]==1 && values[1]==2 && values[2]==3,"GPU consumer counters were stale or incorrect");
    }
    device.destroy_buffer(resident); device.destroy_buffer(counters); device.destroy_buffer(readback);
    device.collect_garbage();
    auto replacement=owned.ids[3]=device.create_buffer({.size=64,.name="new allocation generation"});
    check(replacement!=resident,"destroyed buffer ID/generation was reused");
    check(!device.is_id_valid(resident),"stale destroyed buffer ID remained valid after generation replacement");
    device.destroy_buffer(replacement);
    check(context.validation_errors.load()==0,"Khronos validation reported errors");
}
}
