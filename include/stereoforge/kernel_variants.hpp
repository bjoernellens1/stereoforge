#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <string_view>
namespace stereoforge {
// Exact execution identity. Hash fields refer to actual compiler/SPIR-V bytes;
// daxa includes the applied patch identity. Never use a marketing GPU name.
struct KernelIdentity {
    std::string physical_gpu, driver, daxa, slang, compiler_sha256, shader_sha256, specialization;
    uint32_t source_width=0, source_height=0, target_width=0, target_height=0, disparities=0, paths=0;
    std::string precision, quality_profile;
    uint32_t subgroup=0;
    bool operator==(KernelIdentity const&) const = default;
    static bool hash(std::string_view value) {
        if(value.size()!=64) return false;
        for(char c:value) if(!((c>='0'&&c<='9')||(c>='a'&&c<='f'))) return false;
        return true;
    }
    bool complete() const {
        return !physical_gpu.empty() && !driver.empty() && !daxa.empty() && !slang.empty() &&
            hash(compiler_sha256) && hash(shader_sha256) && !specialization.empty() &&
            source_width && source_height && target_width && target_height &&
            !precision.empty() && !quality_profile.empty() && subgroup;
    }
};
inline bool cache_matches(KernelIdentity const& stored, KernelIdentity const& current) {
    return stored.complete() && current.complete() && stored==current;
}
// Compute-stage subgroup support is queried from Vulkan. Only power-of-two
// sizes within queried bounds apply. This never forces a shader subgroup size.
inline bool supports_subgroup(uint32_t requested,uint32_t minimum,uint32_t maximum,uint32_t compute_stages) {
    return requested && !(requested&(requested-1)) && minimum && requested>=minimum &&
        requested<=maximum && (compute_stages & 32u); // VK_SHADER_STAGE_COMPUTE_BIT
}
struct KernelVariant {
    std::string_view name;
    std::array<std::string_view,2> entrypoints;
    uint32_t dispatches;
    std::array<uint32_t,3> workgroup;
};
// Local metadata gather only: output target-grid ID/age, no geometry changes.
// Split retains the same read-only owner/valid/source inputs across both passes.
inline constexpr std::array kernel_variants{
    KernelVariant{"metadata-fused-64", {"metadata_fused",""},1,{64,1,1}},
    KernelVariant{"metadata-split-64", {"metadata_ids","metadata_ages"},2,{64,1,1}},
};
inline KernelVariant const* find_kernel_variant(std::string_view name) {
    for(auto const& variant:kernel_variants) if(variant.name==name) return &variant;
    return nullptr;
}
}
