#include <stereoforge/kernel_variants.hpp>
#include <iostream>
#include <stdexcept>
using namespace stereoforge;
void require(bool ok) { if(!ok) throw std::runtime_error("variant contract failed"); }
int main() {
    KernelIdentity key{"uuid", "driver", "daxa+patch", "slang", std::string(64,'a'), std::string(64,'b'), "specialization", 1280,720,1280,720,128,4,"float32","frozen-profile",64};
    require(key.complete());
    auto changed=key;
    auto reject=[&] { require(!cache_matches(key,changed)); changed=key; };
    changed.physical_gpu="other";reject();changed.driver="other";reject();changed.daxa="other";reject();
    changed.slang="other";reject();changed.compiler_sha256="other";reject();changed.shader_sha256="other";reject();
    changed.specialization="other";reject();changed.source_width++;reject();changed.source_height++;reject();
    changed.target_width++;reject();changed.target_height++;reject();changed.disparities++;reject();changed.paths++;reject();
    changed.precision="other";reject();changed.quality_profile="other";reject();changed.subgroup=32;reject();
    changed.driver="";require(!cache_matches(changed,changed));
    require(cache_matches(key,key));
    changed=key;changed.shader_sha256="label-not-actual-hash";require(!changed.complete());
    changed=key;changed.compiler_sha256="";require(!changed.complete());
    require(find_kernel_variant("metadata-fused-64") != nullptr);
    require(find_kernel_variant("metadata-split-64") != nullptr);
    require(find_kernel_variant("unknown") == nullptr);
    require(supports_subgroup(64,32,64,32));
    require(!supports_subgroup(48,32,64,32));
    require(!supports_subgroup(64,32,64,0));
    std::cout << "kernel identity and curated variants passed\n";
}
