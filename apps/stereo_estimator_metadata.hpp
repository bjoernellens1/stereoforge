#pragma once
#include <stereoforge/daxa_depth.hpp>
#if __has_include(<json.hpp>)
#include <json.hpp>
#else
#include <nlohmann/json.hpp>
#endif
#include <filesystem>
#include <stdexcept>
#include <string>

namespace stereoforge::native_estimator {
// Setup or post-measurement only: this reads compiled shader bytes from disk.
inline nlohmann::json shader_hashes(const std::string& directory,const std::string& prefix){
    auto hashes=nlohmann::json::object();
    for(auto const& entry:std::filesystem::directory_iterator(directory))
        if(entry.is_regular_file()&&entry.path().extension()==".spv"&&entry.path().filename().string().starts_with(prefix))
            hashes[entry.path().filename().string()]=daxa_preprocess_file_sha256(entry.path());
    if(hashes.empty())throw std::runtime_error("compiled shader family unavailable");
    return hashes;
}
inline nlohmann::json instantaneous(){return {{"stabilization",{{"status","available"},{"value",0}}},
    {"tracking",{{"status","available"},{"value",false}}},{"filling",{{"status","available"},{"value",false}}}};}
// Only instantiated for the independently matched, non-temporal SGM path.
inline nlohmann::json profile(const DaxaDepthConfig& config,const nlohmann::json& temporal,
                               const nlohmann::json& stereo_shaders,const nlohmann::json& depth_shaders,
                               const std::string& compiler_sha256){
    config.validate();
    if(config.stereo_variant!=DaxaSgmVariant::PackedSpecialized || config.right_reference_mode!=DaxaRightReferenceMode::IndependentSGM || stereo_shaders.empty() || depth_shaders.empty() || compiler_sha256.empty())
        throw std::invalid_argument("independent SGM shader identity required");
    auto const& s=config.stereo;
    return {{"algorithm","independent_sgm"},{"width",s.width},{"height",s.height},{"disparities",s.disparities},
        {"paths",s.paths},{"p1",s.p1},{"p2",s.p2},{"subpixel",s.subpixel},{"lr_tolerance",config.lr_tolerance},
        {"right_reference_mode","independent_sgm"},{"depth_units","axial_Z_metres_rectified_rays"},{"temporal",temporal},
        {"compiler_sha256",compiler_sha256},{"stereo_spirv_sha256",stereo_shaders},{"depth_spirv_sha256",depth_shaders}};
}
}
