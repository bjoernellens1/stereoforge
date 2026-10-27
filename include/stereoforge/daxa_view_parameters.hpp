#pragma once
#include <stereoforge/geometry.hpp>
#include <cstddef>
namespace stereoforge::detail {
struct ViewParameters {
    uint64_t address_source_depth=0;
    uint64_t address_source_confidence=0;
    uint64_t address_source_rgb=0;
    uint64_t address_source_provenance=0;
    uint64_t address_source_rays=0;
    uint64_t address_zbuffer=0;
    uint64_t address_owner=0;
    uint64_t address_output_rgb=0;
    uint64_t address_output_valid=0;
    uint64_t address_output_source=0;
    uint64_t address_output_provenance=0;
    uint64_t address_output_depth=0;
    uint64_t address_output_confidence=0;
    uint64_t address_layer_z=0;
    uint64_t address_layer_owner=0;
    uint64_t address_ratio_histogram=0;
    uint64_t address_exposure_gain=0;
    uint64_t address_preview_indices=0;
    uint64_t address_output_rgb_valid=0;
    uint64_t address_output_approximate=0;
    uint64_t address_packed_readback=0;
    uint64_t address_source_ids=0;
    uint64_t address_source_ages=0;
    uint64_t address_output_source_ids=0;
    uint64_t address_output_age_ns=0;
    uint32_t field_src_width{};
    uint32_t field_src_height{};
    uint32_t field_target_width{};
    uint32_t field_target_height{};
    uint32_t field_projection{};
    float field_fx{};
    float field_fy{};
    float field_cx{};
    float field_cy{};
    float field_hfov_radians{};
    Pose4 field_source_to_virtual_left{};
    Pose4 field_source_to_virtual_right{};
    uint32_t field_preview_enabled{};
    uint32_t field_exposure_enabled{};
};
static_assert(sizeof(ViewParameters)==376);
static_assert(offsetof(ViewParameters,field_src_width)==200);
static_assert(offsetof(ViewParameters,field_source_to_virtual_left)==240);
static_assert(offsetof(ViewParameters,field_source_to_virtual_right)==304);
static_assert(offsetof(ViewParameters,field_preview_enabled)==368);
static_assert(offsetof(ViewParameters,field_exposure_enabled)==372);
}
