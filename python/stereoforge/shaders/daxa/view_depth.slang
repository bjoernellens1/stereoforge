// Daxa physical-buffer bindings only; algorithm lives in ../ops/projection.slang.
struct ViewParameters {
    uint64_t address_source_depth;
    uint64_t address_source_confidence;
    uint64_t address_source_rgb;
    uint64_t address_source_provenance;
    uint64_t address_source_rays;
    uint64_t address_zbuffer;
    uint64_t address_owner;
    uint64_t address_output_rgb;
    uint64_t address_output_valid;
    uint64_t address_output_source;
    uint64_t address_output_provenance;
    uint64_t address_output_depth;
    uint64_t address_output_confidence;
    uint64_t address_layer_z;
    uint64_t address_layer_owner;
    uint64_t address_ratio_histogram;
    uint64_t address_exposure_gain;
    uint64_t address_preview_indices;
    uint64_t address_output_rgb_valid;
    uint64_t address_output_approximate;
    uint64_t address_packed_readback;
    uint64_t address_source_ids;
    uint64_t address_source_ages;
    uint64_t address_output_source_ids;
    uint64_t address_output_age_ns;
    uint field_src_width;
    uint field_src_height;
    uint field_target_width;
    uint field_target_height;
    uint field_projection;
    float field_fx;
    float field_fy;
    float field_cx;
    float field_cy;
    float field_hfov_radians;
    float4x4 field_source_to_virtual_left;
    float4x4 field_source_to_virtual_right;
    uint field_preview_enabled;
    uint field_exposure_enabled;
    uint64_t rectified_right_rgb,original_left_rgb,original_right_rgb;
    uint original_width,original_height;
};
struct Push { uint64_t parameters; };
[[vk::push_constant]] ConstantBuffer<Push> pc;
#define params (*(ViewParameters*)pc.parameters)
#define source_depth ((float*)params.address_source_depth)
#define source_confidence ((float*)params.address_source_confidence)
#define source_rgb ((uint*)params.address_source_rgb)
#define source_provenance ((uint*)params.address_source_provenance)
#define source_rays ((float3*)params.address_source_rays)
#define zbuffer ((uint*)params.address_zbuffer)
#define owner ((uint*)params.address_owner)
#define output_rgb ((uint*)params.address_output_rgb)
#define output_valid ((uint*)params.address_output_valid)
#define output_source ((uint*)params.address_output_source)
#define output_provenance ((uint*)params.address_output_provenance)
#define output_depth ((float*)params.address_output_depth)
#define output_confidence ((float*)params.address_output_confidence)
#define layer_z ((uint*)params.address_layer_z)
#define layer_owner ((uint*)params.address_layer_owner)
#define ratio_histogram ((uint*)params.address_ratio_histogram)
#define exposure_gain ((float*)params.address_exposure_gain)
#define preview_indices ((uint*)params.address_preview_indices)
#define output_rgb_valid ((uint*)params.address_output_rgb_valid)
#define output_approximate ((uint*)params.address_output_approximate)
#define packed_readback ((uint4*)params.address_packed_readback)
#define source_ids ((uint2*)params.address_source_ids)
#define source_ages ((uint2*)params.address_source_ages)
#define output_source_ids ((uint2*)params.address_output_source_ids)
#define output_age_ns ((uint2*)params.address_output_age_ns)
#define src_width (params.field_src_width)
#define src_height (params.field_src_height)
#define target_width (params.field_target_width)
#define target_height (params.field_target_height)
#define projection (params.field_projection)
#define fx (params.field_fx)
#define fy (params.field_fy)
#define cx (params.field_cx)
#define cy (params.field_cy)
#define hfov_radians (params.field_hfov_radians)
#define source_to_virtual_left (params.field_source_to_virtual_left)
#define source_to_virtual_right (params.field_source_to_virtual_right)
#define preview_enabled (params.field_preview_enabled)
#define exposure_enabled (params.field_exposure_enabled)
uint read_metric_rgb(uint index){uint n=src_width*src_height;return index<n?source_rgb[index]:((uint*)params.rectified_right_rgb)[index-n];}
uint read_preview_rgb(uint index){uint n=params.original_width*params.original_height;return (index<n?((uint*)params.original_left_rgb)[index]:((uint*)params.original_right_rgb)[index-n])&0x00ffffff;}
#define SF_TYPED_VIEW 1
#define SF_METRIC_RGB(index) read_metric_rgb(index)
#define SF_PREVIEW_RGB(index) read_preview_rgb(index)
#define SF_PREVIEW_BOUND (2*params.original_width*params.original_height)
#define SF_PREVIEW_OWNER(index) 0xffffffff
#include "../ops/projection.slang"

// Explicit downstream diagnostic consumer. Only four final counters are exported.
[shader("compute")]
[numthreads(64,1,1)]
void inspect_outputs(uint3 tid:SV_DispatchThreadID)
{
    uint t=tid.x;if(t>=target_width*target_height)return;
    uint* counters=(uint*)params.address_packed_readback;
    uint ignored;
    if(output_valid[t]!=0) {
        InterlockedAdd(counters[0],1,ignored);
        uint2 id=output_source_ids[t],age=output_age_ns[t];
        InterlockedXor(counters[2],id.x^id.y^age.x^age.y,ignored);
    }
    if(output_approximate[t]!=0)InterlockedAdd(counters[1],1,ignored);
    if(output_valid[t]==0 && (any(output_source_ids[t]!=uint2(0xffffffff,0xffffffff))||any(output_age_ns[t]!=uint2(0,0))))
        InterlockedAdd(counters[3],1,ignored);
}
