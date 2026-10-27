// Canonical scalar-layout stereo parameter ABI: 168 bytes, shared by all wrappers.
#ifndef SF_STEREO_PARAMETERS
#define SF_STEREO_PARAMETERS
struct Parameters {
    uint64_t left_address,right_address,left_desc_address,right_desc_address,left_valid_address,right_valid_address;
    uint64_t costs_address,sums_address,disparity_address,confidence_address,validity_address,depth_address,depth_valid_address,counters_address;
    uint width,height,disparities,paths,p1,p2,subpixel,metric_enabled;
    float fx,baseline_m,cx_offset;
    uint packing_bits;
    uint64_t penalties_address;
};
#endif
