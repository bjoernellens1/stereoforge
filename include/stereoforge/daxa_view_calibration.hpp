#pragma once
#include <stereoforge/daxa_preprocess.hpp>
#include <stereoforge/geometry.hpp>
namespace stereoforge {
// Setup-only OpenCV4.14 initUndistortRectifyMap/convertMaps effective 1/32-pixel
// association. This validates supplied models/maps, not physical calibration truth.
struct DaxaViewCalibration {
    std::vector<Ray3> rays;
    std::vector<uint32_t> preview_indices;
    uint32_t original_width=0,original_height=0;
    uint64_t preview_valid_count=0;
    VirtualView effective_view;
    std::string association="opencv4.14-effective-quantized-map-model-validation";
};
DaxaViewCalibration prepare_daxa_view_calibration(const DaxaRectification&,GeometryConfig,VirtualView const&);
// Setup-only typed source admission. Canonical RGB validates retained camera/grid
// association without requiring raw YUYV remap tables or a packed source stride.
DaxaViewCalibration prepare_daxa_view_calibration(const DaxaRectification&,GeometryConfig,VirtualView const&,const CaptureFrame&);
}
