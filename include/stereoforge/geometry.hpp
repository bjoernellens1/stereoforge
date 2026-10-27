#pragma once
#include <stereoforge/contracts.hpp>
#include <array>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace stereoforge {
using Pose4 = std::array<float,16>; // row-major, source-to-target, metres
inline Pose4 identity_pose() { return {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}; }
struct Ray3 { float x,y,z; }; // undistorted source ray, z=1; tightly packed 12 bytes
struct Pinhole { float fx=1,fy=1,cx=0,cy=0; };
enum class Projection { Perspective, Cylindrical };
enum class ViewMode { Strict, Preview };
struct VirtualView {
    Pinhole intrinsics;
    Projection projection=Projection::Perspective;
    float hfov_radians=1.57079632679f;
    Pose4 source_to_virtual_left=identity_pose(), source_to_virtual_right=identity_pose();
    ViewMode mode=ViewMode::Strict; // Native implementation rejects Preview.
};
struct GeometryConfig { std::uint32_t source_width,source_height,target_width,target_height; };
// Arrays concatenate left then right, each source_width*source_height elements.
// RGB packs R in bits0..7, G in8..15, B in16..23. Invalid depth is NaN or <=0.
struct GeometrySources {
    std::span<const float> depth, confidence;
    std::span<const std::uint32_t> rgb, provenance;
    std::span<const Ray3> rays;
    std::span<const std::int64_t> source_ids, age_ns;
    std::int64_t timestamp_ns=0;
    std::string calibration_id;
};
// Rectified pinhole geometry; all arrays have source_width*source_height elements.
struct TemporalInput {
    std::span<const float> depth, confidence, disparity;
    std::span<const std::uint32_t> validity, provenance;
    std::span<const std::int64_t> source_ids, age_ns;
    std::int64_t timestamp_ns=0;
    std::string calibration_id;
};
struct MetricDisparity { float fx,baseline_m,principal_offset=0; };
enum class GeometryOperation { Project, Reproject, Fuse };
enum class DepthConvention { AxialZ, RayRange };
struct GeometryMetadata;
struct GeometryFrame {
    BufferView depth,confidence,disparity,validity,provenance,rgb,source,owner,choose_previous;
    // Original metadata remains available on GPU; owner/choose_previous identify
    // how to gather it. No claim of direct final-id/age GPU materialization.
    BufferView current_ids,current_ages,previous_ids,previous_ages;
    CompletionToken completion;
    std::uint32_t width=0,height=0;
    GeometryOperation operation=GeometryOperation::Project;
    DepthConvention depth_convention=DepthConvention::AxialZ;
    std::int64_t timestamp_ns=0,observation_elapsed_ns=0;
    std::string calibration_id;
    std::shared_ptr<const GeometryMetadata> metadata;
};
struct GeometryReadback {
    std::vector<float> depth,confidence,disparity;
    std::vector<std::uint32_t> validity,provenance,rgb,source,owner,choose_previous;
    std::vector<std::int64_t> source_ids,age_ns;
};
// Strict nearest-surface projection and temporal kernels; no exposure matching,
// seam blending, or approximate RGB synthesis. Independent device, host upload;
// two leased persistent slots. Submit from one thread; explicit readback blocks.
class NativeGeometry {
    struct Impl;
    std::shared_ptr<Impl> impl_;
public:
    NativeGeometry(GeometryConfig config,const std::string& shader_directory,bool vulkan_validation=false);
    GeometryFrame project(const GeometrySources& sources,const VirtualView& view);
    GeometryFrame reproject(const TemporalInput& previous,Pinhole camera,const Pose4& previous_to_current,
        std::int64_t timestamp_ns,std::optional<MetricDisparity> metric=std::nullopt);
    GeometryFrame fuse(const TemporalInput& current,const TemporalInput& prediction,
        std::int64_t max_age_ns=200000000,float relative_tolerance=.03f);
    GeometryReadback readback(const GeometryFrame& frame) const;
    std::uint64_t device_id() const;
    std::string device_name() const;
};
} // namespace stereoforge
