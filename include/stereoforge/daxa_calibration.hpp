#pragma once
#include <stereoforge/contracts.hpp>
#include <array>
#include <string>
#include <span>
#include <vector>
namespace stereoforge {
struct DaxaCameraModel {
    uint32_t width=0,height=0;
    double fx=0,fy=0,cx=0,cy=0;
    std::vector<double> distortion;
};
struct DaxaStereoCalibration {
    std::array<DaxaCameraModel,2> eyes;
    std::array<double,16> right_to_left{}; // Row-major optical pose, metres.
};
struct DaxaRectifiedGeometry {
    DaxaStereoCalibration original,rectified;
    std::array<double,9> R1{},R2{};
    std::array<double,12> P1{},P2{};
    std::string original_identity,rectified_identity;
    // Setup only. Rotations/pinholes and dimensionless baseline-normalized
    // translations/lengths use abs1e-6+rel1e-5*abs(expected). Normalize before
    // rotating t. Affine bottom rows retain abs1e-8+rel1e-5*abs(expected).
    // Projection fourth-column structural zeros and rectified distortion are
    // exactly zero; P2 Tx is checked by its ratio to -fx*rectified baseline.
    void validate() const;
    MetricCalibration metric_calibration() const;
    // Setup-only OpenCV effective float/nearest-even 1/32-pixel association.
    // Maps are Lx,Ly,Rx,Ry on this geometry's rectified grid.
    void validate_map_association(std::array<std::span<const float>,4>) const;
};
namespace detail {
// Shared with View preview; preserve the calibrated projection arithmetic.
std::array<double,2> daxa_calibration_distort(const DaxaCameraModel&,double x,double y);
}
}
