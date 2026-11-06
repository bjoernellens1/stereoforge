#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <stereoforge/daxa_calibration.hpp>
#include <vector>
namespace stereoforge {
// ROS/PCL RGB is a bit payload in a FLOAT32 field. Coordinates are axial metres
// in the left rectified optical frame: x right, y down, z forward.
struct XyzRgb {
  float x, y, z;
  uint32_t rgb;
};
static_assert(sizeof(XyzRgb) == 16 && offsetof(XyzRgb, rgb) == 12);
inline uint32_t point_cloud_rgb(uint32_t rgba) {
  return ((rgba & 255u) << 16) | (rgba & 0xff00u) | ((rgba >> 16) & 255u);
}
struct PointCloudLayout {
  uint32_t source_width, source_height, stride, width, height, row_step;
  size_t bytes;
  PointCloudLayout(uint32_t w, uint32_t h, uint32_t sampling)
      : source_width(w), source_height(h), stride(sampling) {
    if (!w || !h || w > 8192 || h > 8192 || uint64_t(w) * h > 16'777'216 ||
        !sampling || sampling > 8192)
      throw std::invalid_argument("invalid bounded cloud grid/stride");
    width = 1 + (w - 1) / sampling;
    height = 1 + (h - 1) / sampling;
    row_step = width * sizeof(XyzRgb);
    bytes = size_t(row_step) * height;
  }
  size_t source_index(size_t sampled_index) const {
    if (sampled_index >= size_t(width) * height)
      throw std::invalid_argument("cloud sample outside layout");
    return (sampled_index / width) * stride * source_width +
           (sampled_index % width) * stride;
  }
};
// Calibration setup only. Double arithmetic followed by float32 conversion
// matches the preserved CPU preview ray table; no per-frame CPU reconstruction.
inline std::vector<std::array<float, 2>>
point_cloud_rays(const DaxaCameraModel &k) {
  PointCloudLayout grid(k.width, k.height, 1);
  if (!std::isfinite(k.fx) || !std::isfinite(k.fy) || k.fx <= 0 || k.fy <= 0 ||
      !std::isfinite(k.cx) || !std::isfinite(k.cy))
    throw std::invalid_argument(
        "cloud needs finite positive rectified pinhole intrinsics");
  for (auto d : k.distortion)
    if (!std::isfinite(d) || d != 0)
      throw std::invalid_argument(
          "cloud rays require rectified undistorted input");
  std::vector<std::array<float, 2>> rays(size_t(grid.width) * grid.height);
  for (uint32_t y = 0; y < k.height; ++y)
    for (uint32_t x = 0; x < k.width; ++x) {
      auto ray = std::array{float((double(x) - k.cx) / k.fx),
                            float((double(y) - k.cy) / k.fy)};
      if (!std::isfinite(ray[0]) || !std::isfinite(ray[1]))
        throw std::invalid_argument("cloud ray exceeds float32 domain");
      rays[size_t(y) * k.width + x] = ray;
    }
  return rays;
}
} // namespace stereoforge
