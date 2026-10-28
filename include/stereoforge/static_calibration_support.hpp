#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace stereoforge {

// Calibration-only masks: these say nothing about dynamic frame support, depth,
// occlusion or matching validity. Map order is left x/y, then right x/y.
// Source dimensions are generic: callers must independently validate their
// geometry, including DaxaRectification::validate when applicable.
struct StaticCalibrationSupport {
    std::array<std::vector<std::uint8_t>, 2> sampling;
    std::array<std::vector<std::uint8_t>, 2> census5x5;
};

inline StaticCalibrationSupport derive_static_calibration_support(
    std::uint32_t source_width, std::uint32_t source_height,
    std::uint32_t width, std::uint32_t height,
    const std::array<std::vector<float>, 4>& maps) {
    if (!source_width || !source_height || !width || !height ||
        source_width >= 32767 || source_height >= 32767 ||
        width >= 32767 || height >= 32767)
        throw std::invalid_argument("invalid calibration support dimensions");
    const std::size_t n = std::size_t(width) * height;
    for (const auto& map : maps) {
        if (map.size() != n) throw std::invalid_argument("calibration map shape mismatch");
        for (float value : map)
            if (!std::isfinite(value) || std::abs(value) > 32768.f)
                throw std::invalid_argument("invalid calibration map coordinate");
    }

    // Mirrors preprocess_quantize: float multiplication, then round to nearest,
    // ties to even. Map coordinates were validated before any mask allocation.
    const auto quantize = [](float coordinate) {
        const float scaled = coordinate * 32.f;
        const float base = std::floor(scaled);
        const float fraction = scaled - base;
        const int value = static_cast<int>(base);
        return value + int(fraction > .5f || (fraction == .5f && (value & 1)));
    };
    StaticCalibrationSupport result;
    for (std::size_t eye = 0; eye < 2; ++eye) {
        auto& sampling = result.sampling[eye];
        auto& census = result.census5x5[eye];
        sampling.assign(n, 0);
        census.assign(n, 0);
        for (std::size_t i = 0; i < n; ++i) {
            const int qx = quantize(maps[eye * 2][i]);
            const int qy = quantize(maps[eye * 2 + 1][i]);
            // Arithmetic shift by five in Slang is floor division, including negatives.
            const int x = qx >= 0 ? qx / 32 : -((-qx + 31) / 32);
            const int y = qy >= 0 ? qy / 32 : -((-qy + 31) / 32);
            const unsigned fx = unsigned(qx - x * 32), fy = unsigned(qy - y * 32);
            const unsigned weights[4] = {(32-fx)*(32-fy), fx*(32-fy), (32-fx)*fy, fx*fy};
            bool supported = true;
            for (unsigned tap = 0; tap < 4; ++tap) {
                const int xx = x + int(tap & 1), yy = y + int(tap >> 1);
                if (weights[tap] && (xx < 0 || yy < 0 || xx >= int(source_width) || yy >= int(source_height))) supported = false;
            }
            sampling[i] = std::uint8_t(supported);
        }
        for (std::uint32_t y = 2; y + 2 < height; ++y)
            for (std::uint32_t x = 2; x + 2 < width; ++x) {
                bool valid = true;
                for (int dy = -2; dy <= 2; ++dy)
                    for (int dx = -2; dx <= 2; ++dx)
                        valid &= sampling[std::size_t(int(y) + dy) * width + std::size_t(int(x) + dx)] != 0;
                census[std::size_t(y) * width + x] = std::uint8_t(valid);
            }
    }
    return result;
}
} // namespace stereoforge
