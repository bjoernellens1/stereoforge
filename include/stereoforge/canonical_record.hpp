#pragma once

#include <stereoforge/canonical_rgb.hpp>
#include <stereoforge/daxa_calibration.hpp>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace stereoforge {

struct CanonicalRecordFile {
    std::filesystem::path path; // Absolute checked bundle path.
    std::string sha256;
    uint64_t size_bytes = 0;
};

struct CanonicalRecordInput {
    RectifiedRgbFrame frame;
    DaxaRectifiedGeometry full_geometry, coarse_geometry;
    uint64_t ordinal = 0;
    std::string manifest_sha256, record_sha256;
    std::string geometry_supplement_sha256, coarse_calibration_sha256;
    std::string producer;
    std::vector<CanonicalRecordFile> files;
};

// Cold admission retains immutable plane bytes through frame's sidecar backing.
CanonicalRecordInput load_canonical_record(const std::filesystem::path& manifest);

// Explicit post-run boundary: rehash every file checked during admission.
void verify_canonical_record_files(const CanonicalRecordInput&);

} // namespace stereoforge
