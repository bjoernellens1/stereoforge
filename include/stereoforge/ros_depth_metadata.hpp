#pragma once
#include <stereoforge/fixed_json_uint_fields.hpp>
namespace stereoforge {
struct RosDepthDiagnosticStamp {
  int64_t source_id;
  uint64_t capture_ns, ingress_ns, header_ns, images_published_ns, pack_ns;
};
// Exact quoted zero-padded uint64 fields, without per-frame JSON
// allocation/dump.
class RosDepthDiagnosticMetadata {
  FixedJsonUintFields<6> fields_;

public:
  static constexpr std::string_view placeholder() {
    return FixedJsonUintFields<6>::placeholder();
  }
  explicit RosDepthDiagnosticMetadata(std::string value)
      : fields_(std::move(value),
                {"source_id", "capture_timestamp_ns",
                 "host_ingress_timestamp_ns", "mapped_header_stamp_ns",
                 "image_publish_calls_completed_ns", "diagnostic_pack_ns"}) {}
  std::string_view template_bytes() const { return fields_.template_bytes(); }
  void stamp(std::span<char> output,
             const RosDepthDiagnosticStamp &value) const {
    if (value.source_id < 0 || !value.capture_ns || !value.ingress_ns ||
        value.images_published_ns < value.ingress_ns)
      throw std::invalid_argument("invalid diagnostic source/timestamps");
    fields_.stamp(output, {uint64_t(value.source_id), value.capture_ns,
                           value.ingress_ns, value.header_ns,
                           value.images_published_ns, value.pack_ns});
  }
};
} // namespace stereoforge
