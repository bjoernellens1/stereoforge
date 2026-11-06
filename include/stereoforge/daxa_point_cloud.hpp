#pragma once
#include <span>
#include <stereoforge/daxa_observation.hpp>
#include <stereoforge/point_cloud_contract.hpp>
#include <string>
namespace stereoforge {
struct DaxaPointCloudConfig {
  unsigned slots = 2;
  uint32_t export_stride = 0;
}; // 0: no CPU export
struct DaxaPointCloudMemory {
  size_t resident_bytes = 0, export_bytes = 0;
  uint32_t export_memory_type = 0, export_memory_flags = 0;
};
struct DaxaPointCloudTiming {
  double generation_gpu_ms = 0, export_gpu_ms = 0;
};
// Same shared Daxa device/main queue and owner thread as the observation
// producer. Full-resolution XYZRGB remains resident. Optional sampled host
// output is an explicit publication boundary. Geometry never infers depth or
// changes source confidence, provenance or age; those remain on the retained
// observations.
class DaxaPointCloud {
  struct Slot;
  struct Impl;
  std::shared_ptr<Impl> impl_;

public:
  class Frame {
    std::shared_ptr<Impl> owner_;
    std::shared_ptr<Slot> slot_;
    uint64_t generation_ = 0;
    Frame(std::shared_ptr<Impl>, std::shared_ptr<Slot>);
    void validate() const;
    friend class DaxaPointCloud;

  public:
    Frame() = default;
    Frame(const Frame &) = default;
    Frame(Frame &&) noexcept = default;
    ~Frame() noexcept;
    Frame &operator=(Frame) noexcept;
    explicit operator bool() const;
    daxa::BufferId buffer() const;
    size_t point_bytes() const;
    uint32_t width() const;
    uint32_t height() const;
    DaxaObservationFrame observations() const;
    uint64_t submit_ticket() const;
    bool ready() const;
    void wait() const;
    bool has_export() const;
    uint32_t export_width() const;
    uint32_t export_height() const;
    std::span<const std::byte> export_bytes() const;
    DaxaPointCloudTiming timings() const;
    // Trusted caller: register every downstream same-main-queue read before
    // releasing its lease. A ticket cannot attest which resources it reads.
    void retain_until(uint64_t downstream_ticket);
  };
  static std::string shader_directory();
  DaxaPointCloud(std::shared_ptr<DaxaFoundation>, const DaxaObservationFrame &,
                 DaxaPointCloudConfig = {});
  ~DaxaPointCloud() noexcept;
  // Generate full-resolution geometry each time; suppress an unrequested CPU
  // export.
  Frame enqueue(const DaxaObservationFrame &, bool request_export = true);
  bool has_capacity() const;
  size_t collect();
  void drain();
  size_t requested_export_bytes() const;
  DaxaPointCloudMemory memory() const;
};
} // namespace stereoforge
