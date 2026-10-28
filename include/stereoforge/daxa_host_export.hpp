#pragma once
#include <stereoforge/daxa_observation.hpp>
#include <cstddef>
#include <span>
namespace stereoforge {
struct DaxaHostExportConfig { unsigned slots=2; bool depth=true; bool rgb=false; };
struct DaxaHostExportMemory { uint32_t type_index=0,property_flags=0; size_t bytes_per_slot=0; bool host_cached=false; };
// Selective native publication boundary, not a freshness or ROS delivery claim.
// Setup/enqueue/collect/drain share one submission thread with native producers.
class DaxaHostExport {
    struct Slot;struct Impl;std::shared_ptr<Impl> impl_;
public:
    class Frame {
        // Children must be released before the resource/context owner.
        std::shared_ptr<Impl> owner_;std::shared_ptr<Slot> slot_;uint64_t generation_=0;
        Frame(std::shared_ptr<Impl>,std::shared_ptr<Slot>);void validate()const;
        friend class DaxaHostExport;
    public:
        Frame()=default;Frame(const Frame&)=default;Frame(Frame&&)noexcept=default;
        Frame& operator=(Frame)noexcept;
        explicit operator bool()const;
        bool ready()const;void wait()const;
        std::span<const std::byte> depth_bytes()const;std::span<const std::byte> rgba_bytes()const;
        uint32_t width()const;uint32_t height()const;
        uint64_t producer_ticket()const;uint64_t copy_ticket()const;
        CaptureLease source()const;std::shared_ptr<const DaxaRectification> calibration()const;
    };
    DaxaHostExport(std::shared_ptr<DaxaFoundation>,const DaxaObservationFrame&,DaxaHostExportConfig);
    DaxaHostExport(std::shared_ptr<DaxaFoundation>,const DaxaFrame&,DaxaHostExportConfig);
    Frame enqueue(const DaxaObservationFrame&);Frame enqueue(const DaxaFrame&);
    size_t collect();void drain();
    DaxaHostExportMemory memory()const;size_t requested_bytes()const;
};
}
