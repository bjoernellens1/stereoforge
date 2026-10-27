#pragma once
#include <stereoforge/daxa_foundation.hpp>
#include <stereoforge/geometry.hpp>
#include <stereoforge/capture.hpp>
#ifdef SF_DAXA_VIEW_HAS_DEPTH
#include <stereoforge/daxa_depth.hpp>
#endif
#include <memory>
namespace stereoforge {
// Single submission thread. Immutable retained input snapshot/configuration.
// Create a new compositor for changed calibration/shape: live leases retain the
// old context and allocations until their downstream consumers release them.
class DaxaView {
    struct Impl;
    struct Slot;
    std::shared_ptr<Impl> impl_;
public:
    struct Diagnostics {unsigned errors=0,warnings=0;bool closed=false;};
    class Frame {
        std::shared_ptr<Impl> owner_;std::shared_ptr<Slot> slot_;uint64_t generation_=0;
        Frame(std::shared_ptr<Impl>,std::shared_ptr<Slot>);void validate()const;
        friend class DaxaView;
    public:
        Frame()=default;
        Frame(Frame const&)=default;Frame(Frame&&)noexcept=default;
        Frame& operator=(Frame)noexcept;
        void wait() const;
        double gpu_ms() const; // Valid after wait; includes graph barriers, no upload/dump.
        // Borrowed ID for same-context downstream work only. Lease retains storage.
        daxa::BufferId buffer(std::string const& plane) const;
        uint64_t generation() const;
        uint64_t context_id() const;
        int64_t timestamp_ns() const;
        std::string calibration_id() const;
        GeometryConfig config() const;
        CaptureLease source()const;
        DepthConvention depth_convention()const;
        uint64_t start_tick()const;uint64_t end_tick()const;
    };
    static std::string shader_directory();
    static void validate(GeometryConfig,GeometrySources const&,VirtualView const&,unsigned slots);
    DaxaView(GeometryConfig,GeometrySources const&,VirtualView const&,unsigned slots=2,bool validation=false);
#ifdef SF_DAXA_VIEW_HAS_DEPTH
    DaxaView(GeometryConfig,VirtualView const&,std::shared_ptr<DaxaFoundation>,DaxaDepthPipeline::Frame const& prototype,unsigned slots=2);
    Frame project(DaxaDepthPipeline::Frame const&);
#endif
    size_t collect();bool can_project()const;
    static std::string depth_shader_identity();
    Frame project(); // Bounded: throws if all slots retained or still executing; no wait.
    std::array<uint32_t,4> inspect(Frame const&) const; // Downstream GPU count/hash/default validation.
    GeometryReadback readback(Frame const&) const; // Explicit diagnostic transfer and wait.
    std::string device_report() const;
    uint64_t allocated_bytes() const;
    unsigned validation_errors() const;
    std::shared_ptr<Diagnostics const> diagnostics() const;
};
}
