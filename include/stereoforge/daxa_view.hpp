#pragma once
#include <stereoforge/daxa_foundation.hpp>
#include <stereoforge/geometry.hpp>
#include <memory>
namespace stereoforge {
// Single submission thread. Immutable retained input snapshot/configuration.
// Create a new compositor for changed calibration/shape: live leases retain the
// old context and allocations until their downstream consumers release them.
class DaxaView {
    struct Impl;
    struct Lease;
    std::shared_ptr<Impl> impl_;
public:
    struct Diagnostics {unsigned errors=0,warnings=0;bool closed=false;};
    class Frame {
        std::shared_ptr<Lease> lease_;
        explicit Frame(std::shared_ptr<Lease> lease):lease_(std::move(lease)){}
        friend class DaxaView;
    public:
        Frame()=default;
        void wait() const;
        double gpu_ms() const; // Valid after wait; includes graph barriers, no upload/dump.
        // Borrowed ID for same-context downstream work only. Lease retains storage.
        daxa::BufferId buffer(std::string const& plane) const;
        uint64_t generation() const;
        uint64_t context_id() const;
        int64_t timestamp_ns() const;
        std::string calibration_id() const;
        GeometryConfig config() const;
    };
    static std::string shader_directory();
    static void validate(GeometryConfig,GeometrySources const&,VirtualView const&,unsigned slots);
    DaxaView(GeometryConfig,GeometrySources const&,VirtualView const&,unsigned slots=2,bool validation=false);
    Frame project(); // Bounded: throws if all slots retained or still executing; no wait.
    std::array<uint32_t,4> inspect(Frame const&) const; // Downstream GPU count/hash/default validation.
    GeometryReadback readback(Frame const&) const; // Explicit diagnostic transfer and wait.
    std::string device_report() const;
    uint64_t allocated_bytes() const;
    unsigned validation_errors() const;
    std::shared_ptr<Diagnostics const> diagnostics() const;
};
}
