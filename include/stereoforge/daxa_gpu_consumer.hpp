#pragma once
#include <stereoforge/daxa_observation.hpp>
#include <cstddef>
namespace stereoforge {
// Same-device, in-process GPU copies. Not ROS IPC or a cross-device handle.
// All operations and destruction of every live Frame, the consumer, and its
// last owners belong to the producer submission thread. Wrong-thread teardown
// terminates the process rather than freeing resources with uncertain ownership. A downstream main-queue submission must be registered
// with retain_until before its Frame lease is released. A slot is reusable only
// after its copy and all registered downstream submissions have completed and
// all Frame leases have been released. A CPU lease alone cannot protect an
// unregistered asynchronous submission after the lease itself is released.
// Only submissions on the shared Daxa QUEUE_MAIN are supported; external queues
// and cross-process handles require an independent synchronization protocol.
class DaxaGpuConsumer {
    struct Slot; struct Impl; std::shared_ptr<Impl> impl_;
public:
    class Frame {
        std::shared_ptr<Impl> owner_; std::shared_ptr<Slot> slot_; uint64_t generation_=0;
        Frame(std::shared_ptr<Impl>,std::shared_ptr<Slot>); void validate()const;
        friend class DaxaGpuConsumer;
    public:
        Frame()=default; Frame(const Frame&)=default; Frame(Frame&&)noexcept=default;
        ~Frame()noexcept;
        Frame& operator=(Frame)noexcept;
        explicit operator bool()const;
        daxa::BufferId depth()const; daxa::BufferId rgba()const;
        uint32_t width()const; uint32_t height()const;
        uint64_t producer_ticket()const; uint64_t copy_ticket()const;
        bool ready()const;
        // Trusted caller contract: register every downstream submission on the same
        // Daxa QUEUE_MAIN that uses these buffers before releasing this lease.
        // A queue ticket alone cannot prove which resources commands reference.
        // If submission outcome is uncertain, retain the lease until resolved;
        // an uncertain ticket may never signal and teardown then fails closed.
        void retain_until(uint64_t consumer_ticket);
        CaptureLease source()const; std::shared_ptr<const DaxaRectification> calibration()const;
    };
    DaxaGpuConsumer(std::shared_ptr<DaxaFoundation>,const DaxaObservationFrame&,unsigned slots=2);
    ~DaxaGpuConsumer()noexcept;
    // False means every bounded slot is retained or pending. Wrong-thread and
    // device/query failures throw so callers cannot mistake faults for drops.
    bool has_capacity()const;
    Frame enqueue(const DaxaObservationFrame&);
    size_t collect(); void drain(); size_t requested_bytes()const;
};
}
