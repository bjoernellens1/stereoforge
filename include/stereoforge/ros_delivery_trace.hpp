#pragma once
// Bounded POD telemetry for a single native submission or ROS callback thread.
// JSON serialization and file writes belong after the measured loop.
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace stereoforge {
enum class RosThreadCpuStatus : std::uint8_t { Disabled, Available, ClockReadFailed, Nonmonotonic };
enum class RosDeliveryTraceKind : std::uint8_t { Published, Received };
enum class RosDeliveryOutput : std::uint8_t { Depth, Rgb };
enum class RosDeliveryFreshness : std::uint8_t { MeasuredSingleSource, Historical, Unknown };
struct RosDeliveryTraceRecord {
    RosDeliveryTraceKind kind=RosDeliveryTraceKind::Published;
    RosDeliveryOutput output=RosDeliveryOutput::Depth;
    RosDeliveryFreshness freshness=RosDeliveryFreshness::Unknown;
    RosThreadCpuStatus thread_cpu_status=RosThreadCpuStatus::Disabled;
    std::uint64_t thread_cpu_copy_begin_ns=0,thread_cpu_copy_end_ns=0,thread_cpu_publish_begin_ns=0,thread_cpu_publish_end_ns=0;
    std::int64_t source_id=0;
    std::uint64_t at_ns=0,capture_ns=0,host_ingress_ns=0,mapped_capture_ns=0,mapping_error_ns=0;
    std::uint64_t producer_ticket=0,copy_ticket=0,producer_complete_observed_ns=0,copy_complete_observed_ns=0;
    std::uint64_t producer_enqueue_begin_ns=0,producer_enqueue_end_ns=0,export_enqueue_end_ns=0;
    std::uint64_t copy_begin_ns=0,copy_end_ns=0,publish_begin_ns=0,publish_end_ns=0,arrival_system_ns=0;
    std::uint32_t capture_queue=0,pending=0,width=0,height=0,step=0;
    std::uint64_t payload_bytes=0;
};
static_assert(std::is_trivially_copyable_v<RosDeliveryTraceRecord>);
class RosDeliveryTrace {
    std::vector<RosDeliveryTraceRecord> records_;
    std::size_t limit_;
    std::uint64_t dropped_=0;
public:
    // The event limit includes the separately serialized start and stop events.
    explicit RosDeliveryTrace(std::size_t event_capacity):limit_(event_capacity) {
        if(event_capacity<2 || event_capacity>2'000'000)throw std::invalid_argument("invalid trace event capacity");
        records_.reserve(event_capacity-2);
    }
    RosDeliveryTrace(const RosDeliveryTrace&)=delete;
    RosDeliveryTrace& operator=(const RosDeliveryTrace&)=delete;
    bool append(const RosDeliveryTraceRecord& record) noexcept {
        if(records_.size()>=limit_-2) {
            if(dropped_!=std::numeric_limits<std::uint64_t>::max())++dropped_;
            return false;
        }
        records_.push_back(record);
        return true;
    }
    std::span<const RosDeliveryTraceRecord> records() const noexcept {return records_;}
    std::size_t event_capacity() const noexcept {return limit_;}
    // Requested payload only; allocator overhead/rounding are not represented.
    std::size_t requested_bytes() const noexcept {return (limit_-2)*sizeof(RosDeliveryTraceRecord);}
    std::uint64_t dropped() const noexcept {return dropped_;}
};
}
