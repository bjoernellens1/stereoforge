#pragma once
#include <stereoforge/ros_delivery_trace.hpp>
#if __has_include(<json.hpp>)
#include <json.hpp>
#else
#include <nlohmann/json.hpp>
#endif
#include <stdexcept>
#include <string_view>

namespace stereoforge {
// Post-loop serialization only. Caller validates actual ROS message metadata
// against the configured layout before appending a received record.
inline nlohmann::json ros_delivery_event_json(const RosDeliveryTraceRecord& r,
    std::string_view run_id,std::string_view role,const nlohmann::json& config) {
    using nlohmann::json;
    if(role!="publisher" && role!="subscriber")throw std::invalid_argument("invalid trace role");
    if((role=="publisher")!=(r.kind==RosDeliveryTraceKind::Published))throw std::invalid_argument("trace role/kind mismatch");
    if(r.output!=RosDeliveryOutput::Depth && r.output!=RosDeliveryOutput::Rgb)throw std::invalid_argument("invalid trace output");
    const char* output=r.output==RosDeliveryOutput::Depth?"depth":"rgb";
    json e={{"schema","stereoforge.ros_delivery_event.v1"},{"kind",role=="publisher"?"published":"received"},
            {"run_id",run_id},{"role",role},{"at_ns",r.at_ns},{"output",output},{"mapped_capture_ns",r.mapped_capture_ns}};
    if(role=="publisher") {
        const char* freshness=r.freshness==RosDeliveryFreshness::MeasuredSingleSource?"measured_single_source":
            r.freshness==RosDeliveryFreshness::Historical?"historical":"unknown";
        e.update(json{{"source_id",r.source_id},{"capture_ns",r.capture_ns},{"capture_clock",config.at("clock_mapping").at("source_clock")},
            {"host_ingress_ns",r.host_ingress_ns},{"mapping_error_ns",r.mapping_error_ns},{"producer_ticket",r.producer_ticket},
            {"copy_ticket",r.copy_ticket},{"producer_enqueue_begin_ns",r.producer_enqueue_begin_ns},
            {"producer_enqueue_end_ns",r.producer_enqueue_end_ns},{"export_enqueue_end_ns",r.export_enqueue_end_ns},
            {"producer_complete_observed_ns",r.producer_complete_observed_ns},
            {"copy_complete_observed_ns",r.copy_complete_observed_ns},{"copy_begin_ns",r.copy_begin_ns},
            {"copy_end_ns",r.copy_end_ns},{"publish_begin_ns",r.publish_begin_ns},{"publish_end_ns",r.publish_end_ns},
            {"capture_queue",r.capture_queue},{"pending",r.pending},{"freshness",freshness}});
        if(config.value("thread_cpu_timing",false)) {
            if(r.thread_cpu_status==RosThreadCpuStatus::Available)
                e["thread_cpu_timing"]={{"status","available"},{"copy_begin_ns",r.thread_cpu_copy_begin_ns},
                    {"copy_end_ns",r.thread_cpu_copy_end_ns},{"publish_begin_ns",r.thread_cpu_publish_begin_ns},
                    {"publish_end_ns",r.thread_cpu_publish_end_ns}};
            else if(r.thread_cpu_status==RosThreadCpuStatus::ClockReadFailed || r.thread_cpu_status==RosThreadCpuStatus::Nonmonotonic)
                e["thread_cpu_timing"]={{"status","unavailable"},{"reason",r.thread_cpu_status==RosThreadCpuStatus::ClockReadFailed?"clock_read_failed":"nonmonotonic"}};
            else throw std::invalid_argument("missing thread cpu sample");
        } else if(r.thread_cpu_status!=RosThreadCpuStatus::Disabled)throw std::invalid_argument("undeclared thread cpu sample");
    } else {
        auto const& output_config=config.at("outputs").at(output);
        e.update(json{{"arrival_system_ns",r.arrival_system_ns},{"width",r.width},{"height",r.height},{"step",r.step},
            {"encoding",output_config.at("encoding")},{"frame_id",output_config.at("frame_id")},{"payload_bytes",r.payload_bytes}});
    }
    return e;
}
}
