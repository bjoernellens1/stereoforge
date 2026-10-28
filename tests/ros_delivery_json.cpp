#include <stereoforge/ros_delivery_json.hpp>
#include <iostream>
#include <stdexcept>
using namespace stereoforge;
int main(){
    using nlohmann::json;
    json c={{"outputs",{{"depth",{{"encoding","32FC1"},{"frame_id","left_optical"}}}}}};
    RosDeliveryTraceRecord r{};r.kind=RosDeliveryTraceKind::Received;r.output=RosDeliveryOutput::Depth;
    r.at_ns=200;r.mapped_capture_ns=123;r.arrival_system_ns=300;r.width=2;r.height=1;r.step=8;r.payload_bytes=8;
    auto e=ros_delivery_event_json(r,"run","subscriber",c);
    if(e.at("kind")!="received"||e.at("frame_id")!="left_optical"||e.at("payload_bytes")!=8)throw std::runtime_error("serialization mismatch");
    bool rejected=false;try{ros_delivery_event_json(r,"run","publisher",c);}catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)throw std::runtime_error("role mismatch accepted");
    c["clock_mapping"]={{"source_clock","camera"}};
    r.kind=RosDeliveryTraceKind::Published;r.source_id=42;r.capture_ns=111;r.host_ingress_ns=122;
    r.producer_enqueue_begin_ns=124;r.producer_enqueue_end_ns=125;r.export_enqueue_end_ns=126;
    r.mapping_error_ns=7;r.producer_ticket=3;r.copy_ticket=4;r.producer_complete_observed_ns=130;
    r.copy_complete_observed_ns=131;r.copy_begin_ns=132;r.copy_end_ns=133;r.publish_begin_ns=134;r.publish_end_ns=200;
    r.freshness=RosDeliveryFreshness::MeasuredSingleSource;
    auto published=ros_delivery_event_json(r,"run","publisher",c);
    if(published.at("source_id")!=42||published.at("copy_ticket")!=4||published.at("capture_clock")!="camera")throw std::runtime_error("publisher serialization mismatch");
    c["thread_cpu_timing"]=true;
    r.thread_cpu_status=RosThreadCpuStatus::Available;
    r.thread_cpu_copy_begin_ns=10;r.thread_cpu_copy_end_ns=20;
    r.thread_cpu_publish_begin_ns=21;r.thread_cpu_publish_end_ns=41;
    auto cpu=ros_delivery_event_json(r,"run","publisher",c).at("thread_cpu_timing");
    if(cpu.at("status")!="available"||cpu.at("publish_end_ns")!=41)throw std::runtime_error("thread cpu serialization mismatch");
    r.thread_cpu_status=RosThreadCpuStatus::ClockReadFailed;
    cpu=ros_delivery_event_json(r,"run","publisher",c).at("thread_cpu_timing");
    if(cpu.at("status")!="unavailable"||cpu.at("reason")!="clock_read_failed")throw std::runtime_error("thread cpu failure mismatch");
    std::cout<<json::array({e,published}).dump()<<'\n';
}
