#pragma once
#include <json.hpp>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace stereoforge {
// Reporting only: call after collecting the existing measured vectors. No device
// access, timing queries, or modifications to legacy samples/summaries occur here.
inline void add_native_timing_metadata(nlohmann::json& report, unsigned frames,
    unsigned warmup, std::string_view summary_policy,
    nlohmann::json cold_prepare = nullptr,
    std::initializer_list<std::string_view> inactive = {}) {
    using json = nlohmann::json;
    if(frames<1 || frames>300 || warmup>30)
        throw std::invalid_argument("native timing population outside benchmark bounds");
    auto unavailable=[](char const* reason) { return json{{"status","unavailable"},{"reason",reason}}; };
    json columns=json::object();
    auto column=[&](std::string const& key,json const& values,std::string const& scope) {
        bool disabled=false;
        for(auto name:inactive)if(name==key)disabled=true;
        if(values.is_null()) { columns[key]=unavailable("column_not_measured_for_this_variant_or_schedule");return; }
        if(!values.is_array() || values.size()!=frames)
            throw std::invalid_argument("native timing column population mismatch");
        for(auto const& value:values)
            if(!value.is_number() || !std::isfinite(value.get<double>()) || value.get<double>()<0)
                throw std::invalid_argument("invalid native timing sample");
        columns[key]=disabled?unavailable("inactive_stage_legacy_zero_placeholder"):
            json{{"status","available"},{"value",scope}};
    };
    for(auto const& item:report.at("raw_samples_ms").items()) {
        auto const& key=item.key();
        auto scope=key=="host_wall_ms"?"host_enqueue_start_through_final_wait_and_timestamp_query":
            key=="enqueue_cpu_ms"?"host_admission_retirement_and_submission":
            key=="wait_cpu_ms"?"host_final_submit_wait":
            key=="query_cpu_ms"||key=="timestamp_query_cpu_ms"?"host_timestamp_query_and_validation":
            key=="upload_cpu_ms"?"host_source_upload_submission":
            key=="gpu_combined_queue_span_ms"?"gpu_upload_start_through_final_metric_end_including_queue_gaps":
            "gpu_named_stage_timestamp_interval";
        column(key,item.value(),scope);
    }
    if(report.contains("gpu_paired_stereo_raw_samples_ms"))
        column("gpu_paired_stereo_raw_samples_ms",report.at("gpu_paired_stereo_raw_samples_ms"),"gpu_paired_stereo_timestamp_interval");
    for(char const* group:{"gpu_paired_stereo_phases_raw_samples_ms","stereo_stage_raw_samples_ms"})
        if(report.contains(group))for(auto const& item:report.at(group).items())
            column(std::string(group)+"."+item.key(),item.value(),"gpu_named_stereo_phase_timestamp_interval");
    if(!cold_prepare.is_null() && (!cold_prepare.is_number() ||
        !std::isfinite(cold_prepare.get<double>()) || cold_prepare.get<double>()<0))
        throw std::invalid_argument("invalid native setup duration");
    report["timing_metadata"]={{"schema","stereoforge.native_timing_metadata.v1"},
        {"units","ms"},{"sample_population","measured_iterations_only"},
        {"sample_order","chronological"},{"measured_count",frames},
        {"excluded_warmup_count",warmup},{"phase",warmup?"warm":"unwarmed"},
        {"source_reuse","static_retained_record"},
        {"completion_boundary","final_submit_wait_and_query"},
        {"readback_in_span",false},{"file_io_in_span",false},{"diagnostic_retrieval_in_span",false},
        {"legacy_summary_policy",summary_policy},{"columns",std::move(columns)},
        {"cold_prepare",cold_prepare.is_null()?unavailable("setup_duration_not_measured"):json{{"status","available"},{"value",cold_prepare}}},
        {"cold_prepare_scope",cold_prepare.is_null()?"unavailable":"foundation_and_pipeline_construction_only"},
        {"cache_temperature",unavailable("cache_temperature_not_measured")},
        {"allocation_activity",unavailable("runtime_allocation_activity_not_measured")},
        {"host_output_properties",unavailable("host_output_allocations_outside_timing_span_not_measured")}};
}
} // namespace stereoforge
