#pragma once
// SDK-free opaque API sampler. No completion fence is implied by API return.
#include <json.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

namespace sf::zed_benchmark {
template<class Code> struct Samples {
    std::array<double, 100> samples{};
    int warmup_completed = 0;
    int measured_count = 0;
    std::optional<Code> failure;
};

template<class Code, class HostClock = std::chrono::steady_clock, class Grab>
Samples<Code> sample(int warmup, int frames, Grab&& grab, Code success) {
    if (warmup < 0 || warmup > 100 || frames < 1 || frames > 100)
        throw std::invalid_argument("sampler counts outside bounds");
    Samples<Code> result;
    for (int ordinal = 0; ordinal < warmup + frames; ++ordinal) {
        const auto start = HostClock::now();
        const Code code = grab();
        const auto finish = HostClock::now();
        if (code != success) { result.failure = code; break; }
        if (ordinal < warmup) ++result.warmup_completed;
        else result.samples[result.measured_count++] = std::chrono::duration<double, std::milli>(finish - start).count();
    }
    return result;
}

// Post-loop report construction, copies/sorting and serialization stay outside sampling.
template<class Code>
nlohmann::json report(const Samples<Code>& result, int warmup, int frames, bool depth) {
    using Json = nlohmann::json;
    const auto unavailable = [](const char* reason) { return Json{{"available", false}, {"reason", reason}}; };
    std::vector<double> raw(result.samples.begin(), result.samples.begin() + result.measured_count);
    Json summary = unavailable("no successful measured grab samples");
    if (!raw.empty()) {
        auto sorted = raw;
        std::sort(sorted.begin(), sorted.end());
        double sum = 0; for (double value : raw) sum += value;
        const auto rank = [&](double fraction) { return sorted[static_cast<size_t>(std::ceil(fraction * sorted.size())) - 1]; };
        summary = {{"mean_ms", sum / raw.size()}, {"p50_ms", rank(.50)}, {"p95_ms", rank(.95)},
            {"p99_ms", rank(.99)}, {"max_ms", sorted.back()}, {"policy", "nearest_rank"},
            {"tail_guarantee", false}};
    }
    const auto internal = unavailable("SDK internal GPU stages and GPU-wide completion are unproven");
    return {{"schema", "stereoforge.zed_comparison.sdk_timing.v1"}, {"qualification", "unqualified"},
        {"operation", "benchmark"}, {"timing_lane", "opaque-api"}, {"lane", "opaque_api"},
        {"completion_boundary", "opaque_api_return"}, {"gpu_completion", internal},
        {"scope", "opaque blocking grab API wall time; GPU-wide completion unproven"},
        {"complete_wall_scope", "same host API-return span as opaque_grab_api_ms; not GPU completion"},
        {"requested_outputs", Json::array({depth ? "sdk_depth_api" : "no_depth_capture"})},
        {"depth_enabled", depth}, {"warmup_requested", warmup}, {"warmup_completed", result.warmup_completed},
        {"excluded_warmup_count", result.warmup_completed}, {"frames_requested", frames},
        {"sample_count", result.measured_count}, {"phase", result.warmup_completed > 0 ? "warm" : "unwarmed"},
        {"sample_population", "successful_measured_grabs_only"}, {"sample_order", "chronological"}, {"units", "ms"},
        {"opaque_grab_api_ms", raw}, {"complete_wall_ms", raw}, {"summary", summary},
        {"gpu_total_ms", internal}, {"gpu_upload_ms", internal}, {"gpu_readback_ms", internal},
        {"gpu_stages", internal}, {"cpu_retrieval_api_ms", unavailable("no diagnostic retrieval performed")},
        {"file_io_in_span", false}, {"diagnostic_retrieval_in_span", false}, {"playback_pacing", false},
        {"repeated_frame_identity", unavailable("per-grab identity not queried in timing-only loop")},
        {"replay_identity", unavailable("per-grab replay identity not verified")},
        {"cache_temperature", unavailable("not measured")}, {"allocation_activity", unavailable("SDK internal activity not measured")},
        {"host_output_properties", unavailable("no CPU output requested")}};
}
} // namespace sf::zed_benchmark
