// Optional proprietary-SDK reference tool. Diagnostic captures are never canonical records.
#include <sl/Camera.hpp>
#include <json.hpp>
#include "export_io.hpp"
#include "sdk_benchmark.hpp"
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef __linux__
#include <sys/utsname.h>
#endif

#ifndef SF_ZED_EXPORT_SOURCE_SHA256
#define SF_ZED_EXPORT_SOURCE_SHA256 "unavailable"
#endif
#ifndef SF_ZED_EXPORT_IO_SHA256
#define SF_ZED_EXPORT_IO_SHA256 "unavailable"
#endif
#ifndef SF_ZED_EXPORT_BENCHMARK_SHA256
#define SF_ZED_EXPORT_BENCHMARK_SHA256 "unavailable"
#endif
#ifndef SF_ZED_EXPORT_COMMIT
#define SF_ZED_EXPORT_COMMIT "unavailable"
#endif
#ifndef SF_ZED_EXPORT_DIRTY
#define SF_ZED_EXPORT_DIRTY "unavailable"
#endif

namespace fs = std::filesystem;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(std::endian::native == std::endian::little, "F32LE export requires little-endian host");

namespace {
Json unavailable(const char* reason) { return {{"available", false}, {"reason", reason}}; }
std::string sdk_string(const sl::String& value) { return value.c_str(); }
using namespace sf::zed_io;
double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void write_metadata(const fs::path& root, const Json& data) {
    std::ofstream out;
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.open(root / "diagnostic.json", std::ios::binary | std::ios::trunc);
    out << data.dump(2) << '\n'; out.flush(); out.close();
}
fs::path fresh_root(const fs::path& requested) {
    const fs::path root = fs::absolute(requested).lexically_normal();
    require(!root.filename().empty(), "output must name a fresh directory");
    fs::path cursor;
    for (const auto& part : root) {
        cursor /= part;
        require(!fs::is_symlink(fs::symlink_status(cursor)), "output path contains a symlink");
    }
    require(!fs::exists(fs::symlink_status(root)), "output root already exists");
    require(fs::is_directory(root.parent_path()), "output parent must already exist");
    require(fs::create_directory(root), "cannot create fresh output root");
    return root;
}

struct Options {
    std::string operation = "export", svo, output, mode = "NONE", profile = "instantaneous", settings_dir;
    std::string timing_lane = "opaque-api";
    int frames = 2, warmup = 0;
    bool self_calibration = true;
    float minimum = 0.3f, maximum = 8.0f;
};
void help() {
    std::cout << "export_sdk --operation capture-diagnostic|benchmark --svo FILE --output FRESH_DIR\n"
        "  [--mode NONE|NEURAL_LIGHT|NEURAL|NEURAL_PLUS|PERFORMANCE] [--frames 1..100]\n"
        "  [--profile instantaneous] [--self-calibration on|off] [--settings-dir PRIVATE_DIR]\n"
        "  [--warmup 0..100 (benchmark default 5)] [--timing-lane opaque-api (benchmark only)]\n"
        "  [--depth-min 0.3] [--depth-max 8] | --version | --help\n"
        "Diagnostic output is unqualified; canonical export requires proven raw pose and rectification support.\n";
}
Options parse(int argc, char** argv) {
    Options o;
    std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        require(seen.insert(key).second, "duplicate option");
        require(i + 1 < argc, "option requires a value");
        const std::string value = argv[++i];
        if (key == "--operation") o.operation = value;
        else if (key == "--svo") o.svo = value;
        else if (key == "--output") o.output = value;
        else if (key == "--mode") o.mode = value;
        else if (key == "--timing-lane") o.timing_lane = value;
        else if (key == "--profile") o.profile = value;
        else if (key == "--settings-dir") o.settings_dir = value;
        else if (key == "--self-calibration") {
            require(value == "on" || value == "off", "self-calibration must be on or off");
            o.self_calibration = value == "on";
        } else if (key == "--frames" || key == "--warmup") {
            require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos, "invalid frame count");
            size_t used = 0; const int count = std::stoi(value, &used);
            require(used == value.size() && count >= (key == "--frames" ? 1 : 0) && count <= 100, "frames must be 1..100; warmup must be 0..100");
            if (key == "--frames") o.frames = count; else o.warmup = count;
        } else if (key == "--depth-min" || key == "--depth-max") {
            size_t used = 0; const float number = std::stof(value, &used);
            require(used == value.size() && std::isfinite(number), "invalid depth range");
            if (key == "--depth-min") o.minimum = number; else o.maximum = number;
        } else throw SafeError("unknown option");
    }
    require(o.operation == "capture-diagnostic" || o.operation == "benchmark", "canonical export requires proven raw pose and rectification support; use --operation capture-diagnostic");
    require(o.timing_lane == "opaque-api", "unsupported timing lane; only opaque-api is implemented; GPU-wide completion unavailable");
    if (o.operation == "benchmark" && !seen.contains("--warmup")) o.warmup = 5;
    if (o.operation == "capture-diagnostic")
        require(!seen.contains("--warmup") && !seen.contains("--timing-lane"), "timing options require --operation benchmark");
    require(!o.svo.empty() && !o.output.empty(), "--svo and --output are required");
    require(o.profile == "instantaneous", "only instantaneous profile is implemented");
    require(o.minimum >= 0.3f && o.maximum <= 8.0f && o.minimum < o.maximum, "depth range must lie within 0.3..8 metres");
    require(o.mode == "NONE" || o.mode == "NEURAL_LIGHT" || o.mode == "NEURAL" || o.mode == "NEURAL_PLUS" || o.mode == "PERFORMANCE", "unsupported requested mode");
    return o;
}
sl::DEPTH_MODE depth_mode(const std::string& name) {
    if (name == "NONE") return sl::DEPTH_MODE::NONE;
    if (name == "NEURAL_LIGHT") return sl::DEPTH_MODE::NEURAL_LIGHT;
    if (name == "NEURAL") return sl::DEPTH_MODE::NEURAL;
    if (name == "NEURAL_PLUS") return sl::DEPTH_MODE::NEURAL_PLUS;
    if (name == "PERFORMANCE") return sl::DEPTH_MODE::PERFORMANCE;
    throw SafeError("unsupported requested mode");
}
Json init_settings(const sl::InitParameters& p) {
    return {{"depth_mode", sdk_string(sl::toString(p.depth_mode))},
        {"coordinate_units", sdk_string(sl::toString(p.coordinate_units))},
        {"coordinate_system", sdk_string(sl::toString(p.coordinate_system))},
        {"depth_stabilization", p.depth_stabilization}, {"svo_real_time_mode", p.svo_real_time_mode},
        {"camera_disable_self_calib", p.camera_disable_self_calib}, {"camera_image_flip", p.camera_image_flip},
        {"depth_minimum_distance", p.depth_minimum_distance}, {"depth_maximum_distance", p.depth_maximum_distance},
        {"depth_precision", sdk_string(sl::toString(p.depth_precision))}, {"allow_depth_cuda_graph", p.allow_depth_cuda_graph},
        {"sdk_gpu_id", p.sdk_gpu_id}};
}
Json runtime_settings(const sl::RuntimeParameters& p) {
    return {{"enable_depth", p.enable_depth}, {"enable_fill_mode", p.enable_fill_mode},
        {"confidence_threshold", p.confidence_threshold}, {"texture_confidence_threshold", p.texture_confidence_threshold},
        {"remove_saturated_areas", p.remove_saturated_areas},
        {"measure3D_reference_frame", sdk_string(sl::toString(p.measure3D_reference_frame))}};
}
void assert_init(sl::Camera& camera, const sl::InitParameters& requested) {
    const auto actual = camera.getInitParameters();
    require(!(actual.depth_mode != requested.depth_mode), "requested depth mode unavailable: SDK selected a different mode");
    require(!(actual.depth_stabilization != 0) && !camera.isPositionalTrackingEnabled(), "instantaneous temporal/tracking contract violated");
    require(!(actual.coordinate_units != sl::UNIT::METER) && actual.coordinate_system == sl::COORDINATE_SYSTEM::IMAGE,
        "effective coordinate contract violated");
    require(!actual.svo_real_time_mode && actual.camera_disable_self_calib == requested.camera_disable_self_calib
        && actual.camera_image_flip == requested.camera_image_flip, "effective playback/calibration contract violated");
    if (requested.depth_mode != sl::DEPTH_MODE::NONE) {
        require(actual.depth_minimum_distance == requested.depth_minimum_distance && actual.depth_maximum_distance == requested.depth_maximum_distance,
            "effective depth range differs from request");
    }
}
void assert_runtime(const sl::RuntimeParameters& actual, bool depth) {
    require(actual.enable_depth == depth && !actual.enable_fill_mode && actual.confidence_threshold == 95
        && actual.texture_confidence_threshold == 100 && !actual.remove_saturated_areas
        && actual.measure3D_reference_frame == sl::REFERENCE_FRAME::CAMERA, "effective runtime contract violated");
}
// Preserve unknown lens models and all coefficients. No normalization or guessed pose direction.
Json number(double value) {
    if (std::isnan(value)) return "NaN";
    if (std::isinf(value)) return value > 0 ? "+Inf" : "-Inf";
    return value;
}
Json intrinsics(const sl::CameraParameters& p) {
    Json coefficients = Json::array();
    for (int i = 0; i < 12; ++i) coefficients.push_back(number(p.disto[i]));
    return {{"fx", number(p.fx)}, {"fy", number(p.fy)}, {"cx", number(p.cx)}, {"cy", number(p.cy)},
        {"width", p.image_size.width}, {"height", p.image_size.height},
        {"h_fov_degrees", number(p.h_fov)}, {"v_fov_degrees", number(p.v_fov)}, {"d_fov_degrees", number(p.d_fov)},
        {"focal_length_metric_mm", number(p.focal_length_metric)},
        {"lens_model", sdk_string(sl::toString(p.lens_distortion_model))},
        {"lens_model_sdk_enum", static_cast<int>(p.lens_distortion_model)},
        {"distortion_order", "k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4"}, {"distortion", coefficients}};
}
Json calibration(const sl::CalibrationParameters& p) {
    Json matrix = Json::array();
    for (int row = 0; row < 4; ++row) {
        Json values = Json::array();
        for (int col = 0; col < 4; ++col) values.push_back(number(p.stereo_transform(row, col)));
        matrix.push_back(values);
    }
    return {{"left", intrinsics(p.left_cam)}, {"right", intrinsics(p.right_cam)},
        {"sdk_stereo_transform", matrix}, {"sdk_camera_baseline", number(p.getCameraBaseline())},
        {"transform_normalization", unavailable("raw pose direction and rectification rotations require proof")},
        {"pixel_center_convention", unavailable("not established against SDK pixels")}};
}
Json calibration_snapshot(const sl::CameraInformation& info) {
    return {{"raw", calibration(info.camera_configuration.calibration_parameters_raw)},
        {"rectified", calibration(info.camera_configuration.calibration_parameters)},
        {"units", "metres"}, {"coordinate_system", "IMAGE"}};
}
void check_mat(const sl::Mat& mat, sl::MAT_TYPE expected, const sl::Resolution& resolution, size_t row_bytes) {
    require(mat.isInit() && !(mat.getDataType() != expected), "unexpected SDK matrix type or allocation");
    require(resolution.width > 0 && resolution.height > 0, "invalid SDK native resolution");
    require(mat.getWidth() == static_cast<size_t>(resolution.width)
        && mat.getHeight() == static_cast<size_t>(resolution.height), "unexpected SDK matrix dimensions");
    const size_t stride = mat.getStepBytes(sl::MEM::CPU);
    require(stride >= row_bytes, "SDK row stride shorter than logical row");
    checked_product(stride, resolution.height);
    require(mat.getPtr<sl::uchar1>(sl::MEM::CPU) != nullptr, "SDK CPU buffer unavailable");
}
Json write_plane(const fs::path& root, const std::string& name, const sl::Mat& mat,
                 sl::MAT_TYPE expected, const sl::Resolution& resolution, size_t bytes_per_pixel, const char* format) {
    const size_t row_bytes = checked_product(resolution.width, bytes_per_pixel);
    checked_product(row_bytes, resolution.height);
    check_mat(mat, expected, resolution, row_bytes);
    const size_t stride = mat.getStepBytes(sl::MEM::CPU);
    const auto* base = reinterpret_cast<const std::uint8_t*>(mat.getPtr<sl::uchar1>(sl::MEM::CPU));
    Writer writer(root, name);
    write_rows(writer, base, checked_product(stride, resolution.height), row_bytes, resolution.height, stride);
    Json result = writer.finish();
    result.update({{"format", format}, {"width", resolution.width}, {"height", resolution.height},
        {"row_stride_bytes", row_bytes}, {"sdk_row_stride_bytes", stride}});
    return result;
}

// Serial is used only to locate the actual SDK settings file; no private path/serial is published.
Json settings_snapshot(sl::Camera& camera, unsigned int serial, const fs::path& root, int ordinal) {
    const auto init = camera.getInitParameters();
    const std::string configured = sdk_string(init.optional_settings_path);
    const fs::path directory = configured.empty() ? fs::path("/usr/local/zed/settings") : fs::path(configured);
    if (serial == 0) return unavailable("SDK did not expose a usable settings-file identity");
    const fs::path resolved = directory / ("SN" + std::to_string(serial) + ".conf");
    if (!fs::is_regular_file(resolved)) return unavailable("resolved settings file absent in effective SDK settings directory");
    Json file = copy_and_hash(resolved, root, "settings-" + std::to_string(ordinal) + ".conf");
    file["available"] = true;
    file["directory_selection"] = configured.empty() ? "documented_linux_default" : "effective_optional_settings_path";
    return file;
}
struct CameraGuard { sl::Camera camera; ~CameraGuard() { camera.close(); } };
void sdk_success(sl::ERROR_CODE code, Json& diagnostic, const char* stage) {
    if (code == sl::ERROR_CODE::SUCCESS) return;
    diagnostic["sdk_failure"] = {{"stage", stage}, {"error_code", static_cast<int>(code)}, {"error", sdk_string(sl::toString(code))}};
    throw SafeError("SDK operation failed; requested result unavailable (see diagnostic)");
}
Json build_identity() {
    Json os = unavailable("OS identity unavailable");
#ifdef __linux__
    struct utsname identity{};
    if (uname(&identity) == 0) os = {{"system", identity.sysname}, {"release", identity.release}, {"machine", identity.machine}};
#endif
    return {{"source_sha256", SF_ZED_EXPORT_SOURCE_SHA256}, {"benchmark_sha256", SF_ZED_EXPORT_BENCHMARK_SHA256}, {"io_sha256", SF_ZED_EXPORT_IO_SHA256}, {"commit", SF_ZED_EXPORT_COMMIT},
        {"dirty", SF_ZED_EXPORT_DIRTY}, {"compiler", __VERSION__}, {"os", os},
        {"gpu_device_identity", unavailable("not queried by this exporter; effective SDK device ordinal recorded separately")},
        {"driver_identity", unavailable("not exposed by this capture contract")}};
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") { help(); return 0; }
    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::cout << "export_sdk diagnostic-v1; SDK " << sl::Camera::getSDKVersion() << '\n'; return 0;
    }
    fs::path root;
    Json diagnostic;
    try {
        const Options o = parse(argc, argv);
        require(fs::is_regular_file(o.svo), "SVO input must be a regular file");
        root = fresh_root(o.output);
        diagnostic = {{"schema", "stereoforge.zed_comparison.sdk_diagnostic.v1"}, {"qualification", "unqualified"},
            {"canonical_export", false}, {"operation", o.operation}, {"status", "in_progress"},
            {"sdk_version", sdk_string(sl::Camera::getSDKVersion())}, {"build", build_identity()},
            {"requested_mode", o.mode}, {"requested_result", unavailable("capture has not completed")}, {"profile", o.profile}, {"frames_requested", o.frames}, {"frames", Json::array()},
            {"source_codec", unavailable("no external codec attestation provided")},
            {"source_lossless", unavailable("no external lossless attestation provided")},
            {"model_identity", unavailable("SDK does not expose a model file identity through this API")},
            {"rgb_geometric_support", unavailable("SDK rectification sampling support is not established")},
            {"raw_pose_normalization", unavailable("SDK raw stereo pose convention requires independent proof")},
            {"timestamp_mapping", unavailable("no sensor-to-host mapping established")},
            {"timing_contract", "opaque grab API wall time; CPU retrieval and serialization/I/O separate; no GPU kernel timing or FPS claim"}};
        if (o.operation == "benchmark") {
            diagnostic.update(sf::zed_benchmark::report(sf::zed_benchmark::Samples<sl::ERROR_CODE>{}, o.warmup, o.frames, o.mode != "NONE"));
            diagnostic["cpu_buffer_prepare_ms"] = unavailable("no diagnostic CPU buffers requested");
            diagnostic["cold_setup_scopes"] = {{"cold_open_api_ms", "camera.open API only"},
                {"source_copy_hash_io_ms", "preserved SVO copy and hash IO"},
                {"cpu_buffer_prepare_ms", "not performed: no diagnostic CPU buffers requested"}};
        }
        write_metadata(root, diagnostic);
        std::string extension = fs::path(o.svo).extension().string();
        for (char& ch : extension) if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
        diagnostic["source_container_hint"] = extension == ".svo2" ? "SVO2_extension" : extension == ".svo" ? "legacy_SVO_extension" : "unavailable";
        const auto source_start = Clock::now();
        diagnostic["source"] = copy_and_hash(o.svo, root, extension == ".svo2" ? "source.svo2" : "source.svo");
        diagnostic["source_copy_hash_io_ms"] = elapsed_ms(source_start);
        int major = 0, minor = 0, patch = 0; sl::Camera::getSDKVersion(major, minor, patch);
        require(major == 5 && minor == 5 && patch == 0, "exporter requires pinned SDK 5.5.0");

        CameraGuard guard; auto& camera = guard.camera;
        sl::InitParameters init;
        // Replay the exact preserved bytes just hashed, rather than reopening a mutable original.
        const std::string preserved_source = (root / diagnostic["source"]["path"].get<std::string>()).string();
        init.input.setFromSVOFile(preserved_source.c_str()); init.svo_real_time_mode = false; init.sdk_verbose = 0;
        init.coordinate_units = sl::UNIT::METER; init.coordinate_system = sl::COORDINATE_SYSTEM::IMAGE;
        init.depth_mode = depth_mode(o.mode); init.depth_stabilization = 0;
        init.camera_disable_self_calib = !o.self_calibration; init.camera_image_flip = sl::FLIP_MODE::OFF;
        init.depth_minimum_distance = o.minimum; init.depth_maximum_distance = o.maximum;
        init.depth_precision = sl::DEPTH_PRECISION::FP16; init.allow_depth_cuda_graph = false;
        std::string settings_path;
        if (!o.settings_dir.empty()) {
            settings_path = fs::path(o.settings_dir).lexically_normal().generic_string();
            if (settings_path.back() != '/') settings_path += '/';
            init.optional_settings_path = settings_path.c_str();
        }
        sl::RuntimeParameters runtime;
        runtime.enable_depth = o.mode != "NONE"; runtime.enable_fill_mode = false;
        runtime.confidence_threshold = 95; runtime.texture_confidence_threshold = 100;
        runtime.remove_saturated_areas = false; runtime.measure3D_reference_frame = sl::REFERENCE_FRAME::CAMERA;
        diagnostic["requested_settings"] = {{"init", init_settings(init)}, {"runtime", runtime_settings(runtime)},
            {"settings_directory", o.settings_dir.empty() ? "SDK_default" : "private_explicit_directory"}};
        write_metadata(root, diagnostic);
        const auto open_start = Clock::now();
        const auto open_result = camera.open(init);
        diagnostic["cold_open_api_ms"] = elapsed_ms(open_start);
        sdk_success(open_result, diagnostic, "open");
        diagnostic["effective_settings_after_open"] = {{"init", init_settings(camera.getInitParameters())},
            {"positional_tracking_enabled", camera.isPositionalTrackingEnabled()},
            {"runtime", unavailable("no successful grab yet")}};
        assert_init(camera, init);
        const auto initial_info = camera.getCameraInformation();
        const auto resolution = initial_info.camera_configuration.resolution;
        require(resolution.width > 0 && resolution.height > 0, "invalid SDK native resolution");
        const size_t width = static_cast<size_t>(resolution.width);
        const size_t height = static_cast<size_t>(resolution.height);
        const size_t pixels = checked_product(resolution.width, resolution.height);
        checked_product(pixels, size_t(4));
        diagnostic["camera"] = {{"model", sdk_string(sl::toString(initial_info.camera_model))},
            {"width", resolution.width}, {"height", resolution.height}, {"nominal_fps", number(initial_info.camera_configuration.fps)},
            {"serial_identity", unavailable("private serial intentionally omitted")}};
        diagnostic["initial_calibration"] = calibration_snapshot(initial_info);
        if (o.operation == "benchmark") {
            diagnostic["svo_cursor_before"] = camera.getSVOPosition();
            diagnostic["settings_file_before"] = settings_snapshot(camera, initial_info.serial_number, root, 0);
            const auto samples = sf::zed_benchmark::sample<sl::ERROR_CODE>(o.warmup, o.frames,
                [&] { return camera.grab(runtime); }, sl::ERROR_CODE::SUCCESS);
            diagnostic.update(sf::zed_benchmark::report(samples, o.warmup, o.frames, runtime.enable_depth));
            diagnostic["svo_cursor_after"] = camera.getSVOPosition();
            // Preserve partial samples before post-loop assertions or error serialization.
            if (samples.failure) sdk_success(*samples.failure, diagnostic, "benchmark_grab");
            assert_init(camera, init);
            const auto effective_runtime = camera.getRuntimeParameters();
            assert_runtime(effective_runtime, runtime.enable_depth);
            diagnostic["effective_settings_after_benchmark"] = {{"init", init_settings(camera.getInitParameters())},
                {"runtime", runtime_settings(effective_runtime)}, {"positional_tracking_enabled", camera.isPositionalTrackingEnabled()}};
            diagnostic["final_calibration"] = calibration_snapshot(camera.getCameraInformation());
            diagnostic["settings_file_after"] = settings_snapshot(camera, initial_info.serial_number, root, 1);
            diagnostic["status"] = "benchmarked_unqualified";
            diagnostic["requested_result"] = {{"available", true}, {"qualification", "unqualified"}};
            write_metadata(root, diagnostic);
            std::cout << "Unqualified opaque API timing written; GPU-wide completion remains unproven.\n";
            return 0;
        }
        const auto prepare_start = Clock::now();
        std::array<sl::Mat, 4> images;
        for (auto& image : images) image.alloc(resolution, sl::MAT_TYPE::U8_C4, sl::MEM::CPU);
        sl::Mat depth, confidence;
        if (runtime.enable_depth) {
            depth.alloc(resolution, sl::MAT_TYPE::F32_C1, sl::MEM::CPU);
            confidence.alloc(resolution, sl::MAT_TYPE::F32_C1, sl::MEM::CPU);
        }
        std::vector<std::uint8_t> finite_positive(runtime.enable_depth ? pixels : 0);
        diagnostic["cpu_buffer_prepare_ms"] = elapsed_ms(prepare_start);
        const std::array<sl::VIEW, 4> views{sl::VIEW::LEFT_UNRECTIFIED, sl::VIEW::RIGHT_UNRECTIFIED, sl::VIEW::LEFT, sl::VIEW::RIGHT};
        const std::array<const char*, 4> labels{"raw-left", "raw-right", "rectified-left", "rectified-right"};
        std::string previous_calibration;
        for (int ordinal = 0; ordinal < o.frames; ++ordinal) {
            Json frame = {{"ordinal", ordinal}, {"status", "in_progress"}, {"files", Json::object()}};
            diagnostic["frames"].push_back(frame);
            auto& saved = diagnostic["frames"].back();
            const auto grab_start = Clock::now();
            const auto grab_result = camera.grab(runtime);
            saved["opaque_grab_api_ms"] = elapsed_ms(grab_start);
            sdk_success(grab_result, diagnostic, "grab");
            const auto effective_init = camera.getInitParameters();
            const auto effective_runtime = camera.getRuntimeParameters();
            saved["effective_settings"] = {{"init", init_settings(effective_init)}, {"runtime", runtime_settings(effective_runtime)},
                {"positional_tracking_enabled", camera.isPositionalTrackingEnabled()}};
            assert_init(camera, init); assert_runtime(effective_runtime, runtime.enable_depth);
            saved["svo_cursor"] = camera.getSVOPosition();
            const auto image_ns = camera.getTimestamp(sl::TIME_REFERENCE::IMAGE).getNanoseconds();
            saved["image_timestamp"] = {{"value", image_ns}, {"unit", "ns"}, {"clock_domain", "SDK_IMAGE_UNIX_ns_from_SVO"},
                {"semantics", "latest grabbed IMAGE timestamp; no host ingress mapping"}};
            saved["ingress_timestamp"] = unavailable("actual mapped ingress clock unavailable");
            saved["source_observation_id"] = unavailable("SDK does not expose observation ancestry IDs");
            require(camera.getSVOPosition() >= 0 && image_ns != 0, "SDK frame cursor or IMAGE timestamp unavailable");
            const auto info = camera.getCameraInformation();
            require(info.camera_configuration.resolution.width == resolution.width
                && info.camera_configuration.resolution.height == resolution.height, "native SDK resolution changed during capture");
            saved["calibration"] = calibration_snapshot(info);
            const auto calib_text = saved["calibration"].dump();
            const auto calib_sha = hash_text(calib_text);
            saved["calibration_sha256"] = calib_sha;
            saved["calibration_stratum"] = calib_sha;
            saved["calibration_changed_from_previous"] = !previous_calibration.empty() && previous_calibration != calib_sha;
            previous_calibration = calib_sha;
            const auto retrieval_start = Clock::now();
            for (size_t i = 0; i < images.size(); ++i)
                sdk_success(camera.retrieveImage(images[i], views[i], sl::MEM::CPU, resolution), diagnostic, "retrieveImage");
            if (runtime.enable_depth) {
                sdk_success(camera.retrieveMeasure(depth, sl::MEASURE::DEPTH, sl::MEM::CPU, resolution), diagnostic, "retrieveDepth");
                sdk_success(camera.retrieveMeasure(confidence, sl::MEASURE::CONFIDENCE, sl::MEM::CPU, resolution), diagnostic, "retrieveConfidence");
            }
            saved["cpu_retrieval_api_ms"] = elapsed_ms(retrieval_start);
            const auto io_start = Clock::now();
            const std::string prefix = "frame-" + std::to_string(ordinal) + "-";
            for (size_t i = 0; i < images.size(); ++i) {
                auto plane = write_plane(root, prefix + labels[i] + ".bgra8", images[i], sl::MAT_TYPE::U8_C4, resolution, 4, "BGRA8");
                plane["channel_order"] = "BGRA"; plane["alpha_is_validity"] = false;
                plane["pixel_origin"] = "SDK decoded post-ISP; not transport Bayer bytes";
                saved["files"][labels[i]] = plane;
            }
            saved["settings_file"] = settings_snapshot(camera, info.serial_number, root, ordinal);
            saved["capture_stratum"] = hash_text(Json{{"calibration_sha256", calib_sha},
                {"settings_sha256", saved["settings_file"].value("sha256", std::string("unavailable"))}}.dump());
            if (runtime.enable_depth) {
                saved["files"]["depth"] = write_plane(root, prefix + "depth.f32le", depth, sl::MAT_TYPE::F32_C1, resolution, 4, "F32LE");
                saved["files"]["depth"]["unit"] = "metres";
                saved["files"]["depth"]["invalid_encoding"] = "SDK NaN/Inf bits retained exactly";
                saved["files"]["depth"]["axis_convention"] = unavailable("axial-Z versus range needs independent off-axis validation");
                saved["files"]["confidence"] = write_plane(root, prefix + "confidence.f32le", confidence, sl::MAT_TYPE::F32_C1, resolution, 4, "F32LE");
                saved["confidence_direction"] = "low_is_good";
                const auto* base = reinterpret_cast<const std::uint8_t*>(depth.getPtr<sl::uchar1>(sl::MEM::CPU));
                const size_t stride = depth.getStepBytes(sl::MEM::CPU);
                for (size_t y = 0; y < height; ++y) {
                    for (size_t x = 0; x < width; ++x) {
                        float z; std::memcpy(&z, base + y * stride + x * sizeof(float), sizeof(float));
                        finite_positive[y * width + x] = std::isfinite(z) && z > 0 ? 1 : 0;
                    }
                }
                Writer mask(root, prefix + "sdk-finite-positive.u8");
                mask.write(finite_positive.data(), finite_positive.size());
                saved["files"]["sdk_finite_positive"] = mask.finish();
                saved["files"]["sdk_finite_positive"].update({{"format", "U8"}, {"width", resolution.width}, {"height", resolution.height},
                    {"row_stride_bytes", resolution.width}, {"semantics", "finite positive SDK depth only; not geometric RGB support or observation proof"}});
                saved["observation"] = unavailable("SDK does not expose per-pixel observation provenance");
            } else saved["observation"] = unavailable("depth disabled (NONE)");
            saved["texture_map"] = unavailable("texture threshold is a setting, not an exposed map");
            saved["depth_age"] = unavailable("not exposed by SDK API");
            saved["ancestry"] = unavailable("not exposed by SDK API");
            saved["rgb_geometric_support"] = unavailable("no proven rectification footprint support");
            saved["serialization_io_ms"] = elapsed_ms(io_start);
            saved["status"] = "captured_unqualified";
            const auto metadata_start = Clock::now();
            write_metadata(root, diagnostic);
            saved["metadata_write_ms"] = elapsed_ms(metadata_start);
        }
        diagnostic["status"] = "captured_unqualified";
        diagnostic["requested_result"] = {{"available", true}, {"qualification", "unqualified"}};
        write_metadata(root, diagnostic);
        std::cout << "Unqualified diagnostic capture written; canonical geometry gates remain open.\n";
        return 0;
    } catch (const std::exception& error) {
        const auto* safe = dynamic_cast<const SafeError*>(&error);
        const std::string reason = safe ? safe->what() : "invalid request or filesystem/hash failure";
        // SDK errors are recorded by code, and private filenames/serials are never echoed.
        if (!root.empty()) {
            diagnostic["status"] = "failed";
            diagnostic["failure"] = {{"reason", reason}, {"partial_payloads", "unqualified; must not be indexed"}};
            diagnostic["requested_result"] = {{"available", false}, {"reason", reason}};
            try { write_metadata(root, diagnostic); } catch (...) { std::cerr << "Failed to write diagnostic failure metadata.\n"; }
        }
        std::cerr << "export_sdk failed: " << reason << ".\n";
        if (root.empty()) std::cerr << "Use --help; canonical export requires proven raw pose and rectification support.\n";
        return 1;
    }
}
