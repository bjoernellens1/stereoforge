#include <stereoforge/runtime.hpp>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <unordered_map>

namespace {
template<class T> void write(const std::string& path, const std::vector<T>& data) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()*sizeof(T)));
    if (!stream) throw std::runtime_error("cannot write " + path);
}
std::vector<std::uint32_t> read(const std::string& path, std::size_t count) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() != std::streamoff(count*4)) throw std::runtime_error("expected exactly WxH uint32 pixels in " + path);
    stream.seekg(0);
    std::vector<std::uint32_t> pixels(count);
    stream.read(reinterpret_cast<char*>(pixels.data()), std::streamsize(count*4));
    if (!stream) throw std::runtime_error("cannot read " + path);
    return pixels;
}
}
int main(int argc, char** argv) {
    try {
        std::unordered_map<std::string, std::string> options;
        bool self_test = false, validation = false;
        for (int i=1; i<argc; ++i) {
            const std::string key(argv[i]);
            if (key == "--self-test") { self_test = true; continue; }
            if (key == "--validation") { validation = true; continue; }
            if (key == "--help") {
                std::cout << "native_stereo [--width W --height H --disparities D --paths 4|8 --p1 8 --p2 32 --subpixel 0|1]\n"
                             "              [--left left.u32 --right right.u32 --output PREFIX] [--iterations N] [--shader-directory PATH] [--fx PX --baseline-m M --principal-offset PX] [--self-test] [--validation]\n"
                             "Writes PREFIX.disparity.f32, PREFIX.confidence.f32, PREFIX.validity.u32; no inputs generates shifted texture.\n";
                return 0;
            }
            if (key != "--width" && key != "--height" && key != "--disparities" && key != "--paths" &&
                key != "--p1" && key != "--p2" && key != "--subpixel" && key != "--left" && key != "--right" &&
                key != "--output" && key != "--iterations" && key != "--shader-directory" &&
                key != "--fx" && key != "--baseline-m" && key != "--principal-offset") throw std::invalid_argument("unknown option " + key);
            if (++i == argc) throw std::invalid_argument("missing value for " + key);
            options[key] = argv[i];
        }
        auto integer = [&](const char* name, std::uint32_t fallback) {
            if (!options.contains(name)) return fallback;
            const auto& text = options.at(name);
            std::size_t end = 0;
            auto result = std::stoul(text, &end);
            if (end != text.size() || result > UINT32_MAX || text.front() == '-') throw std::invalid_argument(name);
            return std::uint32_t(result);
        };
        stereoforge::StereoConfig config{integer("--width", 64), integer("--height", 32), integer("--disparities", 16),
            integer("--paths", 4), integer("--p1", 8), integer("--p2", 32), integer("--subpixel", 1) != 0};
        config.validate();
        const auto count = std::size_t(config.width)*config.height;
        std::vector<std::uint32_t> left(count), right(count);
        if (options.contains("--left") != options.contains("--right")) throw std::invalid_argument("left and right inputs required together");
        if (options.contains("--left")) {
            left = read(options.at("--left"), count); right = read(options.at("--right"), count);
        } else {
            std::uint32_t state = 1234567;
            for (auto& pixel : left) { state ^= state<<13; state ^= state>>17; state ^= state<<5; pixel = state & 255; }
            for (std::uint32_t y=0; y<config.height; ++y) for (std::uint32_t x=0; x<config.width; ++x)
                right[y*config.width+x] = x+4<config.width ? left[y*config.width+x+4] : 0;
        }
        std::optional<stereoforge::MetricCalibration> metric;
        if(options.contains("--fx")!=options.contains("--baseline-m")||
            (options.contains("--principal-offset")&&!options.contains("--fx")))
            throw std::invalid_argument("metric depth requires --fx and --baseline-m together");
        auto number=[&](const char* name){std::size_t end=0;const auto& text=options.at(name);const auto value=std::stof(text,&end);if(end!=text.size())throw std::invalid_argument(name);return value;};
        if(options.contains("--fx")) {
            metric=stereoforge::MetricCalibration{number("--fx"),number("--baseline-m"),options.contains("--principal-offset")?number("--principal-offset"):0.f};
            metric->validate();
        }
        const auto start = std::chrono::steady_clock::now();
        stereoforge::NativeStereo engine(config, options.contains("--shader-directory") ? options.at("--shader-directory") : STEREOFORGE_SHADER_DIRECTORY, validation, metric);
        const auto initialized = std::chrono::steady_clock::now();
        const auto iterations = integer("--iterations", 1);
        if (!iterations) throw std::invalid_argument("iterations must be positive");
        stereoforge::StereoReadback output;
        for (std::uint32_t i=0; i<iterations; ++i) {
            auto frame = engine.enqueue(left, right);
            if(bool(frame.depth.native)!=metric.has_value()||bool(frame.depth_validity.native)!=metric.has_value())throw std::runtime_error("metric depth capability mismatch");
            if (self_test && i == 0) {
                auto second = engine.enqueue(left, right);
                bool rejected = false;
                try { auto third = engine.enqueue(left, right); } catch (const std::runtime_error&) { rejected = true; }
                if (!rejected) throw std::runtime_error("bounded queue accepted a third leased frame");
                auto second_output = engine.readback(second);
                if (second_output.validity.size() != count) throw std::runtime_error("second slot readback failed");
            }
            output = engine.readback(frame);
        }
        const auto done = std::chrono::steady_clock::now();
        const auto valid = std::accumulate(output.validity.begin(), output.validity.end(), std::uint64_t(0));
        if (self_test) {
            std::size_t checked = 0, correct = 0;
            for (std::uint32_t y=4; y+4<config.height; ++y) for (std::uint32_t x=10; x+4<config.width; ++x) {
                const auto i = y*config.width+x;
                if (output.validity[i]) { ++checked; correct += std::abs(output.disparity[i]-4.0f) < 0.6f; }
            }
            if (!checked || float(correct)/checked < 0.98f) throw std::runtime_error("known-shift disparity failed");
            // Views and completion tokens retain their engine after its public
            // facade is destroyed; foreign engine readback must reject them.
            stereoforge::NativeFrame retained;
            {
                stereoforge::NativeStereo temporary(config, options.contains("--shader-directory") ? options.at("--shader-directory") : STEREOFORGE_SHADER_DIRECTORY, validation, metric);
                retained = temporary.enqueue(left, right);
                bool foreign_rejected = false;
                try { auto foreign = engine.readback(retained); } catch (const std::invalid_argument&) { foreign_rejected = true; }
                if (!foreign_rejected) throw std::runtime_error("foreign device readback accepted");
            }
            retained.completion.wait();
            if (!retained.completion.ready()) throw std::runtime_error("retained completion failed");
            auto calibrated_config=config;calibrated_config.subpixel=false;
            stereoforge::NativeStereo calibrated(calibrated_config,options.contains("--shader-directory")?options.at("--shader-directory"):STEREOFORGE_SHADER_DIRECTORY,validation,stereoforge::MetricCalibration{80,.12f,0});
            auto calibrated_frame=calibrated.enqueue(left,right);
            auto calibrated_output=calibrated.readback(calibrated_frame);
            std::size_t depth_checked=0;
            for(std::size_t i=0;i<count;++i) {
                if(calibrated_output.validity[i]) {
                    if(calibrated_output.depth_validity[i]!=1)throw std::runtime_error("valid metric depth lacked depth validity");
                    if(std::abs(calibrated_output.depth[i]-9.6f/calibrated_output.disparity[i])>1e-5f)throw std::runtime_error("GPU metric depth mismatch");
                    if(calibrated_output.disparity[i]==4.f){++depth_checked;if(std::abs(calibrated_output.depth[i]-2.4f)>1e-5f)throw std::runtime_error("known-shift metric depth failed");}
                }else if(calibrated_output.depth_validity[i]!=0||!std::isnan(calibrated_output.depth[i]))throw std::runtime_error("invalid disparity emitted metric depth");
            }
            if(!depth_checked)throw std::runtime_error("no metric-depth observations tested");
            stereoforge::NativeStereo offset_engine(calibrated_config,options.contains("--shader-directory")?options.at("--shader-directory"):STEREOFORGE_SHADER_DIRECTORY,validation,stereoforge::MetricCalibration{80,.12f,4});
            auto offset_frame=offset_engine.enqueue(left,right);
            auto offset_output=offset_engine.readback(offset_frame);
            std::size_t rejected_metric=0;
            for(std::size_t i=0;i<count;++i)if(offset_output.validity[i]&&offset_output.disparity[i]==4.f){
                ++rejected_metric;
                if(offset_output.depth_validity[i]!=0||!std::isnan(offset_output.depth[i]))throw std::runtime_error("invalid corrected disparity remained metric-valid");
            }
            if(!rejected_metric)throw std::runtime_error("no invalid corrected disparities tested");
        }
        if (options.contains("--output")) {
            const auto prefix = options.at("--output");
            write(prefix+".disparity.f32", output.disparity);
            write(prefix+".confidence.f32", output.confidence);
            write(prefix+".validity.u32", output.validity);
            if(!output.depth.empty()){write(prefix+".depth.f32",output.depth);write(prefix+".depth_validity.u32",output.depth_validity);}
        }
        const auto ms = [](auto interval) { return std::chrono::duration<double, std::milli>(interval).count(); };
        std::cout << "{\"backend\":\"vulkan\",\"adapter\":" << std::quoted(engine.device_name()) << ",\"vulkan_validation\":" << (validation ? "true" : "false") << ",\"width\":" << config.width << ",\"height\":" << config.height
                  << ",\"disparities\":" << config.disparities << ",\"paths\":" << config.paths
                  << ",\"metric_depth\":" << (metric?"true":"false") << ",\"valid_pixels\":" << valid << ",\"iterations\":" << iterations
                  << ",\"initialization_ms\":" << ms(initialized-start)
                  << ",\"transfer_inclusive_total_ms\":" << ms(done-initialized) << "}\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "native_stereo: " << error.what() << '\n'; return 1; }
}
