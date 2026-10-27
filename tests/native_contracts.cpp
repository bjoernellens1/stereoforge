#include <stereoforge/contracts.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
struct Completion final : stereoforge::CompletionState {
    bool completed = false;
    explicit Completion(std::uint64_t identity) : CompletionState(identity) {}
    bool ready() const override { return completed; }
    void wait() override { completed = true; }
};
void require(bool value) { if (!value) throw std::runtime_error("native contract failed"); }
}
int main() {
    using namespace stereoforge;
    require(std::abs(metric_depth(20.0f, 500.0f, 0.12f, 2.0f) - 500.0f*0.12f/18.0f) < 1e-6f);
    require(std::isnan(metric_depth(2.0f, 500.0f, 0.12f, 2.0f)));
    require(std::isnan(metric_depth(-1.0f, 500.0f, 0.12f)));
    require(std::isnan(metric_depth(10.0f, 500.0f, -0.12f)));
    MetricCalibration{80,.12f,0}.validate();
    bool invalid_calibration=false;
    try {MetricCalibration{80,-.12f,0}.validate();}catch(const std::invalid_argument&){invalid_calibration=true;}
    require(invalid_calibration);
    auto state = std::make_shared<Completion>(7);
    std::weak_ptr<Completion> lifetime = state;
    CompletionToken token(state);
    auto storage = std::make_shared<int>(42);
    BufferView view{storage.get(), 0, 64, 4, 7, storage, token};
    view.validate(7);
    ImageHandle image{view, 4, 4, 16, ImageFormat::Gray32Uint};
    image.validate(7);
    image.row_stride_bytes = 12;
    bool bad_stride = false;
    try { image.validate(7); } catch (const std::invalid_argument&) { bad_stride = true; }
    require(bad_stride);
    image = {};
    bool rejected = false;
    try { view.validate(8); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
    state.reset(); token = {};
    require(!lifetime.expired());
    require(!view.completion.ready());
    view.completion.wait();
    require(view.completion.ready());
    view = {};
    require(lifetime.expired());
    StereoConfig config{64, 32, 16, 4, 8, 32, false};
    config.validate();
    config.disparities = 257;
    rejected = false;
    try { config.validate(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
    std::cout << "native units, configuration, identity, completion, and ownership contracts passed\n";
}
