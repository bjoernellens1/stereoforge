// Native ZED Open Capture binding. SDK buffers are snapshotted while holding
// their producer mutex by the narrowly generated adapters in Capture.cmake.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "videocapture.hpp"
#include "sensorcapture.hpp"
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <stdexcept>

namespace py = pybind11;
struct FrameTimeout : std::runtime_error { using std::runtime_error::runtime_error; };

class ZedCapture {
    std::unique_ptr<sl_oc::video::VideoCapture> video_;
    std::unique_ptr<sl_oc::sensors::SensorCapture> sensors_;
    std::mutex io_mutex_, sample_mutex_;
    std::thread sensor_thread_;
    std::atomic<bool> stop_{false};
    std::deque<sl_oc::sensors::data::Imu> samples_;
    std::atomic<uint64_t> dropped_{0};
    uint64_t last_timestamp_ = 0;
    int serial_ = 0, width_, height_;
    bool sensors_available_ = false;
    std::string sensor_status_ = "unavailable";
    static constexpr size_t max_samples_ = 8192;

    void sensor_loop() {
        uint64_t previous = 0;
        while (!stop_.load()) {
            auto sample = sensors_->snapshotIMU(1500);
            if (sample.valid == sl_oc::sensors::data::Imu::NEW_VAL && sample.timestamp > previous) {
                previous = sample.timestamp;
                std::lock_guard<std::mutex> lock(sample_mutex_);
                if (samples_.size() == max_samples_) { samples_.pop_front(); ++dropped_; }
                samples_.push_back(sample);
            }
        }
    }
public:
    ZedCapture(int width=1280,int height=720,int fps=60,int device_id=-1):width_(width),height_(height) {
        sl_oc::video::VideoParams params;
        using R = sl_oc::video::RESOLUTION;
        if (width==1280 && height==720) params.res=R::HD720;
        else if (width==1920 && height==1080) params.res=R::HD1080;
        else if (width==2208 && height==1242) params.res=R::HD2K;
        else if (width==672 && height==376) params.res=R::VGA;
        else throw std::invalid_argument("unsupported ZED resolution");
        if ((fps!=15 && fps!=30 && fps!=60 && fps!=100) ||
            (width==2208 && fps!=15) || (width==1920 && fps>30) || (width==1280 && fps>60))
            throw std::invalid_argument("unsupported ZED resolution/framerate combination");
        if (device_id < -1) throw std::invalid_argument("invalid video device id");
        params.fps=static_cast<sl_oc::video::FPS>(fps);
        video_=std::make_unique<sl_oc::video::VideoCapture>(params);
        if (!video_->initializeVideo(device_id)) throw std::runtime_error("cannot initialize ZED Open Capture video; check USB mode, permissions and device ownership");
        int actual_width=0,actual_height=0;
        video_->getFrameSize(actual_width,actual_height);
        if (actual_width!=2*width || actual_height!=height)
            throw std::runtime_error("ZED negotiated an unexpected image size");
        serial_=video_->getSerialNumber();
        if (serial_<=0) throw std::runtime_error("cannot obtain camera serial; refusing unidentified calibration");
        sensors_=std::make_unique<sl_oc::sensors::SensorCapture>();
        if (sensors_->initializeSensors(serial_)) {
            sensors_available_=true;
            sensor_status_=video_->enableSensorSync(sensors_.get()) ? "hardware_sync_enabled" : "sync_unavailable_raw_only";
            sensor_thread_=std::thread(&ZedCapture::sensor_loop,this);
        } else {
            sensor_status_="sensor_open_failed_check_hidraw_permissions";
        }
    }
    ~ZedCapture() { close(); }
    void close() {
        std::lock_guard<std::mutex> io_lock(io_mutex_);
        stop_.store(true);
        if (sensor_thread_.joinable()) sensor_thread_.join();
        // Video may dereference its sensor sync pointer while stopping.
        video_.reset();
        sensors_.reset();
    }
    int serial() const { return serial_; }
    bool sensors_available() const { return sensors_available_; }
    std::string sensor_status() const { return sensor_status_; }
    uint64_t sensor_dropped_count() const { return dropped_.load(); }
    py::dict read(int timeout_ms=1000) {
        if (timeout_ms<=0) throw std::invalid_argument("timeout_ms must be positive");
        std::vector<uint8_t> bytes;
        sl_oc::video::Frame frame;
        std::deque<sl_oc::sensors::data::Imu> samples;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> io_lock(io_mutex_);
            if (!video_) throw std::runtime_error("camera is closed");
            frame=video_->snapshotFrame(bytes,static_cast<uint64_t>(timeout_ms));
            if (!frame.timestamp || frame.timestamp==last_timestamp_)
                throw FrameTimeout("no new ZED frame within timeout (last frame repeated or camera disconnected)");
            if (frame.width!=2*width_ || frame.height!=height_ || frame.channels!=2)
                throw std::runtime_error("unexpected ZED YUYV frame format");
            last_timestamp_=frame.timestamp;
            std::lock_guard<std::mutex> sample_lock(sample_mutex_);
            // Include all collected samples up to this observation. Samples newer
            // than the frame stay queued for the next frame rather than misassociate.
            while (!samples_.empty() && samples_.front().timestamp<=frame.timestamp) {
                samples.push_back(samples_.front()); samples_.pop_front();
            }
        }
        py::array_t<uint8_t> yuyv({height_,2*width_,2});
        std::memcpy(yuyv.mutable_data(),bytes.data(),bytes.size());
        py::list sensor_samples;
        for (const auto& sample:samples) {
            py::dict item;
            item["timestamp_ns"]=sample.timestamp;
            item["acceleration_m_s2"]=py::make_tuple(sample.aX,sample.aY,sample.aZ);
            item["angular_velocity_deg_s"]=py::make_tuple(sample.gX,sample.gY,sample.gZ);
            item["temperature_c"]=sample.temp;
            item["sync"]=sample.sync;
            sensor_samples.append(item);
        }
        py::dict result;
        result["yuyv"]=std::move(yuyv);
        result["timestamp_ns"]=frame.timestamp;
        result["frame_id"]=frame.frame_id;
        result["sensor_samples"]=std::move(sensor_samples);
        result["sensor_dropped_count"]=dropped_.load();
        return result;
    }
};

PYBIND11_MODULE(_zed_capture,m) {
    m.doc()="ZED Open Capture unrectified post-ISP stereo and raw IMU; no proprietary SDK";
    py::register_exception<FrameTimeout>(m,"CaptureTimeout",PyExc_TimeoutError);
    py::class_<ZedCapture>(m,"ZedCapture")
      .def(py::init<int,int,int,int>(),py::arg("width")=1280,py::arg("height")=720,py::arg("fps")=60,py::arg("device_id")=-1)
      .def_property_readonly("serial",&ZedCapture::serial)
      .def_property_readonly("sensors_available",&ZedCapture::sensors_available)
      .def_property_readonly("sensor_status",&ZedCapture::sensor_status)
      .def_property_readonly("sensor_dropped_count",&ZedCapture::sensor_dropped_count)
      .def("read",&ZedCapture::read,py::arg("timeout_ms")=1000)
      .def("close",&ZedCapture::close,py::call_guard<py::gil_scoped_release>())
      .def("__enter__",[](ZedCapture& self)->ZedCapture& {return self;},py::return_value_policy::reference_internal)
      .def("__exit__",[](ZedCapture& self,py::object,py::object,py::object) {py::gil_scoped_release release;self.close();});
}
