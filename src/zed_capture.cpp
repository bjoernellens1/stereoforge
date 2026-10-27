// Native ZED Open Capture binding. SDK buffers are snapshotted while holding
// their producer mutex by the narrowly generated adapters in Capture.cmake.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "videocapture.hpp"
#include "sensorcapture.hpp"
#include <atomic>
#include <array>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <stdexcept>

namespace py = pybind11;
struct FrameTimeout : std::runtime_error { using std::runtime_error::runtime_error; };

namespace {
using Eye = sl_oc::video::CAM_SENS_POS;
struct Range { int minimum,maximum; };
const std::map<std::string,Range> control_ranges{
    {"exposure_left",{0,100}},{"exposure_right",{0,100}},{"gain_left",{0,100}},{"gain_right",{0,100}},
    {"white_balance",{2800,6500}},{"brightness",{0,8}},{"contrast",{0,8}},
    {"saturation",{0,8}},{"hue",{0,11}},{"sharpness",{0,8}},{"gamma",{1,9}}
};
struct ControlRequest {
    std::map<std::string,int> integers;
    std::map<std::string,bool> booleans;
    std::map<std::string,std::optional<std::array<uint16_t,4>>> rois;
};
long long strict_integer(py::handle value,const std::string& name) {
    if(!PyLong_CheckExact(value.ptr()))throw py::type_error(name+" must be an integer, excluding bool");
    const auto number=PyLong_AsLongLong(value.ptr());
    if(PyErr_Occurred())throw py::error_already_set();
    return number;
}
ControlRequest parse_controls(int width,int height,const py::dict& controls) {
    if(width<=0||height<=0||width>65535||height>65535)throw std::invalid_argument("invalid per-eye control dimensions");
    ControlRequest request;
    for(auto item:controls) {
        if(!py::isinstance<py::str>(item.first))throw py::type_error("control names must be strings");
        const auto name=py::cast<std::string>(item.first);
        if(control_ranges.contains(name)) {
            const auto value=strict_integer(item.second,name);const auto range=control_ranges.at(name);
            if(value<range.minimum||value>range.maximum)throw std::invalid_argument(name+" is outside ["+std::to_string(range.minimum)+","+std::to_string(range.maximum)+"]");
            request.integers[name]=static_cast<int>(value);
        } else if(name=="auto_exposure_gain"||name=="auto_white_balance"||name=="led") {
            if(!PyBool_Check(item.second.ptr()))throw py::type_error(name+" must be bool");
            request.booleans[name]=item.second.ptr()==Py_True;
        } else if(name=="roi_left"||name=="roi_right") {
            if(item.second.is_none()){request.rois[name]=std::nullopt;continue;}
            if(!PyList_Check(item.second.ptr())&&!PyTuple_Check(item.second.ptr()))throw py::type_error(name+" must be [x,y,width,height] or None");
            auto sequence=py::reinterpret_borrow<py::sequence>(item.second);
            if(sequence.size()!=4)throw std::invalid_argument(name+" needs four integers");
            std::array<long long,4> roi{};
            for(int i=0;i<4;++i)roi[i]=strict_integer(sequence[i],name);
            if(roi[0]<0||roi[1]<0||roi[2]<=0||roi[3]<=0||roi[0]>=width||roi[1]>=height||
                roi[2]>width-roi[0]||roi[3]>height-roi[1])throw std::invalid_argument(name+" must be positive and inside the per-eye image");
            // Pinned Open Capture rejects area <=100 before writing registers.
            if(roi[2]*roi[3]<=100)throw std::invalid_argument(name+" area must exceed 100 pixels");
            request.rois[name]=std::array<uint16_t,4>{static_cast<uint16_t>(roi[0]),static_cast<uint16_t>(roi[1]),static_cast<uint16_t>(roi[2]),static_cast<uint16_t>(roi[3])};
        } else throw std::invalid_argument("unknown or read-only control: "+name);
    }
    if(request.booleans.contains("auto_exposure_gain")&&request.booleans.at("auto_exposure_gain"))
        for(const auto* name:{"exposure_left","exposure_right","gain_left","gain_right"})
            if(request.integers.contains(name))throw std::invalid_argument("manual exposure/gain conflicts with auto_exposure_gain=True");
    if(request.integers.contains("white_balance")&&request.booleans.contains("auto_white_balance")&&request.booleans.at("auto_white_balance"))
        throw std::invalid_argument("manual white balance conflicts with auto_white_balance=True");
    return request;
}
struct ControlSnapshot {
    std::map<std::string,std::optional<int>> integers;
    std::map<std::string,std::optional<std::array<uint16_t,4>>> rois;
    bool auto_exposure_gain=false,auto_white_balance=false;
    std::optional<bool> led;
    std::vector<std::string> read_errors,write_errors;
    std::vector<std::string> readback_limitations{"LED readback omitted: upstream getLEDstatus changes GPIO direction"};
};
py::dict encode_controls(const ControlSnapshot& snapshot) {
    py::dict result;
    result["auto_exposure_gain"]=snapshot.auto_exposure_gain;
    result["auto_white_balance"]=snapshot.auto_white_balance;
    result["led"]=py::cast(snapshot.led);
    for(const auto& [key,value]:snapshot.integers)result[py::str(key)]=py::cast(value);
    for(const auto& [key,value]:snapshot.rois)result[py::str(key)]=py::cast(value);
    result["read_errors"]=snapshot.read_errors;result["write_errors"]=snapshot.write_errors;
    result["readback_limitations"]=snapshot.readback_limitations;
    return result;
}
} // namespace


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

    ControlSnapshot controls_locked() {
        ControlSnapshot state;
        auto read_int=[&](const char* name,int value){
            const auto range=control_ranges.at(name);
            if(value<range.minimum||value>range.maximum){
                state.integers[name]=std::nullopt;
                state.read_errors.push_back(std::string(name)+": out-of-range API readback "+std::to_string(value));
            }else state.integers[name]=value;
        };
        state.auto_exposure_gain=video_->getAECAGC();state.auto_white_balance=video_->getAutoWhiteBalance();
        read_int("exposure_left",video_->getExposure(Eye::LEFT));read_int("exposure_right",video_->getExposure(Eye::RIGHT));
        read_int("gain_left",video_->getGain(Eye::LEFT));read_int("gain_right",video_->getGain(Eye::RIGHT));
        read_int("white_balance",video_->getWhiteBalance());read_int("brightness",video_->getBrightness());
        read_int("contrast",video_->getContrast());read_int("saturation",video_->getSaturation());
        read_int("hue",video_->getHue());read_int("sharpness",video_->getSharpness());read_int("gamma",video_->getGamma());
        // LED intentionally remains unavailable: the upstream getter changes
        // GPIO direction. A control snapshot must not change that setting.
        for(const auto& [name,eye]:{std::pair{"roi_left",Eye::LEFT},std::pair{"roi_right",Eye::RIGHT}}){
            uint16_t x=0,y=0,w=0,h=0;
            const bool success=video_->getROIforAECAGC(eye,x,y,w,h);
            if(!success||w==0||h==0||static_cast<int>(x)+w>width_||static_cast<int>(y)+h>height_||static_cast<uint32_t>(w)*h<=100){
                state.rois[name]=std::nullopt;state.read_errors.push_back(std::string(name)+": API read failed or returned invalid ROI");
            }else state.rois[name]=std::array<uint16_t,4>{x,y,w,h};
        }
        return state;
    }

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
    py::dict get_controls() {
        ControlSnapshot state;
        {py::gil_scoped_release release;std::lock_guard<std::mutex> lock(io_mutex_);
            if(!video_)throw std::runtime_error("camera is closed");
            state=controls_locked();
        }
        return encode_controls(state);
    }
    py::dict set_controls(const py::dict& controls) {
        // Parse the complete request while holding the GIL, before any USB I/O.
        const auto request=parse_controls(width_,height_,controls);
        ControlSnapshot state;
        {py::gil_scoped_release release;std::lock_guard<std::mutex> lock(io_mutex_);
            if(!video_)throw std::runtime_error("camera is closed");
            std::vector<std::string> errors;
            if(request.integers.contains("exposure_left"))video_->setExposure(Eye::LEFT,request.integers.at("exposure_left"));
            if(request.integers.contains("exposure_right"))video_->setExposure(Eye::RIGHT,request.integers.at("exposure_right"));
            if(request.integers.contains("gain_left"))video_->setGain(Eye::LEFT,request.integers.at("gain_left"));
            if(request.integers.contains("gain_right"))video_->setGain(Eye::RIGHT,request.integers.at("gain_right"));
            using Setter=void(sl_oc::video::VideoCapture::*)(int);
            const std::pair<const char*,Setter> setters[]={
                {"brightness",&sl_oc::video::VideoCapture::setBrightness},{"contrast",&sl_oc::video::VideoCapture::setContrast},
                {"saturation",&sl_oc::video::VideoCapture::setSaturation},{"hue",&sl_oc::video::VideoCapture::setHue},
                {"sharpness",&sl_oc::video::VideoCapture::setSharpness},{"gamma",&sl_oc::video::VideoCapture::setGamma},
                {"white_balance",&sl_oc::video::VideoCapture::setWhiteBalance}};
            for(auto [name,setter]:setters)if(request.integers.contains(name))(video_.get()->*setter)(request.integers.at(name));
            if(request.booleans.contains("led")&&video_->setLEDstatus(request.booleans.at("led"))!=0)errors.push_back("led: API write failed");
            for(const auto& [name,roi]:request.rois){
                const auto eye=name=="roi_left"?Eye::LEFT:Eye::RIGHT;
                const bool success=roi?video_->setROIforAECAGC(eye,(*roi)[0],(*roi)[1],(*roi)[2],(*roi)[3]):video_->resetROIforAECAGC(eye);
                if(!success)errors.push_back(name+": API write failed");
            }
            // Manual exposure/gain and white balance may disable automatic modes.
            // Explicit automatic flags therefore run last.
            if(request.booleans.contains("auto_exposure_gain")&&video_->setAECAGC(request.booleans.at("auto_exposure_gain"))!=0)errors.push_back("auto_exposure_gain: API write failed");
            if(request.booleans.contains("auto_white_balance"))video_->setAutoWhiteBalance(request.booleans.at("auto_white_balance"));
            state=controls_locked();state.write_errors=std::move(errors);
        }
        return encode_controls(state);
    }
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
    m.def("validate_controls",[](int width,int height,const py::dict& controls){parse_controls(width,height,controls);},
      py::arg("width"),py::arg("height"),py::arg("controls"),
      "Validate a complete control request offline without opening a device. ROI must fit the per-eye canvas and have area >100 pixels.");
    py::register_exception<FrameTimeout>(m,"CaptureTimeout",PyExc_TimeoutError);
    py::class_<ZedCapture>(m,"ZedCapture")
      .def(py::init<int,int,int,int>(),py::arg("width")=1280,py::arg("height")=720,py::arg("fps")=60,py::arg("device_id")=-1)
      .def_property_readonly("serial",&ZedCapture::serial)
      .def_property_readonly("sensors_available",&ZedCapture::sensors_available)
      .def_property_readonly("sensor_status",&ZedCapture::sensor_status)
      .def_property_readonly("sensor_dropped_count",&ZedCapture::sensor_dropped_count)
      .def("get_controls",&ZedCapture::get_controls,
        "Read camera API controls. Bool automatic getters cannot distinguish I/O failure; values are API-reported, not physically verified. Known read failures are None with read_errors. LED is None with readback_limitations because its upstream getter changes GPIO direction.")
      .def("set_controls",&ZedCapture::set_controls,py::arg("controls"),
        "Validate all settings before USB writes, then return flat API readback/errors. Void upstream setters cannot prove write success; compare readback. Explicit LED writes have no safe readback verification. Automatic bool getters cannot distinguish I/O failure.")
      .def("read",&ZedCapture::read,py::arg("timeout_ms")=1000)
      .def("close",&ZedCapture::close,py::call_guard<py::gil_scoped_release>())
      .def("__enter__",[](ZedCapture& self)->ZedCapture& {return self;},py::return_value_policy::reference_internal)
      .def("__exit__",[](ZedCapture& self,py::object,py::object,py::object) {py::gil_scoped_release release;self.close();});
}
