#include <stereoforge/capture.hpp>
#include <algorithm>
#include <atomic>
#include <bit>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <limits.h>
#ifdef SF_NATIVE_CAPTURE_ZED
#include "videocapture.hpp"
#include "sensorcapture.hpp"
#endif
namespace stereoforge {
namespace {
uint64_t ingress_now(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
constexpr size_t max_imu=8192,max_string=65536,max_payload=256*1024*1024;
bool zed_sysfs_serial(const std::string& expected){
    std::error_code error;
    for(const auto& entry:std::filesystem::directory_iterator("/sys/class/hidraw",error)){
        if(error)break;
        auto link=entry.path()/"device";
        char resolved[PATH_MAX];
        if(!::realpath(link.c_str(),resolved))continue;
        auto path=std::filesystem::path(resolved);
        for(unsigned depth=0;depth<8&&!path.empty();++depth,path=path.parent_path()){
            std::ifstream vendor(path/"idVendor"),product(path/"idProduct"),serial(path/"serial");
            std::string v,p,s;std::getline(vendor,v);std::getline(product,p);std::getline(serial,s);
            if(v=="2b03"&&p=="f781"&&s==expected)return true;
        }
    }
    return false;
}
void validate(const CaptureFrame& f){
    if(f.source_kind!=CaptureSourceKind::PackedYuyv||f.canonical_rgb)
        throw std::invalid_argument("raw capture requires packed YUYV source");
    if(!f.source_width||f.source_width%4||f.source_width>8192||!f.height||f.height>8192||f.stride_bytes!=f.source_width*2||f.packed_yuyv.size()!=size_t(f.stride_bytes)*f.height||f.packed_yuyv.size()>max_payload||!f.camera_timestamp_ns||f.camera_clock_domain.empty()||f.host_clock_domain.empty()||f.imu_samples.size()>max_imu)
        throw std::invalid_argument("invalid packed YUYV frame dimensions, timestamp, clock or sample count");
    for(auto* s:{&f.camera_clock_domain,&f.host_clock_domain,&f.calibration_identity,&f.device_identity,&f.controls_json,&f.control_status,&f.controls_clock_domain,&f.sensor_status,&f.imu_clock_domain,&f.imu_association})
        if(s->size()>max_string)throw std::invalid_argument("capture metadata exceeds bound");
}
void validate_config(const CaptureConfig& c){
    if(!c.eye_width||c.eye_width%2||c.eye_width>4096||!c.height||c.height>8192||!c.fps||c.fps>100||c.device_id < -1||!c.queue_capacity||c.queue_capacity>64||c.pool_capacity<=c.queue_capacity||c.pool_capacity>128||c.calibration_identity.size()>max_string||c.device_serial_override.size()>32||c.source_timeout.count()<=0||c.source_timeout.count()>1000||size_t(c.eye_width)*4*c.height>max_payload||size_t(c.eye_width)*4*c.height*(c.pool_capacity+1)>512*1024*1024)
        throw std::invalid_argument("invalid capture config: dimensions, bounded pool/queue or source timeout");
}
#ifdef SF_NATIVE_CAPTURE_ZED
class CameraSource {
    CaptureConfig config_;
    std::unique_ptr<sl_oc::video::VideoCapture> video_;
    std::unique_ptr<sl_oc::sensors::SensorCapture> sensors_;
    std::thread sensor_thread_;
    std::atomic<bool> stopping_{false};
    std::mutex mutex_;
    std::array<CaptureImuSample,max_imu> samples_{};
    size_t sample_count_=0;
    uint64_t sensor_drops_=0,last_timestamp_=0;
    std::string device_,sensor_status_="unavailable",controls_;
    uint64_t control_time_=0;
public:
    explicit CameraSource(const CaptureConfig& c):config_(c){
        sl_oc::video::VideoParams p;using R=sl_oc::video::RESOLUTION;
        if(c.eye_width==1280&&c.height==720)p.res=R::HD720;
        else if(c.eye_width==1920&&c.height==1080)p.res=R::HD1080;
        else if(c.eye_width==2208&&c.height==1242)p.res=R::HD2K;
        else if(c.eye_width==672&&c.height==376)p.res=R::VGA;
        else throw std::invalid_argument("unsupported ZED resolution");
        if((c.fps!=15&&c.fps!=30&&c.fps!=60&&c.fps!=100)||(c.eye_width==2208&&c.fps!=15)||(c.eye_width==1920&&c.fps>30)||(c.eye_width==1280&&c.fps>60))throw std::invalid_argument("unsupported ZED resolution/fps");
        p.fps=static_cast<sl_oc::video::FPS>(c.fps);
        video_=std::make_unique<sl_oc::video::VideoCapture>(p);
        if(!video_->initializeVideo(c.device_id))throw std::runtime_error("cannot initialize ZED video: USB mode, permissions or device ownership");
        int w=0,h=0;video_->getFrameSize(w,h);
        if(w!=int(c.eye_width*2)||h!=int(c.height))throw std::runtime_error("unexpected negotiated camera dimensions");
        auto serial=video_->getSerialNumber();
        bool overridden=false;
        if(serial<=0){
            constexpr std::string_view supported_serial="26041639";
            constexpr std::string_view supported_identity="40ed5e0bce10cb535e5c252c6e028d2a34b374d9f8af6a08161fffbb750d9390";
            if(c.device_serial_override!=supported_serial||c.calibration_identity!=supported_identity||!zed_sysfs_serial(c.device_serial_override))
                throw std::runtime_error("unidentified camera serial; explicit supported serial, calibration, and HID/sysfs ZED2 identity are required");
            serial=std::stoi(std::string(supported_serial));overridden=true;
        }
        device_="zed:"+std::to_string(serial)+(overridden?":serial_override":"");
        // Cache controls once. These are API readbacks, not physically verified;
        // automatic bool getters cannot distinguish I/O failure. LED getter mutates GPIO.
        std::ostringstream controls;controls<<"{\"auto_exposure_gain\":"<<(video_->getAECAGC()?"true":"false")<<",\"auto_white_balance\":"<<(video_->getAutoWhiteBalance()?"true":"false");
        using Eye=sl_oc::video::CAM_SENS_POS;
        auto integer=[&](const char* name,int v,int lo,int hi){controls<<",\""<<name<<"\":";if(v<lo||v>hi)controls<<"null";else controls<<v;};
        integer("exposure_left",video_->getExposure(Eye::LEFT),0,100);integer("exposure_right",video_->getExposure(Eye::RIGHT),0,100);
        integer("gain_left",video_->getGain(Eye::LEFT),0,100);integer("gain_right",video_->getGain(Eye::RIGHT),0,100);
        integer("white_balance",video_->getWhiteBalance(),2800,6500);integer("brightness",video_->getBrightness(),0,8);
        integer("contrast",video_->getContrast(),0,8);integer("saturation",video_->getSaturation(),0,8);
        integer("hue",video_->getHue(),0,11);integer("sharpness",video_->getSharpness(),0,8);integer("gamma",video_->getGamma(),1,9);
        for(auto [name,eye]:{std::pair{"roi_left",Eye::LEFT},std::pair{"roi_right",Eye::RIGHT}}){uint16_t x=0,y=0,rw=0,rh=0;bool ok=video_->getROIforAECAGC(eye,x,y,rw,rh);controls<<",\""<<name<<"\":";if(!ok||!rw||!rh||uint32_t(x)+rw>c.eye_width||uint32_t(y)+rh>c.height||uint32_t(rw)*rh<=100)controls<<"null";else controls<<'['<<x<<','<<y<<','<<rw<<','<<rh<<']';}
        controls<<",\"device_serial_override\":"<<(overridden?"true":"false")<<",\"led\":null,\"readback_limitations\":[\"LED getter changes GPIO direction\",\"automatic bool getters cannot distinguish I/O failure\",\"null numeric/ROI values indicate invalid API readback\"]}";controls_=controls.str();control_time_=ingress_now();
        sensors_=std::make_unique<sl_oc::sensors::SensorCapture>();
        if(!sensors_->initializeSensors(serial)){sensor_status_="sensor_open_failed_check_hidraw_permissions";return;}
        sensor_status_=video_->enableSensorSync(sensors_.get())?"hardware_sync_enabled_raw_not_pose":"sync_unavailable_raw_only";
        sensor_thread_=std::thread([this]{uint64_t last=0;while(!stopping_){auto s=sensors_->snapshotIMU(1500);if(s.valid!=sl_oc::sensors::data::Imu::NEW_VAL||s.timestamp==last)continue;last=s.timestamp;std::lock_guard lock(mutex_);if(sample_count_==max_imu){++sensor_drops_;continue;}samples_[sample_count_++]={s.timestamp,{s.aX,s.aY,s.aZ},{s.gX,s.gY,s.gZ},s.temp,s.sync};}});
    }
    ~CameraSource(){stopping_=true;if(sensor_thread_.joinable())sensor_thread_.join();video_.reset();sensors_.reset();}
    CaptureSourceStatus read(CaptureFrame& f,std::chrono::milliseconds timeout){
        auto s=video_->snapshotFrame(f.packed_yuyv,uint64_t(timeout.count()),last_timestamp_);
        if(!s.timestamp||s.timestamp==last_timestamp_)return CaptureSourceStatus::timeout;
        last_timestamp_=s.timestamp;
        if(s.channels!=2)throw std::runtime_error("unexpected camera pixel format");
        f.source_width=s.width;f.height=s.height;f.stride_bytes=s.width*2;f.frame_id=s.frame_id;f.camera_timestamp_ns=s.timestamp;
        // Upstream adjusts UVC timestamps against its host reference. No independently
        // validated mapping to steady time or IMU clock is claimed here.
        f.camera_clock_domain="zed_open_capture_video_unvalidated";f.device_identity=device_;
        f.controls_json=controls_;f.control_status="cached_api_readback_not_physically_verified";f.controls_timestamp_ns=control_time_;f.controls_clock_domain="host_steady";
        f.sensor_status=sensor_status_;f.imu_clock_domain="zed_open_capture_imu_unvalidated";f.imu_association="arrival_batch_unknown_raw_not_pose";
        {std::lock_guard lock(mutex_);f.imu_samples.assign(samples_.begin(),samples_.begin()+sample_count_);sample_count_=0;f.sensor_dropped_count=sensor_drops_;}
        return CaptureSourceStatus::frame;
    }
};
#endif
CaptureSource camera_source(const CaptureConfig& c){validate_config(c);
#ifdef SF_NATIVE_CAPTURE_ZED
    auto camera=std::make_shared<CameraSource>(c);return [camera](CaptureFrame& f,auto t){return camera->read(f,t);};
#else
    throw std::runtime_error("native camera source disabled at build time");
#endif
}
}
void CaptureFrame::require_calibration()const{if(calibration_identity.empty())throw std::invalid_argument("processing requires explicit validated calibration identity");}
struct NativeCapture::Impl {
    CaptureConfig config;
    CaptureSource source;
    std::vector<std::shared_ptr<CaptureFrame>> pool;
    CaptureFrame discard;
    mutable std::mutex mutex;
    std::mutex stop_mutex;
    std::condition_variable ready;
    std::vector<CaptureLease> queue;
    size_t queue_head=0,queue_size=0;
    CaptureMetrics stats;
    std::atomic<bool> stopping{false};
    std::exception_ptr error;
    std::thread thread;
    Impl(CaptureConfig c,CaptureSource s):config(std::move(c)),source(std::move(s)){
        validate_config(config);queue.resize(config.queue_capacity);if(!source)throw std::invalid_argument("empty capture source");
        auto prepare=[&](CaptureFrame& f){f.packed_yuyv.reserve(size_t(config.eye_width)*4*config.height);f.imu_samples.reserve(max_imu);f.calibration_identity=config.calibration_identity;};
        pool.reserve(config.pool_capacity);for(size_t i=0;i<config.pool_capacity;++i){auto f=std::make_shared<CaptureFrame>();prepare(*f);pool.push_back(std::move(f));}prepare(discard);
        stats.pool_capacity=pool.size();stats.packed_bytes_per_slot=size_t(config.eye_width)*4*config.height;stats.running=true;
        thread=std::thread([this]{run();});
    }
    void run(){
        uint64_t last_id=0,last_ts=0;bool seen=false;std::string last_clock;
        try{while(!stopping){
            std::shared_ptr<CaptureFrame> slot;
            {std::unique_lock lock(mutex);
                if(config.queue_policy==CaptureQueuePolicy::wait_for_consumer){
                    uint64_t wait_start=0;
                    while(!stopping){
                        bool free=false;for(const auto& p:pool)if(p.use_count()==1){free=true;break;}
                        if(queue_size<queue.size()&&free)break;
                        // Leases may be released without facade access; bounded timed wait
                        // observes that release without placing the facade in the lease.
                        if(!wait_start){wait_start=ingress_now();++stats.backpressure_waits;}
                        ready.wait_for(lock,std::chrono::milliseconds{1});
                    }
                    if(wait_start)stats.backpressure_wait_ns+=ingress_now()-wait_start;
                    if(stopping)break;
                }
                for(auto& p:pool)if(p.use_count()==1){slot=p;break;}
            }
            auto& frame=slot?*slot:discard;
            frame.imu_samples.clear();
            frame.host_ingress_timestamp_ns=0;
            auto status=source(frame,config.source_timeout);
            auto ingress=ingress_now();
            if(status==CaptureSourceStatus::end)break;
            std::lock_guard lock(mutex);
            if(status==CaptureSourceStatus::timeout){++stats.source_timeouts;continue;}
            try{validate(frame);if(frame.source_width!=config.eye_width*2||frame.height!=config.height)throw std::invalid_argument("source dimensions changed");}catch(...){++stats.invalid_frames;throw;}
            ++stats.source_records;
            bool repeated=seen&&frame.camera_clock_domain==last_clock&&frame.frame_id==last_id&&frame.camera_timestamp_ns==last_ts;
            if(repeated){++stats.repeated_frames;if(status!=CaptureSourceStatus::recorded_frame)continue;}
            else ++stats.source_frames;
            if(seen&&frame.camera_clock_domain==last_clock&&frame.frame_id>last_id&&frame.frame_id-last_id>1)stats.source_frame_gaps+=frame.frame_id-last_id-1;
            seen=true;last_id=frame.frame_id;last_ts=frame.camera_timestamp_ns;last_clock=frame.camera_clock_domain;
            // Replay retains original ingress and domains; hardware leaves it unset.
            if(status!=CaptureSourceStatus::recorded_frame&&!frame.host_ingress_timestamp_ns){frame.host_ingress_timestamp_ns=ingress;frame.host_clock_domain="host_steady";}
            if(!config.calibration_identity.empty()&&frame.calibration_identity!=config.calibration_identity)throw std::invalid_argument("source calibration identity differs from capture config");
            if(!slot){++stats.no_slot_drops;discard.host_ingress_timestamp_ns=0;continue;}
            if(queue_size==config.queue_capacity){queue[queue_head].reset();queue_head=(queue_head+1)%queue.size();--queue_size;++stats.queue_drops;}
            queue[(queue_head+queue_size)%queue.size()]=std::move(slot);++queue_size;stats.queue_high_watermark=std::max(stats.queue_high_watermark,queue_size);ready.notify_one();
        }}catch(...){std::lock_guard lock(mutex);error=std::current_exception();}
        {std::lock_guard lock(mutex);stats.running=false;}ready.notify_all();
    }
    void stop(){std::lock_guard stop_lock(stop_mutex);stopping=true;ready.notify_all();if(thread.joinable())thread.join();source={};}
    ~Impl(){stop();}
};
NativeCapture::NativeCapture(CaptureConfig c):NativeCapture(c,camera_source(c)){}
NativeCapture::NativeCapture(CaptureConfig c,CaptureSource s):impl_(std::make_unique<Impl>(std::move(c),std::move(s))){}
NativeCapture::~NativeCapture()=default;
void NativeCapture::stop(){impl_->stop();}
CaptureMetrics NativeCapture::metrics()const{std::lock_guard lock(impl_->mutex);auto result=impl_->stats;result.queued_frames=impl_->queue_size;return result;}
CaptureLease NativeCapture::read(std::chrono::milliseconds timeout){
    if(timeout.count()<=0)throw std::invalid_argument("read timeout must be positive");
    std::unique_lock lock(impl_->mutex);impl_->ready.wait_for(lock,timeout,[&]{return impl_->queue_size!=0||!impl_->stats.running||impl_->stopping.load();});
    if(impl_->error)std::rethrow_exception(impl_->error);
    if(!impl_->queue_size)return {};
    auto f=std::move(impl_->queue[impl_->queue_head]);impl_->queue_head=(impl_->queue_head+1)%impl_->queue.size();--impl_->queue_size;++impl_->stats.delivered_frames;impl_->ready.notify_all();return f;
}
namespace {
// Explicit byte order and exact float bits; FNV-1a only detects accidental corruption.
struct RecordIO {
    std::istream* in=nullptr;std::ostream* out=nullptr;uint64_t hash=14695981039346656037ull;
    void bytes(char* p,size_t n){if(in){in->read(p,n);if(!*in)throw std::runtime_error("truncated packed capture");}else{out->write(p,n);if(!*out)throw std::runtime_error("packed capture write failed");}for(size_t i=0;i<n;++i){hash^=static_cast<uint8_t>(p[i]);hash*=1099511628211ull;}}
    template<class T> void number(T& value){std::array<char,sizeof(T)> b{};if(out)for(size_t i=0;i<b.size();++i)b[i]=char((value>>(i*8))&255);bytes(b.data(),b.size());if(in){value=0;for(size_t i=0;i<b.size();++i)value|=T(static_cast<uint8_t>(b[i]))<<(i*8);}}
    void text(std::string& s){uint32_t n=uint32_t(s.size());number(n);if(n>max_string)throw std::runtime_error("packed metadata too large");if(in)s.resize(n);bytes(s.data(),n);}
    void floating(float& f){uint32_t bits=std::bit_cast<uint32_t>(f);number(bits);if(in)f=std::bit_cast<float>(bits);}
};
void record(RecordIO& io,CaptureFrame& f){
    if(io.in&&(f.source_kind!=CaptureSourceKind::PackedYuyv||f.canonical_rgb))
        throw std::invalid_argument("raw replay requires packed YUYV destination");
    io.number(f.source_width);io.number(f.height);io.number(f.stride_bytes);io.number(f.frame_id);io.number(f.camera_timestamp_ns);io.number(f.host_ingress_timestamp_ns);
    io.text(f.camera_clock_domain);io.text(f.host_clock_domain);io.text(f.calibration_identity);io.text(f.device_identity);io.text(f.controls_json);io.text(f.control_status);io.number(f.controls_timestamp_ns);io.text(f.controls_clock_domain);
    io.text(f.sensor_status);io.text(f.imu_clock_domain);io.text(f.imu_association);io.number(f.sensor_dropped_count);
    uint32_t count=uint32_t(f.imu_samples.size());io.number(count);if(count>max_imu)throw std::runtime_error("packed IMU count exceeds bound");if(io.in)f.imu_samples.resize(count);
    for(auto& s:f.imu_samples){io.number(s.timestamp_ns);for(auto& v:s.acceleration_m_s2)io.floating(v);for(auto& v:s.angular_velocity_deg_s)io.floating(v);io.floating(s.temperature_c);uint8_t sync=s.sync;io.number(sync);if(sync>1)throw std::runtime_error("invalid IMU sync flag");if(io.in)s.sync=bool(sync);}
    uint64_t size=f.packed_yuyv.size();io.number(size);if(size>max_payload||size!=uint64_t(f.stride_bytes)*f.height)throw std::runtime_error("invalid packed payload size");if(io.in)f.packed_yuyv.resize(size);
    io.bytes(reinterpret_cast<char*>(f.packed_yuyv.data()),size);auto expected=io.hash,stored=expected;io.number(stored);if(io.in&&stored!=expected)throw std::runtime_error("packed capture FNV-1a-64 checksum mismatch");validate(f);
}
constexpr char magic[8]={'S','F','Y','U','Y','V','1','\0'};
void header(std::istream* in,std::ostream* out,uint64_t& count){RecordIO io{in,out};char m[8];std::copy(std::begin(magic),std::end(magic),m);io.bytes(m,8);if(!std::equal(std::begin(m),std::end(m),std::begin(magic)))throw std::runtime_error("invalid packed capture magic");uint32_t version=1;io.number(version);if(version!=1)throw std::runtime_error("unsupported packed capture version");io.number(count);if(in&&(count==UINT64_MAX||count>10000000))throw std::runtime_error("nonfinalized or excessive packed capture count");}
struct Reader {
    std::ifstream file;uint64_t remaining=0;
    explicit Reader(const std::filesystem::path& d):file(d/"capture.sfp",std::ios::binary){if(!file)throw std::runtime_error("cannot open packed capture");header(&file,nullptr,remaining);}
    bool next(CaptureFrame& f){if(!remaining){if(file.peek()!=std::char_traits<char>::eof())throw std::runtime_error("trailing packed capture bytes");return false;}RecordIO io{&file,nullptr};record(io,f);--remaining;if(!remaining&&file.peek()!=std::char_traits<char>::eof())throw std::runtime_error("trailing packed capture bytes");return true;}
};
}
struct PackedCaptureWriter::Impl {
    std::fstream file;uint64_t count=0;bool closed=false;
    explicit Impl(const std::filesystem::path& d){if(std::filesystem::exists(d)&&!std::filesystem::is_empty(d))throw std::invalid_argument("recording directory must be new or empty");std::filesystem::create_directories(d);file.open(d/"capture.sfp",std::ios::binary|std::ios::out|std::ios::in|std::ios::trunc);if(!file)throw std::runtime_error("cannot create packed recording");uint64_t pending=UINT64_MAX;header(nullptr,&file,pending);}
};
PackedCaptureWriter::PackedCaptureWriter(const std::filesystem::path& d):impl_(std::make_unique<Impl>(d)){}
PackedCaptureWriter::~PackedCaptureWriter(){/* Explicit close finalizes. An interrupted recording is rejected. */}
void PackedCaptureWriter::append(const CaptureFrame& f){if(impl_->closed)throw std::runtime_error("recording closed");validate(f);RecordIO io{nullptr,&impl_->file};record(io,const_cast<CaptureFrame&>(f));++impl_->count;}
void PackedCaptureWriter::close(){if(impl_->closed)return;impl_->file.flush();if(!impl_->file)throw std::runtime_error("recording flush failed");impl_->file.seekp(12);RecordIO io{nullptr,&impl_->file};io.number(impl_->count);impl_->file.flush();if(!impl_->file)throw std::runtime_error("recording finalization failed");impl_->closed=true;impl_->file.close();}
std::vector<CaptureFrame> read_packed_capture(const std::filesystem::path& d,size_t max_frames,size_t max_bytes){
    Reader r(d);if(r.remaining>max_frames)throw std::runtime_error("fixture exceeds frame bound; use streaming source");
    std::vector<CaptureFrame> frames;size_t total=0;CaptureFrame f;
    while(r.next(f)){
        size_t bytes=sizeof(f)+f.packed_yuyv.size()+f.imu_samples.size()*sizeof(CaptureImuSample);
        for(auto* s:{&f.camera_clock_domain,&f.host_clock_domain,&f.calibration_identity,&f.device_identity,&f.controls_json,&f.control_status,&f.controls_clock_domain,&f.sensor_status,&f.imu_clock_domain,&f.imu_association})bytes+=s->size();
        if(bytes>max_bytes-total)throw std::runtime_error("fixture exceeds byte bound; use streaming source");
        total+=bytes;frames.push_back(std::move(f));f=CaptureFrame{};
    }return frames;
}
CaptureSource packed_replay_source(const std::filesystem::path& d){auto reader=std::make_shared<Reader>(d);return [reader](CaptureFrame& f,auto){return reader->next(f)?CaptureSourceStatus::recorded_frame:CaptureSourceStatus::end;};}
}
