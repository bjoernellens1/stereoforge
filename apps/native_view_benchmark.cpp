// Standalone C++20 Slang RHI fusion benchmark. No Python interpreter is embedded.
#include <stereoforge/geometry.hpp>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs=std::filesystem;
using namespace stereoforge;
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
static std::string quote(const std::string& value){
    std::ostringstream out;out<<'"';
    for(unsigned char c:value){if(c=='"'||c=='\\')out<<'\\'<<c;else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c)<<std::dec;else out<<c;}
    out<<'"';return out.str();
}
struct Options {fs::path fixture,output,dump;unsigned frames=5,warmup=2;std::string mode="strict",readback="gpu";};
static unsigned number(const std::string& text,const char* name,bool positive){
    std::size_t used=0;unsigned long value;
    try{value=std::stoul(text,&used);}catch(...){throw std::invalid_argument(std::string("invalid ")+name);}
    if(used!=text.size()||value>100000||(!text.empty()&&text.front()=='-')||(positive&&value==0))throw std::invalid_argument(std::string(name)+(positive?" must be positive and <=100000":" must be between 0 and 100000"));
    return static_cast<unsigned>(value);
}
static Options options(int argc,char** argv){
    Options out;
    for(int i=1;i<argc;++i){std::string key=argv[i];
        if(key!="--fixture"&&key!="--output"&&key!="--dump-prefix"&&key!="--frames"&&key!="--warmup"&&key!="--mode"&&key!="--readback")throw std::invalid_argument("unknown argument: "+key);
        if(++i==argc)throw std::invalid_argument("missing value for "+key);
        std::string value=argv[i];
        if(key=="--fixture")out.fixture=value;else if(key=="--output")out.output=value;else if(key=="--dump-prefix")out.dump=value;
        else if(key=="--frames")out.frames=number(value,"frames",true);else if(key=="--warmup")out.warmup=number(value,"warmup",false);
        else if(key=="--mode")out.mode=value;else if(key=="--readback")out.readback=value;
    }
    if(out.mode!="strict"&&out.mode!="preview")throw std::invalid_argument("mode must be strict or preview");
    if(out.readback!="gpu"&&out.readback!="cpu")throw std::invalid_argument("readback must be gpu or cpu");
    if(out.fixture.empty()||out.output.empty())throw std::invalid_argument("--fixture and --output are required");
    return out;
}
template<class T>static std::vector<T> load(const fs::path& path,std::size_t count){
    if(!fs::is_regular_file(path)||fs::file_size(path)!=count*sizeof(T))throw std::invalid_argument("fixture array byte count mismatch: "+path.string());
    std::vector<T> result(count);std::ifstream in(path,std::ios::binary);
    if(!in.read(reinterpret_cast<char*>(result.data()),static_cast<std::streamsize>(count*sizeof(T))))throw std::runtime_error("cannot read fixture: "+path.string());
    return result;
}
struct Fixture {
    GeometryConfig config{};VirtualView view{};std::int64_t timestamp=0;std::string calibration;
    std::vector<float> depth,confidence;std::vector<std::uint32_t> rgb,provenance,preview;
    std::vector<Ray3> rays;std::vector<std::int64_t> ids,ages;
    GeometrySources sources()const{return {depth,confidence,rgb,provenance,rays,ids,ages,timestamp,calibration,preview};}
};
static Fixture fixture(const fs::path& directory){
    static_assert(std::endian::native==std::endian::little,"fixture binary ABI requires little endian");
    static_assert(sizeof(Ray3)==12);
    Fixture out;std::ifstream in(directory/"header.txt");
    in>>out.config.source_width>>out.config.source_height>>out.config.target_width>>out.config.target_height
      >>out.view.intrinsics.fx>>out.view.intrinsics.fy>>out.view.intrinsics.cx>>out.view.intrinsics.cy>>out.view.hfov_radians;
    for(auto& x:out.view.source_to_virtual_left)in>>x;
    for(auto& x:out.view.source_to_virtual_right)in>>x;
    in>>out.timestamp>>out.calibration;
    std::string extra;if(!in||in>>extra)throw std::invalid_argument("invalid fixture header");
    const auto n=std::uint64_t(out.config.source_width)*out.config.source_height,m=std::uint64_t(out.config.target_width)*out.config.target_height;
    if(!n||!m||n>UINT32_MAX/2||m>UINT32_MAX/2)throw std::invalid_argument("invalid fixture dimensions");
    if(out.timestamp<0||out.calibration.empty()||!std::isfinite(out.view.hfov_radians)||out.view.hfov_radians<=0||out.view.hfov_radians>=3.141592654f)throw std::invalid_argument("invalid fixture header values");
    for(float x:{out.view.intrinsics.fx,out.view.intrinsics.fy,out.view.intrinsics.cx,out.view.intrinsics.cy})if(!std::isfinite(x))throw std::invalid_argument("invalid fixture intrinsics");
    if(out.view.intrinsics.fx<=0||out.view.intrinsics.fy<=0)throw std::invalid_argument("invalid fixture intrinsics");
    for(const auto* pose:{&out.view.source_to_virtual_left,&out.view.source_to_virtual_right})for(float x:*pose)if(!std::isfinite(x))throw std::invalid_argument("invalid fixture pose");
    out.depth=load<float>(directory/"source_depth.f32",2*n);out.confidence=load<float>(directory/"confidence.f32",2*n);
    out.rgb=load<std::uint32_t>(directory/"rgb.u32",2*n);out.provenance=load<std::uint32_t>(directory/"provenance.u32",2*n);
    out.rays=load<Ray3>(directory/"rays.f32",2*n);out.ids=load<std::int64_t>(directory/"source_ids.i64",2*n);
    out.ages=load<std::int64_t>(directory/"age_ns.i64",2*n);out.preview=load<std::uint32_t>(directory/"preview_indices.u32",2*m);
    return out;
}
template<class T>static void binary(const fs::path& path,const std::vector<T>& values){
    std::ofstream out(path,std::ios::binary);if(!out.write(reinterpret_cast<const char*>(values.data()),static_cast<std::streamsize>(values.size()*sizeof(T))))throw std::runtime_error("cannot write "+path.string());
}
static void dump(const fs::path& prefix,const GeometryReadback& out){
    if(!prefix.parent_path().empty())fs::create_directories(prefix.parent_path());
    const auto p=prefix.string();
    binary(p+".depth.f32",out.depth);binary(p+".confidence.f32",out.confidence);binary(p+".rgb.u32",out.rgb);
    binary(p+".validity.u32",out.validity);binary(p+".rgb_validity.u32",out.rgb_validity);
    binary(p+".provenance.u32",out.provenance);binary(p+".source.u32",out.source);binary(p+".approximate.u32",out.approximate);
    binary(p+".source_ids.i64",out.source_ids);binary(p+".age_ns.i64",out.age_ns);binary(p+".owner.u32",out.owner);
}
struct Sample {double enqueue=0,wait=0,readback=0,total=0;};
static double quantile(std::vector<double> values,double fraction){std::sort(values.begin(),values.end());const auto pos=fraction*(values.size()-1);const auto lo=std::size_t(pos),hi=std::min(lo+1,values.size()-1);return values[lo]+(values[hi]-values[lo])*(pos-lo);}
static void summary(std::ostream& out,const std::vector<Sample>& samples,double Sample::*field){
    std::vector<double> values;for(const auto& sample:samples)values.push_back(sample.*field);
    out<<"{\"p50\":"<<quantile(values,.5)<<",\"p95\":"<<quantile(values,.95)<<",\"minimum\":"<<*std::min_element(values.begin(),values.end())<<",\"maximum\":"<<*std::max_element(values.begin(),values.end())<<'}';
}
int main(int argc,char** argv){try{
    const auto opt=options(argc,argv);const auto load_start=Clock::now();auto input=fixture(opt.fixture);const auto load_end=Clock::now();
    input.view.mode=opt.mode=="preview"?ViewMode::Preview:ViewMode::Strict;
    const auto init_start=Clock::now();NativeGeometry engine(input.config,STEREOFORGE_SHADER_DIRECTORY);const auto init_end=Clock::now();
    const auto sources=input.sources();
    for(unsigned i=0;i<opt.warmup;++i){auto frame=engine.project(sources,input.view);frame.completion.wait();if(opt.readback=="cpu")engine.readback(frame);}
    std::vector<Sample> samples;samples.reserve(opt.frames);std::optional<GeometryFrame> last;GeometryReadback last_cpu;
    for(unsigned i=0;i<opt.frames;++i){
        last.reset();const auto start=Clock::now();last=engine.project(sources,input.view);const auto submitted=Clock::now();
        last->completion.wait();const auto completed=Clock::now();
        if(opt.readback=="cpu")last_cpu=engine.readback(*last);
        const auto readback=Clock::now();samples.push_back({ms(start,submitted),ms(submitted,completed),opt.readback=="cpu"?ms(completed,readback):0.,ms(start,readback)});
    }
    double final_dump_readback=0;
    if(!opt.dump.empty()){
        const auto start=Clock::now();if(opt.readback=="gpu")last_cpu=engine.readback(*last);final_dump_readback=ms(start,Clock::now());dump(opt.dump,last_cpu);
    }
    if(!opt.output.parent_path().empty())fs::create_directories(opt.output.parent_path());
    std::ofstream out(opt.output);if(!out)throw std::runtime_error("cannot create report");out<<std::setprecision(12);
    out<<"{\"schema\":1,\"runtime\":\"C++20 Slang RHI Vulkan\",\"python_interpreter_used\":false,\"fixture\":"<<quote(opt.fixture.string())<<",\"mode\":"<<quote(opt.mode)<<",\"readback\":"<<quote(opt.readback)
       <<",\"device\":"<<quote(engine.device_name())<<",\"source_width\":"<<input.config.source_width<<",\"source_height\":"<<input.config.source_height<<",\"target_width\":"<<input.config.target_width<<",\"target_height\":"<<input.config.target_height
       <<",\"timestamp_ns\":"<<input.timestamp<<",\"calibration_id\":"<<quote(input.calibration)<<",\"warmup\":"<<opt.warmup<<",\"frames\":"<<opt.frames<<",\"fixture_load_ms\":"<<ms(load_start,load_end)<<",\"initialization_ms\":"<<ms(init_start,init_end)
       <<",\"scope\":\"Fusion with host source uploads; GPU case waits for completion without output readback. Excludes stereo, camera, display and initialization.\",\"timing_contract\":\"enqueue includes input validation, metadata copies, host uploads and submission; completion_wait starts at project return; cpu readback includes full output planes and metric identity gathering.\""
       <<",\"dump_readback_after_measurement_ms\":"<<final_dump_readback<<",\"dump_prefix\":"<<quote(opt.dump.string())<<",\"summary_ms\":{\"enqueue\":";summary(out,samples,&Sample::enqueue);
    out<<",\"completion_wait\":";summary(out,samples,&Sample::wait);out<<",\"readback\":";summary(out,samples,&Sample::readback);out<<",\"total\":";summary(out,samples,&Sample::total);out<<"},\"samples_ms\":[";
    for(std::size_t i=0;i<samples.size();++i){if(i)out<<',';const auto& s=samples[i];out<<"{\"enqueue\":"<<s.enqueue<<",\"completion_wait\":"<<s.wait<<",\"readback\":"<<s.readback<<",\"total\":"<<s.total<<'}';}
    out<<"]}\n";if(!out)throw std::runtime_error("report write failed");std::cout<<opt.output<<'\n';return 0;
}catch(const std::exception& error){std::cerr<<"native_view_benchmark: "<<error.what()<<'\n';return 2;}}
