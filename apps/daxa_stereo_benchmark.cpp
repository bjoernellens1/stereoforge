#include <stereoforge/daxa_stereo.hpp>
#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
using namespace stereoforge;
namespace {
using Clock=std::chrono::steady_clock;
double ms(Clock::duration d){return std::chrono::duration<double,std::milli>(d).count();}
uint32_t integer(const std::string& s){uint32_t value=0;auto [p,e]=std::from_chars(s.data(),s.data()+s.size(),value);if(e!=std::errc{}||p!=s.data()+s.size())throw std::invalid_argument("invalid unsigned integer: "+s);return value;}
float number(const std::string& s){size_t end=0;auto v=std::stof(s,&end);if(end!=s.size()||!std::isfinite(v))throw std::invalid_argument("invalid finite number");return v;}
std::vector<uint32_t> read(const std::string& name,size_t n){std::ifstream file(name,std::ios::binary|std::ios::ate);if(!file||file.tellg()!=std::streamoff(n*4))throw std::invalid_argument("luma input must contain exactly WxH uint32 pixels");file.seekg(0);std::vector<uint32_t> data(n);file.read(reinterpret_cast<char*>(data.data()),n*4);if(!file)throw std::runtime_error("luma input read failed");return data;}
void fresh_destination(const std::string& name){if(std::filesystem::symlink_status(name).type()!=std::filesystem::file_type::not_found)throw std::invalid_argument("destination must be a new file: "+name);}
void write_bytes(const std::string& name,const char* bytes,size_t size){
    fresh_destination(name);int fd=::open(name.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0666);if(fd<0)throw std::runtime_error("exclusive destination create failed: "+name);
    size_t offset=0;while(offset<size){auto count=::write(fd,bytes+offset,size-offset);if(count<0&&errno==EINTR)continue;if(count<=0){::close(fd);throw std::runtime_error("destination write failed: "+name);}offset+=size_t(count);}
    if(::close(fd))throw std::runtime_error("destination close failed: "+name);
}
template<class T>void write(const std::string& name,const std::vector<T>& data){write_bytes(name,reinterpret_cast<const char*>(data.data()),data.size()*sizeof(T));}
struct Summary {double p50,p95,mean;};
Summary summary(std::vector<double> v){std::sort(v.begin(),v.end());return {v[v.size()/2],v[std::min(v.size()-1,v.size()*95/100)],std::accumulate(v.begin(),v.end(),0.)/v.size()};}
void stats(std::ostream& file,const Summary& s){file<<"{\"p50\":"<<s.p50<<",\"p95\":"<<s.p95<<",\"mean\":"<<s.mean<<'}';}
}
int main(int argc,char** argv){try{
    if(argc==2&&std::string(argv[1])=="--shader-directory"){std::cout<<DaxaStereo::shader_directory()<<'\n';return 0;}
    std::map<std::string,std::string> options;bool validation=false,intermediate=false,check_reference=false,check_integer_reference=false;
    for(int i=1;i<argc;++i){std::string key=argv[i];if(key=="--validation"){validation=true;continue;}if(key=="--dump-intermediates"){intermediate=true;continue;}if(key=="--check-integer-reference"){check_integer_reference=true;continue;}if(key=="--check-reference"){check_reference=true;continue;}if(key=="--help"){std::cout<<"daxa_stereo_benchmark --variant baseline|wave|packed|packed-specialized|packed-d128|packed-quad|packed-shuffle [--reference left|right] --output FILE [--width 1280 --height 720 --disparities 128 --paths 4|8 --p1 8 --p2 32 --subpixel 0|1] [--frames 30 --warmup 3 --slots 1|2 --wave-size 0|32|64] [--left FILE --right FILE] [--dump-prefix PREFIX --dump-intermediates] [--check-reference|--check-integer-reference] [--validation] [--fx PX --baseline-m M --principal-offset PX] [--shader-directory]\nSingle-reference retained-luma GPU profile; excludes capture/preprocess, second reference, LR consistency, upload, inspect and diagnostic dump.\n";return 0;}
        if(key!="--reference"&&key!="--wave-size"&&key!="--variant"&&key!="--output"&&key!="--width"&&key!="--height"&&key!="--disparities"&&key!="--paths"&&key!="--p1"&&key!="--p2"&&key!="--subpixel"&&key!="--frames"&&key!="--warmup"&&key!="--slots"&&key!="--left"&&key!="--right"&&key!="--dump-prefix"&&key!="--fx"&&key!="--baseline-m"&&key!="--principal-offset")throw std::invalid_argument("unknown stereo benchmark option");if(++i==argc)throw std::invalid_argument("missing option value");options[key]=argv[i];}
    auto value=[&](const char* key,uint32_t fallback){return options.contains(key)?integer(options.at(key)):fallback;};
    StereoConfig config{value("--width",1280),value("--height",720),value("--disparities",128),value("--paths",4),value("--p1",8),value("--p2",32),value("--subpixel",1)!=0};config.validate();auto frames=value("--frames",30),warmup=value("--warmup",3),slots=value("--slots",1);if(!frames||frames>100000||warmup>10000)throw std::invalid_argument("invalid bounded frame/warmup counts");
    auto variant=options.contains("--variant")?options.at("--variant"):"baseline";if(variant!="baseline"&&variant!="wave"&&variant!="packed"&&variant!="packed-specialized"&&variant!="packed-d128"&&variant!="packed-quad"&&variant!="packed-shuffle")throw std::invalid_argument("unknown SGM variant");if(!options.contains("--output"))throw std::invalid_argument("--output required");fresh_destination(options.at("--output"));
    std::optional<MetricCalibration> metric;if(options.contains("--fx")!=options.contains("--baseline-m")||(options.contains("--principal-offset")&&!options.contains("--fx")))throw std::invalid_argument("metric depth needs explicit focal pixels and baseline metres");if(options.contains("--fx")){metric=MetricCalibration{number(options.at("--fx")),number(options.at("--baseline-m")),options.contains("--principal-offset")?number(options.at("--principal-offset")):0};metric->validate();}
    auto selected_variant=variant=="baseline"?DaxaSgmVariant::Baseline:variant=="wave"?DaxaSgmVariant::Wave:variant=="packed"?DaxaSgmVariant::Packed:variant=="packed-specialized"?DaxaSgmVariant::PackedSpecialized:variant=="packed-d128"?DaxaSgmVariant::PackedD128:variant=="packed-quad"?DaxaSgmVariant::PackedQuad:DaxaSgmVariant::PackedShuffle;
    auto reference_name=options.contains("--reference")?options.at("--reference"):"left";if(reference_name!="left"&&reference_name!="right")throw std::invalid_argument("unknown stereo reference");auto reference_geometry=reference_name=="right"?DaxaStereoReference::Right:DaxaStereoReference::Left;
    DaxaStereo::validate_shape(config,slots,metric,selected_variant,reference_geometry);
    if(options.contains("--dump-prefix")){auto prefix=options.at("--dump-prefix");std::vector<std::string> destinations{options.at("--output"),prefix+".disparity.f32",prefix+".confidence.f32",prefix+".validity.u32"};if(metric){destinations.push_back(prefix+".depth.f32");destinations.push_back(prefix+".depth_validity.u32");}if(intermediate){destinations.push_back(prefix+".costs.u32");destinations.push_back(prefix+".sums.u32");if(selected_variant!=DaxaSgmVariant::Baseline&&selected_variant!=DaxaSgmVariant::Wave)destinations.push_back(prefix+".packed_penalties.u32");}std::vector<std::filesystem::path> identities;for(auto const& name:destinations)identities.push_back(std::filesystem::weakly_canonical(name));std::sort(identities.begin(),identities.end());if(std::adjacent_find(identities.begin(),identities.end())!=identities.end())throw std::invalid_argument("report and dump destinations overlap");for(auto const& name:destinations)fresh_destination(name);}
    if(options.contains("--left")!=options.contains("--right"))throw std::invalid_argument("both retained luma files required");std::array<std::vector<uint32_t>,2> luma;if(options.contains("--left")){auto n=size_t(config.width)*config.height;luma[0]=read(options.at("--left"),n);luma[1]=read(options.at("--right"),n);}else luma=daxa_stereo_fixture(config);
    DaxaStereo::validate(config,luma[0],luma[1],slots,metric,selected_variant,reference_geometry);
    std::ostringstream output;auto init_begin=Clock::now();auto context=std::make_shared<DaxaFoundation>(validation);auto engine=std::make_unique<DaxaStereo>(config,luma[0],luma[1],selected_variant,slots,validation,metric,context,value("--wave-size",0),reference_geometry);double initialization=ms(Clock::now()-init_begin);
    DaxaStereo::Frame frame;for(uint32_t i=0;i<warmup;++i){frame={};frame=engine->process();frame.wait();}
    std::vector<DaxaStereoTiming> gpu(frames);std::vector<double> submit(frames),wait(frames),query(frames),roundtrip(frames),total_gpu(frames);
    for(uint32_t i=0;i<frames;++i){
        frame={};auto a=Clock::now();frame=engine->process();auto b=Clock::now();frame.wait();auto c=Clock::now();gpu[i]=frame.timings();auto d=Clock::now();submit[i]=ms(b-a);wait[i]=ms(c-b);query[i]=ms(d-c);roundtrip[i]=ms(d-a);total_gpu[i]=gpu[i].total_gpu_ms;
    }
    auto inspect_begin=Clock::now();auto counters=engine->inspect(frame);double inspect_ms=ms(Clock::now()-inspect_begin),dump_ms=0,reference_ms=0;bool reference_passed=false,integer_reference_passed=false,packed_reference_passed=false,reconstructed_intermediates=false;
    if(options.contains("--dump-prefix")||check_reference||check_integer_reference){auto begin=Clock::now();auto result=engine->readback(frame,intermediate||check_reference||check_integer_reference);dump_ms=ms(Clock::now()-begin);reconstructed_intermediates=result.intermediates_reconstructed;
        if(check_reference||check_integer_reference){
            begin=Clock::now();auto reference=daxa_stereo_reference(config,luma[0],luma[1],metric,reference_geometry);
            if(result.reference!=reference_geometry||frame.reference()!=reference_geometry||engine->reference()!=reference_geometry||result.census!=reference.census||result.census_valid!=reference.census_valid||result.costs!=reference.costs||result.sums!=reference.sums||result.validity!=reference.validity)throw std::runtime_error("GPU scalar integer/Census/mask parity failed");
            if(engine->packing_bits()){
                auto bits=engine->packing_bits(),components=32/bits,words=(config.disparities+components-1)/components;std::vector<uint32_t> expected(size_t(config.width)*config.height*words);
                // This oracle independently packs SCALAR full sums minus paths*C;
                // it does not reuse the packed lane model or reconstructed GPU sums.
                for(size_t i=0;i<size_t(config.width)*config.height;++i)for(uint32_t d=0;d<config.disparities;++d){auto index=i*config.disparities+d;auto base=config.paths*reference.costs[index];if(reference.sums[index]<base||reference.sums[index]-base>config.paths*config.p2)throw std::runtime_error("scalar residual violates proven normalized penalty bound");expected[i*words+d/components]|=(reference.sums[index]-base)<<((d%components)*bits);}
                if(result.packed_penalties!=expected)throw std::runtime_error("actual GPU packed penalty words differ from independent scalar oracle");packed_reference_passed=true;
            }
            integer_reference_passed=true;
            if(check_reference){auto equal=[](const auto& a,const auto& b){if(a.size()!=b.size())return false;for(size_t i=0;i<a.size();++i)if(std::bit_cast<uint32_t>(a[i])!=std::bit_cast<uint32_t>(b[i]))return false;return true;};if(!equal(result.disparity,reference.disparity)||!equal(result.confidence,reference.confidence)||!equal(result.depth,reference.depth)||result.depth_validity!=reference.depth_validity)throw std::runtime_error("GPU scalar output bit parity failed");reference_passed=true;}
            reference_ms=ms(Clock::now()-begin);
        }
        if(options.contains("--dump-prefix")){auto prefix=options.at("--dump-prefix");write(prefix+".disparity.f32",result.disparity);write(prefix+".confidence.f32",result.confidence);write(prefix+".validity.u32",result.validity);if(metric){write(prefix+".depth.f32",result.depth);write(prefix+".depth_validity.u32",result.depth_validity);}if(intermediate){write(prefix+".costs.u32",result.costs);write(prefix+".sums.u32",result.sums);if(!result.packed_penalties.empty())write(prefix+".packed_penalties.u32",result.packed_penalties);}}
    }
    auto device_report=engine->device_report();auto allocated_bytes=engine->allocated_bytes();auto required_wave=engine->required_wave_size();auto packing_bits=engine->packing_bits();
    std::array<const char*,DaxaStereoTiming::max_stages> stage_names{};for(unsigned i=0;i<gpu[0].stage_count;++i)stage_names[i]=engine->stage_name(i);
    frame={};engine.reset();context->close();
    auto errors=context->validation_errors.load(),warnings=context->validation_warnings.load();
    if(errors||warnings)throw std::runtime_error("Vulkan validation errors or warnings, including resource/device teardown");
    output<<std::setprecision(10)<<"{\n\"metric_calibration\":";if(metric)output<<"{\"focal_px\":"<<metric->fx<<",\"baseline_m\":"<<metric->baseline_m<<",\"principal_offset_px\":"<<metric->cx_offset<<'}';else output<<"null";
    output<<",\n\"schema\":\"stereoforge.daxa_stereo.v1\",\n\"scope\":\"single_reference_retained_luma_no_lr_consistency\",\n\"excluded\":\"source_upload,capture,rectification,second_reference,LR_consistency,metric_calibration_loading,inspect,diagnostic_export\",\n\"variant\":"<<std::quoted(variant)<<",\n\"reference\":"<<std::quoted(reference_name)<<",\n\"disparity_convention\":\"positive_x_left_minus_x_right\",\n\"width\":"<<config.width<<",\"height\":"<<config.height<<",\"disparities\":"<<config.disparities<<",\"paths\":"<<config.paths<<",\"p1\":"<<config.p1<<",\"p2\":"<<config.p2<<",\"subpixel\":"<<(config.subpixel?"true":"false")<<",\"metric_depth\":"<<(metric?"true":"false")<<",\n\"frames\":"<<frames<<",\"warmup\":"<<warmup<<",\"slots\":"<<slots<<",\"required_wave_size\":"<<required_wave<<",\"packing_bits\":"<<packing_bits<<",\"packed_penalty_layout\":\"pixel_major_disparity_lsb_u32\",\"packed_penalty_words_per_pixel\":"<<(packing_bits?(config.disparities+32/packing_bits-1)/(32/packing_bits):0)<<",\"packed_penalty_word_count\":"<<(packing_bits?uint64_t(config.width)*config.height*((config.disparities+32/packing_bits-1)/(32/packing_bits)):0)<<",\"packed_penalty_bound\":"<<config.paths*config.p2<<",\"full_subgroups_required\":"<<(required_wave?"true":"false")<<",\"resident_allocated_bytes\":"<<allocated_bytes<<",\n\"shader_source_sha256_pair\":"<<std::quoted(DaxaStereo::shader_identity())<<",\n\"initialization_and_upload_ms\":"<<initialization<<",\n\"gpu_graph_ms\":";stats(output,summary(total_gpu));output<<",\n\"cpu_submit_ms\":";stats(output,summary(submit));output<<",\n\"cpu_wait_ms\":";stats(output,summary(wait));output<<",\n\"cpu_timestamp_query_ms\":";stats(output,summary(query));output<<",\n\"submit_wait_and_query_wall_ms\":";stats(output,summary(roundtrip));output<<",\n\"stages\":{";
    for(unsigned stage=0;stage<gpu[0].stage_count;++stage){std::vector<double> samples(frames);for(uint32_t i=0;i<frames;++i)samples[i]=gpu[i].stage_gpu_ms[stage];if(stage)output<<',';output<<'\n'<<std::quoted(stage_names[stage])<<':';stats(output,summary(std::move(samples)));}
    output<<"\n},\n\"inspect_after_timing_ms\":"<<inspect_ms<<",\"inspect_readback_bytes\":16,\"inspect_counters\":["<<counters[0]<<','<<counters[1]<<','<<counters[2]<<','<<counters[3]<<"],\n\"diagnostic_readback_after_timing_ms\":"<<dump_ms<<",\"diagnostic_intermediates_reconstructed\":"<<(reconstructed_intermediates?"true":"false")<<",\"diagnostic_host_memory_policy\":\"prefer_HOST_VISIBLE_COHERENT_CACHED_else_explicit_COHERENT\",\n\"scalar_reference_after_timing_ms\":"<<reference_ms<<",\"scalar_reference_checked\":"<<(reference_passed?"true":"false")<<",\"compound_float_reference_checked\":"<<(reference_passed?"true":"false")<<",\"integer_reference_checked\":"<<(integer_reference_passed?"true":"false")<<",\"packed_penalties_reference_checked\":"<<(packed_reference_passed?"true":"false")<<",\n\"device\":"<<device_report<<",\n\"validation_errors_after_teardown\":"<<errors<<",\"validation_warnings_after_teardown\":"<<warnings<<"\n}\n";
    if(!output)throw std::runtime_error("profile report formatting failed");auto report=output.str();write_bytes(options.at("--output"),report.data(),report.size());
    return 0;
}catch(const std::exception& e){std::cerr<<"daxa_stereo_benchmark: "<<e.what()<<'\n';return 1;}}
