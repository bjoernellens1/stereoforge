#include <stereoforge/canonical_record.hpp>
#include <json.hpp>
#include <openssl/evp.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace stereoforge {
namespace {
using Json=nlohmann::json;
constexpr uint64_t json_limit=16*1024*1024,plane_limit=512*1024*1024;
void require(bool v,const char* message){if(!v)throw std::invalid_argument(message);}
void keys(const Json& j,std::initializer_list<const char*> names){require(j.is_object()&&j.size()==names.size(),"unexpected JSON object keys");for(auto name:names)require(j.contains(name),"missing JSON object key");}
std::string text(const Json& j){require(j.is_string(),"expected string");auto s=j.get<std::string>();require(!s.empty(),"empty string");return s;}
uint64_t integer(const Json& j,uint64_t max=UINT64_MAX){require(j.is_number_unsigned()||(j.is_number_integer()&&j.get<int64_t>()>=0),"expected nonnegative integer");auto n=j.get<uint64_t>();require(n<=max,"integer outside range");return n;}
double number(const Json& j){require(j.is_number()&&!j.is_boolean(),"expected finite number");double n=j.get<double>();require(std::isfinite(n),"expected finite number");return n;}
std::string digest_text(const Json& j){auto s=text(j);require(s.size()==64&&s.find_first_not_of("0123456789abcdef")==std::string::npos,"invalid SHA256");return s;}
struct Hash {
    EVP_MD_CTX* context=EVP_MD_CTX_new();
    Hash(){if(!context||EVP_DigestInit_ex(context,EVP_sha256(),nullptr)!=1){EVP_MD_CTX_free(context);context=nullptr;throw std::runtime_error("SHA256 initialization failed");}}
    ~Hash(){EVP_MD_CTX_free(context);}
    void add(const void* p,size_t n){if(EVP_DigestUpdate(context,p,n)!=1)throw std::runtime_error("SHA256 update failed");}
    std::string finish(){unsigned char bytes[EVP_MAX_MD_SIZE];unsigned size=0;if(EVP_DigestFinal_ex(context,bytes,&size)!=1||size!=32)throw std::runtime_error("SHA256 failed");std::string s;for(unsigned i=0;i<size;++i){s+="0123456789abcdef"[bytes[i]>>4];s+="0123456789abcdef"[bytes[i]&15];}return s;}
};
std::string digest(std::string const& s){Hash h;h.add(s.data(),s.size());return h.finish();}
// Check lexical components before following any path. Absolute manifest paths
// are permitted, but none of their components may be symlinks either.
void safe_absolute(const std::filesystem::path& path){
    require(path.native().find('\0')==std::string::npos,"NUL in input path");auto current=path.root_path();
    for(auto const& part:path.relative_path()){require(part!="."&&part!=".."&&!part.empty(),"unsafe path component");current/=part;struct stat s{};require(lstat(current.c_str(),&s)==0&&!S_ISLNK(s.st_mode),"missing or symlink file component");}
}
std::filesystem::path relative_file(const std::filesystem::path& base,const Json& value){auto s=text(value);require(s.front()!='/'&&s.find('\\')==std::string::npos&&s.find('\0')==std::string::npos,"unsafe relative file path");size_t start=0;for(;;){auto end=s.find('/',start);auto part=s.substr(start,end==std::string::npos?end:end-start);require(!part.empty()&&part!="."&&part!="..","unsafe relative file path");if(end==std::string::npos)break;start=end+1;}auto p=base/s;safe_absolute(p);return p;}
bool same(struct stat const&a,struct stat const&b){return a.st_dev==b.st_dev&&a.st_ino==b.st_ino&&a.st_mode==b.st_mode&&a.st_size==b.st_size&&a.st_mtim.tv_sec==b.st_mtim.tv_sec&&a.st_mtim.tv_nsec==b.st_mtim.tv_nsec&&a.st_ctim.tv_sec==b.st_ctim.tv_sec&&a.st_ctim.tv_nsec==b.st_ctim.tv_nsec;}
struct Read {std::vector<uint8_t> bytes;std::string hash;uint64_t size;};
Read read(const std::filesystem::path& path,uint64_t limit,bool retain=true){
    safe_absolute(path);struct stat before{},opened{},after{},named{};require(lstat(path.c_str(),&before)==0&&S_ISREG(before.st_mode)&&before.st_size>=0,"regular input file required");
    require(uint64_t(before.st_size)<=limit,"input file exceeds bound");
    int fd=::open(path.c_str(),O_RDONLY|O_NOFOLLOW|O_CLOEXEC);require(fd>=0,"cannot open input file");
    struct Close{int fd;~Close(){::close(fd);}} close{fd};
    require(fstat(fd,&opened)==0&&same(before,opened),"input changed while opening");Read result{{},{},uint64_t(before.st_size)};if(retain)result.bytes.resize(result.size);
    Hash hash;std::array<uint8_t,65536> chunk{};uint64_t offset=0;
    for(;;){auto count=::read(fd,chunk.data(),chunk.size());require(count>=0,"input read failed");if(!count)break;require(uint64_t(count)<=result.size-offset,"input changed while reading");hash.add(chunk.data(),size_t(count));if(retain)std::copy_n(chunk.data(),count,result.bytes.data()+offset);offset+=count;}
    require(offset==result.size&&fstat(fd,&after)==0&&lstat(path.c_str(),&named)==0&&same(before,after)&&same(before,named),"input changed while reading");safe_absolute(path);result.hash=hash.finish();return result;
}
Json parse(Read const& file){
    require(file.size<=json_limit,"JSON exceeds size bound");
    unsigned depth=0;bool string=false,escape=false;
    for(uint8_t c:file.bytes){if(string){if(escape)escape=false;else if(c=='\\')escape=true;else if(c=='"')string=false;}else if(c=='"')string=true;else if(c=='{'||c=='['){require(++depth<=64,"JSON nesting exceeds bound");}else if(c=='}'||c==']'){require(depth>0,"invalid JSON nesting");--depth;}}
    std::vector<std::set<std::string>> objects;
    auto callback=[&](int,Json::parse_event_t event,Json& value){if(event==Json::parse_event_t::object_start)objects.emplace_back();else if(event==Json::parse_event_t::object_end)objects.pop_back();else if(event==Json::parse_event_t::key)require(objects.back().insert(value.get<std::string>()).second,"duplicate JSON key");return true;};
    return Json::parse(file.bytes.begin(),file.bytes.end(),callback);
}
struct Loader {
    std::filesystem::path base;std::vector<CanonicalRecordFile> files;
    Read descriptor(const Json& desc,uint64_t limit,bool retain=true,bool plane=false){
        if(plane)keys(desc,{"path","sha256","size_bytes","width","height","stride_bytes","format"});else keys(desc,{"path","sha256","size_bytes"});
        auto size=integer(desc.at("size_bytes"));require(size<=limit,"descriptor exceeds bound");auto hash=digest_text(desc.at("sha256"));auto path=relative_file(base,desc.at("path"));auto data=read(path,limit,retain);require(data.size==size&&data.hash==hash,"file descriptor hash/size mismatch");
        auto it=std::find_if(files.begin(),files.end(),[&](auto const& f){return f.path==path;});if(it!=files.end())require(it->sha256==hash&&it->size_bytes==size,"conflicting file descriptors");else files.push_back({path,hash,size});return data;
    }
    Json json(const Json& desc){return parse(descriptor(desc,json_limit));}
};
template<size_t N>std::array<double,N> matrix(const Json& j,unsigned rows,unsigned columns){require(j.is_array()&&j.size()==rows,"matrix row shape invalid");std::array<double,N> result{};for(unsigned r=0;r<rows;++r){require(j[r].is_array()&&j[r].size()==columns,"matrix column shape invalid");for(unsigned c=0;c<columns;++c)result[r*columns+c]=number(j[r][c]);}return result;}
DaxaStereoCalibration calibration(const Json& j,const char* space){
    keys(j,{"schema","space","left","right","right_to_left","units","pixel_centres","coordinate_axes"});require(j.at("schema")=="stereoforge.zed_comparison.calibration.v1"&&j.at("space")==space&&j.at("units")=="m"&&j.at("pixel_centres")=="integer_centres"&&j.at("coordinate_axes")=="x_right_y_down_z_forward","unsupported normalized calibration");
    DaxaStereoCalibration result;unsigned i=0;for(auto name:{"left","right"}){auto const& c=j.at(name);keys(c,{"width","height","fx","fy","cx","cy","projection","distortion"});auto& eye=result.eyes[i++];eye.width=integer(c.at("width"),32766);eye.height=integer(c.at("height"),32766);require(eye.width&&eye.height&&c.at("projection")=="perspective","invalid camera dimensions/projection");eye.fx=number(c.at("fx"));eye.fy=number(c.at("fy"));eye.cx=number(c.at("cx"));eye.cy=number(c.at("cy"));auto const& distortion=c.at("distortion");keys(distortion,{"model","coefficients"});auto model=text(distortion.at("model"));auto const& coeff=distortion.at("coefficients");require(coeff.is_array(),"distortion array required");auto n=coeff.size();require((model=="none"&&n==0)||(model=="brown_conrady"&&(n==4||n==5))||(model=="rational_polynomial"&&n==8),"unsupported distortion model");if(std::string(space)=="rectified")require(model=="none","rectified distortion must be none");for(auto const& v:coeff)eye.distortion.push_back(number(v));}
    result.right_to_left=matrix<16>(j.at("right_to_left"),4,4);return result;
}
void equal_calibration(const DaxaStereoCalibration& a,const DaxaStereoCalibration& b){require(a.right_to_left==b.right_to_left,"coarse pose mismatch");for(unsigned i=0;i<2;++i){auto const& x=a.eyes[i];auto const& y=b.eyes[i];require(x.width==y.width&&x.height==y.height&&x.fx==y.fx&&x.fy==y.fy&&x.cx==y.cx&&x.cy==y.cy&&x.distortion==y.distortion,"coarse calibration derivation mismatch");}}
std::pair<uint64_t,std::string> timestamp(const Json& j){keys(j,{"value_ns","clock_domain"});auto n=integer(j.at("value_ns"));auto domain=text(j.at("clock_domain"));return {n,domain};}
constexpr std::array<const char*,6> image_names{"original_left","original_right","rectified_left","rectified_right","support_left","support_right"};
std::string pixel_identity(const Json& images,unsigned begin,unsigned end){Json payload=Json::object();for(unsigned i=begin;i<end;++i){payload[image_names[i]]=images.at(image_names[i]);payload[image_names[i]].erase("path");}return digest(payload.dump());}
}
CanonicalRecordInput load_canonical_record(const std::filesystem::path& manifest){
    CanonicalRecordInput result;auto absolute=std::filesystem::absolute(manifest);safe_absolute(absolute);Loader loader{absolute.parent_path(),{}};
    auto manifest_file=read(absolute,json_limit);loader.files.push_back({absolute,manifest_file.hash,manifest_file.size});result.manifest_sha256=manifest_file.hash;auto m=parse(manifest_file);
    keys(m,{"schema","source_record","geometry_supplement","coarse_calibration","sampling_profile","original_pixel_identity","rectified_pixel_identity"});require(m.at("schema")=="stereoforge.canonical_record_input.v1"&&m.at("sampling_profile")==canonical_rgb_sampling_profile,"unsupported canonical record manifest");
    auto record=loader.json(m.at("source_record"));result.record_sha256=digest_text(m.at("source_record").at("sha256"));
    keys(record,{"schema","scene_id","sequence_id","partition","ordinal","source_id","source","calibration","timestamps","images","observation","implementation","input_identity"});text(record.at("scene_id"));text(record.at("sequence_id"));require(record.at("partition")=="development"||record.at("partition")=="held_out","invalid portable partition");require(record.at("schema")=="stereoforge.zed_comparison.frame.v1","unsupported portable record");result.ordinal=integer(record.at("ordinal"));auto id=integer(record.at("source_id"),INT64_MAX);auto identity=digest_text(record.at("input_identity"));
    auto const& source=record.at("source");keys(source,{"kind","artifact","codec","lossless"});require(source.at("kind")=="analytic","only analytic records admitted");text(source.at("codec"));require(source.at("lossless").is_null()||source.at("lossless").is_boolean(),"invalid lossless flag");loader.descriptor(source.at("artifact"),UINT64_MAX,false);
    auto const& cal=record.at("calibration");keys(cal,{"raw","rectified","resolved_file","geometry"});auto raw_json=loader.json(cal.at("raw"));auto rect_json=loader.json(cal.at("rectified"));auto raw_hash=digest_text(cal.at("raw").at("sha256")),rect_hash=digest_text(cal.at("rectified").at("sha256"));
    auto const& resolved=cal.at("resolved_file");if(resolved.at("status")=="available"){keys(resolved,{"status","value"});loader.descriptor(resolved.at("value"),UINT64_MAX,false);}else{keys(resolved,{"status","reason"});require(resolved.at("status")=="unavailable","invalid resolved calibration availability");text(resolved.at("reason"));}
    auto supplement=loader.json(m.at("geometry_supplement"));result.geometry_supplement_sha256=digest_text(m.at("geometry_supplement").at("sha256"));keys(supplement,{"schema","input_identity","record_sha256","raw_calibration_sha256","rectified_calibration_sha256","R1","R2","P1","P2","support_qualification","producer"});
    require(supplement.at("schema")=="stereoforge.canonical_geometry.v1"&&supplement.at("support_qualification")=="analytic_known_geometry"&&digest_text(supplement.at("input_identity"))==identity&&digest_text(supplement.at("record_sha256"))==result.record_sha256&&digest_text(supplement.at("raw_calibration_sha256"))==raw_hash&&digest_text(supplement.at("rectified_calibration_sha256"))==rect_hash,"supplement binding/qualification mismatch");result.producer=text(supplement.at("producer"));
    auto& g=result.full_geometry;g.original=calibration(raw_json,"raw");g.rectified=calibration(rect_json,"rectified");g.original_identity=raw_hash;g.rectified_identity=rect_hash;g.R1=matrix<9>(supplement.at("R1"),3,3);g.R2=matrix<9>(supplement.at("R2"),3,3);g.P1=matrix<12>(supplement.at("P1"),3,4);g.P2=matrix<12>(supplement.at("P2"),3,4);g.validate();g.metric_calibration();
    auto const& geometry=cal.at("geometry");keys(geometry,{"width","height","fx","fy","baseline_m","cx_left","cx_right","projection","depth_convention","units","coordinate_frame","pixel_centres","reference_camera","coordinate_axes"});
    require(geometry.at("projection")=="perspective"&&geometry.at("depth_convention")=="axial_z"&&geometry.at("units")=="m"&&geometry.at("pixel_centres")=="integer_centres"&&geometry.at("reference_camera")=="left"&&geometry.at("coordinate_axes")=="x_right_y_down_z_forward","unsupported record geometry");require(geometry.at("coordinate_frame")=="left_optical","record geometry must use left optical frame");
    auto const& left=g.rectified.eyes[0];auto const& right=g.rectified.eyes[1];require(integer(geometry.at("width"))==left.width&&integer(geometry.at("height"))==left.height&&number(geometry.at("fx"))==left.fx&&number(geometry.at("fy"))==left.fy&&number(geometry.at("cx_left"))==left.cx&&number(geometry.at("cx_right"))==right.cx&&right.fx==left.fx&&right.fy==left.fy&&right.cy==left.cy,"record/calibration geometry mismatch");double baseline=number(geometry.at("baseline_m"));require(baseline>0&&std::abs(g.rectified.right_to_left[3]-baseline)<=1e-9+1e-6*std::abs(baseline),"record/calibration baseline mismatch");double raw_baseline=std::hypot(g.original.right_to_left[3],g.original.right_to_left[7],g.original.right_to_left[11]);require(std::abs(raw_baseline-baseline)<=1e-9+1e-6*std::abs(baseline),"raw baseline disagrees with record");
    require(left.width%2==0&&left.height%2==0,"even rectified dimensions required");
    auto coarse_json=loader.json(m.at("coarse_calibration"));result.coarse_calibration_sha256=digest_text(m.at("coarse_calibration").at("sha256"));auto coarse=calibration(coarse_json,"rectified");result.coarse_geometry=g;auto& cg=result.coarse_geometry;cg.rectified_identity=result.coarse_calibration_sha256;
    for(auto& c:cg.rectified.eyes){c.width/=2;c.height/=2;c.fx/=2;c.fy/=2;c.cx=(c.cx+.5)/2-.5;c.cy=(c.cy+.5)/2-.5;}
    for(auto* p:{&cg.P1,&cg.P2}){for(unsigned i=0;i<8;++i)if(i!=2&&i!=6)(*p)[i]/=2;(*p)[2]=((*p)[2]+.5)/2-.5;(*p)[6]=((*p)[6]+.5)/2-.5;}
    equal_calibration(cg.rectified,coarse);cg.validate();cg.metric_calibration();
    auto const& times=record.at("timestamps");keys(times,{"capture","ingress","sensors"});auto capture=timestamp(times.at("capture"));require(capture.first&&capture.second!="unknown","capture timestamp/domain unavailable");auto ingress=std::pair<uint64_t,std::string>{0,"unknown"};auto const& in=times.at("ingress");if(in.at("status")=="available"){keys(in,{"status","value"});ingress=timestamp(in.at("value"));}else{keys(in,{"status","reason"});require(in.at("status")=="unavailable","invalid ingress status");text(in.at("reason"));}
    auto const& images=record.at("images");keys(images,{"original_left","original_right","rectified_left","rectified_right","support_left","support_right"});
    using Backing=std::array<std::vector<uint8_t>,6>;auto backing=std::make_shared<Backing>();auto sidecar=std::make_shared<CanonicalRgbSource>();uint64_t extent_sum=0;
    // Validate the sum before allocating any plane storage.
    for(unsigned i=0;i<6;++i){auto const& d=images.at(image_names[i]);keys(d,{"path","sha256","size_bytes","width","height","stride_bytes","format"});auto width=integer(d.at("width"),32766),height=integer(d.at("height"),32766),stride=integer(d.at("stride_bytes"),UINT32_MAX),size=integer(d.at("size_bytes"));require(width&&height&&stride>=width*(i<4?3:1)&&size==stride*height&&size<=plane_limit-extent_sum,"invalid plane shape/extent");extent_sum+=size;require(d.at("format")== (i<4?"rgb8":"u8_mask"),"invalid canonical plane format");auto const& camera=i<2?g.original.eyes[i]:g.rectified.eyes[(i-2)%2];require(width==camera.width&&height==camera.height,"plane/calibration dimensions mismatch");}
    for(unsigned i=0;i<6;++i){auto const& d=images.at(image_names[i]);(*backing)[i]=std::move(loader.descriptor(d,plane_limit,true,true).bytes);CanonicalRgbPlane p{(*backing)[i],uint32_t(integer(d.at("width"))),uint32_t(integer(d.at("height"))),uint32_t(integer(d.at("stride_bytes"))),i<4?CanonicalRgbFormat::Rgb8:CanonicalRgbFormat::SupportU8};if(i<4)sidecar->rgb[i]=p;else sidecar->support[i-4]=p;}
    sidecar->backing=backing;sidecar->input_identity=identity;sidecar->original_pixel_identity=digest_text(m.at("original_pixel_identity"));sidecar->rectified_pixel_identity=digest_text(m.at("rectified_pixel_identity"));require(sidecar->original_pixel_identity==pixel_identity(images,0,2)&&sidecar->rectified_pixel_identity==pixel_identity(images,2,6),"pixel identity mismatch");sidecar->original_calibration_identity=raw_hash;sidecar->rectified_calibration_identity=rect_hash;sidecar->coarse_rectified_calibration_identity=result.coarse_calibration_sha256;
    auto frame=std::make_shared<CaptureFrame>();frame->source_kind=CaptureSourceKind::CanonicalRgbMetadata;frame->canonical_rgb=sidecar;frame->source_width=2*g.original.eyes[0].width;frame->height=g.original.eyes[0].height;frame->frame_id=id;frame->camera_timestamp_ns=capture.first;frame->camera_clock_domain=capture.second;frame->host_ingress_timestamp_ns=ingress.first;frame->host_clock_domain=ingress.second;frame->calibration_identity=raw_hash;
    result.frame=RectifiedRgbFrame(frame);result.files=std::move(loader.files);verify_canonical_record_files(result);return result;
}
void verify_canonical_record_files(const CanonicalRecordInput& input){for(auto const& file:input.files){auto value=read(file.path,file.size_bytes,false);require(value.size==file.size_bytes&&value.hash==file.sha256,"canonical record file changed");}}
}
