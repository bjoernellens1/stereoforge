#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <new>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>
#include <stereoforge/ros_depth_diagnostics.hpp>
#include <stereoforge/ros_depth_metadata.hpp>
static bool track_allocations=false;static size_t allocations=0;
[[gnu::noinline]] void* operator new(size_t size){if(track_allocations)++allocations;if(void* p=std::malloc(size?size:1))return p;throw std::bad_alloc();}
[[gnu::noinline]] void operator delete(void* p)noexcept{std::free(p);}[[gnu::noinline]] void operator delete(void* p,size_t)noexcept{std::free(p);}
static void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
template<class F>static void rejects(F fn){bool rejected=false;try{fn();}catch(const std::invalid_argument&){rejected=true;}check(rejected,"expected diagnostic rejection");}
template<class T>static T at(std::span<const std::byte> bytes,size_t i){T v;std::memcpy(&v,bytes.data()+i*sizeof(T),sizeof(T));return v;}
int main(){try{
 using namespace stereoforge;
 constexpr int64_t id=INT64_MAX;constexpr auto nan=std::numeric_limits<float>::quiet_NaN();
 std::array<float,4> native_z{1.2f,3.f,2.5f,nan},published_z{1.2f,nan,2.5f,nan};
 std::array<float,4> disparity{52.75f,21.f,25.25f,nan},confidence{.8f,.5f,.2f,0};
 std::array<uint32_t,4> validity{1,1,1,0},provenance{1,5,1,0};
 std::array<int64_t,4> ids{id,id,id,-1},ages{0,0,0,0};
 auto bytes=[](const auto& v){return std::as_bytes(std::span(v));};
 RosMeasuredDepthInput input{bytes(native_z),bytes(disparity),bytes(confidence),bytes(validity),bytes(provenance),bytes(ids),bytes(ages)};
 RosMeasuredDepthDiagnostics layout(2,2,"left_rectified_optical");
 std::array<std::vector<std::byte>,7> storage;RosMeasuredDepthDiagnostics::Outputs output;
 for(unsigned p=0;p<7;++p){auto const& c=layout.contract(RosDepthDiagnosticPlane(p));storage[p].resize(size_t(c.step)*c.height);output[p]=storage[p];check(c.width==2&&c.height==2&&!c.is_bigendian&&c.frame_id=="left_rectified_optical","common native grid/frame");}
 check(layout.contract(RosDepthDiagnosticPlane::SourceIds).encoding=="32SC2"&&layout.contract(RosDepthDiagnosticPlane::SourceIds).step==16,"exact signed64 word encoding");
 check(layout.contract(RosDepthDiagnosticPlane::AgeNs).encoding=="32SC2","age word encoding");
 check(layout.contract(RosDepthDiagnosticPlane::Validity).encoding=="32SC1","binary validity encoding");
 const auto before_native=native_z;const auto before_published=published_z;
 track_allocations=true;layout.pack(id,bytes(published_z),input,output);track_allocations=false;check(allocations==0,"diagnostic packing allocated");
 auto plane=[&](RosDepthDiagnosticPlane p)->std::span<const std::byte>{return storage[unsigned(p)];};
 for(size_t i:{size_t(0),size_t(2)}){
  check(at<float>(plane(RosDepthDiagnosticPlane::Disparity),i)==disparity[i],"actual matched disparity unchanged");
  check(at<float>(plane(RosDepthDiagnosticPlane::Confidence),i)==confidence[i],"raw confidence unchanged");
  check(at<uint32_t>(plane(RosDepthDiagnosticPlane::Validity),i)==1&&at<uint32_t>(plane(RosDepthDiagnosticPlane::Provenance),i)==1,"measured masks");
  check(at<int64_t>(plane(RosDepthDiagnosticPlane::SourceIds),i)==id,"source ID high bits unchanged");
  check(at<int64_t>(plane(RosDepthDiagnosticPlane::AgeNs),i)==0,"current producer-relative age");
  check(std::isnan(at<float>(plane(RosDepthDiagnosticPlane::Uncertainty),i)),"confidence never masquerades as metric uncertainty");
 }
 for(size_t i:{size_t(1),size_t(3)}){
  check(at<uint32_t>(plane(RosDepthDiagnosticPlane::Validity),i)==0&&at<uint32_t>(plane(RosDepthDiagnosticPlane::Provenance),i)==0,"speckle rejection clears masks");
  check(at<float>(plane(RosDepthDiagnosticPlane::Confidence),i)==0&&at<int64_t>(plane(RosDepthDiagnosticPlane::SourceIds),i)==-1&&at<int64_t>(plane(RosDepthDiagnosticPlane::AgeNs),i)==0,"rejected identity/age/confidence clear");
  check(std::isnan(at<float>(plane(RosDepthDiagnosticPlane::Disparity),i))&&std::isnan(at<float>(plane(RosDepthDiagnosticPlane::Uncertainty),i)),"rejected floats unknown");
 }
 check(std::memcmp(before_native.data(),native_z.data(),sizeof(native_z))==0&&std::memcmp(before_published.data(),published_z.data(),sizeof(published_z))==0,"input planes immutable");
 provenance[2]=5;layout.pack(id,bytes(published_z),input,output);check(at<uint32_t>(plane(RosDepthDiagnosticPlane::Provenance),2)==5,"correlated stereo provenance retained");provenance[2]=1;layout.pack(id,bytes(published_z),input,output);
 // Native metric validity permits d=0 with negative principal offset.
 disparity[2]=0.f;layout.pack(id,bytes(published_z),input,output);
 check(at<float>(plane(RosDepthDiagnosticPlane::Disparity),2)==0.f,"valid native zero disparity preserved");
 disparity[2]=25.25f;layout.pack(id,bytes(published_z),input,output);
 const auto stable=storage;
 auto reject_unchanged=[&]{rejects([&]{layout.pack(id,bytes(published_z),input,output);});check(storage==stable,"rejection cannot partly replace a packet");};
 ids[2]=id-1;reject_unchanged();ids[2]=id;
 ages[2]=1;reject_unchanged();ages[2]=0;
 provenance[2]=6;reject_unchanged();provenance[2]=1;
 validity[2]=2;reject_unchanged();validity[2]=1;
 disparity[2]=nan;reject_unchanged();disparity[2]=25.25f;
 confidence[2]=1.1f;reject_unchanged();confidence[2]=.2f;
 published_z[2]=2.6f;reject_unchanged();published_z[2]=2.5f;
 published_z[3]=2.f;reject_unchanged();published_z[3]=nan;
 auto alias=output;alias[1]=alias[4];rejects([&]{layout.pack(id,bytes(published_z),input,alias);});
 auto short_output=output;short_output[0]=short_output[0].first(15);rejects([&]{layout.pack(id,bytes(published_z),input,short_output);});
 rejects([&]{layout.pack(-1,bytes(published_z),input,output);});
 rejects([&]{layout.contract(RosDepthDiagnosticPlane(99));});
 rejects([&]{RosMeasuredDepthDiagnostics bad(0,2,"left_rectified_optical");});
 rejects([&]{RosMeasuredDepthDiagnostics bad(2,2,"unknown");});

 const std::string placeholder(20,'0');
 std::string text="{";
 for(auto name:{"source_id","capture_timestamp_ns","host_ingress_timestamp_ns","mapped_header_stamp_ns","image_publish_calls_completed_ns","diagnostic_pack_ns"}){if(text.size()>1)text+=",";text+="\""+std::string(name)+"\":\""+placeholder+"\"";}text+="}";
 RosDepthDiagnosticMetadata metadata(text);std::string payload(metadata.template_bytes());
 track_allocations=true;metadata.stamp(std::span<char>(payload),{INT64_MAX,UINT64_MAX,2,3,4,5});track_allocations=false;check(allocations==0,"metadata formatting allocated");
 check(payload.find("\"source_id\":\"09223372036854775807\"")!=std::string::npos,"metadata ID precision");
 check(payload.find("18446744073709551615")!=std::string::npos,"metadata timestamp precision");
 const auto stable_payload=payload;
 rejects([&]{metadata.stamp(std::span<char>(payload),{-1,1,2,3,4,5});});check(payload==stable_payload,"metadata rejected input immutable");
 rejects([&]{metadata.stamp(std::span<char>(payload).first(payload.size()-1),{1,1,2,3,4,5});});
 metadata.stamp(std::span<char>(payload),{1,1,2,3,4,5});check(payload.find("\"source_id\":\"00000000000000000001\"")!=std::string::npos,"metadata resets previous digits");
 rejects([&]{RosDepthDiagnosticMetadata bad("{}");});
 std::cout<<"ROS measured diagnostic contracts passed\n";return 0;
}catch(const std::exception& e){track_allocations=false;std::cerr<<e.what()<<'\n';return 1;}}
