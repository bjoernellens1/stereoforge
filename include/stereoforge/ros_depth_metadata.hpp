#pragma once
#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace stereoforge {
struct RosDepthDiagnosticStamp {
    int64_t source_id;
    uint64_t capture_ns,ingress_ns,header_ns,images_published_ns,pack_ns;
};
// Serialize a setup-time JSON template without per-frame allocation/dump.
// Exact uint64 values are quoted, zero-padded 20-digit decimal strings.
class RosDepthDiagnosticMetadata {
    std::string template_;
    std::array<size_t,6> offsets_{};
    static void require(bool ok,const char* why){if(!ok)throw std::invalid_argument(why);}
public:
    static constexpr std::string_view placeholder(){return "00000000000000000000";}
    explicit RosDepthDiagnosticMetadata(std::string value):template_(std::move(value)){
        constexpr const char* names[]{"source_id","capture_timestamp_ns","host_ingress_timestamp_ns","mapped_header_stamp_ns","image_publish_calls_completed_ns","diagnostic_pack_ns"};
        for(unsigned i=0;i<offsets_.size();++i){
            const auto prefix="\""+std::string(names[i])+"\":\"";
            const auto token=prefix+std::string(placeholder())+"\"";
            const auto p=template_.find(token);
            require(p!=std::string::npos&&template_.find(token,p+1)==std::string::npos,"missing or duplicate diagnostic metadata placeholder");
            offsets_[i]=p+prefix.size();
        }
    }
    std::string_view template_bytes()const{return template_;}
    void stamp(std::span<char> output,const RosDepthDiagnosticStamp& value)const{
        require(output.size()==template_.size(),"diagnostic metadata extent mismatch");
        require(value.source_id>=0&&value.capture_ns>0&&value.ingress_ns>0&&value.images_published_ns>=value.ingress_ns,"invalid diagnostic source/timestamps");
        const std::array<uint64_t,6> values{uint64_t(value.source_id),value.capture_ns,value.ingress_ns,value.header_ns,value.images_published_ns,value.pack_ns};
        for(unsigned i=0;i<values.size();++i){
            std::array<char,20> digits{};auto result=std::to_chars(digits.data(),digits.data()+digits.size(),values[i]);
            require(result.ec==std::errc{},"diagnostic integer formatting failed");
            auto field=output.subspan(offsets_[i],20);std::fill(field.begin(),field.end(),'0');
            std::copy(digits.data(),result.ptr,field.end()-(result.ptr-digits.data()));
        }
    }
};
}
