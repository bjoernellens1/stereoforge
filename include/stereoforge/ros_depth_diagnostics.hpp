#pragma once
#include <stereoforge/ros_depth_contract.hpp>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

namespace stereoforge {
enum class RosDepthDiagnosticPlane:unsigned { Disparity,Validity,Confidence,Uncertainty,Provenance,SourceIds,AgeNs };
struct RosMeasuredDepthInput {
    std::span<const std::byte> depth,disparity,confidence,validity,provenance,source_ids,age_ns;
};
// ROS serialization after requested readback; this never estimates geometry.
// Only current-frame measured stereo is admitted. Completion is a separate layer.
class RosMeasuredDepthDiagnostics {
    std::array<RosDepthImageContract,7> contracts_;
    size_t pixels_;
    template<class T>static T load(std::span<const std::byte> bytes,size_t i){
        T value;std::memcpy(&value,bytes.data()+i*sizeof(T),sizeof(T));return value;
    }
    template<class T>static void store(std::span<std::byte> bytes,size_t i,T value){
        std::memcpy(bytes.data()+i*sizeof(T),&value,sizeof(T));
    }
    static bool overlaps(std::span<const std::byte> a,std::span<const std::byte> b){
        if(a.empty()||b.empty())return false;
        const auto x=reinterpret_cast<uintptr_t>(a.data()),y=reinterpret_cast<uintptr_t>(b.data());
        return x<=y?y-x<a.size():x-y<b.size();
    }
public:
    using Outputs=std::array<std::span<std::byte>,7>;
    RosMeasuredDepthDiagnostics(uint32_t width,uint32_t height,std::string_view frame):pixels_(size_t(width)*height){
        using ros_contract_detail::require;
        require(std::endian::native==std::endian::little,"ROS diagnostic words require little endian");
        constexpr const char* encodings[]{"32FC1","32SC1","32FC1","32FC1","32SC1","32SC2","32SC2"};
        for(unsigned i=0;i<contracts_.size();++i){
            auto c=make_ros_depth_contract(width,height,frame);c.encoding=encodings[i];
            if(i>=5)c.step*=2;
            require(uint64_t(c.step)*height<=512ULL*1024*1024,"diagnostic payload extent overflow");
            contracts_[i]=std::move(c);
        }
    }
    const RosDepthImageContract& contract(RosDepthDiagnosticPlane plane)const{
        ros_contract_detail::require(unsigned(plane)<contracts_.size(),"unknown diagnostic plane");return contracts_[unsigned(plane)];
    }
    // Output storage is owned/preallocated by the ROS messages. Reject all
    // inconsistent inputs before writing any byte, including filter-created fills.
    // 32SC2 carries exact signed64 low/high words, never a floating-point ID/age.
    void pack(int64_t current_source_id,std::span<const std::byte> published_depth,
              const RosMeasuredDepthInput& in,const Outputs& out)const{
        using ros_contract_detail::require;
        require(current_source_id>=0,"invalid current source ID");
        require(published_depth.size()==pixels_*4,"published depth extent mismatch");
        const std::array<std::span<const std::byte>,7> inputs{in.depth,in.disparity,in.confidence,in.validity,in.provenance,in.source_ids,in.age_ns};
        for(unsigned i=0;i<inputs.size();++i)require(inputs[i].size()==pixels_*(i>=5?8:4),"native diagnostic extent mismatch");
        for(unsigned i=0;i<out.size();++i){
            require(out[i].size()==size_t(contracts_[i].step)*contracts_[i].height,"ROS diagnostic extent mismatch");
            require(!overlaps(out[i],published_depth),"diagnostic output aliases published depth");
            for(auto input:inputs)require(!overlaps(out[i],input),"diagnostic output aliases native input");
            for(unsigned j=0;j<i;++j)require(!overlaps(out[i],out[j]),"diagnostic outputs overlap");
        }
        for(size_t i=0;i<pixels_;++i){
            const auto valid=load<uint32_t>(in.validity,i);require(valid<=1,"nonbinary native validity");
            const auto z=load<float>(published_depth,i);
            if(!std::isfinite(z)||z<=0)continue;
            require(valid==1,"published measured depth has no native measurement");
            require(load<uint32_t>(published_depth,i)==load<uint32_t>(in.depth,i),"published measured geometry was modified");
            const auto d=load<float>(in.disparity,i),confidence=load<float>(in.confidence,i);
            require(std::isfinite(d)&&std::isfinite(confidence)&&confidence>=0&&confidence<=1,"invalid measured disparity/confidence");
            const auto provenance=load<uint32_t>(in.provenance,i);
            require(provenance==1||provenance==5,"inferred or historical geometry in measured layer");
            require(load<int64_t>(in.source_ids,i)==current_source_id&&load<int64_t>(in.age_ns,i)==0,"measured source identity/relative age mismatch");
        }
        constexpr auto nan=std::numeric_limits<float>::quiet_NaN();
        for(size_t i=0;i<pixels_;++i){
            const auto z=load<float>(published_depth,i);const bool valid=std::isfinite(z)&&z>0;
            store<float>(out[0],i,valid?load<float>(in.disparity,i):nan);
            store<uint32_t>(out[1],i,valid?1u:0u);
            store<float>(out[2],i,valid?load<float>(in.confidence,i):0.f);
            store<float>(out[3],i,nan); // No calibrated metric-uncertainty model is available.
            store<uint32_t>(out[4],i,valid?load<uint32_t>(in.provenance,i):0u);
            store<int64_t>(out[5],i,valid?load<int64_t>(in.source_ids,i):int64_t(-1));
            store<int64_t>(out[6],i,0);
        }
    }
};
}
