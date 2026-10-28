#pragma once
#include <stereoforge/daxa_temporal.hpp>
#include <cmath>
#include <stdexcept>
namespace stereoforge::detail {
inline void temporal_require(bool v,char const* m){if(!v)throw std::invalid_argument(m);}
inline bool temporal_tickets_complete(uint64_t write,uint64_t read,uint64_t oldest){return detail::completed_submission(write,oldest)&&detail::completed_submission(read,oldest);}
inline long temporal_internal_references(bool active_history){return active_history?2:1;}
inline std::array<float,4> temporal_pinhole_f32(DaxaCameraModel const& camera){
    std::array<float,4> k{float(camera.fx),float(camera.fy),float(camera.cx),float(camera.cy)};
    for(auto v:k)temporal_require(std::isfinite(v),"temporal pinhole outside float32 domain");
    temporal_require(k[0]>0&&k[1]>0,"temporal float32 focal lengths must be positive");return k;
}
using TemporalMatrix=std::array<double,16>;
inline TemporalMatrix temporal_multiply(TemporalMatrix const& a,TemporalMatrix const& b){TemporalMatrix c{};for(unsigned i=0;i<4;++i)for(unsigned j=0;j<4;++j)for(unsigned k=0;k<4;++k)c[i*4+j]+=a[i*4+k]*b[k*4+j];return c;}
inline TemporalMatrix temporal_inverse(TemporalMatrix const& a){
    double det=a[0]*(a[5]*a[10]-a[6]*a[9])-a[1]*(a[4]*a[10]-a[6]*a[8])+a[2]*(a[4]*a[9]-a[5]*a[8]);
    temporal_require(std::isfinite(det)&&det!=0,"singular temporal transform");TemporalMatrix b{};b[15]=1;
    b[0]=(a[5]*a[10]-a[6]*a[9])/det;b[1]=(a[2]*a[9]-a[1]*a[10])/det;b[2]=(a[1]*a[6]-a[2]*a[5])/det;
    b[4]=(a[6]*a[8]-a[4]*a[10])/det;b[5]=(a[0]*a[10]-a[2]*a[8])/det;b[6]=(a[2]*a[4]-a[0]*a[6])/det;
    b[8]=(a[4]*a[9]-a[5]*a[8])/det;b[9]=(a[1]*a[8]-a[0]*a[9])/det;b[10]=(a[0]*a[5]-a[1]*a[4])/det;
    for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)b[i*4+3]-=b[i*4+j]*a[j*4+3];for(auto v:b)temporal_require(std::isfinite(v),"nonfinite temporal inverse");return b;
}
// Admission is transactional: prepare is const, commit follows successful submit.
struct TemporalSequence {
    std::string device,clock;uint64_t id=0,time=0;bool accepted=false,has_pose=false;TemporalMatrix previous{};
    struct Admission {bool epoch=false,warp=false;uint64_t elapsed=0;};
    TemporalSequence(){device.reserve(65536);clock.reserve(65536);}
    Admission prepare(CaptureFrame const& c,std::optional<DaxaTimedRigPose> const& pose,int64_t max_age)const;
    void commit(CaptureFrame const& c,std::optional<DaxaTimedRigPose> const& p){device.assign(c.device_identity);clock.assign(c.host_clock_domain);id=c.frame_id;time=c.host_ingress_timestamp_ns;accepted=true;has_pose=p.has_value();if(p)previous=p->world_from_rig;}
    void reset(){accepted=false;has_pose=false;device.clear();clock.clear();}
};
}
