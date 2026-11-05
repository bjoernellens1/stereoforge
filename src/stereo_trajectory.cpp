#include <stereoforge/stereo_trajectory.hpp>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace stereoforge {
namespace {
using Pose=std::array<double,16>;
uint64_t next_trajectory_id() {
    static std::atomic<uint64_t> next{1};auto value=next.load(std::memory_order_relaxed);
    for(;;) {
        if(value==std::numeric_limits<uint64_t>::max())throw std::overflow_error("trajectory identity exhausted");
        if(next.compare_exchange_weak(value,value+1,std::memory_order_relaxed))return value;
    }
}
constexpr Pose identity{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
void require(bool ok,char const* message){if(!ok)throw std::invalid_argument(message);}
bool known(std::string const& s){return !s.empty()&&s!="unknown"&&s!="unspecified"&&s.size()<=65536;}
bool same_domain(StereoFrameIdentity const& a,StereoFrameIdentity const& b) {
    return a.device==b.device&&a.original_calibration==b.original_calibration&&a.rectified_calibration==b.rectified_calibration&&
           a.rig_session==b.rig_session&&a.acquisition_clock==b.acquisition_clock&&a.ingress_clock==b.ingress_clock;
}
bool positive_covariance(std::array<double,36> const& c) {
    std::array<double,36> l{};
    for(unsigned i=0;i<6;++i)for(unsigned j=0;j<=i;++j) {
        double a=c[i*6+j],b=c[j*6+i];
        if(!std::isfinite(a)||!std::isfinite(b)||std::abs(a-b)>1e-9*std::max({std::abs(a),std::abs(b),1e-18}))return false;
        for(unsigned k=0;k<j;++k)a-=l[i*6+k]*l[j*6+k];
        if(i==j) {if(a<=0||!std::isfinite(a))return false;l[i*6+j]=std::sqrt(a);}
        else l[i*6+j]=a/l[j*6+j];
    }
    return true;
}
bool quality(StereoVoFrame const& f,StereoTrajectoryConfig const& cfg) {
    auto const& m=f.motion;
    if(f.guard.status!=StereoMotionGuardStatus::NotEvaluated&&f.guard.status!=StereoMotionGuardStatus::Accepted)return false;
    if(m.status!=StereoMotionStatus::Tracked||f.tracked_features>cfg.max_tracks||f.tracked_features<cfg.min_inliers||
       m.inliers.size()<cfg.min_inliers||m.inliers.size()>f.tracked_features||
       double(m.inliers.size())/double(f.tracked_features)<cfg.min_inlier_fraction||
       !std::isfinite(m.rms_m)||m.rms_m<0||m.rms_m>cfg.max_rms_m||!stereo_rigid_transform(m.current_from_previous)||!positive_covariance(m.covariance))return false;
    std::vector<bool> seen(f.tracked_features);
    for(auto id:m.inliers) {if(id>=seen.size()||seen[id])return false;seen[id]=true;}
    auto const& t=m.current_from_previous;
    double angle=std::acos(std::clamp((t[0]+t[5]+t[10]-1)*.5,-1.,1.));
    return std::hypot(t[3],t[7],t[11])<=cfg.max_translation_m&&angle<=cfg.max_rotation_rad;
}
Pose compose_world(Pose const& world,Pose const& relative) {
    Pose inverse=identity,out{};
    for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)inverse[i*4+j]=relative[j*4+i];
    for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)inverse[i*4+3]-=inverse[i*4+j]*relative[j*4+3];
    for(unsigned i=0;i<4;++i)for(unsigned j=0;j<4;++j)for(unsigned k=0;k<4;++k)out[i*4+j]+=world[i*4+k]*inverse[k*4+j];
    return out;
}
}
std::optional<uint64_t> stereo_acquisition_age_bound(StereoFrameIdentity const& old,StereoFrameIdentity const& current) {
    if(!old.clock_mapping_uncertainty_ns||!current.clock_mapping_uncertainty_ns||!known(old.acquisition_clock)||old.acquisition_clock!=current.acquisition_clock||current.acquisition_timestamp_ns<=old.acquisition_timestamp_ns)return {};
    uint64_t delta=current.acquisition_timestamp_ns-old.acquisition_timestamp_ns,max=std::numeric_limits<uint64_t>::max();
    if(*old.clock_mapping_uncertainty_ns>max-delta)return {};
    delta+=*old.clock_mapping_uncertainty_ns;
    if(*current.clock_mapping_uncertainty_ns>max-delta)return {};
    return delta+*current.clock_mapping_uncertainty_ns;
}
bool stereo_pose_usable(StereoWorldPose const& pose,bool research) {
    if(!pose.world_from_rig||pose.trajectory_id==0||pose.epoch==0||
       (pose.status!=StereoTrajectoryStatus::Anchored&&pose.status!=StereoTrajectoryStatus::Tracked)||
       (!research&&!(pose.temporal_qualified&&pose.covariance_calibrated&&pose.source.clock_mapping_uncertainty_ns))||!stereo_rigid_transform(*pose.world_from_rig))return false;
    try {pose.source.validate();}catch(std::invalid_argument const&){return false;}
    return true;
}
void StereoFrameIdentity::validate() const {
    require(known(device)&&known(original_calibration)&&known(rectified_calibration)&&known(ingress_clock),"trajectory requires known bounded device/calibration/ingress clock");
    require(acquisition_clock.size()<=65536&&rig_session.size()<=65536,"unbounded acquisition clock or rig session");
    require(!clock_mapping_uncertainty_ns||known(acquisition_clock),"unknown acquisition clock cannot have qualified uncertainty");
}
void StereoTrajectoryConfig::validate() const {
    require(max_ingress_gap_ns>0&&max_ingress_gap_ns<=200000000&&min_inliers>=3&&max_tracks>=min_inliers&&max_tracks<=4096,"invalid trajectory bounds");
    require(std::isfinite(min_inlier_fraction)&&min_inlier_fraction>.5&&min_inlier_fraction<=1&&
            std::isfinite(max_rms_m)&&max_rms_m>0&&max_rms_m<=1&&std::isfinite(max_translation_m)&&max_translation_m>0&&max_translation_m<=10&&
            std::isfinite(max_rotation_rad)&&max_rotation_rad>0&&max_rotation_rad<=3.141592653589793,"invalid trajectory quality limits");
}
StereoTrajectory::StereoTrajectory(StereoTrajectoryConfig config):config_(config),world_(identity),trajectory_id_(next_trajectory_id()){config_.validate();}
void StereoTrajectory::reset(){last_.reset();world_=identity;}
StereoWorldPose StereoTrajectory::process(StereoVoFrame const& frame) {
    frame.source.validate();if(frame.previous_source)frame.previous_source->validate();
    bool same=last_&&same_domain(*last_,frame.source);
    if(same)require(frame.source.frame_id>last_->frame_id&&frame.source.ingress_timestamp_ns>last_->ingress_timestamp_ns,"duplicate/late trajectory source");
    StereoWorldPose out;out.source=frame.source;out.trajectory_id=trajectory_id_;
    auto reanchor=[&](StereoTrajectoryStatus status,bool publish_anchor) {
        require(epoch_!=std::numeric_limits<uint64_t>::max(),"trajectory epoch overflow");
        // Copy the owned source before committing any state that depends on it.
        auto next=frame.source;last_=std::move(next);world_=identity;++epoch_;
        out.epoch=epoch_;out.status=status;if(publish_anchor)out.world_from_rig=world_;
        return std::move(out);
    };
    bool acquisition_reversed=same&&known(frame.source.acquisition_clock)&&last_->acquisition_timestamp_ns&&frame.source.acquisition_timestamp_ns<=last_->acquisition_timestamp_ns;
    if(!same||acquisition_reversed||!frame.previous_source)return reanchor(StereoTrajectoryStatus::Anchored,true);
    if(*frame.previous_source!=*last_)return reanchor(StereoTrajectoryStatus::SourceMismatch,false);
    if(frame.source.ingress_timestamp_ns-last_->ingress_timestamp_ns>config_.max_ingress_gap_ns)return reanchor(StereoTrajectoryStatus::HistoryExpired,false);
    if(!quality(frame,config_))return reanchor(StereoTrajectoryStatus::TrackingLost,false);
    auto next_world=compose_world(world_,frame.motion.current_from_previous);
    if(!stereo_rigid_transform(next_world))return reanchor(StereoTrajectoryStatus::TrackingLost,false);
    auto next=frame.source;last_=std::move(next);world_=next_world;
    out.epoch=epoch_;out.status=StereoTrajectoryStatus::Tracked;out.world_from_rig=world_;
    out.local_covariance=frame.motion.covariance;
    // No independence assumption is made for correlated successive stereo fits.
    out.covariance_calibrated=false;out.temporal_qualified=false;
    out.inlier_fraction=double(frame.motion.inliers.size())/double(frame.tracked_features);out.rms_m=frame.motion.rms_m;
    return out;
}
}
