#include <stereoforge/stereo_trajectory.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace stereoforge;
static void check(bool ok,char const* m){if(!ok)throw std::runtime_error(m);}
static StereoFrameIdentity stamp(uint64_t id) {
    return {id,id*33000000,id*33000000+5000000,"rig","original","rectified","administrative-session","camera","host",{}};
}
static StereoVoFrame step(uint64_t id,double tx=.03,double angle=0) {
    StereoVoFrame v;v.source=stamp(id);if(id)v.previous_source=stamp(id-1);v.tracked_features=80;
    v.motion.status=StereoMotionStatus::Tracked;
    double c=std::cos(angle),s=std::sin(angle);v.motion.current_from_previous={c,0,s,tx,0,1,0,0,-s,0,c,0,0,0,0,1};
    for(std::size_t i=0;i<60;++i)v.motion.inliers.push_back(i);
    for(unsigned i=0;i<6;++i)v.motion.covariance[i*6+i]=1e-5;
    v.motion.rms_m=.002;return v;
}
static void chain_and_direction() {
    StereoTrajectory t;auto first=step(0);first.previous_source.reset();auto a=t.process(first);
    check(a.status==StereoTrajectoryStatus::Anchored&&a.world_from_rig.has_value(),"first frame not anchored");
    auto b=t.process(step(1));check(b.status==StereoTrajectoryStatus::Tracked,"valid chain not tracked");
    check(std::abs((*b.world_from_rig)[3]+.03)<1e-12,"relative transform not inverted");
    auto c=t.process(step(2,.04,.1));auto const& w=*c.world_from_rig;
    check(c.epoch==a.epoch&&std::abs(w[3]-(-.03-.04*std::cos(.1)))<1e-12,"world translation composition incorrect");
    check(std::abs(w[11]+.04*std::sin(.1))<1e-12&&std::abs(w[2]+std::sin(.1))<1e-12,"world rotation composition incorrect");
    check(c.local_covariance.has_value()&&!c.covariance_calibrated&&!c.temporal_qualified,"conditional covariance promoted");
    check(c.source==stamp(2)&&!c.source.clock_mapping_uncertainty_ns,"source or unknown clock uncertainty lost");
}
static void chain_failures() {
    StereoTrajectory t;auto a=t.process(step(0));auto lost=step(1);lost.motion.status=StereoMotionStatus::NoConsensus;
    auto b=t.process(lost);check(!b.world_from_rig&&b.epoch>a.epoch&&b.status==StereoTrajectoryStatus::TrackingLost,"pose loss did not break epoch");
    auto c=t.process(step(2));check(c.status==StereoTrajectoryStatus::Tracked&&std::abs((*c.world_from_rig)[3]+.03)<1e-12,"lost epoch reconnected old world");
    auto wrong=step(3);wrong.previous_source->acquisition_timestamp_ns+=1;
    auto d=t.process(wrong);check(d.status==StereoTrajectoryStatus::SourceMismatch&&!d.world_from_rig&&d.epoch>c.epoch,"wrong exact source accepted");
    auto expired=step(20);expired.previous_source=wrong.source;
    auto e=t.process(expired);check(e.status==StereoTrajectoryStatus::HistoryExpired&&!e.world_from_rig,"expired chain reused");
    auto clock=step(21);clock.previous_source=expired.source;clock.source.ingress_clock="new-clock";
    auto f=t.process(clock);check(f.status==StereoTrajectoryStatus::Anchored&&f.epoch>e.epoch,"clock change did not reanchor");
    t.reset();auto g=t.process(step(22));check(g.status==StereoTrajectoryStatus::Anchored&&g.epoch>f.epoch,"explicit reset reused epoch");
}
static void quality_rejections() {
    for(unsigned kind=0;kind<10;++kind) {
        StereoTrajectory t;t.process(step(0));auto v=step(1);
        if(kind==0)v.motion.inliers.resize(10);
        if(kind==1)v.motion.inliers[1]=v.motion.inliers[0];
        if(kind==2)v.motion.inliers[0]=1000;
        if(kind==3)v.motion.current_from_previous[0]=2;
        if(kind==4)v.motion.current_from_previous[3]=.5;
        if(kind==5)v.motion.current_from_previous[3]=std::numeric_limits<double>::quiet_NaN();
        if(kind==6)v.motion.covariance[0]=-1;
        if(kind==7)v.motion.covariance[1]=1;
        if(kind==8)v.motion.rms_m=.05;
        if(kind==9)v.tracked_features=2000;
        auto r=t.process(v);check(r.status==StereoTrajectoryStatus::TrackingLost&&!r.world_from_rig,"bad pose quality admitted");
    }
    StereoTrajectory t;t.process(step(0));t.process(step(1));bool rejected=false;
    try{t.process(step(1));}catch(std::invalid_argument const&){rejected=true;}
    check(rejected,"duplicate trajectory frame accepted");
    check(t.process(step(2)).status==StereoTrajectoryStatus::Tracked,"rejected duplicate mutated trajectory");
}
static void identity_epochs_and_invalid_inputs() {
    StereoTrajectory a,b;
    auto first=a.process(step(0)),other=b.process(step(0));
    check(first.trajectory_id!=0&&first.trajectory_id!=other.trajectory_id,"trajectory gauges share identity");
    auto calibration=step(1);calibration.source.rectified_calibration="changed";
    auto changed=a.process(calibration);
    check(changed.status==StereoTrajectoryStatus::Anchored&&changed.epoch>first.epoch,"calibration change retained world");
    auto session=step(2);session.source.rectified_calibration="changed";session.source.rig_session="another mounting session";
    auto mounted=a.process(session);check(mounted.status==StereoTrajectoryStatus::Anchored&&mounted.epoch>changed.epoch,"mounting session change retained world");
    StereoTrajectory c;c.process(step(1));auto reversed=step(2);reversed.source.acquisition_timestamp_ns=1;
    check(c.process(reversed).status==StereoTrajectoryStatus::Anchored,"reversed acquisition clock retained world");
    auto invalid=step(3);invalid.source.ingress_clock="unspecified";bool rejected=false;
    try {c.process(invalid);}catch(std::invalid_argument const&){rejected=true;}
    check(rejected,"unspecified ingress clock admitted");
    StereoTrajectoryConfig bad;bad.max_ingress_gap_ns=200000001;rejected=false;
    try {StereoTrajectory invalid_config(bad);}catch(std::invalid_argument const&){rejected=true;}
    check(rejected,"unbounded trajectory age admitted");
}
int main(){try{chain_and_direction();chain_failures();quality_rejections();identity_epochs_and_invalid_inputs();std::cout<<"trajectory contracts passed\n";}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
