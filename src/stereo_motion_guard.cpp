#include <stereoforge/stereo_motion_guard.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace stereoforge {
namespace {
struct Coverage {unsigned cells=0;double x=0,y=0;};
Coverage coverage(std::span<StereoImageTrack const> pixels,std::span<std::size_t const> ids,StereoMotionGuardConfig const& cfg,bool previous) {
    std::array<bool,256> cells{};std::array<double,2> lo{1,1},hi{0,0};
    for(auto i:ids) {
        auto const& uv=previous?pixels[i].previous:pixels[i].current;
        for(unsigned axis=0;axis<2;++axis){lo[axis]=std::min(lo[axis],uv[axis]);hi[axis]=std::max(hi[axis],uv[axis]);}
        unsigned x=std::min(cfg.columns-1,unsigned(uv[0]*cfg.columns)),y=std::min(cfg.rows-1,unsigned(uv[1]*cfg.rows));cells[y*cfg.columns+x]=true;
    }
    return {unsigned(std::count(cells.begin(),cells.end(),true)),hi[0]-lo[0],hi[1]-lo[1]};
}
std::array<double,3> project(std::array<double,16> const& t,std::array<double,3> const& point) {
    std::array<double,3> out{};for(unsigned i=0;i<3;++i){out[i]=t[i*4+3];for(unsigned j=0;j<3;++j)out[i]+=t[i*4+j]*point[j];}return out;
}
bool valid_pixel(std::array<double,2> const& uv) {return std::all_of(uv.begin(),uv.end(),[](double v){return std::isfinite(v)&&v>=0&&v<=1;});}
}
void StereoMotionGuardConfig::validate()const {
    if(columns==0||rows==0||columns>16||rows>16||min_cells==0||min_cells>columns*rows||
       !std::isfinite(min_span_x)||!std::isfinite(min_span_y)||min_span_x<=0||min_span_x>1||min_span_y<=0||min_span_y>1||
       min_secondary_inliers<3||min_secondary_inliers>4096||max_secondary_hypotheses==0||max_secondary_hypotheses>4096||
       !std::isfinite(min_secondary_fraction)||min_secondary_fraction<=0||min_secondary_fraction>.5||
       !std::isfinite(min_model_separation_m)||min_model_separation_m<1e-6||min_model_separation_m>1||
       !std::isfinite(min_model_separation_rad)||min_model_separation_rad<1e-6||min_model_separation_rad>3.141592653589793)
        throw std::invalid_argument("invalid bounded motion guard configuration");
}
StereoMotionGuardResult guard_stereo_motion(std::span<StereoMetricTrack const> points,std::span<StereoImageTrack const> pixels,
                                            StereoMotionResult const& candidate,StereoMotionConfig const& estimator,StereoMotionGuardConfig const& cfg) {
    cfg.validate();estimator.validate();StereoMotionGuardResult report;
    if(!cfg.enabled||candidate.status!=StereoMotionStatus::Tracked)return report;
    report.status=StereoMotionGuardStatus::InvalidInput;
    if(points.size()!=pixels.size()||points.size()>estimator.max_tracks||candidate.inliers.size()<estimator.min_inliers||
       candidate.inliers.size()>points.size()||double(candidate.inliers.size())<estimator.min_inlier_fraction*double(points.size())||!stereo_rigid_transform(candidate.current_from_previous))return report;
    for(std::size_t i=0;i<points.size();++i) {
        auto const& p=points[i];
        if(!valid_pixel(pixels[i].previous)||!valid_pixel(pixels[i].current)||!std::isfinite(p.sigma_m)||p.sigma_m<=0||p.sigma_m>1||p.previous[2]<=0||p.current[2]<=0||
           !std::all_of(p.previous.begin(),p.previous.end(),[](double v){return std::isfinite(v)&&std::abs(v)<=1000;})||
           !std::all_of(p.current.begin(),p.current.end(),[](double v){return std::isfinite(v)&&std::abs(v)<=1000;}))return report;
    }
    std::vector<bool> inlier(points.size());
    for(auto id:candidate.inliers){if(id>=points.size()||inlier[id])return report;inlier[id]=true;}
    report.primary_inliers=candidate.inliers.size();
    auto before=coverage(pixels,candidate.inliers,cfg,true),after=coverage(pixels,candidate.inliers,cfg,false);
    report.occupied_cells=std::min(before.cells,after.cells);report.span_x=std::min(before.x,after.x);report.span_y=std::min(before.y,after.y);
    if(report.occupied_cells<cfg.min_cells||report.span_x<cfg.min_span_x||report.span_y<cfg.min_span_y){report.status=StereoMotionGuardStatus::InsufficientImageSupport;return report;}
    report.status=StereoMotionGuardStatus::Accepted;
    std::vector<StereoMetricTrack> remaining;remaining.reserve(points.size()-candidate.inliers.size());
    for(std::size_t i=0;i<points.size();++i)if(!inlier[i])remaining.push_back(points[i]);
    auto required=std::max(cfg.min_secondary_inliers,std::size_t(std::ceil(cfg.min_secondary_fraction*double(points.size()))));
    if(remaining.size()<required)return report;
    auto secondary_config=estimator;secondary_config.min_inliers=required;secondary_config.hypotheses=std::min(estimator.hypotheses,cfg.max_secondary_hypotheses);secondary_config.bearing_refinement=false;
    auto secondary=estimate_stereo_consensus(remaining,secondary_config);
    if(secondary.status!=StereoMotionStatus::Tracked)return report;
    report.secondary_inliers=secondary.inliers.size();std::vector<double> differences;differences.reserve(secondary.inliers.size());
    for(auto i:secondary.inliers){auto a=project(candidate.current_from_previous,remaining[i].previous),b=project(secondary.current_from_previous,remaining[i].previous);differences.push_back(std::hypot(a[0]-b[0],a[1]-b[1],a[2]-b[2]));}
    auto middle=differences.begin()+differences.size()/2;std::nth_element(differences.begin(),middle,differences.end());report.model_separation_m=*middle;
    double trace=0;for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)trace+=candidate.current_from_previous[i*4+j]*secondary.current_from_previous[i*4+j];
    report.model_separation_rad=std::acos(std::clamp((trace-1)*.5,-1.,1.));
    if(report.model_separation_m>=cfg.min_model_separation_m||report.model_separation_rad>=cfg.min_model_separation_rad)report.status=StereoMotionGuardStatus::MultipleMotions;
    return report;
}
}
