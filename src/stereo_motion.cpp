#include <stereoforge/stereo_motion.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>

namespace stereoforge {
namespace {
using Vec = std::array<double, 3>;
template<std::size_t N> using Matrix = std::array<double, N*N>;
template<std::size_t N> struct Eigen { std::array<double, N> values; Matrix<N> vectors; };
// Jacobi rotations for the small symmetric normal matrices used below.
template<std::size_t N> Eigen<N> symmetric_eigen(Matrix<N> a) {
    Matrix<N> v{}; for(std::size_t i=0;i<N;++i) v[i*N+i]=1;
    for(unsigned iteration=0;iteration<100*N*N;++iteration) {
        std::size_t p=0,q=1; double largest=0,scale=0;
        for(std::size_t i=0;i<N;++i) {
            scale=std::max(scale,std::abs(a[i*N+i]));
            for(std::size_t j=i+1;j<N;++j) if(std::abs(a[i*N+j])>largest) {largest=std::abs(a[i*N+j]);p=i;q=j;}
        }
        if(largest<=std::max(1e-30,scale*1e-14)) break;
        double angle=.5*std::atan2(2*a[p*N+q],a[q*N+q]-a[p*N+p]);
        double c=std::cos(angle),s=std::sin(angle),app=a[p*N+p],aqq=a[q*N+q],apq=a[p*N+q];
        for(std::size_t k=0;k<N;++k) if(k!=p&&k!=q) {
            double kp=a[k*N+p],kq=a[k*N+q];
            a[k*N+p]=a[p*N+k]=c*kp-s*kq;
            a[k*N+q]=a[q*N+k]=s*kp+c*kq;
        }
        a[p*N+p]=c*c*app-2*s*c*apq+s*s*aqq;
        a[q*N+q]=s*s*app+2*s*c*apq+c*c*aqq;
        a[p*N+q]=a[q*N+p]=0;
        for(std::size_t k=0;k<N;++k) {
            double kp=v[k*N+p],kq=v[k*N+q];v[k*N+p]=c*kp-s*kq;v[k*N+q]=s*kp+c*kq;
        }
    }
    Eigen<N> result{{},v};for(std::size_t i=0;i<N;++i) result.values[i]=a[i*N+i];return result;
}
double weight(StereoMetricTrack const& p, StereoMotionConfig const& cfg) {
    double sigma=std::max(p.sigma_m,cfg.noise_floor_m);return 1/(sigma*sigma);
}
Vec transform(std::array<double,16> const& t, Vec const& p) {
    Vec out{};for(int i=0;i<3;++i) {out[i]=t[i*4+3];for(int j=0;j<3;++j) out[i]+=t[i*4+j]*p[j];}return out;
}
bool fit(std::span<StereoMetricTrack const> tracks, std::span<std::size_t const> ids,
         StereoMotionConfig const& cfg, double min_spread, std::array<double,16>& t) {
    Vec a{},b{};double total=0;
    for(auto id:ids) {auto const& p=tracks[id];double w=weight(p,cfg);total+=w;for(int i=0;i<3;++i) {a[i]+=w*p.previous[i];b[i]+=w*p.current[i];}}
    for(int i=0;i<3;++i) {a[i]/=total;b[i]/=total;}
    Matrix<3> h{},spread{};
    for(auto id:ids) {
        auto const& p=tracks[id];double w=weight(p,cfg)/total;
        for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
            h[i*3+j]+=w*(p.previous[i]-a[i])*(p.current[j]-b[j]);
            spread[i*3+j]+=w*(p.previous[i]-a[i])*(p.previous[j]-a[j]);
        }
    }
    auto eigen=symmetric_eigen<3>(spread).values;std::sort(eigen.begin(),eigen.end());
    if(eigen[1]<min_spread*min_spread) return false; // A plane is observable; a line is not.
    double xx=h[0],xy=h[1],xz=h[2],yx=h[3],yy=h[4],yz=h[5],zx=h[6],zy=h[7],zz=h[8];
    Matrix<4> n={xx+yy+zz,yz-zy,zx-xz,xy-yx,
                 yz-zy,xx-yy-zz,xy+yx,zx+xz,
                 zx-xz,xy+yx,-xx+yy-zz,yz+zy,
                 xy-yx,zx+xz,yz+zy,-xx-yy+zz};
    auto e=symmetric_eigen<4>(n);auto k=std::max_element(e.values.begin(),e.values.end())-e.values.begin();
    double w=e.vectors[k],x=e.vectors[4+k],y=e.vectors[8+k],z=e.vectors[12+k];
    t={1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),0,
       2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),0,
       2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y),0,0,0,0,1};
    for(int i=0;i<3;++i) {t[i*4+3]=b[i];for(int j=0;j<3;++j) t[i*4+3]-=t[i*4+j]*a[j];}
    return true;
}
struct Consensus { std::vector<std::size_t> ids;double cost=0,squared_error=0; };
Consensus consensus(std::span<StereoMetricTrack const> points, std::array<double,16> const& t, StereoMotionConfig const& cfg) {
    Consensus c;
    for(std::size_t id=0;id<points.size();++id) {
        auto const& p=points[id];auto predicted=transform(t,p.previous);double squared=0;
        for(int i=0;i<3;++i) squared+=(predicted[i]-p.current[i])*(predicted[i]-p.current[i]);
        double gate=std::min(cfg.max_residual_m,3*std::max(p.sigma_m,cfg.noise_floor_m));
        c.cost+=std::min(1.,squared/(gate*gate));
        if(squared<=gate*gate) {c.ids.push_back(id);c.squared_error+=squared;}
    }
    return c;
}
bool invert_normal(Matrix<6> const& info, Matrix<6>& out) {
    auto e=symmetric_eigen<6>(info);double largest=*std::max_element(e.values.begin(),e.values.end());
    if(!std::isfinite(largest)||largest<=0||*std::min_element(e.values.begin(),e.values.end())<=largest*1e-10)return false;
    out={};for(int i=0;i<6;++i) for(int j=0;j<6;++j) for(int k=0;k<6;++k)out[i*6+j]+=e.vectors[i*6+k]*e.vectors[j*6+k]/e.values[k];
    return true;
}
bool refine_bearings(std::span<StereoMetricTrack const> points, std::span<std::size_t const> ids,
                     StereoMotionConfig const& cfg, std::array<double,16>& t, Matrix<6>& uncertainty) {
    for(unsigned iteration=0;iteration<12;++iteration) {
        Matrix<6> info{};std::array<double,6> rhs{};
        for(auto id:ids) {
            auto q=transform(t,points[id].previous);auto const& b=points[id].current;
            double range=std::hypot(q[0],q[1],q[2]),observed=std::hypot(b[0],b[1],b[2]);
            if(range<1e-6)return false;
            double base[3][6]={{1,0,0,0,q[2],-q[1]},{0,1,0,-q[2],0,q[0]},{0,0,1,q[1],-q[0],0}};
            double j[3][6]={},error[3];
            for(int i=0;i<3;++i) {
                error[i]=b[i]/observed-q[i]/range;
                for(int k=0;k<3;++k)for(int c=0;c<6;++c)j[i][c]+=((i==k?1.:0.)-q[i]*q[k]/(range*range))*base[k][c]/range;
            }
            double w=range*range*weight(points[id],cfg);
            for(int i=0;i<6;++i) {
                for(int k=0;k<3;++k)rhs[i]+=w*j[k][i]*error[k];
                for(int c=0;c<6;++c)for(int k=0;k<3;++k)info[i*6+c]+=w*j[k][i]*j[k][c];
            }
        }
        if(!invert_normal(info,uncertainty))return false;
        std::array<double,6> step{};
        for(int i=0;i<6;++i)for(int j=0;j<6;++j)step[i]+=uncertainty[i*6+j]*rhs[j];
        double angle=std::hypot(step[3],step[4],step[5]);
        if(!std::isfinite(angle)||angle>.2||std::hypot(step[0],step[1],step[2])>.5)return false;
        if(std::all_of(step.begin(),step.end(),[](double v){return std::abs(v)<1e-10;}))return true;
        Matrix<3> skew{0,-step[5],step[4],step[5],0,-step[3],-step[4],step[3],0},rotation{};
        double a=angle<1e-8?1:std::sin(angle)/angle,b=angle<1e-8?.5:(1-std::cos(angle))/(angle*angle);
        for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
            rotation[i*3+j]=(i==j?1.:0.)+a*skew[i*3+j];
            for(int k=0;k<3;++k)rotation[i*3+j]+=b*skew[i*3+k]*skew[k*3+j];
        }
        auto old=t;
        for(int i=0;i<3;++i)for(int j=0;j<4;++j) {
            t[i*4+j]=j==3?step[i]:0;
            for(int k=0;k<3;++k)t[i*4+j]+=rotation[i*3+k]*old[k*4+j];
        }
    }
    return false;
}
bool covariance(std::span<StereoMetricTrack const> points, std::span<std::size_t const> ids,
                std::array<double,16> const& t, StereoMotionConfig const& cfg, double rms, Matrix<6>& out) {
    Matrix<6> info{};
    for(auto id:ids) {
        auto p=transform(t,points[id].previous);
        double variance=std::max({points[id].sigma_m*points[id].sigma_m,cfg.noise_floor_m*cfg.noise_floor_m,rms*rms/3});
        double j[3][6]={{1,0,0,0,p[2],-p[1]},{0,1,0,-p[2],0,p[0]},{0,0,1,p[1],-p[0],0}};
        for(int a=0;a<6;++a) for(int b=0;b<6;++b) for(auto const& row:j) info[a*6+b]+=row[a]*row[b]/variance;
    }
    return invert_normal(info,out);
}
}
bool stereo_rigid_transform(std::array<double,16> const& t,double tolerance) {
    if(!std::isfinite(tolerance)||tolerance<=0||tolerance>1e-3||
       !std::all_of(t.begin(),t.end(),[](double v){return std::isfinite(v);})||t[12]!=0||t[13]!=0||t[14]!=0||t[15]!=1)return false;
    for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j) {
        double dot=0;for(unsigned k=0;k<3;++k)dot+=t[k*4+i]*t[k*4+j];
        if(std::abs(dot-double(i==j))>tolerance)return false;
    }
    double det=t[0]*(t[5]*t[10]-t[6]*t[9])-t[1]*(t[4]*t[10]-t[6]*t[8])+t[2]*(t[4]*t[9]-t[5]*t[8]);
    return std::abs(det-1)<=tolerance;
}
void StereoMotionConfig::validate() const {
    if(max_tracks<3||max_tracks>4096||min_inliers<3||min_inliers>max_tracks||hypotheses==0||hypotheses>4096||
       !std::isfinite(min_inlier_fraction)||min_inlier_fraction<=.5||min_inlier_fraction>1||
       !std::isfinite(max_residual_m)||max_residual_m<1e-6||max_residual_m>1||
       !std::isfinite(noise_floor_m)||noise_floor_m<1e-6||noise_floor_m>max_residual_m||
       !std::isfinite(min_spread_m)||min_spread_m<1e-6||min_spread_m>10)
        throw std::invalid_argument("invalid bounded stereo motion configuration");
}
static StereoMotionResult estimate_motion(std::span<StereoMetricTrack const> points,StereoMotionConfig const& cfg,bool majority) {
    cfg.validate();StereoMotionResult result;
    if(points.size()>cfg.max_tracks) {result.status=StereoMotionStatus::InvalidInput;return result;}
    for(auto const& p:points) {
        if(!std::isfinite(p.sigma_m)||p.sigma_m<=0||p.sigma_m>1||p.previous[2]<=0||p.current[2]<=0||
           !std::all_of(p.previous.begin(),p.previous.end(),[](double v){return std::isfinite(v)&&std::abs(v)<=1000;})||
           !std::all_of(p.current.begin(),p.current.end(),[](double v){return std::isfinite(v)&&std::abs(v)<=1000;})) {
            result.status=StereoMotionStatus::InvalidInput;return result;
        }
    }
    if(points.size()<cfg.min_inliers) return result;
    std::vector<std::size_t> all(points.size());std::iota(all.begin(),all.end(),0);
    std::array<double,16> pose{};
    Consensus best;
    bool geometric_fit=fit(points,all,cfg,cfg.min_spread_m,pose);
    if(geometric_fit)best=consensus(points,pose,cfg);
    uint64_t rng=0x9e3779b97f4a7c15ULL;
    auto next=[&](){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return std::size_t(rng%points.size());};
    for(std::size_t h=0;h<cfg.hypotheses;++h) {
        std::array<std::size_t,3> ids{next(),next(),next()};
        if(ids[0]==ids[1]||ids[0]==ids[2]||ids[1]==ids[2]||!fit(points,ids,cfg,1e-5,pose)) continue;
        geometric_fit=true;
        auto c=consensus(points,pose,cfg);
        if(c.ids.size()>best.ids.size()||(c.ids.size()==best.ids.size()&&c.cost<best.cost)) best=std::move(c);
    }
    if(!geometric_fit) {result.status=StereoMotionStatus::Degenerate;return result;}
    result.status=StereoMotionStatus::NoConsensus;
    auto required=majority?std::max(cfg.min_inliers,std::size_t(std::ceil(cfg.min_inlier_fraction*points.size()))):cfg.min_inliers;
    for(unsigned iteration=0;iteration<8;++iteration) {
        if(best.ids.size()<required) return result;
        if(!fit(points,best.ids,cfg,cfg.min_spread_m,pose)) {result.status=StereoMotionStatus::Degenerate;return result;}
        Matrix<6> bearing_covariance{};
        if(cfg.bearing_refinement&&!refine_bearings(points,best.ids,cfg,pose,bearing_covariance)) {result.status=StereoMotionStatus::Degenerate;return result;}
        auto c=consensus(points,pose,cfg);
        if(c.ids!=best.ids) {best=std::move(c);continue;}
        result.rms_m=std::sqrt(c.squared_error/c.ids.size());
        if(!covariance(points,c.ids,pose,cfg,result.rms_m,result.covariance)) {result.status=StereoMotionStatus::Degenerate;return result;}
        if(cfg.bearing_refinement)result.covariance=bearing_covariance;
        result.current_from_previous=pose;result.inliers=std::move(c.ids);result.status=StereoMotionStatus::Tracked;return result;
    }
    return result;
}
StereoMotionResult estimate_stereo_motion(std::span<StereoMetricTrack const> points,StereoMotionConfig const& cfg){return estimate_motion(points,cfg,true);}
StereoMotionResult estimate_stereo_consensus(std::span<StereoMetricTrack const> points,StereoMotionConfig const& cfg){return estimate_motion(points,cfg,false);}

}
